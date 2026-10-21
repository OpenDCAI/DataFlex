"""ClusterLESS: 聚类 + LESS 梯度评分 + softmax 采样。

流程：
  1. 聚类（继承 ClusterBaseSelector）→ cluster_ids
  2. 每簇随机取代表点
  3. 代表点 + eval 集各自收集 Adam 预处理梯度 → TRAK 投影 → L2 归一化
  4. 簇均值梯度 vs eval 梯度 → cosine similarity → 簇分数
  5. softmax(簇分数 / temperature) → 每样本概率 → multinomial 采样
"""

import math
import os
from typing import List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataflex.core.registry import register_selector
from dataflex.utils.logging import logger
from dataflex.utils.selector_io import load_cached_selection, save_selection

from .cluster_base_selector import ClusterBaseSelector, _move_to_device


def _collate_idx(indices, dataset, collator):
    """将 index 列表 → collator({...}) + idx 张量。"""
    examples = [dataset[int(i)] for i in indices]
    batch = collator(examples)
    batch["idx"] = torch.tensor([int(i) for i in indices], dtype=torch.long)
    return batch


@register_selector("cluster_less")
class ClusterLessSelector(ClusterBaseSelector):

    def __init__(
        self,
        # ---- ClusterBaseSelector 参数 ----
        dataset, accelerator, data_collator, cache_dir,
        cluster_feature="embedding", embedding_model_name_or_path=None,
        embedding_model_dtype="bf16", embedding_layer=-1,
        gradient_target_modules=None, gradient_proj_dim=2048,
        cluster_size=64, clustering_max_iter=300, kmeans_n_init=5,
        samples_per_cluster=6, seed=3,
        # ---- LESS 评分参数 ----
        eval_dataset=None,                  # 目标任务验证集（计算梯度相似度）
        gradient_type: str = "adam",        # "adam"（预处理梯度）或 "sgd"
        proj_dim: int = 8192,               # 评分梯度的 TRAK 投影维度
        save_interval: int = 16,
        # ---- 采样参数 ----
        temperature: float = 1.0,           # softmax 温度（越小越锐利）
        replacement: bool = False,
        dedup_across_rounds: bool = False,
        **kwargs,
    ):
        super().__init__(
            dataset=dataset, accelerator=accelerator, data_collator=data_collator,
            cache_dir=cache_dir, cluster_feature=cluster_feature,
            embedding_model_name_or_path=embedding_model_name_or_path,
            embedding_model_dtype=embedding_model_dtype, embedding_layer=embedding_layer,
            gradient_target_modules=gradient_target_modules,
            gradient_proj_dim=gradient_proj_dim, cluster_size=cluster_size,
            clustering_max_iter=clustering_max_iter, kmeans_n_init=kmeans_n_init,
            samples_per_cluster=samples_per_cluster, seed=seed,
        )
        if eval_dataset is None:
            raise ValueError("ClusterLessSelector requires eval_dataset.")
        self.eval_dataset = eval_dataset
        self.gradient_type = gradient_type
        self.proj_dim = proj_dim
        self.save_interval = save_interval
        self.temperature = float(temperature)
        self.replacement = replacement
        self.dedup_across_rounds = dedup_across_rounds
        self._selected_history: set = set()

    def select(self, model, step_id: int, num_samples: int, **kwargs) -> List[int]:
        os.makedirs(self.cache_dir, exist_ok=True)
        save_path = os.path.join(self.cache_dir, f"step_{step_id}.json")

        # 缓存检查
        cached = self._broadcast_bool(
            os.path.exists(save_path) if self.accelerator.is_main_process else False)
        if cached:
            if self.accelerator.is_main_process:
                indices, _ = load_cached_selection(save_path)
            else:
                indices = None
            return self._broadcast_long_tensor(
                torch.tensor(indices, dtype=torch.long) if indices is not None else None
            ).tolist()

        # 1. 聚类 + 代表点
        cluster_ids = self.get_or_create_clusters(model)
        representatives = self.sample_representatives(cluster_ids, step_id)
        rep_dataset = Subset(self.dataset, representatives)

        # 2. 收集梯度（代表点 + eval）
        rep_dir = os.path.join(self.cache_dir, "rep_grads", str(step_id))
        eval_dir = os.path.join(self.cache_dir, "eval_grads", str(step_id))
        optimizer_state = kwargs.get("optimizer_state", None)

        rep_grads = self._collect_projected_grads(
            model, rep_dataset, rep_dir, self.gradient_type, optimizer_state)
        eval_grads = self._collect_projected_grads(
            model, self.eval_dataset, eval_dir, "sgd", None)

        self.accelerator.wait_for_everyone()

        # 3. 评分 + 采样（rank 0）
        if self.accelerator.is_main_process:
            rep_g = torch.load(os.path.join(rep_dir, "grads.pt"), map_location="cpu").float()
            eval_g = torch.load(os.path.join(eval_dir, "grads.pt"), map_location="cpu").float()

            # 簇均值 → L2 归一化 → cosine similarity
            num_clusters = int(cluster_ids.max().item()) + 1
            cluster_grads = torch.zeros(num_clusters, self.proj_dim)
            counts = torch.zeros(num_clusters, 1)
            rep_cids = cluster_ids[torch.tensor(representatives, dtype=torch.long)]
            cluster_grads.index_add_(0, rep_cids, rep_g)
            counts.index_add_(0, rep_cids, torch.ones(len(representatives), 1))
            cluster_grads /= counts.clamp(min=1)
            cluster_grads = torch.nn.functional.normalize(cluster_grads, dim=1)
            eval_g = torch.nn.functional.normalize(eval_g, dim=1)
            cluster_scores = (cluster_grads @ eval_g.T).mean(dim=1)

            # softmax → per-sample prob → multinomial
            probs = torch.softmax(cluster_scores / self.temperature, dim=0)
            sample_probs = probs[cluster_ids]

            # 跨轮去重
            if self.dedup_across_rounds and self._selected_history:
                mask = torch.ones_like(sample_probs, dtype=torch.bool)
                for idx in self._selected_history:
                    if idx < len(mask):
                        mask[idx] = False
                sample_probs[~mask] = 0

            sample_probs = sample_probs / sample_probs.sum().clamp(min=1e-12)
            rng = torch.Generator()
            rng.manual_seed(self.seed + step_id)
            selected = torch.multinomial(
                sample_probs, num_samples=min(num_samples, len(sample_probs)),
                replacement=self.replacement, generator=rng,
            ).tolist()

            if self.dedup_across_rounds:
                self._selected_history.update(selected)

            save_selection(save_path, selected,
                           {"cluster_less_score": cluster_scores[cluster_ids[torch.tensor(selected)]].tolist()},
                           self.accelerator)
            logger.info(f"[ClusterLess] selected {len(selected)} samples, clusters={num_clusters}, representatives={len(representatives)}")
        else:
            selected = None

        return self._broadcast_long_tensor(
            torch.tensor(selected, dtype=torch.long) if selected is not None else None
        ).tolist()

    # ============================================================
    # 梯度收集（内联，不继承 LessSelector）
    # ============================================================

    def _collect_projected_grads(self, model, dataset_to_use, save_dir, gradient_type, optimizer_state):
        """收集 per-sample 梯度 → TRAK 投影 → 存盘。

        gradient_type="adam" 时做 Adam 预处理：grad = (β1*m + (1-β1)*g) / √(β2*v + (1-β2)*g² + ε)
        gradient_type="sgd" 时直接用 raw gradient。
        用 TRAK CudaProjector（GPU 分块投影，不存全矩阵）。
        """
        grads_path = os.path.join(save_dir, "grads.pt")
        cached = self._broadcast_bool(
            os.path.exists(grads_path) if self.accelerator.is_main_process else False)
        if cached:
            return

        os.makedirs(save_dir, exist_ok=True)

        # Adam 状态
        m_vec, v_vec = None, None
        if gradient_type == "adam" and optimizer_state is not None:
            m_vec, v_vec = self._extract_adam_state(model, optimizer_state)

        # TRAK projector
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        from .less_selector import _trak_projectors
        BasicProjector, CudaProjector, ProjectionType = _trak_projectors()
        proj_device = "cpu"
        try:
            projector = CudaProjector(
                grad_dim=total_params, proj_dim=self.proj_dim,
                seed=self.seed, proj_type=ProjectionType.rademacher, device=str(self.device), max_batch_size=16)
            proj_device = str(self.device)
        except Exception:
            projector = BasicProjector(
                grad_dim=total_params, proj_dim=self.proj_dim,
                seed=self.seed, proj_type=ProjectionType.rademacher, device="cpu")

        # batch=1 循环收集
        indexed = list(range(len(dataset_to_use)))
        loader = self.accelerator.prepare(
            DataLoader(indexed, batch_size=1, shuffle=False, num_workers=0,
                       collate_fn=lambda indices: _collate_idx(indices, dataset_to_use, self.data_collator))
        )

        model.eval()
        grads_buf = []
        idxs_buf = []
        for batch in tqdm(loader, desc=f"[ClusterLess] grads ({gradient_type})",
                          disable=not self.accelerator.is_local_main_process):
            idx = batch.pop("idx")  # injected by _collate_idx
            batch = _move_to_device(batch, self.device)
            model.zero_grad()
            # no_sync：避免 DDP 把各卡梯度平均，保证是单样本梯度
            with self.accelerator.no_sync(model):
                model(**batch).loss.backward()

            # 拼接所有 requires_grad 参数的梯度
            grad_flat = torch.cat([
                (p.grad if p.grad is not None else torch.zeros_like(p)).detach().float().view(-1)
                for p in model.parameters() if p.requires_grad
            ]).cpu()

            # Adam 预处理
            if gradient_type == "adam" and m_vec is not None:
                beta1, beta2, eps = 0.9, 0.999, 1e-8
                grad_flat = (beta1 * m_vec + (1 - beta1) * grad_flat) / \
                            (torch.sqrt(beta2 * v_vec + (1 - beta2) * grad_flat ** 2) + eps)

            # TRAK 投影
            projected = projector.project(grad_flat.unsqueeze(0).to(proj_device), model_id=0).squeeze(0).cpu()
            grads_buf.append(projected)
            idxs_buf.append(idx.cpu())

        model.zero_grad()

        grads = torch.stack(grads_buf) if grads_buf else torch.empty(0, self.proj_dim)
        idxs = torch.cat(idxs_buf) if idxs_buf else torch.empty(0, dtype=torch.long)

        # L2 归一化 + 汇合存盘
        n = len(dataset_to_use)
        distributed = dist.is_available() and dist.is_initialized()
        if distributed:
            shard_dir = os.path.join(save_dir, "shards")
            os.makedirs(shard_dir, exist_ok=True)
            rank = dist.get_rank()
            torch.save({"g": grads, "i": idxs}, os.path.join(shard_dir, f"r{rank}.pt"))
            dist.barrier()
            if self.accelerator.is_main_process:
                ordered = torch.empty(n, self.proj_dim)
                for r in range(dist.get_world_size()):
                    d = torch.load(os.path.join(shard_dir, f"r{r}.pt"), map_location="cpu")
                    if d["g"].numel() > 0:
                        ordered[d["i"].view(-1)] = d["g"]
                ordered = torch.nn.functional.normalize(ordered, dim=1)
                torch.save(ordered, grads_path)
            dist.barrier()
            import shutil
            if self.accelerator.is_main_process:
                shutil.rmtree(shard_dir, ignore_errors=True)
        else:
            ordered = torch.empty(n, self.proj_dim)
            ordered[idxs.view(-1)] = grads
            ordered = torch.nn.functional.normalize(ordered, dim=1)
            torch.save(ordered, grads_path)

    def _extract_adam_state(self, model, optimizer_state):
        """从 optimizer.state（key 为参数 tensor）提取 Adam 的 m 和 v；全部取不到时返回 (None, None)。"""
        m_parts, v_parts = [], []
        found = False
        for p in model.parameters():
            if not p.requires_grad:
                continue
            state = optimizer_state.get(p)
            if state and "exp_avg" in state:
                m_parts.append(state["exp_avg"].float().cpu().view(-1))
                v_parts.append(state["exp_avg_sq"].float().cpu().view(-1))
                found = True
            else:
                m_parts.append(torch.zeros(p.numel()))
                v_parts.append(torch.zeros(p.numel()))
        if not found:
            return None, None
        return torch.cat(m_parts), torch.cat(v_parts)
