"""TracIn: 全量（非聚类）SGD 梯度 + 多 checkpoint 累积。

对训练集**全部样本**收集 SGD 梯度（不聚类），与 eval 集梯度做 dot product × learning_rate，
多次调用时累积分数（TracInCP），最后取 top-k。

继承 ClusterBaseSelector 以复用梯度收集基础设施，但 select() 里不调用聚类。
是所有 selector 里最贵的方法（每轮要对全量 N 个样本做前向+反向）。
"""

import math
import os
import shutil
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
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


@register_selector("tracin")
class TracInSelector(ClusterBaseSelector):

    def __init__(
        self,
        # ---- ClusterBaseSelector 参数（聚类部分不使用，但 registry 按名字过滤需要声明） ----
        dataset, accelerator, data_collator, cache_dir,
        cluster_feature="embedding", embedding_model_name_or_path=None,
        embedding_model_dtype="bf16", embedding_layer=-1,
        gradient_target_modules=None, gradient_proj_dim=2048,
        cluster_size=64, clustering_max_iter=300, kmeans_n_init=5,
        samples_per_cluster=6, seed=3,
        # ---- TracIn 参数 ----
        eval_dataset=None,
        gradient_type: str = "sgd",
        proj_dim: int = 8192,
        save_interval: int = 16,
        learning_rate: float = 1.0,
        accumulate_checkpoints: bool = True,
        normalize_scores: bool = False,     # True = cosine; False = raw dot product
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
            raise ValueError("TracInSelector requires eval_dataset.")
        self.eval_dataset = eval_dataset
        self.gradient_type = gradient_type
        self.proj_dim = proj_dim
        self.save_interval = save_interval
        self.learning_rate = float(learning_rate)
        self.accumulate_checkpoints = accumulate_checkpoints
        self.normalize_scores = normalize_scores
        self._accum_scores: Optional[torch.Tensor] = None

    def select(self, model, step_id: int, num_samples: int, **kwargs) -> List[int]:
        """全量 TracIn 选择（不聚类）。"""
        os.makedirs(self.cache_dir, exist_ok=True)
        save_path = os.path.join(self.cache_dir, f"step_{step_id}.json")

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

        # 1. 收集全量 SGD 梯度（不做 L2 归一化）
        train_dir = os.path.join(self.cache_dir, "train_grads", str(step_id))
        eval_dir = os.path.join(self.cache_dir, "eval_grads", str(step_id))
        self._collect_raw_grads(model, self.dataset, train_dir)
        self._collect_raw_grads(model, self.eval_dataset, eval_dir)
        self.accelerator.wait_for_everyone()

        # 2. 评分（rank 0）
        if self.accelerator.is_main_process:
            train_g = torch.load(os.path.join(train_dir, "grads.pt"), map_location="cpu").float()
            eval_g = torch.load(os.path.join(eval_dir, "grads.pt"), map_location="cpu").float()

            # 可选 cosine 归一化
            if self.normalize_scores:
                train_g = F.normalize(train_g, dim=1)
                eval_g = F.normalize(eval_g, dim=1)

            step_scores = self.learning_rate * (train_g @ eval_g.T).mean(dim=1)

            # TracInCP 累积
            if self.accumulate_checkpoints:
                if self._accum_scores is None:
                    self._accum_scores = step_scores.clone()
                else:
                    self._accum_scores += step_scores
                scores = self._accum_scores
            else:
                scores = step_scores

            # top-k
            topk = torch.topk(scores, k=min(num_samples, len(scores)), largest=True)
            selected = topk.indices.tolist()

            save_selection(save_path, selected,
                           {"tracin_score": topk.values.tolist()}, self.accelerator)
            logger.info(f"[TracIn] selected {len(selected)} samples, accumulate={self.accumulate_checkpoints}")
        else:
            selected = None

        return self._broadcast_long_tensor(
            torch.tensor(selected, dtype=torch.long) if selected is not None else None
        ).tolist()

    # ============================================================
    # 梯度收集（和 ClusterTracIn 共用逻辑，但这里作用在全量上）
    # ============================================================

    def _collect_raw_grads(self, model, dataset_to_use, save_dir):
        """收集 per-sample SGD 梯度 → TRAK 投影 → 存盘（不做 L2 归一化）。"""
        grads_path = os.path.join(save_dir, "grads.pt")
        cached = self._broadcast_bool(
            os.path.exists(grads_path) if self.accelerator.is_main_process else False)
        if cached:
            return

        os.makedirs(save_dir, exist_ok=True)

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

        indexed = list(range(len(dataset_to_use)))
        loader = self.accelerator.prepare(
            DataLoader(indexed, batch_size=1, shuffle=False, num_workers=0,
                       collate_fn=lambda indices: _collate_idx(indices, dataset_to_use, self.data_collator))
        )

        model.eval()
        grads_buf, idxs_buf = [], []
        for batch in tqdm(loader, desc="[TracIn] grads",
                          disable=not self.accelerator.is_local_main_process):
            idx = batch.pop("idx")  # injected by _collate_idx
            batch = _move_to_device(batch, self.device)
            model.zero_grad()
            with self.accelerator.no_sync(model):
                model(**batch).loss.backward()
            grad_flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().float().view(-1)
                                   for p in model.parameters() if p.requires_grad])
            grads_buf.append(projector.project(grad_flat.unsqueeze(0).to(proj_device), model_id=0).squeeze(0).cpu())
            idxs_buf.append(idx.cpu())
        model.zero_grad()

        grads = torch.stack(grads_buf) if grads_buf else torch.empty(0, self.proj_dim)
        idxs = torch.cat(idxs_buf) if idxs_buf else torch.empty(0, dtype=torch.long)
        n = len(dataset_to_use)

        distributed = dist.is_available() and dist.is_initialized()
        if distributed:
            shard_dir = os.path.join(save_dir, "shards")
            os.makedirs(shard_dir, exist_ok=True)
            torch.save({"g": grads, "i": idxs}, os.path.join(shard_dir, f"r{dist.get_rank()}.pt"))
            dist.barrier()
            if self.accelerator.is_main_process:
                ordered = torch.empty(n, self.proj_dim)
                for r in range(dist.get_world_size()):
                    d = torch.load(os.path.join(shard_dir, f"r{r}.pt"), map_location="cpu")
                    if d["g"].numel() > 0:
                        ordered[d["i"].view(-1)] = d["g"]
                torch.save(ordered, grads_path)
            dist.barrier()
            if self.accelerator.is_main_process:
                shutil.rmtree(shard_dir, ignore_errors=True)
        else:
            ordered = torch.empty(n, self.proj_dim)
            ordered[idxs.view(-1)] = grads
            torch.save(ordered, grads_path)
