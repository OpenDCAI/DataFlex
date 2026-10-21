"""ClusterLoss: 聚类 + forward-only loss 评分 + softmax 采样。

最便宜的 cluster 方法：不需要梯度、不需要 eval_dataset。
只用代表点的前向 loss 给每个簇打分，z-score 归一化后 softmax 采样。
"""

import os
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataflex.core.registry import register_selector
from dataflex.utils.logging import logger
from dataflex.utils.selector_io import load_cached_selection, save_selection

from .cluster_base_selector import ClusterBaseSelector, _move_to_device


@register_selector("cluster_loss")
class ClusterLossSelector(ClusterBaseSelector):

    def __init__(
        self,
        # ---- ClusterBaseSelector 参数 ----
        dataset, accelerator, data_collator, cache_dir,
        cluster_feature="embedding", embedding_model_name_or_path=None,
        embedding_model_dtype="bf16", embedding_layer=-1,
        gradient_target_modules=None, gradient_proj_dim=2048,
        cluster_size=64, clustering_max_iter=300, kmeans_n_init=5,
        samples_per_cluster=6, seed=3,
        # ---- 评分参数 ----
        standardize: bool = True,       # 簇 loss 做 z-score 归一化
        score_clip: float = 3.0,        # z-score 截断
        temperature: float = 1.0,       # softmax 温度
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
        self.standardize = standardize
        self.score_clip = float(score_clip)
        self.temperature = float(temperature)
        self.replacement = replacement
        self.dedup_across_rounds = dedup_across_rounds
        self._selected_history: set = set()

    def select(self, model, step_id: int, num_samples: int, **kwargs) -> List[int]:
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

        # 1. 聚类 + 代表点
        cluster_ids = self.get_or_create_clusters(model)
        representatives = self.sample_representatives(cluster_ids, step_id)

        # 2. 代表点 forward-only loss（所有卡参与，分布式归约）
        rep_losses = self._representative_losses(model, representatives)

        # 3. 评分 + 采样（rank 0）
        if self.accelerator.is_main_process:
            num_clusters = int(cluster_ids.max().item()) + 1
            # 簇均值 loss
            cluster_loss = torch.zeros(num_clusters)
            counts = torch.zeros(num_clusters)
            rep_cids = cluster_ids[torch.tensor(representatives, dtype=torch.long)]
            cluster_loss.index_add_(0, rep_cids, rep_losses)
            counts.index_add_(0, rep_cids, torch.ones(len(representatives)))
            # 空簇用全局均值兜底
            empty = counts == 0
            counts = counts.clamp(min=1)
            cluster_loss /= counts
            if empty.any():
                cluster_loss[empty] = cluster_loss[~empty].mean()

            # z-score 归一化 + 截断（簇数 > 1 时才有意义）
            signal = cluster_loss
            if self.standardize and num_clusters > 1:
                mu, std = signal.mean(), signal.std().clamp(min=1e-6)
                signal = ((signal - mu) / std).clamp(-self.score_clip, self.score_clip)

            # softmax → per-sample prob → multinomial
            probs = torch.softmax(signal / self.temperature, dim=0)
            sample_probs = probs[cluster_ids]

            if self.dedup_across_rounds and self._selected_history:
                for idx in self._selected_history:
                    if idx < len(sample_probs):
                        sample_probs[idx] = 0

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
                           {"cluster_loss": cluster_loss.tolist()[:20]}, self.accelerator)
            logger.info(f"[ClusterLoss] selected {len(selected)} samples, clusters={num_clusters}")
        else:
            selected = None

        return self._broadcast_long_tensor(
            torch.tensor(selected, dtype=torch.long) if selected is not None else None
        ).tolist()

    # ============================================================
    # 代表点 loss 计算
    # ============================================================

    def _representative_losses(self, model, rep_indices: List[int]) -> torch.Tensor:
        """forward-only 计算代表点的 per-sample token-mean CE。

        不传 labels 给模型（避开 Liger 融合 CE），手动从 logits 算。
        所有卡参与、分布式 all_reduce 求和。
        """
        distributed = dist.is_available() and dist.is_initialized()
        world = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        device = self.device

        # 每卡分一段
        my_indices = rep_indices[rank::world]
        rep_dataset = Subset(self.dataset, my_indices)
        loader = DataLoader(rep_dataset, batch_size=8, shuffle=False,
                            num_workers=0, collate_fn=self.data_collator)

        model.eval()
        losses_local = torch.zeros(len(rep_indices), device=device)
        with torch.no_grad():
            offset = 0
            for batch in loader:
                batch = _move_to_device(batch, device)
                labels = batch.pop("labels", None)
                logits = model(**batch).logits
                if labels is None:
                    offset += logits.size(0)
                    continue
                # 手动 per-sample CE
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                for i in range(logits.size(0)):
                    valid = shift_labels[i] != -100
                    if valid.any():
                        ce = F.cross_entropy(shift_logits[i][valid], shift_labels[i][valid])
                        # 写到全局位置
                        global_idx = rank + (offset + i) * world
                        if global_idx < len(rep_indices):
                            losses_local[global_idx] = ce
                offset += logits.size(0)

        if distributed:
            dist.all_reduce(losses_local, op=dist.ReduceOp.SUM)
        return losses_local.cpu()
