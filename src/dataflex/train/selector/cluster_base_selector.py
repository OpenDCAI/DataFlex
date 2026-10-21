"""聚类基类：embedding / gradient 两种特征 + MiniBatchKMeans + 代表点随机采样。

所有 cluster_* selector 都继承这个基类。它只负责「把训练集聚成簇、每簇取几个代表点」，
不含任何评分 / 选择逻辑（那些在各子类的 select() 里自包含）。

聚类只做一次（首轮建、后续从缓存读），代表点每轮重新随机抽。
"""

import hashlib
import math
import os
import shutil
import time
from abc import abstractmethod
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from dataflex.utils.logging import logger
from .base_selector import Selector


# ============================================================
# 工具函数
# ============================================================

def _move_to_device(batch, device):
    """递归地把 dict / list / tuple / tensor 搬到指定 device。"""
    if isinstance(batch, dict):
        return {k: _move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [_move_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(_move_to_device(v, device) for v in batch)
    if hasattr(batch, "to"):
        return batch.to(device)
    return batch


class ClusterBaseSelector(Selector):
    """聚类基类（不注册为 selector，不能直接使用）。

    提供两个公开方法给子类：
      - get_or_create_clusters(model) → cluster_ids[N]
      - sample_representatives(cluster_ids, step_id) → List[int]

    子类在 select() 里调用这两个方法拿到簇和代表点后，自行实现评分 + 采样。
    """

    def __init__(
        self,
        dataset,
        accelerator,
        data_collator,
        cache_dir: str,
        # ---- 聚类特征 ----
        cluster_feature: str = "embedding",         # "embedding": 用 hidden state; "gradient": 用 per-sample 梯度
        embedding_model_name_or_path: Optional[str] = None,  # 独立 embedding 模型；None = 用训练模型
        embedding_model_dtype: str = "bf16",
        embedding_layer: int = -1,                   # 取哪层 hidden state（-1 = 最后一层）
        gradient_target_modules: Optional[list] = None,  # 梯度特征用哪些参数名（None = 全部 requires_grad）
        gradient_proj_dim: int = 2048,               # 梯度 TRAK 投影维度
        # ---- 聚类 ----
        cluster_size: int = 64,                      # 每簇约多少样本（num_clusters = N / cluster_size）
        clustering_max_iter: int = 300,
        kmeans_n_init: int = 5,
        # ---- 代表点 ----
        samples_per_cluster: int = 6,                # 每簇取几个代表点
        seed: int = 3,
        **kwargs,
    ):
        super().__init__(dataset=dataset, accelerator=accelerator,
                         data_collator=data_collator, cache_dir=cache_dir)
        self.cluster_feature = cluster_feature
        self.embedding_model_name_or_path = embedding_model_name_or_path
        self.embedding_model_dtype = embedding_model_dtype
        self.embedding_layer = int(embedding_layer)
        self.gradient_target_modules = gradient_target_modules
        self.gradient_proj_dim = int(gradient_proj_dim)
        self.cluster_size = cluster_size
        self.clustering_max_iter = int(clustering_max_iter)
        self.kmeans_n_init = int(kmeans_n_init)
        self.samples_per_cluster = int(samples_per_cluster)
        self.seed = int(seed)

        # 一次性缓存（内存）
        self._embedding_model = None
        self._cluster_ids: Optional[torch.Tensor] = None
        self._features: Optional[torch.Tensor] = None

    @property
    def device(self):
        return self.accelerator.device

    # ============================================================
    # 公开接口
    # ============================================================

    def get_or_create_clusters(self, model) -> torch.Tensor:
        """返回 cluster_ids[N]（long tensor）。首次调用时提取特征并聚类，之后缓存。"""
        if self._cluster_ids is not None:
            return self._cluster_ids

        cluster_dir = os.path.join(self.cache_dir, "cluster", "init")
        cluster_path = os.path.join(cluster_dir, "train_cluster_ids.pt")

        cached = self._broadcast_bool(
            os.path.exists(cluster_path) if self.accelerator.is_main_process else False
        )

        if cached:
            if self.accelerator.is_main_process:
                logger.info(f"[ClusterBase] loaded clusters from cache: {cluster_path}")
            cluster_ids = None
            if self.accelerator.is_main_process:
                cluster_ids = torch.load(cluster_path, map_location="cpu")
        else:
            # 提取特征
            t0 = time.perf_counter()
            if self.cluster_feature == "gradient":
                features = self._extract_gradient_features(model)
            else:
                features = self._extract_embedding_features(model)
            logger.info(f"[ClusterBase] feature extraction took {time.perf_counter() - t0:.1f}s")

            # 聚类（只在 rank 0）
            cluster_ids = None
            if self.accelerator.is_main_process:
                t0 = time.perf_counter()
                cluster_ids = self._run_kmeans(features)
                os.makedirs(cluster_dir, exist_ok=True)
                torch.save(cluster_ids, cluster_path)
                k = int(cluster_ids.max().item()) + 1
                logger.info(f"[ClusterBase] built {k} clusters in {time.perf_counter() - t0:.1f}s")

        self._cluster_ids = self._broadcast_long_tensor(cluster_ids)
        return self._cluster_ids

    def sample_representatives(self, cluster_ids: torch.Tensor, step_id: int) -> List[int]:
        """每簇随机取 samples_per_cluster 个样本索引。"""
        rng = torch.Generator(device="cpu")
        rng.manual_seed(self.seed + int(step_id) + 17)
        reps = []
        for cid in range(int(cluster_ids.max().item()) + 1):
            members = torch.where(cluster_ids == cid)[0]
            if len(members) == 0:
                continue
            take = min(self.samples_per_cluster, len(members))
            perm = torch.randperm(len(members), generator=rng)
            reps.extend(members[perm[:take]].tolist())
        return reps

    @abstractmethod
    def select(self, model, step_id: int, num_samples: int, **kwargs) -> List[int]:
        """子类必须实现：评分 + 采样。"""
        ...

    # ============================================================
    # 特征提取
    # ============================================================

    def _extract_embedding_features(self, model) -> torch.Tensor:
        """用独立 embedding 模型（或训练模型）取 hidden state → mean pooling → L2 normalize。

        多进程分片提取，通过磁盘 shard 汇合，只在 rank 0 上拼成完整 [N, D] 矩阵。
        """
        feature_path = self._embedding_cache_path()
        cached = self._broadcast_bool(
            os.path.exists(feature_path) if self.accelerator.is_main_process else False
        )
        if cached:
            self.accelerator.wait_for_everyone()
            return torch.load(feature_path, map_location="cpu")

        os.makedirs(os.path.dirname(feature_path), exist_ok=True)
        emb_model = self._get_embedding_model(model)

        # 用 accelerator.prepare 自动分片到各卡
        indexed = list(range(len(self.dataset)))
        def collate_fn(indices):
            examples = [self.dataset[int(i)] for i in indices]
            return torch.tensor(indices, dtype=torch.long), self.data_collator(examples)

        loader = self.accelerator.prepare(
            DataLoader(indexed, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn)
        )

        was_training = emb_model.training
        emb_model.eval()
        feats, idxs = [], []
        with torch.no_grad():
            for indices, batch in tqdm(loader, desc="[ClusterBase] embeddings",
                                       disable=not self.accelerator.is_local_main_process):
                batch = _move_to_device(batch, self.device)
                hidden = emb_model(**batch, output_hidden_states=True).hidden_states[self.embedding_layer]
                mask = batch.get("attention_mask")
                if mask is not None:
                    mask = mask.unsqueeze(-1).float()
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                else:
                    pooled = hidden.mean(1)
                feats.append(pooled.float().cpu())
                idxs.append(indices.cpu())
        if was_training and self.embedding_model_name_or_path is None:
            emb_model.train()

        feats = torch.cat(feats) if feats else torch.empty(0)
        idxs = torch.cat(idxs) if idxs else torch.empty(0, dtype=torch.long)
        return self._gather_and_save_features(feats, idxs, feature_path)

    def _extract_gradient_features(self, model) -> torch.Tensor:
        """用 batch=1 forward+backward 取 per-sample SGD 梯度 → TRAK 随机投影 → L2 normalize。

        只取 gradient_target_modules 匹配的 nn.Linear 层。
        用 TRAK CudaProjector（GPU 分块投影，不需要全矩阵在内存里）。
        """
        feature_path = self._gradient_cache_path()
        cached = self._broadcast_bool(
            os.path.exists(feature_path) if self.accelerator.is_main_process else False
        )
        if cached:
            self.accelerator.wait_for_everyone()
            return torch.load(feature_path, map_location="cpu")

        os.makedirs(os.path.dirname(feature_path), exist_ok=True)

        # 找目标 Linear 层
        target_layers = []
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if self.gradient_target_modules is not None:
                if not any(t in name for t in self.gradient_target_modules):
                    continue
            target_layers.append((name, module))

        if not target_layers:
            raise ValueError("[ClusterBase] gradient_target_modules matched no nn.Linear layers")

        total_params = sum(m.weight.numel() for _, m in target_layers)
        logger.info(f"[ClusterBase] gradient: {len(target_layers)} Linear layers, "
                    f"{total_params:,} params, projected to {self.gradient_proj_dim} dims")

        # TRAK projector（GPU 分块投影，内存安全）
        from .less_selector import _trak_projectors
        BasicProjector, CudaProjector, ProjectionType = _trak_projectors()
        proj_device = "cpu"
        try:
            projector = CudaProjector(
                grad_dim=total_params, proj_dim=self.gradient_proj_dim,
                seed=self.seed + 7, proj_type=ProjectionType.rademacher,
                device=str(self.device), max_batch_size=16,
            )
            proj_device = str(self.device)
        except Exception:
            projector = BasicProjector(
                grad_dim=total_params, proj_dim=self.gradient_proj_dim,
                seed=self.seed + 7, proj_type=ProjectionType.rademacher,
                device="cpu",
            )

        indexed = list(range(len(self.dataset)))
        def collate_fn(indices):
            examples = [self.dataset[int(i)] for i in indices]
            return torch.tensor(indices, dtype=torch.long), self.data_collator(examples)

        loader = self.accelerator.prepare(
            DataLoader(indexed, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
        )

        was_training = model.training
        model.eval()
        feats, idxs = [], []
        for indices, batch in tqdm(loader, desc="[ClusterBase] grad features",
                                    disable=not self.accelerator.is_local_main_process):
            batch = _move_to_device(batch, self.device)
            model.zero_grad()
            with self.accelerator.no_sync(model):
                model(**batch).loss.backward()
            # 拼接目标层的梯度
            grad_parts = []
            for _, module in target_layers:
                if module.weight.grad is not None:
                    grad_parts.append(module.weight.grad.detach().float().view(-1))
                else:
                    grad_parts.append(torch.zeros(module.weight.numel(), device=self.device))
            grad_flat = torch.cat(grad_parts).unsqueeze(0).to(proj_device)  # [1, total_params]
            projected = projector.project(grad_flat, model_id=0)  # [1, proj_dim]
            feats.append(projected.squeeze(0).cpu())
            idxs.append(indices.cpu())
        if was_training:
            model.train()
        model.zero_grad()

        feats = torch.stack(feats) if feats else torch.empty(0, self.gradient_proj_dim)
        idxs = torch.cat(idxs) if idxs else torch.empty(0, dtype=torch.long)
        return self._gather_and_save_features(feats, idxs, feature_path)

    # ============================================================
    # 聚类
    # ============================================================

    def _run_kmeans(self, features: torch.Tensor) -> torch.Tensor:
        """sklearn MiniBatchKMeans，只在 rank 0 调用。"""
        n = len(features)
        k = max(1, min(int(math.ceil(n / max(1, self.cluster_size))), n))
        km = MiniBatchKMeans(
            n_clusters=k,
            max_iter=self.clustering_max_iter,
            n_init=self.kmeans_n_init,
            random_state=self.seed,
            batch_size=min(1024, n),
        )
        km.fit(features.numpy())
        return torch.tensor(km.labels_, dtype=torch.long)

    # ============================================================
    # 内部工具
    # ============================================================

    def _get_embedding_model(self, training_model):
        if self.embedding_model_name_or_path is None:
            return training_model
        if self._embedding_model is not None:
            return self._embedding_model
        dtype_map = {"auto": "auto", "fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        self._embedding_model = AutoModelForCausalLM.from_pretrained(
            self.embedding_model_name_or_path,
            torch_dtype=dtype_map.get(self.embedding_model_dtype, "auto"),
            trust_remote_code=True,
        ).to(self.device).eval()
        return self._embedding_model

    def _embedding_cache_path(self) -> str:
        emb_id = str(self.embedding_model_name_or_path or "train_model")
        h = hashlib.md5(emb_id.encode()).hexdigest()[:10]
        root = os.environ.get("DATAFLEX_EMBEDDING_CACHE_ROOT",
                              os.path.join(self.cache_dir, "embeddings_shared"))
        return os.path.join(root, f"emb_{h}__N{len(self.dataset)}__L{self.embedding_layer}",
                            "train_embeddings.pt")

    def _gradient_cache_path(self) -> str:
        modules_str = ",".join(self.gradient_target_modules or ["all"])
        h = hashlib.md5(modules_str.encode()).hexdigest()[:10]
        root = os.path.join(self.cache_dir, "grad_features_shared")
        return os.path.join(root, f"grad_{h}__N{len(self.dataset)}__D{self.gradient_proj_dim}",
                            "train_grad_features.pt")

    def _gather_and_save_features(self, feats: torch.Tensor, idxs: torch.Tensor,
                                   save_path: str) -> torch.Tensor:
        """多卡分片 → 通过磁盘 shard 汇合 → rank 0 拼成 [N, D] → L2 normalize → 存盘。"""
        n = len(self.dataset)
        distributed = dist.is_available() and dist.is_initialized()

        if distributed:
            shard_dir = os.path.join(os.path.dirname(save_path), "shards")
            os.makedirs(shard_dir, exist_ok=True)
            rank = dist.get_rank()
            torch.save({"f": feats, "i": idxs}, os.path.join(shard_dir, f"shard_{rank}.pt"))
            del feats, idxs
            dist.barrier()

            if self.accelerator.is_main_process:
                ordered = None
                for r in range(dist.get_world_size()):
                    d = torch.load(os.path.join(shard_dir, f"shard_{r}.pt"), map_location="cpu")
                    if d["f"].numel() == 0:
                        continue
                    if ordered is None:
                        ordered = torch.empty(n, d["f"].shape[1], dtype=d["f"].dtype)
                    ordered[d["i"]] = d["f"]
                ordered = F.normalize(ordered, dim=1)
                torch.save(ordered, save_path)
            dist.barrier()
            if self.accelerator.is_main_process:
                shutil.rmtree(shard_dir, ignore_errors=True)
            return torch.load(save_path, map_location="cpu")

        # 单卡
        ordered = torch.empty(n, feats.shape[1], dtype=feats.dtype)
        ordered[idxs] = feats
        ordered = F.normalize(ordered, dim=1)
        torch.save(ordered, save_path)
        return ordered

    def _broadcast_bool(self, value: bool) -> bool:
        if not (dist.is_available() and dist.is_initialized()):
            return value
        obj = [value if self.accelerator.is_main_process else None]
        dist.broadcast_object_list(obj, src=0)
        return bool(obj[0])

    def _broadcast_long_tensor(self, tensor: Optional[torch.Tensor]) -> torch.Tensor:
        if not (dist.is_available() and dist.is_initialized()):
            return tensor
        device = self.device
        length = torch.tensor([tensor.numel() if tensor is not None else 0],
                              dtype=torch.long, device=device)
        dist.broadcast(length, src=0)
        n = int(length.item())
        if self.accelerator.is_main_process:
            payload = tensor.to(device=device, dtype=torch.long)
        else:
            payload = torch.empty(n, dtype=torch.long, device=device)
        dist.broadcast(payload, src=0)
        return payload.cpu()
