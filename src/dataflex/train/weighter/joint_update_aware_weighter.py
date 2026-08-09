from typing import Any, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from dataflex.core.registry import register_weighter
from dataflex.utils.logging import logger

from .base_weighter import Weighter


@register_weighter("joint_update_aware")
class JointUpdateAwareWeighter(Weighter):
    """
    Joint-Update-Aware 数据加权：以样本 embedding 之间的交互（冗余）作为代理，
    在单纯形上求解带熵正则的加权目标：
        max_w  s^T w - (beta / 2) w^T S w + tau * H(w)
    其中 S 是 L2 归一化 embedding 的余弦 Gram 矩阵，s_i = <u, z_i> 为样本对
    anchor(验证)集平均向量 u 的对齐度。唯一内部最优是定点 w = softmax([s - beta S w] / tau)，
    用阻尼迭代求解。
    """

    def __init__(
        self,
        dataset=None,
        eval_dataset=None,
        accelerator=None,
        data_collator=None,
        beta: float = 0.1,                 # 交互/冗余强度
        tau: float = 0.05,                 # 熵温度（需 > 0），越小权重越锐利
        fixed_point_iters: int = 5,        # 阻尼定点迭代次数
        damping: float = 1.0,              # 阻尼系数 rho ∈ (0, 1]，1.0 表示不阻尼
        target_update_step: int = 50,      # 每多少步刷新一次 eval 目标向量
        target_batch_size: int = 1,        # 计算目标向量时的前向 batch 大小
        target_num_batches: int = 1,       # 每次刷新平均多少个 eval batch，<=0 表示遍历整个 eval 集
        embed_normalize: bool = True,      # 是否对 embedding 做 L2 归一化
        pooling: str = "last_token",       # 句向量池化方式：last_token / mean_pool
        embed_layer: int = -1,             # 取哪一层 hidden state，-1 为最后一层
        objective_mode: str = "full",      # full / align_only / diverse_only / uniform
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.eval_dataset = self._unwrap_eval_dataset(eval_dataset)
        self.accelerator = accelerator
        self.data_collator = data_collator

        self.beta = float(beta)
        self.tau = float(tau)
        self.fixed_point_iters = int(fixed_point_iters)
        self.damping = float(damping)
        self.target_update_step = max(1, int(target_update_step))
        self.target_batch_size = max(1, int(target_batch_size))
        self.target_num_batches = int(target_num_batches)
        self.embed_normalize = bool(embed_normalize)
        self.pooling = str(pooling)
        self.embed_layer = int(embed_layer)
        self.objective_mode = str(objective_mode)

        allowed_modes = {"full", "align_only", "diverse_only", "uniform"}
        if self.objective_mode not in allowed_modes:
            raise ValueError(f"objective_mode 必须是 {allowed_modes} 之一，但得到 {self.objective_mode}")

        allowed_pooling = {"last_token", "mean_pool"}
        if self.pooling not in allowed_pooling:
            raise ValueError(f"pooling 必须是 {allowed_pooling} 之一，但得到 {self.pooling}")

        if self.tau <= 0:
            raise ValueError("tau 必须为正数（熵正则的 softmax 更新要求）。")
        if not 0.0 < self.damping <= 1.0:
            raise ValueError("damping 必须落在 (0, 1] 区间内。")

        self._eval_loader = None
        self._eval_iter = None
        self._cached_target = None
        self._cached_target_step = -1
        self._warned_eval_fallback = False

        logger.info(
            "[Dataflex] JointUpdateAwareWeighter initialized "
            f"(beta={self.beta}, tau={self.tau}, iters={self.fixed_point_iters}, "
            f"damping={self.damping}, pooling={self.pooling}, "
            f"embed_layer={self.embed_layer}, objective_mode={self.objective_mode})."
        )

    # ====== 工具函数 ======
    @staticmethod
    def _unwrap_eval_dataset(eval_dataset):
        if isinstance(eval_dataset, dict):
            if not eval_dataset:
                return None
            first_key = next(iter(eval_dataset))
            logger.info(f"[Dataflex] JointUpdateAwareWeighter using eval_dataset['{first_key}'] as target data.")
            return eval_dataset[first_key]
        return eval_dataset

    @staticmethod
    def _dist_info():
        dist_on = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if dist_on else 1
        rank = dist.get_rank() if dist_on else 0
        return dist_on, world_size, rank

    # ====== Embedding 提取 ======
    def _pool_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        labels: torch.Tensor | None,
    ) -> torch.Tensor:
        """将 (B, T, D) 的 hidden states 池化为每样本句向量 (B, D)。"""
        B, T, D = hidden_states.shape

        if self.pooling == "last_token":
            if attention_mask is not None:
                last_idx = (attention_mask.sum(dim=1).long() - 1).clamp(min=0)  # (B,)
            else:
                last_idx = torch.full((B,), T - 1, device=hidden_states.device, dtype=torch.long)
            return hidden_states[torch.arange(B, device=hidden_states.device), last_idx]  # (B, D)

        # mean_pool：优先在 response token（labels != -100）上取均值，否则在非 padding token 上取
        if labels is not None:
            mask = (labels != -100).float()  # (B, T)
            if mask.size(1) != T:  # labels 相对 hidden states 可能有偏移
                if mask.size(1) > T:
                    mask = mask[:, :T]
                else:
                    pad = torch.zeros(B, T - mask.size(1), device=mask.device)
                    mask = torch.cat([mask, pad], dim=1)
        elif attention_mask is not None:
            mask = attention_mask.float()
        else:
            mask = torch.ones(B, T, device=hidden_states.device)

        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
        return (hidden_states * mask.unsqueeze(-1)).sum(dim=1) / denom  # (B, D)

    def _extract_embeddings(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        """前向一次，提取 detach 后的每样本句向量 (B, D)。"""
        batch = {
            key: value.to(device) if torch.is_tensor(value) and value.device != device else value
            for key, value in inputs.items()
        }

        # 取 embedding 只是为了构造 S 与 s，必须走 eval 模式：train 模式会打开 dropout，
        # 让同一批样本的余弦 Gram 矩阵带上随机噪声。与 _compute_eval_target 保持一致。
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                outputs = model(
                    **{k: v for k, v in batch.items() if k != "labels"},
                    labels=batch.get("labels"),
                    output_hidden_states=True,
                )
        finally:
            if was_training:
                model.train()

        layer_hidden = outputs.hidden_states[self.embed_layer]  # (B, T, D)
        embeddings = self._pool_hidden_states(layer_hidden, batch.get("attention_mask"), batch.get("labels"))
        return embeddings.detach().float()

    # ====== 归一化与跨卡收集 ======
    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        if not self.embed_normalize:
            return features
        return F.normalize(features, p=2, dim=-1, eps=1e-12)

    def _gather_features(self, local_features: torch.Tensor):
        dist_on, world_size, rank = self._dist_info()
        local_count = torch.tensor([local_features.size(0)], device=local_features.device, dtype=torch.long)
        if not dist_on:
            return local_features, [int(local_count.item())], rank, world_size

        count_bufs = [torch.zeros_like(local_count) for _ in range(world_size)]
        dist.all_gather(count_bufs, local_count)
        counts = [int(x.item()) for buf in count_bufs for x in buf]
        max_count = max(counts)

        if local_features.size(0) < max_count:
            pad = torch.zeros(
                max_count - local_features.size(0),
                local_features.size(1),
                device=local_features.device,
                dtype=local_features.dtype,
            )
            padded = torch.cat([local_features, pad], dim=0)
        else:
            padded = local_features

        feat_bufs = [torch.zeros_like(padded) for _ in range(world_size)]
        dist.all_gather(feat_bufs, padded)
        global_features = torch.cat([buf[:count] for buf, count in zip(feat_bufs, counts)], dim=0)
        return global_features, counts, rank, world_size

    # ====== eval / 目标向量逻辑 ======
    def _get_eval_loader(self):
        if self.eval_dataset is None or self.data_collator is None:
            return None
        if self._eval_loader is None:
            self._eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=self.target_batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=self.data_collator,
            )
            self._eval_iter = iter(self._eval_loader)
        return self._eval_loader

    def _next_eval_batch(self):
        loader = self._get_eval_loader()
        if loader is None:
            return None
        try:
            return next(self._eval_iter)
        except StopIteration:
            self._eval_iter = iter(loader)
            return next(self._eval_iter)

    def _embed_eval_batch(self, ctx, model: nn.Module, batch, device):
        """对单个 eval batch 前向取句向量，返回 (sum_of_embeddings, num_samples)。"""
        if ctx is not None and hasattr(ctx, "_prepare_inputs"):
            batch = ctx._prepare_inputs(batch)
        else:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(
                **{k: v for k, v in batch.items() if k != "labels"},
                labels=batch.get("labels"),
                output_hidden_states=True,
            )

        layer_hidden = outputs.hidden_states[self.embed_layer]
        embeddings = self._pool_hidden_states(layer_hidden, batch.get("attention_mask"), batch.get("labels"))
        embeddings = embeddings.detach().float()
        return embeddings.sum(dim=0), embeddings.size(0)

    def _compute_eval_target(self, ctx, model: nn.Module, device):
        """计算 anchor/eval 集的平均句向量 u，作为对齐方向。

        每次刷新平均 ``target_num_batches`` 个 eval batch；设为 <=0 时遍历整个 eval 集。
        默认只取 1 个 batch，因此 u 的方差较大——若 eval 集较小，建议把
        ``target_num_batches`` 设为 0 以使用全集均值。
        """
        if self.eval_dataset is None or self.data_collator is None:
            return None

        loader = self._get_eval_loader()
        if loader is None:
            return None

        if self.target_num_batches <= 0:
            num_batches = len(loader)
            # 遍历全集时从头开始，保证每次刷新覆盖同一批 anchor 样本
            self._eval_iter = iter(loader)
        else:
            num_batches = self.target_num_batches

        was_training = model.training
        model.eval()
        try:
            total = None
            count = 0
            for _ in range(max(1, num_batches)):
                batch = self._next_eval_batch()
                if batch is None:
                    break
                batch_sum, batch_n = self._embed_eval_batch(ctx, model, batch, device)
                total = batch_sum if total is None else total + batch_sum
                count += batch_n
            if total is None or count == 0:
                return None
            target = total / count  # (D,) 按样本数加权的真实均值
        finally:
            if was_training:
                model.train()

        dist_on, world_size, _ = self._dist_info()
        if dist_on:
            dist.all_reduce(target, op=dist.ReduceOp.SUM)
            target.div_(world_size)
        return target

    def _get_target(self, ctx, model: nn.Module, device, global_features: torch.Tensor):
        """取对齐方向 u：每 target_update_step 步用 eval 集重算一次并缓存。"""
        step = int(getattr(getattr(ctx, "state", None), "global_step", 0) or 0)
        should_refresh = (
            self._cached_target is None
            or self._cached_target_step < 0
            or step - self._cached_target_step >= self.target_update_step
        )
        if should_refresh:
            target = self._compute_eval_target(ctx, model, device)
            if target is None:
                if not self._warned_eval_fallback:
                    logger.warning(
                        "[Dataflex] eval_dataset/data_collator 不可用，回退为使用当前 batch 的平均 embedding 作为目标。"
                    )
                    self._warned_eval_fallback = True
                target = global_features.mean(dim=0)
            self._cached_target = target.detach()
            self._cached_target_step = step
        target = self._cached_target.to(device)

        if self.embed_normalize:
            target = F.normalize(target, p=2, dim=0, eps=1e-12)
        return target

    # ====== 带熵正则的定点求解 ======
    def _solve_weights(self, s: torch.Tensor, k_matrix: torch.Tensor):
        """阻尼定点迭代求解 w_i = softmax([s - beta * S w] / tau)_i。"""
        n = s.numel()
        if n <= 0:
            return s
        if self.objective_mode == "uniform":
            return torch.full_like(s, 1.0 / n)

        weights = torch.full_like(s, 1.0 / n)
        for _ in range(max(1, self.fixed_point_iters)):
            if self.objective_mode == "align_only":
                score = s
            elif self.objective_mode == "diverse_only":
                score = -self.beta * (k_matrix @ weights)
            else:
                score = s - self.beta * (k_matrix @ weights)

            score = score - score.max()  # 数值稳定
            proposal = torch.softmax(score / self.tau, dim=0)
            weights = (1.0 - self.damping) * weights + self.damping * proposal

        return weights.detach()

    # ====== 主入口 ======
    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        *,
        ctx: Any = None,
        model: nn.Module | None = None,
        inputs: dict[str, Union[torch.Tensor, Any]] | None = None,
    ) -> torch.Tensor:
        # 兼容：标量或非张量 → 不加权
        if (not torch.is_tensor(losses)) or losses.dim() == 0:
            return losses
        if model is None or inputs is None:
            return losses.mean() if losses.dim() > 0 else losses

        if losses.dim() > 1:
            losses = losses.view(-1)

        device = losses.device

        # 从 trainer 包装的模型上取 embedding（自动处理 DDP/FSDP 的 unwrap）
        feature_model = model
        if ctx is not None:
            candidate = getattr(ctx, "model_wrapped", None) or getattr(ctx, "model", None)
            if candidate is not None:
                feature_model = candidate

        local_features = self._extract_embeddings(feature_model, inputs, device)
        local_features = self._normalize_features(local_features)
        global_features, counts, rank, world_size = self._gather_features(local_features)

        if global_features.size(0) <= 1:
            return losses.mean()

        target = self._get_target(ctx, feature_model, device, global_features)

        with torch.no_grad():
            s = global_features @ target                     # 每样本效用 s_i = <u, z_i>
            k_matrix = global_features @ global_features.T    # 余弦 Gram 矩阵 S = Z Z^T
            weights = self._solve_weights(s, k_matrix)

            start = sum(counts[:rank])
            end = start + losses.numel()
            local_weights = weights[start:end].to(device=device, dtype=losses.dtype)

        if ctx is not None and ctx.args.local_rank in [-1, 0]:
            logger.info(f"[Dataflex] JointUpdateAware weights (first sample): {float(local_weights[0])}")

        # DDP 会对各卡梯度取平均，这里 × world_size 让全局加权目标的尺度与单机一致
        return torch.sum(local_weights * losses) * world_size
