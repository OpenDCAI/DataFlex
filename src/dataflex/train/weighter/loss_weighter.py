from dataflex.core.registry import register_weighter
from dataflex.utils.logging import logger
from typing import Any, Union
from torch import nn
from accelerate.utils import DistributedType
from transformers.training_args import OptimizerNames
from transformers.utils import is_apex_available
if is_apex_available():
    from apex import amp
import torch
import torch.distributed as dist


@register_weighter('loss')
class LossWeighter:
    """
    参考论文
    Dynamic Loss-Based Sample Reweighting for Improved Large Language Model Pretraining (ICLR2025)
    """
    def __init__(
        self,
        strategy: str = "linupper",
        delta: float = 1.0,
    ):
        self.strategy = strategy
        self.delta = float(delta)
        self.r = 1

    # ====== 与参考一致的三个函数 ======
    @staticmethod
    def scale_losses(losses: torch.Tensor, r: float) -> torch.Tensor:
        # Exponential scaling: exp(loss / r)
        return torch.exp(losses / r)

    @staticmethod
    def normalize_losses(
        losses: torch.Tensor, delta: float = 1.0, l_min: float = 0.0, l_max: float = 1.0
    ) -> torch.Tensor:
        denom = max(l_max - l_min, 1e-6)
        return 2.0 * delta * losses / denom - delta * (l_max + l_min) / denom

    def apply_strategy(self, losses: torch.Tensor, delta: float = 1.0, strategy: str | None = None) -> torch.Tensor:
        s = strategy or self.strategy
        if s == "linupper":
            return torch.minimum(losses + delta, delta * torch.ones_like(losses))
        elif s == "uniform":
            return losses
        elif s == "quadratic":
            return 1 - (losses ** 2) / (delta ** 2)
        elif s == "extremes":
            return torch.abs(losses)
        else:
            raise NotImplementedError(f"Unknown strategy: {s}")

    def _brief(x):
        if torch.is_tensor(x):
            if x.dim() == 0:
                return float(x.detach().cpu())
            else:
                return x.detach().float().cpu()[:5].tolist()
        return x
    
    def _per_sample_loss_from_logits(self, logits, labels, ignore_index: int = -100):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        num_active = (shift_labels != ignore_index).sum(dim=1)  # (B,)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        tok_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1).long()
        )
        return tok_loss.view(shift_logits.size(0), -1).sum(dim=1) / torch.clamp(num_active, min=1)

    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        *,
        ctx: Any = None,      # 可选：Trainer 上下文，便于获取 global_step 等
        model: nn.Module | None = None,  # 可选：未来扩展用（梯度/激活/正则等）
        inputs: dict[str, Union[torch.Tensor, Any]] | None = None,  # 可选：未来扩展用
    ) -> torch.Tensor:
        """
        参数:
          losses: 本卡的 per-sample loss (B,)。若是标量/非张量，直接原样返回（不加权）。
          ctx: 传入 Trainer（或具备 state/global_step 的对象），用于 r 调度。
        返回:
          标量 loss（已经完成分布式加权，含 ×world_size）。
        """
        # 兼容：标量或非张量 → 不加权
        if (not torch.is_tensor(losses)) or (losses.dim() == 0):
            return losses

        # 确保是一维向量
        if losses.dim() > 1:
            losses = losses.view(-1)

        device_losses = losses  # (B,)
        device = device_losses.device
        dtype = device_losses.dtype

        dist_on = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if dist_on else 1
        local_rank = dist.get_rank() if dist_on else 0

        # 1) 收集本批次所有 GPU 的 per-sample loss -> [W, B]
        if dist_on:
            gathered = torch.zeros(world_size, device_losses.numel(), device=device, dtype=dtype)
            if hasattr(dist, "all_gather_into_tensor"):
                dist.all_gather_into_tensor(gathered, device_losses.detach())
            else:
                bufs = [torch.zeros_like(device_losses) for _ in range(world_size)]
                dist.all_gather(bufs, device_losses.detach())
                gathered = torch.stack(bufs, dim=0)
        else:
            gathered = device_losses.detach().unsqueeze(0)  # [1, B]

        r = self.r

        # 3) 规范化 → 策略加权 → 指数缩放（数值稳定：先减 max）→ 归一化 → 切回本卡
        with torch.no_grad():
            min_loss = gathered.min().item()
            max_loss = gathered.max().item()

            normalized = self.normalize_losses(
                gathered.view(-1), delta=1.0, l_min=min_loss, l_max=max_loss
            )
            reweighted = self.apply_strategy(normalized, delta=1.0, strategy=self.strategy)

            centered = reweighted - float(reweighted.max().item())  # 数值稳定
            scaled = self.scale_losses(centered, r=r)

            weights = scaled / torch.clamp(scaled.sum(), min=1e-12)
            device_weights = weights.view(world_size, -1)[local_rank, :]  # (B,)

        # 4) 本卡加权并 × world_size（保持与参考实现一致）
        return torch.sum(device_weights * device_losses) * world_size

    def training_step(self, ctx, model, inputs, num_items_in_batch=None, use_weighter=False):
        model.train()
        if hasattr(ctx.optimizer, "train") and callable(ctx.optimizer.train):
            ctx.optimizer.train()

        inputs = ctx._prepare_inputs(inputs)

        # 预先保存一份 labels（防止某些实现里被 pop 掉）
        labels_for_weighter = inputs.get("labels", None)

        with ctx.compute_loss_context_manager():
            # 关键：拿到 outputs
            loss, outputs = ctx.compute_loss(
                model, inputs, num_items_in_batch=num_items_in_batch, return_outputs=True
            )

        if use_weighter:
            # 1) 如果 compute_loss 已经返回的是 (B,) 向量，直接用
            if torch.is_tensor(loss) and loss.dim() == 1:
                per_sample = loss
            else:
                # 2) 否则用 logits+labels 现场算每样本 loss（不需要二次前向）
                logits = getattr(outputs, "logits", None) if outputs is not None else None
                labels = inputs.get("labels", None)
                if labels is None:
                    labels = labels_for_weighter
                per_sample = None
                if logits is not None and labels is not None:
                    per_sample = self._per_sample_loss_from_logits(logits, labels)

            if per_sample is not None:
                # 日志仅主进程打
                if ctx.args.local_rank in [-1, 0]:
                    ps = per_sample.detach().float().cpu().view(-1)[0]
                    logger.info(f"[Datafex] Before weighting per-sample (first sample): {ps}")
                # 分布式加权（你的 get_weighted_loss 实现）
                loss = self.get_weighted_loss(per_sample, ctx=ctx, model=model, inputs=inputs)
                if ctx.args.local_rank in [-1, 0]:
                    logger.info(f"[Datafex] After weighting (first sample): {float(loss.detach().cpu())}")
            else:
                if ctx.args.local_rank in [-1, 0]:
                    logger.info("[Datafex] Could not form per-sample losses; fallback to scalar loss (no reweight).")

        del inputs

        if ctx.args.torch_empty_cache_steps is not None and ctx.state.global_step % ctx.args.torch_empty_cache_steps == 0:
            ctx._empty_cache()

        kwargs = {}
        if ctx.args.n_gpu > 1:
            loss = loss.mean()

        if getattr(ctx, "use_apex", False):
            with amp.scale_loss(loss, ctx.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            # 注意：尽量按“本次实际累积的 micro-batches 数”来除，如果你已经做了这类修正，请对应替换这里的除数
            loss = loss / ctx.args.gradient_accumulation_steps
            if ctx.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False
            ctx.accelerator.backward(loss, **kwargs)

        return loss.detach()
