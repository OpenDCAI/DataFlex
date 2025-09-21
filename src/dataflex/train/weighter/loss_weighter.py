from dataflex.core.registry import register_weighter
from dataflex.utils.logging import logger
from typing import Any, Union
from torch import nn
from transformers.trainer_utils import is_sagemaker_mp_enabled
from transformers.trainer_pt_utils import smp_forward_backward
from transformers.integrations import DistributedType
from transformers.optimization import OptimizerNames
from transformers.utils import is_apex_available
if is_apex_available():
    from apex import amp
import torch
import torch.distributed as dist


@register_weighter('loss')
class LossWeighter:
    """
    可扩展的加权器：
    - 保持 training_step 在 weighter 里：拿得到 ctx(Trainer-like)、model、inputs，方便未来实现基于梯度/激活等信号的加权。
    - get_weighted_loss 严格按参考流程：gather→normalize→strategy→exp→softmax→slice(local)→sum×world_size。
    """

    # ====== 配置 ======
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

    # ====== 关键：分布式加权（严格按参考顺序） ======
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

    # ====== 保持“HF 版 training_step 流程”，但放在 weighter 里 ======
    # 这样未来你可以在 weighter 内部获取 model/ctx/inputs 做更复杂的加权（如基于梯度等）
    def training_step(
        self,
        ctx: Any,  # 必传：HF Trainer 或具有相同接口的上下文
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch=None,
    ) -> torch.Tensor:
        """
        与你现有的 training_step 结构对齐：外部调用 + 内部调用 get_weighted_loss。
        只有在 loss 是向量时才进行分布式加权；否则保持原逻辑。
        """
        logger.info(f"[Datafex] Using WeightTrainer.training_step (weighter-owned)")

        model.train()
        if hasattr(ctx.optimizer, "train") and callable(ctx.optimizer.train):
            ctx.optimizer.train()

        inputs = ctx._prepare_inputs(inputs)

        # SageMaker MP 分支：保持原样（一般难以做逐样本加权）
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, ctx.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(ctx.args.device)

        # 前向 & 计算 loss（可能是标量，可能是 per-sample 向量，取决于你的 compute_loss 实现）
        with ctx.compute_loss_context_manager():
            loss = ctx.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        # 在 weighter 里做分布式加权（仅当 loss 是向量）
        # 输出前5个loss在加权前后的变化情况（只在主进程中输出）
        if ctx.args.local_rank in [-1, 0]:
            logger.info(f"[Datafex] Before weighting, loss[:5]: {loss[:5] if torch.is_tensor(loss) and loss.dim()>0 else loss}")
        loss = self.get_weighted_loss(loss, ctx=ctx, model=model, inputs=inputs)
        if ctx.args.local_rank in [-1, 0]:
            logger.info(f"[Datafex] After weighting, loss[:5]: {loss[:5] if torch.is_tensor(loss) and loss.dim()>0 else loss}")

        # 清理
        del inputs

        # empty cache
        if ctx.args.torch_empty_cache_steps is not None and ctx.state.global_step % ctx.args.torch_empty_cache_steps == 0:
            ctx._empty_cache()

        kwargs = {}

        # LOMO/ADALOMO
        if ctx.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = ctx._get_learning_rate()

        # 多卡场景：保持原有写法（对标量无影响；对我们这里的标量同样无影响）
        if ctx.args.n_gpu > 1:
            loss = loss.mean()

        # 反传（AMP/Accelerate/Deepspeed 逻辑保持一致）
        if getattr(ctx, "use_apex", False):
            with amp.scale_loss(loss, ctx.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            if not getattr(ctx, "model_accepts_loss_kwargs", False) and getattr(ctx, "compute_loss_func", None) is None:
                loss = loss / ctx.args.gradient_accumulation_steps

            if ctx.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            ctx.accelerator.backward(loss, **kwargs)

        return loss.detach()
