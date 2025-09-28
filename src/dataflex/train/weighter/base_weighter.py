from abc import ABC, abstractmethod
from typing import Any, Union
from torch import nn
import torch


class Weighter(ABC):
    """
    数据加权器的抽象基类，定义了加权器的基本接口和公共功能。
    """
    
    def __init__(self, **kwargs):
        """
        基类构造函数
        
        Args:
            **kwargs: 子类特定的参数
        """
        # 子类可以在这里定义公共的初始化逻辑
        pass
    
    def _per_sample_loss_from_logits(self, logits, labels, ignore_index: int = -100):
        """
        从 logits 和 labels 计算每个样本的损失
        
        Args:
            logits: 模型输出的 logits
            labels: 真实标签
            ignore_index: 忽略的标签索引
            
        Returns:
            torch.Tensor: 每个样本的损失 (B,)
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        num_active = (shift_labels != ignore_index).sum(dim=1)  # (B,)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        tok_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1).long()
        )
        return tok_loss.view(shift_logits.size(0), -1).sum(dim=1) / torch.clamp(num_active, min=1)
    
    @abstractmethod
    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        *,
        ctx: Any = None,
        model: nn.Module | None = None,
        inputs: dict[str, Union[torch.Tensor, Any]] | None = None,
    ) -> torch.Tensor:
        """
        核心加权方法，子类必须实现此方法
        
        Args:
            losses: 本卡的 per-sample loss (B,)
            ctx: Trainer 上下文，可获取 global_step 等信息
            model: 当前模型
            inputs: 输入数据
            
        Returns:
            torch.Tensor: 加权后的总损失（标量）
        """
        pass
    
    def training_step(self, ctx, model, inputs, num_items_in_batch=None, use_weighter=False):
        """
        执行训练步骤，包含前向传播、损失计算、加权和反向传播
        
        Args:
            ctx: Trainer 上下文
            model: 模型
            inputs: 输入数据
            num_items_in_batch: 批次中的样本数量
            use_weighter: 是否使用加权器
            
        Returns:
            本步骤的损失值
        """
        from dataflex.utils.logging import logger
        from transformers.utils import is_apex_available
        from accelerate.utils import DistributedType
        
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
                    logger.info(f"[Dataflex] Before weighting per-sample (first sample): {ps}")
                # 分布式加权
                loss = self.get_weighted_loss(per_sample, ctx=ctx, model=model, inputs=inputs)
                if ctx.args.local_rank in [-1, 0]:
                    logger.info(f"[Dataflex] After weighting (first sample): {float(loss.detach().cpu())}")
            else:
                if ctx.args.local_rank in [-1, 0]:
                    logger.info("[Dataflex] Could not form per-sample losses; fallback to scalar loss (no reweight).")

        del inputs

        if ctx.args.torch_empty_cache_steps is not None and ctx.state.global_step % ctx.args.torch_empty_cache_steps == 0:
            ctx._empty_cache()

        kwargs = {}
        if ctx.args.n_gpu > 1:
            loss = loss.mean()

        if getattr(ctx, "use_apex", False):
            if is_apex_available():
                from apex import amp
                with amp.scale_loss(loss, ctx.optimizer) as scaled_loss:
                    scaled_loss.backward()
        else:
            loss = loss / ctx.args.gradient_accumulation_steps
            if ctx.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False
            ctx.accelerator.backward(loss, **kwargs)

        return loss.detach()
