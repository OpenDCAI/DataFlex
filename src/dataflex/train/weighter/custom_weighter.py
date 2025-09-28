from dataflex.core.registry import register_weighter
from dataflex.utils.logging import logger
from typing import Any, Union
from torch import nn
import torch
from .base_weighter import Weighter

@register_weighter("custom")
class CustomWeighter(Weighter):
    def __init__(self, strategy: str = "uniform", **kwargs):
        """
        自定义加权器的构造函数
        
        Args:
            strategy: 加权策略，如 "uniform"、"loss_based" 等
            **kwargs: 传递给基类的其他参数
        """
        super().__init__(**kwargs)
        self.strategy = strategy
        logger.info(f"CustomWeighter initialized with strategy: {strategy}")
    
    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        *,
        ctx: Any = None,
        model: nn.Module | None = None,
        inputs: dict[str, Union[torch.Tensor, Any]] | None = None,
    ) -> torch.Tensor:
        """
        核心加权逻辑。
        根据样本损失计算加权后的总损失。
        
        Args:
            losses: 本卡的 per-sample loss (B,)
            ctx: Trainer 上下文，可获取 global_step 等信息
            model: 当前模型
            inputs: 输入数据
            
        Returns:
            加权后的总损失（标量）
        """
        # 示例逻辑：简单的均匀加权
        if not torch.is_tensor(losses) or losses.dim() == 0:
            return losses
            
        # 这里可以实现您的自定义加权策略
        # 例如：基于损失大小、梯度信息、样本难度等
        weights = torch.ones_like(losses) / losses.numel()
        weighted_loss = torch.sum(weights * losses)
        
        return weighted_loss
