from dataflex.core.registry import register_weighter
from dataflex.utils.logging import logger
from typing import Any, Optional, Union
from torch import nn
from torch.utils.data import DataLoader
import torch
from .base_weighter import Weighter


@register_weighter('adapt')
class AdaptWeighter(Weighter):
    """
    参考论文
    Rethinking Data Curation in LLM Training: Online Reweighting Offers Better Generalization (ICLR2026)
    以「训练样本与 anchor(验证)集的表征相似度」作为质量信号，经温度化 sigmoid 得到每样本的绝对权重。
    """
    def __init__(
        self,
        tau: float = 1.0,                # 温度，越小权重区分越锐利
        refresh_interval: int = 50,      # 每多少步用当前模型刷新一次 anchor 向量
        anchor_batch_size: int = 8,      # 计算句向量时的前向 batch 大小
        clip: Optional[float] = None,    # 可选权重上限，防梯度爆炸
        eps: float = 1e-8,
        eval_dataset=None,
        data_collator=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if eval_dataset is None:
            raise ValueError("AdaptWeighter 需要 anchor 集，请在配置中提供 eval_dataset。")
        self.tau = float(tau)
        self.refresh_interval = int(refresh_interval)
        self.anchor_batch_size = int(anchor_batch_size)
        self.clip = clip
        self.eps = float(eps)
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.anchor_emb = None  # (M, H) 归一化后的 anchor 向量，每 refresh_interval 步刷新

    @torch.no_grad()
    def _embed(self, model, input_ids, attention_mask):
        """位置加权均值池化 + L2 归一化，得到句向量 (B, H)。越靠后的 token 权重越大。"""
        hidden = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True).hidden_states[-1]  # (B, L, H)
        mask = attention_mask.to(hidden.dtype)
        pos = torch.arange(1, hidden.size(1) + 1, device=hidden.device, dtype=hidden.dtype) * mask
        w = pos / pos.sum(dim=1, keepdim=True).clamp_min(self.eps)   # (B, L)
        phi = (w.unsqueeze(-1) * hidden).sum(dim=1)                  # (B, H)
        return phi / phi.norm(dim=-1, keepdim=True).clamp_min(self.eps)

    @torch.no_grad()
    def _refresh_anchors(self, model):
        """用当前模型重算 anchor 集的句向量。"""
        loader = DataLoader(self.eval_dataset, batch_size=self.anchor_batch_size,
                            collate_fn=self.data_collator)
        device = next(model.parameters()).device
        embs = [self._embed(model, b["input_ids"].to(device), b["attention_mask"].to(device))
                for b in loader]
        self.anchor_emb = torch.cat(embs, dim=0)  # (M, H)

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
        losses = losses.view(-1)

        # 每 refresh_interval 步用当前模型刷新 anchor 向量（在线更新）
        step = ctx.state.global_step if ctx is not None else 0
        if self.anchor_emb is None or step % self.refresh_interval == 0:
            self._refresh_anchors(model)

        # 样本句向量 → 与 anchor 的平均余弦相似度 → 温度化 sigmoid 得到绝对权重
        phi = self._embed(model, inputs["input_ids"], inputs["attention_mask"])  # (B, H)
        score = (phi @ self.anchor_emb.t().to(phi.dtype)).mean(dim=1)            # (B,)
        weights = torch.sigmoid(score / max(self.tau, self.eps))                # (B,)
        if self.clip is not None:
            weights = weights.clamp(max=self.clip)

        if ctx is not None and ctx.args.local_rank in [-1, 0]:
            logger.info(f"[Dataflex] ADAPT weights (first sample): {float(weights[0])}")

        # 权重作为常量缩放每样本 loss（per-sample learning rate）
        return torch.sum(weights.detach().to(losses.dtype) * losses)
