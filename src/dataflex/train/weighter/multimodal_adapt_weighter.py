"""多模态 ADAPT 重加权: 训练样本与 anchor 集的表征相似度。

参考 ADAPT (ICLR2026): Rethinking Data Curation in LLM Training.
信号不是 loss，而是 sigmoid(mean_cosine(训练样本, anchor 集) / tau)。

核心改造（相对原文纯文本版）：
  - _embed 收整条 batch（包含 pixel_values 等多模态键），不只是 input_ids / attention_mask
  - pool: all / text / answer 三种池化范围
  - weight_norm: mean1 跨卡归一到均值 1（与其他方法同量级）/ none（原文行为）
  - _last_refresh_step 守卫：同一个 optimizer step 的多个 micro-step 只刷新 anchor 一次
"""

from typing import Any, Optional, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dataflex.core.registry import register_weighter
from dataflex.utils.logging import logger
from .base_weighter import Weighter

IGNORE_INDEX = -100


class _AnchorDataset(Dataset):
    """让 DataLoader 能逐条取 anchor 样本（data_collator 会做多模态 batch 组装）。"""
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, i):
        return self.ds[i]


@register_weighter("multimodal_adapt")
class MultimodalAdaptWeighter(Weighter):
    """多模态 ADAPT 重加权。"""

    def __init__(
        self,
        eval_dataset=None,                      # anchor 集（通过 yaml 的 eval_dataset 注入）
        data_collator=None,                     # runtime 注入
        tau: float = 1.0,                       # sigmoid 温度（越小越锐利）
        refresh_interval: int = 50,             # 每多少步刷新 anchor 向量
        anchor_batch_size: int = 8,             # anchor 前向 batch size
        clip: Optional[float] = None,           # 权重上限（None = 不限）
        pool: str = "all",                      # 池化范围: all / text / answer
        weight_norm: str = "mean1",             # mean1 = 跨卡归一到均值 1; none = 原文 sigmoid 绝对值
        image_token_id: Optional[int] = None,   # None 时从 model.config 自动取（pool=text 需要）
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if eval_dataset is None:
            raise ValueError("MultimodalAdaptWeighter requires eval_dataset as the anchor set.")
        if data_collator is None:
            raise ValueError("MultimodalAdaptWeighter requires data_collator.")
        if pool not in ("all", "text", "answer"):
            raise ValueError(f"pool must be one of all/text/answer, got '{pool}'")
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.tau = float(tau)
        self.refresh_interval = int(refresh_interval)
        self.anchor_batch_size = int(anchor_batch_size)
        self.clip = clip
        self.pool = pool
        self.weight_norm = weight_norm
        self.image_token_id = image_token_id
        self.eps = float(eps)

        self.anchor_emb = None              # (M, H) L2 归一化后的 anchor 向量
        self._last_refresh_step = -1        # 同一 step 的多个 micro-step 只刷一次
        logger.info(f"[MultimodalAdapt] tau={tau}, refresh={refresh_interval}, pool={pool}, "
                    f"weight_norm={weight_norm}, anchors={len(eval_dataset)}")

    # ---- 句向量 ----

    def _pool_mask(self, batch):
        """返回参与池化的位置 (B, L) bool。"""
        keep = batch["attention_mask"] != 0
        if self.pool == "text":
            if self.image_token_id is None:
                raise RuntimeError("pool='text' requires image_token_id.")
            keep = keep & (batch["input_ids"] != self.image_token_id)
        elif self.pool == "answer":
            keep = keep & (batch["labels"] != IGNORE_INDEX)
        return keep

    @torch.no_grad()
    def _embed(self, model, batch):
        """位置加权均值池化 + L2 归一化 → 句向量 (B, H)。

        整条 batch（含 pixel_values）喂给模型，多模态键自动带上。
        位置权重线性递增（1, 2, 3, ...），越靠后权重越大。
        """
        model_inputs = {k: v for k, v in batch.items() if k not in ("labels", "idx")}
        hidden = model(**model_inputs, output_hidden_states=True, use_cache=False).hidden_states[-1]
        keep = self._pool_mask(batch)[:, :hidden.size(1)].to(hidden.dtype)
        pos = torch.arange(1, hidden.size(1) + 1, device=hidden.device, dtype=hidden.dtype) * keep
        w = pos / pos.sum(dim=1, keepdim=True).clamp_min(self.eps)
        phi = (w.unsqueeze(-1) * hidden).sum(dim=1)
        return phi / phi.norm(dim=-1, keepdim=True).clamp_min(self.eps)

    @torch.no_grad()
    def _refresh_anchors(self, model):
        """用当前模型重算 anchor 集的句向量。每卡各算一遍全集（前向确定性）。"""
        loader = DataLoader(_AnchorDataset(self.eval_dataset), batch_size=self.anchor_batch_size,
                            shuffle=False, collate_fn=self.data_collator)
        device = next(model.parameters()).device
        embs = []
        for b in loader:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            embs.append(self._embed(model, b))
        self.anchor_emb = torch.cat(embs, dim=0)

    # ---- 加权 ----

    def get_weighted_loss(self, losses, *, ctx=None, model=None, inputs=None):
        if (not torch.is_tensor(losses)) or losses.dim() == 0:
            return losses
        losses = losses.view(-1)

        # 自动取 image_token_id
        if self.image_token_id is None and model is not None:
            cfg = getattr(ctx.accelerator.unwrap_model(model), "config", None) if ctx else None
            if cfg:
                self.image_token_id = getattr(cfg, "image_token_id", None) or \
                                      getattr(cfg, "image_token_index", None)

        # 刷新 anchor（同一 step 只刷一次）
        step = ctx.state.global_step if ctx else 0
        if self.anchor_emb is None or (step != self._last_refresh_step
                                        and step % self.refresh_interval == 0):
            self._refresh_anchors(model)
            self._last_refresh_step = step

        # 样本句向量 → 与全部 anchor 的平均 cosine → sigmoid
        phi = self._embed(model, inputs)
        score = (phi @ self.anchor_emb.t().to(phi.dtype)).mean(dim=1)
        weights = torch.sigmoid(score / max(self.tau, self.eps))
        if self.clip is not None:
            weights = weights.clamp(max=self.clip)
        weights = weights.detach().to(losses.dtype)

        # 跨卡归一到均值 1
        if self.weight_norm == "mean1":
            distributed = dist.is_available() and dist.is_initialized()
            if distributed:
                stat = torch.stack([weights.sum(),
                                    torch.tensor(float(weights.numel()), device=weights.device, dtype=weights.dtype)])
                dist.all_reduce(stat, op=dist.ReduceOp.SUM)
                mean_w = stat[0] / stat[1].clamp_min(1)
            else:
                mean_w = weights.mean()
            weights = weights / mean_w.clamp_min(self.eps)

        if ctx and ctx.args.local_rank in [-1, 0]:
            logger.info(f"[MultimodalAdapt] weights first={float(weights[0]):.4f}, "
                        f"batch mean={float(weights.mean()):.4f}")

        if self.weight_norm == "mean1":
            return torch.mean(weights * losses)
        return torch.sum(weights * losses)
