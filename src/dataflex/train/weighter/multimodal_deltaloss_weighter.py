"""多模态 DeltaLoss 重加权: s = L(R|Q) − L(R|Q,I)，两次前向。

每 refresh_interval 步，用当前训练中的模型对训练集全量样本做两次前向：
  - 有图：原样输入 → L(R|Q,I)
  - 无图：删掉视觉 token → L(R|Q)
delta = L(R|Q) − L(R|Q,I)，加了图之后 loss 降得越多的样本权重越大。

权重公式（内联，不继承 LossWeighter）：
  分位截断 → min-max 映射到 [-δ,δ] → linupper: min(x+δ, δ) → exp((·−max)/r)
  → 全局归一化到均值 1
"""

import hashlib
import math
import os
from typing import Any, Optional, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dataflex.train.data.loader import split_batch
from dataflex.core.registry import register_weighter
from dataflex.utils.logging import logger
from dataflex.utils.multimodal_scoring import strip_vision_tokens, vision_token_ids
from .base_weighter import IndexedWeightLifecycle, Weighter

IGNORE_INDEX = -100


def _sample_keys(batch):
    """每个样本的 16 字节摘要：去 padding 的 input_ids + 该样本图像（pixel_values 按样本排列时）。"""
    ids, am = batch["input_ids"], batch.get("attention_mask")
    pv = batch.get("pixel_values")
    per_sample_img = pv is not None and pv.dim() == 4 and pv.size(0) == ids.size(0)
    keys = []
    for i in range(ids.size(0)):
        h = hashlib.blake2b(digest_size=16)
        row = ids[i][am[i].bool()] if am is not None else ids[i]
        h.update(row.long().cpu().numpy().tobytes())
        if per_sample_img:
            h.update(pv[i].to(torch.bfloat16).view(torch.int16).cpu().numpy().tobytes())
        keys.append(h.digest())
    return keys


# ---------- 工具 ----------

class _PairedIndexedDataset(Dataset):
    """按 order 依次返回 (原始 idx, 主样本)。"""
    def __init__(self, dataset, order):
        self.dataset = dataset
        self.order = order
    def __len__(self):
        return len(self.order)
    def __getitem__(self, i):
        idx = self.order[i]
        return idx, self.dataset[idx]


@register_weighter("multimodal_deltaloss")
class MultimodalDeltalossWeighter(IndexedWeightLifecycle, Weighter):
    """多模态 DeltaLoss 重加权。"""

    def __init__(
        self,
        dataset=None,                           # 训练集（runtime 注入）
        data_collator=None,                     # runtime 注入
        accelerator=None,                       # runtime 注入
        refresh_interval: int = 244,            # 每多少步全池重新打分
        delta: float = 1.0,                     # 权重管线强度（δ=1 时最大/最小权重差约 2.7 倍）
        r: float = 1.0,                         # exp 的分母
        clip_quantile: float = 0.01,            # min-max 前的分位截断
        score_batch_size: int = 8,              # 打分 batch size
        score_num_workers: int = 4,
        loss_reduction: str = "sample",         # "sample" 或 "token"（对齐 HF 标准 loss）
        cache_dir: Optional[str] = None,        # 每轮分数/权重落盘
        **kwargs,
    ):
        super().__init__(**kwargs)
        if dataset is None or data_collator is None:
            raise ValueError("MultimodalDeltalossWeighter requires dataset and data_collator (injected by WeightTrainer).")
        self.dataset = dataset
        self.data_collator = data_collator
        self.accelerator = accelerator
        self.refresh_interval = int(refresh_interval)
        self.delta = float(delta)
        self.r = float(r)
        self.clip_quantile = float(clip_quantile)
        self.score_batch_size = int(score_batch_size)
        self.score_num_workers = int(score_num_workers)
        self.loss_reduction = loss_reduction
        self.cache_dir = cache_dir

        self.W = None                   # (N,) CPU 权重表，均值 1
        self._weight_map = {}           # {input_ids_bytes: weight_scalar} 用于 training_step 查表
        self._last_refresh_step = -1
        self._order = None              # 打分顺序（按长度降序，长样本先跑）
        self._vision_ids = None         # (vision_start_id, vision_end_id, image_token_id)
        logger.info(f"[MultimodalDeltaloss] refresh_interval={refresh_interval}, delta={delta}, r={r}")

    # ---- 训练入口 ----

    def get_weighted_loss(self, losses, *, ctx=None, model=None, inputs=None):
        if (not torch.is_tensor(losses)) or losses.dim() == 0:
            return losses
        losses = losses.view(-1)
        w = self._batch_weights(inputs, losses, _sample_keys)
        if self.loss_reduction == "token":
            n = (inputs["labels"][..., 1:] != IGNORE_INDEX).sum(dim=1).to(losses.dtype)
            return torch.sum(w * losses * n) / n.sum().clamp_min(1)
        return torch.mean(w * losses)

    # ---- 全量打分 ----

    @torch.no_grad()
    def _refresh(self, ctx, model, step):
        loss_full = self._source_scores("loss_full")
        if loss_full is not None:
            loss_noimg = self._source_scores("loss_no_image")
            score = loss_noimg - loss_full
            self.W = self._score_to_weight(score)
            self._weight_map = {}
            self._log_and_save(step, score, loss_full, loss_noimg)
            return
        acc = ctx.accelerator
        if self._vision_ids is None:
            self._vision_ids = vision_token_ids(acc.unwrap_model(model).config)

        n = len(self.dataset)
        order = self._get_length_order()
        loader = DataLoader(
            _PairedIndexedDataset(self.dataset, order),
            batch_size=self.score_batch_size, shuffle=False,
            num_workers=self.score_num_workers,
            collate_fn=lambda items: {
                "idx": torch.tensor([it[0] for it in items], dtype=torch.long),
                "main": self.data_collator([it[1] for it in items]),
            },
        )
        loader = acc.prepare(loader)

        logger.info(f"[MultimodalDeltaloss] step {step}: scoring {n} samples")
        was_training = model.training
        model.eval()
        idx_l, key_l, full_l, noimg_l = [], [], [], []
        for batch in loader:
            main = batch["main"]
            full_l.append(self._forward_loss(model, main))
            noimg_l.append(self._forward_loss(model, self._strip_vision(main)))
            idx_l.append(batch["idx"])
            if "idx" not in main:
                key_l.append(torch.frombuffer(bytearray(b"".join(_sample_keys(main))), dtype=torch.uint8)
                             .view(-1, 16).to(acc.device))
        if was_training:
            model.train()
        torch.cuda.empty_cache()

        all_idx = acc.gather(torch.cat(idx_l)).cpu()
        all_keys = acc.gather(torch.cat(key_l)).cpu().numpy() if key_l else None
        def _align(parts):
            vals = acc.gather(torch.cat(parts)).float().cpu()
            out = torch.full((n,), float("nan"))
            out[all_idx] = vals
            return out

        loss_full = _align(full_l)
        loss_noimg = _align(noimg_l)
        score = loss_noimg - loss_full  # delta: 加了图 loss 降得越多越高

        self.W = self._score_to_weight(score)
        self._weight_map = ({k.tobytes(): float(self.W[i]) for k, i in zip(all_keys, all_idx.tolist())}
                            if all_keys is not None else {})
        self._log_and_save(step, score, loss_full, loss_noimg)

    # ---- 权重公式（内联） ----

    def _score_to_weight(self, score):
        valid = torch.isfinite(score)
        if not valid.any():
            raise ValueError("Cannot refresh weights: no finite answer-token losses")
        s = score.clone()
        lo = torch.quantile(s[valid], self.clip_quantile).item()
        hi = torch.quantile(s[valid], 1 - self.clip_quantile).item()
        s = s.clamp(lo, hi)
        denom = max(hi - lo, 1e-6)
        x = 2 * self.delta * s / denom - self.delta * (hi + lo) / denom  # normalize to [-delta, delta]
        x = torch.minimum(x + self.delta, self.delta * torch.ones_like(x))  # linupper
        w = torch.exp((x - x[valid].max()) / self.r)
        w = w / w[valid].mean()
        w[~valid] = 1.0
        return w

    # ---- 前向 ----

    def _forward_loss(self, model, batch):
        inputs, _ = split_batch(batch)
        inputs.pop("labels", None)
        logits = model(**inputs, use_cache=False).logits
        return self._answer_loss(logits, batch["labels"])

    def _answer_loss(self, logits, labels):
        """per-sample answer-token-mean CE (B,)"""
        shift_logits, shift_labels = logits[:, :-1], labels[:, 1:]
        out = []
        for i in range(labels.size(0)):
            m = shift_labels[i] != IGNORE_INDEX
            if m.any():
                out.append(F.cross_entropy(shift_logits[i][m].float(), shift_labels[i][m]))
            else:
                out.append(torch.tensor(float("nan"), device=logits.device))
        return torch.stack(out)

    # ---- 去图 ----

    def _strip_vision(self, batch):
        """Delete visual tokens, using the same condition as LEGO scoring."""
        return strip_vision_tokens(batch, self._vision_ids)

    # ---- 工具 ----


    def _log_and_save(self, step, score, loss_full, loss_noimg):
        if not dist.is_initialized() or dist.get_rank() == 0:
            v = torch.isfinite(score)
            both = v & torch.isfinite(loss_full) & torch.isfinite(loss_noimg)
            logger.info(f"[MultimodalDeltaloss] step {step}: valid={int(v.sum())}/{len(score)}, "
                        f"score mean={score[v].mean():.4f}, weight [{self.W.min():.3f}, {self.W.max():.3f}], "
                        f"L(Q,I)={loss_full[both].mean():.4f}, L(Q)={loss_noimg[both].mean():.4f}, "
                        f"frac L(Q)>L(Q,I)={(loss_noimg[both] > loss_full[both]).float().mean():.3f}")
            if self.cache_dir:
                os.makedirs(self.cache_dir, exist_ok=True)
                torch.save({"step": step, "loss_full": loss_full, "loss_noimg": loss_noimg,
                             "score": score, "weight": self.W},
                            os.path.join(self.cache_dir, f"refresh_step{step}.pt"))
