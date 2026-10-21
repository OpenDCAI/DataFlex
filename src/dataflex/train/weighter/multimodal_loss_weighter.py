"""多模态 Loss 重加权: s = L(R|Q,I)，一次有图前向。

和 DeltaLoss 结构相同但只做一次有图前向，不含 _strip_vision。
高 loss 的样本权重更大（linupper 上权）。
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

from dataflex.core.registry import register_weighter
from dataflex.utils.logging import logger
from .base_weighter import Weighter

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


class _IndexedDataset(Dataset):
    """按 order 依次返回 (原始 idx, 主样本)。"""
    def __init__(self, dataset, order):
        self.dataset = dataset
        self.order = order
    def __len__(self):
        return len(self.order)
    def __getitem__(self, i):
        idx = self.order[i]
        return idx, self.dataset[idx]


@register_weighter("multimodal_loss")
class MultimodalLossWeighter(Weighter):
    """多模态 Loss 重加权：s = L(R|Q,I)，一次有图前向。"""

    def __init__(
        self,
        dataset=None,
        data_collator=None,
        accelerator=None,
        refresh_interval: int = 244,
        delta: float = 1.0,
        r: float = 1.0,
        clip_quantile: float = 0.01,
        score_batch_size: int = 8,
        score_num_workers: int = 4,
        loss_reduction: str = "sample",
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if dataset is None or data_collator is None:
            raise ValueError("MultimodalLossWeighter requires dataset and data_collator.")
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

        self.W = None
        self._weight_map = {}
        self._last_refresh_step = -1
        self._order = None
        logger.info(f"[MultimodalLoss] refresh_interval={refresh_interval}, delta={delta}")

    def training_step(self, ctx, model, inputs, num_items_in_batch=None, use_weighter=False):
        step = ctx.state.global_step
        if use_weighter and step != self._last_refresh_step and \
                (self.W is None or step % self.refresh_interval == 0):
            self._refresh(ctx, model, step)
            self._last_refresh_step = step
        return super().training_step(ctx, model, inputs, num_items_in_batch, use_weighter)

    def get_weighted_loss(self, losses, *, ctx=None, model=None, inputs=None):
        if (not torch.is_tensor(losses)) or losses.dim() == 0:
            return losses
        if not self._weight_map:
            return losses.mean()
        losses = losses.view(-1)
        w = torch.tensor([self._weight_map.get(k, 1.0) for k in _sample_keys(inputs)],
                         dtype=losses.dtype, device=losses.device)
        if self.loss_reduction == "token":
            n = (inputs["labels"][..., 1:] != IGNORE_INDEX).sum(dim=1).to(losses.dtype)
            return torch.sum(w * losses * n) / n.sum().clamp_min(1)
        return torch.mean(w * losses)

    @torch.no_grad()
    def _refresh(self, ctx, model, step):
        acc = ctx.accelerator
        n = len(self.dataset)
        order = self._get_length_order()
        loader = DataLoader(
            _IndexedDataset(self.dataset, order),
            batch_size=self.score_batch_size, shuffle=False,
            num_workers=self.score_num_workers,
            collate_fn=lambda items: {
                "idx": torch.tensor([it[0] for it in items], dtype=torch.long),
                "main": self.data_collator([it[1] for it in items]),
            },
        )
        loader = acc.prepare(loader)

        logger.info(f"[MultimodalLoss] step {step}: scoring {n} samples")
        was_training = model.training
        model.eval()
        idx_l, key_l, loss_l = [], [], []
        for batch in loader:
            main = batch["main"]
            inputs = {k: v for k, v in main.items() if k != "labels"}
            logits = model(**inputs, use_cache=False).logits
            # per-sample answer-token-mean CE
            shift_logits, shift_labels = logits[:, :-1], main["labels"][:, 1:]
            per_loss = []
            for i in range(logits.size(0)):
                m = shift_labels[i] != IGNORE_INDEX
                if m.any():
                    per_loss.append(F.cross_entropy(shift_logits[i][m].float(), shift_labels[i][m]))
                else:
                    per_loss.append(torch.tensor(float("nan"), device=logits.device))
            loss_l.append(torch.stack(per_loss))
            idx_l.append(batch["idx"])
            key_l.append(torch.frombuffer(bytearray(b"".join(_sample_keys(main))), dtype=torch.uint8)
                         .view(-1, 16).to(acc.device))
        if was_training:
            model.train()
        torch.cuda.empty_cache()

        all_idx = acc.gather(torch.cat(idx_l)).cpu()
        all_keys = acc.gather(torch.cat(key_l)).cpu().numpy()
        vals = acc.gather(torch.cat(loss_l)).float().cpu()
        score = torch.full((n,), float("nan"))
        score[all_idx] = vals

        self.W = self._score_to_weight(score)
        self._weight_map = {k.tobytes(): float(self.W[i]) for k, i in zip(all_keys, all_idx.tolist())}

        if not dist.is_initialized() or dist.get_rank() == 0:
            v = torch.isfinite(score)
            logger.info(f"[MultimodalLoss] step {step}: valid={int(v.sum())}/{n}, "
                        f"L(Q,I) mean={score[v].mean():.4f}, weight [{self.W.min():.3f}, {self.W.max():.3f}]")
            if self.cache_dir:
                os.makedirs(self.cache_dir, exist_ok=True)
                torch.save({"step": step, "score": score, "weight": self.W},
                            os.path.join(self.cache_dir, f"refresh_step{step}.pt"))

    def _score_to_weight(self, score):
        valid = torch.isfinite(score)
        s = score.clone()
        lo = torch.quantile(s[valid], self.clip_quantile).item()
        hi = torch.quantile(s[valid], 1 - self.clip_quantile).item()
        s = s.clamp(lo, hi)
        denom = max(hi - lo, 1e-6)
        x = 2 * self.delta * s / denom - self.delta * (hi + lo) / denom
        x = torch.minimum(x + self.delta, self.delta * torch.ones_like(x))
        w = torch.exp((x - x[valid].max()) / self.r)
        w = w / w[valid].mean()
        w[~valid] = 1.0
        return w

    def _get_length_order(self):
        if self._order is None:
            lens = self.dataset.map(
                lambda b: {"_n": [len(x) for x in b["input_ids"]]},
                batched=True, remove_columns=self.dataset.column_names, keep_in_memory=True,
            )["_n"]
            self._order = torch.tensor(lens).argsort(descending=True).tolist()
        return self._order
