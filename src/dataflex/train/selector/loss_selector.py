from dataflex.core.registry import register_selector
from dataflex.utils.selector_io import load_cached_selection, save_selection
from dataflex.utils.logging import logger
from dataflex.utils.loss_utils import UNSCORED, per_sample_loss_from_outputs
from .base_selector import Selector

import math
import torch
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import json
import os

class IndexedDataset(Dataset):
    def __init__(self, original_dataset, indices=None):
        self.dataset = original_dataset
        self.indices = list(indices) if indices is not None else None

    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.dataset)

    def __getitem__(self, i):
        index = self.indices[i] if self.indices is not None else i
        data = self.dataset[index]
        return {"idx": index, **data}

@register_selector('loss')
class LossSelector(Selector):
    def __init__(
        self,
        dataset,
        accelerator,
        data_collator,
        cache_dir,
        focus: str = "high",              # "high" | "medium" | "low"
        focus_weight: float = 5.0,        # Weight multiplier
        quantiles: tuple = (0.33, 0.66),  # Low/medium/high quantiles
        replacement: bool = False,        # Whether to use replacement sampling
        temperature: float = 1.0,         # Temperature control
        score_batch_size: int = 1,        # Samples per scoring forward pass
        score_num_workers: int = 2,       # DataLoader workers for scoring
        score_ascending: bool = None,     # Required for focus=medium when exporting order scores.
    ):
        super().__init__(dataset, accelerator, data_collator, cache_dir)

        # New sampling control parameters
        self.focus = str(focus).lower()
        if self.focus not in {"low", "medium", "high"}:
            raise ValueError("focus must be 'low', 'medium' or 'high'")
        self.focus_weight = focus_weight
        self.quantiles = quantiles
        self.replacement = replacement
        self.temperature = temperature
        self.score_batch_size = int(score_batch_size)
        self.score_num_workers = int(score_num_workers)
        if score_ascending is not None and not isinstance(score_ascending, bool):
            raise ValueError("score_ascending must be a boolean or None")
        if self.focus != "medium" and score_ascending is not None and score_ascending != (self.focus == "low"):
            raise ValueError("score_ascending must agree with the loss selector's low/high focus")
        self.score_ascending = score_ascending if self.focus == "medium" else self.focus == "low"

        logger.info(f"LossSelector initialized.")

    def _compute_losses(self, positions, model, step_id: int):
        """Per-sample loss for `positions`, one value per position.

        The old implementation was pinned to `batch_size=1` because it read
        `model(**inputs).loss`, which is a batch *mean* and only coincides with
        the per-sample loss when the batch holds one sample. Going through
        logits+labels makes the result independent of how the pool is batched,
        so `score_batch_size` above 1 is a pure speedup.
        """
        positions = list(positions)
        if not positions:
            return []

        dataloader = DataLoader(
            IndexedDataset(self.dataset, positions),
            batch_size=self.score_batch_size,
            shuffle=False,
            num_workers=self.score_num_workers,
            collate_fn=self.data_collator,
        )
        dataloader = self.accelerator.prepare(dataloader)

        if self.accelerator.is_main_process:
            logger.info(
                f"[Dataflex] Calculating loss for {len(positions)} samples using "
                f"{self.accelerator.num_processes} GPUs (batch_size={self.score_batch_size})"
            )
        local_losses, local_indices = [], []
        for batch in tqdm(
            dataloader,
            desc=f"[Selector step {step_id}]",
            disable=not self.accelerator.is_main_process,
            dynamic_ncols=True,
        ):
            idx = batch["idx"]
            if not torch.is_tensor(idx):
                idx = torch.tensor(idx, dtype=torch.long, device=self.accelerator.device)
            idx = idx.view(-1).to(dtype=torch.long)

            with torch.no_grad():
                model_inputs = {k: v for k, v in batch.items() if k not in ("idx", "domain_id")}
                outputs = model(**model_inputs)
                loss = per_sample_loss_from_outputs(outputs, model_inputs).view(-1)

            local_losses.append(loss.float())
            local_indices.append(idx)

        if local_losses:
            local_losses = torch.cat(local_losses, dim=0)
            local_indices = torch.cat(local_indices, dim=0)
        else:
            local_losses = torch.zeros(0, device=self.accelerator.device)
            local_indices = torch.zeros(0, dtype=torch.long, device=self.accelerator.device)

        all_losses = self.accelerator.gather(local_losses).detach().cpu().tolist()
        all_indices = self.accelerator.gather(local_indices).detach().cpu().tolist()

        # gather pads the last batch, so the same idx can come back more than
        # once; keeping the first occurrence makes the result independent of the
        # world size. Non-finite values are left out so that a duplicate
        # carrying a real number can still fill the index in, and anything still
        # missing afterwards reads as UNSCORED.
        by_index = {}
        for value, i in zip(all_losses, all_indices):
            i = int(i)
            if i not in by_index and math.isfinite(value):
                by_index[i] = float(value)

        if self.accelerator.is_main_process:
            logger.info(f"[Dataflex] Loss calculation finished")
        return [by_index.get(p, UNSCORED) for p in positions]

    def select(self, model, step_id: int, num_samples: int, **kwargs):
        model.eval()
        os.makedirs(self.cache_dir, exist_ok=True)
        save_path = os.path.join(self.cache_dir, f"step_{step_id}.json")
        if self._score_sink is None and getattr(self, "reuse_selection_cache", True) and os.path.exists(save_path):
            if self.accelerator.is_main_process:
                cached_indices, _ = load_cached_selection(save_path)
            else:
                cached_indices = None
            cached_indices_list = [cached_indices]
            if dist.is_available() and dist.is_initialized():
                dist.broadcast_object_list(cached_indices_list, src=0)
                cached_indices = cached_indices_list[0]
            else:
                cached_indices = cached_indices or []
            return cached_indices

        # Only score the candidate pool. When used independently, the candidate pool is the entire dataset; when combined with other stages, the upstream may have already filtered out most data, so it's not necessary to calculate the loss for them again.
        positions = self.candidate_positions()
        scores = self._scores_for(positions, model, step_id)

        # ========= Main process: sampling based on distribution =========
        if self.accelerator.is_main_process:
            logger.info(f"[Dataflex] focus={self.focus}, focus_weight={self.focus_weight}")
            losses = torch.tensor(scores, dtype=torch.float32)
            valid_mask = torch.isfinite(losses)

            if valid_mask.sum().item() == 0:
                probs = torch.full((len(losses),), 1.0 / max(1, len(losses)))
            else:
                valid_losses = losses[valid_mask]
                q1 = torch.quantile(valid_losses, self.quantiles[0])
                q2 = torch.quantile(valid_losses, self.quantiles[1])

                low_mask    = (losses <= q1) & valid_mask
                medium_mask = (losses > q1) & (losses <= q2) & valid_mask
                high_mask   = (losses > q2) & valid_mask

                weights = torch.zeros_like(losses).float()
                weights[low_mask]    = 1.0
                weights[medium_mask] = 1.0
                weights[high_mask]   = 1.0

                if self.focus == "low":
                    weights[low_mask] *= self.focus_weight
                elif self.focus == "medium":
                    weights[medium_mask] *= self.focus_weight
                else:
                    weights[high_mask] *= self.focus_weight

                weights[~valid_mask] = 0.0
                eps = 1e-12
                probs = (weights + eps) ** (1.0 / self.temperature)
                if probs.sum() == 0.0:
                    probs = valid_mask.float()
                probs = probs / probs.sum()

            available = int((probs > 0).sum().item())
            effective_replacement = self.replacement
            if not effective_replacement and num_samples > available:
                effective_replacement = True
                logger.info(
                    f"[Dataflex] Effective sample size {available} is less than the requested number {num_samples},"
                    f"automatically changed to replacement sampling."
                )

            gen = torch.Generator()
            gen.manual_seed(self.seed + int(step_id))
            picked = torch.multinomial(
                probs.cpu(), num_samples=num_samples,
                replacement=effective_replacement, generator=gen
            ).tolist()
            # multinomial gives the indices in the candidate pool, map back to the dataset indices
            sel = [positions[i] for i in picked]

            # ========= Save (only save "selected indices + corresponding metric") =========
            metric_payload = {"loss": [float(scores[i]) for i in picked]}
            self.publish_scores(sel, metric_payload["loss"], step_id,
                                metric="loss", ascending=self.score_ascending)
            save_selection(save_path, sel, metric_payload, self.accelerator)
        else:
            sel = None

        # Broadcast selected samples
        sel_list = [sel]
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(sel_list, src=0)
            sel = sel_list[0]
        else:
            sel = sel or []

        return sel
