"""One place where per-sample scores are computed and shared.

A loss-based selector, a dynamic reorder and a loss weighter all want the same
per-sample signal. Run independently each does its own full forward pass over the
pool, so composing three of them triples the scoring bill for no extra
information. The ScoreBoard computes a named metric at most once per planning
boundary and hands the same array to every stage that asks.

Scores are stored densely over dataset positions so a stage can index them
directly, with NaN for "not scored". Staleness is tracked per metric so a stage
can opt into reusing a slightly old signal (`max_age`) rather than forcing a
recompute.
"""

import json
import os
import random
import weakref
from contextlib import contextmanager, nullcontext
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataflex.train.data.loader import split_batch
from dataflex.utils.logging import logger
from dataflex.utils.loss_utils import per_sample_loss_from_outputs
from dataflex.utils.multimodal_scoring import move_to_device, scoring_state
from dataflex.utils.multimodal_scoring import strip_vision_tokens, vision_token_ids


@contextmanager
def _scoring_environment(model):
    """Extra scoring must not advance the training/collator random streams."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with scoring_state(model):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class _IndexedDataset(Dataset):
    """Attach the originating index to each item.

    Scoring is sharded across ranks and gathered back in an order nobody
    controls, so every sample has to carry its own identity.
    """

    def __init__(self, dataset, indices: Optional[Sequence[int]] = None):
        self.dataset = dataset
        self.indices = list(indices) if indices is not None else None

    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.dataset)

    def __getitem__(self, i):
        idx = self.indices[i] if self.indices is not None else i
        item = self.dataset[idx]
        if isinstance(item, dict):
            return {**item, "idx": idx}
        return {"idx": idx, "item": item}


class ScoreBoard:
    """Dense per-sample score cache with staleness tracking."""

    def __init__(
        self,
        dataset=None,
        accelerator=None,
        data_collator=None,
        dataset_size: int = 0,
        batch_size: int = 8,
        num_workers: int = 2,
        cache_dir: Optional[str] = None,
        seed: int = 42,
        score_sharing: str = "shared",
    ):
        self.dataset = dataset
        self.accelerator = accelerator
        self.data_collator = data_collator
        self.dataset_size = int(dataset_size or (len(dataset) if dataset is not None else 0))
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.cache_dir = cache_dir
        self.seed = int(seed)
        if score_sharing not in {"shared", "independent"}:
            raise ValueError("score_sharing must be 'shared' or 'independent'")
        if self.batch_size <= 0 or self.num_workers < 0:
            raise ValueError("Scoring batch size must be positive and worker count nonnegative")
        self.score_sharing = score_sharing
        self._model_ref = None

        self._values: Dict[str, np.ndarray] = {}
        #: When each position was last scored, -1 for never. Per position rather
        #: than per metric because a partial fill would otherwise re-date the
        #: whole array: one stage with a permissive `max_age` triggering a fill
        #: for a handful of positions would make arbitrarily old values look
        #: fresh to the next stage that asked for age 0.
        self._scored_at: Dict[str, np.ndarray] = {}
        self._step: Dict[str, int] = {}
        #: How many times a metric was actually computed, versus served from
        #: cache. Exposed so a run can show that sharing is working.
        self.compute_calls: Dict[str, int] = {}
        self.cache_hits: Dict[str, int] = {}
        # Selector outputs are algorithm state, not raw-loss compute caches.
        # Reorder consumes them in BOTH shared and independent modes.
        self.selection_scores = {}

    @property
    def scoring_contract(self):
        return {
            "version": 2,
            "loss": "answer-token-mean-ce-fp32-eval",
            "no_image": "delete-visual-tokens-v1",
            "sharing": self.score_sharing,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "seed": self.seed,
        }

    def _loss_key(self, metric, consumer):
        if metric not in {"loss_full", "loss_no_image"}:
            raise ValueError(f"Unknown LEGO loss condition: {metric}")
        if not consumer:
            raise ValueError("A stable component identity is required for scoring")
        return metric if self.score_sharing == "shared" else f"{consumer}::{metric}"

    def _bind_model(self, model):
        core = self.accelerator.unwrap_model(model) if self.accelerator is not None else model
        if self._model_ref is not None and self._model_ref() is not core:
            self._values.clear()
            self._scored_at.clear()
            self._step.clear()
            self.selection_scores.clear()
        self._model_ref = weakref.ref(core)
        return core

    def loss_scores(self, model, positions, step_id, *, consumer, metric="loss_full"):
        """Exact current-step scores; namespaces isolate independent components.

        Preprocessing must be deterministic. Refresh clocks and component-level
        baselines/weights remain owned by the components, not this cache.
        """
        positions = list(positions)
        if step_id < 0 or any(i < 0 or i >= self.dataset_size for i in positions):
            raise ValueError("Invalid scoring step or global sample IDs")
        self._bind_model(model)
        key = self._loss_key(metric, consumer)

        def compute(missing):
            values = self._compute_losses(model, missing, step_id, metric=metric)
            return [values[i] for i in missing]

        return self.get(key, positions, step_id, compute, max_age=0, fill="nan", cache_invalid=True)

    def put_selection_scores(self, consumer, payload):
        indices = np.asarray(payload["indices"], dtype=np.int64)
        scores = None if payload["scores"] is None else np.asarray(payload["scores"], dtype=np.float64)
        if (indices.ndim != 1 or len(np.unique(indices)) != len(indices)
                or (indices < 0).any() or (indices >= self.dataset_size).any()):
            raise ValueError("Selector scores have invalid or duplicate sample IDs")
        if scores is not None and (scores.shape != indices.shape or not np.isfinite(scores).all()):
            raise ValueError("Selector scores must be finite and aligned with sample IDs")
        if payload["step"] < 0 or payload["metric"] not in {None, "loss", "delta_loss"}:
            raise ValueError("Invalid selector score step or metric")
        if scores is not None and type(payload["ascending"]) is not bool:
            raise ValueError("Selector score direction must be explicit")
        if scores is None and (payload["metric"] is not None or payload.get("reason") != "random"):
            raise ValueError("Only Random selection can publish no scores")
        order = np.argsort(indices)
        self.selection_scores[consumer] = {
            **payload, "indices": indices[order].copy(),
            "scores": None if scores is None else scores[order].copy(),
        }

    def selection_scores_for(self, consumer, positions, selection_step):
        saved = self.selection_scores.get(consumer)
        if saved is None or saved["step"] != selection_step:
            raise ValueError("Missing selector scores for the active selection step")
        positions = np.asarray(list(positions), dtype=np.int64)
        ids = saved["indices"]
        offsets = np.searchsorted(ids, positions)
        if (offsets >= len(ids)).any() or not np.array_equal(ids[offsets], positions):
            raise ValueError("Reorder candidate IDs are not covered by the selector scores")
        return saved, None if saved["scores"] is None else saved["scores"][offsets].copy()

    def state_dict(self):
        return {
            "contract": self.scoring_contract,
            "values": self._values,
            "scored_at": self._scored_at,
            "step": self._step,
            "compute_calls": self.compute_calls,
            "cache_hits": self.cache_hits,
            "selection_scores": self.selection_scores,
        }

    def load_state_dict(self, state):
        if state.get("contract") != self.scoring_contract:
            raise ValueError("ScoreBoard checkpoint scoring contract differs or is missing")
        values = {k: np.asarray(v).copy() for k, v in state["values"].items()}
        stamps = {k: np.asarray(v).copy() for k, v in state["scored_at"].items()}
        if values.keys() != stamps.keys() or any(
            v.shape != (self.dataset_size,) for table in (values, stamps) for v in table.values()
        ):
            raise ValueError("ScoreBoard checkpoint does not match the dataset")
        self._values, self._scored_at = values, stamps
        self._step = dict(state["step"])
        self.compute_calls, self.cache_hits = dict(state["compute_calls"]), dict(state["cache_hits"])
        self.selection_scores = {}
        for consumer, payload in state.get("selection_scores", {}).items():
            self.put_selection_scores(consumer, payload)
        self._model_ref = None

    # ------------------------------------------------------------------
    # Cache protocol
    # ------------------------------------------------------------------

    def age(self, metric: str, step_id: int) -> Optional[int]:
        if metric not in self._step:
            return None
        return step_id - self._step[metric]

    def has_fresh(self, metric: str, step_id: int, max_age: Optional[int] = 0) -> bool:
        age = self.age(metric, step_id)
        if age is None:
            return False
        if max_age is None:
            return True
        return age <= max_age

    def put(self, metric: str, indices: Sequence[int], values: Sequence[float], step_id: int) -> None:
        arr = self._values.get(metric)
        if arr is None:
            arr = np.full(self.dataset_size, np.nan, dtype=np.float64)
        stamps = self._scored_at.get(metric)
        if stamps is None:
            stamps = np.full(self.dataset_size, -1, dtype=np.int64)
        idx = np.asarray(list(indices), dtype=np.int64)
        arr[idx] = np.asarray(list(values), dtype=np.float64)
        stamps[idx] = int(step_id)
        self._values[metric] = arr
        self._scored_at[metric] = stamps
        self._step[metric] = int(step_id)

    def get(
        self,
        metric: str,
        indices: Sequence[int],
        step_id: int = 0,
        compute_fn: Optional[Callable[[Sequence[int]], Sequence[float]]] = None,
        max_age: Optional[int] = 0,
        fill: str = "median",
        cache_invalid: bool = False,
    ) -> np.ndarray:
        """Return scores for `indices`, computing them only if necessary.

        Args:
            metric: cache key, e.g. "loss".
            indices: positions to score.
            step_id: current global step, for staleness bookkeeping.
            compute_fn: called with the positions that still need a value. When
                omitted and nothing is cached, raises.
            max_age: reuse a cached value if it is at most this many steps old.
                `0` means only this boundary's value counts; `None` means any
                cached value is acceptable.
            fill: what to do about positions that remain unscored —
                "median" (neutral: keeps them mid-curriculum) or "nan".
        """
        indices = list(indices)
        if not indices:
            return np.zeros(0, dtype=np.float64)

        idx = np.asarray(indices, dtype=np.int64)
        cached = self._values.get(metric)
        stamps = self._scored_at.get(metric)

        if cached is None or stamps is None:
            to_compute = indices
        else:
            # A position is usable when it holds a real number that is young
            # enough. Judging each position on its own stamp is what keeps a
            # partial fill from passing stale values off as fresh.
            usable = (stamps[idx] >= 0) & (stamps[idx] <= step_id)
            if not cache_invalid:
                usable &= ~np.isnan(cached[idx])
            if max_age is not None:
                usable &= (step_id - stamps[idx]) <= max_age
            stale = idx[~usable]
            if stale.size == 0:
                self.cache_hits[metric] = self.cache_hits.get(metric, 0) + 1
                logger.info(
                    f"[Dataflex][Lego] '{metric}' served from cache "
                    f"(oldest={int((step_id - stamps[idx]).max())} steps, {len(indices)} positions)"
                )
                return cached[idx].copy()
            to_compute = stale.tolist()

        if compute_fn is None:
            if cached is None:
                raise KeyError(f"no scores for metric '{metric}' and no compute_fn given")
            to_compute = []

        if to_compute:
            self.compute_calls[metric] = self.compute_calls.get(metric, 0) + 1
            logger.info(
                f"[Dataflex][Lego] computing '{metric}' for {len(to_compute)} positions "
                f"(compute #{self.compute_calls[metric]} this run)"
            )
            to_compute = sorted(set(to_compute))
            values = compute_fn(to_compute)
            self.put(metric, to_compute, values, step_id)

        arr = self._values[metric]
        out = arr[idx].copy()
        if np.isnan(out).any():
            if fill == "median":
                finite = out[np.isfinite(out)]
                out[np.isnan(out)] = float(np.median(finite)) if finite.size else 0.0
            # "nan" leaves them for the caller to deal with
        return out

    # ------------------------------------------------------------------
    # The built-in metric: per-sample loss under the current model
    # ------------------------------------------------------------------

    def per_sample_loss_fn(self, model, step_id: int, max_samples: Optional[int] = None):
        """Build a `compute_fn` for `get("loss", ...)`."""

        def compute(positions: Sequence[int]) -> List[float]:
            target = list(positions)
            if max_samples is not None and len(target) > max_samples:
                rng = np.random.default_rng(self.seed + int(step_id))
                picked = sorted(rng.choice(len(target), size=max_samples, replace=False).tolist())
                subset = [target[i] for i in picked]
                logger.info(
                    f"[Dataflex][Lego] scoring a {len(subset)}/{len(target)} subsample "
                    f"(max_samples={max_samples}); the rest stay unscored and take the median"
                )
                scored = self._compute_losses(model, subset, step_id)
                return [scored.get(p, np.nan) for p in target]

            scored = self._compute_losses(model, target, step_id)
            return [scored.get(p, np.nan) for p in target]

        return compute

    def _compute_losses(self, model, positions: Sequence[int], step_id: int, metric="loss_full") -> Dict[int, float]:
        if model is None:
            raise ValueError("per-sample loss needs a model")
        if not positions:
            return {}
        acc = self.accelerator
        core = acc.unwrap_model(model) if acc is not None else model
        device = acc.device if acc is not None else next(model.parameters()).device
        # Local DataLoader RNG avoids advancing the Trainer's RNG even at
        # iterator construction. scoring_state also restores model/rope state.
        with _scoring_environment(model):
            loader = DataLoader(
                _IndexedDataset(self.dataset, positions),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.data_collator,
                generator=torch.Generator().manual_seed(self.seed + int(step_id)),
            )
            if acc is not None:
                loader = acc.prepare(loader)
            local_losses, local_indices = [], []
            for batch in tqdm(
                loader, desc=f"[Lego {metric} @ step {step_id}]",
                disable=acc is not None and not acc.is_main_process, dynamic_ncols=True,
            ):
                batch = move_to_device(batch, device)
                idx = torch.as_tensor(batch["idx"], dtype=torch.long, device=device).view(-1)
                inputs, _ = split_batch(batch)
                if metric == "loss_no_image":
                    inputs = strip_vision_tokens(inputs, vision_token_ids(core.config))
                labels = inputs.pop("labels")
                with torch.no_grad(), acc.autocast() if acc is not None else nullcontext():
                    outputs = model(**{**inputs, "use_cache": False})
                    if getattr(outputs, "logits", None) is None:
                        raise ValueError("LEGO scoring requires logits; a scalar batch loss is not a per-sample score")
                    if outputs.logits.shape[:2] != labels.shape:
                        raise ValueError("LEGO scoring requires logits aligned with response labels")
                    per_sample = per_sample_loss_from_outputs(outputs, {"labels": labels})
                local_losses.append(per_sample)
                local_indices.append(idx)

            losses = torch.cat(local_losses).float() if local_losses else torch.zeros(0, device=device)
            indices = torch.cat(local_indices) if local_indices else torch.zeros(0, dtype=torch.long, device=device)
            if acc is not None:
                losses, indices = acc.gather(losses), acc.gather(indices)
            out = {}
            for i, value in zip(indices.cpu().tolist(), losses.cpu().tolist()):
                # Accelerate pads the tail; preserve the first occurrence,
                # including NaN for samples with no supervised answer tokens.
                out.setdefault(int(i), float(value))
            if any(i not in out for i in positions):
                raise RuntimeError("Distributed scoring did not return every requested sample ID")
            return out

    # ------------------------------------------------------------------
    # Persistence, reusing the selector cache layout
    # ------------------------------------------------------------------

    def save(self, step_id: int, indices: Optional[Sequence[int]] = None) -> Optional[str]:
        """Dump the board in the `{"indices", "metric": {...}}` selector format.

        Same shape `selector_io.save_selection` writes, so a selector run and a
        lego run produce interchangeable artifacts and either can seed the
        other.
        """
        if not self.cache_dir:
            return None
        if self.accelerator is not None and not self.accelerator.is_main_process:
            return None
        if not self._values:
            return None

        idx = list(indices) if indices is not None else list(range(self.dataset_size))
        payload = {
            "indices": [int(i) for i in idx],
            "metric": {
                name: [None if np.isnan(v) else float(v) for v in arr[np.asarray(idx, dtype=np.int64)]]
                for name, arr in self._values.items()
            },
            "computed_at_step": {k: int(v) for k, v in self._step.items()},
        }
        # Preserve the selector-cache convention for the one shared full-loss
        # signal. Independent namespaces must not masquerade as a single cache.
        if "loss_full" in payload["metric"]:
            payload["metric"].setdefault("loss", payload["metric"]["loss_full"])
        os.makedirs(self.cache_dir, exist_ok=True)
        path = os.path.join(self.cache_dir, f"scoreboard_step_{step_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        logger.info(f"[Dataflex][Lego] scoreboard saved to {path}")
        return path

    def summary(self) -> str:
        parts = []
        for metric in sorted(set(self.compute_calls) | set(self.cache_hits)):
            parts.append(
                f"{metric}: computed {self.compute_calls.get(metric, 0)}x, "
                f"reused {self.cache_hits.get(metric, 0)}x"
            )
        return "; ".join(parts) if parts else "no metrics requested"
