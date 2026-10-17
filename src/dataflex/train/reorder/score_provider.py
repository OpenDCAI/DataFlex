"""Sources of the per-sample score vector that drives ordering.

The ordering patterns are pure functions of a score vector, so where that vector
comes from is a separate, swappable concern. Four sources are provided:

    precomputed  - a quality score that already exists (FineWeb-Edu edu score,
                   QuRating, a KenLM perplexity, an LQS output). Free, and what
                   the paper actually uses.
    model_loss   - per-sample loss under the current model. Costs a forward pass
                   over the pool but needs no data plumbing and reflects the
                   model's current state, which is what makes dynamic reordering
                   possible.
    model_perplexity - exp(token-mean causal NLL) per sample under the current
                   model, with the same scoring schedule as model_loss.
    cached       - reuse scores a selector already wrote to its cache, i.e. the
                   paper's "reuse pre-computed scores at zero extra cost"
                   argument expressed with DataFlex's own artifacts.

Note on polarity: these return raw scores. Interpreting whether high means good
is `Reorder.polarity`, not the provider's job.
"""

import json
import math
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, default_collate
from tqdm import tqdm

from dataflex.utils.logging import logger


class _IndexedDataset(Dataset):
    """Wrap a dataset so each item carries the index it came from.

    Scoring runs sharded across ranks and the results come back gathered in an
    order nobody controls, so every sample has to say who it is. Same trick the
    loss/delta-loss selectors use.
    """

    def __init__(self, dataset, indices: Optional[Sequence[int]] = None):
        self.dataset = dataset
        self.indices = list(indices) if indices is not None else None

    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.dataset)

    def __getitem__(self, i):
        idx = self.indices[i] if self.indices is not None else i
        return {"idx": idx, **self.dataset[idx]}


class _IndexedCollator:
    """Keep scoring indices outside the training collator's column filtering."""

    def __init__(self, data_collator):
        self.data_collator = data_collator or default_collate

    def __call__(self, features):
        indices = torch.tensor([feature["idx"] for feature in features], dtype=torch.long)
        rows = [{k: v for k, v in feature.items() if k != "idx"} for feature in features]
        return {**self.data_collator(rows), "idx": indices}


class ScoreProvider(ABC):
    """Produces a score for each position it is asked about."""

    #: True when scoring depends on the model, i.e. scores must be recomputed as
    #: training progresses rather than fetched once.
    is_dynamic: bool = False

    @abstractmethod
    def scores_for(self, positions: Sequence[int], model=None, step_id: int = 0) -> List[float]:
        """Return one score per entry of `positions`, in the same order."""
        raise NotImplementedError

    def describe(self) -> str:
        return self.__class__.__name__


class PrecomputedScoreProvider(ScoreProvider):
    """Scores that already exist, keyed by position.

    Accepts either a dense vector covering positions `0..N-1` (a `.npy`, a
    `.json` list, or a jsonl field), which is how the static path reads a score
    column straight off the raw dataset.
    """

    is_dynamic = False

    def __init__(
        self,
        scores: Optional[Sequence[float]] = None,
        score_path: Optional[str] = None,
        score_field: str = "score",
        expected_size: Optional[int] = None,
    ):
        if scores is None and score_path is None:
            raise ValueError("PrecomputedScoreProvider needs either `scores` or `score_path`")

        if scores is None:
            scores = self._load(score_path, score_field)

        self.scores = np.asarray(scores, dtype=np.float64)
        if not np.all(np.isfinite(self.scores)):
            n_bad = int((~np.isfinite(self.scores)).sum())
            raise ValueError(f"score vector contains {n_bad} non-finite values")

        if expected_size is not None and len(self.scores) != expected_size:
            raise ValueError(
                f"score vector has {len(self.scores)} entries but the dataset has {expected_size}. "
                f"Scores must line up one-to-one with dataset rows."
            )
        logger.info(
            f"[Dataflex][Reorder] loaded {len(self.scores)} precomputed scores "
            f"(min={self.scores.min():.4f}, max={self.scores.max():.4f}, mean={self.scores.mean():.4f})"
        )

    @staticmethod
    def _load(path: str, score_field: str) -> Sequence[float]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"score file not found: {path}")

        if path.endswith(".npy"):
            return np.load(path).reshape(-1).tolist()

        if path.endswith(".jsonl"):
            out = []
            with open(path, "r", encoding="utf-8") as f:
                for lineno, line in enumerate(f):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if score_field not in row:
                        raise KeyError(f"{path}:{lineno + 1} has no field '{score_field}'")
                    out.append(float(row[score_field]))
            return out

        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                # Also accept the selector cache layout {"indices": [...], "metric": {...}}.
                if score_field in payload:
                    return payload[score_field]
                metric = payload.get("metric") or {}
                if score_field in metric:
                    return metric[score_field]
                raise KeyError(f"{path} has no '{score_field}' at top level or under 'metric'")
            return payload

        raise ValueError(f"unsupported score file type: {path} (expected .npy/.json/.jsonl)")

    def scores_for(self, positions, model=None, step_id: int = 0) -> List[float]:
        positions = np.asarray(list(positions), dtype=np.int64)
        if positions.size and (positions.max() >= len(self.scores) or positions.min() < 0):
            raise IndexError(
                f"position out of range for score vector of length {len(self.scores)} "
                f"(min={positions.min()}, max={positions.max()})"
            )
        return self.scores[positions].tolist()

    def describe(self) -> str:
        return f"precomputed(n={len(self.scores)})"


class CachedSelectionScoreProvider(PrecomputedScoreProvider):
    """Reuse a score a selector already computed and cached.

    `save_selection` writes `{"indices": [...], "metric": {"loss": [...]}}`, so a
    reorder can consume a selector's work instead of paying for a second
    scoring pass. Positions absent from the cache get `fill_value`.
    """

    is_dynamic = False

    def __init__(self, cache_path: str, metric_key: str = "loss", expected_size: Optional[int] = None,
                 fill_value: float = 0.0):
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"selection cache not found: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        indices = payload.get("indices")
        metric = (payload.get("metric") or {}).get(metric_key)
        if indices is None or metric is None:
            available = ", ".join((payload.get("metric") or {}).keys()) or "<none>"
            raise KeyError(f"{cache_path} lacks 'indices' or metric '{metric_key}'. Available metrics: {available}")
        if len(indices) != len(metric):
            raise ValueError(f"{cache_path}: {len(indices)} indices but {len(metric)} '{metric_key}' values")

        size = expected_size if expected_size is not None else (max(indices) + 1 if indices else 0)
        dense = np.full(size, float(fill_value), dtype=np.float64)
        for idx, value in zip(indices, metric):
            if 0 <= idx < size:
                dense[idx] = float(value)

        covered = len(set(i for i in indices if 0 <= i < size))
        logger.info(
            f"[Dataflex][Reorder] reusing '{metric_key}' from {cache_path}: "
            f"{covered}/{size} positions covered, rest filled with {fill_value}"
        )
        super().__init__(scores=dense, expected_size=expected_size)
        self.metric_key = metric_key

    def describe(self) -> str:
        return f"cached_selection(metric={self.metric_key}, n={len(self.scores)})"


class ModelLossScoreProvider(ScoreProvider):
    """Per-sample loss under the current model.

    This is what makes ordering interactive: the score reflects the model as it
    is now, not a fixed judgement made before training. Higher loss means the
    sample is currently harder, so pair it with `polarity: lower_is_better` if
    you want "easy first".

    Scoring the whole pool every interval is the dominant cost, so `max_samples`
    subsamples the pool and unscored positions inherit the pool median (which
    leaves them mid-curriculum rather than biasing them to either end).
    """

    is_dynamic = True

    def __init__(
        self,
        dataset,
        accelerator,
        data_collator=None,
        batch_size: int = 8,
        num_workers: int = 2,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.accelerator = accelerator
        self.data_collator = data_collator
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.max_samples = max_samples
        self.seed = int(seed)

    def scores_for(self, positions, model=None, step_id: int = 0) -> List[float]:
        positions = list(positions)
        if not positions:
            return []
        if model is None:
            raise ValueError("ModelLossScoreProvider requires a model")

        target = positions
        if self.max_samples is not None and len(positions) > self.max_samples:
            rng = np.random.default_rng(self.seed + int(step_id))
            target = sorted(rng.choice(len(positions), size=self.max_samples, replace=False).tolist())
            target = [positions[i] for i in target]
            logger.info(
                f"[Dataflex][Reorder] scoring a {len(target)}/{len(positions)} subsample of the pool "
                f"(max_samples={self.max_samples})"
            )

        losses = self._compute_losses(model, target, step_id)

        if len(target) == len(positions):
            return [losses[p] for p in positions]

        # Unscored positions sit at the median so subsampling does not push them
        # to the front or the back of the curriculum.
        finite = [v for v in losses.values() if np.isfinite(v)]
        fill = float(np.median(finite)) if finite else 0.0
        return [losses.get(p, fill) for p in positions]

    def _compute_losses(self, model, positions: Sequence[int], step_id: int) -> Dict[int, float]:
        was_training = model.training
        model.eval()

        loader = DataLoader(
            _IndexedDataset(self.dataset, positions),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
        )
        loader = self.accelerator.prepare(loader)

        local_losses, local_indices = [], []
        for batch in tqdm(
            loader,
            desc=f"[Reorder scoring @ step {step_id}]",
            disable=not self.accelerator.is_main_process,
            dynamic_ncols=True,
        ):
            idx = batch["idx"]
            if not torch.is_tensor(idx):
                idx = torch.tensor(idx, dtype=torch.long, device=self.accelerator.device)
            idx = idx.view(-1).to(dtype=torch.long)

            inputs = {k: v for k, v in batch.items() if k != "idx"}
            with torch.no_grad():
                outputs = model(**inputs)
                per_sample = self._per_sample_loss(outputs, inputs)

            local_losses.append(per_sample)
            local_indices.append(idx)

        if local_losses:
            local_losses = torch.cat(local_losses, dim=0).float()
            local_indices = torch.cat(local_indices, dim=0)
        else:
            local_losses = torch.zeros(0, device=self.accelerator.device)
            local_indices = torch.zeros(0, dtype=torch.long, device=self.accelerator.device)

        all_losses = self.accelerator.gather(local_losses).detach().cpu().tolist()
        all_indices = self.accelerator.gather(local_indices).detach().cpu().tolist()

        # Distributed gather pads the last batch, so the same index can come back
        # more than once. First occurrence wins, which keeps this deterministic.
        out: Dict[int, float] = {}
        for value, idx in zip(all_losses, all_indices):
            if idx not in out and np.isfinite(value):
                out[int(idx)] = float(value)

        if was_training:
            model.train()

        missing = len(positions) - len(out)
        if missing > 0:
            logger.warning(f"[Dataflex][Reorder] {missing} positions produced no score; they will use the median")
        return out

    @staticmethod
    def _per_sample_loss(outputs, inputs) -> torch.Tensor:
        """Token-mean cross-entropy per sample, so long samples are not penalised.

        Falls back to the model's own scalar loss broadcast across the batch when
        logits/labels are unavailable.
        """
        logits = getattr(outputs, "logits", None)
        labels = inputs.get("labels", None)
        if logits is None or labels is None:
            loss = getattr(outputs, "loss", None)
            if loss is None:
                raise ValueError("model returned neither logits+labels nor a loss")
            batch = next(iter(inputs.values())).size(0)
            return loss.detach().view(1).expand(batch).clone()

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        tok_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1).long(),
        ).view(shift_labels.size(0), -1)
        active = (shift_labels != -100).sum(dim=1)
        return (tok_loss.sum(dim=1) / torch.clamp(active, min=1)).detach()

    def describe(self) -> str:
        return f"model_loss(batch_size={self.batch_size}, max_samples={self.max_samples})"


class ModelPerplexityScoreProvider(ModelLossScoreProvider):
    """Per-sample perplexity: exp(mean causal negative log-likelihood).

    Only supervised, non-padding targets count. For SFT this is response PPL
    conditioned on the prompt when prompt labels are -100. For packed data the
    score describes the whole preprocessed row, not each original document.
    Logits and labels are required: a batch-level scalar loss cannot recover
    individual sample perplexities. Invalid rows inherit the scored pool median;
    a pool with no valid scores raises instead of assigning a misleading PPL=1.

    Use higher_later for an easy-to-hard difficulty curriculum. Exponentiation
    preserves NLL ranks; it does not itself improve the curriculum.
    """

    @staticmethod
    def _per_sample_loss(outputs, inputs) -> torch.Tensor:
        logits = getattr(outputs, "logits", None)
        labels = inputs.get("labels")
        if logits is None or labels is None:
            raise ValueError("model_perplexity requires per-token logits and labels, not a batch scalar loss")
        if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
            raise ValueError("model_perplexity expects logits [batch, tokens, vocab] and labels [batch, tokens]")

        # FP32 CE avoids half-precision overflow before exponentiating in FP64.
        shift_logits = logits[:, :-1, :].float().contiguous()
        shift_labels = labels[:, 1:].to(device=logits.device, dtype=torch.long).clone()
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            if attention_mask.shape != labels.shape:
                raise ValueError("model_perplexity expects attention_mask [batch, tokens] matching labels")
            mask = attention_mask.to(device=logits.device)
            # Exclude padding targets AND a first token predicted from left pad.
            valid = (mask[:, 1:] != 0) & (mask[:, :-1] != 0)
            shift_labels.masked_fill_(~valid, -100)

        token_losses = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).reshape_as(shift_labels)
        active = (shift_labels != -100).sum(dim=1)
        mean_nll = token_losses.sum(dim=1) / active.clamp_min(1)
        return mean_nll.masked_fill(active == 0, float("nan")).detach()

    def _compute_losses(self, model, positions: Sequence[int], step_id: int) -> Dict[int, float]:
        # Keep the existing ModelLossScoreProvider scoring path unchanged. PPL
        # needs its own collator so the training collator cannot drop row IDs,
        # and its invalid rows must be filled in PPL space.
        was_training = model.training
        model.eval()
        try:
            loader = DataLoader(
                _IndexedDataset(self.dataset, positions),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=_IndexedCollator(self.data_collator),
            )
            loader = self.accelerator.prepare(loader)

            local_nll, local_indices = [], []
            for batch in tqdm(
                loader,
                desc=f"[Reorder PPL scoring @ step {step_id}]",
                disable=not self.accelerator.is_main_process,
                dynamic_ncols=True,
            ):
                index = batch["idx"]
                if not torch.is_tensor(index):
                    index = torch.tensor(index, dtype=torch.long, device=self.accelerator.device)
                inputs = {key: value for key, value in batch.items() if key != "idx"}
                with torch.no_grad():
                    nll = self._per_sample_loss(model(**inputs), inputs)
                local_nll.append(nll)
                local_indices.append(index.view(-1).to(device=self.accelerator.device, dtype=torch.long))

            if local_nll:
                local_nll = torch.cat(local_nll).float()
                local_indices = torch.cat(local_indices)
            else:
                local_nll = torch.zeros(0, device=self.accelerator.device)
                local_indices = torch.zeros(0, dtype=torch.long, device=self.accelerator.device)

            local_nll = self.accelerator.pad_across_processes(local_nll, dim=0, pad_index=float("nan"))
            local_indices = self.accelerator.pad_across_processes(local_indices, dim=0, pad_index=-1)
            nll_values = self.accelerator.gather(local_nll).detach().cpu().tolist()
            indices = self.accelerator.gather(local_indices).detach().cpu().tolist()
            return self._ppl_scores(nll_values, indices, positions)
        finally:
            model.train(was_training)

    @staticmethod
    def _ppl_scores(nll_values, indices, positions) -> Dict[int, float]:
        scores: Dict[int, float] = {}
        for nll, index in zip(nll_values, indices):
            if index < 0 or index in scores or not math.isfinite(nll):
                continue
            try:
                scores[int(index)] = math.exp(nll)
            except OverflowError as exc:
                raise ValueError("model_perplexity overflowed; check logits or use model_loss") from exc
        if not scores:
            raise ValueError("model_perplexity produced no valid sample scores")

        # Fill invalid rows before the inherited scores_for lookup. For a
        # subsample, its existing fill logic uses the PPL values returned here.
        ordered = sorted(scores.values())
        middle = len(ordered) // 2
        fill = ordered[middle] if len(ordered) % 2 else ordered[middle - 1] / 2 + ordered[middle] / 2
        missing = len(set(positions) - scores.keys())
        if missing:
            logger.warning(f"[Dataflex][Reorder] {missing} rows have no valid PPL; using the median")
        return {int(index): scores.get(index, fill) for index in positions}

    def describe(self) -> str:
        return f"model_perplexity(batch_size={self.batch_size}, max_samples={self.max_samples})"


def build_score_provider(
    source: str = "precomputed",
    *,
    dataset=None,
    accelerator=None,
    data_collator=None,
    expected_size: Optional[int] = None,
    scores: Optional[Sequence[float]] = None,
    **params,
) -> ScoreProvider:
    """Build a score provider from config.

    Args:
        source: "precomputed" | "cached_selection" | "model_loss" | "model_perplexity".
        scores: an in-memory vector, used by the static path which reads the
            score column straight off the raw dataset.
        **params: forwarded to the chosen provider.
    """
    source = (source or "precomputed").lower()

    if source == "precomputed":
        return PrecomputedScoreProvider(
            scores=scores,
            score_path=params.get("score_path"),
            score_field=params.get("score_field", "score"),
            expected_size=expected_size,
        )

    if source == "cached_selection":
        return CachedSelectionScoreProvider(
            cache_path=params["cache_path"],
            metric_key=params.get("metric_key", "loss"),
            expected_size=expected_size,
            fill_value=params.get("fill_value", 0.0),
        )

    if source == "model_loss":
        return ModelLossScoreProvider(
            dataset=dataset,
            accelerator=accelerator,
            data_collator=data_collator,
            batch_size=params.get("batch_size", 8),
            num_workers=params.get("num_workers", 2),
            max_samples=params.get("max_samples"),
            seed=params.get("seed", 42),
        )

    if source == "model_perplexity":
        return ModelPerplexityScoreProvider(
            dataset=dataset,
            accelerator=accelerator,
            data_collator=data_collator,
            batch_size=params.get("batch_size", 8),
            num_workers=params.get("num_workers", 2),
            max_samples=params.get("max_samples"),
            seed=params.get("seed", 42),
        )

    raise ValueError(
        f"unknown score source '{source}'. Available: precomputed, cached_selection, "
        "model_loss, model_perplexity"
    )
