"""Input/RNG/budget boundaries around the unchanged published Adapt functions."""

import contextlib
import io
import random
from types import SimpleNamespace

import numpy as np

from dataflex.utils.author_adapt import get_ccs_samples, obtain_ig_score
from dataflex.utils.multimodal_scoring import answer_statistics


class _ImageArgumentAdapter:
    def __init__(self, model):
        self.model = model

    def __call__(self, **batch):
        # LLaVA author's `images` and HF's `pixel_values` are input API names.
        # Keep all visual token positions and the same processed zero tensor.
        pixels = batch.pop("images")
        result = self.model(**{**batch, "pixel_values": pixels})
        if getattr(result, "loss", None) is not None:
            return result
        # Tiny model fixtures may expose logits only; real HF models return loss.
        return SimpleNamespace(loss=answer_statistics(result.logits, batch["labels"])["loss"])


def author_zero_image_ig(model, batch, image_loss):
    if "pixel_values" not in batch:
        raise ValueError("Author IG requires processed image pixels")
    inputs = {k: v for k, v in batch.items() if k != "pixel_values"}
    inputs["images"] = batch["pixel_values"]
    # The upstream function replaces only the shallow copy's image tensor.
    return obtain_ig_score(_ImageArgumentAdapter(model), inputs, image_loss.detach().exp(), 1)


def author_ccs_indices(scores, budget, rng, bins=50):
    """Published interval/ceil/redistribution rules, with isolated Python RNG."""
    scores = np.asarray(scores)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("CCS needs a finite nonempty vector")
    if type(budget) is not int or budget < 0 or bins < 1:
        raise ValueError("Invalid CCS budget/bins")
    saved = random.getstate()
    random.setstate(rng.getstate())
    try:
        # Suppress author console output, not any sampling operation.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = get_ccs_samples(scores, bins, budget)
    finally:
        rng.setstate(random.getstate())
        random.setstate(saved)
    return result.astype(np.int64).tolist()


def fit_selection_budget(selected, positions, budget, rng, policy):
    """Keep raw author output, then explicitly enforce a caller's exact budget.

    `native` returns the original count. `exact` is the experiment's unique
    record/exposure constraint; every truncation/fill is recorded separately.
    This must never be represented as a property of the author's CCS algorithm.
    """
    if policy not in ("native", "exact"):
        raise ValueError("selection_budget must be native or exact")
    pool = set(positions)
    if len(selected) != len(set(selected)) or not set(selected) <= pool:
        raise ValueError("Invalid author selection indices")
    raw_count = len(selected)
    result = list(selected)
    truncated = filled = 0
    if policy == "exact":
        if budget > len(pool):
            raise ValueError("Unique budget exceeds candidate pool")
        truncated = max(0, raw_count - budget)
        result = result[:budget]
        filled = max(0, budget - len(result))
        if filled:
            used = set(result)
            remaining = [i for i in positions if i not in used]
            result.extend(rng.sample(remaining, filled))
    if not result and budget:
        raise ValueError("Published CCS selected no samples; use an explicit exact-budget policy if required")
    return result, {
        "policy": policy,
        "author_selected_count": raw_count,
        "truncated_count": truncated,
        "filled_count": filled,
    }
