"""Input and RNG adapters for the published Adapt-inf functions."""

# Uniform cluster-budget allocation is adapted from the upstream source.
# MIT License
#
# Copyright (c) 2024 Adyasha Maharana
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import contextlib
import io
import math
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


def author_cluster_budgets(capacities, budget):
    """Released uniform cluster-budget allocation, including ceil rounding.

    Removes the upstream CLI-global args.n_clusters dependency only. The
    selected count is not subsequently truncated or filled to fit the budget.
    """
    capacities = np.asarray(capacities)
    clusters = len(capacities)
    if not clusters or np.any(capacities <= 0) or budget < 0:
        raise ValueError("Cluster capacities must be positive and budget nonnegative")
    if int(capacities.min()) * clusters >= budget:
        return [math.ceil(budget / clusters)] * clusters
    order = np.argsort(capacities)
    per_cluster = math.ceil(budget / clusters)
    budgets = []
    for i, index in enumerate(order):
        size = int(capacities[index])
        if size < per_cluster:
            budgets.append(size)
            if i + 1 < clusters:
                per_cluster += math.ceil((per_cluster - size) / (clusters - i - 1))
        else:
            budgets.append(per_cluster)
    return [budgets[index] for index in np.argsort(order)]
