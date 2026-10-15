"""Published zero-image IG and CCS are evaluated against pinned source."""

import contextlib
import io
import math
import random

import numpy as np
import torch
from test_author_alignment import source_function
from test_multimodal_llava_integration import tiny_llava
from test_multimodal_selection import fixture_dataset

from dataflex.offline_selector.offline_presel_selector import paired_inputs
from dataflex.utils.author_adapters import author_ccs_indices, author_zero_image_ig, fit_selection_budget


def test_zero_image_ig_matches_author_on_real_tiny_llava_and_preserves_inputs(tmp_path):
    model, processor = tiny_llava()
    model.eval()
    records, _ = fixture_dataset(tmp_path, 1)
    batch = paired_inputs(processor, records[0], records[0]["image"])
    original_pixels = batch["pixel_values"].clone()
    original_ids = batch["input_ids"].clone()
    observed = []
    hook = model.model.vision_tower.register_forward_pre_hook(
        lambda module, args, kwargs: observed.append(kwargs.get("pixel_values", args[0] if args else None).clone()),
        with_kwargs=True,
    )
    with torch.no_grad():
        loss = model(**batch).loss
        observed.clear()
        actual = author_zero_image_ig(model, batch, loss)
        assert len(observed) == 1 and torch.count_nonzero(observed[0]) == 0
        upstream = source_function("adapt-inf", "llava/grad_utils.py", "obtain_ig_score", {"torch": torch})

        class ReferenceAdapter:
            def __call__(self, **inputs):
                return model(
                    **{**{k: v for k, v in inputs.items() if k != "images"}, "pixel_values": inputs["images"]}
                )

        reference_batch = {k: v for k, v in batch.items() if k != "pixel_values"}
        reference_batch["images"] = original_pixels
        expected = upstream(ReferenceAdapter(), reference_batch, loss.exp(), 1)
    hook.remove()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(batch["pixel_values"], original_pixels) and torch.equal(batch["input_ids"], original_ids)


def published_ccs():
    threshold = source_function("adapt-inf", "data_selection/adapt-inf.py", "set_outlier_threshold", {"np": np})
    return source_function(
        "adapt-inf",
        "data_selection/adapt-inf.py",
        "get_ccs_samples",
        {"np": np, "math": math, "random": random, "tqdm": lambda x, **kwargs: x, "set_outlier_threshold": threshold},
    )


def test_ccs_raw_indices_and_python_rng_match_author_including_overshoot():
    scores = np.linspace(0, 1, 2000)
    old = random.getstate()
    random.seed(42)
    with contextlib.redirect_stdout(io.StringIO()):
        expected = published_ccs()(scores, 3, 5)
    random.setstate(old)
    actual = author_ccs_indices(scores, 5, random.Random(42), bins=3)
    assert random.getstate() == old
    assert actual == expected.tolist() and len(actual) == 6  # exact original ceil behavior
    constant = author_ccs_indices(np.ones(100), 5, random.Random(42))
    assert constant == []


def test_framework_budget_adapter_is_explicit_and_keeps_native_output():
    raw = [8, 5, 3, 1, 2, 4]
    pool = list(range(10))
    unchanged, metadata = fit_selection_budget(raw, pool, 5, random.Random(42), "native")
    assert unchanged == raw and metadata["truncated_count"] == metadata["filled_count"] == 0
    capped, metadata = fit_selection_budget(raw, pool, 5, random.Random(42), "exact")
    assert capped == raw[:5] and metadata["author_selected_count"] == 6 and metadata["truncated_count"] == 1
    filled, metadata = fit_selection_budget([8, 5], pool, 5, random.Random(42), "exact")
    assert filled[:2] == [8, 5] and len(set(filled)) == 5 and metadata["filled_count"] == 3


def test_native_selector_keeps_raw_count_and_reuses_cache(tmp_path):
    from test_multimodal_selection import Collator, CPUAccelerator, TinyVisionModel

    from dataflex.train.selector.adapt_inf_selector import AdaptInfSelector

    _, dataset = fixture_dataset(tmp_path, 12)
    # A one-cluster real score pass can legitimately give a variable native
    # count. A fixture patch changes only CCS output, exercising native cache I/O.
    from unittest.mock import patch

    model = TinyVisionModel()
    selector = AdaptInfSelector(
        dataset,
        CPUAccelerator(),
        Collator(),
        str(tmp_path / "cache"),
        num_clusters=1,
        proj_dim=8,
        parameter_pattern="lora_",
        selection_budget="native",
    )
    with patch("dataflex.train.selector.adapt_inf_selector.author_ccs_indices", return_value=[0, 1, 2]):
        first = selector.select(model, 1, 2)
    assert len(first) == 3
    assert selector.select(model, 1, 2) == first and selector.last_metadata["cache_hit"]


def test_author_ig_with_nonreentrant_llava_checkpointing(tmp_path):
    from dataflex.utils.multimodal_scoring import answer_statistics, scoring_state

    model, processor = tiny_llava()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    records, _ = fixture_dataset(tmp_path, 1)
    batch = paired_inputs(processor, records[0], records[0]["image"])
    batch["use_cache"] = False
    with scoring_state(model, checkpoint_llama=True):
        outputs = model(**{k: v for k, v in batch.items() if k != "labels"})
        stats = answer_statistics(outputs.logits, batch["labels"])
        torch.autograd.grad(stats["loss"], list(model.model.language_model.layers[0].parameters()))
        result = author_zero_image_ig(model, batch, stats["loss"])
        assert torch.isfinite(result) and result > 0
