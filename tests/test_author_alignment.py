"""Compare published functions, candidate composition and portable history."""

import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_multimodal_selection import Collator, CPUAccelerator, TinyVisionModel, fixture_dataset

from dataflex.train.selector.strict_adapt_inf_selector import StrictAdaptInfSelector
from dataflex.utils.author_adapt import obtain_entropy_score
from dataflex.utils.multimodal_selection import fingerprint, score_distribution_entropy


UPSTREAM = Path(__file__).parent / "fixtures/upstream"


def source_function(project, relative, name, scope):
    source = UPSTREAM / project / relative
    manifest = json.loads((UPSTREAM / "manifest.json").read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest[project]["source_sha256"][relative]
    node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    return scope[name]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_predictive_entropy_matches_published_full_sequence(dtype):
    upstream = source_function("adapt-inf", "llava/grad_utils.py", "obtain_entropy_score", {"torch": torch})
    logits = torch.randn(1, 19, 23, generator=torch.Generator().manual_seed(42)).to(dtype)
    outputs = SimpleNamespace(logits=logits)
    torch.testing.assert_close(obtain_entropy_score(outputs), upstream(outputs), rtol=0, atol=0)
    # A prompt change must affect the expert even when answer logits stay fixed.
    changed = logits.clone()
    changed[:, :15] = 0
    assert not torch.equal(obtain_entropy_score(SimpleNamespace(logits=changed)), obtain_entropy_score(outputs))


@pytest.mark.parametrize("scale", [0.01, 0.1, 1.0, 10.0, 100.0])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distribution_entropy_matches_upstream_edges_and_outliers(scale, dtype):
    upstream = source_function("adapt-inf", "data_selection/adapt-inf.py", "get_dist_entropy", {"np": np})
    values = np.random.default_rng(42).normal(size=300).astype(dtype) * scale
    values[-1] = 100 * scale
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = upstream(values)
    actual = score_distribution_entropy(values)
    assert (np.isnan(actual) and np.isnan(expected)) or actual == expected


def strict(dataset, cache, pool=None):
    accelerator = CPUAccelerator()
    accelerator.is_main_process = True
    selector = StrictAdaptInfSelector(
        dataset,
        accelerator,
        Collator(),
        str(cache),
        unique_budget=6,
        proj_dim=16,
        projection="countsketch",
        projection_backend="cpu",
        parameter_pattern="lora_",
        cluster_grid=[2],
        num_clusters=2,
        dataset_revision="fixture-v1",
    )
    if pool is not None:
        selector.set_candidate_pool(pool)
    return selector


def test_strict_candidate_pool_applies_to_warmup_scoring_and_successive_updates(tmp_path):
    _, dataset = fixture_dataset(tmp_path)
    pool = [1, 2, 3, 4, 5, 6, 7, 8]
    selector = strict(dataset, tmp_path / "cache", pool)
    warmup = selector.warmup(2)
    model = TinyVisionModel()
    first = selector.select(model, 2, 2)
    second = selector.select(model, 4, 2)
    assert selector.get_candidate_pool() == pool
    assert set(warmup + first + second) <= set(pool)
    assert len(set(warmup + first + second)) == 6
    for path in (tmp_path / "cache").glob("step_*/dataset_positions.npy"):
        assert set(np.load(path)) <= set(pool)
    with pytest.raises(ValueError, match="No unused"):
        selector.select(model, 6, 1)


def test_strict_history_restores_into_empty_cache_and_rejects_missing_or_corrupt_state(tmp_path):
    _, dataset = fixture_dataset(tmp_path)
    pool = [1, 2, 3, 4, 5, 6, 7, 8]
    original = strict(dataset, tmp_path / "original", pool)
    warmup = original.warmup(2)
    model = TinyVisionModel()
    first = original.select(model, 2, 2)
    state = original.export_checkpoint_state(global_step=3, opened_step=2, active_indices=first)
    restored = strict(dataset, tmp_path / "relocated", pool)
    restored.restore_checkpoint_state(state, global_step=3, opened_step=2, active_indices=first)
    expected = original.select(model, 4, 4)  # remaining budget clips to 2
    actual = restored.select(model, 4, 4)
    assert actual == expected and len(actual) == 2 and not set(actual) & set(warmup + first)
    with pytest.raises(ValueError, match="state missing"):
        restored.restore_checkpoint_state(None, global_step=3, opened_step=2, active_indices=first)
    corrupt = copy.deepcopy(state)
    corrupt["stages"][0]["indices"][0] = first[0]
    corrupt["digest"] = fingerprint({k: v for k, v in corrupt.items() if k != "digest"})
    with pytest.raises(ValueError, match="overlapping"):
        restored.restore_checkpoint_state(corrupt, global_step=3, opened_step=2, active_indices=first)


def test_published_expert_choice_including_nan_and_tie_order():
    upstream_entropy = source_function("adapt-inf", "data_selection/adapt-inf.py", "get_dist_entropy", {"np": np})
    upstream = source_function(
        "adapt-inf", "data_selection/adapt-inf.py", "get_mwp_score", {"np": np, "get_dist_entropy": upstream_entropy}
    )
    rng = np.random.default_rng(42)
    cases = [
        {"entropy": np.ones(30), "el2n": np.arange(30) / 100, "ppl": np.arange(30) * 3.0, "ig": rng.normal(size=30)},
        {k: rng.normal(size=200) * scale for k, scale in zip(["entropy", "el2n", "ppl", "ig"], [0.1, 0.3, 1, 10])},
    ]
    for scores in cases:
        expected = upstream(scores, np.arange(len(scores["ppl"])))
        columns = [scores[k] for k in ["ppl", "el2n", "entropy", "ig"]]
        entropies = [score_distribution_entropy(x) for x in columns]
        order = [2, 1, 0, 3]
        expert = order[int(np.argmax([entropies[i] for i in order]))]
        np.testing.assert_array_equal(columns[expert], expected)


def test_vendored_adapt_function_bodies_are_unmodified():
    import dataflex.utils.author_adapt as vendor

    vendored = ast.parse(Path(vendor.__file__).read_text())
    for name, relative in [
        ("obtain_entropy_score", "llava/grad_utils.py"),
        ("get_dist_entropy", "data_selection/adapt-inf.py"),
        ("obtain_ig_score", "llava/grad_utils.py"),
        ("set_outlier_threshold", "data_selection/adapt-inf.py"),
        ("get_ccs_samples", "data_selection/adapt-inf.py"),
    ]:
        source = ast.parse((UPSTREAM / "adapt-inf" / relative).read_text())
        original = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == name)
        current = next(n for n in vendored.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(current) == ast.dump(original)
