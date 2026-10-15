import numpy as np
import pytest
import torch

from dataflex.train.selector.strict_adapt_inf_selector import CachedRademacher, remaining_record_budget


def test_registry_injects_runtime_dependencies(tmp_path):
    from types import SimpleNamespace

    from dataflex.core.registry import REGISTRY

    runtime = {
        "dataset": [{"input_ids": [1]}] * 20,
        "accelerator": SimpleNamespace(is_main_process=True),
        "data_collator": lambda x: x,
    }
    selector = REGISTRY.build(
        "selector",
        "adapt_inf_strict",
        runtime=runtime,
        cfg={"cache_dir": str(tmp_path), "unique_budget": 10, "dataset_revision": "fixture"},
    )
    assert selector.unique_budget == 10 and selector.dataset is runtime["dataset"]


def test_total_budget_disjoint_three_intervals():
    count, used = remaining_record_budget(93690, [list(range(12800))], 48896)
    assert count == 48896 and len(used) == 12800
    count, used = remaining_record_budget(93690, [list(range(12800)), list(range(12800, 61696))], 48896)
    assert count == 31994 and len(used) == 61696
    assert 12800 + 48896 + count == 93690
    with pytest.raises(ValueError):
        remaining_record_budget(10, [[1, 2], [2, 3]], 5)


def test_rademacher_projection_is_dense_seeded_and_scaled():
    gradients = [torch.arange(3, dtype=torch.float32), torch.tensor([2.0, -1.0])]
    one = CachedRademacher(gradients, 128, 42, "cpu")
    two = CachedRademacher(gradients, 128, 42, "cpu")
    assert torch.equal(one.matrix, two.matrix)
    assert set(one.matrix.abs().float().unique().tolist()) == {float(torch.tensor(128**-0.5, dtype=torch.bfloat16))}
    assert torch.count_nonzero(one.matrix) == 5 * 128
    expected = (torch.cat(gradients).to(torch.bfloat16)[None] @ one.matrix)[0].float().numpy()
    np.testing.assert_array_equal(one(gradients), expected)


def test_training_marker_collator_never_modifies_source_features():
    from dataflex.train.dataset.record_tracking import IndexedCollator, IndexedDataset

    dataset = IndexedDataset([{"input_ids": [1, 2], "labels": [1, 2]}])
    row = dataset[0]

    def base(features):
        assert "__source_position__" not in features[0]
        return {"input_ids": torch.tensor([x["input_ids"] for x in features])}

    collator = IndexedCollator(base)
    result = collator([row])
    assert result["__record_indices__"].tolist() == [0]
    assert row["__source_position__"] == 0
