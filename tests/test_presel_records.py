import ast
from pathlib import Path

import numpy as np
import pytest

from dataflex.offline_selector.record_presel_selector import (
    combine_selection,
    importance_by_record,
    official_typical_selector,
    reference_indices,
    select_task,
    task_quotas,
    without_question_record,
)


AUTHOR = Path(__file__).parent / "fixtures/upstream/PreSel"


def pool(n=1000):
    # Many different records share one physical image, as in the real pool.
    return [
        {
            "task": f"official_task_{i % 10 + 1}",
            "image": "shared.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\nQuestion"},
                {"from": "gpt", "value": f"Answer {i}"},
            ],
        }
        for i in range(n)
    ]


def test_record_budget_reference_inclusion_and_no_image_expansion():
    records = pool()
    reference, quotas = reference_indices(records, 42)
    assert len(reference) == 50 and all(q == 5 for q in quotas.values())
    assert reference_indices(records, 42)[0] == reference
    extra = [i for i in range(1000) if i not in reference][:80]
    final, fill = combine_selection(records, reference, [extra], 42)
    assert len(final) == 150 and len(set(final)) == 150
    assert set(reference) <= set(final) and len(fill) == 20
    assert len(final) < len(records)  # A repeated image must never expand to every record.
    with pytest.raises(ValueError):
        combine_selection(records, reference, [extra + extra], 42)


def test_irs_is_mean_of_record_ratios_and_author_task_budget_rounding():
    records = pool(100)
    reference = list(range(20))
    scores = [{"index": i, "loss_with_question": float(i + 1), "loss_without_question": 2.0} for i in reference]
    importance, t = importance_by_record(records, reference, scores)
    assert importance["official_task_1"]["mean_irs"] == 3.0
    assert t == pytest.approx(1 / np.sqrt(10))
    tasks = list(importance)
    q = task_quotas(importance, tasks, 1000)
    weights = np.array([importance[x]["weight"] for x in tasks])
    expected = (weights * 1000).astype(int)
    expected[np.argmax(weights)] += 1000 - expected.sum()
    assert list(q.values()) == expected.tolist() and sum(q.values()) == 1000


def test_no_question_transform_matches_author_multiturn_preprocess():
    row = pool(1)[0]
    row["conversations"] += [{"from": "human", "value": "Second question"}, {"from": "gpt", "value": "Second answer"}]
    tree = ast.parse((AUTHOR / "data_process/remove_instruction.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "preprocess")
    scope = {"tqdm": lambda x: x}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "author_remove_instruction", "exec"), scope)
    author = scope["preprocess"]([{**row, "id": "x", "unique_idx": 0}])[0]
    assert without_question_record(row)["conversations"] == author["conversations"]


def test_clustering_nc_and_cluster_rounding_use_author_function():
    rng = np.random.default_rng(42)
    features = rng.normal(size=(500, 16)).astype(np.float32)
    features[1] = features[0]  # Duplicate-image records stay independently eligible.
    ids = np.arange(500)
    reference = list(range(25))
    selected, labels, _ = select_task("official_task_1", ids, features, reference, 50, AUTHOR, 42)
    normalized = features / np.linalg.norm(features, axis=1, keepdims=True)
    expected = official_typical_selector(AUTHOR)(ids, normalized, labels, 50, set(reference))
    assert official_typical_selector()(ids, normalized, labels, 50, set(reference)) == expected
    assert selected == expected and len(selected) == 50 and not set(selected) & set(reference)
