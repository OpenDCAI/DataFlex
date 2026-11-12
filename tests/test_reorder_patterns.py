import numpy as np
import pytest

from dataflex.train.reorder import DynamicReorder, StaticReorder
from dataflex.train.reorder.patterns import sort_positions


@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize(
    ("positions", "scores"),
    [
        ([], []),
        ([17], [2.0]),
        ([10, 5, 8], [1.0, 3.0, 2.0]),
        ([11, 3, 8, 99, 2], [1.0, 2.0, 2.0, 1.0, 3.0]),
        ([11, 3, 8, 99, 2], [1.0] * 5),
        ([4, 1, 7, 2, 9, 3], [-1.0, 0.0, -0.0, 1.0, 1.0, -1.0]),
    ],
)
def test_sort_positions_matches_stable_sort(positions, scores, ascending):
    expected = [positions[i] for i in sorted(range(len(scores)), key=scores.__getitem__, reverse=not ascending)]
    score_array = np.asarray(scores, dtype=np.float64)
    original_scores = score_array.copy()

    assert sort_positions(positions, score_array, ascending=ascending) == expected
    np.testing.assert_array_equal(score_array, original_scores)


@pytest.mark.parametrize(
    ("polarity", "ascending"),
    [("higher_later", False), ("higher_earlier", True), ("lower_is_better", True)],
)
def test_static_reorder_preserves_tied_raw_rows(polarity, ascending):
    reorder = StaticReorder(pattern="sorting", polarity=polarity, ascending=ascending)

    assert reorder.order_rows([1.0, 2.0, 2.0, 1.0, 3.0]) == [4, 1, 2, 0, 3]


@pytest.mark.parametrize("reorder_class", [StaticReorder, DynamicReorder])
def test_reorder_serves_tied_indices_in_original_order(reorder_class):
    options = {"apply_at": "index"} if reorder_class is StaticReorder else {}
    reorder = reorder_class(
        pattern="sorting",
        polarity="higher_earlier",
        score_source="precomputed",
        score_params={"scores": [1.0, 2.0, 2.0, 1.0, 3.0]},
        dataset=list(range(5)),
        **options,
    )

    assert reorder.next_indices(model=None, step_id=0, num_samples=3) == [4, 1, 2]
    assert reorder.next_indices(model=None, step_id=1, num_samples=2) == [0, 3]
