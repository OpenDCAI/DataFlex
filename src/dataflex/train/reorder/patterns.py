"""Score-based data ordering patterns.

Every function here is a pure permutation over a list of *positions*. A position
is an opaque integer: for the static path it is a row of the raw dataset, for the
dynamic path it is an index into the preprocessed ``train_dataset``. Nothing in
this module touches a model, a dataset object, or the filesystem.

Ported from the reference implementation of "Demystifying Data Organization for
Enhanced LLM Training" (https://github.com/microsoft/data-efficacy). The paper's
four guidances map onto the patterns as follows:

    G1 Boundary Sharpening  -> segment (SEG)
    G2 Cyclic Scheduling    -> folding (FO)
    G3 Curriculum Continuity-> zigzag (ZIG)
    G4 Local Diversity      -> window_size > 1 (JIT), composable with any pattern
    G1+G2+G4                -> stair (STR)
    G1+G2+G3+G4             -> saw (SAW)
"""

import random
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

PATTERNS = ("shuffle", "sorting", "folding", "zigzag", "segment", "stair", "saw")

# The reference implementation exposes the fold count under two different config
# keys (``folding_layer`` for FO/STR/SAW, ``zigzag_layer`` for ZIG). We accept a
# single ``layers`` key and treat those two as aliases so existing paper configs
# can be pasted in unchanged.
_LAYER_ALIASES = ("layers", "folding_layer", "zigzag_layer")


def _validate_layers(layers: int, name: str = "layers") -> int:
    layers = int(layers)
    if layers < 1:
        raise ValueError(f"{name} must be >= 1, got {layers}")
    return layers


def resolve_layers(params: Dict, default: int = 1) -> int:
    for key in _LAYER_ALIASES:
        if params.get(key) is not None:
            return _validate_layers(params[key], key)
    return _validate_layers(default)


def sort_positions(
    positions: Sequence[int],
    scores: Sequence[float],
    ascending: bool = True,
) -> List[int]:
    """Order positions by score. Ties keep their incoming relative order."""
    if len(positions) != len(scores):
        raise ValueError(f"positions/scores length mismatch: {len(positions)} vs {len(scores)}")
    order = np.argsort(np.asarray(scores, dtype=np.float64), kind="stable")
    if not ascending:
        order = order[::-1]
    positions = np.asarray(positions)
    return positions[order].tolist()


def window_shuffle(positions: Sequence[int], window_size: int = 0, seed: int = 42) -> List[int]:
    """Jittering Ordering (JIT, G4): shuffle inside contiguous windows of size w.

    Preserves the global trend while restoring gradient diversity inside a
    mini-batch. A single RNG is threaded through all windows, matching the
    reference implementation.
    """
    if not window_size or window_size <= 1:
        return list(positions)

    rng = np.random.default_rng(seed)
    out: List[int] = []
    for start in range(0, len(positions), window_size):
        chunk = list(positions[start : start + window_size])
        rng.shuffle(chunk)
        out.extend(chunk)
    return out


def folding_order(sorted_positions: Sequence[int], layers: int) -> List[int]:
    """Folding Ordering (FO, G2): strided partition into ``layers`` cycles.

    Each cycle sweeps the whole score spectrum low to high, so the model
    periodically revisits easy material instead of monotonically leaving it
    behind.
    """
    layers = _validate_layers(layers, "folding_layer")
    out: List[int] = []
    for layer in range(layers):
        out.extend(sorted_positions[layer::layers])
    return out


def zigzag_order(sorted_positions: Sequence[int], layers: int) -> List[int]:
    """Zig-zag Ordering (ZIG, G3): FO with odd cycles reversed.

    Turns FO's sawtooth into a triangle wave so consecutive cycles meet at
    similar scores, removing the gradient-norm spike at cycle boundaries.
    """
    layers = _validate_layers(layers, "zigzag_layer")
    out: List[int] = []
    for layer in range(layers):
        cycle = list(sorted_positions[layer::layers])
        if layer % 2 == 1:
            cycle.reverse()
        out.extend(cycle)
    return out


def cross_guidance_order(
    sorted_positions: Sequence[int],
    num_sections: int,
    transition_ratio: float,
    layers: int,
    mode: str,
) -> List[int]:
    """Shared body of STR and SAW.

    The sequence stays monotonic except inside ``num_sections - 1`` transition
    regions of radius ``transition_ratio * N`` centred on evenly spaced split
    points, where FO (``mode="folding"``, STR) or ZIG (``mode="zigzag"``, SAW)
    is applied locally.
    """
    num_sections = _validate_layers(num_sections, "num_sections")
    layers = _validate_layers(layers, "folding_layer")
    if transition_ratio < 0:
        raise ValueError(f"transition_ratio must be >= 0, got {transition_ratio}")
    if mode not in {"folding", "zigzag"}:
        raise ValueError(f"Unsupported transition mode: {mode}")

    n_items = len(sorted_positions)
    if n_items == 0 or num_sections == 1 or transition_ratio == 0:
        return list(sorted_positions)

    # The radius is a fraction of the whole dataset, not of a section, so
    # regions overlap once 2*rho exceeds a section width. The reference code
    # silently merges them via a cursor clamp; we refuse instead, because a
    # merged region means the realised pattern is not the configured one.
    if 2.0 * transition_ratio >= 1.0 / num_sections:
        raise ValueError(
            f"transition regions overlap: 2*transition_ratio ({2 * transition_ratio:.4f}) "
            f"must be < 1/num_sections ({1.0 / num_sections:.4f}). "
            f"Lower transition_ratio or num_sections."
        )

    split_points = [round(n_items * s / num_sections) for s in range(1, num_sections)]
    radius = round(n_items * transition_ratio)
    transform = folding_order if mode == "folding" else zigzag_order

    out: List[int] = []
    cursor = 0
    for split_point in split_points:
        start = max(cursor, split_point - radius)
        end = min(n_items, split_point + radius)
        if cursor < start:
            out.extend(sorted_positions[cursor:start])
        out.extend(transform(list(sorted_positions[start:end]), layers))
        cursor = end

    if cursor < n_items:
        out.extend(sorted_positions[cursor:n_items])
    return out


def segment_order(
    sorted_positions: Sequence[int],
    x_pct: float = 10,
    y_pct: float = 10,
    front_is_high: bool = False,
    back_is_high: bool = True,
    seed: int = 42,
) -> List[int]:
    """Segment Ordering (SEG, G1): control the score distribution at the boundaries.

    Draws the leading ``x_pct`` and trailing ``y_pct`` of training from a chosen
    score tail, shuffles each of the three segments independently, and leaves the
    rest in the middle. The default (low first, high last) is the paper's
    SEG(l10-h10).
    """
    # stdlib RNG, matching the reference implementation of SEG bit for bit.
    rng = random.Random(seed)

    total = len(sorted_positions)
    n_front = int(total * x_pct // 100)
    n_back = int(total * y_pct // 100)

    if n_front + n_back > total:
        ratio = total / (n_front + n_back)
        n_front = int(n_front * ratio)
        n_back = total - n_front

    def take(seq: Sequence[int], n: int, high: bool):
        if n <= 0:
            return [], list(seq)
        if high:
            return list(seq[-n:]), list(seq[:-n])
        return list(seq[:n]), list(seq[n:])

    if bool(front_is_high) == bool(back_is_high):
        # Both ends come from the same tail; split it randomly between them.
        selected, middle = take(sorted_positions, n_front + n_back, high=bool(front_is_high))
        rng.shuffle(selected)
        front, back = selected[:n_front], selected[n_front:]
    else:
        front, remaining = take(sorted_positions, n_front, high=bool(front_is_high))
        back, middle = take(remaining, n_back, high=bool(back_is_high))

    rng.shuffle(front)
    rng.shuffle(middle)
    rng.shuffle(back)
    return front + middle + back


def _pattern_shuffle(positions, scores, params) -> List[int]:
    out = list(positions)
    random.Random(params.get("seed", 42)).shuffle(out)
    return out


def _pattern_sorting(positions, scores, params) -> List[int]:
    return sort_positions(positions, scores, ascending=params.get("ascending", True))


def _pattern_folding(positions, scores, params) -> List[int]:
    ordered = sort_positions(positions, scores, ascending=params.get("ascending", True))
    return folding_order(ordered, resolve_layers(params, default=5))


def _pattern_zigzag(positions, scores, params) -> List[int]:
    ordered = sort_positions(positions, scores, ascending=params.get("ascending", True))
    return zigzag_order(ordered, resolve_layers(params, default=5))


def _pattern_segment(positions, scores, params) -> List[int]:
    # The reference SEG always sorts ascending and drops `ascending`; we honour
    # it so polarity handling stays uniform across every pattern.
    ordered = sort_positions(positions, scores, ascending=params.get("ascending", True))
    return segment_order(
        ordered,
        x_pct=params.get("x_pct", 10),
        y_pct=params.get("y_pct", 10),
        front_is_high=params.get("front_is_high", False),
        back_is_high=params.get("back_is_high", True),
        seed=params.get("seed", 42),
    )


def _cross_guidance(positions, scores, params, mode: str) -> List[int]:
    ordered = sort_positions(positions, scores, ascending=params.get("ascending", True))
    return cross_guidance_order(
        ordered,
        num_sections=params.get("num_sections", 2),
        transition_ratio=params.get("folding_ratio", 0.0),
        layers=resolve_layers(params, default=2),
        mode=mode,
    )


_DISPATCH: Dict[str, Callable] = {
    "shuffle": _pattern_shuffle,
    "sorting": _pattern_sorting,
    "folding": _pattern_folding,
    "zigzag": _pattern_zigzag,
    "segment": _pattern_segment,
    "stair": lambda p, s, prm: _cross_guidance(p, s, prm, "folding"),
    "saw": lambda p, s, prm: _cross_guidance(p, s, prm, "zigzag"),
}


def assert_permutation(positions: Sequence[int], out: Sequence[int]) -> None:
    """Verify `out` is a true permutation of `positions`.

    The reference implementation only compares lengths, which cannot catch a
    pattern that duplicates one element and drops another.
    """
    if len(positions) != len(out):
        raise ValueError(f"ordering changed size from {len(positions)} to {len(out)}")
    if sorted(positions) != sorted(out):
        raise ValueError("ordering result is not a permutation of its input")


def apply_pattern(
    pattern: str,
    positions: Sequence[int],
    scores: Optional[Sequence[float]] = None,
    **params,
) -> List[int]:
    """Order `positions` by `scores` under the named pattern, then apply JIT.

    Args:
        pattern: one of `PATTERNS`.
        positions: opaque integer ids to permute.
        scores: one score per position. Only `shuffle` may omit it.
        **params: pattern hyperparameters plus the shared `window_size`/`seed`.

    Returns:
        A permutation of `positions`.
    """
    if pattern not in _DISPATCH:
        raise ValueError(f"unknown pattern '{pattern}'. Available: {', '.join(PATTERNS)}")

    positions = list(positions)
    if not positions:
        return []

    if scores is None:
        if pattern != "shuffle":
            raise ValueError(f"pattern '{pattern}' requires scores")
        scores = [0.0] * len(positions)

    out = _DISPATCH[pattern](positions, scores, params)
    out = window_shuffle(out, window_size=params.get("window_size", 0), seed=params.get("seed", 42))
    assert_permutation(positions, out)
    return out
