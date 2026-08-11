"""Data ordering components.

Importing this package is what registers the reorderers, so every reorderer must
be imported here. Registry names must be unique or `REGISTRY.register` raises at
import time.
"""

from .base_reorderer import POLARITIES, Reorderer
from .dynamic_reorderer import DynamicReorderer
from .patterns import PATTERNS, apply_pattern
from .score_provider import (
    CachedSelectionScoreProvider,
    ModelLossScoreProvider,
    PrecomputedScoreProvider,
    ScoreProvider,
    build_score_provider,
)
from .static_reorderer import StaticReorderer

#: Registered reorderer classes. Unlike the other families, a reorderer preset in
#: components.yaml is named after the *ordering* it produces (`saw`, `folding`,
#: ...) rather than after its class, because the ordering is chosen by config.
#: `kind` is what maps a preset onto one of these.
KINDS = ("static", "dynamic")


def resolve_reorderer_kind(component_name: str, params: dict) -> str:
    """Decide which reorderer class a components.yaml preset refers to.

    Falls back to the preset name so a preset named directly after a class still
    works without a `kind`, matching how selectors/mixers/weighters behave.
    """
    kind = params.pop("kind", None) or component_name
    if kind not in KINDS:
        raise ValueError(
            f"reorderer preset '{component_name}' resolves to kind '{kind}', which is not registered. "
            f"Set `kind` to one of {list(KINDS)} in its params block."
        )
    return kind


__all__ = [
    "POLARITIES",
    "PATTERNS",
    "KINDS",
    "Reorderer",
    "StaticReorderer",
    "DynamicReorderer",
    "ScoreProvider",
    "PrecomputedScoreProvider",
    "CachedSelectionScoreProvider",
    "ModelLossScoreProvider",
    "build_score_provider",
    "apply_pattern",
    "resolve_reorderer_kind",
]
