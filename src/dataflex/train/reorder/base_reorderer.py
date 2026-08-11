from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

from dataflex.utils.logging import logger

from .patterns import PATTERNS, apply_pattern

#: Which end of the raw score the curriculum should finish on.
#:
#: Every pattern arranges samples along one axis that advances as training
#: progresses; `polarity` says how the raw score maps onto that axis, and it
#: exists because "high score" means opposite things for different sources.
#:
#:   higher_later   - a larger raw score is presented later. Correct for a
#:                    quality score (FineWeb-Edu, QuRating: finish on the best
#:                    data, which is the paper's G1) and also for loss used as
#:                    difficulty (finish on the hardest, i.e. easy-to-hard).
#:   higher_earlier - a larger raw score is presented earlier. Needed when the
#:                    raw number runs backwards relative to the axis, e.g. a
#:                    perplexity standing in for quality, where low is good.
#:
#: `higher_is_better` / `lower_is_better` are accepted as aliases because they
#: read naturally for quality scores.
POLARITIES = ("higher_later", "higher_earlier")

_POLARITY_ALIASES = {
    "higher_is_better": "higher_later",
    "lower_is_better": "higher_earlier",
}


class Reorderer(ABC):
    """Base class for data ordering components.

    A reorderer turns a set of *positions* plus their scores into a permutation.
    Positions are opaque integers whose meaning depends on the path: raw dataset
    rows for the static path, indices into the preprocessed ``train_dataset`` for
    the dynamic path.

    Unlike a selector it never changes the number of samples, and unlike a mixer
    it knows nothing about domains. It only decides *order*.
    """

    def __init__(
        self,
        pattern: str = "sorting",
        window_size: int = 0,
        polarity: str = "higher_later",
        ascending: bool = True,
        seed: int = 42,
        pattern_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if pattern not in PATTERNS:
            raise ValueError(f"unknown pattern '{pattern}'. Available: {', '.join(PATTERNS)}")
        polarity = _POLARITY_ALIASES.get(polarity, polarity)
        if polarity not in POLARITIES:
            raise ValueError(
                f"unknown polarity '{polarity}'. Available: {', '.join(POLARITIES)} "
                f"(aliases: {', '.join(_POLARITY_ALIASES)})"
            )

        self.pattern = pattern
        self.window_size = int(window_size or 0)
        self.polarity = polarity
        self.ascending = bool(ascending)
        self.seed = int(seed)

        # Pattern hyperparameters (layers, num_sections, folding_ratio, x_pct,
        # ...) are passed straight through so a new pattern needs no signature
        # change here. Explicit `pattern_params` wins over loose kwargs.
        self.pattern_params: Dict[str, Any] = {
            k: v for k, v in kwargs.items() if k not in {"dataset", "accelerator", "data_collator"}
        }
        self.pattern_params.update(pattern_params or {})

        # ---- composition hooks (see set_* below) ----
        self._candidate_pool: Optional[List[int]] = None
        self._domain_ids: Optional[Sequence[int]] = None
        self._signals: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Composition hooks
    #
    # These exist so reorder can later be chained with select/mix/weight into a
    # single data pipeline without reworking the component. They are all no-ops
    # by default, so a standalone reorderer behaves exactly as if they were
    # absent.
    # ------------------------------------------------------------------

    def set_candidate_pool(self, indices: Optional[Sequence[int]]) -> None:
        """Restrict ordering to a subset of positions.

        This is the seam for chaining after a selector: the selector decides
        *which* samples survive, the reorderer then decides their order. When
        unset, the reorderer orders everything it is given.
        """
        self._candidate_pool = list(indices) if indices is not None else None

    def get_candidate_pool(self) -> Optional[List[int]]:
        return list(self._candidate_pool) if self._candidate_pool is not None else None

    def set_domain_ids(self, domain_ids: Optional[Sequence[int]]) -> None:
        """Attach a domain label per position.

        The seam for chaining with a mixer: with domain labels available a
        future strategy can order within each domain and interleave according to
        the mixer's proportions, rather than ordering the pooled stream. Nothing
        consumes this yet.
        """
        self._domain_ids = domain_ids

    def get_domain_ids(self) -> Optional[Sequence[int]]:
        return self._domain_ids

    def observe(self, **signals: Any) -> None:
        """Receive training-time feedback.

        The trainer calls this at each update boundary with whatever it has
        (``grad_norm``, ``global_step``, ``loss``, ...). The base implementation
        only records the latest values; this is the seam for an adaptive
        controller that tunes ``layers`` / ``folding_ratio`` / ``window_size``
        online from forgetting and optimizer-shock signals. No current subclass
        acts on it.
        """
        self._signals.update(signals)

    @property
    def signals(self) -> Dict[str, Any]:
        return dict(self._signals)

    # ------------------------------------------------------------------
    # Ordering
    # ------------------------------------------------------------------

    def _effective_ascending(self) -> bool:
        """Fold polarity into the sort direction.

        `ascending=True` means the curriculum advances along the axis, i.e. the
        end of training gets the samples `polarity` marks as belonging last.
        """
        if self.polarity == "higher_earlier":
            return not self.ascending
        return self.ascending

    def apply(self, positions: Sequence[int], scores: Optional[Sequence[float]]) -> List[int]:
        """Run the configured pattern over `positions`, honouring polarity and JIT."""
        params = dict(self.pattern_params)
        params.update(
            ascending=self._effective_ascending(),
            window_size=self.window_size,
            seed=self.seed,
        )
        return apply_pattern(self.pattern, positions, scores, **params)

    def describe(self) -> str:
        parts = [f"pattern={self.pattern}", f"polarity={self.polarity}", f"ascending={self.ascending}"]
        if self.window_size > 1:
            parts.append(f"jit_window={self.window_size}")
        for key in ("layers", "folding_layer", "zigzag_layer", "num_sections", "folding_ratio", "x_pct", "y_pct"):
            if self.pattern_params.get(key) is not None:
                parts.append(f"{key}={self.pattern_params[key]}")
        return ", ".join(parts)

    def log_schedule(
        self,
        order: Sequence[int],
        score_by_position: Optional[Dict[int, float]],
        num_buckets: int = 10,
        tag: str = "",
    ) -> None:
        """Log the realised score trajectory so the curriculum shape is visible.

        Prints the mean score of consecutive buckets of the ordered sequence,
        which is the discrete version of the score-index plots in the paper: a
        single ramp is CL, a repeating ramp is FO, a triangle wave is ZIG.
        """
        if not score_by_position or not len(order):
            return
        size = max(1, len(order) // max(1, num_buckets))
        means = []
        for start in range(0, len(order), size):
            bucket = [score_by_position[p] for p in order[start : start + size] if p in score_by_position]
            if bucket:
                means.append(sum(bucket) / len(bucket))
        if means:
            logger.info(
                f"[Dataflex][Reorder]{tag} score trajectory ({len(means)} buckets): "
                + " -> ".join(f"{m:.3f}" for m in means)
            )

    def warmup_indices(self, num_samples: int) -> List[int]:
        """Indices for the warmup phase, before the component takes over.

        The default is the head of the ordering, which is right whenever the
        ordering does not depend on the model. Subclasses that score with the
        model must override this, because at warmup the model has not trained
        yet and its scores carry no signal.
        """
        return self.next_indices(model=None, step_id=0, num_samples=num_samples)

    @abstractmethod
    def next_indices(self, model, step_id: int, num_samples: int, **kwargs) -> List[int]:
        """Return the next `num_samples` dataset indices to train on, in order.

        Args:
            model: the model being trained, for score providers that need it.
            step_id: current global step.
            num_samples: how many indices the trainer wants for this interval.
            **kwargs: trainer context (`current_update_times`, `update_times`, ...).

        Returns:
            An ordered list of dataset indices.
        """
        raise NotImplementedError
