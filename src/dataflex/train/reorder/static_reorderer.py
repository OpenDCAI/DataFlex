from typing import List, Optional, Sequence

from dataflex.core.registry import register_reorderer
from dataflex.utils.logging import logger

from .base_reorderer import Reorderer
from .score_provider import build_score_provider


@register_reorderer("static")
class StaticReorderer(Reorderer):
    """Order the dataset once, from scores that do not depend on the model.

    This is the faithful reproduction of the paper: a single global permutation
    computed up front and then consumed in order. It has two entry points, which
    differ only in what a "position" means:

    - ``order_rows(scores)`` is called by the dataset loader before tokenization
      and permutes *raw* rows. This is the preferred route for a score that
      lives in the raw JSONL, because the score column is deleted during
      preprocessing and mapping it onto preprocessed indices afterwards is not
      safe (rows can be dropped or packed together).
    - ``next_indices(...)`` is called by the trainer and permutes *dataset*
      indices, streaming the permutation out in chunks. Used when the scores
      come from a file that is already aligned to the preprocessed dataset, or
      from a selector cache.

    Either way the permutation is computed exactly once; the trainer only walks
    a cursor along it.
    """

    def __init__(
        self,
        pattern: str = "sorting",
        window_size: int = 0,
        polarity: str = "higher_is_better",
        ascending: bool = True,
        seed: int = 42,
        score_source: str = "precomputed",
        score_params: Optional[dict] = None,
        pattern_params: Optional[dict] = None,
        apply_at: str = "raw",
        dataset=None,
        accelerator=None,
        data_collator=None,
        **kwargs,
    ):
        super().__init__(
            pattern=pattern,
            window_size=window_size,
            polarity=polarity,
            ascending=ascending,
            seed=seed,
            pattern_params=pattern_params,
            **kwargs,
        )
        if apply_at not in ("raw", "index"):
            raise ValueError(f"apply_at must be 'raw' or 'index', got '{apply_at}'")

        self.apply_at = apply_at
        self.score_source = score_source
        self.score_params = dict(score_params or {})
        self.dataset = dataset
        self.accelerator = accelerator
        self.data_collator = data_collator

        self._order: Optional[List[int]] = None
        self._cursor = 0

        logger.info(
            f"[Dataflex][Reorder] StaticReorderer({self.describe()}, "
            f"score_source={score_source}, apply_at={apply_at})"
        )

    # ------------------------------------------------------------------
    # Entry point A: permute raw rows before preprocessing
    # ------------------------------------------------------------------

    def order_rows(self, scores: Sequence[float]) -> List[int]:
        """Return a permutation of raw dataset rows given their scores."""
        positions = list(range(len(scores)))
        order = self.apply(positions, scores)
        self.log_schedule(order, dict(zip(positions, scores)), tag="[raw]")
        logger.info(f"[Dataflex][Reorder] permuted {len(order)} raw rows with pattern '{self.pattern}'")
        return order

    # ------------------------------------------------------------------
    # Entry point B: permute dataset indices, stream in chunks
    # ------------------------------------------------------------------

    def _ensure_order(self, model=None, step_id: int = 0) -> List[int]:
        if self._order is not None:
            return self._order

        pool = self.get_candidate_pool()
        if pool is None:
            if self.dataset is None:
                raise ValueError("StaticReorderer needs a dataset or an explicit candidate pool")
            pool = list(range(len(self.dataset)))

        if self.apply_at == "raw":
            # The loader already permuted the raw rows, so the dataset is stored
            # in curriculum order and reordering again would undo it.
            self._order = list(pool)
            logger.info(
                f"[Dataflex][Reorder] ordering was applied at the raw stage; "
                f"serving {len(self._order)} samples sequentially"
            )
            return self._order

        provider = build_score_provider(
            self.score_source,
            dataset=self.dataset,
            accelerator=self.accelerator,
            data_collator=self.data_collator,
            expected_size=len(self.dataset) if self.dataset is not None else None,
            **self.score_params,
        )
        if provider.is_dynamic:
            logger.warning(
                f"[Dataflex][Reorder] score source '{self.score_source}' depends on the model but "
                f"StaticReorderer scores only once, at the first update. Use the 'dynamic' reorderer "
                f"to re-score as training progresses."
            )

        scores = provider.scores_for(pool, model=model, step_id=step_id)
        self._order = self.apply(pool, scores)
        self.log_schedule(self._order, dict(zip(pool, scores)))
        logger.info(
            f"[Dataflex][Reorder] built a global order over {len(self._order)} samples "
            f"using {provider.describe()}"
        )
        return self._order

    def next_indices(self, model, step_id: int, num_samples: int, **kwargs) -> List[int]:
        """Hand out the next `num_samples` of the global permutation.

        Wraps around when the permutation is exhausted, so a run configured for
        more steps than one pass keeps going instead of starving.
        """
        order = self._ensure_order(model=model, step_id=step_id)
        if not order:
            return []

        out: List[int] = []
        while len(out) < num_samples:
            remaining = len(order) - self._cursor
            if remaining <= 0:
                self._cursor = 0
                remaining = len(order)
                logger.info("[Dataflex][Reorder] permutation exhausted, restarting from the beginning")
            take = min(num_samples - len(out), remaining)
            out.extend(order[self._cursor : self._cursor + take])
            self._cursor += take

        progress = 100.0 * self._cursor / max(1, len(order))
        logger.info(
            f"[Dataflex][Reorder] step {step_id}: serving {len(out)} samples, "
            f"cursor at {self._cursor}/{len(order)} ({progress:.1f}% of one pass)"
        )
        return out

    def reorder(self, positions: Sequence[int], scores: Sequence[float], step_id: int = 0, **ctx) -> List[int]:
        """Stateless ordering of an arbitrary subset. Used for composition."""
        return self.apply(positions, scores)
