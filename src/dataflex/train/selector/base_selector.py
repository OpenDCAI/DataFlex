from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Sequence
import torch
from torch import distributed as dist

from dataflex.utils.logging import logger


class Selector(ABC):
    def __init__(self, dataset, accelerator, data_collator, cache_dir):
        self.dataset = dataset
        self.accelerator = accelerator
        self.data_collator = data_collator
        self.cache_dir = cache_dir
        self.seed = 42

        # ---- composition hooks (see set_* below) ----
        self._candidate_pool: Optional[List[int]] = None
        self._score_source: Optional[Callable[[Sequence[int]], Sequence[float]]] = None
        self._score_sink = None

    def set_score_sink(self, fn) -> None:
        """Optional LEGO consumer of the selected rows' actual scoring signal."""
        self._score_sink = fn

    def publish_scores(self, indices, scores, step_id, *, metric, ascending, reason=None):
        if self._score_sink is not None:
            self._score_sink(dict(indices=list(indices), scores=list(scores), step=int(step_id),
                                  metric=metric, ascending=ascending, reason=reason))

    # ------------------------------------------------------------------
    # Composition hooks
    #
    # Mirrors the reorder base class so both families expose the same seams.
    # Both are no-ops by default, so a selector used on its own behaves exactly
    # as if they were absent.
    # ------------------------------------------------------------------

    def set_candidate_pool(self, indices: Optional[Sequence[int]]) -> None:
        """Restrict this selector to a subset of the dataset.

        When several strategies run in one pipeline an earlier stage may already
        have discarded most of the data. Without this the selector would still
        score all N samples and then choose from a pool that no longer exists.

        Anything a subclass derived from the previous pool is invalidated, since
        an upstream stage that re-filters every boundary would otherwise be
        ignored after the first call.
        """
        new_pool = list(indices) if indices is not None else None
        changed = new_pool != self._candidate_pool
        self._candidate_pool = new_pool
        if changed:
            self._on_candidate_pool_changed()

    def get_candidate_pool(self) -> Optional[List[int]]:
        return list(self._candidate_pool) if self._candidate_pool is not None else None

    def _on_candidate_pool_changed(self) -> None:
        """Drop state derived from the old candidate pool. Overridden by subclasses."""
        return None

    def candidate_positions(self) -> List[int]:
        """The positions this selector should be scoring and choosing from."""
        if self._candidate_pool is not None:
            return list(self._candidate_pool)
        return list(range(len(self.dataset)))

    def set_score_source(self, fn: Optional[Callable[[Sequence[int]], Sequence[float]]]) -> None:
        """Take per-sample scores from somewhere else instead of computing them.

        A loss-based selector and a dynamic reorder stage want the same signal,
        so running both means two forward passes over the pool for one piece of
        information. Injecting a shared source collapses them into one.

        `fn(positions) -> scores`, one score per position, same order.
        """
        self._score_source = fn

    def has_score_source(self) -> bool:
        return self._score_source is not None

    def _scores_for(self, positions: Sequence[int], model, step_id: int) -> List[float]:
        """Per-sample scores, from the shared source when one is attached.

        Subclasses implement `_compute_losses` for the standalone case; this is
        the single place that decides whether to call it.
        """
        positions = list(positions)
        if not positions:
            return []
        if self._score_source is not None:
            scores = list(self._score_source(positions))
            if len(scores) != len(positions):
                raise ValueError(
                    f"score source returned {len(scores)} values for {len(positions)} positions"
                )
            logger.info(f"[Dataflex] {type(self).__name__} took {len(scores)} scores from the shared source")
            return scores
        return list(self._compute_losses(positions, model, step_id))

    def _compute_losses(self, positions: Sequence[int], model, step_id: int) -> List[float]:
        """Standalone scoring. Subclasses that score must implement this."""
        raise NotImplementedError(
            f"{type(self).__name__} has no _compute_losses; it cannot score without a score source"
        )

    # ------------------------------------------------------------------

    def warmup(self, num_samples: int, replacement: bool) -> List[int]:
        if self.accelerator.is_main_process:
            dataset_size = len(self.dataset)
            gen = torch.Generator()
            gen.manual_seed(self.seed)

            if replacement:
                full_indices = torch.randint(
                    low=0, high=dataset_size, size=(num_samples,), generator=gen
                ).tolist()
            else:
                if num_samples > dataset_size:
                    raise ValueError(
                        f"Cannot sample {num_samples} without replacement from {dataset_size} samples"
                    )
                full_indices = torch.randperm(dataset_size, generator=gen)[:num_samples].tolist()
        else:
            full_indices = None

        obj = [full_indices]
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(obj, src=0)
            full_indices = obj[0]
        else:
            full_indices = full_indices or []

        return full_indices

    @abstractmethod
    def select(self, model, step_id: int, num_samples: int, **kwargs):
        """
        Select samples from the dataset for the model in 'step_id'.

        Args:
            model: The model object used in the selection process.
            step_id (int): The ID of the current training step or stage.
            num_samples (int): The number of samples to select.
            **kwargs: Additional keyword arguments, allowing for flexible expansion by subclasses.

        Returns:
            List[int]: A list of the selected sample indices.
        """
        pass
