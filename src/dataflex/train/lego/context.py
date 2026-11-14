"""Everything a stage may need, in one object.

Each family currently reaches for a different subset of an untyped `**kwargs`:
`less` wants `optimizer_state`, `delta_loss` wants `current_update_times` and
`update_times`, the mixers want `batch` / `domain_ids` / `data_collator`. When
strategies run one at a time the trainer can hand each its own bespoke dict, but
composing them needs a single object that satisfies all of them at once — and
that any future stage can read from without changing a signature.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


__all__ = ["PlanContext"]


@dataclass
class PlanContext:
    """Read-mostly view of the training state at one planning boundary."""

    # ---- what we are planning over ----
    dataset: Any = None
    dataset_size: int = 0
    #: Per-sample domain label aligned with `dataset`, or None for a single-domain
    #: run. This is what lets a mixer work by index arithmetic instead of needing
    #: one dataset object per source.
    domain_ids: Optional[np.ndarray] = None
    domain_names: Optional[List[str]] = None

    # ---- training state ----
    model: Any = None
    accelerator: Any = None
    data_collator: Any = None
    step_id: int = 0
    #: How many samples the trainer needs for the upcoming interval, i.e.
    #: global_batch_size * update_step. The budget contract says the finished
    #: plan must contain exactly this many.
    interval_size: int = 0
    current_update_times: int = 1
    update_times: int = 1

    # ---- optimiser state, for gradient-based components ----
    optimizer_state: Any = None
    scheduler_state: Any = None

    # ---- the most recent training batch, for components that need it ----
    batch: Any = None

    # ---- shared services ----
    scoreboard: Any = None

    #: Free-form signals recorded by the trainer (grad_norm, learning_rate, ...).
    signals: Dict[str, Any] = field(default_factory=dict)

    #: Cached `{domain: indices}` for the pool most recently asked about. A quota
    #: draw asks once per domain, and rebuilding the array and mask each time
    #: walked the pool K times for a K-domain run; grouping once makes it one
    #: pass regardless of K.
    _group_cache: Optional[Dict[int, np.ndarray]] = None
    _group_cache_key: Optional[tuple] = None
    #: The pool the cache was built for, held only to keep it alive. The key uses
    #: `id()`, and CPython reuses the address of a freed object, so without this
    #: reference a later pool of the same length could collide with the entry and
    #: silently receive another pool's grouping.
    _group_cache_pool: Optional[Any] = None

    #: Keep the selector's tokenizer available without exposing the full Trainer.
    tokenizer: Any = None

    @classmethod
    def from_trainer(cls, trainer, *, model=None, batch=None, **overrides) -> "PlanContext":
        """Snapshot the runtime shared by standalone and LEGO components.

        Components read step_id and accelerator.device/is_main_process. This
        is not a Trainer replacement: backward/optimizer operations still
        belong to the Trainer. Explicit overrides describe planning boundaries.
        """
        dataset = trainer.train_dataset
        values = dict(
            dataset=dataset,
            dataset_size=len(dataset) if dataset is not None else 0,
            domain_ids=getattr(trainer, "domain_ids", None),
            domain_names=getattr(trainer, "domain_names", None),
            model=model if model is not None else trainer.model,
            accelerator=trainer.accelerator,
            data_collator=trainer.data_collator,
            tokenizer=getattr(trainer, "processing_class", None),
            step_id=trainer.state.global_step,
            update_times=getattr(getattr(trainer, "finetuning_args", None), "update_times", 1),
            batch=batch,
        )
        values.update(overrides)
        return cls(**values)

    def domain_pool(self, domain: int, pool: Optional[Sequence[int]] = None) -> np.ndarray:
        """Indices belonging to `domain`, optionally restricted to `pool`.

        The primitive a mixer needs: allocating a quota per domain is just
        drawing from these groups. Returns an array rather than a list because
        the caller only slices and samples from it — converting would rebuild
        every index as a Python object, which across all domains is the whole
        pool and dominates the cost of the draw.
        """
        if self.domain_ids is None:
            if pool is None:
                return np.arange(self.dataset_size, dtype=np.int64)
            return np.asarray(pool, dtype=np.int64)
        return self._grouped(pool).get(int(domain), np.empty(0, dtype=np.int64))

    def _grouped(self, pool: Optional[Sequence[int]]) -> Dict[int, np.ndarray]:
        # Key on the pool *before* touching it. Converting a Python list to an
        # array is itself O(N), so doing it ahead of the cache check would keep
        # the per-domain cost linear in the pool and defeat the point.
        key = (id(pool), self.dataset_size if pool is None else len(pool))
        if self._group_cache_key == key and self._group_cache is not None:
            return self._group_cache

        pool_arr = (
            np.arange(self.dataset_size, dtype=np.int64)
            if pool is None
            else np.asarray(pool, dtype=np.int64)
        )
        labels = np.asarray(self.domain_ids)[pool_arr]
        order = np.argsort(labels, kind="stable")
        sorted_labels = labels[order]
        sorted_pool = pool_arr[order]
        boundaries = np.flatnonzero(np.diff(sorted_labels)) + 1
        groups = {
            int(chunk[0]): sorted_pool[start:stop]
            for chunk, start, stop in zip(
                np.split(sorted_labels, boundaries),
                np.concatenate(([0], boundaries)),
                np.concatenate((boundaries, [len(sorted_labels)])),
            )
            if len(chunk)
        }
        self._group_cache = groups
        self._group_cache_key = key
        self._group_cache_pool = pool
        return groups

    @property
    def num_domains(self) -> int:
        if self.domain_names:
            return len(self.domain_names)
        if self.domain_ids is not None and len(self.domain_ids):
            return int(np.max(self.domain_ids)) + 1
        return 1

    def as_component_kwargs(self) -> Dict[str, Any]:
        """The `**kwargs` the existing families expect.

        Lets the stage adapters call `selector.select(...)` / `mixer.mix(...)`
        unchanged, so no component has to be rewritten to be composable.
        """
        return dict(
            optimizer_state=self.optimizer_state,
            scheduler_state=self.scheduler_state,
            current_update_times=self.current_update_times,
            update_times=self.update_times,
            batch=self.batch,
            domain_ids=self.domain_ids,
            data_collator=self.data_collator,
            dataset=self.dataset,
            tokenizer=self.tokenizer,
        )
