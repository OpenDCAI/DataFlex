from typing import List, Optional, Sequence

import numpy as np
import torch.distributed as dist

from dataflex.core.registry import register_reorderer
from dataflex.utils.logging import logger

from .base_reorderer import Reorderer
from .score_provider import build_score_provider


@register_reorderer("dynamic")
class DynamicReorderer(Reorderer):
    """Re-score and re-order the remaining pool as training progresses.

    The static path fixes the whole curriculum before the first step, so the
    order reflects a judgement made by some other model at some other time. Here
    the score is recomputed from the model being trained, at every update
    interval, and the pattern is re-applied to whatever has not been consumed
    yet. A sample's place in the curriculum can therefore change as the model's
    view of it changes.

    Consume-once semantics are maintained explicitly: whatever is handed out is
    removed from the pool, so one pass over the data stays one pass. When the
    pool empties it refills (with a log line), so a run configured for more
    steps than one epoch keeps going.

    Cost: one forward pass over the pool per interval. Control it with
    `score_params.max_samples` (subsample the pool) and `reorder_every` (re-score
    every k-th interval instead of every one).
    """

    def __init__(
        self,
        pattern: str = "sorting",
        window_size: int = 0,
        polarity: str = "lower_is_better",
        ascending: bool = True,
        seed: int = 42,
        score_source: str = "model_loss",
        score_params: Optional[dict] = None,
        pattern_params: Optional[dict] = None,
        reorder_every: int = 1,
        consume_once: bool = True,
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
        self.score_source = score_source
        self.score_params = dict(score_params or {})
        self.reorder_every = max(1, int(reorder_every))
        self.consume_once = bool(consume_once)
        self.dataset = dataset
        self.accelerator = accelerator
        self.data_collator = data_collator

        self._provider = None
        self._pool: Optional[List[int]] = None
        self._pending: List[int] = []
        self._call_count = 0

        logger.info(
            f"[Dataflex][Reorder] DynamicReorderer({self.describe()}, score_source={score_source}, "
            f"reorder_every={self.reorder_every}, consume_once={self.consume_once})"
        )

    def _ensure_provider(self):
        if self._provider is None:
            self._provider = build_score_provider(
                self.score_source,
                dataset=self.dataset,
                accelerator=self.accelerator,
                data_collator=self.data_collator,
                expected_size=len(self.dataset) if self.dataset is not None else None,
                **self.score_params,
            )
            logger.info(f"[Dataflex][Reorder] score provider: {self._provider.describe()}")
        return self._provider

    def _ensure_pool(self) -> List[int]:
        if self._pool is None:
            pool = self.get_candidate_pool()
            if pool is None:
                if self.dataset is None:
                    raise ValueError("DynamicReorderer needs a dataset or an explicit candidate pool")
                pool = list(range(len(self.dataset)))
            self._pool = list(pool)
        return self._pool

    def _consume(self, served: Sequence[int]) -> None:
        """Remove served samples from the pool so one pass stays one pass."""
        if not self.consume_once or not served:
            return
        served_set = set(served)
        self._pool = [i for i in (self._pool or []) if i not in served_set]

    def _refill(self) -> None:
        pool = self.get_candidate_pool()
        if pool is None:
            pool = list(range(len(self.dataset)))
        self._pool = list(pool)
        logger.info(f"[Dataflex][Reorder] pool exhausted, refilled with {len(self._pool)} samples")

    def _broadcast(self, values: Optional[List[int]]) -> List[int]:
        """Agree on one ordering across ranks.

        Scoring is gathered on every rank, but float non-determinism could still
        let two ranks disagree on a tie, and they must consume identical index
        lists. Rank 0 decides.
        """
        if not (dist.is_available() and dist.is_initialized()):
            return values or []
        payload = [values]
        dist.broadcast_object_list(payload, src=0)
        return payload[0] or []

    def warmup_indices(self, num_samples: int) -> List[int]:
        """Random sample, not a scored ordering.

        Scoring here would rank samples by an untrained model's loss, which is
        mostly a function of sequence length and token frequency rather than
        anything about learning. This is the same reason every other DataFlex
        component treats warmup as a separate phase. Drawn without replacement
        and removed from the pool, so consume-once still holds.
        """
        pool = self._ensure_pool()
        rng = np.random.default_rng(self.seed)
        take = min(num_samples, len(pool))
        picked = [pool[i] for i in rng.choice(len(pool), size=take, replace=False)]

        if take < num_samples:
            logger.warning(
                f"[Dataflex][Reorder] warmup wanted {num_samples} samples but the pool only has {take}"
            )
        self._consume(picked)
        logger.info(
            f"[Dataflex][Reorder] warmup draws {len(picked)} random samples "
            f"(the model has not trained yet, so its scores carry no signal); "
            f"{len(self._pool)} left in the pool"
        )
        return picked

    def next_indices(self, model, step_id: int, num_samples: int, **kwargs) -> List[int]:
        self._ensure_pool()
        self._call_count += 1

        # Serve from the already-ordered remainder when re-scoring is throttled.
        should_rescore = (self._call_count - 1) % self.reorder_every == 0
        if not should_rescore and len(self._pending) >= num_samples:
            out, self._pending = self._pending[:num_samples], self._pending[num_samples:]
            self._consume(out)
            logger.info(
                f"[Dataflex][Reorder] step {step_id}: reusing the previous ordering "
                f"({len(self._pending)} samples still pending, {len(self._pool)} left in the pool, "
                f"reorder_every={self.reorder_every})"
            )
            return out

        out: List[int] = []
        while len(out) < num_samples:
            if not self._pool:
                if not self.consume_once:
                    break
                self._refill()
                if not self._pool:
                    break

            provider = self._ensure_provider()
            scores = provider.scores_for(self._pool, model=model, step_id=step_id)

            is_main = self.accelerator is None or self.accelerator.is_main_process
            ordered = self.apply(self._pool, scores) if is_main else None
            ordered = self._broadcast(ordered) if (self.accelerator is not None) else (ordered or [])

            if is_main:
                self.log_schedule(ordered, dict(zip(self._pool, scores)), tag=f"[step {step_id}]")
                finite = [s for s in scores if np.isfinite(s)]
                if finite:
                    logger.info(
                        f"[Dataflex][Reorder] step {step_id}: scored {len(finite)} samples "
                        f"(min={min(finite):.4f}, mean={sum(finite) / len(finite):.4f}, max={max(finite):.4f})"
                    )

            take = min(num_samples - len(out), len(ordered))
            out.extend(ordered[:take])
            self._pending = ordered[take:]

            if not self.consume_once:
                break
            # Only the served samples leave the pool; the rest go back and are
            # re-scored next interval, which is the point of doing this online.
            self._consume(ordered[:take])

        logger.info(
            f"[Dataflex][Reorder] step {step_id}: serving {len(out)} samples, "
            f"{len(self._pool)} left in the pool"
        )
        return out

    def reorder(self, positions: Sequence[int], scores: Sequence[float], step_id: int = 0, **ctx) -> List[int]:
        """Stateless ordering of an arbitrary subset. Used for composition."""
        return self.apply(positions, scores)
