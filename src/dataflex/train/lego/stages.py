"""Adapters that let existing components act as pipeline stages.

Nothing here reimplements an algorithm. Each adapter wraps a component built
from the ordinary `selectors:` / `mixers:` / `reorders:` / `weighters:` bucket
and translates between that family's own signature and the two composition
sockets:

    IndexStage.apply(plan, ctx) -> plan      transforms the index list
    LossStage.weights(losses, ctx) -> tensor per-sample multipliers

That is what makes the bricks interchangeable: adding a new selector to DataFlex
automatically makes it usable as a pipeline stage, with no work here.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist

from dataflex.utils.logging import logger

from .context import PlanContext
from .plan import DataPlan


#: Index-space stages run in this order regardless of how the YAML lists them.
#: The order is forced by the algebra, not preference: proportions should be
#: computed over survivors, a chunk must be known before it can be arranged, and
#: ordering has to come last or a later stage would destroy it.
INDEX_ORDER = ("selector", "mixer", "reorder")
LOSS_FAMILIES = ("weighter",)


class Stage(ABC):
    """Common scheduling behaviour for both socket types."""

    family: str = ""
    #: True when this stage reduces the pool to exactly `ctx.interval_size`.
    #: Exactly one index stage in a pipeline must be a taker.
    takes_budget: bool = False

    def __init__(
        self,
        component,
        name: str,
        warmup_step: int = 0,
        every: int = 1,
        score_max_age: int = 0,
        **_ignored,
    ):
        self.component = component
        self.name = name
        self.warmup_step = int(warmup_step or 0)
        self.every = max(1, int(every or 1))
        #: Kept for explicit validation of older configurations. Exact LEGO
        #: loss scoring accepts only zero; component refresh clocks are separate.
        self.score_max_age = int(score_max_age or 0)
        self._last_run_step: Optional[int] = None
        self._cached: Optional[Any] = None

    def is_active(self, step_id: int) -> bool:
        """Whether this stage participates at all yet."""
        return step_id >= self.warmup_step

    def should_run(self, step_id: int) -> bool:
        """Whether this stage wants to recompute at `step_id`."""
        if not self.is_active(step_id):
            return False
        if self._last_run_step is None:
            return True
        return (step_id - self._last_run_step) >= self.every

    def mark_run(self, step_id: int) -> None:
        self._last_run_step = int(step_id)

    def observe(self, **signals) -> None:
        """Forward training feedback to the component if it accepts it."""
        fn = getattr(self.component, "observe", None)
        if callable(fn):
            try:
                fn(**signals)
            except Exception as exc:
                logger.warning(f"[Dataflex][Lego] {self.name}.observe failed: {exc}")

    def board_score_fn(self, ctx: PlanContext):
        """Current-step answer loss through the common scoring backend."""
        if ctx.scoreboard is None:
            return None
        self.validate_score_options()
        consumer = getattr(self, "score_consumer", f"{self.family}:{self.name}")

        def fn(positions, model=None, step_id=None, metric="loss_full"):
            return ctx.scoreboard.loss_scores(
                model if model is not None else ctx.model,
                positions, ctx.step_id if step_id is None else step_id,
                consumer=consumer, metric=metric,
            ).tolist()

        return fn

    def validate_score_options(self):
        params = getattr(self.component, "score_params", {}) or {}
        if self.score_max_age != 0 or params.get("max_samples") is not None:
            raise ValueError("LEGO loss scoring requires score_max_age: 0 and no max_samples approximation")

    def describe(self) -> str:
        bits = [f"{self.family}={self.name}"]
        if self.warmup_step:
            bits.append(f"warmup={self.warmup_step}")
        if self.every > 1:
            bits.append(f"every={self.every}")
        if self.takes_budget:
            bits.append("taker")
        return " ".join(bits)


class IndexStage(Stage, ABC):
    @abstractmethod
    def apply(self, plan: DataPlan, ctx: PlanContext) -> DataPlan:
        raise NotImplementedError


class LossStage(Stage, ABC):
    @abstractmethod
    def weights(self, losses: torch.Tensor, ctx: PlanContext, model=None, inputs=None) -> Optional[torch.Tensor]:
        """Per-sample multipliers such that `(w * losses).mean()` is the intended loss."""
        raise NotImplementedError


def _broadcast_indices(indices, accelerator) -> List[int]:
    """Make every rank agree on one index list.

    Components already broadcast internally, but a stage may also derive indices
    locally (quota draws, uniform takes). Ranks must consume identical lists or
    they train on different data, so rank 0 decides.

    Accepts a list or an ndarray; note that `indices or []` would raise on an
    array, so emptiness is tested explicitly.
    """
    as_list = [] if indices is None else (
        indices.tolist() if isinstance(indices, np.ndarray) else list(indices)
    )
    if not (dist.is_available() and dist.is_initialized()):
        return as_list
    payload = [as_list]
    dist.broadcast_object_list(payload, src=0)
    return list(payload[0] or [])


# ----------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------


class SelectStage(IndexStage):
    """Filter the candidate pool, or draw the interval outright.

    Two modes, because "selection" means different things depending on whether
    anything runs after it:

    - ``filter`` requests `keep_ratio * N` samples and leaves the interval draw
      to a later stage. ADAPT-INF retains the actual author-CCS result, whose
      size can differ after outlier filtering and cluster-budget rounding.
    - ``draw`` returns exactly `interval_size` samples. In that case this stage
      owns the interval budget, including when the selector is ADAPT-INF.
    """

    family = "selector"

    def __init__(self, component, name, mode: str = "filter", keep_ratio: float = 0.5, **kw):
        super().__init__(component, name, **kw)
        if mode not in ("filter", "draw"):
            raise ValueError(f"select stage mode must be 'filter' or 'draw', got '{mode}'")
        self.mode = mode
        self.keep_ratio = float(keep_ratio)
        if self.mode == "filter" and not (0.0 < self.keep_ratio <= 1.0):
            raise ValueError(f"keep_ratio must be in (0, 1], got {self.keep_ratio}")
        self.takes_budget = mode == "draw"
        if mode == "filter" and getattr(component, "replacement", False):
            raise ValueError("LEGO filter selection requires replacement: false")

    def _share_scores(self, plan: DataPlan, ctx: PlanContext) -> None:
        """Point the selector at the current pool and at the shared board.

        Both hooks are no-ops on a selector that does not score (`random`,
        `near`, `tsds`, `custom`), and the board source is ignored by anything
        whose signal is not per-sample loss (`less` and `nice` use projected
        gradients).
        """
        setter = getattr(self.component, "set_candidate_pool", None)
        if callable(setter):
            setter(plan.as_list())

        inject = getattr(self.component, "set_score_source", None)
        if not callable(inject):
            return
        fn = self.board_score_fn(ctx)
        if fn is None:
            return
        inject(lambda positions: fn(positions, model=ctx.model, step_id=ctx.step_id))

    def apply(self, plan: DataPlan, ctx: PlanContext) -> DataPlan:
        self._share_scores(plan, ctx)
        published = []
        if getattr(self, "publish_selection_scores", False):
            self.component.set_score_sink(published.append)

        if self.mode == "draw":
            num = ctx.interval_size
        else:
            num = max(1, int(len(plan) * self.keep_ratio))

        kwargs = ctx.as_component_kwargs()
        # Components see the original row space, without LEGO's attached IDs.
        kwargs["dataset"] = getattr(self.component, "dataset", ctx.dataset)
        kwargs["data_collator"] = getattr(self.component, "data_collator", ctx.data_collator)
        try:
            indices = self.component.select(
                model=ctx.model,
                step_id=ctx.step_id,
                num_samples=num,
                **kwargs,
            )
        finally:
            if getattr(self, "publish_selection_scores", False):
                self.component.set_score_sink(None)
        indices = list(indices or [])

        if getattr(self, "publish_selection_scores", False):
            payload = published[-1] if published else None
            if self.name == "random":
                payload = dict(indices=indices, scores=None, metric=None,
                               ascending=None, step=ctx.step_id, reason="random")
            # DeltaLoss's selection signal is produced on rank 0, just like
            # its chosen IDs. Broadcast both, not independently recomputed deltas.
            obj = [payload]
            if dist.is_available() and dist.is_initialized():
                dist.broadcast_object_list(obj, src=0)
            payload = obj[0]
            if payload is None or payload["indices"] != indices:
                raise ValueError("Selector did not publish scores aligned with its selected IDs")
            ctx.scoreboard.put_selection_scores(self.score_consumer, payload)

        if self.name in {"loss", "delta_loss", "adapt_inf", "random"}:
            pool = set(plan.as_list())
            # Author CCS trims outliers and rounds per-cluster budgets, so its
            # result need not equal the requested count. A filter passes that
            # actual pool downstream; it does not own the training interval's
            # exact budget. Never fill or truncate CCS to force an exact count.
            variable_filter = self.name == "adapt_inf" and self.mode == "filter"
            valid_count = bool(indices) if variable_filter else len(indices) == num
            if not valid_count or any(i not in pool for i in indices):
                raise ValueError(f"{self.name} returned invalid IDs or a different selection budget")
            if self.mode == "filter" and len(set(indices)) != len(indices):
                raise ValueError(f"{self.name} returned duplicate IDs in filter mode")
            if variable_filter and len(indices) != num:
                logger.info(
                    f"[Dataflex][Lego] adapt_inf filter requested {num}, retaining "
                    f"{len(indices)} samples returned by author CCS"
                )

        if self.mode == "filter":
            # A selector may return duplicates (several sample with replacement).
            # As a pool filter that is meaningless, so collapse them while
            # keeping first-seen order for determinism.
            seen = set()
            deduped = [i for i in indices if not (i in seen or seen.add(i))]
            if len(deduped) != len(indices):
                logger.info(
                    f"[Dataflex][Lego] {self.name} returned {len(indices) - len(deduped)} duplicate "
                    f"indices; collapsed for pool filtering"
                )
            indices = deduped

        logger.info(
            f"[Dataflex][Lego] select({self.name}, mode={self.mode}): "
            f"{len(plan)} -> {len(indices)} samples"
        )
        return plan.replace_indices(indices, stage=f"select:{self.name}")


# ----------------------------------------------------------------------
# Mixture
# ----------------------------------------------------------------------


class MixStage(IndexStage):
    """Turn a mixer's domain proportions into an actual per-domain quota.

    The mixer itself is unchanged: it still returns a probability vector over
    domains. What changes is how that vector is realised. `MixedProportionManager`
    materialises it by sampling *with replacement* from per-source datasets and
    then shuffling, both of which are incompatible with composition — the first
    breaks one-pass coverage, the second erases any ordering. Here the same
    vector becomes a draw without replacement from the domain groups of the
    current pool, and the order is left entirely to a later reorder stage.

    A mixer that reweights instead of resampling (DoReMi returns uniform
    proportions and applies its weights in the loss) therefore passes the pool
    through untouched, which is correct: its effect lands in loss space.
    """

    family = "mixer"
    takes_budget = True

    def __init__(self, component, name, allow_replacement: bool = False, **kw):
        super().__init__(component, name, **kw)
        self.allow_replacement = bool(allow_replacement)
        self._last_probs: Optional[np.ndarray] = None
        #: Separate clock for `update_from_batch`, so driving the loss-space
        #: update from the training step does not consume the index-space
        #: schedule that `should_run` reports to the pipeline.
        self._last_batch_update: Optional[int] = None

    @property
    def reweights_in_loss_space(self) -> bool:
        """True for a mixer whose decision is a loss coefficient, not a quota.

        DoReMi is the case: `mix()` returns uniform proportions and the real
        output is `get_current_doremi_weights()`. It also cannot run at a
        planning boundary at all, because updating those weights means comparing
        per-token losses against a reference model over a *live batch*, and a
        boundary has no batch to give it. So the update is driven from the
        training step and index space is left untouched here, which is exactly
        what its uniform proportions would have done anyway.
        """
        return callable(getattr(self.component, "get_current_doremi_weights", None))

    def due_for_batch_update(self, step_id: int) -> bool:
        """Whether the loss-space update should run at `step_id`.

        Same `warmup_step` / `every` semantics as index space, on its own clock.
        """
        if not self.is_active(step_id):
            return False
        if self._last_batch_update is None:
            return True
        return (step_id - self._last_batch_update) >= self.every

    def update_from_batch(self, model, step_id: int, batch, domain_ids) -> bool:
        """Let a reweighting mixer update itself from a live training batch.

        Mirrors what `MixTrainer` does inside its own loop. Note that DoReMi
        wants the *batch's* domain labels, not the dataset-wide array that
        `PlanContext` carries, which is why this takes them explicitly.
        """
        if not self.reweights_in_loss_space:
            return False
        if batch is None or domain_ids is None:
            return False
        if not self.due_for_batch_update(step_id):
            return False
        try:
            self.component.mix(model=model, step_id=step_id, batch=batch, domain_ids=domain_ids)
        except Exception as exc:
            logger.warning(f"[Dataflex][Lego] mix({self.name}) batch update failed: {exc}")
            return False
        self._last_batch_update = int(step_id)
        return True

    def apply(self, plan: DataPlan, ctx: PlanContext) -> DataPlan:
        if self.reweights_in_loss_space:
            logger.info(
                f"[Dataflex][Lego] mix({self.name}): reweights by domain, so the pool passes "
                f"through unchanged ({len(plan)} samples); its update runs from the training step"
            )
            return plan.replace_indices(plan.indices, stage=f"mix:{self.name}:loss-space")

        probs = self.component.mix(
            model=ctx.model,
            step_id=ctx.step_id,
            **ctx.as_component_kwargs(),
        )
        probs = np.asarray(probs, dtype=np.float64).reshape(-1)
        if probs.size == 0 or not np.all(np.isfinite(probs)) or probs.sum() <= 0:
            logger.warning(f"[Dataflex][Lego] {self.name} returned unusable proportions; keeping uniform")
            probs = np.ones(max(1, ctx.num_domains), dtype=np.float64)
        probs = probs / probs.sum()
        self._last_probs = probs

        if ctx.domain_ids is None:
            logger.warning(
                f"[Dataflex][Lego] {self.name} has no domain labels to act on; "
                f"drawing the interval uniformly instead"
            )
            indices = self._uniform_draw(plan.as_list(), ctx)
            return plan.replace_indices(indices, stage=f"mix:{self.name}", mix_probs=probs.tolist())

        indices = self._quota_draw(plan.as_list(), probs, ctx)
        names = ctx.domain_names or [str(i) for i in range(len(probs))]
        logger.info(
            f"[Dataflex][Lego] mix({self.name}): target "
            + ", ".join(f"{n}={p:.3f}" for n, p in zip(names, probs))
        )
        return plan.replace_indices(indices, stage=f"mix:{self.name}", mix_probs=probs.tolist())

    def _target_size(self, pool: List[int], probs: np.ndarray, ctx: PlanContext) -> int:
        """How many samples to hand downstream.

        As the budget taker (nothing runs after) the answer is one interval. As
        an intermediate stage it reshapes the pool for a downstream budget
        taker that orders a larger pool before choosing its interval.

        So reshape instead: keep the pool as large as the target proportions
        allow without drawing any sample twice. The binding constraint is the
        scarcest domain relative to its share, `min_d(|group_d| / p_d)`.
        """
        if self.takes_budget:
            return ctx.interval_size

        limits = []
        for domain, p in enumerate(probs):
            if p <= 0:
                continue
            size = len(ctx.domain_pool(domain, pool))
            limits.append((size / p, domain))

        if not limits:
            return min(len(pool), max(ctx.interval_size, 1))

        cap, binding = min(limits)
        target = int(min(len(pool), np.floor(cap)))
        # Never hand downstream less than one interval, even if that means
        # over-representing a scarce domain.
        target = max(target, min(ctx.interval_size, len(pool)))

        if target < len(pool):
            names = ctx.domain_names or [str(i) for i in range(len(probs))]
            logger.info(
                f"[Dataflex][Lego] mix({self.name}): reshaping the pool to {target} of {len(pool)} "
                f"samples; domain '{names[binding]}' is the binding constraint at these proportions"
            )
        return target

    def _quota_draw(self, pool: List[int], probs: np.ndarray, ctx: PlanContext) -> List[int]:
        budget = self._target_size(pool, probs, ctx)
        rng = np.random.default_rng(ctx.step_id + 12345)

        # Largest-remainder allocation, so the quotas sum to exactly `budget`
        # rather than drifting by a few samples through repeated flooring.
        raw = probs * budget
        quota = np.floor(raw).astype(np.int64)
        remainder = budget - int(quota.sum())
        if remainder > 0:
            order = np.argsort(-(raw - quota))
            quota[order[:remainder]] += 1

        chunks: List[np.ndarray] = []
        shortfall = 0
        for domain in range(len(quota)):
            want = int(quota[domain])
            if want <= 0:
                continue
            group = ctx.domain_pool(domain, pool)
            if len(group) == 0:
                shortfall += want
                continue
            if want <= len(group):
                take = rng.choice(len(group), size=want, replace=False)
            elif self.allow_replacement:
                take = rng.choice(len(group), size=want, replace=True)
            else:
                # Honour one-pass semantics: take what exists and make the
                # difference up from other domains rather than repeating samples.
                take = rng.permutation(len(group))
                shortfall += want - len(group)
            chunks.append(group[take])

        picked = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)

        if shortfall > 0:
            pool_arr = np.asarray(pool, dtype=np.int64)
            remaining = pool_arr[~np.isin(pool_arr, picked)]
            if len(remaining):
                extra = rng.choice(len(remaining), size=min(shortfall, len(remaining)), replace=False)
                picked = np.concatenate([picked, remaining[extra]])
            logger.info(
                f"[Dataflex][Lego] mix({self.name}): {shortfall} samples could not be met from their "
                f"domain quota (domain exhausted); backfilled from the rest of the pool"
            )
        return _broadcast_indices(picked, ctx.accelerator)

    def _uniform_draw(self, pool: List[int], ctx: PlanContext) -> List[int]:
        # Without domain labels there is nothing to reallocate, so as an
        # intermediate stage this is a pass-through rather than a shrink.
        if not self.takes_budget:
            return list(pool)
        rng = np.random.default_rng(ctx.step_id + 12345)
        n = min(ctx.interval_size, len(pool))
        take = rng.choice(len(pool), size=n, replace=False)
        return _broadcast_indices([pool[i] for i in np.asarray(take).tolist()], ctx.accelerator)

    def domain_weights(self) -> Optional[np.ndarray]:
        """Loss-space weights, for mixers that reweight rather than resample."""
        fn = getattr(self.component, "get_current_doremi_weights", None)
        if callable(fn):
            try:
                return np.asarray(fn(), dtype=np.float64).reshape(-1)
            except Exception as exc:
                logger.warning(f"[Dataflex][Lego] {self.name}.get_current_doremi_weights failed: {exc}")
        return None


# ----------------------------------------------------------------------
# Ordering
# ----------------------------------------------------------------------


class ReorderStage(IndexStage):
    """Arrange the chunk. Always last among index stages.

    The reorder base class already exposes `set_candidate_pool`, which is exactly the
    seam needed here: it orders whatever the upstream stages left rather than
    assuming it owns the whole dataset.
    """

    family = "reorder"
    takes_budget = True

    def __init__(self, component, name, **kw):
        super().__init__(component, name, **kw)
        if getattr(component, "apply_at", "index") == "raw":
            raise ValueError(
                f"reorder stage '{name}' uses apply_at='raw', which permutes raw rows before "
                f"tokenization and so cannot compose with stages that run during training. "
                f"Use apply_at: index for pipelines."
            )

    def apply(self, plan: DataPlan, ctx: PlanContext) -> DataPlan:
        if not self.takes_budget and not len(plan):
            return plan
        setter = getattr(self.component, "set_candidate_pool", None)
        if callable(setter):
            if not self.takes_budget:
                # Restart from the full interval even when its IDs match an
                # already consumed candidate pool.
                setter(None)
            setter(plan.as_list())
        domain_setter = getattr(self.component, "set_domain_ids", None)
        if callable(domain_setter) and ctx.domain_ids is not None:
            domain_setter(ctx.domain_ids)
        self._share_scores(ctx)

        num = ctx.interval_size
        if not self.takes_budget:
            num = len(plan)
        elif len(plan):
            num = min(num, len(plan))
        indices = self.component.next_indices(
            model=ctx.model,
            step_id=ctx.step_id,
            num_samples=num,
            **ctx.as_component_kwargs(),
        )
        indices = list(indices or [])
        logger.info(
            f"[Dataflex][Lego] reorder({self.name}): arranged {len(indices)} samples "
            f"from a pool of {len(plan)}"
        )
        # Flag the plan as ordered so downstream budget enforcement trims from
        # the head (which is the next part of the curriculum) instead of
        # subsampling and breaking the arrangement.
        return plan.replace_indices(indices, stage=f"reorder:{self.name}", ordered=True)

    def _share_scores(self, ctx: PlanContext) -> None:
        """Swap the reorder's ScoreProvider for one backed by the shared board."""
        if getattr(self.component, "score_source", None) == "selector_score":
            selector = self.selection_stage
            selection_step = selector._last_run_step
            if selection_step is None or selection_step > ctx.step_id:
                raise ValueError("Reorder requires an active upstream selection")
            board, consumer = ctx.scoreboard, selector.score_consumer
            snapshot, _ = board.selection_scores_for(consumer, [], selection_step)
            fallback = snapshot.get("reason") in {"random", "delta_baseline"}
            if fallback and self.component.score_params.get("fallback") != "model_loss":
                raise ValueError(f"Selector has no ranking signal ({snapshot['reason']}); configure fallback: model_loss")
            logger.info(
                f"[Dataflex][Lego] reorder uses selector={consumer}, score_step={selection_step}, "
                f"metric={snapshot['metric'] or 'loss'}, ascending={snapshot['ascending'] if not fallback else True}, "
                f"fallback={snapshot.get('reason')}; selector scores are reused in both sharing modes"
            )

            class _SelectorProvider:
                is_dynamic = False
                ascending = True if fallback else snapshot["ascending"]

                def scores_for(self, positions, model=None, step_id=0):
                    _, scores = board.selection_scores_for(consumer, positions, selection_step)
                    if scores is None:
                        # Random has no model score. Delta's baseline already
                        # contains raw loss, so it needs no additional forward.
                        return board.loss_scores(model, positions, step_id, consumer=consumer).tolist()
                    return scores.tolist()

                def describe(self):
                    return f"selector_score({consumer}, step={selection_step})"

            self.component._provider = _SelectorProvider()
            return
        if getattr(self.component, "score_source", None) != "model_loss":
            return
        if getattr(self.component, "_provider", "missing") == "missing":
            return
        fn = self.board_score_fn(ctx)
        if fn is None:
            return

        class _BoardProvider:
            is_dynamic = True

            def describe(self):
                return "scoreboard(loss_full)"

            def scores_for(self, positions, model=None, step_id: int = 0):
                return fn(positions, model=model, step_id=step_id)

        self.component._provider = _BoardProvider()

    def warmup_indices(self, num_samples: int, pool: Optional[List[int]] = None) -> Optional[List[int]]:
        setter = getattr(self.component, "set_candidate_pool", None)
        if callable(setter) and pool is not None:
            setter(pool)
        fn = getattr(self.component, "warmup_indices", None)
        return list(fn(num_samples)) if callable(fn) else None


# ----------------------------------------------------------------------
# Reweighting
# ----------------------------------------------------------------------


class WeightStage(LossStage):
    """Per-sample loss multipliers from a weighter.

    Prefers the composable `get_sample_weights`; falls back to signalling that
    only the legacy scalar path is available, which the pipeline then uses
    directly (and refuses to combine with anything else).
    """

    family = "weighter"

    def __init__(self, component, name, **kwargs):
        super().__init__(component, name, **kwargs)
        if self.every != 1:
            raise ValueError("Weight stages apply every batch; configure refresh_interval on the weighter")

    def weights(self, losses: torch.Tensor, ctx: PlanContext, model=None, inputs=None) -> Optional[torch.Tensor]:
        fn = getattr(self.component, "get_sample_weights", None)
        if not callable(fn):
            return None
        return fn(losses, ctx=ctx, model=model, inputs=inputs)

    def scalar_loss(self, losses: torch.Tensor, ctx: PlanContext, model=None, inputs=None) -> torch.Tensor:
        return self.component.get_weighted_loss(losses, ctx=ctx, model=model, inputs=inputs)


class DomainWeightStage(LossStage):
    """Loss-space contribution of a reweighting mixer (DoReMi).

    DoReMi is configured as a mixer but its `mix()` returns uniform proportions;
    the domain weight is applied to the loss. Representing that as a loss-space
    stage is what lets it coexist with a sample-level weighter instead of the two
    being mutually exclusive.
    """

    family = "mixer"

    def __init__(self, mix_stage: MixStage, **kw):
        super().__init__(mix_stage.component, f"{mix_stage.name}:domain_weights", **kw)
        self.mix_stage = mix_stage

    def weights(self, losses: torch.Tensor, ctx: PlanContext, model=None, inputs=None) -> Optional[torch.Tensor]:
        domain_weights = self.mix_stage.domain_weights()
        if domain_weights is None or inputs is None:
            return None
        domain_ids = inputs.get("domain_id")
        if domain_ids is None:
            return None
        if not torch.is_tensor(domain_ids):
            domain_ids = torch.tensor(domain_ids, dtype=torch.long, device=losses.device)
        domain_ids = domain_ids.view(-1).to(device=losses.device, dtype=torch.long)
        table = torch.tensor(domain_weights, dtype=losses.dtype, device=losses.device)
        return table[domain_ids]


STAGE_TYPES: Dict[str, Any] = {
    "selector": SelectStage,
    "mixer": MixStage,
    "reorder": ReorderStage,
    "weighter": WeightStage,
}
