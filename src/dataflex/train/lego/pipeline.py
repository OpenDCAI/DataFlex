"""The baseplate: holds the stages, validates them, and runs them."""

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from dataflex.core.registry import REGISTRY
from dataflex.utils.load_component import load_component
from dataflex.utils.logging import logger

from .context import PlanContext
from .plan import DataPlan
from .stages import (
    INDEX_ORDER,
    LOSS_FAMILIES,
    STAGE_TYPES,
    DomainWeightStage,
    IndexStage,
    LossStage,
    MixStage,
    ReorderStage,
    SelectStage,
    WeightStage,
)


def _provides_sample_weights(component) -> bool:
    """Whether a weighter actually implements per-sample multipliers.

    An absent method is scalar-only. If the base later gains a placeholder,
    inheriting it still must not count as implementing sample weights.
    """
    from dataflex.train.weighter.base_weighter import Weighter

    own = getattr(type(component), "get_sample_weights", None)
    return own is not None and own is not getattr(Weighter, "get_sample_weights", None)


class SchedulePipeline:
    """An ordered chain of index-space stages plus a loss-space stack.

    Built from a `pipelines:` preset in components.yaml. Each entry names an
    existing `(family, component)` pair, so every selector/mixer/reorder/
    weighter already in DataFlex is usable as a brick with no extra code.
    """

    def __init__(
        self,
        index_stages: Sequence[IndexStage],
        loss_stages: Sequence[LossStage],
        scoreboard=None,
        save_plans: bool = False,
    ):
        self.index_stages: List[IndexStage] = list(index_stages)
        self.loss_stages: List[LossStage] = list(loss_stages)
        self.scoreboard = scoreboard
        self.save_plans = bool(save_plans)

        self._validate()
        for i, stage in enumerate((*self.index_stages, *self.loss_stages)):
            # Stable even when the same algorithm occurs twice in a pipeline.
            stage.score_consumer = f"{i}:{stage.family}:{stage.name}"
            if scoreboard is not None and (
                isinstance(stage, SelectStage)
                or (isinstance(stage, ReorderStage) and getattr(stage.component, "score_source", None) == "model_loss")
                or stage.name in {"multimodal_loss", "multimodal_deltaloss"}
            ):
                stage.validate_score_options()

        selector = None
        for stage in self.index_stages:
            if isinstance(stage, SelectStage):
                selector = stage
            if not isinstance(stage, ReorderStage) or getattr(stage.component, "score_source", None) != "selector_score":
                continue
            from dataflex.train.reorder.dynamic_reorder import DynamicReorder

            if scoreboard is None or selector is None or selector.name not in {"loss", "delta_loss", "random"}:
                raise ValueError("selector_score requires ScoreBoard and an upstream loss/delta_loss/random selector")
            if type(stage.component) is not DynamicReorder or stage.component.reorder_every != 1:
                raise ValueError("selector_score requires dynamic reorder with reorder_every: 1")
            if stage.warmup_step < selector.warmup_step:
                raise ValueError("selector_score reorder cannot start before its selector")
            if selector.name == "loss" and selector.component.score_ascending is None:
                raise ValueError("Loss focus=medium requires explicit score_ascending for selector_score reorder")
            if stage.component.score_params.get("fallback") not in {None, "model_loss"}:
                raise ValueError("selector_score fallback must be model_loss or omitted")
            stage.validate_score_options()
            selector.publish_selection_scores = True
            stage.selection_stage = selector

        self._plan: Optional[DataPlan] = None
        #: What each index stage last handed downstream, one slot per stage, so a
        #: stage that fires while its upstream does not can resume from its own
        #: predecessor's output. A single shared slot would hand it whatever ran
        #: last instead: with three stages that means feeding a stage its own
        #: previous output, and the pool then shrinks at every boundary. Sized
        #: after `_validate`, which may append a taker.
        self._stage_out: List[Optional[np.ndarray]] = [None] * len(self.index_stages)

        logger.info(f"[Dataflex][Lego] pipeline: {self.describe()}")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        stages_cfg: Sequence[Dict[str, Any]],
        components_cfg_file: str,
        runtime: Dict[str, Any],
        scoreboard=None,
        save_plans: bool = False,
        cache_dir: Optional[str] = None,
    ) -> "SchedulePipeline":
        index_stages: List[IndexStage] = []
        loss_stages: List[LossStage] = []

        for entry in stages_cfg:
            entry = dict(entry)
            family = entry.pop("family", None)
            component_name = entry.pop("component", None)
            if not family or not component_name:
                raise ValueError(f"each pipeline stage needs `family` and `component`, got {entry}")
            if family not in STAGE_TYPES:
                raise ValueError(
                    f"unknown stage family '{family}'. Available: {', '.join(sorted(STAGE_TYPES))}"
                )
            if entry.pop("enabled", True) is False:
                logger.info(f"[Dataflex][Lego] stage {family}:{component_name} disabled, skipping")
                continue

            bucket = f"{family}s" if not family.endswith("s") else family
            params = load_component(bucket, components_cfg_file, component_name, runtime_vars={})

            # Give each stage its own cache directory. Selectors cache by
            # `step_{id}.json` alone, so a component reused across pipelines — or
            # used in `filter` mode here after a `draw` mode run elsewhere — would
            # otherwise silently load a stale artifact of the wrong size.
            stage_runtime = dict(runtime)
            if cache_dir:
                stage_runtime["cache_dir"] = os.path.join(cache_dir, f"{family}_{component_name}")

            component = cls._build_component(family, component_name, params, stage_runtime)
            if family == "selector" and component_name in {"loss", "delta_loss"}:
                # Step-only JSON caches cannot identify a model/run. LEGO resume
                # restores its actual stage state instead of accepting those files.
                component.reuse_selection_cache = False

            stage_cls = STAGE_TYPES[family]
            stage = stage_cls(component, component_name, **entry)

            if isinstance(stage, LossStage):
                loss_stages.append(stage)
            else:
                index_stages.append(stage)
                # A reweighting mixer also contributes in loss space. Register
                # that side too rather than making mix and weight exclusive.
                if isinstance(stage, MixStage) and stage.domain_weights() is not None:
                    # Inherit the mixer's schedule: without it the loss-space
                    # twin would start weighting from step 0 while the mixer
                    # itself is still in warmup.
                    loss_stages.append(
                        DomainWeightStage(stage, warmup_step=stage.warmup_step, every=stage.every)
                    )
                    logger.info(
                        f"[Dataflex][Lego] {component_name} reweights by domain; "
                        f"registered as a loss-space contributor as well"
                    )

        return cls(index_stages, loss_stages, scoreboard=scoreboard, save_plans=save_plans)

    @staticmethod
    def _build_component(family: str, name: str, params: Dict[str, Any], runtime: Dict[str, Any]):
        """Instantiate a component the same way its dedicated trainer would."""
        if family == "selector":
            from dataflex.train.data.loader import DomainLabeledDataset

            # The wrapper adds global IDs for training/weight lookup. Selectors
            # already return positions, so give them the same rows without those
            # extra fields. No subset or reindexing: candidate positions remain
            # global. Their own IndexedDataset/collator bookkeeping stays intact.
            runtime = dict(runtime)
            dataset = runtime.get("dataset")
            if isinstance(dataset, DomainLabeledDataset):
                runtime["dataset"] = dataset.dataset

        if family == "reorder":
            from dataflex.train.reorder import resolve_reorder_kind

            kind = resolve_reorder_kind(name, params)
            component = REGISTRY.build("reorder", kind, runtime=runtime, cfg=params)
            # The stage needs to reject apply_at='raw'; surface it for the check.
            setattr(component, "apply_at", params.get("apply_at", "index"))
            return component

        if family == "mixer":
            # Mixers take a mixture_manager positionally today. In a pipeline the
            # domain information lives on the context instead, so pass a light
            # stand-in that answers the attribute lookups they actually make.
            cfg = dict(params)
            cfg.setdefault("mixture_manager", runtime.get("mixture_manager"))
            return REGISTRY.build("mixer", name, runtime=runtime, cfg=cfg)

        return REGISTRY.build(family, name, runtime=runtime, cfg=params)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        families = [s.family for s in self.index_stages]

        for family in INDEX_ORDER:
            if families.count(family) > 1:
                raise ValueError(
                    f"pipeline has {families.count(family)} '{family}' index stages; at most one is supported"
                )

        # Enforce the canonical order rather than trusting the YAML: proportions
        # over survivors, arrange a known chunk, ordering last.
        rank = {f: i for i, f in enumerate(INDEX_ORDER)}
        actual = [rank[f] for f in families if f in rank]
        if actual != sorted(actual):
            listed = " -> ".join(families)
            canonical = " -> ".join(f for f in INDEX_ORDER if f in families)
            logger.warning(
                f"[Dataflex][Lego] stages listed as {listed}, reordering to the canonical "
                f"{canonical} (select before mix before reorder is required for the composition "
                f"to be well defined)"
            )
            self.index_stages.sort(key=lambda s: rank.get(s.family, len(rank)))

        takers = [s for s in self.index_stages if s.takes_budget]
        if len(takers) > 1:
            # Only the last taker actually determines the interval, so the
            # earlier ones must not shrink it to the budget.
            for s in takers[:-1]:
                if isinstance(s, SelectStage):
                    raise ValueError(
                        f"select stage '{s.name}' is in draw mode but is followed by another stage that "
                        f"takes the interval budget. Use mode: filter so it acts as a pool filter."
                    )
                s.takes_budget = False
            logger.info(
                f"[Dataflex][Lego] {takers[-1].family}:{takers[-1].name} is the budget taker"
            )

        if not any(s.takes_budget for s in self.index_stages):
            logger.info(
                "[Dataflex][Lego] no configured stage takes the interval budget; "
                "appending a uniform taker so the plan is always the right size"
            )
            self.index_stages.append(_UniformTakeStage())

        scalar_only = [
            s for s in self.loss_stages if isinstance(s, WeightStage) and not _provides_sample_weights(s.component)
        ]
        if len(self.loss_stages) > 1 and scalar_only:
            raise ValueError(
                f"loss-space stages {[s.name for s in scalar_only]} only provide a scalar loss and cannot "
                f"be combined with other loss-space contributors. Implement get_sample_weights on them."
            )

    def describe(self) -> str:
        index_part = " -> ".join(s.describe() for s in self.index_stages) or "(none)"
        loss_part = " x ".join(s.describe() for s in self.loss_stages) or "(none)"
        return f"index[{index_part}] loss[{loss_part}]"

    def set_data_collator(self, data_collator) -> None:
        """Use the Trainer's resolved collator for every scoring path as well."""
        # MetadataCollator only restores fields present in its input rows.
        # Selector rows have no LEGO IDs; indices added by selectors themselves
        # are still preserved for their existing scoring/gather logic.
        if self.scoreboard is not None:
            self.scoreboard.data_collator = data_collator
        for stage in (*self.index_stages, *self.loss_stages):
            component = stage.component
            setter = getattr(component, "set_data_collator", None)
            if callable(setter):
                setter(data_collator)
            elif hasattr(component, "data_collator"):
                component.data_collator = data_collator

    @property
    def has_reorder(self) -> bool:
        return any(isinstance(s, ReorderStage) for s in self.index_stages)

    # ------------------------------------------------------------------
    # Index space
    # ------------------------------------------------------------------

    def build_plan(self, ctx: PlanContext) -> DataPlan:
        """Run the index stages and return the plan for the coming interval.

        Firing is upward-closed: because every stage consumes the previous
        stage's output, once one stage recomputes, all later stages must too.
        Stages before the first firing one keep their cached pool.
        """
        first = self._first_firing(ctx.step_id)

        if first is None and self._plan is not None:
            logger.info(f"[Dataflex][Lego] step {ctx.step_id}: no stage due, reusing the previous plan")
            return self._plan

        start = 0 if first is None else first
        # Resume from the output of the stage immediately before the first firing
        # one. Falling back to the whole dataset is correct when that predecessor
        # has never run, which happens while it is still in warmup.
        resume = self._stage_out[start - 1] if start > 0 else None
        if resume is not None:
            plan = DataPlan.over(resume, reused_prefix=start)
        else:
            plan = DataPlan.over(range(ctx.dataset_size))

        for i, stage in enumerate(self.index_stages):
            if i < start:
                continue
            if not stage.is_active(ctx.step_id):
                logger.info(
                    f"[Dataflex][Lego] step {ctx.step_id}: {stage.family}:{stage.name} still in warmup "
                    f"(until {stage.warmup_step}), passing the pool through"
                )
                self._stage_out[i] = plan.indices
                continue
            plan = stage.apply(plan, ctx)
            stage.mark_run(ctx.step_id)
            self._stage_out[i] = plan.indices

        plan = self._enforce_budget(plan, ctx)
        self._plan = plan

        logger.info(f"[Dataflex][Lego] step {ctx.step_id}: plan = {plan.describe(ctx.domain_ids)}")
        if self.save_plans and self.scoreboard is not None:
            self.scoreboard.save(ctx.step_id, plan.indices)
        return plan

    def _first_firing(self, step_id: int) -> Optional[int]:
        for i, stage in enumerate(self.index_stages):
            if stage.should_run(step_id):
                return i
        return None

    def _enforce_budget(self, plan: DataPlan, ctx: PlanContext) -> DataPlan:
        """The plan handed to the trainer must be exactly one interval long."""
        budget = ctx.interval_size
        if budget <= 0 or len(plan) == budget:
            return plan

        if len(plan) > budget:
            if plan.meta.get("ordered"):
                # A reorder stage arranged this, so the head is exactly the part
                # of the curriculum that comes next.
                logger.info(f"[Dataflex][Lego] plan has {len(plan)} samples, taking the leading {budget}")
                return plan.replace_indices(plan.indices[:budget], stage="budget:head")

            # Unordered pool: taking the head would bias the draw toward whatever
            # happens to sit at low indices — with concatenated domains that is
            # the first domain only. Sample instead.
            logger.info(
                f"[Dataflex][Lego] plan has {len(plan)} samples and no ordering stage; "
                f"drawing {budget} of them uniformly"
            )
            rng = np.random.default_rng(ctx.step_id + 4242)
            take = rng.choice(len(plan.indices), size=budget, replace=False)
            picked = plan.indices[take]
            return plan.replace_indices(picked, stage="budget:sample")

        # Short: repeat the plan rather than starving the interval.
        need = budget - len(plan)
        if len(plan.indices) == 0:
            logger.warning("[Dataflex][Lego] plan is empty; falling back to a uniform draw")
            rng = np.random.default_rng(ctx.step_id)
            picked = rng.choice(max(1, ctx.dataset_size), size=budget, replace=True).tolist()
            return plan.replace_indices(picked, stage="budget:fallback")

        logger.info(f"[Dataflex][Lego] plan is {need} short of the budget {budget}; cycling it")
        reps = -(-budget // len(plan.indices))  # ceil
        filled = np.tile(plan.indices, reps)[:budget]
        return plan.replace_indices(filled, stage="budget:cycle")

    def warmup_plan(self, ctx: PlanContext) -> List[int]:
        """Indices for the warmup interval, before any stage has fired.

        A reorder stage knows how it wants to start (head of the curriculum, or a
        random draw when its scores need a trained model), so defer to it if
        present. Otherwise draw uniformly, which matches what every other family
        does during warmup.
        """
        for stage in self.index_stages:
            if isinstance(stage, ReorderStage):
                indices = stage.warmup_indices(ctx.interval_size, pool=list(range(ctx.dataset_size)))
                if indices:
                    logger.info(f"[Dataflex][Lego] warmup deferred to reorder stage {stage.name}")
                    return indices

        rng = np.random.default_rng(ctx.step_id or 42)
        size = max(1, ctx.dataset_size)
        replace = ctx.interval_size > size
        picked = rng.choice(size, size=ctx.interval_size, replace=replace).tolist()
        logger.info(f"[Dataflex][Lego] warmup draws {len(picked)} samples uniformly")
        return picked

    # ------------------------------------------------------------------
    # Loss space
    # ------------------------------------------------------------------

    def has_loss_stages(self, step_id: int) -> bool:
        return any(s.is_active(step_id) for s in self.loss_stages)

    def prepare_loss_stages(self, ctx: PlanContext, model) -> None:
        """Refresh before the differentiable training forward, never in evaluation."""
        for stage in self.loss_stages:
            if isinstance(stage, WeightStage) and stage.is_active(ctx.step_id):
                setter = getattr(stage.component, "set_score_source", None)
                if callable(setter):
                    if ctx.scoreboard is not None and stage.component.dataset is not ctx.scoreboard.dataset:
                        raise ValueError("LEGO weighter and ScoreBoard must reference the same global training dataset")
                    setter(stage.board_score_fn(ctx))
                prepare = getattr(stage.component, "prepare", None)
                if callable(prepare):
                    prepare(ctx, model)

    def weighted_loss(
        self,
        per_sample_losses: torch.Tensor,
        ctx: PlanContext,
        model=None,
        inputs=None,
    ) -> Optional[torch.Tensor]:
        """Combine every active loss-space contributor into a scalar loss.

        Weights multiply, then reduce by mean:  L = mean_i ( prod_k w_i^(k) * l_i ).
        Returns None when nothing is active, so the caller keeps the plain loss.
        """
        active = [s for s in self.loss_stages if s.is_active(ctx.step_id)]
        if not active:
            return None

        combined: Optional[torch.Tensor] = None
        contributors: List[str] = []

        for stage in active:
            w = stage.weights(per_sample_losses, ctx, model=model, inputs=inputs)
            if w is None:
                continue
            w = w.view(-1).to(device=per_sample_losses.device, dtype=per_sample_losses.dtype)
            if w.numel() != per_sample_losses.numel():
                logger.warning(
                    f"[Dataflex][Lego] {stage.name} returned {w.numel()} weights for "
                    f"{per_sample_losses.numel()} samples; ignoring it this step"
                )
                continue
            combined = w if combined is None else combined * w
            contributors.append(stage.name)

        if combined is not None:
            if len(contributors) > 1:
                logger.info(f"[Dataflex][Lego] loss weights from {' x '.join(contributors)}")
            return (combined * per_sample_losses).mean()

        # Nobody could produce weights. Fall back to the single scalar weighter,
        # which is still valid on its own.
        for stage in active:
            if isinstance(stage, WeightStage):
                return stage.scalar_loss(per_sample_losses, ctx, model=model, inputs=inputs)
        return None

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def observe(self, **signals) -> None:
        for stage in list(self.index_stages) + list(self.loss_stages):
            stage.observe(**signals)

    def mixer_stages(self) -> List[MixStage]:
        return [s for s in self.index_stages if isinstance(s, MixStage)]

    def on_train_end(self, output_dir: str) -> None:
        """Give components a chance to persist whatever they accumulated."""
        for stage in self.mixer_stages():
            fn = getattr(stage.component, "save_average_weights", None)
            if callable(fn):
                try:
                    fn(output_dir)
                    logger.info(f"[Dataflex][Lego] {stage.name}.save_average_weights done")
                except Exception as exc:
                    logger.warning(f"[Dataflex][Lego] {stage.name}.save_average_weights failed: {exc}")
        if self.scoreboard is not None:
            logger.info(f"[Dataflex][Lego] scoreboard usage — {self.scoreboard.summary()}")


class _UniformTakeStage(IndexStage):
    """Draws the interval when no configured stage does.

    Keeps the budget contract total: a pipeline of only filters still yields a
    correctly sized plan, and a pipeline with no index stages at all degenerates
    to ordinary random sampling.
    """

    family = "take"
    takes_budget = True

    def __init__(self):
        super().__init__(component=None, name="uniform", warmup_step=0, every=1)

    def apply(self, plan: DataPlan, ctx: PlanContext) -> DataPlan:
        pool = plan.indices if len(plan.indices) else np.arange(ctx.dataset_size, dtype=np.int64)
        rng = np.random.default_rng(ctx.step_id + 999)
        replace = ctx.interval_size > len(pool)
        take = rng.choice(len(pool), size=ctx.interval_size, replace=replace)
        picked = np.asarray(pool, dtype=np.int64)[take]
        return plan.replace_indices(picked, stage="take:uniform")
