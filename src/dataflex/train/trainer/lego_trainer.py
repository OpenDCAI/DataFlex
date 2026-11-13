import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from transformers.trainer_utils import get_last_checkpoint
from typing_extensions import override

from dataflex.train.data.loader import SAMPLE_METADATA_KEYS, MetadataCollator, split_batch
from dataflex.train.hooks import group_by_length_requested
from dataflex.train.lego import PlanContext, SchedulePipeline, ScoreBoard
from dataflex.train.lego.checkpoint import dataset_identity, pipeline_state, restore_pipeline, save_atomic
from dataflex.train.lego.domain_view import DomainView
from dataflex.train.weighter.base_weighter import scoring_mode
from dataflex.utils.load_component import load_component
from dataflex.utils.logging import logger
from dataflex.utils.multimodal_selection import fingerprint

from .select_trainer import SelectTrainer


class _PipelineAsSelector:
    """Present the pipeline through the index-provider protocol the loop expects.

    `SelectTrainer._inner_training_loop` already does the hard part: every
    `update_step` optimizer steps it asks a component for indices, wraps them in
    `Subset` (which respects list order) and swaps the iterator. That is exactly
    the shape a plan needs, so the pipeline borrows the loop instead of the
    codebase gaining a fifth ~550-line copy of it.
    """

    checkpoint_stage = True

    def __init__(self, trainer: "LegoTrainer"):
        self.trainer = trainer
        self._data_collator = trainer.data_collator

    @property
    def data_collator(self):
        return self._data_collator

    @data_collator.setter
    def data_collator(self, value):
        self._data_collator = value
        self.trainer.pipeline.set_data_collator(value)

    def checkpoint_signature(self):
        return self.trainer.lego_checkpoint_signature()

    def export_checkpoint_state(self, **kwargs):
        return {"version": 1}

    def restore_checkpoint_state(self, state, *, global_step, **kwargs):
        if state != {"version": 1} or getattr(self.trainer, "_lego_restored_step", None) != global_step:
            raise ValueError("The selection cursor is missing its matching LEGO checkpoint")

    def warmup(self, num_samples: int, replacement: bool = False) -> List[int]:
        ctx = self.trainer.plan_context(num_samples=num_samples, step_id=0)
        return self.trainer.pipeline.warmup_plan(ctx)

    def select(self, model, step_id: int, num_samples: int, **kwargs) -> List[int]:
        ctx = self.trainer.plan_context(
            num_samples=num_samples,
            step_id=step_id,
            model=model,
            **kwargs,
        )
        with scoring_mode(model):
            plan = self.trainer.pipeline.build_plan(ctx)
        self.trainer._active_plan = plan
        return plan.as_list()


class LegoTrainer(SelectTrainer):
    """Trainer for `train_type: dynamic_lego`.

    Runs a chain of data-centric strategies in one training run. Index-space
    stages build the plan for each interval; loss-space stages multiply their
    per-sample weights at each step.
    """

    def __init__(self, finetuning_args, processor=None, gen_kwargs=None, model_args=None, **kwargs):
        # Extra keys `lego_get_dataset` attached to the dataset module arrive
        # here because run_pt/run_sft splat it into the constructor.
        domain_ids = kwargs.pop("domain_ids", None)
        domain_names = kwargs.pop("domain_names", None)

        # Skip SelectTrainer.__init__, which would build a single selector.
        super(SelectTrainer, self).__init__(
            finetuning_args=finetuning_args,
            processor=processor,
            model_args=model_args,
            gen_kwargs=gen_kwargs,
            **kwargs,
        )

        self.domain_ids = np.asarray(domain_ids, dtype=np.int64) if domain_ids is not None else None
        self.domain_names = list(domain_names) if domain_names else None
        self._active_plan = None
        self._last_signals = {}
        self.data_collator = MetadataCollator(self.data_collator)

        name = finetuning_args.component_name
        cfg = load_component("pipelines", finetuning_args.components_cfg_file, name, runtime_vars={})
        cfg.setdefault("score_sharing", "shared")
        stages_cfg = cfg.get("stages")
        if not stages_cfg:
            raise ValueError(
                f"pipeline '{name}' has no `stages`. A pipeline is a list of "
                f"{{family, component}} entries under `params.stages`."
            )
        self._lego_config = {"pipeline": cfg, "components": [
            load_component(f"{entry['family']}s", finetuning_args.components_cfg_file, entry["component"], {})
            for entry in stages_cfg if entry.get("enabled", True)
        ]}
        self._lego_signature = None

        self.scoreboard = ScoreBoard(
            dataset=self.train_dataset,
            accelerator=self.accelerator,
            data_collator=self.data_collator,
            dataset_size=len(self.train_dataset) if self.train_dataset is not None else 0,
            batch_size=int(cfg.get("score_batch_size", 8)),
            num_workers=int(cfg.get("score_num_workers", 2)),
            cache_dir=cfg.get("cache_dir"),
            seed=int(cfg.get("seed", 42)),
            score_sharing=cfg["score_sharing"],
        )

        # A stand-in for `mixture_manager`: the mixers read `names` /
        # `initial_proportions` / `sources` off it, and the composition
        # representation can answer all three without per-source datasets.
        self.domain_view = DomainView(
            names=self.domain_names or ["all"],
            domain_ids=self.domain_ids
            if self.domain_ids is not None
            else np.zeros(len(self.train_dataset) if self.train_dataset is not None else 0, dtype=np.int64),
            dataset=getattr(self.train_dataset, "dataset", self.train_dataset),
            initial_proportions=getattr(self.args, "init_mixture_proportions", None)
            or getattr(finetuning_args, "init_mixture_proportions", None),
        )

        runtime = dict(
            dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            accelerator=self.accelerator,
            data_collator=self.data_collator,
            output_dir=self.args.output_dir,
            mixture_manager=self.domain_view,
        )

        self.pipeline = SchedulePipeline.from_config(
            stages_cfg,
            components_cfg_file=finetuning_args.components_cfg_file,
            runtime=runtime,
            scoreboard=self.scoreboard,
            save_plans=bool(cfg.get("save_plans", False)),
            cache_dir=cfg.get("cache_dir") or str(Path(self.args.output_dir) / "lego_cache"),
        )
        if any(stage.family == "reorder" and not stage.takes_budget for stage in self.pipeline.index_stages):
            # The budget policy changes the planned samples under the same YAML.
            self._lego_config["budget_before_reorder"] = True

        # The loop drives `self.selector`; the adapter routes it to the pipeline
        # without editing select_trainer.py.
        self.selector = _PipelineAsSelector(self)

        logger.info(f"[LegoTrainer] pipeline={name}")
        if self.domain_names:
            logger.info(f"[LegoTrainer] domains: {self.domain_view.describe()}")
        else:
            logger.info("[LegoTrainer] single-domain run; mixture stages will have nothing to allocate over")
        logger.info("[Dataflex] LegoTrainer initialized")

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------

    def plan_context(self, num_samples: int, step_id: int, model=None, **kwargs) -> PlanContext:
        return PlanContext.from_trainer(
            self,
            model=model,
            data_collator=self.selector.data_collator,
            tokenizer=kwargs.get("tokenizer", self.processing_class),
            step_id=step_id,
            interval_size=num_samples,
            current_update_times=kwargs.get("current_update_times", 1),
            update_times=kwargs.get("update_times", self.finetuning_args.update_times),
            optimizer_state=kwargs.get("optimizer_state"),
            scheduler_state=kwargs.get("scheduler_state"),
            batch=kwargs.get("batch"),
            scoreboard=self.scoreboard,
            signals=dict(self._last_signals),
        )

    # ------------------------------------------------------------------
    # Index space
    # ------------------------------------------------------------------

    def dataflex_initial_indices(self):
        # LEGO can cycle a small pool; standalone SelectTrainer's no-replacement
        # size restriction does not describe the pipeline's budget contract.
        count = self._warmup_samples or self._total_train_batch_size * self.finetuning_args.update_step
        return self.selector.warmup(count, replacement=True)

    def lego_checkpoint_signature(self):
        if self._lego_signature is None:
            def semantic(value):
                if isinstance(value, dict):
                    return {key: semantic(item) for key, item in value.items()
                            if key not in {"cache_dir", "output_dir", "save_plans"}}
                if isinstance(value, (list, tuple)):
                    return [semantic(item) for item in value]
                return value

            collator = self.data_collator
            tokenizer = getattr(collator, "tokenizer", None)
            processor = getattr(collator, "processor", None)
            image_processor = getattr(processor, "image_processor", None)
            self._lego_signature = fingerprint({
                "version": 2, "config": semantic(self._lego_config),
                "scoring": self.scoreboard.scoring_contract,
                "precision": {key: getattr(self.args, key, None) for key in ("bf16", "fp16", "tf32")},
                "data": dataset_identity(self.train_dataset), "eval": dataset_identity(self.eval_dataset),
                "domains": self.domain_names,
                "collator": f"{type(collator.collator).__module__}.{type(collator.collator).__qualname__}",
                "loader": {key: getattr(self.args, key, None) for key in (
                    "dataloader_num_workers", "dataloader_persistent_workers", "dataloader_prefetch_factor",
                    "dataloader_drop_last",
                )},
                "tokenizer": tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else str(tokenizer),
                "padding_side": getattr(tokenizer, "padding_side", None),
                "template": repr(getattr(collator, "template", None)),
                "image_processor": image_processor.to_dict() if hasattr(image_processor, "to_dict") else None,
            })
        return self._lego_signature

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        # These are consumed by the collator, before model.forward. Extending
        # the signature covers both HF Dataset pruning and RemoveColumnsCollator.
        self._signature_columns = list(dict.fromkeys([
            *self._signature_columns, *SAMPLE_METADATA_KEYS, "images", "videos", "audios",
        ]))

    def _get_collator_with_removed_columns(self, data_collator, description=None):
        if isinstance(data_collator, MetadataCollator):
            data_collator = data_collator.collator
        collator = super()._get_collator_with_removed_columns(data_collator, description=description)
        return MetadataCollator(collator)

    @override
    def _get_train_sampler(self, train_dataset=None) -> Optional[torch.utils.data.Sampler]:
        """Sequential exactly when the plan's order is meaningful.

        With a reorder stage the plan order *is* the curriculum and a shuffling
        sampler would silently discard it. Without one, order carries no
        information and shuffling is the better default.
        """
        if train_dataset is None:
            train_dataset = self.train_dataset
        if train_dataset is None:
            return None

        if self.pipeline.has_reorder:
            if group_by_length_requested(self.args):
                logger.warning(
                    "[Dataflex][Lego] length-grouped batching would reorder batches by length and "
                    "override the curriculum; ignoring it."
                )
            return torch.utils.data.SequentialSampler(train_dataset)

        return super()._get_train_sampler(train_dataset)

    # ------------------------------------------------------------------
    # Loss space
    # ------------------------------------------------------------------

    @override
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Apply the loss-space stack.

        Scheduling metadata is kept for weight stages, outside model.forward.
        """
        if not model.training or not self.pipeline.has_loss_stages(self.state.global_step):
            return self._forward_without_metadata(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )

        model_inputs, metadata = split_batch(inputs)
        labels = model_inputs.get("labels")

        outputs = model(**model_inputs)
        if labels is None or getattr(outputs, "logits", None) is None:
            logger.warning("[Dataflex][Lego] cannot form per-sample losses; using the model's scalar loss")
            return self._forward_without_metadata(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )

        # Must stay in the graph, so this is the differentiable counterpart of
        # `per_sample_loss_from_outputs` (which detaches for scoring).
        per_sample = self._per_sample_loss_grad(outputs, labels)

        ctx = self.plan_context(
            num_samples=0,
            step_id=self.state.global_step,
            model=model,
            batch=inputs,
        )
        weight_inputs = {**model_inputs, **metadata}

        loss = self.pipeline.weighted_loss(per_sample, ctx, model=model, inputs=weight_inputs)
        if loss is None:
            loss = per_sample.mean()

        return (loss, outputs) if return_outputs else loss

    def _forward_without_metadata(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        model_inputs, _ = split_batch(inputs)
        return super().compute_loss(
            model, model_inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
        )

    @staticmethod
    def _per_sample_loss_grad(outputs, labels) -> torch.Tensor:
        """Token-mean cross-entropy per sample, keeping the graph.

        Upcast to float32 for the same reason as `per_sample_loss_from_logits`:
        under bf16 the loss quantises coarsely enough to distort the per-sample
        weights the loss stack is about to compute from it.
        """
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].float().contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        tok_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1).long(),
        ).view(shift_labels.size(0), -1)
        active = (shift_labels != -100).sum(dim=1)
        return tok_loss.sum(dim=1) / torch.clamp(active, min=1)

    @override
    def training_step(self, model, inputs, num_items_in_batch=None):
        ctx = self.plan_context(num_samples=0, step_id=self.state.global_step, model=model, batch=inputs)
        with self.compute_loss_context_manager():
            self.pipeline.prepare_loss_stages(ctx, model)
        # Our scalar is reduced from per-sample losses, not the model's
        # num_items_in_batch-normalized loss. Let HF apply accumulation scaling
        # even for models whose forward accepts loss kwargs (e.g. Qwen).
        weighted = self.pipeline.has_loss_stages(self.state.global_step)
        accepts_loss_kwargs = self.model_accepts_loss_kwargs
        compute_loss_func = self.compute_loss_func
        try:
            if weighted:
                self.model_accepts_loss_kwargs = False
                self.compute_loss_func = None
            loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        finally:
            self.model_accepts_loss_kwargs = accepts_loss_kwargs
            self.compute_loss_func = compute_loss_func

        self._update_reweighting_mixers(model, inputs)

        # ODM attributes its bandit reward per batch. MixTrainer does this inside
        # its own loop; here it belongs in the step, which is the only place that
        # sees both the loss and the batch's domain.
        for stage in self.pipeline.mixer_stages():
            fn = getattr(stage.component, "update_batch_info", None)
            if not callable(fn):
                continue
            domain_id = inputs.get("domain_id")
            if domain_id is None:
                continue
            if torch.is_tensor(domain_id):
                domain_id = domain_id.view(-1)
                domain_id = int(domain_id[0].item()) if domain_id.numel() else None
            if domain_id is None:
                continue
            try:
                fn(float(loss.detach().float().cpu()), domain_id)
            except Exception as exc:
                logger.warning(f"[Dataflex][Lego] {stage.name}.update_batch_info failed: {exc}")

        return loss

    def _update_reweighting_mixers(self, model, inputs) -> None:
        """Advance a domain-reweighting mixer using the batch just trained on.

        DoReMi's update compares per-token losses against a reference model, so
        it needs a real batch and the batch's own domain labels. A planning
        boundary has neither, which is why `MixStage.apply` leaves index space
        alone for such a mixer and the work happens here instead.
        """
        stages = [s for s in self.pipeline.mixer_stages() if s.reweights_in_loss_space]
        if not stages:
            return
        domain_id = inputs.get("domain_id")
        if domain_id is None:
            return
        if not torch.is_tensor(domain_id):
            domain_id = torch.as_tensor(domain_id, dtype=torch.long)
        batch, _ = split_batch(inputs)
        for stage in stages:
            stage.update_from_batch(model, self.state.global_step, batch, domain_id.view(-1))

    # ------------------------------------------------------------------
    # Feedback and teardown
    # ------------------------------------------------------------------

    def _save_checkpoint(self, model, trial, *args, **kwargs):
        super()._save_checkpoint(model, trial, *args, **kwargs)
        output = Path(self._get_output_dir(trial=trial)) / f"checkpoint-{self.state.global_step}"
        save_atomic(output / f"dataflex_lego_rank_{self.args.process_index}.pt", {
            "version": 1, "global_step": self.state.global_step,
            "signature": self._selection_run_signature(),
            "pipeline": pipeline_state(self.pipeline), "signals": self._last_signals,
        })

    @override
    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, *args, **kwargs):
        self._last_signals = dict(
            global_step=self.state.global_step,
            grad_norm=float(grad_norm) if grad_norm is not None else None,
            learning_rate=self._get_learning_rate(),
        )
        self.pipeline.observe(**self._last_signals)
        return super()._maybe_log_save_evaluate(
            tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, *args, **kwargs
        )

    @override
    def train(self, *args, **kwargs):
        fa = self.finetuning_args
        if fa.update_step <= 0 or fa.warmup_step < 0:
            raise ValueError("LEGO requires update_step > 0 and warmup_step >= 0")
        if any(s.family == "selector" and s.name == "delta_loss" for s in self.pipeline.index_stages):
            if fa.update_times < 2:
                raise ValueError("delta_loss selector requires update_times >= 2 for its baseline/window schedule")
        resume = kwargs.get("resume_from_checkpoint", args[0] if args else None)
        if resume is True:
            resume = get_last_checkpoint(self.args.output_dir)
        if resume:
            path = Path(resume) / f"dataflex_lego_rank_{self.args.process_index}.pt"
            if not path.exists():
                raise ValueError(f"LEGO state is missing: {path}")
            state = torch.load(path, map_location="cpu", weights_only=True)
            if state.get("pipeline", {}).get("version") != 2:
                raise ValueError("Unsupported LEGO checkpoint schema; checkpoints from before shared scoring need a new run")
            trainer_state = json.loads((Path(resume) / "trainer_state.json").read_text())
            if (state.get("version") != 1 or state.get("signature") != self._selection_run_signature()
                    or state.get("global_step") != trainer_state["global_step"]):
                raise ValueError("LEGO checkpoint dataset/config/schedule/step does not match")
            restore_pipeline(self.pipeline, state["pipeline"], len(self.train_dataset))
            self._active_plan = self.pipeline._plan
            self._last_signals = state["signals"]
            self._lego_restored_step = state["global_step"]
        result = super().train(*args, **kwargs)
        try:
            self.pipeline.on_train_end(self.args.output_dir)
        except Exception as exc:
            logger.warning(f"[Dataflex][Lego] on_train_end failed: {exc}")
        return result
