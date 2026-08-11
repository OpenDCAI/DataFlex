from typing import List, Optional

import torch
from typing_extensions import override

from dataflex.core.registry import REGISTRY
from dataflex.utils.load_component import load_component
from dataflex.utils.logging import logger

from dataflex.train.reorderer import resolve_reorderer_kind  # also registers the reorderers

from .select_trainer import SelectTrainer


class _ReordererAsSelector:
    """Adapt a reorderer to the index-provider protocol `SelectTrainer` expects.

    `SelectTrainer` already does exactly what a reorderer needs: every
    `update_step` steps it asks a component for a list of indices, wraps them in
    `torch.utils.data.Subset` (which respects list order) and swaps the
    iterator. Reusing that loop rather than copying a fourth ~550-line
    `_inner_training_loop` is the whole point of this adapter.

    Only two methods are needed, and both differ from a real selector:
    `warmup` returns the head of the curriculum rather than a random sample,
    and `select` returns an ordered chunk rather than a scored subset.
    """

    def __init__(self, reorderer, accelerator=None):
        self.reorderer = reorderer
        self.accelerator = accelerator
        self.data_collator = None  # assigned by get_train_dataloader

    def warmup(self, num_samples: int, replacement: bool = False) -> List[int]:
        # The base Selector.warmup samples randomly with replacement, which for a
        # score-ordered run would start the curriculum in the middle. The
        # reorderer decides instead: the head of the ordering when scores are
        # model-independent, a random draw when they are not.
        return self.reorderer.warmup_indices(num_samples)

    def select(self, model, step_id: int, num_samples: int, **kwargs) -> List[int]:
        return self.reorderer.next_indices(model=model, step_id=step_id, num_samples=num_samples, **kwargs)

    def __getattr__(self, item):
        # Forward anything else (e.g. observe) to the reorderer. Guarded so an
        # access before __init__ finishes raises instead of recursing forever.
        if item == "reorderer":
            raise AttributeError(item)
        return getattr(self.reorderer, item)


class ReorderTrainer(SelectTrainer):
    """Trainer for the `dynamic_reorder` train type.

    Inherits `SelectTrainer`'s loop wholesale and changes only two things: which
    component supplies the indices, and the sampler.
    """

    def __init__(self, finetuning_args, processor=None, gen_kwargs=None, model_args=None, **kwargs):
        # Skip SelectTrainer.__init__, which would build a selector, and call the
        # LlamaFactory trainer directly. It takes both `model_args` (pt stage)
        # and `gen_kwargs` (sft stage) as optionals, so one call covers both.
        super(SelectTrainer, self).__init__(
            finetuning_args=finetuning_args,
            processor=processor,
            model_args=model_args,
            gen_kwargs=gen_kwargs,
            **kwargs,
        )

        name = finetuning_args.component_name
        params = load_component("reorderers", finetuning_args.components_cfg_file, name, runtime_vars={})
        kind = resolve_reorderer_kind(name, params)

        runtime = dict(
            dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            accelerator=self.accelerator,
            data_collator=self.data_collator,
        )
        self.reorderer = REGISTRY.build("reorderer", kind, runtime=runtime, cfg=params)

        # SelectTrainer's loop calls `self.selector`; the adapter makes the
        # reorderer answer to that protocol without editing select_trainer.py.
        self.selector = _ReordererAsSelector(self.reorderer, accelerator=self.accelerator)

        logger.info(f"[ReorderTrainer] reorderer={name} (kind={kind}), params={params}")
        logger.info("[Dataflex] ReorderTrainer initialized")

    @override
    def _get_train_sampler(self, train_dataset=None) -> Optional[torch.utils.data.Sampler]:
        """Always sequential.

        The reorderer's output order *is* the curriculum, so any shuffling
        sampler silently discards it and the run degrades into the random
        baseline while still looking correct. This is forced rather than left to
        `disable_shuffling` because that failure is invisible in the logs.
        """
        if train_dataset is None:
            train_dataset = self.train_dataset
        if train_dataset is None:
            return None

        if self.args.group_by_length:
            logger.warning(
                "[Dataflex][Reorder] `group_by_length` reorders batches by length and would override the "
                "curriculum; ignoring it."
            )
        if not self.finetuning_args.disable_shuffling:
            logger.info("[Dataflex][Reorder] forcing SequentialSampler so the ordering survives.")

        return torch.utils.data.SequentialSampler(train_dataset)

    @override
    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, *args, **kwargs):
        # Feed the training signal back to the reorderer. Nothing consumes it
        # yet; this is the seam for an adaptive controller that would react to
        # gradient-norm spikes at cycle boundaries.
        try:
            self.reorderer.observe(
                global_step=self.state.global_step,
                grad_norm=float(grad_norm) if grad_norm is not None else None,
                learning_rate=self._get_learning_rate(),
            )
        except Exception as exc:
            logger.warning(f"[Dataflex][Reorder] observe() failed: {exc}")

        return super()._maybe_log_save_evaluate(
            tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, *args, **kwargs
        )
