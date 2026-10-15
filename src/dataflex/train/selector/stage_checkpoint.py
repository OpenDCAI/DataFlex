"""Opt-in checkpointing for selectors whose stage must survive a restart.

The upstream Trainer skips its full-pool loader on resume; DataFlex reads a
different, selected loader. Its indices and batch cursor must be restored too.
Other selectors, reorder, mixture and Lego retain their existing behavior.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers.trainer_utils import get_last_checkpoint

from dataflex.utils.multimodal_selection import fingerprint, write_json


class SelectionCheckpointMixin:
    @property
    def _checkpoint_selection_stage(self):
        return bool(getattr(getattr(self, "selector", None), "checkpoint_stage", False))

    def _selection_run_signature(self):
        args, fa = self.args, self.finetuning_args
        return fingerprint(
            {
                "selector": self.selector.checkpoint_signature(),
                "schedule": [fa.warmup_step, fa.update_step, fa.update_times, self.dataflex_step_budget()],
                "batch": [args.per_device_train_batch_size, args.gradient_accumulation_steps, args.world_size],
                "sampling": [
                    args.seed,
                    args.data_seed,
                    fa.disable_shuffling,
                    args.dataloader_drop_last,
                    getattr(args, "train_sampling_strategy", None),
                    getattr(args, "group_by_length", False),
                ],
            }
        )

    def train(self, *args, **kwargs):
        if not self._checkpoint_selection_stage:
            return super().train(*args, **kwargs)
        resume = kwargs.get("resume_from_checkpoint", args[0] if args else None)
        if resume is True:
            resume = get_last_checkpoint(self.args.output_dir)
        self._dataflex_iterator = None
        self._selection_pending_state = None
        self._selection_stage_state = None
        if resume:
            path = Path(resume) / f"dataflex_selection_rank_{self.args.process_index}.json"
            if not path.exists():
                raise ValueError(
                    f"selection stage state is missing: {path}; cannot resume this selected loader safely"
                )
            state = json.loads(path.read_text())
            trainer_state = json.loads((Path(resume) / "trainer_state.json").read_text())
            if state.get("version") != 1 or state.get("run_signature") != self._selection_run_signature():
                raise ValueError("selection checkpoint dataset/algorithm/schedule/batch settings do not match")
            if state.get("global_step") != trainer_state["global_step"]:
                raise ValueError("selection checkpoint cursor does not match Trainer global_step")
            indices = state.get("indices")
            if (
                not isinstance(indices, list)
                or not indices
                or any(type(i) is not int or not 0 <= i < len(self.train_dataset) for i in indices)
                or type(state.get("batches_seen")) is not int
                or state["batches_seen"] < 0
            ):
                raise ValueError("invalid selection checkpoint indices/batch cursor")
            restore = getattr(self.selector, "restore_checkpoint_state", None)
            if restore is not None:
                restore(state.get("selector_state"), global_step=state["global_step"],
                        opened_step=state["opened_step"], active_indices=indices)
            self._selection_pending_state = state
        # Upstream must not skip its unrelated full-pool iterator as well.
        previous_skip = self.args.ignore_data_skip
        self.args.ignore_data_skip = True
        try:
            return super().train(*args, **kwargs)
        finally:
            self.args.ignore_data_skip = previous_skip

    def _dataflex_open_stage(self, indices):
        if self._checkpoint_selection_stage:
            step = self.state.global_step
            seed = self.args.seed if self.args.data_seed is None else self.args.data_seed
            self._selection_stage_seed = getattr(self, "_selection_restoring_seed", int(seed) + step)
            self._selection_stage_state = {
                "indices": list(indices),
                "opened_step": step,
                "sampler_seed": self._selection_stage_seed,
                "batches_seen": 0,
            }
        return super()._dataflex_open_stage(indices)

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        pending = getattr(self, "_selection_pending_state", None)
        if pending is not None:
            self._selection_pending_state = None
            self._selection_stage_seed = pending["sampler_seed"]
            # Opening normally seeds with the current step; resume needs the
            # seed of the step at which this subset was originally opened.
            self._selection_restoring_seed = pending["sampler_seed"]
            python_rng, numpy_rng = random.getstate(), np.random.get_state()
            try:
                with torch.random.fork_rng():
                    self._dataflex_open_stage(pending["indices"])
                    for _ in range(pending["batches_seen"]):
                        next(self._dataflex_iterator)
            finally:
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
                del self._selection_restoring_seed
            self._selection_stage_state = {
                key: pending[key] for key in ("indices", "opened_step", "sampler_seed", "batches_seen")
            }
        batches, count = super().get_batch_samples(epoch_iterator, num_batches, device)
        if self._checkpoint_selection_stage:
            self._selection_stage_state["batches_seen"] += len(batches)
        return batches, count

    def _save_checkpoint(self, model, trial, *args, **kwargs):
        super()._save_checkpoint(model, trial, *args, **kwargs)
        if self._checkpoint_selection_stage and self._selection_stage_state is not None:
            output = Path(self._get_output_dir(trial=trial)) / f"checkpoint-{self.state.global_step}"
            write_json(
                output / f"dataflex_selection_rank_{self.args.process_index}.json",
                {
                    "version": 1,
                    "global_step": self.state.global_step,
                    "run_signature": self._selection_run_signature(),
                    **self._selection_stage_state,
                    "selector_state": self.selector.export_checkpoint_state(
                        global_step=self.state.global_step,
                        opened_step=self._selection_stage_state["opened_step"],
                        active_indices=self._selection_stage_state["indices"],
                    ) if hasattr(self.selector, "export_checkpoint_state") else None,
                },
            )
