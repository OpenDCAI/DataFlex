# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dynamic selection: a component picks which samples the next interval trains on.

The schedule is a repeating stage of `warmup_step + update_step * update_times`
optimizer steps. It opens with a warmup draw, then re-selects every `update_step`
steps. `ReorderTrainer` and `LegoTrainer` reuse all of it by presenting themselves
as a selector.
"""

import functools
import json
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from torch.utils.data import DataLoader, RandomSampler
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.trainer_utils import get_last_checkpoint, has_length, seed_worker
from transformers.utils import is_datasets_available
from typing_extensions import override

# Registers the built-in selectors into REGISTRY as a side effect.
import dataflex.train.selector  # noqa: F401
from dataflex.core.registry import REGISTRY
from dataflex.train.hooks import DataflexTrainerMixin, group_by_length_requested
from dataflex.utils.load_component import load_component
from dataflex.utils.logging import logger
from dataflex.utils.multimodal_selection import fingerprint, write_json


if is_datasets_available():
    import datasets


class SelectTrainer(DataflexTrainerMixin, CustomSeq2SeqTrainer):
    def __init__(self, finetuning_args, processor=None, gen_kwargs=None, **kwargs):
        super().__init__(finetuning_args=finetuning_args, processor=processor, gen_kwargs=gen_kwargs, **kwargs)
        name = finetuning_args.component_name
        # Params of the selector (can substitute ${output_dir})
        sel_params = load_component(
            'selectors',
            finetuning_args.components_cfg_file,
            name,
            runtime_vars={}
        )

        # Supply the runtime dependencies; static components ignore what they do not need.
        runtime = dict(
            dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            accelerator=self.accelerator,
            data_collator=self.data_collator,
        )

        self.selector = REGISTRY.build("selector", name, runtime=runtime, cfg=sel_params)
        if self._checkpoint_selection_stage and (
            finetuning_args.warmup_step < 0 or finetuning_args.update_step <= 0
        ):
            raise ValueError("Adapt-inf requires warmup_step >= 0 and update_step > 0")
        logger.info(f"[SelectTrainer] selector={name}, params={sel_params}")
        logger.info("[Dataflex] SelectTrainer initialized")

    # ------------------------------------------------------------------
    # Schedule
    # ------------------------------------------------------------------

    @property
    def _total_train_batch_size(self) -> int:
        return self.get_total_train_batch_size(self.args)

    @property
    def _warmup_samples(self) -> int:
        return self._total_train_batch_size * self.finetuning_args.warmup_step

    @property
    def _stage_steps(self) -> Optional[int]:
        """Length of one warmup-then-select stage, in optimizer steps."""
        fa = self.finetuning_args
        if fa.update_times > 0:
            return max(1, fa.warmup_step + fa.update_step * fa.update_times)
        return None

    @override
    def dataflex_step_budget(self) -> Optional[int]:
        fa = self.finetuning_args
        if fa.train_step > 0:
            if self.args.num_train_epochs != 1:
                logger.warning("[Dataflex] train_step is set; num_train_epochs will be ignored.")
            return fa.train_step

        stage_steps = self._stage_steps
        if stage_steps is not None:
            return int(np.ceil(self.args.num_train_epochs * stage_steps))

        # No selection schedule given: fall back to one pass over the dataset.
        steps_per_epoch = max(1, int(np.ceil(len(self.train_dataset) / self._total_train_batch_size)))
        return int(np.ceil(self.args.num_train_epochs * steps_per_epoch))

    @override
    def dataflex_initial_indices(self) -> Optional[List[int]]:
        fa = self.finetuning_args
        dataset_size = len(self.train_dataset)

        # A stage cannot draw more than the dataset holds.
        update_samples = self._total_train_batch_size * fa.update_step
        if update_samples > dataset_size:
            raise ValueError(
                f"[Dataflex] update_step draws {update_samples} samples but the dataset holds {dataset_size}. "
                "Lower `update_step` or the batch size."
            )
        if self._warmup_samples > dataset_size:
            raise ValueError(
                f"[Dataflex] warmup_step draws {self._warmup_samples} samples but the dataset holds {dataset_size}. "
                "Lower `warmup_step` or the batch size."
            )

        logger.info(
            f"[Dataflex] warmup_step={fa.warmup_step}, {self._warmup_samples} warmup samples in total"
        )
        return self.selector.warmup(self._warmup_samples, replacement=True)

    @override
    def dataflex_next_indices(self, model, step_id: int) -> Optional[List[int]]:
        fa = self.finetuning_args
        stage_steps = self._stage_steps
        budget = self.state.max_steps

        if step_id >= budget or (step_id == 0 and not (self._checkpoint_selection_stage and fa.warmup_step == 0)):
            return None

        step_in_stage = step_id % stage_steps if stage_steps is not None else step_id

        # Stage rollover: draw a fresh warmup subset.
        if stage_steps is not None and step_in_stage == 0 and not (
            self._checkpoint_selection_stage and fa.warmup_step == 0
        ):
            self._dataflex_sync()
            return self.selector.warmup(self._warmup_samples, replacement=True)

        # Interval boundary: ask the selector.
        at_boundary = step_in_stage == fa.warmup_step or (
            step_in_stage > fa.warmup_step
            and (step_in_stage - fa.warmup_step) % fa.update_step == 0
        )
        if not at_boundary:
            return None

        self._dataflex_sync()

        current_update_times = (step_in_stage - fa.warmup_step) // fa.update_step + 1
        effective_update_times = fa.update_times
        if effective_update_times <= 0 and stage_steps is not None:
            effective_update_times = max(
                1,
                int(np.ceil(max(stage_steps - fa.warmup_step, 0) / fa.update_step)),
            )

        logger.info(
            f"[Dataflex] Model training paused, starting the {current_update_times}th dynamic data selection..."
        )
        indices = self.selector.select(
            model=model,
            step_id=step_id,
            num_samples=self._total_train_batch_size * fa.update_step,
            optimizer_state=self.optimizer.state,
            scheduler_state=self.lr_scheduler.state_dict(),
            current_update_times=current_update_times,
            update_times=effective_update_times,
            tokenizer=self.processing_class,
        )
        if self.accelerator.is_main_process:
            logger.info(
                f"[Dataflex] Updated dataloader at step {step_id}, {len(indices)} samples in total."
            )
        return indices

    def _dataflex_sync(self) -> None:
        """Line the ranks up before a component looks at the model."""
        self.accelerator.wait_for_everyone()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    @override
    def _get_train_sampler(self, train_dataset=None) -> Optional[torch.utils.data.Sampler]:
        if train_dataset is None:
            train_dataset = self.train_dataset
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(train_dataset)
        if train_dataset is None or not has_length(train_dataset):
            return None

        if group_by_length_requested(self.args):
            if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
                lengths = (
                    train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = (
                self.processing_class.model_input_names[0] if self.processing_class is not None else None
            )
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
                generator=torch.Generator().manual_seed(getattr(self, "_selection_stage_seed", self.args.seed))
                if self._checkpoint_selection_stage else None,
            )

        if self._checkpoint_selection_stage:
            from accelerate.data_loader import SeedableRandomSampler

            seed = getattr(self, "_selection_stage_seed", self.args.seed)
            return SeedableRandomSampler(train_dataset, data_seed=seed)
        return RandomSampler(train_dataset)

    @override
    def get_train_dataloader(self, indices: Optional[List[int]] = None) -> DataLoader:
        """Training DataLoader, restricted to `indices` when given."""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if indices is not None:
            train_dataset = torch.utils.data.Subset(train_dataset, indices)

        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        # `_remove_unused_columns` may have replaced the collator, so the selector
        # needs the resolved one to build its own scoring loaders.
        self.selector.data_collator = data_collator

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        if self._checkpoint_selection_stage:
            dataloader_params["generator"] = torch.Generator().manual_seed(
                getattr(self, "_selection_stage_seed", self.args.seed)
            )

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler(train_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            if self.args.dataloader_num_workers > 0:
                # seed_worker takes (worker_id, num_workers, rank); DataLoader only passes worker_id.
                dataloader_params["worker_init_fn"] = functools.partial(
                    seed_worker,
                    num_workers=self.args.dataloader_num_workers,
                    rank=self.args.process_index,
                )
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    # Selection checkpoints retain the selected indices and batch cursor on resume.
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
