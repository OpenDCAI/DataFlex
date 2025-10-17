# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import contextlib
import copy
import functools
import glob
import importlib.metadata
import os
import random
import re
import shutil
import numpy as np
from torch import nn
import time
import torch.distributed as dist
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, List
from torch.utils.data import Subset, DataLoader, IterableDataset
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union
from packaging import version
import numpy as np
import torch
from typing_extensions import override
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.packages import is_transformers_version_greater_than
from llamafactory.train.callbacks import SaveProcessorCallback
from llamafactory.train.trainer_utils import create_custom_optimizer, create_custom_scheduler
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer

from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS, default_hp_search_backend
from transformers.image_processing_utils import BaseImageProcessor
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.optimization import Adafactor, get_scheduler
from transformers.processing_utils import ProcessorMixin
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS,
    is_torch_greater_or_equal_than_2_3,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    ExportableState,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    EvalLoopContainer,
    IterableDatasetShard,
    LabelSmoother,
    LayerWiseDummyOptimizer,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
    remove_dummy_checkpoint,
    set_rng_state_for_device,
)
from transformers.trainer_utils import (
    TrainOutput,
    speed_metrics,
    seed_worker,
    has_length
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    is_accelerate_available,
    is_apex_available,
    is_apollo_torch_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_galore_torch_available,
    is_grokadamw_available,
    is_in_notebook,
    is_ipex_available,
    is_liger_kernel_available,
    is_lomo_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_schedulefree_available,
    is_torch_compile_available,
    is_torch_hpu_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_xla_available,
    is_torch_xpu_available,
    is_torchao_available,
    strtobool,
)

from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.quantization_config import QuantizationMethod

from dataflex.utils.load_component import load_component
from dataflex.core.registry import REGISTRY
import dataflex.train.selector

TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCALER_NAME = "scaler.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from llamafactory.hparams import FinetuningArguments
from dataflex.utils.logging import logger

if is_peft_available():
    from peft import PeftModel

def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False

if is_sagemaker_mp_enabled():
    # import smdistributed.modelparallel.torch as smp
    # from smdistributed.modelparallel import __version__ as SMP_VERSION

    # IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    # from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
    pass
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.state import AcceleratorState
    from accelerate.utils import (
        AutocastKwargs,
        DistributedDataParallelKwargs,
        DistributedType,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("1.3.0"):
        from accelerate.utils import TorchTensorParallelPlugin
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

if is_torch_xla_available():
    # import torch_xla.core.xla_model as xm
    # import torch_xla.debug.metrics as met
    # from torch_xla import __version__ as XLA_VERSION

    # IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(XLA_FSDPV2_MIN_VERSION)
    # if IS_XLA_FSDPV2_POST_2_2:
    #     import torch_xla.distributed.spmd as xs
    #     import torch_xla.runtime as xr
    pass
else:
    IS_XLA_FSDPV2_POST_2_2 = False


class SelectTrainer(CustomSeq2SeqTrainer):
    def __init__(self, finetuning_args, processor=None, gen_kwargs=None, **kwargs):
        # 初始化父类
        super().__init__(finetuning_args=finetuning_args, processor=processor, gen_kwargs=gen_kwargs, **kwargs)
        name = finetuning_args.component_name
        # 取该 selector 的 params（可替换 ${output_dir}）
        sel_params = load_component(
            'selectors',
            finetuning_args.components_cfg_file,
            name,
            runtime_vars={}
        )

        # 统一提供“动态运行期依赖”，静态类会自动忽略
        runtime = dict(
            dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            accelerator=self.accelerator,
            data_collator=self.data_collator,
        )

        # 实例化（无任何 if/else）
        self.selector = REGISTRY.build("selector", name, runtime=runtime, cfg=sel_params)
        logger.info(f"[SelectTrainer] selector={name}, params={sel_params}")
        logger.info("[Dataflex] SelectTrainer initialized")

    @override
    def _get_train_sampler(self, train_dataset) -> Optional[torch.utils.data.Sampler]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(train_dataset)
        if train_dataset is None or not has_length(train_dataset):
            return None

        # Build the sampler.
        if self.args.group_by_length:
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
            )

        else:
            return RandomSampler(train_dataset)

    @override
    def get_train_dataloader(self, indices: Optional[List[int]] = None) -> DataLoader:
        """
        返回训练 DataLoader。
        如果传入 indices，则在 train_dataset 上构造子集 DataLoader。
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if indices is not None:
            train_dataset = torch.utils.data.Subset(train_dataset, indices)
        # print(len(train_dataset))
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")
        
        self.selector.data_collator = data_collator

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler(train_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))


    # 这个函数也是分别在每个gpu上执行的
    @override
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        # 这个batchsize就是per_gpu batchsize!
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the initial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        # _train_batch_size = micro batch size
        # 这个是global batch size
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        if total_train_batch_size * self.finetuning_args.update_step > len(self.train_dataset):
            assert False, f"Total updated samples {total_train_batch_size * self.finetuning_args.update_step} is larger than dataset size {len(self.train_dataset)}"

        logger.info(f"[Dataflex] Dynamic training mode")
        total_warmup_samples = total_train_batch_size * self.finetuning_args.warmup_step

        if total_warmup_samples > len(self.train_dataset):
            assert False, f"Total warmup samples {total_warmup_samples} is larger than dataset size {len(self.train_dataset)}"

        logger.info(f"[Dataflex] Warmup step {self.finetuning_args.warmup_step}, warmup samples: {total_warmup_samples} in total")
        warmup_indices = self.selector.warmup(total_warmup_samples, replacement=True)
        train_dataloader = self.get_train_dataloader(warmup_indices)

        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)
        (
            num_train_epochs,
            num_update_steps_per_epoch, # 等于len_dataloader // acc (或len(dataset)/worldsize/microbatchsize/acc)
            num_examples, # 等于数据集长度
            num_train_samples, # 等于数据集长度 * epoch数
            epoch_based,
            len_dataloader, # 等于数据集长度/worldsize/micro_batchsize
            max_steps,
        ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)
        max_steps = (self.finetuning_args.warmup_step + self.finetuning_args.update_step * self.finetuning_args.update_times)
        epoch_based = False
        logger.info(f"[Dataflex]Set max train steps to {max_steps}")
        logger.info(f"[Dataflex]Set epoch_based = False")
        num_train_tokens = None

        # 这里是每个gpu的tokens
        if self.args.include_tokens_per_second:
            num_train_tokens = self.num_tokens(train_dataloader, None if epoch_based else max_steps)
            # If going by epochs, multiply tokens linearly
            if len_dataloader is not None and epoch_based:
                num_train_tokens *= args.num_train_epochs
            # Otherwise since its steps, we just multiply by grad accum
            else:
                num_train_tokens *= args.gradient_accumulation_steps

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # Can't delay optimizer creation when using FSDP2: https://github.com/huggingface/accelerate/blob/3f636d626063ffcf9a337c7d3624d61b7d187d59/src/accelerate/accelerator.py#L1404
        is_fsdp2 = self.is_fsdp_enabled and (getattr(self.accelerator.state.fsdp_plugin, "fsdp_version", 1) == 2)
        if is_fsdp2:
            delay_optimizer_creation = False

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        self.state.compute_steps(args, max_steps)

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if use_accelerator_prepare and self.is_fsdp_enabled:
            # In case of auto_find_batch_size=True
            # Remove FSDP wrapping from sub-models.
            self.model = unwrap_model(self.model, recursive=True)

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                # configure fsdp plugin for qlora if any
                self._fsdp_qlora_plugin_updates()
                if self.accelerator.mixed_precision != "fp8":
                    self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        self._load_scaler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Dynamic Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        for attr in ("model", "optimizer", "lr_scheduler"):
            setattr(self.callback_handler, attr, getattr(self, attr))
        self.callback_handler.train_dataloader = train_dataloader

        self.state.init_training_references(self, max_steps, num_train_epochs, trial)

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0, device=args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        learning_rate = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        # 放弃epoch逻辑，相当于只训练一个epoch，通过step来训练
        current_dataloader = train_dataloader
        epoch = 0
        if self.state.global_step < self.finetuning_args.warmup_step:
            logger.info("[Dataflex] Model warmup in progress...")
        # if hasattr(current_dataloader, "set_epoch"):
        #     current_dataloader.set_epoch(epoch)
        
        # Reset the past mems state at the beginning of each epoch if necessary.
        if args.past_index >= 0:
            self._past = None

        steps_in_epoch = max_steps * args.gradient_accumulation_steps
        self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

        if resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
            self._load_rng_state(resume_from_checkpoint)

        rng_to_sync = False
        steps_skipped = 0
        if steps_trained_in_current_epoch > 0:
            current_dataloader = skip_first_batches(current_dataloader, steps_trained_in_current_epoch)
            steps_skipped = steps_trained_in_current_epoch
            steps_trained_in_current_epoch = 0
            rng_to_sync = True

        step = -1
        current_iterator = iter(current_dataloader)
        # We chunkify the epoch iterator into gradient accumulation steps `n` batches
        remainder = num_examples % args.gradient_accumulation_steps
        if remainder == 0:
            remainder = args.gradient_accumulation_steps
        update_step = -1
        # 一个epoch中的模型总更新次数
        total_updates = steps_in_epoch // args.gradient_accumulation_steps + 1
        if args.gradient_accumulation_steps == 1:
            total_updates -= 1
        for _ in range(total_updates):
            update_step += 1
            # 当前应该拿到的batch数，一般情况是gradient_accumulation_steps，每这么多个batch反向传播一次梯度，每个batch有batch_size个样本
            num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder

            batch_samples, num_items_in_batch = self.get_batch_samples(current_iterator, num_batches, args.device)
            # 遍历当前批次的样本
            for i, inputs in enumerate(batch_samples):
                step += 1  # 每次迭代时增加全局步数

                # 判断是否达到同步步数，或者是当前epoch的最后一个步数
                do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
                
                # 由于我们使用了预取（prefetching），我们需要手动设置同步梯度
                self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                # 如果需要记录输入的token数量
                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")  # 获取模型的主输入名称（默认为input_ids）
                    
                    # 检查模型的输入是否包含主输入名称
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        # 计算当前输入的tokens数量，并将其加入到已看到的总token数中
                        input_tokens = inputs[main_input_name].numel()  # 计算当前输入的tokens数量
                        input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)  # 转换为张量
                        self.state.num_input_tokens_seen += self.accelerator.gather(input_tokens).sum().item()  # 累加已看到的token数量

                # 如果需要同步随机数生成器（用于恢复训练）
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)  # 从检查点加载随机数生成器的状态
                    rng_to_sync = False  # 重置同步标志

                # 如果恢复训练且当前epoch还有未训练的步数，跳过已训练的步骤
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1  # 减少剩余的训练步数
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)  # 更新已训练步数的进度条
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)  # 恢复检查点的随机数生成器状态
                    continue  # 跳过这次迭代，进入下一次迭代
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()  # 关闭已训练步数的进度条
                    steps_trained_progress_bar = None  # 重置进度条

                # 每当步数达到梯度累积步骤数时，执行一次同步操作
                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)  # 执行步骤开始的回调

                # 在生成训练时避免依赖`accelerator.accumulate`，显式设置是否同步
                context = (
                    functools.partial(self.accelerator.no_sync, model=model)  # 如果不是最后一个批次，则不进行同步
                    if i != len(batch_samples) - 1
                    and self.accelerator.distributed_type != DistributedType.DEEPSPEED
                    else contextlib.nullcontext  # 否则不使用同步
                )
                
                with context():  # 在非同步上下文中进行训练
                    tr_loss_step = self.training_step(model, inputs, num_items_in_batch)  # 执行一次训练步骤，返回该步的损失值

                # 检查损失是否为NaN或Infinity，如果是，使用之前的损失值替代
                if (
                    args.logging_nan_inf_filter
                    and not is_torch_xla_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)  # 如果损失为NaN或Inf，则使用平均损失
                else:
                    if tr_loss.device != tr_loss_step.device:
                        raise ValueError(
                            f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                        )  # 检查计算的损失是否在原始设备上
                    tr_loss = tr_loss + tr_loss_step  # 将当前步的损失加入总损失

                # 累加浮点数操作的数量
                self.current_flos += float(self.floating_point_ops(inputs))

                # step达到acc，同步梯度
                if do_sync_step:
                    # Since we perform prefetching, we need to manually set sync_gradients to True
                    self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        if is_sagemaker_mp_enabled() and args.fp16:
                            _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            _grad_norm = nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            _grad_norm = self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                        if (
                            is_accelerate_available()
                            and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                        ):
                            grad_norm = model.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if hasattr(grad_norm, "item"):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = _grad_norm

                    self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                    self.optimizer.step()

                    self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                    # get leaning rate before update
                    learning_rate = self._get_learning_rate()

                    if not self.accelerator.optimizer_step_was_skipped:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    # 同步精度然后反向传播，此时每个gpu上处理了per_gpu_batch_size * acc个数据，global_step+1
                    self.state.global_step += 1
                    self.state.epoch = (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    self._maybe_log_save_evaluate(
                        tr_loss,
                        grad_norm,
                        model,
                        trial,
                        epoch,
                        ignore_keys_for_eval,
                        start_time,
                        # learning_rate=learning_rate,
                    )

                    # 动态训练更新
                    if (
                        self.state.global_step < max_steps and (
                        self.state.global_step == self.finetuning_args.warmup_step or
                        (self.state.global_step > self.finetuning_args.warmup_step and
                        (self.state.global_step - self.finetuning_args.warmup_step) % self.finetuning_args.update_step == 0))
                    ):
                        self.accelerator.wait_for_everyone()
                        torch.cuda.empty_cache()
                        if dist.is_initialized():
                            dist.barrier()

                        current_update_times = (self.state.global_step - self.finetuning_args.warmup_step) // self.finetuning_args.update_step + 1
                        logger.info(f"[Dataflex] Model training paused, starting the {current_update_times}th dynamic data selection...")
                        # 这里传一些特定的selector参数
                        extra_args = dict(
                            optimizer_state=self.optimizer.state,
                            scheduler_state=self.lr_scheduler.state_dict(),
                            current_update_times=current_update_times,
                            update_times=self.finetuning_args.update_times
                        )
                        new_indices = self.selector.select(
                            model=model,
                            step_id=self.state.global_step,
                            num_samples=total_train_batch_size * self.finetuning_args.update_step,
                            **extra_args
                        )

                        # 每个进程根据 local_indices 构造 dataloader
                        train_loader = self.get_train_dataloader(indices=new_indices)
                        current_iterator = iter(train_loader)

                        if self.accelerator.is_main_process:
                            logger.info(f"[Dataflex] Updated dataloader at step {self.state.global_step}, {len(new_indices)} samples in total.")

                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                # PyTorch/XLA relies on the data loader to insert the mark_step for
                # each step. Since we are breaking the loop early, we need to manually
                # insert the mark_step here.
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            # We also need to break out of the nested loop
            if self.control.should_epoch_stop or self.control.should_training_stop:
                if is_torch_xla_available():
                    xm.mark_step()
                break
        if step < 0:
            logger.warning(
                "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                f" num_steps ({max_steps}) higher than the number of available samples."
            )
            self.control.should_training_stop = True

        self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
        self._maybe_log_save_evaluate(
            tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time
        )

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            if is_torch_xla_available():
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())
            else:
                logger.warning(
                    "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                    "configured. Check your training configuration if this is unexpected."
                )
            # if self.control.should_training_stop:
            #     break
        # 结束主训练循环
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)
