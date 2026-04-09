
import os
from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np
from datasets import Dataset, load_dataset, load_from_disk

from llamafactory.extras.constants import FILEEXT2TYPE
from llamafactory.extras.misc import check_version, has_tokenized_data
from llamafactory.data.converter import align_dataset
from llamafactory.data.data_utils import get_dataset_module, merge_dataset, read_cloud_json, split_dataset
from llamafactory.data.parser import get_dataset_list
from llamafactory.data.processor import (
    FeedbackDatasetProcessor,
    PackedSupervisedDatasetProcessor,
    PairwiseDatasetProcessor,
    PretrainDatasetProcessor,
    SupervisedDatasetProcessor,
    UnsupervisedDatasetProcessor,
)
from llamafactory.data.loader import _get_merged_dataset, _get_preprocessed_dataset

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import PreTrainedTokenizer, ProcessorMixin, Seq2SeqTrainingArguments

    from llamafactory.hparams import DataArguments, ModelArguments
    from llamafactory.data.data_utils import DatasetModule
    from llamafactory.data.parser import DatasetAttr
    from llamafactory.data.processor import DatasetProcessor
    from llamafactory.data.template import Template

import logging
import sys
logging.basicConfig(level=logging.INFO)
handler = logging.StreamHandler(sys.stdout)
logger = logging.getLogger(__name__)
logger.addHandler(handler)

from ..dataset.mixed_proportion_manager import MixedProportionManager

def get_dataset(
    template: "Template",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
) -> "DatasetModule":
    r"""Get the train dataset and optionally gets the evaluation dataset."""
    logger.info_rank0(f"[DATAFLEX DEBUG] DataFlex get_dataset called with stage={stage}")
    logger.info_rank0(f"[DATAFLEX DEBUG] data_args.dataset={data_args.dataset}")
    logger.info_rank0(f"[DATAFLEX DEBUG] data_args.mixture_sample_rule={getattr(data_args, 'mixture_sample_rule', 'NOT_SET')}")
    logger.info_rank0(f"[DATAFLEX DEBUG] data_args.init_mixture_proportions={getattr(data_args, 'init_mixture_proportions', 'NOT_SET')}")
    # Load tokenized dataset if path exists
    if data_args.tokenized_path is not None:
        if has_tokenized_data(data_args.tokenized_path):
            logger.warning_rank0("Loading dataset from disk will ignore other data arguments.")
            tokenized_data = load_from_disk(data_args.tokenized_path)
            dataset_module = get_dataset_module(tokenized_data)
            if data_args.streaming:
                dataset_module["train_dataset"] = dataset_module["train_dataset"].to_iterable_dataset()

            logger.info_rank0(f"Loaded tokenized dataset from {data_args.tokenized_path}.")
            return dataset_module

        if data_args.streaming:
            raise ValueError("Turn off `streaming` when saving dataset to disk.")

    # Load and preprocess dataset
    with training_args.main_process_first(desc="load dataset", local=(not data_args.data_shared_file_system)):
        dataset = _get_merged_dataset(data_args.dataset, model_args, data_args, training_args, stage)
        eval_dataset = _get_merged_dataset(
            data_args.eval_dataset,
            model_args,
            data_args,
            training_args,
            stage,
            return_dict=data_args.eval_on_each_dataset,
        )
        per_source_raw = None
        logger.info_rank0("[Dataflex] Mixture enabled: building per-source raw datasets for dynamic mixing.")
        per_source_raw = _get_merged_dataset(
            data_args.dataset, model_args, data_args, training_args, stage, return_dict=True
        )
        logger.info_rank0(f"[Dataflex] Loaded per-source raw datasets: {list(per_source_raw.keys())} "
                            f"(num_sources={len(per_source_raw)})")

    with training_args.main_process_first(desc="pre-process dataset", local=(not data_args.data_shared_file_system)):
        dataset = _get_preprocessed_dataset(
            dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=False
        )
        if isinstance(eval_dataset, dict):
            for eval_name, eval_data in eval_dataset.items():
                eval_dataset[eval_name] = _get_preprocessed_dataset(
                    eval_data, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
                )
        else:
            eval_dataset = _get_preprocessed_dataset(
                eval_dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
            )

        dataset_dict = split_dataset(dataset, eval_dataset, data_args, seed=training_args.seed)
        if data_args.tokenized_path is not None:  # save tokenized dataset to disk
            if training_args.should_save:
                dataset_dict.save_to_disk(data_args.tokenized_path)
                logger.info_rank0(f"Tokenized dataset is saved at {data_args.tokenized_path}.")
                logger.info_rank0(f"Please launch the training with `tokenized_path: {data_args.tokenized_path}`.")

        dataset_module = get_dataset_module(dataset_dict)

    logger.info_rank0("[Dataflex] Preprocessing per-source datasets for dynamic mixing...")
    per_source_pp = {
        name: _get_preprocessed_dataset(
            ds, data_args, training_args, stage, template, tokenizer, processor, is_eval=False
        )
        for name, ds in per_source_raw.items()
    }
    sizes_str = {name: len(ds) for name, ds in per_source_pp.items()}
    logger.info_rank0(f"[Dataflex] Per-source preprocessed sizes: {sizes_str}")

    # 打印初始比例配置
    logger.info_rank0(f"[Dataflex] sample_rule={data_args.mixture_sample_rule} | "
                        f"proportions={data_args.init_mixture_proportions} | "
                        f"seed={training_args.seed}")

    manager = MixedProportionManager(
        per_source=per_source_pp,
        sample_rule=data_args.mixture_sample_rule,
        proportions=data_args.init_mixture_proportions,
        seed=training_args.seed,
        logger=logger,
    )

    # ── Load independent eval datasets for mixer (e.g. gate load evaluation) ──
    mixer_eval_names = getattr(data_args, 'mixer_eval_dataset', None)
    if mixer_eval_names:
        logger.info_rank0(f"[Dataflex] Loading mixer eval datasets: {mixer_eval_names}")
        with training_args.main_process_first(desc="load mixer eval dataset", local=(not data_args.data_shared_file_system)):
            mixer_eval_raw = _get_merged_dataset(
                mixer_eval_names, model_args, data_args, training_args, stage, return_dict=True
            )
        mixer_eval_pp = {}
        for name, ds in mixer_eval_raw.items():
            mixer_eval_pp[name] = _get_preprocessed_dataset(
                ds, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
            )
        # Map eval dataset names back to training domain names:
        # e.g. "code_eval" -> "code", so dynamic_moe_mixer can look up by domain name
        mixer_eval_by_domain = {}
        for eval_name, eval_ds in mixer_eval_pp.items():
            domain = eval_name.replace("_eval", "")
            mixer_eval_by_domain[domain] = eval_ds
            logger.info_rank0(f"[Dataflex] Mixer eval: '{eval_name}' -> domain '{domain}' ({len(eval_ds)} samples)")
        manager.mixer_eval_datasets = mixer_eval_by_domain

    # 可选：把 manager 留给外部（方便在 callback 里重建）
    # 例如附在 dataset_module 上（Trainer 不会用到这个字段）
    dataset_module["train_dataset"] = None # 先占位，trainer里会rebuild
    dataset_module["mixture_manager"] = manager
    logger.info_rank0("[Dataflex] Exposed mixture_manager for runtime re-mixing.")

    return dataset_module
