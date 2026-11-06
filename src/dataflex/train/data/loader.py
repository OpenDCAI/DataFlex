
import json
import os
from collections.abc import Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Literal, Optional, Union

import numpy as np
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from torch.utils.data import Dataset as TorchDataset

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

#: Written last, so a half-finished snapshot is never mistaken for a usable one.
_SNAPSHOT_MANIFEST = "dataflex_tokenized.json"
_SNAPSHOT_FORMAT = "dataflex-domains-v1"


def _split_dataset(dataset, eval_dataset, data_args, seed) -> "DatasetDict":
    """LlamaFactory >=0.9.5 returns `(train_dict, eval_dict)`; older versions return a `DatasetDict`.

    Without this, `get_dataset_module` takes the tuple through its untyped
    `else: # single dataset` branch and hands it back as `train_dataset`, which
    raises nothing and yields a 2-sample training set.
    """
    out = split_dataset(dataset, eval_dataset, data_args, seed=seed)
    if isinstance(out, tuple):
        return DatasetDict({**out[0], **out[1]})
    return out


def _snapshot_signature(model_args, data_args, stage) -> dict:
    """The inputs that decide what the tokenized bytes are.

    Reusing a snapshot produced under a different tokenizer or `cutoff_len`
    would train on silently wrong data, and reusing one whose `dataset:` list
    has a different order is worse still: `init_mixture_proportions` is aligned
    by position, so every domain would get another domain's weight. Both paths
    therefore refuse a snapshot whose signature does not match, rather than
    following LlamaFactory's "ignore other data arguments" warning.
    """
    return {
        "format": _SNAPSHOT_FORMAT,
        "stage": stage,
        "dataset": list(data_args.dataset or []),
        "template": data_args.template,
        "cutoff_len": data_args.cutoff_len,
        "packing": bool(data_args.packing),
        "max_samples": data_args.max_samples,
        "tokenizer": model_args.model_name_or_path,
    }


def _snapshot_ready(tokenized_path: Optional[str]) -> bool:
    return bool(tokenized_path) and os.path.isfile(os.path.join(tokenized_path, _SNAPSHOT_MANIFEST))


def _save_tokenized_domains(tokenized_path: str, per_source_pp: dict, eval_dataset, signature: dict) -> None:
    """Persist the per-domain datasets, keeping their order structural.

    Domains go to `domains/0`, `domains/1`, ... and the manifest holds the names
    in that order. Deriving order from a directory listing would put it at the
    mercy of string sorting, which is exactly the mistake that misaligns
    proportions.
    """
    os.makedirs(tokenized_path, exist_ok=True)
    names = list(per_source_pp.keys())
    for i, name in enumerate(names):
        per_source_pp[name].save_to_disk(os.path.join(tokenized_path, "domains", str(i)))

    if isinstance(eval_dataset, dict):
        eval_layout = list(eval_dataset.keys())
        for name, ds in eval_dataset.items():
            ds.save_to_disk(os.path.join(tokenized_path, "eval", name))
    elif eval_dataset is not None:
        eval_layout = "single"
        eval_dataset.save_to_disk(os.path.join(tokenized_path, "eval"))
    else:
        eval_layout = None

    payload = dict(signature)
    payload["domains"] = names
    payload["sizes"] = [len(per_source_pp[n]) for n in names]
    payload["eval"] = eval_layout
    with open(os.path.join(tokenized_path, _SNAPSHOT_MANIFEST), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info_rank0(
        f"[Dataflex] tokenized snapshot saved to {tokenized_path} "
        f"(domains={names}, sizes={payload['sizes']}). Reuse it by keeping the same "
        f"`tokenized_path` in the config."
    )


def _load_tokenized_domains(tokenized_path: str, signature: dict):
    """Read a snapshot back, or refuse it if it was made for something else."""
    with open(os.path.join(tokenized_path, _SNAPSHOT_MANIFEST), encoding="utf-8") as f:
        manifest = json.load(f)

    mismatch = {
        key: (manifest.get(key), value) for key, value in signature.items() if manifest.get(key) != value
    }
    if mismatch:
        details = "; ".join(f"{k}: snapshot={old!r} config={new!r}" for k, (old, new) in mismatch.items())
        raise ValueError(
            f"[Dataflex] the tokenized snapshot at {tokenized_path} was produced under different "
            f"settings and cannot be reused ({details}). Point `tokenized_path` somewhere else, or "
            f"delete that directory to rebuild it."
        )

    names = manifest["domains"]
    per_source_pp = {
        name: load_from_disk(os.path.join(tokenized_path, "domains", str(i)))
        for i, name in enumerate(names)
    }
    for name, expected in zip(names, manifest.get("sizes", [])):
        if len(per_source_pp[name]) != expected:
            raise ValueError(
                f"[Dataflex] snapshot domain '{name}' holds {len(per_source_pp[name])} rows but the "
                f"manifest says {expected}; the directory looks incomplete."
            )

    layout = manifest.get("eval")
    if layout is None:
        eval_dataset = None
    elif layout == "single":
        eval_dataset = load_from_disk(os.path.join(tokenized_path, "eval"))
    else:
        eval_dataset = {name: load_from_disk(os.path.join(tokenized_path, "eval", name)) for name in layout}

    logger.info_rank0(
        f"[Dataflex] loaded tokenized snapshot from {tokenized_path} "
        f"(domains={names}, sizes={manifest.get('sizes')}); skipping load and tokenization."
    )
    return per_source_pp, eval_dataset


def _merged_train_set(per_source_pp, data_args):
    """The concatenated training set, built only if something will read it.

    `split_dataset` is the sole consumer, and it only touches the train side when
    `val_size > 0`. Otherwise the merged set becomes `dataset_dict["train"]`,
    which `get_dataset` immediately replaces with None so the trainer can rebuild
    from the mixture manager. Returning None in that case is what removes an
    entire tokenization pass over the corpus.

    When a split is wanted, the domains are concatenated *after* tokenizing
    rather than before. Under packing the two are not identical -- blocks are
    formed within a map batch, so merging first shifts boundaries where one
    domain ends and the next begins -- which is acceptable for a held-out slice
    and avoids paying for the corpus twice.
    """
    if data_args.val_size <= 1e-6:
        return None

    if data_args.mix_strategy != "concat":
        logger.warning_rank0(
            f"[Dataflex] `val_size` with mix_strategy='{data_args.mix_strategy}' is not "
            f"expressible on the mixture path; carving the validation split from a plain "
            f"concatenation of the domains instead."
        )
    logger.warning_rank0(
        "[Dataflex] `val_size` splits a validation set out of the merged corpus, but the "
        "mixer keeps sampling from the full per-domain datasets, so those rows are still "
        "trained on. Prefer a separate `eval_dataset`."
    )
    return concatenate_datasets(list(per_source_pp.values()))


def get_dataset(
    template: "Template",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
) -> "DatasetModule":
    r"""Get the train dataset and optionally gets the evaluation dataset.

    The mixture path trains on the *per-domain* datasets: `MixedProportionManager`
    samples from them every time proportions change, and `train_dataset` below is
    handed to the trainer as None precisely so the trainer rebuilds from it. So
    the domains are the only tokenized copy that has a consumer, and this
    function walks the corpus exactly once to produce them.
    """
    signature = _snapshot_signature(model_args, data_args, stage)

    if _snapshot_ready(data_args.tokenized_path):
        per_source_pp, eval_dataset = _load_tokenized_domains(data_args.tokenized_path, signature)
    else:
        if data_args.tokenized_path is not None and data_args.streaming:
            raise ValueError("Turn off `streaming` when saving dataset to disk.")

        with training_args.main_process_first(desc="load dataset", local=(not data_args.data_shared_file_system)):
            # One load of the training corpus, kept split by domain. A second
            # merged load would repeat `align_dataset` -- a full row-wise map --
            # for every domain, and `overwrite_cache: true` in the pre-training
            # configs stops the datasets cache from absorbing it.
            per_source_raw = _get_merged_dataset(
                data_args.dataset, model_args, data_args, training_args, stage, return_dict=True
            )
            if not per_source_raw:
                raise ValueError("train_type: dynamic_mix requires at least one dataset in `dataset`.")
            logger.info_rank0(f"[Dataflex] Loaded per-source raw datasets: {list(per_source_raw.keys())} "
                                f"(num_sources={len(per_source_raw)})")
            eval_dataset = _get_merged_dataset(
                data_args.eval_dataset,
                model_args,
                data_args,
                training_args,
                stage,
                return_dict=data_args.eval_on_each_dataset,
            )

        with training_args.main_process_first(desc="pre-process dataset", local=(not data_args.data_shared_file_system)):
            logger.info_rank0("[Dataflex] Preprocessing per-source datasets for dynamic mixing...")
            per_source_pp = {
                name: _get_preprocessed_dataset(
                    ds, data_args, training_args, stage, template, tokenizer, processor, is_eval=False
                )
                for name, ds in per_source_raw.items()
            }
            sizes_str = {name: len(ds) for name, ds in per_source_pp.items()}
            logger.info_rank0(f"[Dataflex] Per-source preprocessed sizes: {sizes_str}")

            if isinstance(eval_dataset, dict):
                for eval_name, eval_data in eval_dataset.items():
                    eval_dataset[eval_name] = _get_preprocessed_dataset(
                        eval_data, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
                    )
            else:
                eval_dataset = _get_preprocessed_dataset(
                    eval_dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
                )

        if data_args.tokenized_path is not None and training_args.should_save:
            _save_tokenized_domains(data_args.tokenized_path, per_source_pp, eval_dataset, signature)

    dataset_dict = _split_dataset(
        _merged_train_set(per_source_pp, data_args), eval_dataset, data_args, seed=training_args.seed
    )
    dataset_module = get_dataset_module(dataset_dict)

    # Print initial proportion configuration
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

    # Optional: expose manager to external code (for callback reconstruction)
    # e.g. attached to dataset_module (Trainer doesn't use this field)
    dataset_module["train_dataset"] = None # Placeholder, trainer will rebuild
    dataset_module["mixture_manager"] = manager
    logger.info_rank0("[Dataflex] Exposed mixture_manager for runtime re-mixing.")

    return dataset_module


# ======================================================================
# Reordering
# ======================================================================


@contextmanager
def _capture_raw_scores(score_field: str):
    """Snapshot a score column from each raw dataset before it is discarded.

    `align_dataset` is the last thing `_load_single_dataset` does, and it maps
    with `remove_columns=column_names`, so any score field in the raw JSONL dies
    there. Wrapping it lets us read the column while reusing LlamaFactory's
    loading path verbatim.

    The capture happens after `num_samples` / `max_samples` truncation, so the
    captured vector lines up with the rows that survive, and datasets are
    captured in `dataset:` order, which is the order `concat` merges them in.
    """
    import llamafactory.data.loader as lf_loader

    original = lf_loader.align_dataset
    captured: List[Optional[np.ndarray]] = []

    def wrapper(dataset, dataset_attr, data_args, training_args):
        try:
            column_names = getattr(dataset, "column_names", None) or []
            if score_field in column_names:
                captured.append(np.asarray(dataset[score_field], dtype=np.float64))
            else:
                captured.append(None)
                logger.warning_rank0(
                    f"[Dataflex][Reorder] dataset '{dataset_attr}' has no field '{score_field}' "
                    f"(available: {list(column_names)})"
                )
        except Exception as exc:  # never let score capture break dataset loading
            captured.append(None)
            logger.warning_rank0(f"[Dataflex][Reorder] could not read '{score_field}': {exc}")
        return original(dataset, dataset_attr, data_args, training_args)

    lf_loader.align_dataset = wrapper
    try:
        yield captured
    finally:
        lf_loader.align_dataset = original


def _concat_raw_scores(captured: List[Optional[np.ndarray]], score_field: str) -> np.ndarray:
    if not captured or any(part is None for part in captured):
        raise ValueError(
            f"[Dataflex][Reorder] score field '{score_field}' is missing from at least one dataset. "
            f"Either add it to every source, or switch the reorder to "
            f"`apply_at: index` with an explicit `score_path`."
        )
    return np.concatenate(captured, axis=0)


def make_reorder_get_dataset(reorder_factory):
    """Build a `get_dataset` that permutes raw rows before tokenization.

    Why here and not in the trainer: the score lives in the raw JSONL and is
    deleted during preprocessing, and preprocessing is not index-preserving
    (malformed rows are dropped, packing merges rows). Permuting the raw dataset
    sidesteps the mapping entirely, because both `align_dataset` and
    `_get_preprocessed_dataset` preserve order, so whatever survives keeps its
    relative position.

    Args:
        reorder_factory: zero-arg callable returning a reorder exposing
            `order_rows(scores) -> permutation` and a `score_params` dict.
    """

    def reorder_get_dataset(
        template: "Template",
        model_args: "ModelArguments",
        data_args: "DataArguments",
        training_args: "Seq2SeqTrainingArguments",
        stage: Literal["pt", "sft", "rm", "ppo", "kto"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"] = None,
    ) -> "DatasetModule":
        if data_args.tokenized_path is not None and has_tokenized_data(data_args.tokenized_path):
            logger.warning_rank0(
                "[Dataflex][Reorder] loading an already tokenized dataset; its stored order is used as is. "
                "Use a distinct `tokenized_path` per ordering variant."
            )
            tokenized_data = load_from_disk(data_args.tokenized_path)
            dataset_module = get_dataset_module(tokenized_data)
            if data_args.streaming:
                dataset_module["train_dataset"] = dataset_module["train_dataset"].to_iterable_dataset()
            return dataset_module

        if data_args.streaming:
            raise ValueError("[Dataflex][Reorder] reordering requires `streaming: false`.")

        reorder = reorder_factory()
        score_field = reorder.score_params.get("score_field", "score")

        with training_args.main_process_first(desc="load dataset", local=(not data_args.data_shared_file_system)):
            with _capture_raw_scores(score_field) as captured:
                dataset = _get_merged_dataset(data_args.dataset, model_args, data_args, training_args, stage)

            eval_dataset = _get_merged_dataset(
                data_args.eval_dataset,
                model_args,
                data_args,
                training_args,
                stage,
                return_dict=data_args.eval_on_each_dataset,
            )

            if dataset is not None:
                score_path = reorder.score_params.get("score_path")
                if score_path:
                    from ..reorder.score_provider import PrecomputedScoreProvider

                    scores = np.asarray(
                        PrecomputedScoreProvider(
                            score_path=score_path, score_field=score_field, expected_size=len(dataset)
                        ).scores,
                        dtype=np.float64,
                    )
                else:
                    scores = _concat_raw_scores(captured, score_field)

                if len(scores) != len(dataset):
                    raise ValueError(
                        f"[Dataflex][Reorder] captured {len(scores)} scores but the merged dataset has "
                        f"{len(dataset)} rows. Check `mix_strategy` (use 'concat') and that every source "
                        f"carries '{score_field}'."
                    )

                permutation = reorder.order_rows(scores)
                dataset = dataset.select(permutation)
                logger.info_rank0(
                    f"[Dataflex][Reorder] applied '{reorder.pattern}' to {len(permutation)} raw rows "
                    f"before preprocessing."
                )

        with training_args.main_process_first(
            desc="pre-process dataset", local=(not data_args.data_shared_file_system)
        ):
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

            if data_args.val_size > 1e-6:
                logger.warning_rank0(
                    "[Dataflex][Reorder] `val_size > 0` splits with a shuffle, which destroys the ordering. "
                    "Use a separate `eval_dataset` instead."
                )

            dataset_dict = _split_dataset(dataset, eval_dataset, data_args, seed=training_args.seed)
            if data_args.tokenized_path is not None and training_args.should_save:
                dataset_dict.save_to_disk(data_args.tokenized_path)
                logger.info_rank0(f"[Dataflex][Reorder] tokenized dataset saved at {data_args.tokenized_path}.")

            dataset_module = get_dataset_module(dataset_dict)

        train_size = len(dataset_module["train_dataset"]) if dataset_module.get("train_dataset") is not None else 0
        logger.info_rank0(f"[Dataflex][Reorder] ordered training set ready: {train_size} samples.")
        return dataset_module

    return reorder_get_dataset


# ======================================================================
# Lego
# ======================================================================


SAMPLE_METADATA_KEYS = ("idx", "domain_id")
# Some multimodal collators attach this bookkeeping field themselves.
BATCH_METADATA_KEYS = (*SAMPLE_METADATA_KEYS, "__record_indices__")


def split_batch(batch):
    """Return shallow copies of model inputs and metadata, without a model-field allowlist.

    Labels and all vision/audio fields remain model inputs. Tensor objects are
    shared with the original batch; removing keys never mutates that batch.
    """
    return (
        {key: value for key, value in batch.items() if key not in BATCH_METADATA_KEYS},
        {key: value for key, value in batch.items() if key in BATCH_METADATA_KEYS},
    )


class MetadataCollator:
    """Let the original collator handle modalities, then restore row metadata.

    The wrapped collator must keep one output sequence per input row, in order.
    This is the contract of the preprocessed PT/SFT collators used by LEGO;
    collators which expand, pack or reorder rows here need a separate mapping.
    Eval/anchor batches without training IDs are valid. Partially labelled
    batches fail instead of silently misaligning IDs and model inputs.
    """

    def __init__(self, collator):
        self.collator = collator

    def __call__(self, features):
        metadata = {}
        for key in SAMPLE_METADATA_KEYS:
            present = [key in feature for feature in features]
            if any(present):
                if not all(present):
                    raise ValueError(f"Inconsistent {key} metadata within a batch")
                metadata[key] = torch.tensor([feature[key] for feature in features], dtype=torch.long)
        rows = [
            {key: value for key, value in feature.items() if key not in SAMPLE_METADATA_KEYS} for feature in features
        ]
        batch = dict(self.collator(rows))
        input_ids = batch.get("input_ids")
        if metadata and input_ids is not None and len(input_ids) != len(features):
            raise ValueError("LEGO collator changed the number of rows; sample IDs cannot be aligned")
        batch.update(metadata)
        return batch

    def __getattr__(self, name):
        # __dict__.get avoids recursion during unpickling in DataLoader workers.
        collator = self.__dict__.get("collator")
        if collator is None:
            raise AttributeError(name)
        try:
            return getattr(collator, name)
        except AttributeError:
            # Hugging Face's RemoveColumnsCollator does not forward attributes.
            delegate = getattr(collator, "data_collator", None)
            if delegate is None:
                raise
            return getattr(delegate, name)


class DomainLabeledDataset(TorchDataset):
    """Attach identity in the final, preprocessed training dataset's index space.

    Wrap once after concatenating domains, before Subset/sampling/reordering.
    Repeated draws keep the same idx. These IDs identify processed rows (packed
    sequences when packing is enabled), not raw records or content hashes, and
    are only stable for the same processed dataset and domain order.
    """

    def __init__(self, dataset, domain_ids):
        self.dataset = dataset
        self.domain_ids = np.asarray(domain_ids, dtype=np.int64)
        if self.domain_ids.ndim != 1 or len(self.domain_ids) != len(dataset):
            raise ValueError("domain_ids must be a one-dimensional label array aligned with the dataset")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        idx = int(idx)
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        item = self.dataset[idx]
        if not isinstance(item, Mapping):
            raise TypeError("LEGO requires dataset rows to be mappings")
        return {**item, "idx": idx, "domain_id": int(self.domain_ids[idx])}

    @property
    def column_names(self):
        names = getattr(self.dataset, "column_names", None)
        return list(dict.fromkeys([*names, *SAMPLE_METADATA_KEYS])) if names is not None else None


def lego_get_dataset(
    template: "Template",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
) -> "DatasetModule":
    r"""Build one concatenated training set plus a per-sample domain label.

    Reuses LlamaFactory's own loading and preprocessing, then concatenates the
    per-source datasets in the order they appear in `dataset:` so that domain
    index order matches `init_mixture_proportions`. Unlike the mixture path this
    returns a real `train_dataset`, samples nothing, and shuffles nothing —
    all of that is the pipeline's job.
    """
    if data_args.streaming:
        raise ValueError("[Dataflex][Lego] composition requires `streaming: false`.")

    signature = _snapshot_signature(model_args, data_args, stage)

    if _snapshot_ready(data_args.tokenized_path):
        # Same on-disk format the mixture path writes: domains kept separate and
        # ordered by the manifest. Lego concatenates them below, so a snapshot is
        # interchangeable between the two paths for the same tokenizer settings.
        per_source_pp, eval_dataset = _load_tokenized_domains(data_args.tokenized_path, signature)
    else:
        with training_args.main_process_first(desc="load dataset", local=(not data_args.data_shared_file_system)):
            per_source_raw = _get_merged_dataset(
                data_args.dataset, model_args, data_args, training_args, stage, return_dict=True
            )
            eval_dataset = _get_merged_dataset(
                data_args.eval_dataset,
                model_args,
                data_args,
                training_args,
                stage,
                return_dict=data_args.eval_on_each_dataset,
            )

        with training_args.main_process_first(desc="pre-process dataset", local=(not data_args.data_shared_file_system)):
            per_source_pp = {
                name: _get_preprocessed_dataset(
                    ds, data_args, training_args, stage, template, tokenizer, processor, is_eval=False
                )
                for name, ds in (per_source_raw or {}).items()
            }

            if isinstance(eval_dataset, dict):
                for eval_name, eval_data in eval_dataset.items():
                    eval_dataset[eval_name] = _get_preprocessed_dataset(
                        eval_data, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
                    )
            else:
                eval_dataset = _get_preprocessed_dataset(
                    eval_dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
                )

        if data_args.tokenized_path is not None and training_args.should_save:
            _save_tokenized_domains(data_args.tokenized_path, per_source_pp, eval_dataset, signature)

    domain_names = list(per_source_pp.keys())
    sizes = [len(per_source_pp[name]) for name in domain_names]
    domain_ids = np.concatenate(
        [np.full(n, i, dtype=np.int64) for i, n in enumerate(sizes)]
    ) if sizes else np.zeros(0, dtype=np.int64)

    if len(domain_names) == 1:
        train_dataset = per_source_pp[domain_names[0]]
    else:
        train_dataset = concatenate_datasets([per_source_pp[name] for name in domain_names])

    if data_args.val_size > 1e-6:
        raise ValueError(
            "[Dataflex][Lego] `val_size > 0` splits with a shuffle, which breaks the alignment "
            "between domain_ids and train_dataset. Use a separate `eval_dataset`."
        )

    dataset_dict = _split_dataset(train_dataset, eval_dataset, data_args, seed=training_args.seed)
    dataset_module = get_dataset_module(dataset_dict)

    plan = list(zip(domain_names, sizes))
    logger.info_rank0(
        f"[Dataflex][Lego] domains in `dataset:` order (name, rows): {plan}; total={sum(sizes)}"
    )

    # Assign IDs once in the concatenated, preprocessed index space, including
    # single-domain runs. Subsequent Subset views must never renumber them.
    dataset_module["train_dataset"] = DomainLabeledDataset(dataset_module["train_dataset"], domain_ids)
    logger.info_rank0("[Dataflex][Lego] train_dataset wrapped to carry idx and domain_id per sample.")

    dataset_module["domain_ids"] = domain_ids
    dataset_module["domain_names"] = domain_names
    return dataset_module
