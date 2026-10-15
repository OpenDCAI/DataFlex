"""Image-budget offline reference/IRS/DINOv2/NC pipeline.

Budget is explicitly in unique images. Image-to-instruction expansion is recorded,
not silently treated as an equal instruction budget.

For the paper's LLaVA unique_idx record budget, use record_presel_selector.
Grouping then expanding instructions is not the official LLaVA 15% selection.
"""

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from dataflex.utils.multimodal_selection import (
    allocate_budget,
    cluster_features,
    file_fingerprint,
    fingerprint,
    neighbor_centrality,
    write_json,
)


def load_records(path):
    records = json.loads(Path(path).read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("input must be a nonempty JSON array")
    return records


def resolve_image_path(image_root, image, parent_cache, path_cache):
    """Resolve shared directory symlinks once instead of statting each ancestor per row."""
    source = Path(image_root) / image
    if source not in path_cache:
        parent = source.parent
        if parent not in parent_cache:
            parent_cache[parent] = parent.resolve()
        path = parent_cache[parent] / source.name
        path_cache[source] = path.resolve() if path.is_symlink() else path
    return path_cache[source]


def image_groups(records, image_root, task_field="task", cross_task_policy="reject"):
    """Group duplicate instructions by image; preserve original dataset positions.

    A missing taxonomy, text-only row, multi-image row, bad image, or overlapping
    task labels fails with its position. Resolving those requires an explicit
    dataset adapter; the generic pipeline never invents task labels.
    """
    from PIL import Image

    if cross_task_policy not in ("reject", "task-image"):
        raise ValueError("cross_task_policy must be reject or task-image")
    groups = {}
    validated_paths = {}
    parent_cache, path_cache = {}, {}
    for index, record in enumerate(records):
        task = record.get(task_field)
        if task is None or not str(task).strip():
            raise ValueError(f"row {index}: missing task field {task_field!r}")
        images = record.get("images", record.get("image"))
        images = [images] if isinstance(images, str) else images
        if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
            raise ValueError(
                f"row {index}: exactly one image path is required; text-only/multi-image need a dataset adapter"
            )
        path = resolve_image_path(image_root, images[0], parent_cache, path_cache)
        physical_path = str(path)
        image_id = (
            json.dumps([str(task), physical_path], ensure_ascii=False)
            if cross_task_policy == "task-image"
            else physical_path
        )
        if image_id not in groups:
            if physical_path not in validated_paths:
                try:
                    with Image.open(path) as image:
                        image.verify()
                except (OSError, ValueError) as exc:
                    raise ValueError(f"row {index}: invalid image {images[0]!r}: {exc}") from exc
                validated_paths[physical_path] = path.stat()
            stat = validated_paths[physical_path]
            groups[image_id] = {
                "task": str(task),
                "indices": [],
                "path": physical_path,
                "file_size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        if groups[image_id]["task"] != str(task):
            raise ValueError(
                f"row {index}: one image has multiple tasks; an explicit cross-task mapping policy is required"
            )
        groups[image_id]["indices"].append(index)
    return groups


def choose_reference(groups, count, seed):
    tasks = sorted({group["task"] for group in groups.values()})
    by_task = [[key for key, group in groups.items() if group["task"] == task] for task in tasks]
    if count < len(tasks):
        raise ValueError("reference count must cover every task")
    # Guarantee a reference image for each task, then allocate the remaining
    # budget by task size. This small-task safeguard is an explicit adaptation.
    quotas = allocate_budget([len(ids) - 1 for ids in by_task], count - len(tasks), [len(ids) for ids in by_task])
    rng = np.random.default_rng(seed)
    return [str(key) for ids, quota in zip(by_task, quotas) for key in rng.choice(ids, quota + 1, replace=False)]


def task_importance(records, groups, reference_ids, scores, temperature=None):
    expected = {index for key in reference_ids for index in groups[key]["indices"]}
    by_index = {}
    for score in scores:
        index = score["index"]
        if index in by_index or index not in expected:
            raise ValueError("reference scores contain duplicate or unexpected dataset indices")
        with_q, without_q = float(score["loss_with_question"]), float(score["loss_without_question"])
        if not math.isfinite(with_q) or not math.isfinite(without_q) or with_q < 0 or without_q <= 1e-12:
            raise ValueError(f"row {index}: IRS requires finite losses and a positive denominator")
        if score.get("answer_tokens", 0) <= 0 or score.get("answers_match") is not True:
            raise ValueError("IRS scores must certify identical nonempty response token supervision")
        by_index[index] = with_q / without_q
    if set(by_index) != expected:
        raise ValueError("reference score coverage does not match the exact reference instruction set")
    # Average instructions within an image, then images within each task. This
    # preserves the paper's image unit when one image has several instructions.
    per_task = defaultdict(list)
    for key in reference_ids:
        group = groups[key]
        per_task[group["task"]].append(float(np.mean([by_index[index] for index in group["indices"]])))
    tasks = sorted({group["task"] for group in groups.values()})
    if set(per_task) != set(tasks):
        raise ValueError("reference set must cover every task")
    means = np.array([np.mean(per_task[task]) for task in tasks])
    temperature = 1 / math.sqrt(len(tasks)) if temperature is None else float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    logits = -means / temperature
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    return {
        task: {"mean_irs": float(mean), "weight": float(weight), "reference_images": len(per_task[task])}
        for task, mean, weight in zip(tasks, means, weights)
    }, temperature


def select_images(
    groups,
    features,
    image_ids,
    importance,
    reference_ids,
    total_budget,
    include_reference,
    seed,
    cluster_fraction=0.01,
):
    if not 0 < cluster_fraction <= 1:
        raise ValueError("cluster_fraction must be in (0, 1]")
    image_ids = list(image_ids)
    if len(set(image_ids)) != len(image_ids) or set(image_ids) != set(groups):
        raise ValueError("embedding image IDs must exactly match the input image pool")
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or len(features) != len(image_ids) or not np.isfinite(features).all():
        raise ValueError("embedding features must be a finite matrix aligned to image_ids")
    if (np.linalg.norm(features, axis=1) < 1e-12).any():
        raise ValueError("zero image embeddings are invalid")
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    reference = set(reference_ids)
    if len(reference) != len(reference_ids) or not reference <= set(groups):
        raise ValueError("reference IDs must be unique members of the image pool")
    selected = list(reference_ids) if include_reference else []
    budget = total_budget - len(selected)
    tasks = sorted(importance)
    if set(tasks) != {group["task"] for group in groups.values()}:
        raise ValueError("task importance and data taxonomy differ")
    available = [
        [i for i, key in enumerate(image_ids) if groups[key]["task"] == task and key not in reference]
        for task in tasks
    ]
    quotas = allocate_budget([len(rows) for rows in available], budget, [importance[task]["weight"] for task in tasks])
    details = []
    for task, rows, quota in zip(tasks, available, quotas):
        # Match official clustering over the whole task, excluding reference
        # members only when ranking available images within each cluster.
        all_rows = [i for i, key in enumerate(image_ids) if groups[key]["task"] == task]
        clusters = max(1, int(len(all_rows) * cluster_fraction))
        labels = cluster_features(features[all_rows], clusters, seed, minibatch=False)
        labels_by_row = dict(zip(all_rows, labels.tolist()))
        cluster_ids = sorted({labels_by_row[row] for row in rows})
        members = [[row for row in rows if labels_by_row[row] == label] for label in cluster_ids]
        capacities = [len(group) for group in members]
        cluster_quotas = allocate_budget(capacities, quota, capacities)
        for label, group, cluster_budget in zip(cluster_ids, members, cluster_quotas):
            nc = neighbor_centrality(features[group])
            order = np.argsort(-nc, kind="stable")
            chosen = [image_ids[group[i]] for i in order[:cluster_budget]]
            selected.extend(chosen)
            details.append(
                {
                    "task": task,
                    "cluster": label,
                    "size": len(group),
                    "budget": cluster_budget,
                    "selected_image_ids": chosen,
                    "nc": [float(nc[i]) for i in order[:cluster_budget]],
                }
            )
    if len(selected) != total_budget or len(set(selected)) != len(selected):
        raise RuntimeError("PreSel failed exact unique-image budget validation")
    indices = sorted(index for key in selected for index in groups[key]["indices"])
    return selected, indices, {"task_budgets": dict(zip(tasks, quotas)), "clusters": details}


def _messages(record):
    if "messages" in record:
        messages = record["messages"]
    elif "conversations" in record:
        roles = {"human": "user", "gpt": "assistant", "system": "system"}
        messages = [
            {"role": roles.get(item["from"], item["from"]), "content": item["value"]}
            for item in record["conversations"]
        ]
    elif "question" in record and "response" in record:
        messages = [
            {"role": "user", "content": record["question"]},
            {"role": "assistant", "content": record["response"]},
        ]
    else:
        raise ValueError("score needs messages, conversations, or question/response fields")
    if not messages or messages[-1]["role"] != "assistant" or not any(item["role"] == "user" for item in messages):
        raise ValueError("score requires user/assistant conversations ending with an assistant response")
    if any(
        item["role"] not in ("system", "user", "assistant") or not isinstance(item["content"], str)
        for item in messages
    ):
        raise ValueError("score requires text conversation contents")
    dialogue = messages[1:] if messages[0]["role"] == "system" else messages
    if len(dialogue) % 2 or any(
        item["role"] != ("user" if i % 2 == 0 else "assistant") for i, item in enumerate(dialogue)
    ):
        raise ValueError("conversation must alternate user/assistant with an optional initial system message")
    return messages


def export_training_dataset(output, stem, records, groups, indices, dataset_name):
    """Export a registered ShareGPT dataset without changing original ID space.

    Raw records/indices remain separate artifacts. Absolute media paths avoid
    dependence on the training job's dataset_dir or working directory.
    """
    output = Path(output)
    by_index = {
        index: (key, group["task"], group["path"])
        for key, group in groups.items()
        for index in group["indices"]
    }
    training, mapping = [], []
    for position, index in enumerate(indices):
        key, task, physical_path = by_index[index]
        messages = [dict(item) for item in _messages(records[index])]
        for item in messages:
            item["content"] = item["content"].replace("<image>", "")
        next(item for item in messages if item["role"] == "user")["content"] = (
            "<image>\n" + next(item for item in messages if item["role"] == "user")["content"]
        )
        training.append({"messages": messages, "images": [physical_path]})
        mapping.append({"training_position": position, "source_position": index, "image_id": key, "task": task})
    filename = f"{stem}_training.json"
    write_json(output / filename, training)
    write_json(output / f"{stem}_training_index_map.json", mapping)
    registry_path = output / "dataset_info.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    registry[dataset_name] = {
        "file_name": filename,
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }
    write_json(registry_path, registry)
    return {
        "dataset": dataset_name,
        "dataset_dir": str(output.resolve()),
        "training_file": filename,
        "index_map": f"{stem}_training_index_map.json",
        "training_fingerprint": fingerprint(training),
    }


def llava_paired_inputs(processor, record, image_path, without_question=False, cutoff_len=None):
    """Use the same LLaMA-Factory LLaVA template/expansion as SFT.

    Encoding turn segments separately matters for Vicuna's SentencePiece
    boundaries. A generic full-string prefix mask can otherwise disagree with
    the training labels, especially when the question is removed.
    """
    import torch
    from llamafactory.data.processor.processor_utils import infer_seqlen
    from llamafactory.data.template import TEMPLATES

    template = TEMPLATES["llava"]
    messages, image_added = [], False
    for item in _messages(record):
        text = item["content"].replace("<image>", "")
        if item["role"] == "user":
            if without_question:
                text = ""
            if not image_added:
                text = "<image>\n" + text
                image_added = True
        messages.append({"role": item["role"], "content": text})
    images = [image_path]
    messages = template.mm_plugin.process_messages(messages, images, [], [], processor)
    encoded = list(template.encode_multiturn(processor.tokenizer, messages))
    budget_pairs = encoded
    if without_question and cutoff_len is not None:
        original = []
        image_added = False
        for item in _messages(record):
            text = item["content"].replace("<image>", "")
            if item["role"] == "user" and not image_added:
                text = "<image>\n" + text
                image_added = True
            original.append({"role": item["role"], "content": text})
        original = template.mm_plugin.process_messages(original, images, [], [], processor)
        budget_pairs = list(template.encode_multiturn(processor.tokenizer, original))
    ids, labels = [], []
    budget_used = 0
    for (prompt, answer), (budget_prompt, budget_answer) in zip(encoded, budget_pairs):
        if answer != budget_answer:
            raise ValueError("question removal changed encoded response")
        if cutoff_len is not None:
            if budget_used >= cutoff_len:
                break
            source_len, target_len = infer_seqlen(len(budget_prompt), len(budget_answer), cutoff_len - budget_used)
            prompt, answer = prompt[:source_len], answer[:target_len]
            budget_used += source_len + target_len
        ids.extend(prompt + answer)
        labels.extend([-100] * len(prompt) + answer)
    if cutoff_len is not None:
        image_id = processor.tokenizer.convert_tokens_to_ids("<image>")
        expected = sum(pair[0].count(image_id) for pair in encoded)
        if ids.count(image_id) != expected or not any(label != -100 for label in labels):
            raise ValueError("LLaVA scoring cutoff removed image tokens or all response supervision")
    inputs = template.mm_plugin.get_mm_inputs(images, [], [], [1], [0], [0], [ids], processor)
    inputs.update(
        input_ids=torch.tensor([ids]),
        labels=torch.tensor([labels]),
        attention_mask=torch.ones(1, len(ids), dtype=torch.long),
    )
    return inputs


def paired_inputs(processor, record, image_path, without_question=False, cutoff_len=None):
    """Use the model chat template and supervise only assistant spans.

    Prefix agreement and paired response equality are checked explicitly; a
    template that cannot support these invariants needs a dedicated adapter.
    """
    import torch

    if type(processor).__name__ == "LlavaProcessor":
        return llava_paired_inputs(processor, record, image_path, without_question, cutoff_len)
    if cutoff_len is not None:
        raise ValueError("paired scoring cutoff currently requires the LLaVA adapter")

    messages, image_added = [], False
    for item in _messages(record):
        text = item["content"].replace("<image>", "")
        if item["role"] == "user" and without_question:
            text = ""
        content = [{"type": "text", "text": text}]
        if item["role"] == "user" and not image_added:
            content.insert(0, {"type": "image", "url": image_path})
            image_added = True
        messages.append({"role": item["role"], "content": content})

    def encode(history, generation=False):
        return processor.apply_chat_template(
            history, tokenize=True, add_generation_prompt=generation, return_dict=True, return_tensors="pt"
        )

    inputs = dict(encode(messages))
    ids = inputs["input_ids"]
    labels = torch.full_like(ids, -100)
    for i, item in enumerate(messages):
        if item["role"] != "assistant":
            continue
        prompt = encode(messages[:i], generation=True)["input_ids"]
        completed = encode(messages[: i + 1])["input_ids"]
        start, stop = prompt.shape[1], completed.shape[1]
        if stop <= start or not torch.equal(completed[:, :start], prompt) or not torch.equal(ids[:, :stop], completed):
            raise ValueError(
                "chat template does not preserve prefix alignment; a dedicated scoring adapter is required"
            )
        labels[:, start:stop] = ids[:, start:stop]
    inputs["labels"] = labels
    return inputs


def run_reference(args):
    if getattr(args, "budget_unit", "image") == "record":
        from .record_presel_pipeline import run_reference as run_record
        return run_record(args)
    records = load_records(args.data)
    policy = getattr(args, "cross_task_policy", "reject")
    groups = image_groups(records, args.image_root, args.task_field, policy)
    reference = choose_reference(groups, args.count, args.seed)
    indices = sorted(index for key in reference for index in groups[key]["indices"])
    output = Path(args.output_dir)
    write_json(output / "reference_ids.json", reference)
    write_json(output / "reference_indices.json", indices)
    write_json(output / "reference_data.json", [records[i] for i in indices])
    training = export_training_dataset(output, "reference", records, groups, indices, "presel_reference")
    write_json(
        output / "reference_manifest.json",
        {
            "seed": args.seed,
            "budget_unit": "task_image" if policy == "task-image" else "image",
            "cross_task_policy": policy,
            "image_count": len(reference),
            "instruction_count": len(indices),
            "pool_fingerprint": fingerprint(groups),
            "data_fingerprint": fingerprint(records),
            "requires_reference_training": True,
            "training_dataset": training,
        },
    )


def run_extract(args):
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    records = load_records(args.data)
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if getattr(args, "budget_unit", "image") == "record":
        from .record_presel_pipeline import record_groups
        groups = record_groups(records, args.image_root, args.task_field)
    else:
        groups = image_groups(records, args.image_root, args.task_field, getattr(args, "cross_task_policy", "reject"))
    processor = AutoImageProcessor.from_pretrained(args.model, revision=args.revision)
    model = AutoModel.from_pretrained(args.model, revision=args.revision).to(args.device).eval()
    if getattr(model.config, "model_type", "") != "dinov2":
        raise ValueError("PreSel requires a DINOv2 encoder")
    ids = list(groups)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    embeddings = np.lib.format.open_memmap(
        output / "features.npy", mode="w+", dtype=np.float32, shape=(len(ids), model.config.hidden_size)
    )
    rows_by_path = defaultdict(list)
    for row, key in enumerate(ids):
        rows_by_path[groups[key]["path"]].append(row)
    physical_paths = list(rows_by_path)
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(physical_paths), args.batch_size):
            images = []
            batch_paths = physical_paths[start : start + args.batch_size]
            for path in batch_paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt").to(args.device)
            features = model(**inputs).last_hidden_state[:, 0].float().cpu().numpy()
            for path, feature in zip(batch_paths, features):
                embeddings[rows_by_path[path]] = feature
            if start == 0 or start % (args.batch_size * 100) == 0 or start + len(images) == len(physical_paths):
                embeddings.flush()
                write_json(output / "progress.json", {
                    "completed_physical_images": start + len(images), "total_physical_images": len(physical_paths),
                    "seconds": time.monotonic() - started,
                })
    embeddings.flush()
    write_json(output / "image_ids.json", ids)
    write_json(
        output / "manifest.json",
        {
            "encoder": "dinov2",
            "budget_unit": getattr(args, "budget_unit", "image"),
            "model": args.model,
            "revision": args.revision,
            "resolved_revision": getattr(model.config, "_commit_hash", None),
            "pool_fingerprint": fingerprint(groups),
            "processor": processor.to_dict(),
            "feature": "last_layer_normalized_cls",
            "physical_images_encoded": len(physical_paths),
            "features_fingerprint": file_fingerprint(output / "features.npy"),
            "image_ids_fingerprint": fingerprint(ids),
        },
    )


def validate_reference_training(manifest, reference_ids, reference_model):
    required = {"checkpoint", "reference_ids_fingerprint", "optimizer_steps", "gpu_hours", "config"}
    if not isinstance(manifest, dict) or not required <= set(manifest):
        raise ValueError(
            "reference training manifest requires checkpoint, reference_ids_fingerprint, optimizer_steps, gpu_hours, config"
        )
    if manifest["checkpoint"] != reference_model or manifest["reference_ids_fingerprint"] != fingerprint(
        reference_ids
    ):
        raise ValueError("reference training manifest does not match this checkpoint/reference set")
    if (
        manifest["optimizer_steps"] <= 0
        or not math.isfinite(manifest["gpu_hours"])
        or manifest["gpu_hours"] < 0
        or not manifest["config"]
    ):
        raise ValueError("reference training steps/config/cost must be recorded")


def load_reference_model(checkpoint, revision=None, dtype="auto"):
    """Load adapters through PEFT so saved full projectors are restored.

    Transformers' automatic adapter loading can remap LLaVA checkpoint keys
    while missing PEFT modules_to_save keys. Loading the base then PEFT avoids
    silently replacing the trained reference projector with fresh parameters.
    """
    import torch
    from transformers import AutoModelForImageTextToText
    from transformers.utils.peft_utils import find_adapter_config_file

    adapter_path = find_adapter_config_file(checkpoint, revision=revision)
    if adapter_path is None:
        return AutoModelForImageTextToText.from_pretrained(checkpoint, revision=revision, dtype=dtype)
    from peft import PeftConfig, PeftModel
    from peft.utils.save_and_load import get_peft_model_state_dict, load_peft_weights

    config = PeftConfig.from_pretrained(checkpoint, revision=revision)
    base = AutoModelForImageTextToText.from_pretrained(
        config.base_model_name_or_path, revision=config.revision, dtype=dtype
    )
    model = PeftModel.from_pretrained(base, checkpoint, revision=revision, is_trainable=False)
    # Verify the full saved modules, not just a finite forward pass: a missing
    # projector can still produce finite but invalid reference scores.
    saved = load_peft_weights(checkpoint, device="cpu", revision=revision)
    restored = get_peft_model_state_dict(model)
    verified = 0
    for key, value in saved.items():
        if any(f".{name}." in key for name in config.modules_to_save or []):
            current = restored.get(key)
            if current is None or not torch.equal(current.detach().cpu(), value.to(current.dtype)):
                raise ValueError(f"reference adapter did not restore saved module tensor: {key}")
            verified += 1
    if config.modules_to_save and not verified:
        raise ValueError("reference adapter declares full saved modules but contains none")
    model._dataflex_saved_module_verification = {"verified_tensors": verified, "checkpoint": checkpoint}
    return model


def run_score(args):
    if getattr(args, "budget_unit", "image") == "record":
        from .record_presel_pipeline import run_score as run_record
        return run_record(args)
    import torch
    from transformers import AutoProcessor

    from dataflex.utils.multimodal_scoring import answer_statistics, assert_same_answers, move_to_device

    records = load_records(args.data)
    groups = image_groups(records, args.image_root, args.task_field, getattr(args, "cross_task_policy", "reject"))
    reference = json.loads(Path(args.reference_ids).read_text())
    if len(set(reference)) != len(reference) or not set(reference) <= set(groups):
        raise ValueError("invalid reference image IDs")
    training_manifest = json.loads(Path(args.reference_training_manifest).read_text())
    validate_reference_training(training_manifest, reference, args.reference_model)
    processor = AutoProcessor.from_pretrained(args.reference_model, revision=args.revision)
    minimum = getattr(args, "image_min_pixels", None)
    maximum = getattr(args, "image_max_pixels", None)
    if minimum is not None or maximum is not None:
        image_processor = getattr(processor, "image_processor", None)
        size = dict(getattr(image_processor, "size", {}))
        if not {"shortest_edge", "longest_edge"} <= size.keys():
            raise ValueError("pixel limits require a Qwen-VL image processor with pixel-area size bounds")
        size["shortest_edge"] = minimum if minimum is not None else size["shortest_edge"]
        size["longest_edge"] = maximum if maximum is not None else size["longest_edge"]
        if not 0 < size["shortest_edge"] <= size["longest_edge"]:
            raise ValueError("image pixel limits must be positive and min <= max")
        image_processor.size = size
        # Slow and fast Qwen processors expose different compatibility fields.
        if hasattr(image_processor, "min_pixels"):
            image_processor.min_pixels = size["shortest_edge"]
        if hasattr(image_processor, "max_pixels"):
            image_processor.max_pixels = size["longest_edge"]
    dtype = getattr(args, "dtype", "auto")
    if dtype != "auto":
        dtype = getattr(torch, dtype)
    model = load_reference_model(args.reference_model, args.revision, dtype).to(args.device).eval()
    results = []
    started = time.monotonic()
    with torch.inference_mode():
        completed = 0
        total = sum(len(groups[key]["indices"]) for key in reference)
        for key in reference:
            for index in groups[key]["indices"]:
                path = groups[key]["path"]
                cutoff = getattr(args, "cutoff_len", None)
                with_q = move_to_device(paired_inputs(processor, records[index], path, cutoff_len=cutoff), args.device)
                without_q = move_to_device(paired_inputs(processor, records[index], path, True, cutoff), args.device)
                assert_same_answers(with_q["labels"], without_q["labels"])
                losses = [
                    answer_statistics(model(**inputs).logits, inputs["labels"]) for inputs in (with_q, without_q)
                ]
                results.append(
                    {
                        "index": index,
                        "loss_with_question": float(losses[0]["loss"]),
                        "loss_without_question": float(losses[1]["loss"]),
                        "answer_tokens": losses[0]["answer_tokens"],
                        "answers_match": True,
                    }
                )
                completed += 1
                if completed == 1 or completed % 100 == 0 or completed == total:
                    write_json(Path(args.output_dir) / "progress.json", {
                        "completed": completed, "total": total, "seconds": time.monotonic() - started,
                    })
    output = Path(args.output_dir)
    write_json(output / "reference_scores.json", results)
    write_json(
        output / "score_manifest.json",
        {
            "reference_model": args.reference_model,
            "revision": args.revision,
            "resolved_revision": getattr(model.config, "_commit_hash", None),
            "reference_ids_fingerprint": fingerprint(reference),
            "model_dtype": str(model.dtype),
            "cutoff_len": getattr(args, "cutoff_len", None),
            "saved_module_verification": getattr(model, "_dataflex_saved_module_verification", None),
            "image_processor": processor.image_processor.to_dict(),
            "pool_fingerprint": fingerprint(groups),
            "data_fingerprint": fingerprint(records),
            "scoring_seconds": time.monotonic() - started,
            "reference_training_manifest": training_manifest,
            "scores_fingerprint": file_fingerprint(output / "reference_scores.json"),
        },
    )


def run_select(args):
    if getattr(args, "budget_unit", "image") == "record":
        from .record_presel_pipeline import run_select as run_record
        return run_record(args)
    started = time.monotonic()
    records = load_records(args.data)
    policy = getattr(args, "cross_task_policy", "reject")
    groups = image_groups(records, args.image_root, args.task_field, policy)
    reference = json.loads(Path(args.reference_ids).read_text())
    score_manifest = json.loads(Path(args.score_manifest).read_text())
    feature_dir = Path(args.features_dir)
    feature_manifest = json.loads((feature_dir / "manifest.json").read_text())
    if feature_manifest.get("encoder") != "dinov2":
        raise ValueError("features must identify a DINOv2 encoder")
    for manifest in (score_manifest, feature_manifest):
        if manifest.get("pool_fingerprint") != fingerprint(groups):
            raise ValueError("cached artifacts do not match the image/task pool")
    if score_manifest.get("reference_ids_fingerprint") != fingerprint(reference) or score_manifest.get(
        "data_fingerprint"
    ) != fingerprint(records):
        raise ValueError("score cache does not match reference IDs or instruction content")
    validate_reference_training(
        score_manifest.get("reference_training_manifest"), reference, score_manifest.get("reference_model")
    )
    if score_manifest.get("scores_fingerprint") != file_fingerprint(args.scores):
        raise ValueError("reference score file does not match its manifest")
    scores = json.loads(Path(args.scores).read_text())
    importance, temperature = task_importance(records, groups, reference, scores, args.temperature)
    image_ids = json.loads((feature_dir / "image_ids.json").read_text())
    if feature_manifest.get("features_fingerprint") != file_fingerprint(
        feature_dir / "features.npy"
    ) or feature_manifest.get("image_ids_fingerprint") != fingerprint(image_ids):
        raise ValueError("embedding files do not match their manifest")
    features = np.load(feature_dir / "features.npy", mmap_mode="r")
    selected, indices, stats = select_images(
        groups,
        features,
        image_ids,
        importance,
        reference,
        args.budget,
        args.reference_budget == "included",
        args.seed,
        args.cluster_fraction,
    )
    output = Path(args.output_dir)
    write_json(output / "selected_indices.json", indices)
    write_json(output / "selected_data.json", [records[i] for i in indices])
    write_json(output / "selected_image_ids.json", selected)
    write_json(output / "task_importance.json", importance)
    write_json(output / "selection_stats.json", stats)
    training = export_training_dataset(output, "selected", records, groups, indices, "presel_selected")
    write_json(
        output / "manifest.json",
        {
            "method": "presel",
            "mode": "flex-adapted-offline",
            "budget_unit": "task_image" if policy == "task-image" else "unique_image",
            "cross_task_policy": policy,
            "physical_image_count": len({groups[key]["path"] for key in selected}),
            "image_count": len(selected),
            "instruction_count": len(indices),
            "reference_budget": args.reference_budget,
            "reference_image_count": len(reference),
            "reference_instruction_count": sum(len(groups[key]["indices"]) for key in reference),
            "temperature": temperature,
            "seed": args.seed,
            "cluster_fraction": args.cluster_fraction,
            "data_fingerprint": fingerprint(records),
            "pool_fingerprint": fingerprint(groups),
            "feature_manifest": feature_manifest,
            "score_manifest": score_manifest,
            "selection_seconds": time.monotonic() - started,
            "training_dataset": training,
            "adaptations": [
                "exact capacity-aware rounding",
                "instructions averaged per image",
                "single offline round; no new instruction generation",
                "single image placed in first user turn for registered ShareGPT training",
            ],
        },
    )


def run_training_manifest(args):
    checkpoint = Path(args.checkpoint)
    trainer = json.loads((checkpoint/"trainer_state.json").read_text())
    if trainer.get("global_step", 0) <= 0 or abs(trainer.get("epoch", 0) - 1) > 1e-6:
        raise ValueError("Reference training must complete one epoch before IRS")
    ids = json.loads(Path(args.reference_ids).read_text())
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Invalid reference IDs")
    config = json.loads(Path(args.training_config).read_text())
    if Path(config["output_dir"]).resolve() != checkpoint.resolve():
        raise ValueError("Training config output_dir does not match reference checkpoint")
    value = {"checkpoint":args.checkpoint, "reference_ids_fingerprint":fingerprint(ids),
             "optimizer_steps":trainer["global_step"], "gpu_hours":args.gpu_hours, "config":config}
    validate_reference_training(value, ids, args.checkpoint)
    write_json(args.output_file, value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, prog="dataflex-cli presel" if argv is not None else None)
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser("training-manifest", help="Certify completed reference training")
    manifest.add_argument("--reference-ids", required=True)
    manifest.add_argument("--checkpoint", required=True)
    manifest.add_argument("--training-config", required=True, help="The JSON config used for this reference run")
    manifest.add_argument("--gpu-hours", type=float, required=True)
    manifest.add_argument("--output-file", required=True)
    manifest.set_defaults(run=run_training_manifest)
    for name, function in [
        ("reference", run_reference),
        ("extract", run_extract),
        ("score", run_score),
        ("select", run_select),
    ]:
        command = subparsers.add_parser(name)
        command.set_defaults(run=function)
        command.add_argument("--budget-unit", choices=["record", "image"], default="record",
                             help="record: official LLaVA 5% reference / 15% final; image: legacy expansion")
        command.add_argument("--data", required=True)
        command.add_argument("--image-root", required=True)
        command.add_argument("--task-field", default="task")
        command.add_argument(
            "--cross-task-policy", choices=["reject", "task-image"], default="reject",
            help="task-image preserves each official task membership of a shared physical image; budgets count task/image pairs",
        )
        command.add_argument("--output-dir", required=True)
        if name in ("reference", "select"):
            command.add_argument("--seed", type=int, default=42)
        if name == "reference":
            command.add_argument("--count", type=int, help="record: fixed 5%; image: required image count")
        if name in ("extract", "score"):
            command.add_argument("--revision")
            command.add_argument("--device", default="cpu")
        if name == "extract":
            command.add_argument("--model", required=True, help="DINOv2 checkpoint")
            command.add_argument("--batch-size", type=int, default=8)
        if name in ("score", "select"):
            command.add_argument("--reference-ids", required=True)
        if name == "score":
            command.add_argument("--cutoff-len", type=int, help="LLaVA paired response truncation using with-question lengths")
            command.add_argument("--reference-model", required=True, help="model AFTER reference fine-tuning")
            command.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
            command.add_argument("--image-min-pixels", type=int, help="Qwen-VL minimum image pixel area")
            command.add_argument("--image-max-pixels", type=int, help="Qwen-VL maximum image pixel area")
            command.add_argument(
                "--reference-training-manifest",
                required=True,
                help="JSON recording reference IDs/config/checkpoint/cost",
            )
        if name == "select":
            command.add_argument("--scores", required=True)
            command.add_argument("--score-manifest", required=True)
            command.add_argument("--features-dir", required=True)
            command.add_argument("--budget", type=int, help="record: fixed 15%; image: required image budget")
            command.add_argument("--reference-budget", choices=["included", "additional"], default="included")
            command.add_argument("--temperature", type=float)
            command.add_argument("--cluster-fraction", type=float, default=0.01)
    args = parser.parse_args(argv)
    if getattr(args, "budget_unit", None) == "image":
        if args.command == "reference" and args.count is None:
            parser.error("image budget mode requires --count")
        if args.command == "select" and args.budget is None:
            parser.error("image budget mode requires --budget")
    args.run(args)


if __name__ == "__main__":
    main()
