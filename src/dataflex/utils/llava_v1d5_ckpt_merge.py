#!/usr/bin/env python
"""Merge a LLaVA-1.5 LoRA adapter and normalize the checkpoint for evaluation.

LLaMA-Factory exports produced under Transformers 5.x store the CLIP vision
tower with flattened parameter names::

    vision_tower.embeddings.patch_embedding.weight
    vision_tower.encoder.layers.0.self_attn.q_proj.weight

vLLM and the Transformers 4.x evaluation stacks expect the classic nested
layout::

    vision_tower.vision_model.embeddings.patch_embedding.weight
    vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight

The language-model and projector parameters are identical in both layouts, so
only the vision tower has to be rewritten. The rewrite copies every tensor
unchanged and only renames keys, hence no numerical change is introduced.

Typical use::

    # merge LoRA, then convert the exported checkpoint for evaluation
    python -m dataflex.utils.llava_v1d5_ckpt_merge \
        --base /models/llava-v1.5-7b-hf \
        --adapter /runs/presel/checkpoint-732 \
        --out /runs/presel_merged/checkpoint-732

    # only post-process an existing checkpoint (a full fine-tune or the base model)
    python -m dataflex.utils.llava_v1d5_ckpt_merge --convert-only \
        --source /models/llava-v1.5-7b-hf \
        --out /models/llava-v1.5-7b-hf-vllm

The merge step must run in the environment used for training: the adapter's
``target_modules`` are Transformers 5 module paths (``model.language_model...``).
The conversion step is plain safetensors I/O and works in any environment with
``safetensors`` installed.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


INDEX_NAME = "model.safetensors.index.json"
SINGLE_WEIGHT_NAME = "model.safetensors"
FLAT_VISION_PREFIX = "vision_tower."
NESTED_VISION_PREFIX = "vision_tower.vision_model."
VOCAB_KEYS = ("language_model.model.embed_tokens.weight", "language_model.lm_head.weight")
# Non-weight files copied when the export did not carry them over from the base.
META_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "chat_template.jinja",
    "chat_template.json",
    "added_tokens.json",
    "special_tokens_map.json",
)


def is_flat_vision_key(key):
    """Whether ``key`` is a Transformers 5 flat vision-tower parameter name."""
    return key.startswith(FLAT_VISION_PREFIX) and not key.startswith(NESTED_VISION_PREFIX)


def nested_vision_key(key):
    """Map a flat vision-tower key to the nested layout expected by vLLM."""
    return NESTED_VISION_PREFIX + key[len(FLAT_VISION_PREFIX) :]


def _weight_files(ckpt):
    """Return ``(index or None, shard names)`` for a safetensors checkpoint."""
    ckpt = Path(ckpt)
    index_path = ckpt / INDEX_NAME
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = sorted(set(index["weight_map"].values()))
        return index, shards

    single = ckpt / SINGLE_WEIGHT_NAME
    if single.is_file():
        return None, [single.name]

    raise FileNotFoundError(
        f"{ckpt} contains neither {INDEX_NAME} nor {SINGLE_WEIGHT_NAME}; "
        "the checkpoint must be saved in safetensors format."
    )


def _key_set(ckpt):
    index, shards = _weight_files(ckpt)
    if index is not None:
        return set(index["weight_map"])

    with safe_open(str(Path(ckpt) / shards[0]), framework="pt", device="cpu") as handle:
        return set(handle.keys())


def _vision_counts(ckpt):
    keys = _key_set(ckpt)
    flat = sum(is_flat_vision_key(key) for key in keys)
    nested = sum(key.startswith(NESTED_VISION_PREFIX) for key in keys)
    return flat, nested


def convert_vision_keys(source, destination):
    """Copy ``source`` to ``destination``, renaming flat vision-tower keys.

    Returns the number of renamed tensors. When the checkpoint already uses the
    nested layout, nothing is written and 0 is returned.
    """
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination must differ; keep the training checkpoint untouched")
    if destination.exists() and not destination.is_dir():
        raise FileExistsError(f"{destination} exists and is not a directory")
    if destination.is_dir() and any(destination.iterdir()):
        raise FileExistsError(f"{destination} already exists and is not empty; choose another output directory")

    flat_before, _ = _vision_counts(source)
    if flat_before == 0:
        print(f"{source}: vision tower already uses the nested layout, nothing to convert")
        return 0
    print(f"{source}: {flat_before} flat vision-tower tensors -> {destination}")

    destination.mkdir(parents=True, exist_ok=True)
    index, shards = _weight_files(source)
    weight_map = dict(index["weight_map"]) if index is not None else None
    renamed = 0

    for shard in shards:
        source_path = source / shard
        destination_path = destination / shard
        tensors = None
        metadata = None
        with safe_open(str(source_path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            flat_keys = [key for key in keys if is_flat_vision_key(key)]
            if flat_keys:
                clashes = {nested_vision_key(key) for key in flat_keys} & set(keys)
                if clashes:
                    raise ValueError(f"{shard} contains both layouts: {sorted(clashes)[:3]}")
                metadata = handle.metadata()
                tensors = {}
                for key in keys:
                    new_key = nested_vision_key(key) if is_flat_vision_key(key) else key
                    tensors[new_key] = handle.get_tensor(key)

        if tensors is None:
            # No vision tensor in this shard: copy the bytes instead of rewriting them.
            shutil.copy2(source_path, destination_path)
            continue

        save_file(tensors, str(destination_path), metadata=metadata or {"format": "pt"})
        if weight_map is not None:
            for key in flat_keys:
                weight_map[nested_vision_key(key)] = weight_map.pop(key)
        renamed += len(flat_keys)
        print(f"  {shard}: renamed {len(flat_keys)} tensors")

    if index is not None:
        payload = {**index, "weight_map": weight_map}
        (destination / INDEX_NAME).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Tokenizer/processor/config files are not weights and are carried over as-is.
    for path in source.iterdir():
        if path.is_file() and path.suffix != ".safetensors" and path.name != INDEX_NAME:
            shutil.copy2(path, destination / path.name)

    return renamed


def verify_conversion(source, destination):
    """Check that conversion only renamed vision keys and changed no key set."""
    before = _key_set(source)
    after = _key_set(destination)
    expected = {nested_vision_key(key) if is_flat_vision_key(key) else key for key in before}
    flat_after = sorted(key for key in after if is_flat_vision_key(key))

    ok = True
    _, shards = _weight_files(destination)
    missing_shards = [shard for shard in shards if not (Path(destination) / shard).is_file()]
    if missing_shards:
        print(f"  !! missing weight shards: {missing_shards[:3]}")
        ok = False
    if after != expected:
        missing = sorted(expected - after)[:3]
        extra = sorted(after - expected)[:3]
        print(f"  !! converted key set differs: missing {missing}, extra {extra}")
        ok = False
    if flat_after:
        print(f"  !! {len(flat_after)} flat vision-tower keys remain: {flat_after[:3]}")
        ok = False
    nested = sum(key.startswith(NESTED_VISION_PREFIX) for key in after)
    print(
        f"  keys: {len(before)} -> {len(after)}, nested vision-tower tensors: {nested}, remaining flat: {len(flat_after)}"
    )
    return ok


def verify_merged(base, merged):
    """Check that merging preserved the base checkpoint's parameter names."""
    if not Path(base).is_dir():
        return True
    try:
        base_keys = _key_set(base)
    except FileNotFoundError:
        print(f"  skip base comparison: {base} has no safetensors weights")
        return True

    merged_keys = _key_set(merged)
    missing = sorted(base_keys - merged_keys)[:3]
    extra = sorted(merged_keys - base_keys)[:3]
    if missing or extra:
        print(f"  !! merged key set differs from the base: missing {missing}, extra {extra}")
        return False
    print(f"  merged keys match the base: {len(merged_keys)} tensors")
    return True


def trim_vocab(ckpt, vocab):
    """Truncate ``embed_tokens``/``lm_head`` to ``vocab`` rows in place.

    Useful when the tokenizer used for evaluation has a different vocabulary
    size from the checkpoint's base model.
    """
    ckpt = Path(ckpt)
    index, shards = _weight_files(ckpt)
    by_shard = {}
    for key in VOCAB_KEYS:
        if index is not None:
            if key not in index["weight_map"]:
                raise KeyError(f"{ckpt} has no parameter {key}")
            shard = index["weight_map"][key]
        else:
            shard = shards[0]
        by_shard.setdefault(shard, []).append(key)

    for shard, keys in by_shard.items():
        path = ckpt / shard
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        for key in keys:
            rows = tensors[key].shape[0]
            if vocab > rows:
                raise ValueError(f"{key} has {rows} rows, cannot trim to {vocab}")
            print(f"  {key}: {rows} -> {vocab} rows")
            tensors[key] = tensors[key][:vocab].contiguous()
        save_file(tensors, str(path), metadata=metadata or {"format": "pt"})


def copy_missing_meta(base, out):
    """Copy non-weight files the export did not write, without overwriting its output."""
    for path in sorted(Path(base).iterdir()):
        if not path.is_file() or path.suffix == ".safetensors" or path.name == INDEX_NAME:
            continue
        target = Path(out) / path.name
        if not target.exists():
            shutil.copy2(path, target)
            print(f"  meta: {path.name} <- base")


def copy_meta_from(source, out):
    """Overwrite the listed metadata files from ``source``."""
    for name in META_FILES:
        path = Path(source) / name
        if path.is_file():
            shutil.copy2(path, Path(out) / name)
            print(f"  meta: {name} <- {source}")


def find_llamafactory_cli(explicit=None):
    if explicit:
        return explicit
    found = shutil.which("llamafactory-cli")
    if found:
        return found
    candidate = Path(sys.executable).with_name("llamafactory-cli")
    if candidate.is_file():
        return str(candidate)
    raise FileNotFoundError("llamafactory-cli not found; activate the training environment or pass --llamafactory-cli")


def run_command(command, log_path=None):
    print("$ " + " ".join(command), flush=True)
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if log_path:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("$ " + " ".join(command) + "\n" + process.stdout)
    print(process.stdout, end="", flush=True)
    if process.returncode != 0:
        raise SystemExit(f"command failed with exit code {process.returncode}: {' '.join(command)}")


def export_merged(base, adapter, out, args):
    command = [
        find_llamafactory_cli(args.llamafactory_cli),
        "export",
        "--model_name_or_path",
        str(base),
        "--adapter_name_or_path",
        str(adapter),
        "--template",
        args.template,
        "--finetuning_type",
        "lora",
        "--export_dir",
        str(out),
        "--export_size",
        str(args.export_size),
        "--export_device",
        args.export_device,
    ]
    if args.trust_remote_code:
        command.append("--trust_remote_code")
    run_command(command, args.log)


def derive_eval_dir(out):
    """Default evaluation directory: ``*_merged`` -> ``*_merged_vllm``, else ``*_vllm``."""
    out = Path(out)
    parts = list(out.parts)
    for index in range(len(parts) - 1, -1, -1):
        if "_merged" in parts[index]:
            parts[index] = parts[index].replace("_merged", "_merged_vllm", 1)
            return Path(*parts)
    return out.with_name(out.name + "_vllm")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", help="base model directory (required unless --convert-only)")
    parser.add_argument(
        "--adapter", "--lora", dest="adapter", help="native LoRA checkpoint directory (required unless --convert-only)"
    )
    parser.add_argument(
        "--out", required=True, help="merged output directory; with --convert-only, the converted destination"
    )
    parser.add_argument(
        "--out-vllm",
        default=None,
        help="converted directory; default: --out with _merged -> _merged_vllm, else _vllm suffix",
    )
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="skip LoRA export and only rewrite the vision-tower keys of --source",
    )
    parser.add_argument("--source", default=None, help="checkpoint to rewrite with --convert-only")
    parser.add_argument("--template", default="llava", help="LLaMA-Factory template (default: llava)")
    parser.add_argument("--export-size", type=int, default=5, help="export shard size in GB (default: 5)")
    parser.add_argument(
        "--export-device", default="cpu", choices=("cpu", "auto"), help="device used during export (default: cpu)"
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="pass --trust_remote_code to the export")
    parser.add_argument(
        "--vocab-trim", type=int, default=None, help="truncate embed_tokens/lm_head to N rows in the merged checkpoint"
    )
    parser.add_argument(
        "--meta-from",
        default=None,
        help="copy tokenizer/config files from this directory over the exported checkpoint",
    )
    parser.add_argument("--llamafactory-cli", default=None, help="explicit path to the llamafactory-cli executable")
    parser.add_argument("--log", default=None, help="append the export output to this log file")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.convert_only:
        if not args.source:
            raise SystemExit("--convert-only requires --source")
        source = Path(args.source).expanduser().resolve()
        out = Path(args.out).expanduser().resolve()
        print(f"[1/2] rewrite vision-tower keys: {source} -> {out}")
        renamed = convert_vision_keys(source, out)
        if renamed == 0:
            print("nothing to do; use the source checkpoint directly for evaluation")
            return 0
        print("[2/2] verify")
        return 0 if verify_conversion(source, out) else 1

    if not args.base or not args.adapter:
        raise SystemExit("--base and --adapter are required unless --convert-only is set")

    base = Path(args.base).expanduser().resolve()
    adapter = Path(args.adapter).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out_vllm = Path(args.out_vllm).expanduser().resolve() if args.out_vllm else derive_eval_dir(out)
    if not base.exists():
        raise SystemExit(f"base model not found: {base}")
    if not adapter.exists():
        raise SystemExit(f"adapter checkpoint not found: {adapter}")
    if out.exists() and not out.is_dir():
        raise FileExistsError(f"{out} exists and is not a directory")
    if out.is_dir() and any(out.iterdir()):
        raise FileExistsError(f"{out} already exists and is not empty; choose another output directory")

    print(f"[1/4] merge LoRA: {adapter} -> {out}")
    export_merged(base, adapter, out, args)

    print("[2/4] copy metadata")
    copy_missing_meta(base, out)
    if args.meta_from:
        copy_meta_from(Path(args.meta_from).expanduser().resolve(), out)

    if args.vocab_trim:
        print(f"[3/4] trim vocabulary to {args.vocab_trim} rows")
        trim_vocab(out, args.vocab_trim)
    else:
        print("[3/4] vocabulary unchanged")

    print(f"[4/4] rewrite vision-tower keys: {out} -> {out_vllm}")
    renamed = convert_vision_keys(out, out_vllm)
    if renamed == 0:
        print(f"export already uses the nested layout; evaluation checkpoint is {out}")
        return 0 if verify_merged(base, out) else 1

    print("[verify]")
    ok = verify_merged(base, out)
    ok = verify_conversion(out, out_vllm) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
