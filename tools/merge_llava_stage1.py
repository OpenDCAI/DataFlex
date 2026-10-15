"""Build an HF LLaVA-1.5 stage1 checkpoint from the three specified sources.

Example: python tools/merge_llava_stage1.py --vicuna-dir ... --projector-dir ...
    --clip-dir ... --output-dir ...
Inputs are local snapshots. No stage2/instruction-tuned LLaVA weights are used.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    CLIPImageProcessor,
    CLIPVisionConfig,
    LlamaConfig,
    LlavaConfig,
    LlavaForConditionalGeneration,
    LlavaProcessor,
)


SOURCES = {
    "vicuna": ("lmsys/vicuna-7b-v1.5", "3321f76e3f527bd14065daf69dad9344000a201d"),
    "projector": (
        "liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5",
        "5414da88308e4287a29f2e9609256458afb0a981",
    ),
    "clip": ("openai/clip-vit-large-patch14-336", "ce19dc912ca5cd21c8a653c79e251e808ccabcd1"),
}
WEIGHT_HASHES = {
    "pytorch_model-00001-of-00002.bin": "4133d2fcc5f31286881ea50806d95b721d016b533036a99dedce3f8fe88520e6",
    "pytorch_model-00002-of-00002.bin": "d261d3c35e92d3070d1e61ed821ebfca812a847d2a880757d82728acf005c5ac",
    "mm_projector.bin": "8a8d5a8fc6030bd16d8ee3df3b20a21c79190a3b0932a798d05f52a1ffdfc215",
    "pytorch_model.bin": "c6032c2e0caae3dc2d4fba35535fa6307dbb49df59c7e182b1bc4b3329b81801",
}
METADATA_HASHES = {
    "vicuna": {
        "config.json": "e122b598d0734b489590e99e3b3562a11ce67ea13bce390a7868b4f73ae6e615",
        "tokenizer.model": "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347",
        "tokenizer_config.json": "1cbc84c8a5b41e0ec24e0da66677773569fb0ccfa753d5d184e343e2ec3815a0",
        "special_tokens_map.json": "719833ff26ac897a3ec8ed946028a135de2a351470af59b4008744ab1f0ee9b7",
    },
    "projector": {"config.json": "21e258a7587447ad85ffa9524476250e62f222a1d4e98b2f13e5e8d10290f073"},
    "clip": {
        "config.json": "51b1c14aabcdf639c4a0370eeda1010b773bbe1df78319c7d0f5882c22ac0ac0",
        "preprocessor_config.json": "d253881f65322dc546df59cf925a408e5538b8ecb5a1b496cdd36af9992686d4",
    },
}


def verify_metadata(directories):
    verified = []
    for source, hashes in METADATA_HASHES.items():
        for name, expected in hashes.items():
            with (directories[source] / name).open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != expected:
                raise ValueError(f"{source}/{name}: metadata differs from the pinned snapshot")
            verified.append({"source": source, "file": name, "sha256": digest})
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vicuna", "projector", "clip", "output"):
        parser.add_argument(f"--{name}-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty; existing checkpoints are never overwritten")
    paths = {
        "vicuna": [args.vicuna_dir / name for name in WEIGHT_HASHES if "-of-" in name],
        "projector": [args.projector_dir / "mm_projector.bin"],
        "clip": [args.clip_dir / "pytorch_model.bin"],
    }
    verified_metadata = verify_metadata(
        {"vicuna": args.vicuna_dir, "projector": args.projector_dir, "clip": args.clip_dir}
    )
    verified = []
    for source, files in paths.items():
        for path in files:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != WEIGHT_HASHES[path.name]:
                raise ValueError(f"{path}: weight hash differs from the pinned official snapshot")
            verified.append({"source": source, "file": path.name, "sha256": digest})
            print("Verified", source, path.name, flush=True)
    original = json.loads((args.projector_dir / "config.json").read_text())
    if original["mm_projector_type"] != "mlp2x_gelu" or original["mm_vision_select_layer"] != -2:
        raise ValueError("unexpected projector architecture or vision feature layer")
    # Explicitly use the Vicuna/Llama SentencePiece conversion. Generic auto
    # loading in the training environment previously retained only BPE vocab
    # and merges, losing the boundary/decoder rules without raising an error.
    tokenizer = LlamaTokenizer.from_pretrained(args.vicuna_dir, local_files_only=True, legacy=False)
    # A matching vocabulary is insufficient: a tokenizer can silently lose
    # SentencePiece boundary/decoding rules during backend conversion.
    sample = "The woman is wearing a robe."
    if tokenizer.decode(tokenizer.encode(sample, add_special_tokens=False)) != sample:
        raise ValueError("Vicuna tokenizer conversion lost text boundaries; restore and verify the original tokenizer before merging")
    if len(tokenizer) != 32000:
        raise ValueError("expected the original 32000-token Vicuna tokenizer")
    tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.unk_token
    text_config = LlamaConfig.from_pretrained(args.vicuna_dir, local_files_only=True)
    text_config.vocab_size = len(tokenizer)
    vision_config = CLIPVisionConfig.from_pretrained(args.clip_dir, local_files_only=True)
    config = LlavaConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_index=tokenizer.convert_tokens_to_ids("<image>"),
        vision_feature_layer=-2,
        vision_feature_select_strategy="default",
        image_seq_length=576,
        projector_hidden_act="gelu",
        dtype="float16",
    )
    config.architectures = ["LlavaForConditionalGeneration"]
    with init_empty_weights():
        model = LlavaForConditionalGeneration(config)
    state = {}
    for path in paths["vicuna"]:
        for key, value in torch.load(path, map_location="cpu", weights_only=True).items():
            if key.endswith(".rotary_emb.inv_freq"):
                # Older Transformers persisted this derived buffer per layer;
                # current Llama reconstructs it from the unchanged config.
                head_dim = text_config.hidden_size // text_config.num_attention_heads
                theta = text_config.rope_parameters["rope_theta"]
                expected = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
                if not torch.allclose(value.float(), expected, rtol=5e-4, atol=1e-6):
                    raise ValueError(f"unexpected original rotary buffer: {key}")
                continue
            target = "model.language_model." + key[len("model.") :] if key.startswith("model.") else key
            if key in ("model.embed_tokens.weight", "lm_head.weight"):
                # The original -200 image sentinel had no vocabulary row. HF
                # needs a row for <image>; initialize it with an exact mean.
                value = torch.cat([value, value.float().mean(0, keepdim=True).to(value.dtype)])
            state[target] = value.to(torch.float16)
    clip = torch.load(paths["clip"][0], map_location="cpu", weights_only=True)
    for key, value in clip.items():
        if key == "vision_model.embeddings.position_ids":
            if not torch.equal(value, torch.arange(577).expand(1, -1)):
                raise ValueError("unexpected CLIP positional index buffer")
            continue
        if key.startswith("vision_model."):
            state["model.vision_tower." + key] = value.to(torch.float16)
    del clip
    projector = torch.load(paths["projector"][0], map_location="cpu", weights_only=True)
    for key, value in projector.items():
        suffix = key.split("mm_projector.")[-1]
        parts = suffix.split(".")
        layer = {"0": "linear_1", "2": "linear_2"}.get(parts[0])
        if layer is None or len(parts) != 2:
            raise ValueError(f"unexpected projector key: {key}")
        state[f"model.multi_modal_projector.{layer}.{parts[1]}"] = value.to(torch.float16)
    model.load_state_dict(state, strict=True, assign=True)
    if any(p.is_meta for p in model.parameters()):
        raise ValueError("conversion left uninitialized parameters")
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")
    processor = LlavaProcessor(
        tokenizer=tokenizer,
        image_processor=CLIPImageProcessor.from_pretrained(args.clip_dir, local_files_only=True),
        patch_size=14,
        num_additional_image_tokens=1,
        vision_feature_select_strategy="default",
    )
    processor.save_pretrained(output)
    if hasattr(tokenizer, 'backend_tokenizer'):
        # Preserve the complete serialized backend under a portable class
        # name instead of triggering another Llama conversion at reload.
        tokenizer_path = output / 'tokenizer_config.json'
        tokenizer_config = json.loads(tokenizer_path.read_text())
        tokenizer_config['tokenizer_class'] = 'PreTrainedTokenizerFast'
        tokenizer_path.write_text(json.dumps(tokenizer_config, indent=2))
    reloaded = AutoTokenizer.from_pretrained(output, local_files_only=True)
    for text in [sample, 'Canon', 'A  B', '中文测试。', 'a\nb', 'USER: <image>\nWhat is this?']:
        expected_ids = tokenizer.encode(text, add_special_tokens=False)
        if reloaded.encode(text, add_special_tokens=False) != expected_ids:
            raise ValueError('Saved Vicuna tokenizer changed input ids on reload')
        if reloaded.decode(expected_ids) != tokenizer.decode(expected_ids):
            raise ValueError('Saved Vicuna tokenizer changed decoder behavior on reload')
    manifest = {
        "stage": "pre-VIT stage1, before reference fine-tuning",
        "sources": {name: {"repo": repo, "revision": rev} for name, (repo, rev) in SOURCES.items()},
        "verified_weights": verified,
        "verified_metadata": verified_metadata,
        "conversion": (
            "strict parameter mapping; FP16 storage; added image embedding/head row initialized to mean; "
            "validated legacy derived rotary/position buffers omitted and reconstructed by Transformers"
        ),
        "image_token_id": config.image_token_index,
        "image_sequence_length": config.image_seq_length,
        "output_weights": {},
    }
    for path in sorted(output.glob("*.safetensors")):
        with path.open("rb") as stream:
            manifest["output_weights"][path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    (output / "stage1_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Stage1 checkpoint ready:", output, flush=True)


if __name__ == "__main__":
    main()
