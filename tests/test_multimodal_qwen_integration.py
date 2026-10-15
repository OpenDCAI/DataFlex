"""Real model/processor integration using tiny random CPU checkpoints only."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_multimodal_selection import CPUAccelerator, fixture_dataset

from dataflex.offline_selector.offline_presel_selector import (
    paired_inputs,
    run_extract,
    run_reference,
    run_score,
    run_select,
)
from dataflex.train.selector.adapt_inf_selector import AdaptInfSelector
from dataflex.utils.multimodal_scoring import answer_statistics, assert_same_answers
from dataflex.utils.multimodal_selection import fingerprint, write_json


def run_training_cli(config, path, log_name="cli.log", *overrides, expected_error=None):
    """Run the public entry point in a fresh process, including hparam patches."""
    from dataflex.cli import __file__ as cli_path

    config_path = path / f"{log_name}.json"
    write_json(config_path, config)
    log_path = path / log_name
    with log_path.open("w") as log:
        result = subprocess.run(
            [sys.executable, "-c", "from dataflex.cli import main; main()", "train", str(config_path), *overrides],
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "WANDB_DISABLED": "true",
            },
            cwd=str(Path(cli_path).parents[2]),
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
    if expected_error is None:
        assert result.returncode == 0, log_path.read_text()[-16000:]
    else:
        assert result.returncode != 0 and expected_error in log_path.read_text(), log_path.read_text()[-16000:]


def cli_training_config(checkpoint, dataset_dir, dataset_name, output_dir, steps=20):
    return {
        "model_name_or_path": str(checkpoint),
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_target": "q_proj,v_proj",
        "lora_rank": 2,
        "lora_alpha": 4,
        "dataset": dataset_name,
        "dataset_dir": str(dataset_dir),
        "template": "qwen3_vl_nothink",
        "cutoff_len": 128,
        "packing": False,
        "preprocessing_num_workers": 1,
        "image_min_pixels": 64,
        "image_max_pixels": 64,
        "dataloader_num_workers": 0,
        "output_dir": str(output_dir),
        "report_to": "none",
        "seed": 42,
        "use_cpu": True,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4,
        "num_train_epochs": 1,
        "max_steps": steps,
        "bf16": False,
        "gradient_checkpointing": False,
        "save_strategy": "steps",
        "save_steps": 7,
        "logging_steps": 1,
        "train_type": "static",
        "disable_tqdm": True,
    }


def tiny_qwen_components():
    import transformers
    from llamafactory.data.collator import SFTDataCollatorWith4DAttentionMask
    from llamafactory.data.template import TEMPLATES
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    if not hasattr(transformers, "Qwen3VLForConditionalGeneration"):
        pytest.skip("Qwen3-VL integration requires a Transformers release supporting Qwen3-VL")
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen2VLImageProcessor,
        Qwen3VLConfig,
        Qwen3VLForConditionalGeneration,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )

    vocab = {f"token{i}": i for i in range(32)}
    special = [
        "<pad>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|im_start|>",
        "<|im_end|>",
        "user",
        "assistant",
        "color",
        "red",
        "blue",
        "<unk>",
    ]
    for i, token in enumerate(special):
        del vocab[f"token{i}"]
        vocab[token] = i
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="<|im_end|>",
        unk_token="<unk>",
        additional_special_tokens=special[1:7],
    )
    template = (
        "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\n' }}"
        "{% for c in m['content'] %}{% if c['type'] == 'image' %}"
        "{{ '<|vision_start|><|image_pad|><|vision_end|>' }}{% else %}{{ c['text'] }}{% endif %}{% endfor %}"
        "{{ '<|im_end|>\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(
            patch_size=4, merge_size=2, temporal_patch_size=2, min_pixels=64, max_pixels=64
        ),
        tokenizer=tokenizer,
        video_processor=Qwen3VLVideoProcessor(),
        chat_template=template,
    )
    config = Qwen3VLConfig(
        text_config=dict(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            rope_parameters=dict(rope_type="default", rope_theta=10000.0, mrope_section=[1, 1, 2]),
        ),
        vision_config=dict(
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            patch_size=4,
            spatial_merge_size=2,
            temporal_patch_size=2,
            out_hidden_size=16,
            num_position_embeddings=16,
            deepstack_visual_indexes=[],
        ),
        image_token_id=3,
        video_token_id=4,
        vision_start_token_id=1,
        vision_end_token_id=2,
    )
    model = get_peft_model(
        Qwen3VLForConditionalGeneration(config),
        LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    )
    collator = SFTDataCollatorWith4DAttentionMask(
        tokenizer=tokenizer, model=model, template=TEMPLATES["qwen3_vl_nothink"], processor=processor
    )
    return model, processor, collator


def test_qwen_lora_factory_collator_two_updates_and_training(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model, processor, collator = tiny_qwen_components()
    records, _ = fixture_dataset(tmp_path, 8)
    dataset = [
        {
            "input_ids": [5, 7, 1, 3, 2, 9, 6, 5, 8, 10 + i % 2, 6],
            "labels": [-100] * 9 + [10 + i % 2, 6],
            "attention_mask": [1] * 11,
            "images": [record["image"]],
        }
        for i, record in enumerate(records)
    ]
    selector = AdaptInfSelector(
        dataset,
        CPUAccelerator(),
        collator,
        str(tmp_path / "cache"),
        selection_budget="exact", num_clusters=2,
        proj_dim=16,
        parameter_pattern="lora_",
    )
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.01)
    selected = list(range(8))
    for step in range(20):
        if step in (4, 12):
            selected = selector.select(model, step, 4, tokenizer=processor.tokenizer)
            assert len(selected) == len(set(selected)) == 4
            assert selector.last_metadata["score_statistics"]["ig"]["min"] > 0
        batch = collator([dict(dataset[selected[step % len(selected)]])])
        optimizer.zero_grad()
        outputs = model(**batch)
        assert torch.isfinite(outputs.loss)
        outputs.loss.backward()
        optimizer.step()
    # Paired PreSel scoring supports multiple conversation rounds and image
    # processing through the actual Qwen processor.
    record = {
        "messages": [
            {"role": "user", "content": "color"},
            {"role": "assistant", "content": "red"},
            {"role": "user", "content": "color"},
            {"role": "assistant", "content": "blue"},
        ]
    }
    first = paired_inputs(processor, record, records[0]["image"])
    second = paired_inputs(processor, record, records[0]["image"], True)
    assert_same_answers(first["labels"], second["labels"])
    assert int((first["labels"] != -100).sum()) >= 4
    assert torch.isfinite(answer_statistics(model(**first).logits, first["labels"])["loss"])


def test_presel_complete_offline_pipeline_with_tiny_local_checkpoints(tmp_path):
    from transformers import Dinov2Config, Dinov2Model

    torch.set_num_threads(1)
    torch.manual_seed(42)
    records, _ = fixture_dataset(tmp_path, 8)
    for i, record in enumerate(records):
        record.update(question="color", response="red" if i % 2 == 0 else "blue")
    data = tmp_path / "data.json"
    write_json(data, records)
    common = dict(data=str(data), image_root="/", task_field="task")
    reference_dir = tmp_path / "reference"
    run_reference(SimpleNamespace(**common, count=2, seed=42, output_dir=str(reference_dir)))
    reference_path = reference_dir / "reference_ids.json"
    reference = json.loads(reference_path.read_text())
    model, processor, _ = tiny_qwen_components()
    base = tmp_path / "base_model"
    model.merge_and_unload().save_pretrained(base)
    processor.save_pretrained(base)
    checkpoint = tmp_path / "reference_model"
    reference_config = cli_training_config(base, reference_dir, "presel_reference", checkpoint, steps=2)
    run_training_cli(reference_config, tmp_path, "reference_cli.log")
    provenance = tmp_path / "reference_training_manifest.json"
    write_json(
        provenance,
        {
            "checkpoint": str(checkpoint),
            "reference_ids_fingerprint": fingerprint(reference),
            "optimizer_steps": 2,
            "gpu_hours": 0,
            "config": reference_config,
        },
    )
    score_dir = tmp_path / "scores"
    run_score(
        SimpleNamespace(
            **common,
            reference_ids=str(reference_path),
            reference_model=str(checkpoint),
            reference_training_manifest=str(provenance),
            revision=None,
            device="cpu",
            dtype="auto",
            image_min_pixels=64,
            image_max_pixels=64,
            output_dir=str(score_dir),
        )
    )
    dino_checkpoint = tmp_path / "dino_model"
    config = Dinov2Config(
        hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, image_size=8, patch_size=4
    )
    Dinov2Model(config).save_pretrained(dino_checkpoint)
    # Actual DINOv2 processor/model; random tiny weights are fixture-only.
    from transformers import BitImageProcessor

    processor = BitImageProcessor(size={"shortest_edge": 8}, crop_size={"height": 8, "width": 8})
    processor.save_pretrained(dino_checkpoint)
    feature_dir = tmp_path / "features"
    run_extract(
        SimpleNamespace(
            **common,
            model=str(dino_checkpoint),
            revision=None,
            device="cpu",
            batch_size=2,
            output_dir=str(feature_dir),
        )
    )
    selected_dir = tmp_path / "selected"
    args = SimpleNamespace(
        **common,
        reference_ids=str(reference_path),
        scores=str(score_dir / "reference_scores.json"),
        score_manifest=str(score_dir / "score_manifest.json"),
        features_dir=str(feature_dir),
        budget=5,
        reference_budget="included",
        seed=42,
        temperature=None,
        cluster_fraction=0.25,
        output_dir=str(selected_dir),
    )
    run_select(args)
    indices = json.loads((selected_dir / "selected_indices.json").read_text())
    selected = json.loads((selected_dir / "selected_data.json").read_text())
    assert len(indices) == len(selected) == 5
    assert selected == [records[i] for i in indices]
    assert json.loads((selected_dir / "manifest.json").read_text())["reference_image_count"] == 2
    # The exported registry must work through DataFlex's stock static SFT path,
    # starting at raw conversations rather than hand-tokenized fixture rows.
    static_output = tmp_path / "presel_training"
    run_training_cli(
        cli_training_config(base, selected_dir, "presel_selected", static_output), tmp_path, "presel_cli.log"
    )
    assert json.loads((static_output / "trainer_state.json").read_text())["global_step"] == 20
    mapping = json.loads((selected_dir / "selected_training_index_map.json").read_text())
    assert [row["source_position"] for row in mapping] == indices
    # Artifact corruption must be detected before any selection is performed.
    features = np.load(feature_dir / "features.npy")
    features[0, 0] += 1
    np.save(feature_dir / "features.npy", features)
    with pytest.raises(ValueError, match="embedding files"):
        run_select(args)


@pytest.mark.parametrize("warmup,expected_steps", [(4, [4, 12]), (0, [0, 8, 16])])
def test_actual_dataflex_trainer_schedules_qwen_updates(tmp_path, warmup, expected_steps):
    from datasets import Dataset
    from llamafactory.hparams.training_args import TrainingArguments

    from dataflex.train.hparams.dynamic_params import DynamicFinetuningArguments
    from dataflex.train.trainer.select_trainer import SelectTrainer

    torch.set_num_threads(1)
    torch.manual_seed(42)
    model, processor, collator = tiny_qwen_components()
    records, _ = fixture_dataset(tmp_path, 8)
    dataset = Dataset.from_list(
        [
            {
                "input_ids": [5, 7, 1, 3, 2, 9, 6, 5, 8, 10 + i % 2, 6],
                "labels": [-100] * 9 + [10 + i % 2, 6],
                "attention_mask": [1] * 11,
                "images": [record["image"]],
            }
            for i, record in enumerate(records)
        ]
    )
    config_path = tmp_path / "components.yaml"
    # JSON is also YAML, and avoids another fixture serialization dependency.
    write_json(
        config_path,
        {
            "selectors": {
                "adapt_inf": {
                    "name": "adapt_inf",
                    "params": {
                        "cache_dir": str(tmp_path / "cache"),
                        "selection_budget": "exact", "num_clusters": 2,
                        "proj_dim": 16,
                        "parameter_pattern": "lora_",
                    },
                }
            }
        },
    )
    finetuning = DynamicFinetuningArguments(
        train_type="dynamic_select",
        component_name="adapt_inf",
        components_cfg_file=str(config_path),
        warmup_step=warmup,
        update_step=8,
        update_times=2,
        train_step=20,
    )
    args = TrainingArguments(
        output_dir=str(tmp_path / "training"),
        use_cpu=True,
        report_to=[],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        num_train_epochs=1,
        remove_unused_columns=False,
        save_strategy="no",
        logging_strategy="no",
        disable_tqdm=True,
    )
    trainer = SelectTrainer(
        finetuning_args=finetuning,
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=processor.tokenizer,
        processor=processor,
    )
    result = trainer.train()
    assert result.global_step == 20
    updates = sorted(path.name for path in (tmp_path / "cache").iterdir() if (path / "selection.json").exists())
    assert len(updates) == len(expected_steps)
    assert all(name.startswith(f"step_{step:08d}_") for name, step in zip(updates, expected_steps))


def test_raw_dataset_cli_dynamic_selection_and_mid_interval_resume(tmp_path):
    from safetensors.torch import load_file

    torch.set_num_threads(1)
    torch.manual_seed(42)
    model, processor, _ = tiny_qwen_components()
    base_model = tmp_path / "base"
    model.merge_and_unload().save_pretrained(base_model)
    processor.save_pretrained(base_model)
    records, _ = fixture_dataset(tmp_path, 24)
    for i, record in enumerate(records):
        record.update(question="color", response="red" if i % 2 == 0 else "blue")
    data = tmp_path / "raw.json"
    write_json(data, records)
    registered = tmp_path / "pool"
    run_reference(
        SimpleNamespace(
            data=str(data), image_root="/", task_field="task", count=24, seed=42, output_dir=str(registered)
        )
    )
    components = tmp_path / "components.yaml"
    cache = tmp_path / "cache"
    write_json(
        components,
        {
            "selectors": {
                "adapt_inf": {
                    "name": "adapt_inf",
                    "params": {
                        "cache_dir": str(cache),
                        "selection_budget": "exact", "num_clusters": 2,
                        "proj_dim": 16,
                        "parameter_pattern": "lora_",
                    },
                }
            }
        },
    )
    output = tmp_path / "adapt_training"
    config = cli_training_config(base_model, registered, "presel_reference", output)
    config.update(
        train_type="dynamic_select",
        component_name="adapt_inf",
        components_cfg_file=str(components),
        warmup_step=4,
        update_step=8,
        update_times=2,
        train_step=20,
        gradient_accumulation_steps=2,
    )
    run_training_cli(config, tmp_path, "adapt_cli.log")
    uninterrupted = load_file(output / "adapter_model.safetensors")
    stage = json.loads((output / "checkpoint-7" / "dataflex_selection_rank_0.json").read_text())
    assert stage["global_step"] == 7 and stage["opened_step"] == 4 and stage["batches_seen"] == 6
    # Resume three steps into the selected interval. A wrong warmup subset or
    # double skip changes later gradients, selections and final adapter weights.
    run_training_cli(config, tmp_path, "resume_cli.log", f"resume_from_checkpoint={output / 'checkpoint-7'}")
    resumed = load_file(output / "adapter_model.safetensors")
    assert uninterrupted.keys() == resumed.keys()
    for key in uninterrupted:
        torch.testing.assert_close(uninterrupted[key], resumed[key], rtol=0, atol=0)
    assert len(list(cache.glob("step_*/selection.json"))) == 2
    assert json.loads((output / "trainer_state.json").read_text())["global_step"] == 20
    run_training_cli(
        config,
        tmp_path,
        "mismatch_cli.log",
        f"resume_from_checkpoint={output / 'checkpoint-7'}",
        "gradient_accumulation_steps=1",
        expected_error="batch settings do not match",
    )
    # Public trainer hooks are opt-in: the existing Random component should
    # still run through the same CLI without selection checkpoint sidecars.
    random_components = tmp_path / "random_components.yaml"
    write_json(
        random_components,
        {
            "selectors": {
                "random": {
                    "name": "random",
                    "params": {
                        "cache_dir": str(tmp_path / "random_cache"),
                        "seed": 42,
                    },
                }
            }
        },
    )
    random_output = tmp_path / "random_training"
    random_config = {
        **config,
        "component_name": "random",
        "components_cfg_file": str(random_components),
        "output_dir": str(random_output),
    }
    run_training_cli(random_config, tmp_path, "random_cli.log")
    assert json.loads((random_output / "trainer_state.json").read_text())["global_step"] == 20
    assert not list(random_output.glob("checkpoint-*/dataflex_selection_rank_*.json"))
