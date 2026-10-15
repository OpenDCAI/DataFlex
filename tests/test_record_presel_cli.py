"""Exercise every record-mode public command and train the exported subset."""

import json
import os
import subprocess
import sys

import torch
from test_multimodal_llava_integration import tiny_llava
from test_multimodal_qwen_integration import cli_training_config, run_training_cli
from test_multimodal_selection import fixture_dataset
from transformers import BitImageProcessor, Dinov2Config, Dinov2Model

from dataflex.utils.multimodal_selection import write_json


def command(args, tmp_path, label, expected_error=None):
    path = tmp_path / f"{label}.log"
    with path.open("w") as log:
        result = subprocess.run(
            [sys.executable, "-c", "from dataflex.cli import main; main()", "presel", *map(str, args)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
            timeout=90,
        )
    if expected_error:
        assert result.returncode and expected_error in path.read_text(), path.read_text()[-8000:]
    else:
        assert result.returncode == 0, path.read_text()[-8000:]


def test_record_mode_full_cli_with_repeated_images_and_real_reference_training(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    images, _ = fixture_dataset(tmp_path, 10)
    records = [
        {
            "unique_idx": i,
            "task": f"official_task_{i % 2 + 1}",
            "image": images[i % 10]["image"],
            "conversations": [{"from": "human", "value": "<image>\ntoken5"}, {"from": "gpt", "value": "token6"}],
        }
        for i in range(100)
    ]
    data = tmp_path / "data.json"
    write_json(data, records)
    common = ["--data", data, "--image-root", "/"]  # public default is record mode
    reference = tmp_path / "reference"
    command(["reference", *common, "--output-dir", reference], tmp_path, "reference")
    ids = json.loads((reference / "reference_indices.json").read_text())
    assert len(ids) == 5
    model, processor = tiny_llava()
    base = tmp_path / "base"
    model.save_pretrained(base)
    processor.save_pretrained(base)
    checkpoint = tmp_path / "reference_model"
    config = cli_training_config(base, reference, "presel_reference", checkpoint)
    config.update(
        template="llava",
        max_steps=-1,
        num_train_epochs=1,
        additional_target="multi_modal_projector",
        freeze_multi_modal_projector=False,
        freeze_vision_tower=True,
    )
    config.pop("image_min_pixels")
    config.pop("image_max_pixels")
    config_path = tmp_path / "reference_config.json"
    write_json(config_path, config)
    run_training_cli(config, tmp_path, "reference_train.log")
    provenance = tmp_path / "training_manifest.json"
    command(
        [
            "training-manifest",
            "--reference-ids",
            reference / "reference_ids.json",
            "--checkpoint",
            checkpoint,
            "--training-config",
            config_path,
            "--gpu-hours",
            0,
            "--output-file",
            provenance,
        ],
        tmp_path,
        "provenance",
    )
    scores = tmp_path / "scores"
    command(
        [
            "score",
            *common,
            "--reference-ids",
            reference / "reference_ids.json",
            "--reference-model",
            checkpoint,
            "--reference-training-manifest",
            provenance,
            "--output-dir",
            scores,
        ],
        tmp_path,
        "scores",
    )
    dino = tmp_path / "dino"
    Dinov2Model(
        Dinov2Config(
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            image_size=8,
            patch_size=4,
        )
    ).save_pretrained(dino)
    BitImageProcessor(size={"shortest_edge": 8}, crop_size={"height": 8, "width": 8}).save_pretrained(dino)
    features = tmp_path / "features"
    command(["extract", *common, "--model", dino, "--batch-size", 4, "--output-dir", features], tmp_path, "features")
    manifest = json.loads((features / "manifest.json").read_text())
    assert manifest["physical_images_encoded"] == 10 and manifest["budget_unit"] == "record"
    selected = tmp_path / "selected"
    args = [
        "select",
        *common,
        "--reference-ids",
        reference / "reference_ids.json",
        "--scores",
        scores / "reference_scores.json",
        "--score-manifest",
        scores / "score_manifest.json",
        "--features-dir",
        features,
        "--output-dir",
        selected,
    ]
    command(args, tmp_path, "selection")
    chosen = json.loads((selected / "selected_indices.json").read_text())
    assert len(chosen) == len(set(chosen)) == 15 and set(ids) <= set(chosen)
    assert len(json.loads((selected / "selected_training.json").read_text())) == 15
    command([*args, "--budget", 16], tmp_path, "wrong_budget", expected_error="final budget")
    final = {
        **config,
        "dataset": "presel_selected",
        "dataset_dir": str(selected),
        "output_dir": str(tmp_path / "final_training"),
    }
    run_training_cli(final, tmp_path, "final_train.log")
    state = json.loads((tmp_path / "final_training/trainer_state.json").read_text())
    assert state["global_step"] == 15 and state["epoch"] == 1
