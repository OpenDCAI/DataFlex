"""Fresh-process strict resume after physically removing the original cache."""

import json
import shutil
from types import SimpleNamespace

import torch
from safetensors.torch import load_file
from test_multimodal_qwen_integration import (
    cli_training_config,
    run_training_cli,
    tiny_qwen_components,
)
from test_multimodal_selection import fixture_dataset

from dataflex.offline_selector.offline_presel_selector import run_reference
from dataflex.utils.multimodal_selection import write_json


def test_strict_resume_from_relocated_checkpoint_without_original_cache(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model, processor, _ = tiny_qwen_components()
    base = tmp_path / "base"
    model.merge_and_unload().save_pretrained(base)
    processor.save_pretrained(base)
    records, _ = fixture_dataset(tmp_path, 48)
    for i, r in enumerate(records):
        r.update(question="color", response="red" if i % 2 == 0 else "blue")
    data = tmp_path / "raw.json"
    write_json(data, records)
    registered = tmp_path / "registered"
    run_reference(
        SimpleNamespace(
            data=str(data), image_root="/", task_field="task", count=48, seed=42, output_dir=str(registered)
        )
    )
    cache = tmp_path / "cache"
    components = tmp_path / "components.json"
    params = {
        "cache_dir": str(cache),
        "unique_budget": 40,
        "num_clusters": 2,
        "cluster_grid": [2],
        "proj_dim": 16,
        "projection": "countsketch",
        "projection_backend": "cpu",
        "parameter_pattern": "lora_",
        "dataset_revision": "strict-resume-fixture-v1",
    }
    write_json(components, {"selectors": {"adapt_inf_strict": {"name": "adapt_inf_strict", "params": params}}})
    output = tmp_path / "uninterrupted"
    config = cli_training_config(base, registered, "presel_reference", output)
    config.update(
        train_type="dynamic_select",
        component_name="adapt_inf_strict",
        components_cfg_file=str(components),
        warmup_step=4,
        update_step=8,
        update_times=2,
        train_step=20,
        gradient_accumulation_steps=2,
    )
    run_training_cli(config, tmp_path, "strict_cli.log")
    expected = load_file(output / "adapter_model.safetensors")
    state = json.loads((output / "checkpoint-7/dataflex_selection_rank_0.json").read_text())
    assert [len(s["indices"]) for s in state["selector_state"]["stages"]] == [8, 16]
    relocated = tmp_path / "relocated-checkpoint"
    shutil.copytree(output / "checkpoint-7", relocated)
    shutil.rmtree(cache)
    new_cache = tmp_path / "new-cache"
    params["cache_dir"] = str(new_cache)
    write_json(components, {"selectors": {"adapt_inf_strict": {"name": "adapt_inf_strict", "params": params}}})
    resumed_output = tmp_path / "resumed"
    resumed = {**config, "output_dir": str(resumed_output)}
    run_training_cli(resumed, tmp_path, "relocated_resume.log", f"resume_from_checkpoint={relocated}")
    actual = load_file(resumed_output / "adapter_model.safetensors")
    assert expected.keys() == actual.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    stages = [json.loads(p.read_text()) for p in sorted((new_cache / "record_budget").glob("step_*.json"))]
    ids = [i for s in stages for i in s["indices"]]
    assert [len(s["indices"]) for s in stages] == [8, 16, 16]
    assert len(ids) == len(set(ids)) == 40
    broken = tmp_path / "broken-checkpoint"
    shutil.copytree(relocated, broken)
    sidecar = broken / "dataflex_selection_rank_0.json"
    state = json.loads(sidecar.read_text())
    state.pop("selector_state")
    sidecar.write_text(json.dumps(state))
    run_training_cli(
        {**config, "output_dir": str(tmp_path / "broken-output")},
        tmp_path,
        "missing_history.log",
        f"resume_from_checkpoint={broken}",
        expected_error="Strict record-budget state missing",
    )

    run_training_cli(
        {**config, "output_dir": str(tmp_path / "wrong_budget_output")},
        tmp_path,
        "wrong_budget.log",
        "gradient_accumulation_steps=1",
        expected_error="Strict unique budget must match planned training visits",
    )
