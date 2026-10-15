"""Exercise LLaVA image ablation, projector training and reference scoring."""

import json

import torch
from safetensors.torch import load_file
from test_multimodal_qwen_integration import cli_training_config, run_training_cli
from test_multimodal_selection import fixture_dataset

from dataflex.offline_selector.offline_presel_selector import load_reference_model, paired_inputs
from dataflex.utils.multimodal_scoring import (
    answer_statistics,
    assert_same_answers,
    image_text_only_sample,
    scoring_state,
)
from dataflex.utils.multimodal_selection import write_json


def tiny_llava():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        CLIPImageProcessor,
        LlavaConfig,
        LlavaForConditionalGeneration,
        LlavaProcessor,
        PreTrainedTokenizerFast,
    )

    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<image>": 3, "<unk>": 4}
    vocab.update({f"token{i}": i for i in range(5, 32)})
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        additional_special_tokens=["<image>"],
    )
    processor = LlavaProcessor(
        tokenizer=tokenizer,
        image_processor=CLIPImageProcessor(size={"shortest_edge": 8}, crop_size={"height": 8, "width": 8}),
        patch_size=4,
        num_additional_image_tokens=1,
        vision_feature_select_strategy="default",
    )
    config = LlavaConfig(
        text_config=dict(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        ),
        vision_config=dict(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            image_size=8,
            patch_size=4,
        ),
        image_token_index=3,
        image_seq_length=4,
        vision_feature_layer=-2,
    )
    return LlavaForConditionalGeneration(config), processor


def test_llava_paired_answers_and_true_image_ablation(tmp_path):
    torch.set_num_threads(1)
    model, processor = tiny_llava()
    records, _ = fixture_dataset(tmp_path, 2)
    record = records[0]
    record["messages"] = [
        {"role": "user", "content": "token5"},
        {"role": "assistant", "content": "token6"},
        {"role": "user", "content": "token7"},
        {"role": "assistant", "content": "token8"},
    ]
    with_q = paired_inputs(processor, record, record["image"])
    without_q = paired_inputs(processor, record, record["image"], True)
    assert_same_answers(with_q["labels"], without_q["labels"])
    assert (with_q["input_ids"] == 3).sum() == 4
    assert with_q["pixel_values"].shape == (1, 3, 8, 8)
    sample = {
        "input_ids": with_q["input_ids"][0].tolist(),
        "labels": with_q["labels"][0].tolist(),
        "images": [record["image"]],
    }
    text = image_text_only_sample(sample, processor.tokenizer, "llava")
    assert 3 not in text["input_ids"] and text["images"] == []
    assert_same_answers(with_q["labels"], text["labels"])
    with torch.no_grad():
        visual_logits = model(**with_q).logits
        text_ids = torch.tensor([text["input_ids"]])
        text_logits = model(input_ids=text_ids).logits
        assert torch.isfinite(answer_statistics(visual_logits, with_q["labels"])["loss"])
        assert torch.isfinite(answer_statistics(text_logits, torch.tensor([text["labels"]]))["loss"])


def test_llava_paired_cutoff_keeps_common_answers_and_image_tokens(tmp_path):
    _, processor = tiny_llava()
    records, _ = fixture_dataset(tmp_path, 2)
    record = {**records[0], "messages": [
        {"role": "user", "content": " ".join(["token5"] * 100)},
        {"role": "assistant", "content": " ".join(["token6"] * 120)},
    ]}
    with_q = paired_inputs(processor, record, record["image"], cutoff_len=80)
    without_q = paired_inputs(processor, record, record["image"], True, cutoff_len=80)
    assert with_q["input_ids"].shape[1] == 80
    assert without_q["input_ids"].shape[1] <= 80
    assert_same_answers(with_q["labels"], without_q["labels"])
    assert (with_q["input_ids"] == 3).sum() == (without_q["input_ids"] == 3).sum() == 4


def test_llava_checkpointed_scoring_matches_eval_and_restores_modes(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model, processor = tiny_llava()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    from llamafactory.model.model_utils.checkpointing import get_custom_gradient_checkpointing_func
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    for module in model.modules():
        if isinstance(module, LlamaDecoderLayer):
            module._gradient_checkpointing_func = get_custom_gradient_checkpointing_func(
                module._gradient_checkpointing_func
            )
    records, _ = fixture_dataset(tmp_path, 2)
    inputs = paired_inputs(processor, records[0], records[0]["image"])
    model.train()
    original_modes = [module.training for module in model.modules()]
    with scoring_state(model):
        loss = answer_statistics(model(**inputs).logits, inputs["labels"])["loss"]
        expected = torch.autograd.grad(loss, list(model.parameters()), allow_unused=True)
    with scoring_state(model, checkpoint_llama=True):
        assert all(module.training for module in model.modules() if isinstance(module, LlamaDecoderLayer))
        assert not model.training
        loss_checkpointed = answer_statistics(model(**inputs).logits, inputs["labels"])["loss"]
        actual = torch.autograd.grad(loss_checkpointed, list(model.parameters()), allow_unused=True)
    torch.testing.assert_close(loss, loss_checkpointed, rtol=0, atol=0)
    for reference, value in zip(expected, actual):
        if reference is None:
            assert value is None
        else:
            torch.testing.assert_close(reference, value, rtol=0, atol=0)
    assert original_modes == [module.training for module in model.modules()]


def test_llava_cli_dynamic_updates_train_and_save_projector(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model, processor = tiny_llava()
    checkpoint = tmp_path / "base"
    model.save_pretrained(checkpoint)
    processor.save_pretrained(checkpoint)
    records, _ = fixture_dataset(tmp_path, 20)
    data = tmp_path / "data"
    data.mkdir()
    training = []
    for record in records:
        training.append(
            {
                "messages": [
                    {"role": "user", "content": "<image>\ntoken5"},
                    {"role": "assistant", "content": "token6"},
                ],
                "images": [record["image"]],
            }
        )
    write_json(data / "pool.json", training)
    write_json(
        data / "dataset_info.json",
        {
            "llava_pool": {
                "file_name": "pool.json",
                "formatting": "sharegpt",
                "columns": {"messages": "messages", "images": "images"},
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                },
            }
        },
    )
    components = tmp_path / "components.yaml"
    components.write_text(
        json.dumps(
            {
                "selectors": {
                    "adapt_inf": {
                        "name": "adapt_inf",
                        "params": {
                            "cache_dir": str(tmp_path / "cache"),
                            "selection_budget": "exact", "num_clusters": 2,
                            "proj_dim": 16,
                            "parameter_pattern": "lora_|multi_modal_projector.*modules_to_save",
                            "seed": 42,
                        },
                    }
                }
            }
        )
    )
    output = tmp_path / "training"
    config = cli_training_config(checkpoint, data, "llava_pool", output)
    config.update(
        template="llava",
        learning_rate=2e-4,
        freeze_multi_modal_projector=False,
        freeze_vision_tower=True,
        additional_target="multi_modal_projector",
        train_type="dynamic_select",
        components_cfg_file=str(components),
        component_name="adapt_inf",
        warmup_step=4,
        update_step=8,
        update_times=2,
        train_step=20,
    )
    config.pop("image_min_pixels")
    config.pop("image_max_pixels")
    run_training_cli(config, tmp_path, "llava_cli.log")
    assert len(list((tmp_path / "cache").glob("step_*/selection.json"))) == 2
    saved = load_file(output / "adapter_model.safetensors")
    projector = {key: value for key, value in saved.items() if "multi_modal_projector" in key}
    assert len(projector) == 4
    initial = model.model.multi_modal_projector.state_dict()
    for key, value in projector.items():
        suffix = key.split("multi_modal_projector.")[-1].replace("modules_to_save.", "").replace("default.", "")
        assert torch.isfinite(value).all()
        assert not torch.equal(value, initial[suffix]), key
    assert all("vision_tower" not in key for key in saved)
    from transformers import AutoProcessor

    trained = load_reference_model(str(output))
    restored = trained.base_model.model.model.multi_modal_projector.modules_to_save["default"].state_dict()
    for key, value in projector.items():
        suffix = key.split("multi_modal_projector.")[-1]
        assert torch.equal(restored[suffix], value), key
    trained_processor = AutoProcessor.from_pretrained(output)
    pair = paired_inputs(trained_processor, records[0], records[0]["image"])
    assert torch.isfinite(answer_statistics(trained(**pair).logits, pair["labels"])["loss"])
