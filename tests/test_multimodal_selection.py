import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image

from dataflex.core.registry import REGISTRY
from dataflex.offline_selector.offline_presel_selector import (
    choose_reference,
    export_training_dataset,
    image_groups,
    select_images,
    task_importance,
)
from dataflex.train.selector.adapt_inf_selector import AdaptInfSelector
from dataflex.utils.multimodal_scoring import (
    CachedCountSketch,
    answer_statistics,
    move_to_device,
    project_gradients,
    qwen_text_only_sample,
)
from dataflex.utils.multimodal_selection import (
    allocate_budget,
    coverage_sample,
    neighbor_centrality,
    score_distribution_entropy,
)


def test_move_batch_encoding_and_nested_mapping_to_device():
    from collections import UserDict

    from transformers import BatchEncoding

    batch = BatchEncoding(
        {"input_ids": torch.tensor([[1, 2]]), "nested": UserDict({"pixels": [torch.ones(2)]})}
    )
    moved = move_to_device(batch, "meta")
    assert moved["input_ids"].device.type == "meta"
    assert moved["nested"]["pixels"][0].device.type == "meta"
    assert batch["input_ids"].device.type == "cpu"


class Tokenizer:
    def get_vocab(self):
        return {"<|vision_start|>": 1, "<|vision_end|>": 2, "<|image_pad|>": 3}


class Collator:
    tokenizer = Tokenizer()

    def __call__(self, samples):
        sample = samples[0]
        pixels = np.zeros(3, dtype=np.float32)
        if sample.get("images"):
            with Image.open(sample["images"][0]) as image:
                pixels = np.asarray(image, dtype=np.float32).mean(axis=(0, 1)) / 255
        return {
            "input_ids": torch.tensor([sample["input_ids"]]),
            "labels": torch.tensor([sample["labels"]]),
            "pixel_values": torch.tensor(pixels[None]),
        }


class TinyVisionModel(torch.nn.Module):
    """CPU fixture, not Qwen or a surrogate benchmark result.

    Reads actual image pixels and propagates the text prefix to each prediction.
    The small low-rank adapter is trained through the exact selector interface.
    """

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 8)
        self.vision = torch.nn.Linear(3, 8)
        self.lora_a = torch.nn.Linear(8, 2, bias=False)
        self.lora_b = torch.nn.Linear(2, 16, bias=False)

    def forward(
        self, input_ids, labels=None, pixel_values=None, attention_mask=None, position_ids=None, use_cache=False
    ):
        if pixel_values is None:
            pixel_values = torch.zeros(input_ids.shape[0], 3, device=input_ids.device)
        context = self.embedding(input_ids).cumsum(1) + self.vision(pixel_values)[:, None]
        return SimpleNamespace(logits=self.lora_b(self.lora_a(context)))


class CPUAccelerator:
    device = torch.device("cpu")
    distributed_type = "NO"

    def unwrap_model(self, model):
        return model.module if hasattr(model, "module") else model

    def autocast(self):
        return nullcontext()


def fixture_dataset(directory, count=12):
    records, processed = [], []
    for i in range(count):
        path = Path(directory) / f"image_{i}.png"
        Image.new("RGB", (8, 8), (i * 17 % 255, i * 29 % 255, i * 41 % 255)).save(path)
        records.append(
            {
                "id": f"original-{i * 11}",
                "image": str(path),
                "task": "a" if i < count // 2 else "b",
                "question": "What color?",
                "response": str(i),
            }
        )
        processed.append(
            {
                "input_ids": [0, 1, 3, 2, 4 + i % 3, 7 + i % 5, 6],
                "labels": [-100, -100, -100, -100, -100, 7 + i % 5, 6],
                "images": [str(path)],
            }
        )
    return records, processed


def test_capped_allocation_and_ccs_edge_cases():
    assert allocate_budget([0, 1, 20], 10) == [0, 1, 9]
    assert allocate_budget([10, 10], 6, [1, 2]) == [2, 4]
    assert allocate_budget([], 0) == []
    with pytest.raises(ValueError):
        allocate_budget([1], 2)
    with pytest.raises(ValueError):
        allocate_budget([1], 1, [float("nan")])
    rng = np.random.default_rng(4)
    picks, stats = coverage_sample(np.arange(100), 100, rng)
    assert sorted(picks) == list(range(100))
    assert stats["trimmed_fallback_count"] == 10
    assert np.isnan(score_distribution_entropy([1, 1, 1]))  # published helper: 0*log2(0)
    assert score_distribution_entropy(np.arange(100)) > 0
    assert len(coverage_sample(np.ones(7), 5, rng)[0]) == 5


def test_answer_mask_math_and_gradients():
    logits = torch.zeros(1, 4, 3, requires_grad=True)
    labels = torch.tensor([[-100, -100, 1, 2]])
    stats = answer_statistics(logits, labels)
    assert stats["answer_tokens"] == 2
    assert float(stats["loss"].detach()) == pytest.approx(np.log(3))
    assert stats["entropy"] == pytest.approx(np.log(3))
    assert stats["el2n"] == pytest.approx(np.sqrt(2 / 3))
    (grad,) = torch.autograd.grad(stats["loss"], logits)
    assert torch.count_nonzero(grad[0, 0]) == 0
    assert torch.count_nonzero(grad[0, 3]) == 0
    with pytest.raises(ValueError, match="no supervised"):
        answer_statistics(logits, torch.full_like(labels, -100))


def test_image_ablation_preserves_all_answer_turns(tmp_path):
    _, dataset = fixture_dataset(tmp_path, 2)
    sample = dataset[0]
    sample["input_ids"] += [4, 9]
    sample["labels"] += [-100, 9]
    text = qwen_text_only_sample(sample, Tokenizer())
    assert text["images"] == []
    assert text["input_ids"] == [0, 4, 7, 6, 4, 9]
    assert [x for x in text["labels"] if x != -100] == [7, 6, 9]
    with pytest.raises(ValueError, match="image-only"):
        qwen_text_only_sample({**sample, "images": []}, Tokenizer())


def test_projection_is_fixed_and_exact_mode():
    gradients = [torch.arange(7).float(), torch.arange(3).float()]
    first = project_gradients(gradients, 8, 42)
    np.testing.assert_array_equal(first, project_gradients(gradients, 8, 42))
    assert not np.array_equal(first, project_gradients(gradients, 8, 43))
    np.testing.assert_array_equal(project_gradients(gradients, 8, 42, "none"), np.r_[np.arange(7), np.arange(3)])


def test_selector_registry_training_resume_and_cache_invalidation(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    _, dataset = fixture_dataset(tmp_path)
    model, collator = TinyVisionModel(), Collator()
    selector = REGISTRY.build(
        "selector",
        "adapt_inf",
        runtime={"dataset": dataset, "accelerator": CPUAccelerator(), "data_collator": collator},
        cfg={"cache_dir": str(tmp_path / "cache"), "selection_budget": "exact", "num_clusters": 3, "parameter_pattern": "lora_", "proj_dim": 16},
    )
    assert isinstance(selector, AdaptInfSelector)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.002)
    selections = []
    for step in range(20):
        if step in (4, 12):
            # Pending gradients and mixed module modes must survive scoring.
            model.train()
            model.vision.eval()
            saved_modes = [module.training for module in model.modules()]
            for p in model.parameters():
                p.grad = torch.ones_like(p)
            saved_gradients = [p.grad.clone() for p in model.parameters()]
            saved_rng = torch.get_rng_state().clone()
            indices = selector.select(model, step, 5)
            assert len(indices) == len(set(indices)) == 5
            assert all(0 <= index < len(dataset) for index in indices)
            assert [module.training for module in model.modules()] == saved_modes
            assert all(torch.equal(p.grad, grad) for p, grad in zip(model.parameters(), saved_gradients))
            assert torch.equal(torch.get_rng_state(), saved_rng)
            assert selector.select(model, step, 5) == indices
            assert selector.last_metadata["cache_hit"] is True
            restored = AdaptInfSelector(
                dataset,
                CPUAccelerator(),
                collator,
                str(tmp_path / "cache"),
                selection_budget="exact", num_clusters=3,
                parameter_pattern="lora_",
                proj_dim=16,
            )
            assert restored.select(model, step, 5) == indices
            assert restored.last_metadata["cache_hit"] is True
            selections.append(indices)
        index = selections[-1][step % 5] if selections else step % len(dataset)
        optimizer.zero_grad()
        batch = collator([dataset[index]])
        loss = answer_statistics(model(**batch).logits, batch["labels"])["loss"]
        loss.backward()
        optimizer.step()
    selector.select(model, 12, 5)
    assert selector.last_metadata["cache_hit"] is False
    selector.set_candidate_pool([1, 3, 5, 7, 9])
    assert set(selector.select(model, 12, 5)) == {1, 3, 5, 7, 9}
    with pytest.raises(NotImplementedError, match="permanent pruning"):
        AdaptInfSelector(dataset, CPUAccelerator(), collator, "unused", enable_permanent_pruning=True)


def test_presel_irs_nc_duplicates_and_selected_data_training(tmp_path):
    torch.set_num_threads(1)
    records, processed = fixture_dataset(tmp_path, 10)
    records.append({**records[0], "id": "second-instruction"})
    groups = image_groups(records, "/")
    assert len(groups) == 10
    assert groups[records[0]["image"]]["indices"] == [0, 10]
    reference = choose_reference(groups, 2, 42)
    scores = [
        {
            "index": index,
            "loss_with_question": 1 if group["task"] == "a" else 2,
            "loss_without_question": 2,
            "answer_tokens": 2,
            "answers_match": True,
        }
        for key, group in groups.items()
        if key in reference
        for index in group["indices"]
    ]
    importance, temperature = task_importance(records, groups, reference, scores)
    assert temperature == pytest.approx(1 / np.sqrt(2))
    assert importance["a"]["weight"] > importance["b"]["weight"]
    ids = list(groups)
    features = np.array([[1, i / 10, 0.1] for i in range(10)], dtype=np.float32)
    selected, indices, stats = select_images(groups, features, ids, importance, reference, 6, True, 42, 0.5)
    assert len(selected) == len(set(selected)) == 6
    assert set(reference) <= set(selected)
    assert sum(stats["task_budgets"].values()) == 4
    assert set(indices) == {index for key in selected for index in groups[key]["indices"]}
    # Static selection remains usable for a subsequent 20-step fixture train.
    model, optimizer, collator = TinyVisionModel(), None, Collator()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    for step in range(20):
        batch = collator([processed[indices[step % len(indices)] % 10]])
        optimizer.zero_grad()
        loss = answer_statistics(model(**batch).logits, batch["labels"])["loss"]
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    with pytest.raises(ValueError, match="positive denominator"):
        task_importance(records, groups, reference, [{**score, "loss_without_question": 0} for score in scores])
    with pytest.raises(ValueError, match="reference score coverage"):
        task_importance(records, groups, reference, scores[:-1])
    with pytest.raises(ValueError, match="multiple tasks"):
        image_groups(records + [{**records[0], "task": "other"}], "/")
    np.testing.assert_allclose(neighbor_centrality([[1, 0], [1, 0], [0, 1]], 1), [1, 1, 0])
    assert neighbor_centrality([[1, 0]]).tolist() == [1]
    with pytest.raises(ValueError, match="invalid image"):
        image_groups([{**records[0], "image": str(tmp_path / "missing.png")}], "/")


def test_cached_countsketch_preserves_seeded_map_across_chunks():
    rng = np.random.default_rng(42)
    gradients = [torch.tensor(rng.standard_normal(70000), dtype=torch.float32), torch.arange(17.0)]
    projector = CachedCountSketch(gradients, 32, 42, "cpu")
    expected = project_gradients(gradients, 32, 42)
    actual = projector(gradients)
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=1e-3)
    np.testing.assert_array_equal(actual, projector(gradients))
    with pytest.raises(ValueError, match="layout changed"):
        projector(gradients[:1])


def test_presel_shared_image_preserves_task_membership_and_real_path(tmp_path):
    records, _ = fixture_dataset(tmp_path, 2)
    records = [records[0], {**records[0], "task": "other"}, {**records[0], "id": "duplicate"}]
    groups = image_groups(records, "/", cross_task_policy="task-image")
    assert len(groups) == 2
    assert sorted(group["indices"] for group in groups.values()) == [[0, 2], [1]]
    assert {group["path"] for group in groups.values()} == {records[0]["image"]}
    reference = choose_reference(groups, 2, 42)
    scores = [
        {"index": index, "loss_with_question": 1, "loss_without_question": 2,
         "answer_tokens": 2, "answers_match": True}
        for key in reference for index in groups[key]["indices"]
    ]
    importance, _ = task_importance(records, groups, reference, scores)
    selected, indices, _ = select_images(
        groups, np.array([[1, 0], [1, 0]]), list(groups), importance, reference, 2, True, 42
    )
    assert len(selected) == 2 and indices == [0, 1, 2]
    export_training_dataset(tmp_path, "shared", records, groups, indices, "shared")
    training = json.loads((tmp_path / "shared_training.json").read_text())
    assert len(training) == 3
    assert all(row["images"] == [records[0]["image"]] for row in training)


def _distributed_worker(rank, directory, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=Path(rendezvous).as_uri(), rank=rank, world_size=2)
    try:
        torch.manual_seed(42)
        dataset = json.loads((Path(directory) / "processed.json").read_text())
        selector = AdaptInfSelector(
            dataset,
            CPUAccelerator(),
            Collator(),
            str(Path(directory) / "cache"),
            selection_budget="exact", num_clusters=3,
            parameter_pattern="lora_",
            proj_dim=16,
        )
        model = torch.nn.parallel.DistributedDataParallel(TinyVisionModel())
        result = selector.select(model, 4, 5)
        gathered = [None, None]
        dist.all_gather_object(gathered, result)
        assert gathered[0] == gathered[1]
        assert len(result) == len(set(result)) == 5
        assert selector.select(model, 4, 5) == result
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        # Selection must leave DDP's reducer usable for ordinary training;
        # test both directions of the train -> score -> train transition.
        for step in (5, 12):
            batch = Collator()([dataset[result[rank]]])
            optimizer.zero_grad()
            answer_statistics(model(**batch).logits, batch["labels"])["loss"].backward()
            optimizer.step()
            result = selector.select(model, step, 5)
            dist.all_gather_object(gathered, result)
            assert gathered[0] == gathered[1]
        original_memmap = np.lib.format.open_memmap

        def fail_one_rank(filename, *args, **kwargs):
            if rank == 1 and Path(filename).name == "rank_1_features.npy":
                raise OSError("fixture shard write failure")
            return original_memmap(filename, *args, **kwargs)

        np.lib.format.open_memmap = fail_one_rank
        try:
            with pytest.raises(RuntimeError, match="rank 1.*shard write failure"):
                selector.select(model, 20, 5)
        finally:
            np.lib.format.open_memmap = original_memmap
    finally:
        dist.destroy_process_group()


def test_two_rank_ddp_same_selection_and_cache(tmp_path):
    _, dataset = fixture_dataset(tmp_path, 7)  # Uneven shards are intentional.
    (tmp_path / "processed.json").write_text(json.dumps(dataset))
    mp.spawn(_distributed_worker, args=(str(tmp_path), str(tmp_path / "rendezvous")), nprocs=2, join=True)
