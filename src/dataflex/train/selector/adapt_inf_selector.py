"""Flex periodic Adapt-inf adaptation, with explicit projection/ablation limits."""

import hashlib
import random
import re
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from dataflex.core.registry import register_selector
from dataflex.utils.author_adapt import obtain_entropy_score
from dataflex.utils.author_adapters import author_ccs_indices, author_zero_image_ig, fit_selection_budget
from dataflex.utils.multimodal_scoring import (
    CachedCountSketch,
    answer_statistics,
    move_to_device,
    project_gradients,
    scoring_state,
)
from dataflex.utils.multimodal_selection import (
    allocate_budget,
    cluster_features,
    fingerprint,
    score_distribution_entropy,
    write_json,
)

from .base_selector import Selector


def _broadcast(value):
    payload = [value]
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _model_fingerprint(model, parameters):
    digest = hashlib.sha256()
    config = getattr(model, "config", None)
    digest.update(fingerprint(config.to_dict() if hasattr(config, "to_dict") else str(config)).encode())
    # Include frozen backbone weights too: config/path alone cannot identify a
    # locally replaced base checkpoint. This has a measurable hashing cost.
    for name, param in model.named_parameters():
        digest.update(name.encode())
        digest.update(param.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy())
    digest.update(fingerprint([(name, list(p.shape)) for name, p in parameters]).encode())
    return digest.hexdigest()


@register_selector("adapt_inf")
class AdaptInfSelector(Selector):
    checkpoint_stage = True

    def __init__(
        self,
        dataset,
        accelerator,
        data_collator,
        cache_dir,
        num_clusters=10,
        proj_dim=512,
        projection="countsketch",
        parameter_pattern=".*",
        seed=42,
        histogram_bins=50,
        trim_fraction=0.05,
        replacement=False,
        dataset_revision=None,
        enable_permanent_pruning=False,
        projection_backend="cpu",
        selection_budget="native",
    ):
        super().__init__(dataset, accelerator, data_collator, cache_dir)
        if enable_permanent_pruning:
            raise NotImplementedError(
                "This is Flex-adapted periodic selection; staged permanent pruning is not implemented"
            )
        if num_clusters < 1 or proj_dim < 1 or histogram_bins < 2 or not 0 <= trim_fraction < 0.5:
            raise ValueError("invalid Adapt-inf clustering/projection/histogram parameters")
        if trim_fraction != 0.05:
            raise ValueError("Published CCS uses fixed 5%/95% thresholds; trim_fraction must be 0.05")
        if projection not in ("countsketch", "none", "rademacher"):
            raise ValueError("projection must be countsketch, rademacher or none")
        if projection_backend not in ("cpu", "cuda") or (projection == "none" and projection_backend != "cpu"):
            raise ValueError("projection_backend must be cpu, or cuda with countsketch")
        if projection == "rademacher" and projection_backend != "cuda":
            raise ValueError("rademacher projection requires CUDA")
        self.num_clusters, self.proj_dim, self.projection = int(num_clusters), int(proj_dim), projection
        self.parameter_pattern, self.seed = str(parameter_pattern), int(seed)
        re.compile(self.parameter_pattern)
        self.histogram_bins, self.trim_fraction = int(histogram_bins), float(trim_fraction)
        self.replacement, self.dataset_revision = bool(replacement), dataset_revision
        self.projection_backend = projection_backend
        if selection_budget not in ("native", "exact"):
            raise ValueError("selection_budget must be native or exact")
        self.selection_budget = selection_budget
        self._gradient_projector = None
        self.last_metadata = None

    def checkpoint_signature(self):
        return fingerprint(
            {
                "data": self._data_fingerprint(self.candidate_positions()),
                "scoring_profile": "author-scores-ccs-d1b1b25-v2",
                "selection_budget": self.selection_budget,
                "config": [
                    self.num_clusters,
                    self.proj_dim,
                    self.projection,
                    self.projection_backend,
                    self.parameter_pattern,
                    self.seed,
                    self.histogram_bins,
                    self.trim_fraction,
                    self.replacement,
                ],
            }
        )

    def _data_fingerprint(self, positions):
        revision = self.dataset_revision or getattr(self.dataset, "_fingerprint", None)
        if revision is not None:
            return fingerprint({"revision": revision, "length": len(self.dataset), "positions": positions})
        # Small Python fixtures have no HF fingerprint. Production datasets should
        # supply a content/version fingerprint including their image snapshot.
        return fingerprint([self.dataset[position] for position in positions])

    def select(self, model, step_id, num_samples, **kwargs):
        if num_samples < 0 or int(num_samples) != num_samples:
            raise ValueError("num_samples must be a nonnegative integer")
        kind = str(getattr(self.accelerator, "distributed_type", "NO"))
        if "DEEPSPEED" in kind or "FSDP" in kind:
            raise NotImplementedError(
                "Adapt-inf currently requires replicated weights (single device or DDP), not ZeRO/FSDP"
            )
        if not num_samples:
            return []
        positions = self.candidate_positions()
        if len(set(positions)) != len(positions) or any(p < 0 or p >= len(self.dataset) for p in positions):
            raise ValueError("candidate positions must be unique valid dataset indices")
        if not positions or (num_samples > len(positions) and not self.replacement):
            raise ValueError("requested budget exceeds candidate pool without replacement")
        tokenizer = kwargs.get("tokenizer") or getattr(self.data_collator, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("a tokenizer is required to remove image tokens for IG")
        model = self.accelerator.unwrap_model(model)
        parameters = [
            (n, p) for n, p in model.named_parameters() if p.requires_grad and re.search(self.parameter_pattern, n)
        ]
        if not parameters:
            raise ValueError("parameter_pattern matches no trainable parameters")
        started = time.monotonic()
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        key = fingerprint(
            {
                "version": 7,
                "scoring_profile": "author-scores-ccs-d1b1b25-v2",
                "selection_budget": self.selection_budget,
                "selection_profile": getattr(self, "_strict_signature", None),
                "model": _model_fingerprint(model, parameters),
                "data": self._data_fingerprint(positions),
                "step": int(step_id),
                "budget": int(num_samples),
                "clusters": self.num_clusters,
                "projection": self.projection,
                "projection_backend": self.projection_backend,
                "dimension": self.proj_dim,
                "parameters": self.parameter_pattern,
                "seed": self.seed,
                "bins": self.histogram_bins,
                "trim": self.trim_fraction,
                "replacement": self.replacement,
                "collator": type(self.data_collator).__qualname__,
                "tokenizer_vocabulary": getattr(tokenizer, "tokenizer", tokenizer).get_vocab(),
                "padding_side": getattr(getattr(tokenizer, "tokenizer", tokenizer), "padding_side", None),
                "template": repr(getattr(self.data_collator, "template", None)),
                "image_processor": getattr(
                    getattr(self.data_collator, "processor", None), "image_processor", None
                ).to_dict()
                if getattr(getattr(self.data_collator, "processor", None), "image_processor", None) is not None
                else None,
            }
        )
        directory = Path(self.cache_dir) / f"step_{step_id:08d}_{key[:16]}"
        if dist.is_initialized():
            rank_keys = [None] * world
            dist.all_gather_object(rank_keys, key)
            if len(set(rank_keys)) != 1:
                raise ValueError("Adapt-inf model/data/config fingerprints differ across ranks")
        selection_path = directory / "selection.json"
        import json

        cached = None
        if rank == 0 and selection_path.exists():
            try:
                cached = json.loads(selection_path.read_text())
                ids = cached["selected_indices"]
                pool = set(positions)
                if (
                    cached["fingerprint"] != key
                    or not ids
                    or (self.selection_budget == "exact" and len(ids) != num_samples)
                    or any(p not in pool for p in ids)
                ):
                    raise ValueError("invalid cached selection")
                if not self.replacement and len(set(ids)) != len(ids):
                    raise ValueError("cached selection has duplicates")
            except (ValueError, KeyError, TypeError, OSError):
                cached = None
        cached = _broadcast(cached)
        if cached is not None:
            self.last_metadata = {**cached, "cache_hit": True}
            return cached["selected_indices"]
        rows = np.arange(rank, len(positions), world)
        dimension = self.proj_dim if self.projection != "none" else sum(p.numel() for _, p in parameters)
        # Per-rank disk arrays avoid all_gather of the full gradient pool. A
        # shared cache directory is required for distributed selection.
        feature_path = directory / f"rank_{rank}_features.npy"
        score_path = directory / f"rank_{rank}_scores.npy"
        feature = scores = None
        local = -1
        failure = None
        try:
            directory.mkdir(parents=True, exist_ok=True)
            feature = np.lib.format.open_memmap(
                feature_path, mode="w+", dtype=np.float32, shape=(len(rows), dimension)
            )
            scores = np.lib.format.open_memmap(score_path, mode="w+", dtype=np.float64, shape=(len(rows), 4))
            model_type = getattr(model.config, "model_type", None) if hasattr(model, "config") else None
            with scoring_state(model, checkpoint_llama=model_type == "llava"):
                write_json(
                    directory / f"rank_{rank}_progress.json",
                    {
                        "rank": rank,
                        "completed": 0,
                        "total": len(rows),
                        "step": step_id,
                        "model_type": model_type,
                        "checkpointed_training_layers": sum(
                            bool(getattr(module, "gradient_checkpointing", False) and module.training)
                            for module in model.modules()
                        ),
                    },
                )
                for local, row in enumerate(rows):
                    position = positions[int(row)]
                    sample = self.dataset[position]
                    model_type = getattr(model.config, "model_type", None) if hasattr(model, "config") else None
                    if not sample.get("images") or sample.get("videos") or sample.get("audios"):
                        raise ValueError("Author IG requires image-only training records")
                    batch = move_to_device(self.data_collator([dict(sample)]), self.accelerator.device)
                    batch.pop("__record_indices__", None)
                    batch["use_cache"] = False
                    with self.accelerator.autocast(), torch.enable_grad():
                        # We compute answer-only CE ourselves. Passing labels
                        # also builds an unused model loss graph.
                        outputs = model(**{key: value for key, value in batch.items() if key != "labels"})
                        stats = answer_statistics(outputs.logits, batch["labels"])
                        gradients = torch.autograd.grad(stats["loss"], [p for _, p in parameters], allow_unused=True)
                    gradients = [torch.zeros_like(p) if g is None else g for g, (_, p) in zip(gradients, parameters)]
                    if self.projection_backend == "cuda":
                        if self._gradient_projector is None:
                            projector = CachedCountSketch
                            if self.projection == "rademacher":
                                from dataflex.train.selector.strict_adapt_inf_selector import CachedRademacher

                                projector = CachedRademacher
                            self._gradient_projector = projector(
                                gradients, dimension, self.seed, self.accelerator.device
                            )
                        feature[local] = self._gradient_projector(gradients)
                    else:
                        feature[local] = project_gradients(gradients, dimension, self.seed, self.projection)
                    loss = float(stats["loss"].detach())
                    with torch.no_grad():
                        predictive_entropy = float(obtain_entropy_score(outputs, eval_batch_size=1))
                    del outputs, gradients
                    with self.accelerator.autocast(), torch.no_grad():
                        ig = float(author_zero_image_ig(model, batch, stats["loss"]))
                    scores[local] = [loss, stats["el2n"], predictive_entropy, ig]
                    if not np.isfinite(scores[local]).all() or not np.isfinite(feature[local]).all():
                        raise ValueError("nonfinite multimodal score/gradient")
                    del batch, stats
                    if local == 0 or (local + 1) % 100 == 0 or local + 1 == len(rows):
                        feature.flush()
                        scores.flush()
                        write_json(
                            directory / f"rank_{rank}_progress.json",
                            {
                                "rank": rank,
                                "completed": local + 1,
                                "total": len(rows),
                                "step": step_id,
                                "elapsed_seconds": time.monotonic() - started,
                                "last_dataset_position": position,
                            },
                        )
        except Exception as exc:
            sample_id = positions[int(rows[local])] if local >= 0 else "shard setup"
            failure = f"rank {rank}, sample {sample_id}: {type(exc).__name__}: {exc}"
            try:
                write_json(
                    directory / f"rank_{rank}_failure.json", {"error": failure, "traceback": traceback.format_exc()}
                )
            except OSError:
                pass
        finally:
            try:
                for array in (feature, scores):
                    if array is not None:
                        array.flush()
            except Exception as exc:
                failure = f"rank {rank}, shard flush: {type(exc).__name__}: {exc}"
            del feature, scores
        failures = [failure]
        if dist.is_initialized():
            failures = [None] * world
            dist.all_gather_object(failures, failure)
        if any(failures):
            raise RuntimeError("Adapt-inf scoring failed: " + "; ".join(f for f in failures if f))
        payload = None
        if rank == 0:
            try:
                features = np.lib.format.open_memmap(
                    directory / "features.npy", mode="w+", dtype=np.float32, shape=(len(positions), dimension)
                )
                all_scores = np.lib.format.open_memmap(
                    directory / "scores.npy", mode="w+", dtype=np.float64, shape=(len(positions), 4)
                )
                for worker in range(world):
                    features[worker::world] = np.load(directory / f"rank_{worker}_features.npy", mmap_mode="r")
                    all_scores[worker::world] = np.load(directory / f"rank_{worker}_scores.npy", mmap_mode="r")
                features.flush()
                all_scores.flush()
                # In-place normalization preserves the memmap rather than making
                # an additional N x D heap allocation.
                for start in range(0, len(features), 4096):
                    block = features[start : start + 4096]
                    block /= np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1e-12)
                if hasattr(self, "cluster_paper_features"):
                    labels = self.cluster_paper_features(features, directory)
                else:
                    labels = cluster_features(features, self.num_clusters, self.seed)
                np.save(directory / "cluster_ids.npy", labels)
                np.save(directory / "dataset_positions.npy", np.asarray(positions))
                cluster_ids, capacities = np.unique(labels, return_counts=True)
                quotas = allocate_budget(capacities, min(num_samples, len(positions)))
                rng = random.Random(self.seed + int(step_id))
                selected, details = [], []
                names = ["log_ppl", "el2n", "predictive_entropy", "ig"]
                for cluster_id, quota in zip(cluster_ids, quotas):
                    members = np.flatnonzero(labels == cluster_id)
                    entropy = [
                        score_distribution_entropy(all_scores[members, i], self.histogram_bins) for i in range(4)
                    ]
                    # collect_grads inserts the four released experts in this
                    # order. Preserve its tie/NaN decisions, not column order.
                    expert_order = [2, 1, 0, 3]  # entropy, EL2N, PPL, IG
                    expert = expert_order[int(np.argmax([entropy[i] for i in expert_order]))]
                    picked = (
                        author_ccs_indices(all_scores[members, expert], quota, rng, self.histogram_bins)
                        if quota
                        else []
                    )
                    selected.extend(positions[int(members[i])] for i in picked)
                    details.append(
                        {
                            "cluster": int(cluster_id),
                            "size": len(members),
                            "budget": quota,
                            "expert": names[expert],
                            "expert_entropies": {
                                name: float(value) if np.isfinite(value) else None
                                for name, value in zip(names, entropy)
                            },
                            "nonfinite_entropy_experts": [
                                name for name, value in zip(names, entropy) if not np.isfinite(value)
                            ],
                            "expert_argmax_rule": "upstream entropy/EL2N/PPL/IG order; numpy.argmax including NaN",
                            "author_ccs_selected_count": len(picked),
                        }
                    )
                author_selected = list(selected)
                selected, budget_adapter = fit_selection_budget(
                    selected, positions, min(num_samples, len(positions)), rng, self.selection_budget
                )
                repeat_count = max(0, num_samples - len(selected)) if self.replacement else 0
                if repeat_count:
                    selected.extend(rng.choices(selected, k=repeat_count))
                row_by_position = {position: row for row, position in enumerate(positions)}
                result = {
                    "fingerprint": key,
                    "selected_indices": selected,
                    "cache_hit": False,
                    "indices": selected,  # DataFlex selector_io interchange format
                    "metric": {
                        name: [float(all_scores[row_by_position[p], i]) for p in selected]
                        for i, name in enumerate(names)
                    },
                    "mode": "flex-adapted-periodic",
                    "projection": self.projection,
                    "scoring_profile": "author-scores-ccs-d1b1b25-v2",
                    "selection_budget": self.selection_budget,
                    "budget_adapter": budget_adapter,
                    "remaining_adaptations": [
                        "explicit exact budget only when configured",
                        "single-pool periodic schedule",
                        "configured trainable gradient subspace",
                    ],
                    "gradient_parameters": [n for n, _ in parameters],
                    "gradient_dimension": dimension,
                    "clusters": details,
                    "selection_seconds": time.monotonic() - started,
                    "repeat_count": repeat_count,
                    "world_size": world,
                    "gpu_memory_peak_bytes": torch.cuda.max_memory_allocated(self.accelerator.device)
                    if self.accelerator.device.type == "cuda"
                    else 0,
                    "score_statistics": {
                        name: {
                            "min": float(all_scores[:, i].min()),
                            "max": float(all_scores[:, i].max()),
                            "mean": float(all_scores[:, i].mean()),
                        }
                        for i, name in enumerate(names)
                    },
                }
                write_json(directory / "author_selected_indices.json", author_selected)
                write_json(directory / "selected_indices.json", selected)
                write_json(selection_path, result)
                payload = {"result": result}
            except Exception as exc:
                payload = {"error": f"{type(exc).__name__}: {exc}; distributed cache must be shared across ranks"}
        payload = _broadcast(payload)
        if "error" in payload:
            raise RuntimeError(payload["error"])
        self.last_metadata = payload["result"]
        return payload["result"]["selected_indices"]
