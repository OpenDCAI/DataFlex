# Vendored unmodified function bodies from the upstream adapt-inf source.
# MIT License
#
# Copyright (c) 2024 Adyasha Maharana
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Adapt-inf scoring and selection in DataFlex's periodic training schedule.

Uses the authors' Rademacher projector and CCS output without global record
exclusion or post-selection budget repair. The configured cluster count is
explicit; WSS diagnostics support the paper's 5..50 search without inventing
an automatic elbow rule.
"""

import contextlib
import hashlib
import io
import math
import random
import re
import time
import traceback
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from dataflex.core.registry import register_selector
from dataflex.utils.multimodal_scoring import (
    answer_statistics,
    move_to_device,
    scoring_state,
)
from dataflex.utils.multimodal_selection import (
    fingerprint,
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
        cluster_grid=(5, 10, 15, 20, 25, 30, 35, 40, 45, 50),
        proj_dim=8192,
        parameter_pattern=r"\.layers\.16\..*lora_",
        seed=42,
        projection_seed=0,
        histogram_bins=50,
        dataset_revision=None,
    ):
        super().__init__(dataset, accelerator, data_collator, cache_dir)
        self.cluster_grid = list(cluster_grid)
        if (
            type(num_clusters) is not int
            or num_clusters < 1
            or not self.cluster_grid
            or any(type(k) is not int or k < 1 for k in self.cluster_grid)
            or num_clusters not in self.cluster_grid
            or type(proj_dim) is not int
            or proj_dim < 1
            or histogram_bins != 50
        ):
            raise ValueError("Choose num_clusters from cluster_grid; projection must be positive and CCS uses 50 bins")
        self.num_clusters, self.proj_dim = num_clusters, proj_dim
        self.parameter_pattern, self.seed = str(parameter_pattern), int(seed)
        re.compile(self.parameter_pattern)
        self.projection_seed = int(projection_seed)
        self.histogram_bins = 50
        self.dataset_revision = dataset_revision
        self._gradient_projector = None
        self._projector_type = None
        self.last_metadata = None

    def checkpoint_signature(self):
        return fingerprint(
            {
                "data": self._data_fingerprint(self.candidate_positions()),
                "scoring_profile": "author-scores-ccs-d1b1b25-v2",
                "projection_backend": self._projection_backend(self.accelerator.device).__name__,
                "trak_version": version("traker"),
                "config": [
                    self.num_clusters,
                    self.proj_dim,
                    self.cluster_grid,
                    self.projection_seed,
                    self.parameter_pattern,
                    self.seed,
                    self.histogram_bins,
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
        if not positions or num_samples > len(positions):
            raise ValueError("requested budget exceeds candidate pool without replacement")
        tokenizer = kwargs.get("tokenizer") or getattr(self.data_collator, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("a tokenizer is required to identify the scoring configuration")
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
                "version": 8,
                "scoring_profile": "author-scores-ccs-d1b1b25-v2",
                "model": _model_fingerprint(model, parameters),
                "data": self._data_fingerprint(positions),
                "step": int(step_id),
                "budget": int(num_samples),
                "clusters": self.num_clusters,
                "cluster_grid": self.cluster_grid,
                "projection": "trak-rademacher",
                "projection_backend": self._projection_backend(self.accelerator.device).__name__,
                "trak_version": version("traker"),
                "projection_seed": self.projection_seed,
                "dimension": self.proj_dim,
                "parameters": self.parameter_pattern,
                "seed": self.seed,
                "bins": self.histogram_bins,
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
                    or any(p not in pool for p in ids)
                ):
                    raise ValueError("invalid cached selection")
                if len(set(ids)) != len(ids):
                    raise ValueError("cached selection has duplicates")
            except (ValueError, KeyError, TypeError, OSError):
                cached = None
        cached = _broadcast(cached)
        if cached is not None:
            self.last_metadata = {**cached, "cache_hit": True}
            return cached["selected_indices"]
        rows = np.arange(rank, len(positions), world)
        dimension = self.proj_dim
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
                    feature[local] = self._project_gradients(gradients)
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
            if self._gradient_projector is not None:
                self._gradient_projector.free_memory()
                self._gradient_projector = None
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
                labels = self._cluster_features(features, directory)
                np.save(directory / "cluster_ids.npy", labels)
                np.save(directory / "dataset_positions.npy", np.asarray(positions))
                cluster_ids, capacities = np.unique(labels, return_counts=True)
                quotas = author_cluster_budgets(capacities, num_samples)
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
                if not selected:
                    raise ValueError("Published CCS selected no samples; adjust the requested budget or scoring pool")
                if len(selected) != len(set(selected)) or not set(selected) <= set(positions):
                    raise ValueError("Invalid author selection indices")
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
                    "projection": "trak-rademacher",
                    "projection_seed": self.projection_seed,
                    "scoring_profile": "author-scores-ccs-d1b1b25-v2",
                    "requested_budget": int(num_samples),
                    "selected_count": len(selected),
                    "remaining_adaptations": [
                        "single-pool periodic schedule",
                        "configured trainable gradient subspace",
                    ],
                    "gradient_parameters": [n for n, _ in parameters],
                    "gradient_dimension": dimension,
                    "clusters": details,
                    "selection_seconds": time.monotonic() - started,
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

    def _project_gradients(self, gradients):
        """Use the same TRAK backend selection and seed as the released code."""
        if self._gradient_projector is None:
            from trak.projectors import ProjectionType

            device = gradients[0].device
            projector = self._projection_backend(device)
            self._gradient_projector = projector(
                grad_dim=sum(g.numel() for g in gradients),
                proj_dim=self.proj_dim,
                seed=self.projection_seed,
                proj_type=ProjectionType.rademacher,
                device=device,
                dtype=gradients[0].dtype,
                block_size=128,
                max_batch_size=16,
            )
        vector = torch.cat([g.detach().reshape(-1) for g in gradients]).to(torch.float16)
        return self._gradient_projector.project(vector[None], model_id=0)[0].float().cpu().numpy()

    def _projection_backend(self, device):
        if self._projector_type is None:
            try:
                from trak.projectors import BasicProjector, CudaProjector
            except ImportError as exc:
                raise ImportError("Adapt-inf requires dataflex[multimodal-selection]") from exc
            self._projector_type = BasicProjector
            if device.type == "cuda":
                try:
                    import fast_jl

                    num_sms = torch.cuda.get_device_properties(device.index).multi_processor_count
                    fast_jl.project_rademacher_8(torch.zeros(8, 1000, device=device), 512, 0, num_sms)
                    self._projector_type = CudaProjector
                except Exception:
                    # The authors also fall back to BasicProjector if fast_jl fails.
                    pass
        return self._projector_type

    def _cluster_features(self, features, directory):
        """Report the paper's WSS grid; use the explicitly configured k.

        Neither the paper nor released code specifies an automatic elbow rule.
        num_clusters is chosen from these diagnostics, as with the author's
        --n-clusters argument, rather than an additional selection heuristic.
        """
        from sklearn.cluster import MiniBatchKMeans

        if self.num_clusters > len(features):
            raise ValueError("num_clusters exceeds the scoring pool")
        diagnostics, chosen_labels = [], None
        for k in self.cluster_grid:
            if k > len(features):
                continue
            estimator = MiniBatchKMeans(n_clusters=k, batch_size=10240, random_state=self.seed)
            estimator.fit(features)
            labels = estimator.labels_
            wss = 0.0
            for start in range(0, len(features), 4096):
                block = features[start : start + 4096]
                delta = block - estimator.cluster_centers_[labels[start : start + len(block)]]
                wss += float(np.einsum("ij,ij->", delta, delta, dtype=np.float64))
            diagnostics.append({"k": k, "wss": wss})
            if k == self.num_clusters:
                chosen_labels = labels
        write_json(
            directory / "cluster_grid_selection.json",
            {"grid": diagnostics, "selected_k": self.num_clusters, "rule": "explicit num_clusters; inspect WSS grid"},
        )
        return chosen_labels


def obtain_entropy_score(outputs, eval_batch_size=None):
    # Entropy of the probability vector

    probs = torch.nn.Softmax(dim=-1)(outputs.logits)
    entropy = -1 * probs * torch.log(probs + 1e-10)
    # print(probs.shape, entropy.shape)
    entropy = torch.mean(torch.sum(entropy, dim=-1), dim=-1).detach().cpu().squeeze()
    # print('entropy', entropy.shape, entropy)
    return entropy

def get_dist_entropy(x, nbins=50, verbose=False):
    """
    x is assumed to be an (nsignals, nsamples) array containing integers between
    0 and n_unique_vals
    """

    # normalize with mean and std dev
    nan_count = np.count_nonzero(np.isnan(x))
    if verbose:
        x = x.astype(np.float32)
        print("Replacing %s nan values in input vector" % nan_count)
        print(x, np.mean(x), np.std(x))
    x = np.nan_to_num(x)
    
    x_std = np.std(x)
    x = (x - np.mean(x))/x_std

    counts = np.zeros(nbins)
    bins = np.linspace(-4*x_std, 4*x_std, nbins)
    if verbose:
        print(bins)
        print(x)
    for i in range(1, nbins):
        counts[i] = np.sum((x > bins[i-1]) & (x <= bins[i]))

    epsilon = np.finfo(float).eps
    # divide by number of columns to get the probability of each unique value
    p = counts / np.sum(counts)
    # Replace zeros with epsilon
    if verbose:
        print(p)
    p = np.where(p == 0, epsilon, p)
    if verbose:
        print(p)
    nan_count = np.count_nonzero(np.isnan(p))
    if verbose:
        print("Replacing %s nan values in prob. vector" % nan_count)
    p = np.nan_to_num(p)
    if verbose:
        print(p)
        print(np.log2(p))

    # compute Shannon entropy in bits
    return -np.sum(p * np.log2(p))

def obtain_ig_score(model, batch, ppl_with_image, eval_batch_size): #TODO: Fix

    if eval_batch_size == 1:
        batch['images'] = torch.zeros_like(batch['images'])
        # print(batch.keys())
        with torch.inference_mode():
            loss_no_image = model(**batch).loss
        ppl_wo_image = torch.exp(loss_no_image)
        ig_score = ppl_wo_image/ppl_with_image
        # print('ig_score', ig_score.shape, ig_score)
        return ig_score
    else:
        raise NotImplementedError

def set_outlier_threshold(scores):

    scores = np.sort(scores)
    low_thresh = scores[int(len(scores)*0.05)]
    high_thresh = scores[int(len(scores)*0.95)]
    print("Setting outlier threshold to %s (low) and %s (high)" % (low_thresh, high_thresh))
    return low_thresh, high_thresh

def get_ccs_samples(scores, bins, budget):

    # low, high = set_outlier_threshold(list(scores.values()))
    low, high = set_outlier_threshold(scores)
    right = high

    remaining = budget
    budget = math.ceil(budget/bins)

    selected_idxs = []
    for i in tqdm(range(bins), desc="Getting CCS samples"):
        left = right - ((high-low)/bins)
        # candidates = [d for d in data.keys() if d in scores and scores[d] <= right and scores[d] >= left]
        candidates = [j for j, score in enumerate(scores) if score > left and score <= right]
        if len(candidates) < budget:
            selected_idxs.extend(candidates)
            if (i+1) < bins:
                budget = math.ceil((remaining - len(candidates))/(bins-(i+1)))
                remaining -= len(candidates)
        else:
            selected_idxs.extend(random.sample(candidates, k=budget))
            remaining -= budget
        print("Selected %s samples from between %s and %s values" % (len(selected_idxs), left, right))
        right = left
    
    return np.array(selected_idxs)

class _ImageArgumentAdapter:
    def __init__(self, model):
        self.model = model

    def __call__(self, **batch):
        # LLaVA author's `images` and HF's `pixel_values` are input API names.
        # Keep all visual token positions and the same processed zero tensor.
        pixels = batch.pop("images")
        result = self.model(**{**batch, "pixel_values": pixels})
        if getattr(result, "loss", None) is not None:
            return result
        # Tiny model fixtures may expose logits only; real HF models return loss.
        return SimpleNamespace(loss=answer_statistics(result.logits, batch["labels"])["loss"])

def author_zero_image_ig(model, batch, image_loss):
    if "pixel_values" not in batch:
        raise ValueError("Author IG requires processed image pixels")
    inputs = {k: v for k, v in batch.items() if k != "pixel_values"}
    inputs["images"] = batch["pixel_values"]
    # The upstream function replaces only the shallow copy's image tensor.
    return obtain_ig_score(_ImageArgumentAdapter(model), inputs, image_loss.detach().exp(), 1)

def author_ccs_indices(scores, budget, rng, bins=50):
    """Published interval/ceil/redistribution rules, with isolated Python RNG."""
    scores = np.asarray(scores)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("CCS needs a finite nonempty vector")
    if type(budget) is not int or budget < 0 or bins < 1:
        raise ValueError("Invalid CCS budget/bins")
    saved = random.getstate()
    random.setstate(rng.getstate())
    try:
        # Suppress author console output, not any sampling operation.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = get_ccs_samples(scores, bins, budget)
    finally:
        rng.setstate(random.getstate())
        random.setstate(saved)
    return result.astype(np.int64).tolist()

def author_cluster_budgets(capacities, budget):
    """Released uniform cluster-budget allocation, including ceil rounding.

    Removes the upstream CLI-global args.n_clusters dependency only. The
    selected count is not subsequently truncated or filled to fit the budget.
    """
    capacities = np.asarray(capacities)
    clusters = len(capacities)
    if not clusters or np.any(capacities <= 0) or budget < 0:
        raise ValueError("Cluster capacities must be positive and budget nonnegative")
    if int(capacities.min()) * clusters >= budget:
        return [math.ceil(budget / clusters)] * clusters
    order = np.argsort(capacities)
    per_cluster = math.ceil(budget / clusters)
    budgets = []
    for i, index in enumerate(order):
        size = int(capacities[index])
        if size < per_cluster:
            budgets.append(size)
            if i + 1 < clusters:
                per_cluster += math.ceil((per_cluster - size) / (clusters - i - 1))
        else:
            budgets.append(per_cluster)
    return [budgets[index] for index in np.argsort(order)]

def score_distribution_entropy(scores, bins=50):
    """Published Adapt-inf get_dist_entropy, including its original bin edges.

    The author helper does not trim scores here. trim_fraction remains a CCS
    setting, rather than silently changing expert selection. Constant/empty-bin
    behavior is also inherited from the pinned upstream implementation.
    """

    values = np.asarray(scores)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or bins < 2:
        raise ValueError("expert scores must be a finite nonempty vector; bins >= 2")
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(get_dist_entropy(values, nbins=bins))
