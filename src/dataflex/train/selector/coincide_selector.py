import hashlib
import inspect
import math
import os
import shutil
import time
from typing import List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForImageTextToText
from transformers.integrations.deepspeed import (
    is_deepspeed_zero3_enabled,
    set_hf_deepspeed_config,
    unset_hf_deepspeed_config,
)

from dataflex.core.registry import register_selector
from dataflex.utils.logging import logger
from dataflex.utils.selector_io import load_cached_selection, save_selection

from .base_selector import Selector


def _move_to_device(batch, device):
    if isinstance(batch, dict):
        return {k: _move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [_move_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(_move_to_device(v, device) for v in batch)
    if hasattr(batch, "to"):
        return batch.to(device)
    return batch


@register_selector("coincide")
class CoincideSelector(Selector):
    """
    COINCIDE data selection (arXiv:2406.10995, EMNLP'24).

    An unsupervised coreset selector for visual instruction tuning. A small
    reference LVLM's internal activations group the training data into
    concept-skill clusters; each cluster then receives a sampling budget
    proportional to its transferability (cheap proxy: mean cosine similarity to
    other cluster centroids) and inversely to its density (diversity). Within a
    cluster, samples are drawn by greedily minimising the squared maximum mean
    discrepancy (MMD^2) to the cluster distribution.

    Unlike LESS/ICONS this needs no gradients and no eval_dataset: the
    __init__ deliberately omits an eval_dataset parameter, so the registry (which
    filters kwargs by signature) never injects one.

    Feature (paper Eq. 2-3): from M reference-model layers, tanh the hidden
    state, mean-pool visual and text tokens separately (image tokens located by
    input_ids == config.image_token_id, exactly as transformers injects visual
    features), L2-normalize each, concatenate the 2M blocks and divide by
    sqrt(2M). The paper pools the post-MSA activation; here we approximate it with
    the standard per-block hidden_states, which is architecture-agnostic and
    needs no forward hooks.
    """

    def __init__(
        self,
        dataset,
        accelerator,
        data_collator,
        cache_dir,
        embedding_model_name_or_path: Optional[str] = None,
        embedding_model_dtype: str = "auto",
        # Which hidden-state layers to pool. None -> M=5 layers evenly spaced
        # from the first to the top layer (paper default).
        embedding_layers: Optional[List[int]] = None,
        num_clusters: int = 10000,
        clustering_batch_size: int = 8,
        clustering_num_workers: int = 4,
        clustering_device: str = "auto",
        clustering_max_iter: int = 10,
        assignment_chunk_size: int = 4096,
        temperature: float = 1.0,
        density_chunk_size: int = 2048,
        single_member_density: float = 1.0,
        seed: int = 42,
    ):
        super().__init__(dataset, accelerator, data_collator, cache_dir)

        self.embedding_model_name_or_path = embedding_model_name_or_path or os.environ.get(
            "DATAFLEX_COINCIDE_EMBEDDING_MODEL"
        )
        self.embedding_model_dtype = embedding_model_dtype
        self.embedding_layers = embedding_layers
        self.num_clusters = num_clusters
        self.clustering_batch_size = clustering_batch_size
        self.clustering_num_workers = max(0, int(clustering_num_workers))
        self.clustering_device = str(clustering_device).lower()
        self.clustering_max_iter = clustering_max_iter
        self.assignment_chunk_size = assignment_chunk_size
        self.temperature = float(temperature)
        self.density_chunk_size = int(density_chunk_size)
        self.single_member_density = float(single_member_density)
        self.seed = seed

        self.device = self.accelerator.device

        # One-time, run-once artefacts memoised across select steps.
        self._embedding_model = None
        self._features: Optional[torch.Tensor] = None
        self._cluster_ids: Optional[torch.Tensor] = None
        self._resolved_layers: Optional[List[int]] = None

        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(
            f"[CoincideSelector] initialized. num_clusters={self.num_clusters}, "
            f"temperature={self.temperature}, data_workers={self.clustering_num_workers}. "
            f"Features/clusters cached in {self.cache_dir}"
        )

    # ------------------------------------------------------------------
    # Reference model + layer resolution
    # ------------------------------------------------------------------

    def _get_embedding_model(self, training_model):
        """Resolve the model used for embedding extraction.

        When ``embedding_model_name_or_path`` is set we lazily load that
        standalone LVLM (decoupled from training). Otherwise we fall back to the
        training model. A multimodal model MUST be loaded with
        ``AutoModelForImageTextToText`` so it consumes ``pixel_values``.
        """
        if self.embedding_model_name_or_path is None:
            return training_model
        if self._embedding_model is not None:
            return self._embedding_model
        dtype_map = {
            "auto": "auto",
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        dtype = dtype_map.get(self.embedding_model_dtype, "auto")
        ds_config = None
        if is_deepspeed_zero3_enabled():
            plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
            ds_config = getattr(plugin, "hf_ds_config", None)
            if ds_config is None:
                raise RuntimeError("Cannot isolate the COINCIDE reference model from ZeRO-3.")
            unset_hf_deepspeed_config()
        try:
            self._embedding_model = AutoModelForImageTextToText.from_pretrained(
                self.embedding_model_name_or_path,
                torch_dtype=dtype,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            ).to(self.device)
        finally:
            if ds_config is not None:
                set_hf_deepspeed_config(ds_config)
        self._embedding_model.eval()
        if self.accelerator.is_main_process:
            logger.info(
                f"[CoincideSelector] Loaded standalone embedding model from "
                f"{self.embedding_model_name_or_path} with FlashAttention-2"
            )
        return self._embedding_model

    def _model_config(self, model):
        """Return the underlying HF config, unwrapping any DDP / DeepSpeed engine. """
        unwrapped = self.accelerator.unwrap_model(model)
        config = getattr(unwrapped, "config", None)
        if config is None:
            config = getattr(model, "config", None)
        if config is None:
            raise ValueError(
                f"[CoincideSelector] Cannot find the Hugging Face config on "
                f"{type(model).__name__} or its unwrapped model."
            )
        return config

    def _resolve_layers(self, num_hidden_layers: int) -> List[int]:
        """Layers to pool. hidden_states has length num_hidden_layers+1, index 0
        being the embedding layer and index -1 the last transformer block."""
        if self._resolved_layers is not None:
            return self._resolved_layers

        if self.embedding_layers:
            resolved = []
            for l in self.embedding_layers:
                idx = int(l)
                if idx < 0:
                    idx += num_hidden_layers + 1
                if not (0 <= idx <= num_hidden_layers):
                    raise ValueError(
                        f"[CoincideSelector] embedding layer {l} out of range for a model "
                        f"with {num_hidden_layers} hidden layers."
                    )
                resolved.append(idx)
            resolved = sorted(set(resolved))
        else:
            # M=5 layers evenly spaced across the transformer blocks (skip the
            # index-0 embedding layer), covering first -> top per the paper.
            m = min(5, num_hidden_layers)
            resolved = torch.linspace(1, num_hidden_layers, m).round().long().unique().tolist()

        self._resolved_layers = resolved
        if self.accelerator.is_main_process:
            logger.info(f"[CoincideSelector] Pooling hidden_states layers: {resolved}")
        return resolved

    # ------------------------------------------------------------------
    # Multimodal feature extraction (paper Eq. 2-3)
    # ------------------------------------------------------------------

    def _embedding_cache_path(self) -> str:
        """Step-independent cache path. Embeddings depend only on the embedding
        model, the dataset size, and the set of pooled layers -- never on step."""
        emb_model_id = str(self.embedding_model_name_or_path or "train_model")
        emb_hash = hashlib.md5(emb_model_id.encode("utf-8")).hexdigest()[:10]
        n_samples = len(self.dataset)
        layers = self._resolved_layers if self._resolved_layers is not None else []
        layers_tag = "L" + "-".join(str(x) for x in sorted(layers))
        emb_root = os.environ.get(
            "DATAFLEX_EMBEDDING_CACHE_ROOT",
            os.path.join(self.cache_dir, "embeddings_shared"),
        )
        return os.path.join(
            emb_root,
            f"coincide_emb_{emb_hash}__N{n_samples}__{layers_tag}",
            "train_features.pt",
        )

    def _pool_multimodal_features(self, hidden_states, input_ids, attention_mask, image_token_id) -> torch.Tensor:
        """Split visual/text masked mean-pool over M layers (Eq. 2-3).

        Returns [B, 2*M*D] on CPU float32. Image tokens sit at
        ``input_ids == image_token_id`` (transformers replaces them in-place via
        masked_scatter, so hidden_states aligns token-for-token with input_ids).
        A pure-image or pure-text sample yields a zero vector for the missing
        modality, which is an accepted degeneracy.
        """
        image_mask = (input_ids == image_token_id)  # [B, S]
        if attention_mask is not None:
            valid = attention_mask.bool()
            text_mask = valid & ~image_mask
        else:
            text_mask = ~image_mask

        img_m = image_mask.unsqueeze(-1).to(torch.float32)  # [B, S, 1]
        txt_m = text_mask.unsqueeze(-1).to(torch.float32)
        img_denom = img_m.sum(dim=1).clamp(min=1.0)         # [B, 1]
        txt_denom = txt_m.sum(dim=1).clamp(min=1.0)

        blocks = []
        for l in self._resolved_layers:
            h = torch.tanh(hidden_states[l].float())        # [B, S, D]
            uv = (h * img_m).sum(dim=1) / img_denom          # [B, D]
            ut = (h * txt_m).sum(dim=1) / txt_denom
            uv = uv / uv.norm(dim=1, keepdim=True).clamp(min=1e-12)
            ut = ut / ut.norm(dim=1, keepdim=True).clamp(min=1e-12)
            blocks.append(uv)
            blocks.append(ut)

        m = len(self._resolved_layers)
        feat = torch.cat(blocks, dim=1) / math.sqrt(2 * m)   # [B, 2*M*D]
        return feat.detach().cpu().float()

    def _resolve_image_token_id(self, config) -> int:
        img_id = getattr(config, "image_token_id", None)
        if img_id is None:
            img_id = getattr(config, "image_token_index", None)
        if img_id is None:
            raise ValueError(
                "[CoincideSelector] Could not find image_token_id/image_token_index on the "
                "model config. COINCIDE needs a multimodal reference model."
            )
        return int(img_id)

    def _extract_embedding_features(self, model) -> torch.Tensor:
        # Embeddings are step-independent: cache once, reuse forever.
        feature_path = self._embedding_cache_path()

        cached = os.path.exists(feature_path) if self.accelerator.is_main_process else False
        cached = self._broadcast_bool(cached)
        if cached:
            if self.accelerator.is_main_process:
                logger.info(f"[CoincideSelector] Loading cached train features from {feature_path}")
            self.accelerator.wait_for_everyone()
            return torch.load(feature_path, map_location="cpu")

        os.makedirs(os.path.dirname(feature_path), exist_ok=True)
        emb_model = self._get_embedding_model(model)
        emb_config = self._model_config(emb_model)
        text_config = getattr(emb_config, "text_config", emb_config)
        num_hidden_layers = int(text_config.num_hidden_layers)
        self._resolve_layers(num_hidden_layers)
        image_token_id = self._resolve_image_token_id(emb_config)

        indexed_dataset = list(range(len(self.dataset)))

        def collate_indices(indices):
            examples = [self.dataset[int(i)] for i in indices]
            return torch.tensor(indices, dtype=torch.long), self.data_collator(examples)

        dataloader = DataLoader(
            indexed_dataset,
            batch_size=self.clustering_batch_size,
            shuffle=False,
            num_workers=self.clustering_num_workers,
            pin_memory=True,
            prefetch_factor=2 if self.clustering_num_workers > 0 else None,
            collate_fn=collate_indices,
        )
        # prepare() shards the dataloader across ranks so each rank embeds only
        # 1/world_size of the data.
        dataloader = self.accelerator.prepare(dataloader)

        was_training = emb_model.training
        emb_model.eval()
        base_model = self.accelerator.unwrap_model(emb_model)
        forward_params = inspect.signature(base_model.forward).parameters
        forward_kwargs = {"output_hidden_states": True, "return_dict": True}
        if "logits_to_keep" in forward_params:
            forward_kwargs["logits_to_keep"] = 1
        elif "num_logits_to_keep" in forward_params:
            forward_kwargs["num_logits_to_keep"] = 1
        features = []
        indices_seen = []
        with torch.no_grad():
            for indices, batch in tqdm(
                dataloader,
                desc=f"[Process {self.accelerator.process_index}] COINCIDE embeddings",
                disable=not self.accelerator.is_local_main_process,
                dynamic_ncols=True,
                position=self.accelerator.process_index,
            ):
                batch = _move_to_device(batch, self.device)
                # Selection only needs hidden states.
                model_inputs = {k: v for k, v in batch.items() if k != "labels"}
                outputs = emb_model(**model_inputs, **forward_kwargs)
                pooled = self._pool_multimodal_features(
                    outputs.hidden_states,
                    batch["input_ids"],
                    batch.get("attention_mask"),
                    image_token_id,
                )
                features.append(pooled)
                indices_seen.append(indices.cpu())
                del outputs, pooled, model_inputs, batch

        if was_training and self.embedding_model_name_or_path is None:
            emb_model.train()
        if self.embedding_model_name_or_path is not None:
            self._embedding_model = None
            del base_model, emb_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("[CoincideSelector] Released standalone embedding model from GPU memory.")

        features = torch.cat(features, dim=0) if features else torch.empty(0)
        indices_seen = torch.cat(indices_seen, dim=0) if indices_seen else torch.empty(0, dtype=torch.long)

        n_samples = len(self.dataset)
        distributed = dist.is_available() and dist.is_initialized()

        if n_samples == 0:
            raise ValueError("[CoincideSelector] Cannot extract features from an empty dataset.")

        if distributed:
            # Each rank wrote a disjoint shard (by original index). Gather them on
            # the main process via disk shards, then reorder by index.
            shard_dir = os.path.join(os.path.dirname(feature_path), "shards")
            os.makedirs(shard_dir, exist_ok=True)
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            shard_path = os.path.join(shard_dir, f"shard_rank{rank}.pt")
            torch.save({"features": features, "indices": indices_seen}, shard_path)
            del features, indices_seen
            dist.barrier()

            if self.accelerator.is_main_process:
                ordered = None
                for r in range(world_size):
                    shard_data = torch.load(
                        os.path.join(shard_dir, f"shard_rank{r}.pt"), map_location="cpu"
                    )
                    r_features = shard_data["features"]
                    r_indices = shard_data["indices"]
                    if r_features.numel() == 0:
                        continue
                    if ordered is None:
                        ordered = torch.empty(n_samples, r_features.shape[1], dtype=r_features.dtype)
                    ordered[r_indices] = r_features
                    del shard_data, r_features, r_indices
                # Guard the zero-vector degeneracy so spherical k-means gets unit
                # vectors (the assembled feature is unit-norm by construction).
                norms = ordered.norm(dim=1, keepdim=True).clamp(min=1e-12)
                ordered = ordered / norms
                torch.save(ordered, feature_path)
                logger.info(f"[CoincideSelector] Saved train features to {feature_path}")
            dist.barrier()
            if self.accelerator.is_main_process:
                shutil.rmtree(shard_dir, ignore_errors=True)
            ordered = torch.load(feature_path, map_location="cpu")
            return ordered

        # Single-process path
        if features.numel() == 0:
            raise ValueError(
                "[CoincideSelector] No features were extracted. Check dataset preprocessing."
            )
        ordered = torch.empty(n_samples, features.shape[1], dtype=features.dtype)
        ordered[indices_seen] = features
        norms = ordered.norm(dim=1, keepdim=True).clamp(min=1e-12)
        ordered = ordered / norms
        torch.save(ordered, feature_path)
        logger.info(f"[CoincideSelector] Saved train features to {feature_path}")
        return ordered

    def _get_features(self, model) -> torch.Tensor:
        """Lazily load (and memoise) the clustering features, once per run."""
        if self._features is None:
            self._features = self._extract_embedding_features(model)
        return self._features

    # ------------------------------------------------------------------
    # Spherical k-means clustering
    # ------------------------------------------------------------------

    def _resolve_num_clusters(self, n_samples: int) -> int:
        if n_samples == 0:
            raise ValueError("Cannot cluster an empty training dataset.")
        return max(1, min(int(self.num_clusters), n_samples))

    def _assign_to_centers(self, features: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        assignments = []
        for start in range(0, len(features), self.assignment_chunk_size):
            chunk = features[start:start + self.assignment_chunk_size]
            similarities = chunk @ centers.T
            assignments.append(similarities.argmax(dim=1))
        return torch.cat(assignments, dim=0)

    def _run_spherical_kmeans(self, features: torch.Tensor) -> torch.Tensor:
        use_cuda = self.clustering_device == "cuda" or (
            self.clustering_device == "auto" and self.device.type == "cuda"
        )
        if self.clustering_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("clustering_device must be one of: auto, cpu, cuda")
        if use_cuda:
            return self._run_spherical_kmeans_cuda(features)

        n_samples = len(features)
        k = self._resolve_num_clusters(n_samples)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        init_indices = torch.randperm(n_samples, generator=generator)[:k]
        centers = features[init_indices].clone()
        centers = centers / centers.norm(dim=1, keepdim=True).clamp(min=1e-12)

        for _ in range(max(1, self.clustering_max_iter)):
            assignments = self._assign_to_centers(features, centers)
            new_centers = torch.zeros_like(centers)
            counts = torch.bincount(assignments, minlength=k).float().unsqueeze(1)
            new_centers.index_add_(0, assignments, features)

            empty = counts.squeeze(1) == 0
            counts = counts.clamp(min=1.0)
            new_centers = new_centers / counts
            if empty.any():
                replacement = torch.randperm(n_samples, generator=generator)[: int(empty.sum().item())]
                new_centers[empty] = features[replacement]
            new_centers = new_centers / new_centers.norm(dim=1, keepdim=True).clamp(min=1e-12)
            centers = new_centers

        return self._assign_to_centers(features, centers).to(torch.long)

    def _run_spherical_kmeans_cuda(self, features: torch.Tensor) -> torch.Tensor:
        """Chunked fp16 cosine assignment on GPU with fp32 centroid updates."""
        if not torch.cuda.is_available():
            raise RuntimeError("clustering_device=cuda requested, but CUDA is unavailable")

        n_samples, feature_dim = features.shape
        k = self._resolve_num_clusters(n_samples)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        init_indices = torch.randperm(n_samples, generator=generator)[:k]
        centers = features[init_indices].to(self.device, dtype=torch.float16)
        centers = centers / centers.norm(dim=1, keepdim=True).clamp(min=1e-12)

        logger.info(
            f"[CoincideSelector] Running spherical k-means on {self.device} "
            f"(N={n_samples}, K={k}, D={feature_dim}, fp16 assignment)."
        )
        with torch.inference_mode():
            for iteration in range(max(1, self.clustering_max_iter)):
                new_centers = torch.zeros(k, feature_dim, device=self.device, dtype=torch.float32)
                counts = torch.zeros(k, device=self.device, dtype=torch.float32)
                for start in range(0, n_samples, self.assignment_chunk_size):
                    x = features[start:start + self.assignment_chunk_size].to(
                        self.device, dtype=torch.float16
                    )
                    assignments = (x @ centers.T).argmax(dim=1)
                    new_centers.index_add_(0, assignments, x.float())
                    counts += torch.bincount(assignments, minlength=k).float()
                    del x, assignments

                empty = counts == 0
                new_centers /= counts.clamp(min=1.0).unsqueeze(1)
                if empty.any():
                    replacement = torch.randperm(n_samples, generator=generator)[: int(empty.sum())]
                    new_centers[empty] = features[replacement].to(self.device)
                new_centers /= new_centers.norm(dim=1, keepdim=True).clamp(min=1e-12)
                centers = new_centers.to(torch.float16)
                logger.info(
                    f"[CoincideSelector] CUDA spherical k-means iteration "
                    f"{iteration + 1}/{self.clustering_max_iter} complete."
                )

            del new_centers, counts
            result = torch.empty(n_samples, dtype=torch.long)
            for start in range(0, n_samples, self.assignment_chunk_size):
                x = features[start:start + self.assignment_chunk_size].to(
                    self.device, dtype=torch.float16
                )
                result[start:start + len(x)] = (x @ centers.T).argmax(dim=1).cpu()
                del x

        del centers
        torch.cuda.empty_cache()
        return result

    # ------------------------------------------------------------------
    # Distributed helpers
    # ------------------------------------------------------------------

    def _broadcast_bool(self, value: bool) -> bool:
        if not (dist.is_available() and dist.is_initialized()):
            return value
        obj = [value if self.accelerator.is_main_process else None]
        dist.broadcast_object_list(obj, src=0)
        return bool(obj[0])

    def _broadcast_long_tensor(self, tensor: Optional[torch.Tensor]) -> torch.Tensor:
        if not (dist.is_available() and dist.is_initialized()):
            return tensor
        device = self.device
        length = torch.tensor(
            [tensor.numel() if tensor is not None else 0], dtype=torch.long, device=device
        )
        dist.broadcast(length, src=0)
        n = int(length.item())
        if self.accelerator.is_main_process:
            payload = tensor.to(device=device, dtype=torch.long)
        else:
            payload = torch.empty(n, dtype=torch.long, device=device)
        dist.broadcast(payload, src=0)
        return payload.cpu()

    def _get_or_create_clusters(self, model):
        # In-memory short-circuit: cluster exactly ONCE per run.
        if self._cluster_ids is not None:
            return self._cluster_ids, {
                "embedding_time_sec": 0.0,
                "clustering_time_sec": 0.0,
                "cluster_cache_hit": True,
            }

        cluster_dir = os.path.join(self.cache_dir, "cluster", "init")
        cluster_path = os.path.join(cluster_dir, "train_cluster_ids.pt")
        timing = {"embedding_time_sec": 0.0, "clustering_time_sec": 0.0, "cluster_cache_hit": False}

        clusters_cached = os.path.exists(cluster_path) if self.accelerator.is_main_process else False
        clusters_cached = self._broadcast_bool(clusters_cached)
        timing["cluster_cache_hit"] = clusters_cached

        # Only extract features if we actually need to (re)build clusters.
        if not clusters_cached:
            started = time.perf_counter()
            features = self._get_features(model)
            timing["embedding_time_sec"] = time.perf_counter() - started
        else:
            features = None

        if self.accelerator.is_main_process and clusters_cached:
            cluster_ids = torch.load(cluster_path, map_location="cpu")
            logger.info(f"[CoincideSelector] Loading cached clusters from {cluster_path}")
        elif self.accelerator.is_main_process:
            started = time.perf_counter()
            cluster_ids = self._run_spherical_kmeans(features)
            timing["clustering_time_sec"] = time.perf_counter() - started
            os.makedirs(cluster_dir, exist_ok=True)
            torch.save(cluster_ids, cluster_path)
            logger.info(
                f"[CoincideSelector] Built {int(cluster_ids.max().item()) + 1} clusters "
                f"for {len(cluster_ids)} train samples (spherical k-means)."
            )
        else:
            cluster_ids = None

        cluster_ids = self._broadcast_long_tensor(cluster_ids)
        self._cluster_ids = cluster_ids
        return cluster_ids, timing

    # ------------------------------------------------------------------
    # Transferability, density, budget, intra-cluster sampling
    # ------------------------------------------------------------------

    def _compute_centroids(self, features: torch.Tensor, cluster_ids: torch.Tensor, K: int) -> torch.Tensor:
        F = features.shape[1]
        centroids = torch.zeros(K, F, dtype=torch.float32)
        counts = torch.zeros(K, 1, dtype=torch.float32)
        centroids.index_add_(0, cluster_ids, features)
        counts.index_add_(0, cluster_ids, torch.ones(len(features), 1))
        centroids = centroids / counts.clamp(min=1.0)
        centroids = centroids / centroids.norm(dim=1, keepdim=True).clamp(min=1e-12)
        return centroids

    def _compute_transferability(self, centroids_norm: torch.Tensor) -> torch.Tensor:
        """S_i (Eq. 5): mean cosine similarity of centroid i to all other centroids."""
        K = centroids_norm.shape[0]
        if K == 1:
            return torch.zeros(1, dtype=torch.float32)
        sim = centroids_norm @ centroids_norm.T           # [K, K], diag ~ 1
        S = (sim.sum(dim=1) - 1.0) / max(K - 1, 1)
        return S

    def _compute_density(self, features: torch.Tensor, cluster_ids: torch.Tensor, K: int) -> torch.Tensor:
        """D_i (Eq. 6): mean over off-diagonal Gaussian kernel exp(-||u_p-u_q||^2)
        of all sample pairs in cluster i. Small D_i => diverse cluster."""
        D = torch.empty(K, dtype=torch.float32)
        for i in range(K):
            members = torch.where(cluster_ids == i)[0]
            n_i = len(members)
            if n_i <= 1:
                # A singleton has no pair; treat as maximally dense (exp(0)=1), so
                # exp(S/(tau*D)) shrinks its budget rather than crashing.
                D[i] = self.single_member_density
                continue
            X_i = features[members]
            sum_k = 0.0
            for start in range(0, n_i, self.density_chunk_size):
                block = X_i[start:start + self.density_chunk_size]
                dist2 = torch.cdist(block, X_i).pow(2)     # [b, n_i]
                sum_k += torch.exp(-dist2).sum().item()
            # subtract the n_i self-pairs (exp(0)=1 each); ordered off-diagonal count.
            D[i] = (sum_k - n_i) / (n_i * (n_i - 1))
        return D

    def _allocate_budget(self, S: torch.Tensor, D: torch.Tensor, N_core: int, cluster_sizes: torch.Tensor) -> torch.Tensor:
        """Per-cluster budget (Eq. 6-7): P_i propto exp(S_i/(tau*D_i)); take
        round(N_core * P_i), with largest-remainder rounding and capacity
        redistribution so the total is exactly min(N_core, N)."""
        K = S.shape[0]
        logits = S / (self.temperature * D.clamp(min=1e-12))
        P = torch.softmax(logits, dim=0)                   # == exp(S/(tau*D)) normalized

        def largest_remainder(prob: torch.Tensor, budget: int, cap: torch.Tensor) -> torch.Tensor:
            """Distribute `budget` over clusters proportional to `prob`, capped by
            `cap`, deterministic (ties -> lower index)."""
            if budget <= 0 or prob.sum() <= 0:
                return torch.zeros(K, dtype=torch.long)
            prob = prob / prob.sum()
            target = prob * budget
            base = torch.floor(target).to(torch.long)
            base = torch.minimum(base, cap)
            remaining = budget - int(base.sum().item())
            if remaining > 0:
                frac = target - torch.floor(target)
                # only clusters with spare capacity can take a remainder unit
                spare = (cap - base) > 0
                frac = torch.where(spare, frac, torch.full_like(frac, -1.0))
                # deterministic: sort by (-frac, index)
                order = sorted(range(K), key=lambda j: (-float(frac[j]), j))
                for j in order:
                    if remaining <= 0:
                        break
                    if frac[j] < 0:
                        break
                    base[j] += 1
                    remaining -= 1
            return base

        cap = cluster_sizes.to(torch.long)
        alloc = largest_remainder(P, N_core, cap)
        # Redistribute any shortfall (from clamping) until the budget is met or no
        # capacity remains. Each pass strictly reduces the shortfall.
        for _ in range(K):
            over = N_core - int(alloc.sum().item())
            if over <= 0:
                break
            spare = cap - alloc
            if int((spare > 0).sum().item()) == 0:
                break
            extra = largest_remainder(P * (spare > 0).float(), over, spare)
            alloc = alloc + extra
        return alloc

    def _mmd_greedy_select(self, X_i: torch.Tensor, m: int) -> List[int]:
        """Greedily pick m of n_i members minimising MMD^2(Ci, C'i) (Eq. 7).

        MMD^2 = A(Ci,Ci) + A(C'i,C'i) - 2 A(Ci,C'i), A(X,Y)=mean_{p,q} d(p,q),
        d(p,q)=exp(-||u_p-u_q||^2). A(Ci,Ci) is constant, so we minimise
        A(C'i,C'i) - 2 A(Ci,C'i). Incremental updates keep each step O(n_i).
        """
        n_i = X_i.shape[0]
        if m >= n_i:
            return list(range(n_i))

        # Full kernel matrix (chunk the build if a cluster is very large).
        if n_i <= self.density_chunk_size:
            Kmat = torch.exp(-torch.cdist(X_i, X_i).pow(2))
        else:
            Kmat = torch.empty(n_i, n_i, dtype=torch.float32)
            for start in range(0, n_i, self.density_chunk_size):
                block = X_i[start:start + self.density_chunk_size]
                Kmat[start:start + block.shape[0]] = torch.exp(-torch.cdist(block, X_i).pow(2))

        colsum = Kmat.sum(dim=0)                # [n_i], sum over p of Kmat[p, j]
        diag = torch.diagonal(Kmat)             # [n_i]

        selected: List[int] = []
        chosen_mask = torch.zeros(n_i, dtype=torch.bool)
        crossK = torch.zeros(n_i, dtype=torch.float32)   # sum_{p in S'} Kmat[:, p]
        sum_within = 0.0

        for _ in range(m):
            t = len(selected) + 1
            # A(C'i, C'i) for each candidate j
            a_within = (sum_within + 2.0 * crossK + diag) / (t * t)   # [n_i]
            # A(Ci, C'i): (sum over members->S' + col of j) / (n_i * t)
            a_cross = (crossK.sum() + colsum) / (n_i * t)             # [n_i]
            score = a_within - 2.0 * a_cross                          # minimise (A_full dropped)
            score = score.masked_fill(chosen_mask, float("inf"))
            j = int(torch.argmin(score).item())

            selected.append(j)
            chosen_mask[j] = True
            sum_within += 2.0 * float(crossK[j].item()) + float(diag[j].item())
            crossK += Kmat[:, j]

        return selected

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(self, model, step_id: int, num_samples: int, **kwargs) -> List[int]:
        os.makedirs(self.cache_dir, exist_ok=True)
        save_path = os.path.join(self.cache_dir, f"step_{step_id}.json")

        selection_cached = os.path.exists(save_path) if self.accelerator.is_main_process else False
        selection_cached = self._broadcast_bool(selection_cached)
        if selection_cached:
            if self.accelerator.is_main_process:
                cached_indices, _ = load_cached_selection(save_path)
            else:
                cached_indices = None
            cached = self._broadcast_long_tensor(
                torch.tensor(cached_indices, dtype=torch.long) if cached_indices is not None else None
            )
            return cached.tolist()

        select_started = time.perf_counter()
        cluster_ids, timing = self._get_or_create_clusters(model)
        K = int(cluster_ids.max().item()) + 1
        features = self._get_features(model)

        if self.accelerator.is_main_process:
            started = time.perf_counter()
            centroids = self._compute_centroids(features, cluster_ids, K)
            S = self._compute_transferability(centroids)
            D = self._compute_density(features, cluster_ids, K)
            timing["scoring_time_sec"] = time.perf_counter() - started

            N_core = min(num_samples, len(features))
            cluster_sizes = torch.bincount(cluster_ids, minlength=K)
            budget = self._allocate_budget(S, D, N_core, cluster_sizes)

            started = time.perf_counter()
            selected_indices: List[int] = []
            for i in range(K):
                m = int(budget[i].item())
                if m <= 0:
                    continue
                members = torch.where(cluster_ids == i)[0]
                X_i = features[members]
                chosen_local = self._mmd_greedy_select(X_i, m)
                selected_indices.extend(members[torch.tensor(chosen_local, dtype=torch.long)].tolist())
            timing["sampling_time_sec"] = time.perf_counter() - started
            timing["total_select_time_sec"] = time.perf_counter() - select_started

            metric_payload = {
                "num_clusters": int(K),
                "temperature": float(self.temperature),
                "selected_layers": list(self._resolved_layers or []),
                "budget_per_cluster": [int(b) for b in budget.tolist()],
                "N_core": int(N_core),
                "transferability_summary": {
                    "min": float(S.min().item()), "mean": float(S.mean().item()), "max": float(S.max().item()),
                },
                "density_summary": {
                    "min": float(D.min().item()), "mean": float(D.mean().item()), "max": float(D.max().item()),
                },
                "timing": timing,
            }
            save_selection(save_path, selected_indices, metric_payload, self.accelerator)
            logger.info(
                f"[CoincideSelector] Selected {len(selected_indices)} samples from {K} clusters "
                f"(transferability/density-weighted, MMD^2 greedy intra-cluster)."
            )
        else:
            selected_indices = None

        selected = self._broadcast_long_tensor(
            torch.tensor(selected_indices, dtype=torch.long) if selected_indices is not None else None
        )
        return selected.tolist()
