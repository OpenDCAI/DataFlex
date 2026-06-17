import hashlib
import math
import os
import shutil
import time
from typing import List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from dataflex.core.registry import register_selector
from dataflex.utils.logging import logger
from dataflex.utils.selector_io import load_cached_selection, save_selection

from .less_selector import LessSelector


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


@register_selector("cluster_less")
class ClusterLessSelector(LessSelector):
    """
    Cluster-accelerated LESS selector.

    The original LESS selector computes one gradient per training example. This
    variant first clusters the training set with no-gradient embedding features,
    samples a few representatives from each cluster, averages their projected
    gradients, and scores the whole cluster against validation gradients.
    """

    def __init__(
        self,
        dataset,
        eval_dataset,
        accelerator,
        data_collator,
        cache_dir,
        gradient_type: str = "adam",
        proj_dim: int = 4096,
        save_interval: int = 16,
        seed: int = 42,
        cluster_size: int = 64,
        num_clusters: Optional[int] = None,
        samples_per_cluster: int = 3,
        clustering_batch_size: int = 8,
        clustering_max_iter: int = 20,
        assignment_chunk_size: int = 4096,
        clustering_method: str = "kmeans",
        representative_strategy: str = "random",
        lsh_num_bits: Optional[int] = None,
        # Optional standalone model used ONLY for the one-time clustering
        # embedding extraction. When set, clustering is decoupled from the
        # training model: embeddings are extracted once with this model and
        # the cluster assignment is frozen for the whole run. When None we
        # fall back to the legacy behaviour (use the training model).
        embedding_model_name_or_path: Optional[str] = None,
        embedding_model_dtype: str = "auto",
        # Which transformer layer's hidden states to use as embedding.
        # -1 = last layer (default). Note: when no standalone embedding model
        # is given we use the TRAINING model itself for embedding extraction.
        embedding_layer: int = -1,
    ):
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            accelerator=accelerator,
            data_collator=data_collator,
            cache_dir=cache_dir,
            gradient_type=gradient_type,
            proj_dim=proj_dim,
            save_interval=save_interval,
            seed=seed,
        )
        self.cluster_size = cluster_size
        self.num_clusters = num_clusters
        self.samples_per_cluster = samples_per_cluster
        self.clustering_batch_size = clustering_batch_size
        self.clustering_max_iter = clustering_max_iter
        self.assignment_chunk_size = assignment_chunk_size
        self.clustering_method = clustering_method
        self.representative_strategy = representative_strategy
        self.lsh_num_bits = lsh_num_bits
        self.embedding_model_name_or_path = embedding_model_name_or_path
        self.embedding_model_dtype = embedding_model_dtype
        self.embedding_layer = int(embedding_layer)

        # One-time clustering artefacts, cached in-memory across select steps
        # so we cluster exactly once per run.
        self._embedding_model = None
        self._cluster_ids: Optional[torch.Tensor] = None
        self._features: Optional[torch.Tensor] = None

    def _get_embedding_model(self, training_model):
        """Resolve the model used for embedding extraction.

        When ``embedding_model_name_or_path`` is set we lazily load that
        standalone model (decoupled from training). Otherwise we fall back to
        the training model passed in (legacy behaviour).
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
        self._embedding_model = AutoModelForCausalLM.from_pretrained(
            self.embedding_model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self.device)
        self._embedding_model.eval()
        if self.accelerator.is_main_process:
            logger.info(
                f"[ClusterLessSelector] Loaded standalone embedding model from "
                f"{self.embedding_model_name_or_path}"
            )
        return self._embedding_model

    def _embedding_cache_path(self) -> str:
        """Step-independent cache path for the one-time clustering embeddings.

        The embeddings only depend on (embedding model, dataset size, layer),
        NOT on the training step, so the path deliberately omits step_id. This
        lets the clustering run exactly once and be reused across steps/runs.
        """
        emb_model_id = str(self.embedding_model_name_or_path or "train_model")
        emb_hash = hashlib.md5(emb_model_id.encode("utf-8")).hexdigest()[:10]
        n_samples = len(self.dataset)
        emb_root = os.environ.get(
            "DATAFLEX_EMBEDDING_CACHE_ROOT",
            os.path.join(self.cache_dir, "embeddings_shared"),
        )
        return os.path.join(
            emb_root,
            f"emb_{emb_hash}__N{n_samples}__L{self.embedding_layer}",
            "train_embeddings.pt",
        )

    def _extract_embedding_features(self, model) -> torch.Tensor:
        # Embeddings are step-independent: cache once, reuse forever.
        feature_path = self._embedding_cache_path()

        cached = os.path.exists(feature_path) if self.accelerator.is_main_process else False
        cached = self._broadcast_bool(cached)
        if cached:
            if self.accelerator.is_main_process:
                logger.info(f"[ClusterLessSelector] Loading cached train embeddings from {feature_path}")
            self.accelerator.wait_for_everyone()
            return torch.load(feature_path, map_location="cpu")

        os.makedirs(os.path.dirname(feature_path), exist_ok=True)
        emb_model = self._get_embedding_model(model)
        indexed_dataset = list(range(len(self.dataset)))

        def collate_indices(indices):
            examples = [self.dataset[int(i)] for i in indices]
            return torch.tensor(indices, dtype=torch.long), self.data_collator(examples)

        dataloader = DataLoader(
            indexed_dataset,
            batch_size=self.clustering_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_indices,
        )
        # IMPORTANT: prepare() shards the dataloader across ranks so each rank
        # only embeds 1/world_size of the data (instead of every rank redundantly
        # embedding the whole dataset).
        dataloader = self.accelerator.prepare(dataloader)

        was_training = emb_model.training
        emb_model.eval()
        features = []
        indices_seen = []
        with torch.no_grad():
            for indices, batch in tqdm(
                dataloader,
                desc=f"[Process {self.accelerator.process_index}] Cluster embeddings",
                disable=not self.accelerator.is_local_main_process,
                dynamic_ncols=True,
                position=self.accelerator.process_index,
            ):
                batch = _move_to_device(batch, self.device)
                outputs = emb_model(**batch, output_hidden_states=True, return_dict=True)
                # hidden_states[0] = embedding, [-1] = last layer.
                hidden = outputs.hidden_states[self.embedding_layer]
                attention_mask = batch.get("attention_mask")
                if attention_mask is None:
                    pooled = hidden.mean(dim=1)
                else:
                    mask = attention_mask.to(hidden.device).unsqueeze(-1).float()
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                features.append(pooled.detach().float().cpu())
                indices_seen.append(indices.cpu())

        if was_training and self.embedding_model_name_or_path is None:
            emb_model.train()

        features = torch.cat(features, dim=0) if features else torch.empty(0)
        indices_seen = torch.cat(indices_seen, dim=0) if indices_seen else torch.empty(0, dtype=torch.long)

        n_samples = len(self.dataset)
        distributed = dist.is_available() and dist.is_initialized()

        if n_samples == 0:
            raise ValueError(
                "[ClusterLessSelector] Cannot extract embedding features from an empty dataset."
            )

        if distributed:
            # Each rank wrote a disjoint shard (by original index). Gather them
            # on the main process via disk shards, then reorder by index.
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
                norms = ordered.norm(dim=1, keepdim=True).clamp(min=1e-12)
                ordered = ordered / norms
                torch.save(ordered, feature_path)
                logger.info(f"[ClusterLessSelector] Saved train embeddings to {feature_path}")
            dist.barrier()
            if self.accelerator.is_main_process:
                shutil.rmtree(shard_dir, ignore_errors=True)
            ordered = torch.load(feature_path, map_location="cpu")
            return ordered

        # Single-process path
        if features.numel() == 0:
            raise ValueError(
                "[ClusterLessSelector] No embedding features were extracted. "
                "Please check dataset preprocessing."
            )
        ordered = torch.empty(n_samples, features.shape[1], dtype=features.dtype)
        ordered[indices_seen] = features
        norms = ordered.norm(dim=1, keepdim=True).clamp(min=1e-12)
        ordered = ordered / norms
        torch.save(ordered, feature_path)
        logger.info(f"[ClusterLessSelector] Saved train embeddings to {feature_path}")
        return ordered

    def _get_features(self, model) -> torch.Tensor:
        """Lazily load (and memoise) the clustering embeddings, once per run.

        Whether the clusters were built fresh or loaded from cache, the
        embeddings live at the same step-independent path, so this is the single
        entry point that guarantees exactly one extraction/load per run.
        """
        if self._features is None:
            self._features = self._extract_embedding_features(model)
        return self._features

    def _resolve_num_clusters(self, n_samples: int) -> int:
        if n_samples == 0:
            raise ValueError("Cannot cluster an empty training dataset.")
        if self.num_clusters is not None:
            k = int(self.num_clusters)
        else:
            k = int(math.ceil(n_samples / max(1, self.cluster_size)))
        return max(1, min(k, n_samples))

    def _assign_to_centers(
        self,
        features: torch.Tensor,
        centers: torch.Tensor,
        spherical: bool = False,
    ) -> torch.Tensor:
        assignments = []
        for start in range(0, len(features), self.assignment_chunk_size):
            chunk = features[start:start + self.assignment_chunk_size]
            if spherical:
                similarities = chunk @ centers.T
                assignments.append(similarities.argmax(dim=1))
            else:
                distances = torch.cdist(chunk, centers)
                assignments.append(distances.argmin(dim=1))
        return torch.cat(assignments, dim=0)

    def _run_lloyd_kmeans(self, features: torch.Tensor, spherical: bool) -> torch.Tensor:
        n_samples = len(features)
        k = self._resolve_num_clusters(n_samples)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        init_indices = torch.randperm(n_samples, generator=generator)[:k]
        centers = features[init_indices].clone()
        if spherical:
            centers = centers / centers.norm(dim=1, keepdim=True).clamp(min=1e-12)

        for _ in range(max(1, self.clustering_max_iter)):
            assignments = self._assign_to_centers(features, centers, spherical=spherical)
            new_centers = torch.zeros_like(centers)
            counts = torch.bincount(assignments, minlength=k).float().unsqueeze(1)
            new_centers.index_add_(0, assignments, features)

            empty = counts.squeeze(1) == 0
            counts = counts.clamp(min=1.0)
            new_centers = new_centers / counts
            if empty.any():
                replacement = torch.randperm(n_samples, generator=generator)[: int(empty.sum().item())]
                new_centers[empty] = features[replacement]
            if spherical:
                new_centers = new_centers / new_centers.norm(dim=1, keepdim=True).clamp(min=1e-12)
            centers = new_centers

        return self._assign_to_centers(features, centers, spherical=spherical).to(torch.long)

    def _run_farthest_first(self, features: torch.Tensor) -> torch.Tensor:
        n_samples = len(features)
        k = self._resolve_num_clusters(n_samples)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)

        first = int(torch.randint(n_samples, (1,), generator=generator).item())
        center_indices = [first]
        min_distances = torch.cdist(features, features[first:first + 1]).squeeze(1)
        for _ in range(1, k):
            # Mask already-chosen centers so we never pick the same point twice
            # (their min_distance is 0 but ties could otherwise re-select them).
            scores = min_distances.clone()
            scores[center_indices] = -float("inf")
            next_index = int(scores.argmax().item())
            center_indices.append(next_index)
            distances = torch.cdist(features, features[next_index:next_index + 1]).squeeze(1)
            min_distances = torch.minimum(min_distances, distances)

        centers = features[torch.tensor(center_indices, dtype=torch.long)]
        return self._assign_to_centers(features, centers).to(torch.long)

    def _run_random_projection_lsh(self, features: torch.Tensor) -> torch.Tensor:
        n_samples = len(features)
        k = self._resolve_num_clusters(n_samples)
        bits = self.lsh_num_bits or max(1, int(math.ceil(math.log2(k))))
        bits = min(bits, 30)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + 31)
        planes = torch.randn(features.shape[1], bits, generator=generator, dtype=features.dtype)
        signatures = (features @ planes >= 0).to(torch.long)
        weights = (2 ** torch.arange(bits, dtype=torch.long)).unsqueeze(0)
        buckets = (signatures * weights).sum(dim=1)

        unique_buckets, inverse = torch.unique(buckets, sorted=True, return_inverse=True)
        if len(unique_buckets) <= k:
            return inverse.to(torch.long)
        return (inverse % k).to(torch.long)

    def _run_random_partition(self, features: torch.Tensor) -> torch.Tensor:
        n_samples = len(features)
        k = self._resolve_num_clusters(n_samples)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + 47)
        order = torch.randperm(n_samples, generator=generator)
        assignments = torch.empty(n_samples, dtype=torch.long)
        assignments[order] = torch.arange(n_samples, dtype=torch.long) % k
        return assignments

    def _run_clustering(self, features: torch.Tensor) -> torch.Tensor:
        method = self.clustering_method.lower()
        if method in {"kmeans", "euclidean_kmeans"}:
            return self._run_lloyd_kmeans(features, spherical=False)
        if method in {"spherical_kmeans", "cosine_kmeans"}:
            return self._run_lloyd_kmeans(features, spherical=True)
        if method in {"farthest_first", "kcenter", "k_center"}:
            return self._run_farthest_first(features)
        if method in {"random_projection_lsh", "lsh", "rp_lsh"}:
            return self._run_random_projection_lsh(features)
        if method in {"random_partition", "random"}:
            return self._run_random_partition(features)
        raise ValueError(f"Unknown clustering_method: {self.clustering_method}")

    def _broadcast_bool(self, value: bool) -> bool:
        """Broadcast a boolean flag from rank 0 to all ranks.

        Any cache-existence check that gates a *collective* code path must be
        decided by rank 0 alone and broadcast, otherwise ranks can diverge
        (e.g. a stale/laggy shared FS) and deadlock on mismatched collectives.
        """
        if not (dist.is_available() and dist.is_initialized()):
            return value
        obj = [value if self.accelerator.is_main_process else None]
        dist.broadcast_object_list(obj, src=0)
        return bool(obj[0])

    def _broadcast_long_tensor(self, tensor: Optional[torch.Tensor]) -> torch.Tensor:
        """Broadcast a 1-D long tensor from rank 0 efficiently.

        Avoids broadcast_object_list (pickle) on potentially huge lists by
        broadcasting the length first, then the raw tensor payload.
        """
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
        # 0) In-memory short-circuit: cluster exactly ONCE per run.
        if self._cluster_ids is not None:
            return self._cluster_ids, {
                "embedding_time_sec": 0.0,
                "clustering_time_sec": 0.0,
                "cluster_cache_hit": True,
            }

        # Step-independent on-disk cache: clustering is frozen for the run.
        cluster_dir = os.path.join(self.cache_dir, "cluster", "init")
        cluster_path = os.path.join(cluster_dir, "train_cluster_ids.pt")
        timing = {"embedding_time_sec": 0.0, "clustering_time_sec": 0.0, "cluster_cache_hit": False}

        clusters_cached = os.path.exists(cluster_path) if self.accelerator.is_main_process else False
        clusters_cached = self._broadcast_bool(clusters_cached)
        timing["cluster_cache_hit"] = clusters_cached

        # Only extract embeddings if we actually need to (re)build clusters.
        if not clusters_cached:
            started = time.perf_counter()
            features = self._get_features(model)
            timing["embedding_time_sec"] = time.perf_counter() - started
        else:
            features = None

        if self.accelerator.is_main_process and clusters_cached:
            cluster_ids = torch.load(cluster_path, map_location="cpu")
            logger.info(f"[ClusterLessSelector] Loading cached clusters from {cluster_path}")
        elif self.accelerator.is_main_process:
            started = time.perf_counter()
            # Clustering is run exactly once per run (step-independent), so the
            # RNG is seeded purely from self.seed inside each clustering method.
            cluster_ids = self._run_clustering(features)
            timing["clustering_time_sec"] = time.perf_counter() - started
            os.makedirs(cluster_dir, exist_ok=True)
            torch.save(cluster_ids, cluster_path)
            logger.info(
                f"[ClusterLessSelector] Built {int(cluster_ids.max().item()) + 1} clusters "
                f"for {len(cluster_ids)} train samples with method={self.clustering_method}."
            )
        else:
            cluster_ids = None

        cluster_ids = self._broadcast_long_tensor(cluster_ids)
        # Freeze in memory so subsequent select steps reuse it directly.
        self._cluster_ids = cluster_ids
        return cluster_ids, timing

    def _rank_members_by_strategy(
        self,
        members: torch.Tensor,
        features: Optional[torch.Tensor],
        generator: torch.Generator,
    ) -> torch.Tensor:
        strategy = self.representative_strategy.lower()
        if strategy in {"random", "rand"} or features is None:
            return members[torch.randperm(len(members), generator=generator)]

        member_features = features[members]
        centroid = member_features.mean(dim=0, keepdim=True)
        distances = torch.cdist(member_features, centroid).squeeze(1)

        if strategy in {"nearest", "nearest_to_centroid", "center", "centroid"}:
            order = torch.argsort(distances, descending=False)
            return members[order]
        if strategy in {"farthest", "farthest_from_centroid", "boundary"}:
            order = torch.argsort(distances, descending=True)
            return members[order]
        if strategy in {"mixed", "mixed_center_boundary", "center_boundary"}:
            near = torch.argsort(distances, descending=False).tolist()
            far = torch.argsort(distances, descending=True).tolist()
            merged = []
            seen = set()
            for left, right in zip(near, far):
                for idx in (left, right):
                    if idx not in seen:
                        seen.add(idx)
                        merged.append(idx)
            return members[torch.tensor(merged, dtype=torch.long)]

        raise ValueError(f"Unknown representative_strategy: {self.representative_strategy}")

    def _sample_representatives(
        self,
        cluster_ids: torch.Tensor,
        step_id: int,
        features: Optional[torch.Tensor] = None,
    ) -> List[int]:
        representatives = []
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + int(step_id) + 17)

        for cluster_id in range(int(cluster_ids.max().item()) + 1):
            members = torch.where(cluster_ids == cluster_id)[0]
            if len(members) == 0:
                continue
            take = min(self.samples_per_cluster, len(members))
            ranked_members = self._rank_members_by_strategy(members, features, generator)
            representatives.extend(ranked_members[:take].tolist())

        return representatives

    def select(self, model, step_id: int, num_samples: int, **kwargs) -> List[int]:
        os.makedirs(self.cache_dir, exist_ok=True)
        save_path = os.path.join(self.cache_dir, f"step_{step_id}.json")
        # Decide cache hits on rank 0 only, then broadcast: every branch below
        # runs collectives, so all ranks must agree to avoid deadlock.
        selection_cached = os.path.exists(save_path) if self.accelerator.is_main_process else False
        selection_cached = self._broadcast_bool(selection_cached)
        if selection_cached:
            if self.accelerator.is_main_process:
                cached_indices, _ = load_cached_selection(save_path)
            else:
                cached_indices = None
            cached = self._broadcast_long_tensor(
                torch.tensor(cached_indices, dtype=torch.long)
                if cached_indices is not None
                else None
            )
            return cached.tolist()

        select_started = time.perf_counter()
        # Cluster ONCE (in-memory / disk cached); subsequent steps reuse it.
        cluster_ids, timing = self._get_or_create_clusters(model)
        features = None
        if self.representative_strategy.lower() not in {"random", "rand"}:
            started = time.perf_counter()
            # Reuse the frozen clustering embeddings (loaded once per run and
            # memoised on self._features). On a cluster-cache hit they were not
            # materialised, so load them lazily here exactly once.
            features = self._get_features(model)
            timing["representative_feature_load_time_sec"] = time.perf_counter() - started
        else:
            timing["representative_feature_load_time_sec"] = 0.0

        started = time.perf_counter()
        representatives = self._sample_representatives(cluster_ids, step_id, features)
        timing["representative_sampling_time_sec"] = time.perf_counter() - started
        rep_dataset = Subset(self.dataset, representatives)

        now_train_save_dir = os.path.join(self.cache_dir, "train_representatives", str(step_id))
        now_eval_save_dir = os.path.join(self.cache_dir, "eval", str(step_id))
        rep_grads_path = os.path.join(now_train_save_dir, "all_projected_grads.pt")
        eval_grads_path = os.path.join(now_eval_save_dir, "all_projected_grads.pt")

        # Gate the collective gradient computation on a rank-0 decision.
        rep_grads_cached = os.path.exists(rep_grads_path) if self.accelerator.is_main_process else False
        rep_grads_cached = self._broadcast_bool(rep_grads_cached)
        if not rep_grads_cached:
            os.makedirs(now_train_save_dir, exist_ok=True)
            optimizer_state = kwargs.get("optimizer_state", None)
            started = time.perf_counter()
            self._collect_and_save_projected_gradients(
                model,
                now_train_save_dir,
                rep_dataset,
                self.gradient_type,
                optimizer_state,
            )
            self._merge_and_normalize_info(now_train_save_dir, len(rep_dataset))
            timing["representative_gradient_time_sec"] = time.perf_counter() - started
        else:
            timing["representative_gradient_time_sec"] = 0.0

        self.accelerator.wait_for_everyone()

        eval_grads_cached = os.path.exists(eval_grads_path) if self.accelerator.is_main_process else False
        eval_grads_cached = self._broadcast_bool(eval_grads_cached)
        if not eval_grads_cached:
            os.makedirs(now_eval_save_dir, exist_ok=True)
            started = time.perf_counter()
            self._collect_and_save_projected_gradients(model, now_eval_save_dir, self.eval_dataset, "sgd", None)
            self._merge_and_normalize_info(now_eval_save_dir, len(self.eval_dataset))
            timing["eval_gradient_time_sec"] = time.perf_counter() - started
        else:
            timing["eval_gradient_time_sec"] = 0.0

        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            started = time.perf_counter()
            rep_projected_grads = torch.load(rep_grads_path, map_location="cpu").float()
            eval_projected_grads = torch.load(eval_grads_path, map_location="cpu").float()

            num_clusters = int(cluster_ids.max().item()) + 1
            cluster_grads = torch.zeros(num_clusters, self.proj_dim, dtype=torch.float32)
            cluster_counts = torch.zeros(num_clusters, 1, dtype=torch.float32)
            rep_clusters = cluster_ids[torch.tensor(representatives, dtype=torch.long)]
            cluster_grads.index_add_(0, rep_clusters, rep_projected_grads)
            cluster_counts.index_add_(0, rep_clusters, torch.ones(len(representatives), 1))
            cluster_grads = cluster_grads / cluster_counts.clamp(min=1.0)
            cluster_grads = cluster_grads / cluster_grads.norm(dim=1, keepdim=True).clamp(min=1e-12)

            cluster_scores = (cluster_grads @ eval_projected_grads.T).mean(dim=1)
            sample_scores = cluster_scores[cluster_ids]
            topk = torch.topk(sample_scores, k=min(num_samples, len(sample_scores)), largest=True)
            selected_indices = topk.indices.tolist()
            selected_scores = topk.values.tolist()
            timing["scoring_time_sec"] = time.perf_counter() - started
            timing["total_select_time_sec"] = time.perf_counter() - select_started

            metric_payload = {
                "cluster_less_score": selected_scores,
                "num_clusters": int(num_clusters),
                "num_representatives": int(len(representatives)),
                "samples_per_cluster": int(self.samples_per_cluster),
                "clustering_method": self.clustering_method,
                "representative_strategy": self.representative_strategy,
                "embedding_layer": int(self.embedding_layer),
                "timing": timing,
            }
            save_selection(save_path, selected_indices, metric_payload, self.accelerator)
            logger.info(
                f"[ClusterLessSelector] Selected {len(selected_indices)} samples from "
                f"{num_clusters} clusters using {len(representatives)} representative gradients."
            )
        else:
            selected_indices = None

        selected = self._broadcast_long_tensor(
            torch.tensor(selected_indices, dtype=torch.long)
            if selected_indices is not None
            else None
        )
        return selected.tolist()
