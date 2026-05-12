import math
import os
import time
from typing import List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoModel

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


def _filter_model_inputs(batch):
    if not isinstance(batch, dict):
        return batch
    ignored_keys = {"labels", "label", "label_ids"}
    return {key: value for key, value in batch.items() if key not in ignored_keys}


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
        cluster_update_interval: int = 1,
        embedding_model_name_or_path: Optional[str] = None,
        embedding_model_cache_dir: Optional[str] = None,
        embedding_model_dtype: str = "auto",
        embedding_model_local_files_only: bool = False,
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
        self.cluster_update_interval = max(1, int(cluster_update_interval))
        self.embedding_model_name_or_path = embedding_model_name_or_path
        self.embedding_model_cache_dir = embedding_model_cache_dir
        self.embedding_model_dtype = embedding_model_dtype
        self.embedding_model_local_files_only = bool(embedding_model_local_files_only)

    def _resolve_cluster_step(self, step_id: int, **kwargs) -> int:
        """Return the cached clustering step to use for this dynamic update."""
        if self.embedding_model_name_or_path:
            return 0

        if self.cluster_update_interval <= 1:
            return int(step_id)

        current_update = int(kwargs.get("current_update_times", 1) or 1)
        update_step = int(kwargs.get("update_step", 0) or 0)
        if current_update <= 1 or update_step <= 0:
            return int(step_id)

        offset_updates = (current_update - 1) % self.cluster_update_interval
        return int(step_id) - offset_updates * update_step

    def _pool_last_hidden_state(self, hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attention_mask is None:
            return hidden.mean(dim=1)

        mask = attention_mask.to(hidden.device).unsqueeze(-1).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def _load_embedding_model(self):
        if not self.embedding_model_name_or_path:
            return None

        cache_dir = self.embedding_model_cache_dir or os.path.join(self.cache_dir, "hf_cache")
        dtype = None
        if self.embedding_model_dtype and self.embedding_model_dtype != "auto":
            dtype = getattr(torch, self.embedding_model_dtype, None)
            if dtype is None:
                raise ValueError(f"Unknown embedding_model_dtype: {self.embedding_model_dtype}")

        kwargs = {
            "trust_remote_code": True,
            "cache_dir": cache_dir,
            "local_files_only": self.embedding_model_local_files_only,
        }
        if dtype is not None:
            kwargs["torch_dtype"] = dtype

        if self.accelerator.is_main_process:
            logger.info(
                "[ClusterLessSelector] Loading static embedding model "
                f"from {self.embedding_model_name_or_path}; cache_dir={cache_dir}"
            )
        return AutoModel.from_pretrained(self.embedding_model_name_or_path, **kwargs)

    def _extract_features_with_embedding_model(self, step_id: int) -> torch.Tensor:
        feature_path = os.path.join(self.cache_dir, "cluster", str(step_id), "static_train_embeddings.pt")
        if os.path.exists(feature_path):
            if self.accelerator.is_main_process:
                logger.info(f"[ClusterLessSelector] Loading cached static train embeddings from {feature_path}")
            return torch.load(feature_path, map_location="cpu")

        os.makedirs(os.path.dirname(feature_path), exist_ok=True)
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

        embedding_model = self._load_embedding_model()
        embedding_model.to(self.device)
        embedding_model.eval()

        features = []
        indices_seen = []
        with torch.no_grad():
            for indices, batch in tqdm(
                dataloader,
                desc=f"[Process {self.accelerator.process_index}] Static cluster embeddings",
                disable=not self.accelerator.is_local_main_process,
                dynamic_ncols=True,
                position=self.accelerator.process_index,
            ):
                batch = _move_to_device(batch, self.device)
                outputs = embedding_model(**_filter_model_inputs(batch), return_dict=True)
                hidden = getattr(outputs, "last_hidden_state", None)
                if hidden is None and getattr(outputs, "hidden_states", None):
                    hidden = outputs.hidden_states[-1]
                if hidden is None:
                    raise RuntimeError("Embedding model output does not contain hidden states.")

                pooled = self._pool_last_hidden_state(hidden, batch.get("attention_mask"))
                features.append(pooled.detach().float().cpu())
                indices_seen.append(indices.cpu())

        del embedding_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        features = torch.cat(features, dim=0)
        indices_seen = torch.cat(indices_seen, dim=0)
        ordered = torch.empty_like(features)
        ordered[indices_seen] = features
        ordered = ordered / ordered.norm(dim=1, keepdim=True).clamp(min=1e-12)

        if self.accelerator.is_main_process:
            torch.save(ordered, feature_path)
            logger.info(f"[ClusterLessSelector] Saved static train embeddings to {feature_path}")
        self.accelerator.wait_for_everyone()
        return ordered

    def _extract_embedding_features(self, model, step_id: int) -> torch.Tensor:
        if self.embedding_model_name_or_path:
            return self._extract_features_with_embedding_model(step_id)

        feature_path = os.path.join(self.cache_dir, "cluster", str(step_id), "train_embeddings.pt")
        if os.path.exists(feature_path):
            if self.accelerator.is_main_process:
                logger.info(f"[ClusterLessSelector] Loading cached train embeddings from {feature_path}")
            return torch.load(feature_path, map_location="cpu")

        os.makedirs(os.path.dirname(feature_path), exist_ok=True)
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

        was_training = model.training
        model.eval()
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
                outputs = model(**batch, output_hidden_states=True, return_dict=True)
                hidden = outputs.hidden_states[-1]
                pooled = self._pool_last_hidden_state(hidden, batch.get("attention_mask"))
                features.append(pooled.detach().float().cpu())
                indices_seen.append(indices.cpu())

        if was_training:
            model.train()

        features = torch.cat(features, dim=0)
        indices_seen = torch.cat(indices_seen, dim=0)
        ordered = torch.empty_like(features)
        ordered[indices_seen] = features
        norms = ordered.norm(dim=1, keepdim=True).clamp(min=1e-12)
        ordered = ordered / norms

        if self.accelerator.is_main_process:
            torch.save(ordered, feature_path)
            logger.info(f"[ClusterLessSelector] Saved train embeddings to {feature_path}")
        self.accelerator.wait_for_everyone()
        return ordered

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
        center_norms = None
        if not spherical:
            center_norms = centers.pow(2).sum(dim=1).unsqueeze(0)
        for start in range(0, len(features), self.assignment_chunk_size):
            chunk = features[start:start + self.assignment_chunk_size]
            if spherical:
                similarities = chunk @ centers.T
                assignments.append(similarities.argmax(dim=1))
            else:
                scores = 2.0 * (chunk @ centers.T) - center_norms
                assignments.append(scores.argmax(dim=1))
        return torch.cat(assignments, dim=0)

    def _run_lloyd_kmeans(self, features: torch.Tensor, step_id: int, spherical: bool) -> torch.Tensor:
        n_samples = len(features)
        k = self._resolve_num_clusters(n_samples)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + int(step_id))
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

    def _run_farthest_first(self, features: torch.Tensor, step_id: int) -> torch.Tensor:
        n_samples = len(features)
        k = self._resolve_num_clusters(n_samples)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + int(step_id))

        first = int(torch.randint(n_samples, (1,), generator=generator).item())
        center_indices = [first]
        min_distances = torch.cdist(features, features[first:first + 1]).squeeze(1)
        for _ in range(1, k):
            next_index = int(min_distances.argmax().item())
            center_indices.append(next_index)
            distances = torch.cdist(features, features[next_index:next_index + 1]).squeeze(1)
            min_distances = torch.minimum(min_distances, distances)

        centers = features[torch.tensor(center_indices, dtype=torch.long)]
        return self._assign_to_centers(features, centers).to(torch.long)

    def _run_random_projection_lsh(self, features: torch.Tensor, step_id: int) -> torch.Tensor:
        n_samples = len(features)
        k = self._resolve_num_clusters(n_samples)
        bits = self.lsh_num_bits or max(1, int(math.ceil(math.log2(k))))
        bits = min(bits, 30)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + int(step_id) + 31)
        planes = torch.randn(features.shape[1], bits, generator=generator, dtype=features.dtype)
        signatures = (features @ planes >= 0).to(torch.long)
        weights = (2 ** torch.arange(bits, dtype=torch.long)).unsqueeze(0)
        buckets = (signatures * weights).sum(dim=1)

        unique_buckets, inverse = torch.unique(buckets, sorted=True, return_inverse=True)
        if len(unique_buckets) <= k:
            return inverse.to(torch.long)
        return (inverse % k).to(torch.long)

    def _run_random_partition(self, features: torch.Tensor, step_id: int) -> torch.Tensor:
        n_samples = len(features)
        k = self._resolve_num_clusters(n_samples)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + int(step_id) + 47)
        order = torch.randperm(n_samples, generator=generator)
        assignments = torch.empty(n_samples, dtype=torch.long)
        assignments[order] = torch.arange(n_samples, dtype=torch.long) % k
        return assignments

    def _run_clustering(self, features: torch.Tensor, step_id: int) -> torch.Tensor:
        method = self.clustering_method.lower()
        if method in {"kmeans", "euclidean_kmeans"}:
            return self._run_lloyd_kmeans(features, step_id, spherical=False)
        if method in {"spherical_kmeans", "cosine_kmeans"}:
            return self._run_lloyd_kmeans(features, step_id, spherical=True)
        if method in {"farthest_first", "kcenter", "k_center"}:
            return self._run_farthest_first(features, step_id)
        if method in {"random_projection_lsh", "lsh", "rp_lsh"}:
            return self._run_random_projection_lsh(features, step_id)
        if method in {"random_partition", "random"}:
            return self._run_random_partition(features, step_id)
        raise ValueError(f"Unknown clustering_method: {self.clustering_method}")

    def _get_or_create_clusters(self, model, step_id: int):
        cluster_dir = os.path.join(self.cache_dir, "cluster", str(step_id))
        cluster_path = os.path.join(cluster_dir, "train_cluster_ids.pt")
        timing = {"embedding_time_sec": 0.0, "clustering_time_sec": 0.0, "cluster_cache_hit": bool(os.path.exists(cluster_path))}

        clusters_cached = os.path.exists(cluster_path)
        if dist.is_available() and dist.is_initialized():
            obj = [clusters_cached]
            dist.broadcast_object_list(obj, src=0)
            clusters_cached = obj[0]

        if not clusters_cached:
            started = time.perf_counter()
            features = self._extract_embedding_features(model, step_id)
            timing["embedding_time_sec"] = time.perf_counter() - started
        else:
            features = None

        if self.accelerator.is_main_process and clusters_cached:
            cluster_ids = torch.load(cluster_path, map_location="cpu")
            logger.info(f"[ClusterLessSelector] Loading cached clusters from {cluster_path}")
        elif self.accelerator.is_main_process:
            started = time.perf_counter()
            cluster_ids = self._run_clustering(features, step_id)
            timing["clustering_time_sec"] = time.perf_counter() - started
            os.makedirs(cluster_dir, exist_ok=True)
            torch.save(cluster_ids, cluster_path)
            logger.info(
                f"[ClusterLessSelector] Built {int(cluster_ids.max().item()) + 1} clusters "
                f"for {len(cluster_ids)} train samples with method={self.clustering_method}."
            )
        else:
            cluster_ids = None

        obj = [cluster_ids.tolist() if cluster_ids is not None else None]
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(obj, src=0)
        cluster_ids = torch.tensor(obj[0], dtype=torch.long)
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
        distances = (member_features - centroid).pow(2).sum(dim=1)

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
        if os.path.exists(save_path):
            if self.accelerator.is_main_process:
                cached_indices, _ = load_cached_selection(save_path)
            else:
                cached_indices = None
            obj = [cached_indices]
            if dist.is_available() and dist.is_initialized():
                dist.broadcast_object_list(obj, src=0)
            return obj[0] or []

        select_started = time.perf_counter()
        cluster_step_id = self._resolve_cluster_step(step_id, **kwargs)
        cluster_ids, timing = self._get_or_create_clusters(model, cluster_step_id)
        timing["cluster_step_id"] = int(cluster_step_id)
        timing["cluster_update_interval"] = int(self.cluster_update_interval)
        features = None
        if self.representative_strategy.lower() not in {"random", "rand"}:
            started = time.perf_counter()
            features = self._extract_embedding_features(model, cluster_step_id)
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

        if not os.path.exists(rep_grads_path):
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

        if not os.path.exists(eval_grads_path):
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
            if rep_projected_grads.shape[0] != len(representatives):
                aligned_count = min(rep_projected_grads.shape[0], len(representatives))
                logger.warning(
                    "[ClusterLessSelector] Representative gradient count mismatch: "
                    f"got {rep_projected_grads.shape[0]} gradients for {len(representatives)} "
                    f"representatives. Aligning to first {aligned_count} rows."
                )
                rep_projected_grads = rep_projected_grads[:aligned_count]
                representatives = representatives[:aligned_count]
            rep_clusters = cluster_ids[torch.tensor(representatives, dtype=torch.long)]
            cluster_grads.index_add_(0, rep_clusters, rep_projected_grads)
            cluster_counts.index_add_(0, rep_clusters, torch.ones(len(representatives), 1))
            cluster_grads = cluster_grads / cluster_counts.clamp(min=1.0)
            cluster_grads = cluster_grads / cluster_grads.norm(dim=1, keepdim=True).clamp(min=1e-12)

            cluster_scores = (cluster_grads @ eval_projected_grads.T).mean(dim=1)
            sample_scores = cluster_scores[cluster_ids]
            topk = torch.topk(sample_scores, k=min(num_samples, len(sample_scores)), largest=True)
            selected_indices = topk.indices.tolist()
            timing["scoring_time_sec"] = time.perf_counter() - started
            timing["total_select_time_sec"] = time.perf_counter() - select_started

            metric_payload = {
                "cluster_less_score": [float(sample_scores[i].item()) for i in selected_indices],
                "num_clusters": int(num_clusters),
                "num_representatives": int(len(representatives)),
                "samples_per_cluster": int(self.samples_per_cluster),
                "clustering_method": self.clustering_method,
                "representative_strategy": self.representative_strategy,
                "cluster_update_interval": int(self.cluster_update_interval),
                "embedding_model_name_or_path": self.embedding_model_name_or_path,
                "cluster_step_id": int(cluster_step_id),
                "timing": timing,
            }
            save_selection(save_path, selected_indices, metric_payload, self.accelerator)
            logger.info(
                f"[ClusterLessSelector] Selected {len(selected_indices)} samples from "
                f"{num_clusters} clusters using {len(representatives)} representative gradients."
            )
        else:
            selected_indices = None

        obj = [selected_indices]
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(obj, src=0)
        return obj[0] or []
