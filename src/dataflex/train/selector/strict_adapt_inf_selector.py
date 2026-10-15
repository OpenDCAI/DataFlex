"""Adapt selection with a run-wide budget of distinct training records.

The warmup and each interval are disjoint and are consumed in full. The only
allowed repeats are explicitly counted distributed padding in the last batch.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from dataflex.core.registry import register_selector
from dataflex.utils.multimodal_selection import fingerprint, write_json

from .adapt_inf_selector import AdaptInfSelector


class CachedRademacher:
    """Seeded dense JL projection, signs ±1/sqrt(d), kept on the GPU.

    This matches the Rademacher projection family used by the authors' TRAK
    projector. It does not claim identical matrices across different RNGs.
    """

    def __init__(self, gradients, dimension, seed, device):
        self.size = sum(g.numel() for g in gradients)
        self.dimension = dimension
        self.matrix = torch.empty((self.size, dimension), device=device, dtype=torch.bfloat16)
        generator = torch.Generator(device=device).manual_seed(seed)
        for start in range(0, self.size, 1024):
            block = torch.randint(
                0, 2, (min(1024, self.size - start), dimension), generator=generator, device=device, dtype=torch.int8
            )
            self.matrix[start : start + len(block)].copy_(block)
        self.matrix.mul_(2).sub_(1).mul_(dimension**-0.5)

    def __call__(self, gradients):
        vector = torch.cat([g.detach().reshape(-1) for g in gradients]).to(self.matrix.dtype)
        assert vector.numel() == self.size
        return (vector[None] @ self.matrix)[0].float().cpu().numpy()


def remaining_record_budget(total, previous, requested):
    flat = [int(i) for group in previous for i in group]
    if len(flat) != len(set(flat)) or len(flat) > total:
        raise ValueError("Repeated records or exceeded run-wide budget")
    return min(requested, total - len(flat)), set(flat)


@register_selector("adapt_inf_strict")
class StrictAdaptInfSelector(AdaptInfSelector):
    def __init__(
        self,
        dataset,
        accelerator,
        data_collator,
        cache_dir,
        unique_budget,
        cluster_grid=(5, 10, 15, 20, 25, 30, 35, 40, 45, 50),
        num_clusters=10,
        proj_dim=8192,
        projection="rademacher",
        parameter_pattern=r"\.layers\.16\..*lora_",
        seed=42,
        histogram_bins=50,
        trim_fraction=0.05,
        replacement=False,
        dataset_revision=None,
        enable_permanent_pruning=False,
        projection_backend="cuda",
    ):
        super().__init__(
            dataset,
            accelerator,
            data_collator,
            cache_dir,
            num_clusters,
            proj_dim,
            projection,
            parameter_pattern,
            seed,
            histogram_bins,
            trim_fraction,
            replacement,
            dataset_revision,
            enable_permanent_pruning,
            projection_backend,
            selection_budget="exact",
        )
        self.unique_budget = int(unique_budget)
        self.cluster_grid = list(cluster_grid)
        self.ledger = Path(self.cache_dir) / "record_budget"
        self.ledger.mkdir(parents=True, exist_ok=True)
        if not 0 < self.unique_budget <= len(self.dataset):
            raise ValueError("Invalid distinct-record budget")
        self._strict_signature = fingerprint(
            {
                "parent": super().checkpoint_signature(),
                "unique_budget": self.unique_budget,
                "cluster_grid": self.cluster_grid,
            }
        )

    def checkpoint_signature(self):
        return fingerprint({"algorithm": self._strict_signature, "candidate_pool": self.get_candidate_pool()})

    def _write_stage(self, path, state):
        """Propagate rank-zero I/O failures instead of stranding other ranks."""
        error = None
        if self.accelerator.is_main_process:
            try:
                write_json(path, state)
            except Exception as exc:
                error = f"record ledger write failed: {type(exc).__name__}: {exc}"
        if dist.is_initialized():
            payload = [error]
            dist.broadcast_object_list(payload, src=0)
            error = payload[0]
        if error:
            raise RuntimeError(error)

    def _stages(self, before=None):
        stages = []
        for path in sorted(self.ledger.glob("step_*.json")):
            state = json.loads(path.read_text())
            if state["signature"] != self._strict_signature:
                raise ValueError("Record ledger belongs to another run")
            if before is None or state["step"] < before:
                stages.append(state)
        return stages

    def export_checkpoint_state(self, *, global_step, opened_step, active_indices):
        stages = [
            {k: s[k] for k in ("step", "kind", "signature", "indices")} for s in self._stages(before=opened_step + 1)
        ]
        payload = {"version": 1, "signature": self._strict_signature, "stages": stages}
        payload["digest"] = fingerprint(payload)
        self._validate_checkpoint_state(payload, global_step, opened_step, active_indices)
        return payload

    def _validate_checkpoint_state(self, state, global_step, opened_step, active_indices):
        if not isinstance(state, dict) or state.get("version") != 1:
            raise ValueError("Strict record-budget state missing; cannot resume without complete history")
        if state.get("signature") != self._strict_signature:
            raise ValueError("Record-budget checkpoint configuration differs")
        payload = {k: v for k, v in state.items() if k != "digest"}
        if state.get("digest") != fingerprint(payload):
            raise ValueError("Record-budget checkpoint digest mismatch")
        stages = state.get("stages")
        if not isinstance(stages, list) or not stages or stages[0].get("step") != 0:
            raise ValueError("Record-budget checkpoint is missing its initial stage")
        steps, used = [], set()
        pool = set(range(len(self.dataset)))
        for stage in stages:
            step, ids = stage.get("step"), stage.get("indices")
            if (
                type(step) is not int
                or step < 0
                or step > opened_step
                or stage.get("signature") != self._strict_signature
                or not isinstance(ids, list)
                or any(type(i) is not int or i not in pool for i in ids)
                or len(ids) != len(set(ids))
                or used.intersection(ids)
            ):
                raise ValueError("Invalid or overlapping record-budget checkpoint stages")
            steps.append(step)
            used.update(ids)
        if (
            steps != sorted(set(steps))
            or steps[-1] != opened_step
            or opened_step > global_step
            or stages[-1]["indices"] != active_indices
            or len(used) > self.unique_budget
        ):
            raise ValueError("Record-budget checkpoint does not match the active stage")
        return stages

    def restore_checkpoint_state(self, state, *, global_step, opened_step, active_indices):
        error, stages = None, []
        # Earlier/later ledger entries from a different continuation must never
        # silently change the restored checkpoint's past. Keep future entries:
        # select() ignores them until that boundary is recomputed/overwritten.
        try:
            stages = self._validate_checkpoint_state(state, global_step, opened_step, active_indices)
            expected = {s["step"]: s for s in stages}
            for old in self._stages(before=opened_step + 1):
                new = expected.get(old["step"])
                if new is None or old["indices"] != new["indices"]:
                    raise ValueError("Existing record ledger conflicts with checkpoint history")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = [error]
        if dist.is_initialized():
            errors = [None] * dist.get_world_size()
            dist.all_gather_object(errors, error)
        if any(errors):
            raise ValueError("; ".join(e for e in errors if e))
        for stage in stages:
            self._write_stage(self.ledger / f"step_{stage['step']:08d}.json", stage)

    def warmup(self, num_samples, replacement=False):
        if num_samples > self.unique_budget:
            raise ValueError("Warmup exceeds unique-record budget")
        pool = self.candidate_positions()
        if num_samples > len(pool):
            raise ValueError("Warmup exceeds the upstream candidate pool")
        generator = torch.Generator().manual_seed(self.seed)
        ids = [pool[i] for i in torch.randperm(len(pool), generator=generator)[:num_samples].tolist()]
        path = self.ledger / "step_00000000.json"
        state = {"step": 0, "indices": ids, "kind": "warmup", "signature": self._strict_signature}
        if path.exists() and json.loads(path.read_text())["indices"] != ids:
            raise ValueError("Warmup ledger does not match run")
        self._write_stage(path, state)
        return ids

    def select(self, model, step_id, num_samples, **kwargs):
        previous = [s["indices"] for s in self._stages(before=step_id)]
        count, used = remaining_record_budget(self.unique_budget, previous, num_samples)
        if count <= 0:
            raise ValueError("No unused training budget remains")
        external_pool = self.get_candidate_pool()
        available = [i for i in self.candidate_positions() if i not in used]
        if count > len(available):
            raise ValueError("Requested budget exceeds unused upstream candidates")
        self.set_candidate_pool(available)
        try:
            ids = super().select(model, step_id, count, **kwargs)
        finally:
            # Dense projection is recreated deterministically next round. Do not
            # keep 32 GiB allocated during ordinary DDP training.
            self._gradient_projector = None
            torch.cuda.empty_cache()
            self.set_candidate_pool(external_pool)
        assert len(ids) == count and len(set(ids)) == count and not used.intersection(ids)
        self._write_stage(
            self.ledger / f"step_{step_id:08d}.json",
            {
                "step": step_id,
                "kind": "adapt_selection",
                "signature": self._strict_signature,
                "indices": ids,
                "distinct_records_total": len(used) + count,
                "metadata": self.last_metadata,
            },
        )
        return ids

    def cluster_paper_features(self, features, directory):
        """Paper grid 5..50; a deterministic WSS elbow chooses the cluster count.

        The paper specifies WSS, but no tie/elbow rule. The normalized distance
        from the first-to-last chord is recorded as our explicit resolution.
        """
        from sklearn.cluster import MiniBatchKMeans

        candidates = []
        for k in self.cluster_grid:
            if k > len(features):
                continue
            estimator = MiniBatchKMeans(n_clusters=k, batch_size=10240, random_state=self.seed, n_init=3)
            estimator.fit(features)
            labels = estimator.labels_
            wss = 0.0
            for start in range(0, len(features), 4096):
                block = features[start : start + 4096]
                delta = block - estimator.cluster_centers_[labels[start : start + len(block)]]
                wss += float(np.einsum("ij,ij->", delta, delta, dtype=np.float64))
            np.save(directory / f"cluster_grid_k{k}.npy", labels)
            candidates.append({"k": k, "wss": wss})
            write_json(directory / "cluster_grid_progress.json", candidates)
        if not candidates:
            raise ValueError("Cluster grid exceeds pool size")
        if len(candidates) == 1:
            index = 0
        else:
            x = np.asarray([r["k"] for r in candidates], dtype=float)
            y = np.asarray([r["wss"] for r in candidates], dtype=float)
            xn = (x - x[0]) / (x[-1] - x[0])
            yn = (y - y[-1]) / max(y[0] - y[-1], 1e-12)
            distances = (1 - xn) - yn
            index = int(np.argmax(distances))
        chosen = candidates[index]["k"]
        write_json(
            directory / "cluster_grid_selection.json",
            {
                "grid": candidates,
                "selected_k": chosen,
                "rule": "normalized WSS chord elbow; first maximum",
                "seed": self.seed,
                "projection_dimension": self.proj_dim,
                "gradient_pattern": self.parameter_pattern,
            },
        )
        return np.load(directory / f"cluster_grid_k{chosen}.npy")
