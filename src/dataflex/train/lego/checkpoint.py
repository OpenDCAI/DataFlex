"""Explicit, tensor/primitive-only checkpoints for supported LEGO components."""

import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from dataflex.utils.multimodal_selection import fingerprint

from .plan import DataPlan


def dataset_identity(dataset):
    """Identify processed rows and referenced media snapshots, not just length.

    External files are identified by path/size/mtime, not re-read in full. Their
    snapshot must stay immutable for exact resume (as for ordinary HF training).
    """
    if dataset is None:
        return None
    if isinstance(dataset, dict):
        return {key: dataset_identity(value) for key, value in dataset.items()}
    from dataflex.train.data.loader import DomainLabeledDataset

    if isinstance(dataset, DomainLabeledDataset):
        return fingerprint({"data": dataset_identity(dataset.dataset), "domains": dataset.domain_ids.tolist()})
    revision = getattr(dataset, "_fingerprint", None)
    media = hashlib.sha256()
    seen = set()

    def record(value):
        if isinstance(value, (list, tuple)):
            for item in value:
                record(item)
        elif isinstance(value, dict):
            record(value.get("path"))
        elif isinstance(value, str) and value not in seen:
            seen.add(value)
            path = Path(value)
            stat = path.stat() if path.is_file() else None
            media.update(
                fingerprint([value, stat.st_size if stat else None, stat.st_mtime_ns if stat else None]).encode()
            )

    if revision is not None:
        for field in ("images", "videos", "audios"):
            if field in (getattr(dataset, "column_names", None) or []):
                for value in dataset[field]:
                    record(value)
    else:
        rows = hashlib.sha256()
        for i in range(len(dataset)):
            row = dataset[i]
            rows.update(fingerprint(row).encode())
            for field in ("images", "videos", "audios"):
                record(row.get(field))
        revision = rows.hexdigest()
    return fingerprint({"revision": revision, "length": len(dataset), "media": media.hexdigest()})


def portable(value):
    """Avoid pickling modules, callbacks, or device-dependent objects."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable(item) for item in value]
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return value
    raise TypeError(f"Unsupported LEGO checkpoint value: {type(value).__name__}")


def unsupported_stages(pipeline):
    from dataflex.train.reorder.dynamic_reorder import DynamicReorder
    from dataflex.train.reorder.static_reorder import StaticReorder

    supported = {
        "selector": {"loss", "delta_loss", "adapt_inf", "random"},
        "mixer": {"static", "random", "odm"},
        "weighter": {"loss", "multimodal_loss", "multimodal_deltaloss", "multimodal_adapt"},
        "reorder": {"static", "dynamic"},
    }
    return [
        f"{stage.family}:{stage.name}"
        for stage in (*pipeline.index_stages, *pipeline.loss_stages)
        if stage.component is not None
        and not (stage.family == "reorder" and type(stage.component) in (StaticReorder, DynamicReorder))
        and stage.name not in supported.get(stage.family, set())
    ]


def pipeline_state(pipeline):
    stages = []
    for stage in (*pipeline.index_stages, *pipeline.loss_stages):
        component = stage.component
        export = getattr(component, "state_dict", None)
        stages.append(
            {
                "identity": [stage.family, stage.name],
                "last_run_step": stage._last_run_step,
                "cached": stage._cached,
                "last_probs": getattr(stage, "_last_probs", None),
                "last_batch_update": getattr(stage, "_last_batch_update", None),
                "candidate_pool": getattr(component, "_candidate_pool", None),
                "component": export() if callable(export) else None,
            }
        )
    plan = pipeline._plan
    board = pipeline.scoreboard
    return portable(
        {
            "version": 2,
            "unsupported": unsupported_stages(pipeline),
            "stages": stages,
            "stage_out": pipeline._stage_out,
            "plan": None if plan is None else {"indices": plan.indices, "weights": plan.weights, "meta": plan.meta},
            "scoreboard": None if board is None else board.state_dict(),
        }
    )


def restore_pipeline(pipeline, state, dataset_size):
    if state.get("version") != 2:
        raise ValueError("Unsupported LEGO checkpoint schema; checkpoints from before shared scoring need a new run")
    unsupported = unsupported_stages(pipeline)
    if unsupported:
        raise ValueError(f"LEGO resume is not implemented for {unsupported}")
    if state.get("unsupported"):
        raise ValueError(
            f"LEGO checkpoint lacks resumable state for {state['unsupported']}; a new checkpoint is required"
        )
    stages = [*pipeline.index_stages, *pipeline.loss_stages]
    if len(state["stages"]) != len(stages) or len(state["stage_out"]) != len(pipeline.index_stages):
        raise ValueError("LEGO checkpoint stage count differs")
    for stage, saved in zip(stages, state["stages"]):
        if stage.family == "mixer" and stage.name == "odm" and saved["component"] is None:
            raise ValueError("LEGO checkpoint is missing ODM state; a new checkpoint is required")

    def indices(value):
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.int64)
        if arr.ndim != 1 or (arr < 0).any() or (arr >= dataset_size).any():
            raise ValueError("LEGO checkpoint has invalid dataset indices")
        return arr.copy()

    for stage, saved in zip(stages, state["stages"]):
        if saved["identity"] != [stage.family, stage.name]:
            raise ValueError("LEGO checkpoint stage identity differs")
        stage._last_run_step, stage._cached = saved["last_run_step"], saved["cached"]
        if hasattr(stage, "_last_probs"):
            stage._last_probs = None if saved["last_probs"] is None else np.asarray(saved["last_probs"]).copy()
            stage._last_batch_update = saved["last_batch_update"]
        if saved["component"] is not None:
            stage.component.load_state_dict(saved["component"])
        if hasattr(stage.component, "_candidate_pool"):
            pool = indices(saved["candidate_pool"])
            stage.component._candidate_pool = None if pool is None else pool.tolist()
    pipeline._stage_out = [indices(value) for value in state["stage_out"]]
    plan = state["plan"]
    pipeline._plan = (
        None
        if plan is None
        else DataPlan(
            indices(plan["indices"]),
            None if plan["weights"] is None else np.asarray(plan["weights"]).copy(),
            dict(plan["meta"]),
        )
    )
    saved = state["scoreboard"]
    if saved is not None and pipeline.scoreboard is not None:
        pipeline.scoreboard.load_state_dict(saved)


def save_atomic(path, state):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(portable(state), temporary)
    os.replace(temporary, path)
