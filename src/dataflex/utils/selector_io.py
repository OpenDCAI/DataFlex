import json
import os
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple
from dataflex.utils.logging import logger

def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
        
def load_cached_selection(
    save_path: str
) -> Tuple[Optional[List[int]], Optional[Dict[str, List]]]:
    indices = None
    metric = None
    with open(save_path, "r") as f:
        payload = json.load(f)
    indices = payload.get("indices", [])
    metric = payload.get("metric", {})

    logger.info(f"[Dataflex] Loaded cached selection from {save_path}: {indices is not None}.")
    return indices, metric

def save_selection(
    save_path: str,
    indices: List[int],
    metric: Dict[str, List],
    accelerator,
) -> None:
    """
    以统一格式保存，并仅由主进程落盘。
    存储为标准的 JSON 格式。
    """
    if accelerator.is_main_process:
        _ensure_parent_dir(save_path)
        payload = {
            "indices": list(map(int, indices)),
            "metric": metric,
        }
        with open(save_path, "w") as f:
            json.dump(payload, f, indent=4)  # 保存为漂亮的JSON格式
        logger.info(f"[Dataflex] Saved selection to {save_path}.")
