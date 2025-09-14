import yaml
from typing import Dict, Any, Optional

def load_component(cfg_file: str, name: str, runtime_vars: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    with open(cfg_file, "r", encoding="utf-8") as f:
        root = yaml.safe_load(f) or {}
    bucket = (root.get("selectors") or {})
    if name not in bucket:
        available = ", ".join(sorted(bucket.keys()))
        raise ValueError(f"selector '{name}' not found. Available: {available}")
    params = dict(bucket[name].get("params") or {})

    # 简单占位替换（如 ${output_dir}）
    if runtime_vars:
        def subst(v):
            if isinstance(v, str):
                for k, val in runtime_vars.items(): v = v.replace(k, val)
                return v
            if isinstance(v, dict):  return {kk: subst(vv) for kk, vv in v.items()}
            if isinstance(v, list):  return [subst(x) for x in v]
            return v
        params = subst(params)

    return params
