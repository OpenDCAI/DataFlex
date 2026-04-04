import inspect
import warnings
from difflib import get_close_matches
from typing import Any, Dict, Optional, Type

class Registry:
    def __init__(self):
        self._store: Dict[str, Dict[str, Type]] = {}

    def register(self, kind: str, name: str):
        def deco(cls: Type):
            self._store.setdefault(kind, {})
            if name in self._store[kind]:
                raise ValueError(f"{kind}.{name} already registered")
            self._store[kind][name] = cls
            return cls
        return deco

    def get(self, kind: str, name: str) -> Type:
        return self._store[kind][name]

    def build(self, kind: str, name: str, *, runtime: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None):
        cls = self.get(kind, name)
        cfg = cfg or {}
        sig = inspect.signature(cls.__init__)
        accepted = {
            p.name
            for p in list(sig.parameters.values())[1:]
            if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        }

        self._validate_cfg(kind, name, cls, cfg, accepted)
        self._warn_unknown_runtime(kind, name, runtime, accepted)

        merged = {**cfg, **runtime}  # 运行期依赖优先
        filtered = {k: v for k, v in merged.items() if k in accepted}
        return cls(**filtered)

    def _validate_cfg(self, kind: str, name: str, cls: Type, cfg: Dict[str, Any], accepted: set[str]) -> None:
        if not cfg or getattr(cls, "skip_strict_validation", False):
            return

        unknown = sorted(set(cfg) - accepted)
        if not unknown:
            return

        supported = sorted(accepted)
        hints = []
        for key in unknown:
            matches = get_close_matches(key, supported, n=1)
            if matches:
                hints.append(f"{key!r} -> {matches[0]!r}")

        if not supported:
            msg = (
                f"{kind}.{name} has no configurable parameters, but got: "
                f"{', '.join(repr(k) for k in unknown)}."
            )
        else:
            msg = (
                f"Unknown config parameter(s) for {kind}.{name}: {', '.join(repr(k) for k in unknown)}. "
                f"Supported parameters: {', '.join(repr(k) for k in supported)}."
            )
        if hints:
            msg += f" Did you mean: {', '.join(hints)}?"

        raise ValueError(msg)

    def _warn_unknown_runtime(self, kind: str, name: str, runtime: Dict[str, Any], accepted: set[str]) -> None:
        if not runtime:
            return

        unknown = sorted(set(runtime) - accepted)
        if not unknown:
            return

        warnings.warn(
            (
                f"Ignoring unknown runtime parameter(s) for {kind}.{name}: "
                f"{', '.join(repr(k) for k in unknown)}."
            ),
            stacklevel=3,
        )

REGISTRY = Registry()
def register_selector(name: str): return REGISTRY.register("selector", name)
def register_mixer(name: str):    return REGISTRY.register("mixer", name)
def register_weighter(name: str):    return REGISTRY.register("weighter", name)
