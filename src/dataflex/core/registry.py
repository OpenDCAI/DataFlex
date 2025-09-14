import inspect
from typing import Dict, Type, Any, Optional

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
        merged = {**cfg, **runtime}                     # 运行期依赖优先
        sig = inspect.signature(cls.__init__)
        accepted = {p.name for p in list(sig.parameters.values())[1:]}  # 跳过 self
        filtered = {k: v for k, v in merged.items() if k in accepted}   # 只喂需要的
        return cls(**filtered)

REGISTRY = Registry()
def register_selector(name: str): return REGISTRY.register("selector", name)
def register_mixer(name: str):    return REGISTRY.register("mixer", name)
def register_weighter(name: str):    return REGISTRY.register("weighter", name)
