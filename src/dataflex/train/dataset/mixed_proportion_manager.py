# 文件：llamafactory/data/mixture_dataset_runtime.py
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset as HFDataset

class _MixedSnapshot(TorchDataset):
    """一次 rebuild 结果的只读快照，可直接喂给 DataLoader。"""
    def __init__(self, names: List[str], sources: Dict[str, HFDataset], index_table: List[Tuple[int, int]]):
        self.names = names
        self.sources = sources
        self.index_table = index_table
    def __len__(self): return len(self.index_table)
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        si, row = self.index_table[idx]
        name = self.names[si]
        item = self.sources[name][row]
        # Add domain_id to track which domain this sample belongs to
        # This is needed by DoReMi mixer to compute per-domain excess losses
        if isinstance(item, dict):
            item['domain_id'] = si
        return item

class MixedProportionManager:
    """
    混合管理器：
      - 传入：各来源的『已预处理』HF Dataset（字段已与 collator/模板对齐）
      - set_proportions([...])：更新比例
      - rebuild(num_samples=None, seed=None) -> Dataset：返回新的快照 Dataset
    """
    def __init__(
        self,
        per_source: Dict[str, HFDataset],
        sample_rule: str = "mixture",
        proportions: Optional[List[float]] = None,
        seed: int = 42,
        slice_list: Optional[List[str]] = None,
        logger=None,
    ):
        assert len(per_source) > 0
        self.logger = logger
        all_names = list(per_source.keys())
        if slice_list:
            names = [n for n in all_names if n in set(slice_list)]
        else:
            names = all_names
        self.names = names
        self.k = len(names)
        self.sources = {k: per_source[k] for k in names}
        self.sample_rule = sample_rule
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        self.sizes = {k: len(v) for k, v in self.sources.items()}
        
        # Store initial proportions for mixers to access
        self.initial_proportions = proportions.copy() if proportions is not None else None
        
        self.set_proportions(proportions)

    def set_proportions(self, proportions: Optional[List[float]]):
        k = self.k
        if self.sample_rule == "mixture":
            # mixture: 用用户指定比例，否则均匀
            if proportions is None:
                probs = np.repeat(1.0 / k, k)
            else:
                assert len(proportions) == k, f"proportions 长度 {len(proportions)} != 源数 {k}"
                probs = np.array(proportions, dtype=float)
                probs = probs / probs.sum()

        elif self.sample_rule == "stratified":
            # stratified: 按源数据集大小比例分层
            sizes = np.array([self.sizes[n] for n in self.names], dtype=float)
            probs = sizes / sizes.sum()

        elif self.sample_rule == "uniform":
            # uniform: 强制均匀分布
            probs = np.repeat(1.0 / k, k)

        else:
            raise ValueError(f"Unknown sample_rule={self.sample_rule}")

        self.probs = probs

    def _current_probs(self) -> np.ndarray:
        """
        根据当前 sample_rule 计算“应当使用”的比例，不依赖 self.probs，
        确保在 stratified/uniform 下不会受到旧状态影响。
        仅用作信息打印
        """
        k = self.k
        if self.sample_rule == "mixture":
            # mixture 下仍然使用 set_proportions 设定的 self.probs
            return np.array(self.probs, dtype=float)

        elif self.sample_rule == "stratified":
            sizes = np.array([self.sizes[n] for n in self.names], dtype=float)
            return sizes / sizes.sum()

        elif self.sample_rule == "uniform":
            return np.repeat(1.0 / k, k)

        else:
            raise ValueError(f"Unknown sample_rule={self.sample_rule}")



    def rebuild(self, num_samples: Optional[int] = None, seed: Optional[int] = None) -> TorchDataset:
        if seed is not None:
            self._seed = int(seed)
            self.rng = np.random.default_rng(self._seed)

        sizes = np.array([self.sizes[n] for n in self.names], dtype=int)

        # Safety check: validate self.probs before using
        probs = self.probs
        if (not np.all(np.isfinite(probs))) or probs.sum() <= 0:
            if self.logger:
                self.logger.warning("[MixedProportionManager] probs invalid, reset to uniform")
            probs = np.ones_like(probs, dtype=float) / len(probs)
            self.probs = probs

        # 按比例计算每个数据源的数量，使用更稳定的计算方式避免溢出
        # 先计算每个比例对应的样本数，避免大数相乘
        n_per = np.zeros(len(probs), dtype=np.int64)  # 使用 int64 避免溢出
        remaining = num_samples

        for i in range(len(probs) - 1):
            # 计算当前域的样本数，取整
            n_i = int(np.floor(num_samples * probs[i]))
            # 确保不超过剩余样本数
            n_i = min(n_i, remaining)
            n_per[i] = n_i
            remaining -= n_i

        # 最后一个域得到剩余的所有样本
        n_per[-1] = remaining

        # Additional safety check: ensure no negative values
        if (n_per < 0).any() or not np.all(np.isfinite(n_per)):
            if self.logger:
                self.logger.warning("[MixedProportionManager] n_per invalid, reset to uniform allocation")
            n_per = np.full_like(n_per, num_samples // len(n_per), dtype=int)
            n_per[-1] += num_samples - n_per.sum()

        # 确保没有负数（理论上不应该发生，但作为安全检查）
        n_per = np.maximum(n_per, 0)

        index_table: list[tuple[int, int]] = []

        for si, (name, take) in enumerate(zip(self.names, n_per)):
            cap = self.sizes[name]
            # 无论 take 是否超过 cap，都直接从 0..cap-1 中抽样
            # replace=True 可以保证数量足够
            rows = self.rng.choice(cap, size=take, replace=True).tolist()
            index_table.extend((si, r) for r in rows)

        # 打乱最终索引表
        perm = self.rng.permutation(len(index_table))
        index_table = [index_table[i] for i in perm]

        if self.logger:
            plan = list(zip(self.names, self.probs.tolist(), n_per.tolist(), sizes.tolist()))
            assert len(index_table) == num_samples
            self.logger.info(f"[Mixture] plan (name, prob, take, cap): {plan}; total={len(index_table)}")

        return _MixedSnapshot(self.names, self.sources, index_table)
