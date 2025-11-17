import os
import numpy as np
from dataflex.core.registry import register_selector
from dataflex.utils.logging import logger
from .base_selector import Selector


@register_selector("near")  
class near_Selector(Selector):
    
    def __init__(
        self,
        dataset,
        accelerator,
        data_collator,
        cache_dir,
        indices_path: str = None):
        super().__init__(dataset, accelerator, data_collator, cache_dir)

        # 基本检查
        if not indices_path or not os.path.exists(indices_path):
            raise FileNotFoundError(f"top_indices 文件 {indices_path} 不存在！")
        self.top_indices = np.load(indices_path)
        logger.info(f"[near_Selector] Loaded top_indices from: {indices_path}")
        logger.info(f"[near_Selector] top_indices shape: {self.top_indices.shape}")
        if self.top_indices.ndim != 2:
            raise ValueError(
                f"top_indices 必须是二维矩阵，当前 ndim = {self.top_indices.ndim}"
            )
        if not np.issubdtype(self.top_indices.dtype, np.integer):
            raise TypeError(
                f"top_indices 必须是整数类型，当前 dtype = {self.top_indices.dtype}"
            )
        
        max_idx = int(self.top_indices.max())
        min_idx = int(self.top_indices.min())
        if min_idx < 0 or max_idx >= len(self.dataset):
            raise ValueError(
                f"top_indices 中存在越界索引：范围 [{min_idx}, {max_idx}]，"
                f"但数据集长度为 {len(self.dataset)}"
            )

        logger.info("[near_Selector] Initialization complete.")

    def select(self, model, step_id: int, num_samples: int, **kwargs):
        """
        按列优先顺序，从 top_indices 中选出 num_samples 个不重复的索引：
        - 第 0 列： top_indices[0,0], top_indices[1,0], ..., top_indices[M-1,0]
        - 第 1 列： top_indices[0,1], ...
        - ...
        - 遇到已经选过的 index 就跳过
        - 遍历完所有列仍不足 num_samples，则直接报错阻止运行
        """
        M, K = self.top_indices.shape
        logger.info(
            f"[near_Selector] Selecting {num_samples} samples "
            f"from matrix of shape (M={M}, K={K}) ..."
        )

        selected_ids = []
        selected_set = set()

        for k in range(K):
            for j in range(M):
                idx = int(self.top_indices[j, k])

                if idx in selected_set:
                    continue

                selected_set.add(idx)
                selected_ids.append(idx)

                if len(selected_ids) >= num_samples:
                    logger.info(
                        f"[near_Selector] Col-wise selection reached "
                        f"num_samples={num_samples}."
                    )
                    return selected_ids

        # 扫完所有列还不够,抛异常
        raise ValueError(
            f"{num_samples} 个不重复样本：最多只能得到 {len(selected_ids)} 个。"
        )
