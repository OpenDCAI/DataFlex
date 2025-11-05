import os
import numpy as np
from dataflex.core.registry import register_selector
from dataflex.utils.logging import logger
from .base_selector import Selector

@register_selector("tsds")
class tsds_Selector(Selector):
    ###一个基于外部 TSDS 概率文件（.npy）进行采样的数据选择器。
    def __init__(
        self,
        dataset,
        accelerator,
        data_collator,
        cache_dir,
        probs_path: str = None,
    ): 
        super().__init__(dataset, accelerator, data_collator, cache_dir)
        self.probs_path = probs_path

        if not probs_path or not os.path.exists(probs_path):
            raise FileNotFoundError(f"TSDS概率文件 {probs_path} 不存在！")
        self.probs = np.load(probs_path)
        logger.info(f"[tsds_selector] Loaded probabilities from: {probs_path}")
        logger.info(f"[tsds_selector] Probs shape: {self.probs.shape}")

        if len(self.probs) != len(self.dataset):
            raise ValueError(
                f"概率长度 {len(self.probs)} 与数据集长度 {len(self.dataset)} 不一致！"
            )
        logger.info("[tsds_Selector] Initialization complete.")

    def select(self, model, step_id: int, num_samples: int, **kwargs):
        
        logger.info(
            f"[tsds_selector] Sampling {num_samples} examples using precomputed probs..."
        )
        selected_ids = np.random.choice(
            len(self.probs),
            size=num_samples,
            replace=False,
            p=self.probs,
        )
        logger.info(f"[tsds_selector] Sample complete.")
        return selected_ids.tolist()
