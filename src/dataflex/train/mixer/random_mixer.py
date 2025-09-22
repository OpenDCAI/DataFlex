from dataflex.core.registry import register_mixer
from dataflex.utils.logging import logger
from .base_mixer import Mixer

import numpy as np

@register_mixer("random")
class RandomMixer(Mixer):
    def __init__(self, mixture_manager, seed):
        super().__init__(mixture_manager)
        self.seed = seed
    
    def mix(self, model, step_id: int, **kwargs) -> np.ndarray:
        """
        随机生成一组比例向量。

        Returns:
            np.ndarray: 长度为源数量的归一化比例数组。
        """
        k = len(self.mixture_manager.names)
        np.random.seed(self.seed)
        raw = np.random.random(k)
        probs = raw / raw.sum()  # 归一化
        logger.info(f"[RandomMixer] Step {step_id} Generated proportions: {probs}")

        return probs

