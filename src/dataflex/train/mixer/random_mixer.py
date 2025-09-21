from typing import List
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import logging
import sys
logging.basicConfig(level=logging.INFO)
handler = logging.StreamHandler(sys.stdout)
logger = logging.getLogger(__name__)
logger.addHandler(handler)
from dataflex.core.registry import register_mixer
from dataflex.utils.logging import logger

@register_mixer("random")
class RandomMixer:
    def __init__(self, seed, mixture_manager):
        self.seed = seed
        self.mixture_manager = mixture_manager
    
    def mix(self) -> np.ndarray:
        """
        随机生成一组比例向量。

        Returns:
            np.ndarray: 长度为源数量的归一化比例数组。
        """
        k = len(self.mixture_manager.names)
        raw = np.random.random(k)
        probs = raw / raw.sum()  # 归一化
        logger.info(f"[RandomMixer] Generated proportions: {probs}")

        return probs

