from dataflex.core.registry import register_mixer
from dataflex.utils.logging import logger
from .base_mixer import Mixer

import numpy as np

@register_mixer("static")
class StaticMixer(Mixer):
    def __init__(self, mixture_manager, proportions=None, **kwargs):

        super().__init__(mixture_manager)
        
        k = len(self.mixture_manager.names)
        
        if proportions is None:
            # Use uniform distribution if no proportions specified
            self.proportions = np.ones(k) / k
            logger.info(f"[StaticMixer] No proportions specified, using uniform distribution: {self.proportions}")
        else:
            # Validate and normalize proportions
            proportions = np.array(proportions, dtype=float)
            
            if len(proportions) != k:
                raise ValueError(f"[StaticMixer] Number of proportions ({len(proportions)}) "
                               f"must match number of domains ({k})")
            
            if np.any(proportions < 0):
                raise ValueError("[StaticMixer] All proportions must be non-negative")
            
            if np.sum(proportions) == 0:
                raise ValueError("[StaticMixer] Sum of proportions cannot be zero")
            
            # Normalize to ensure they sum to 1
            self.proportions = proportions / np.sum(proportions)
            
            logger.info(f"[StaticMixer] Using fixed proportions: {self.proportions}")
            logger.info(f"[StaticMixer] Domain names: {self.mixture_manager.names}")
    
    def mix(self, model, step_id: int, **kwargs) -> np.ndarray:

        logger.info(f"[StaticMixer] Step {step_id} Using fixed proportions: {self.proportions}")
        
        # Log domain-wise proportions for clarity
        for i, name in enumerate(self.mixture_manager.names):
            logger.info(f"  {name}: {self.proportions[i]:.4f}")
        
        return self.proportions.copy()
