from dataflex.core.registry import register_mixer
from dataflex.utils.logging import logger
from .base_mixer import Mixer

import numpy as np

@register_mixer("static")
class StaticMixer(Mixer):
    def __init__(self, mixture_manager, proportions=None, **kwargs):
        """
        Initialize Static Mixer for DoReMi Step 1 and Step 3.
        
        Args:
            mixture_manager: The mixture manager object
            proportions: Fixed proportions for each domain. If None, uses uniform distribution.
        """
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
        """
        Return the fixed proportions for all training steps.
        
        This mixer is designed for DoReMi Step 1 (reference model training) 
        and Step 3 (large model training with fixed optimized weights).
        
        Args:
            model: Current model being trained (not used in static mixer)
            step_id: Current training step (not used in static mixer)
            **kwargs: Additional arguments (not used in static mixer)
            
        Returns:
            np.ndarray: Fixed proportions for all domains
        """
        logger.info(f"[StaticMixer] Step {step_id} Using fixed proportions: {self.proportions}")
        
        # Log domain-wise proportions for clarity
        for i, name in enumerate(self.mixture_manager.names):
            logger.info(f"  {name}: {self.proportions[i]:.4f}")
        
        return self.proportions.copy()
    
    def get_proportions(self):
        """Get the current fixed proportions."""
        return self.proportions.copy()
    
    def set_proportions(self, new_proportions):
        """
        Update the fixed proportions (useful for DoReMi Step 3 
        when using optimized weights from Step 2).
        
        Args:
            new_proportions: New proportions to use
        """
        k = len(self.mixture_manager.names)
        new_proportions = np.array(new_proportions, dtype=float)
        
        if len(new_proportions) != k:
            raise ValueError(f"[StaticMixer] Number of proportions ({len(new_proportions)}) "
                           f"must match number of domains ({k})")
        
        if np.any(new_proportions < 0):
            raise ValueError("[StaticMixer] All proportions must be non-negative")
        
        if np.sum(new_proportions) == 0:
            raise ValueError("[StaticMixer] Sum of proportions cannot be zero")
        
        # Normalize and update
        self.proportions = new_proportions / np.sum(new_proportions)
        
        logger.info(f"[StaticMixer] Updated proportions to: {self.proportions}")
        logger.info(f"[StaticMixer] Domain names: {self.mixture_manager.names}")
        
        # Log domain-wise proportions
        for i, name in enumerate(self.mixture_manager.names):
            logger.info(f"  {name}: {self.proportions[i]:.4f}")
    
    def save_proportions(self, output_dir, step_id):
        """Save current proportions to file."""
        try:
            import json
            import os
            
            proportions_file = os.path.join(output_dir, f"static_proportions_step_{step_id}.json")
            
            proportions_data = {
                "step": step_id,
                "domain_names": self.mixture_manager.names,
                "proportions": self.proportions.tolist(),
                "mixer_type": "static"
            }
            
            os.makedirs(output_dir, exist_ok=True)
            with open(proportions_file, 'w') as f:
                json.dump(proportions_data, f, indent=2)
                
            logger.info(f"[StaticMixer] Saved proportions to {proportions_file}")
            
        except Exception as e:
            logger.warning(f"[StaticMixer] Failed to save proportions: {e}")
    
    def load_proportions(self, proportions_file):
        """Load proportions from file."""
        try:
            import json
            with open(proportions_file, 'r') as f:
                proportions_data = json.load(f)
            
            self.proportions = np.array(proportions_data["proportions"])
            
            logger.info(f"[StaticMixer] Loaded proportions from {proportions_file}")
            logger.info(f"[StaticMixer] Loaded proportions: {self.proportions}")
            return True
            
        except Exception as e:
            logger.warning(f"[StaticMixer] Failed to load proportions: {e}")
            return False