from abc import ABC, abstractmethod
import numpy as np

class Mixer(ABC):
    def __init__(self, mixture_manager):
        self.mixture_manager = mixture_manager
    
    @abstractmethod
    def mix(self, model, step_id: int, **kwargs) -> np.ndarray:
        """
        Change the proportions of samples for the model in 'step_id'.

        Args:
            model: The model object used in the selection process.
            step_id (int): The ID of the current training step or stage.
            **kwargs: Additional keyword arguments, allowing for flexible expansion by subclasses.

        Returns:
            np.ndarray: The updated proportions of samples for the model in 'step_id'.
        """
        pass