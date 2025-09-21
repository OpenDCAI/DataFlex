from abc import ABC, abstractmethod
from typing import List
import torch
from torch import distributed as dist

class Selector(ABC):
    def __init__(self, dataset, accelerator, data_collator, cache_dir):
        self.dataset = dataset
        self.accelerator = accelerator
        self.data_collator = data_collator
        self.cache_dir = cache_dir
        self.seed = 42
    
    def warmup(self, num_samples: int, replacement: bool) -> List[List[int]]:
        if self.accelerator.is_main_process:
            dataset_size = len(self.dataset)
            gen = torch.Generator()
            gen.manual_seed(self.seed)

            if replacement:
                full_indices = torch.randint(
                    low=0, high=dataset_size, size=(num_samples,), generator=gen
                ).tolist()
            else:
                if num_samples > dataset_size:
                    raise ValueError(
                        f"Cannot sample {num_samples} without replacement from {dataset_size} samples"
                    )
                full_indices = torch.randperm(dataset_size, generator=gen)[:num_samples].tolist()
        else:
            full_indices = None

        obj = [full_indices]
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(obj, src=0)
            full_indices = obj[0]
        else:
            full_indices = full_indices or []

        return full_indices

    @abstractmethod
    def select(self, model, step_id: int, num_samples: int, **kwargs):
        """
        Select samples from the dataset for the model in 'step_id'.

        Args:
            model: The model object used in the selection process.
            step_id (int): The ID of the current training step or stage.
            num_samples (int): The number of samples to select.
            **kwargs: Additional keyword arguments, allowing for flexible expansion by subclasses.

        Returns:
            List[int]: A list of the selected sample indices.
        """
        pass