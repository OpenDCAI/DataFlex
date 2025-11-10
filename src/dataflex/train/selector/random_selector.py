from dataflex.core.registry import register_selector
from dataflex.utils.logging import logger
from .base_selector import Selector

import torch
import torch.distributed as dist


@register_selector("random")
class RandomSelector(Selector):
    """随机从训练数据中抽取样本的选择器。"""

    def __init__(
        self,
        dataset,
        accelerator,
        data_collator,
        cache_dir,
        seed: int = 42,
        replacement: bool = False,
    ):
        super().__init__(dataset, accelerator, data_collator, cache_dir)
        self.seed = seed
        self.replacement = replacement

    def select(self, model, step_id: int, num_samples: int, **kwargs):
        if self.accelerator.is_main_process:
            dataset_size = len(self.dataset)
            generator = torch.Generator()
            generator.manual_seed(self.seed + int(step_id))

            if self.replacement:
                selected_indices = torch.randint(
                    low=0,
                    high=dataset_size,
                    size=(num_samples,),
                    generator=generator,
                ).tolist()
            else:
                if num_samples > dataset_size:
                    raise ValueError(
                        f"Cannot sample {num_samples} without replacement from {dataset_size} samples"
                    )
                selected_indices = torch.randperm(dataset_size, generator=generator)[:num_samples].tolist()

            logger.info(
                f"[RandomSelector] Selected {len(selected_indices)} samples at step {step_id} with replacement={self.replacement}."
            )
        else:
            selected_indices = None

        indices_obj = [selected_indices]
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(indices_obj, src=0)
            selected_indices = indices_obj[0]
        else:
            selected_indices = selected_indices or []

        return selected_indices