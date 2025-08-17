import torch
from typing import List
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import DataLoader
from llamafactory.extras import logging
import json
import os

logger = logging.get_logger(__name__)

class DynamicSelector:
    def __init__(self, dataset, accelerator, data_collator):
        self.dataset = dataset
        self.accelerator = accelerator
        self.seed = 42
        self.data_collator = data_collator
    
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


    def select(self, model, step_id: int, num_samples: int):
        model.eval()

        save_dir = "/mnt/public2/code/zzy/LLaMA-Flex/saves/dynamic"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"step_{step_id}.json")

        # ========== 加载或计算 gathered_losses ==========
        if os.path.exists(save_path):
            if self.accelerator.is_main_process:
                with open(save_path, "r") as f:
                    saved = json.load(f)
                logger.info(f"[DynamicTrain] Loss exists, loaded from {save_path}")
                gathered_losses = torch.tensor(saved["losses"])
            else:
                gathered_losses = None
        else:
            # 1) 构造 DataLoader
            dataloader = DataLoader(
                self.dataset,
                batch_size=4,
                shuffle=False,
                num_workers=2,
                collate_fn=self.data_collator,
            )
            dataloader = self.accelerator.prepare(dataloader)

            # 2) 收集 loss
            logger.info(f"[DynamicTrain] Calculating loss using {self.accelerator.num_processes} GPUs")
            local_losses = []
            for batch in tqdm(
                dataloader,
                desc=f"[Selector step {step_id}]",
                disable=not self.accelerator.is_main_process,
                dynamic_ncols=True
            ):
                with torch.no_grad():
                    loss = model(**batch).loss.detach().unsqueeze(0)
                local_losses.append(loss)

            local_losses = torch.cat(local_losses)
            gathered_losses = self.accelerator.gather(local_losses)

            # 仅主进程保存
            if self.accelerator.is_main_process:
                with open(save_path, "w") as f:
                    json.dump({"losses": gathered_losses.cpu().tolist()}, f)
                logger.info(f"[DynamicTrain] Loss calculation finished, saved to {save_path}")

        # ========== 所有进程都广播 gathered_losses ==========
        gathered_list = [gathered_losses if self.accelerator.is_main_process else None]
        dist.broadcast_object_list(gathered_list, src=0)
        gathered_losses = gathered_list[0]

        # ========== 主进程执行 topk，并广播 sel ==========
        if self.accelerator.is_main_process:
            topk = torch.topk(gathered_losses, k=num_samples, largest=True)
            sel = topk.indices.tolist()
        else:
            sel = None

        sel_list = [sel]
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(sel_list, src=0)
            sel = sel_list[0]
        else:
            sel = sel or []

        return sel



