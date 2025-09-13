import torch
from typing import List
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import json
import os
from dataflex.core.registry import register_selector

import logging
import sys
logging.basicConfig(level=logging.INFO)
handler = logging.StreamHandler(sys.stdout)
logger = logging.getLogger(__name__)
logger.addHandler(handler)

class IndexedDataset(Dataset):
    def __init__(self, original_dataset):
        self.dataset = original_dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        data = self.dataset[index]
        return {"idx": index, **data}

@register_selector('loss')
class LossSelector:
    def __init__(self, dataset, accelerator, data_collator, cache_dir):
        self.dataset = dataset
        self.accelerator = accelerator
        self.seed = 42
        self.data_collator = data_collator
        self.cache_dir = cache_dir
    
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


    def select(self, model, step_id: int, num_samples: int, **kwargs):
        model.eval()
        save_dir = self.cache_dir
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"step_{step_id}.json")
    
        n = len(self.dataset)    
        # ========= 加载或计算（新逻辑，仅支持新文件结构） =========
        if os.path.exists(save_path):
            if self.accelerator.is_main_process:
                with open(save_path, "r") as f:
                    saved = json.load(f)
                gathered_losses = torch.tensor(saved["losses"], dtype=torch.float32)
                assert len(gathered_losses) == n, "保存的 losses 长度应与 dataset 等长"
            else:
                gathered_losses = None
        else:
            # 1) DataLoader
            dataloader = DataLoader(
                IndexedDataset(self.dataset),
                batch_size=1,            
                shuffle=False,
                num_workers=2,
                collate_fn=self.data_collator, 
            )
            dataloader = self.accelerator.prepare(dataloader)
    
            # 2) 本地收集 loss 与 idx
            logger.info(f"[Dataflex] Calculating loss using {self.accelerator.num_processes} GPUs")
            local_losses, local_indices = [], []
            for batch in tqdm(
                dataloader,
                desc=f"[Selector step {step_id}]",
                disable=not self.accelerator.is_main_process,
                dynamic_ncols=True,
            ):
                idx = batch["idx"]
                if not torch.is_tensor(idx):
                    idx = torch.tensor(idx, dtype=torch.long, device=self.accelerator.device)
                idx = idx.view(-1).to(dtype=torch.long)
    
                with torch.no_grad():
                    # 注意从 batch 中移除 'idx' 再喂给模型
                    model_inputs = {k: v for k, v in batch.items() if k != "idx"}
                    loss = model(**model_inputs).loss.detach().view(-1)  # [B]
    
                local_losses.append(loss)
                local_indices.append(idx)
    
            local_losses  = torch.cat(local_losses,  dim=0)  # [N_local_padded]
            local_indices = torch.cat(local_indices, dim=0)  # [N_local_padded]
    
            # 3) 各进程 gather（按 rank 串联，可能含补齐/重复）
            all_losses  = self.accelerator.gather(local_losses)
            all_indices = self.accelerator.gather(local_indices)
    
            # 4) 主进程按 idx 去重并对齐到 len(dataset)
            if self.accelerator.is_main_process:
                aligned = torch.full((n,), float("inf"), dtype=all_losses.dtype, device=all_losses.device)
                seen = set()
                # 采用“首次出现优先”保证确定性
                for l, i in zip(all_losses.tolist(), all_indices.tolist()):
                    if 0 <= i < n and i not in seen:
                        aligned[i] = l
                        seen.add(i)
                # 若极端情况下有没覆盖到的 idx，仍为 +inf；不会进 largest=True 的 topk
                gathered_losses = aligned
    
                # 5) 保存（新格式，仅 losses 等长 + indices 为 0..n-1）
                with open(save_path, "w") as f:
                    json.dump(
                        {
                            "losses": gathered_losses.cpu().tolist(),
                            "indices": list(range(n)),
                        },
                        f,
                    )
                logger.info(f"[Dataflex] Loss calculation finished, saved to {save_path}")
            else:
                gathered_losses = None
    
        # ========= 广播 gathered_losses（等长张量） =========
        gathered_list = [gathered_losses if self.accelerator.is_main_process else None]
        dist.broadcast_object_list(gathered_list, src=0)
        gathered_losses = gathered_list[0]
    
        # ========= 主进程 topk，并广播 sel =========
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
