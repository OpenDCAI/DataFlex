from dataflex.core.registry import register_selector
from .base_selector import Selector, logger

import torch
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import json
import os

class IndexedDataset(Dataset):
    def __init__(self, original_dataset):
        self.dataset = original_dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        data = self.dataset[index]
        return {"idx": index, **data}

@register_selector('loss')
class LossSelector(Selector):
    def __init__(
        self,
        dataset,
        accelerator,
        data_collator,
        cache_dir,
        focus: str = "high",              # "high" | "medium" | "low"
        focus_weight: float = 5.0,        # 权重倍数
        quantiles: tuple = (0.33, 0.66),  # 低/中/高切分点
        replacement: bool = False,        # 是否放回采样
        temperature: float = 1.0,         # 温度控制
    ):
        super().__init__(dataset, accelerator, data_collator, cache_dir)

        # 新增的采样控制参数
        self.focus = str(focus).lower()
        if self.focus not in {"low", "medium", "high"}:
            raise ValueError("focus 必须是 'low'、'medium' 或 'high'")
        self.focus_weight = focus_weight
        self.quantiles = quantiles
        self.replacement = replacement
        self.temperature = temperature

        logger.info(f"LossSelector initialized.")

    def select(self, model, step_id: int, num_samples: int, **kwargs):
        model.eval()
        os.makedirs(self.cache_dir, exist_ok=True)
        save_path = os.path.join(self.cache_dir, f"step_{step_id}.json")
    
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
    
        # ========= 主进程：基于分布的采样 =========
        if self.accelerator.is_main_process:
            logger.info(f"[Dataflex] focus={self.focus}, focus_weight={self.focus_weight}")
            losses = gathered_losses.clone().detach().float()
            valid_mask = torch.isfinite(losses)

            if valid_mask.sum().item() == 0:
                probs = torch.full((len(losses),), 1.0 / len(losses))
            else:
                valid_losses = losses[valid_mask]
                q1 = torch.quantile(valid_losses, self.quantiles[0])
                q2 = torch.quantile(valid_losses, self.quantiles[1])

                low_mask    = (losses <= q1) & valid_mask
                medium_mask = (losses > q1) & (losses <= q2) & valid_mask
                high_mask   = (losses > q2) & valid_mask

                weights = torch.zeros_like(losses).float()
                weights[low_mask]    = 1.0
                weights[medium_mask] = 1.0
                weights[high_mask]   = 1.0

                if self.focus == "low":
                    weights[low_mask] *= self.focus_weight
                elif self.focus == "medium":
                    weights[medium_mask] *= self.focus_weight
                else:
                    weights[high_mask] *= self.focus_weight

                weights[~valid_mask] = 0.0
                eps = 1e-12
                probs = (weights + eps) ** (1.0 / self.temperature)
                if probs.sum() == 0.0:
                    probs = valid_mask.float()
                probs = probs / probs.sum()

            available = int((probs > 0).sum().item())
            effective_replacement = self.replacement
            if not effective_replacement and num_samples > available:
                effective_replacement = True
                logger.info(
                    f"[Dataflex] 有效样本量 {available} 小于请求数量 {num_samples}，"
                    f"已自动改为放回采样。"
                )

            gen = torch.Generator()
            gen.manual_seed(self.seed + int(step_id))
            sel_tensor = torch.multinomial(
                probs.cpu(), num_samples=num_samples,
                replacement=effective_replacement, generator=gen
            )
            sel = sel_tensor.tolist()
        else:
            sel = None

        # 广播 sel
        sel_list = [sel]
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(sel_list, src=0)
            sel = sel_list[0]
        else:
            sel = sel or []
    
        return sel