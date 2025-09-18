import torch
import os
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from dataflex.core.registry import register_selector
from .base_selector import Selector
from dataflex.utils.logging import logger
from dataflex.utils.selector_io import load_cached_selection, save_selection

def sigmoid(x, k):
    return 1 / (1 + np.exp(-k * (x - 0.5)))

# 计算窗口位置
def calculate_window_position(current_update_times, update_times, dataset_len, window_size=0.2, k=10):
    scaled_iteration = current_update_times / update_times

    delta = sigmoid(scaled_iteration, k) * (dataset_len - window_size * dataset_len)
    
    # 计算窗口的起始和结束位置
    window_start = delta
    window_end = delta + window_size * dataset_len
    return int(window_start), int(window_end)


class IndexedDataset(Dataset):
    def __init__(self, original_dataset):
        self.dataset = original_dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        data = self.dataset[index]
        return {"idx": index, **data}

@register_selector('delta_loss')
class DeltaLossSelector(Selector):
    def __init__(
        self,
        dataset,
        accelerator,
        data_collator,
        cache_dir,
        window_size: float = 0.2,         # 窗口大小，默认20%
    ):
        super().__init__(dataset, accelerator, data_collator, cache_dir)
        self.seed = 42
        self.window_size = window_size
        self.initial_losses = None
        self.first_time = True
        self.path_to_initial_losses = None

    def compute_loss(self, dataloader, model, step_id):
        dataloader = self.accelerator.prepare(dataloader)
        n = len(self.dataset)
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
            logger.info(f"[Dataflex] Loss calculation finished")
        else:
            gathered_losses = None
        return gathered_losses

    def select(self, model, step_id: int, num_samples: int, **kwargs):
        model.eval()
        os.makedirs(self.cache_dir, exist_ok=True)
        save_path = os.path.join(self.cache_dir, f"step_{step_id}.json")

        n = len(self.dataset)

        # ========= 第一次调用select，计算并保存 initial_losses =========
        if self.first_time == True:
            self.first_time = False
            self.path_to_initial_losses = save_path
            # 读取并在main中广播
            if os.path.exists(save_path):
                if self.accelerator.is_main_process:
                    cached_indices, _ = load_cached_selection(save_path)
                else:
                    cached_indices = None
                cached_indices_list = [cached_indices]
                if dist.is_available() and dist.is_initialized():
                    dist.broadcast_object_list(cached_indices_list, src=0)
                    cached_indices = cached_indices_list[0]
                else:
                    cached_indices = cached_indices or []
                return cached_indices
            
            logger.info(f"[Dataflex] Calculating initial losses...")
            dataloader = DataLoader(
                IndexedDataset(self.dataset),
                batch_size=1,            
                shuffle=False,
                num_workers=2,
                collate_fn=self.data_collator, 
            )
            gathered_losses = self.compute_loss(dataloader, model, step_id)
            if self.accelerator.is_main_process:
                logger.info(f"[Dataflex] Got initial_losses. Return random warmup selection.")
                gen = torch.Generator()
                gen.manual_seed(self.seed)
                if num_samples > n:
                    raise ValueError(
                        f"Cannot sample {num_samples} without replacement from {n} samples"
                    )
                full_indices = torch.randperm(n, generator=gen)[:num_samples].tolist()
                sel = full_indices
                metric_payload = {
                    "loss": gathered_losses.tolist()
                }
                # 只有main中才会保存
                save_selection(save_path, sel, metric_payload, self.accelerator)
                
            else:
                full_indices = None

            obj = [full_indices]
            if dist.is_available() and dist.is_initialized():
                dist.broadcast_object_list(obj, src=0)
                full_indices = obj[0]
            else:
                full_indices = full_indices or []

            return full_indices
        
        # ========= 后续调用 select，根据 delta_loss 选择样本 =========
        
        if os.path.exists(save_path):
            if self.accelerator.is_main_process:
                cached_indices, _ = load_cached_selection(save_path)
            else:
                cached_indices = None
            cached_indices_list = [cached_indices]
            if dist.is_available() and dist.is_initialized():
                dist.broadcast_object_list(cached_indices_list, src=0)
                cached_indices = cached_indices_list[0]
            else:
                cached_indices = cached_indices or []
            return cached_indices
        logger.info(f"[Dataflex] Calculating current losses...")
        dataloader = DataLoader(
            IndexedDataset(self.dataset),
            batch_size=1,            
            shuffle=False,
            num_workers=2,
            collate_fn=self.data_collator, 
        )
        
        gathered_losses = self.compute_loss(dataloader, model, step_id)
        
        # ========= 广播 current_losses（等长张量） =========
        # losses_list = [gathered_losses if self.accelerator.is_main_process else None]
        # dist.broadcast_object_list(losses_list, src=0)
        # current_losses = losses_list[0]
        current_losses = gathered_losses
        # ========= Delta Loss 选择 =========
        if self.accelerator.is_main_process:
            logger.info(f"[Dataflex] Loading initial losses from {self.path_to_initial_losses}")
            
            cached_indices, metrics = load_cached_selection(self.path_to_initial_losses)
            
            self.initial_losses = torch.tensor(metrics["loss"], dtype=torch.float32)

            logger.info(f"[Dataflex] Selecting samples based on delta loss.")
            
            # 计算损失差（delta loss）
            delta_loss = self.initial_losses.to(current_losses.device) - current_losses

            # 排序索引
            sorted_indices = torch.argsort(delta_loss, descending=True)

            # 计算滑动窗口的位置
            window_start, window_end = calculate_window_position(kwargs["current_update_times"]-1, kwargs["update_times"]-1, n, window_size=self.window_size)
            invalid_position = (delta_loss[sorted_indices] < 0).nonzero()
            if len(invalid_position) > 0:
                invalid_position = invalid_position[0].item()  # 获取第一个小于0的位置
                window_end = min(window_end, invalid_position)  # 设置窗口右端点
            
            # 输出选择次数、窗口位置
            logger.info(f"[Dataflex] Step {step_id}, Update {kwargs['current_update_times']-1}/{kwargs['update_times']-1}, Window position: [{window_start}, {window_end})")
            # 输出窗口内最大的和最小的五个delta loss及其索引
            window_delta_loss = delta_loss[sorted_indices][window_start:window_end]
            if len(window_delta_loss) > 0:
                logger.info(f"[Dataflex] Window delta loss stats:")
                logger.info(f"  Max 5: {window_delta_loss[:5].cpu().numpy()}")
                logger.info(f"  Min 5: {window_delta_loss[-5:].cpu().numpy()}")
            # 获取窗口内的样本索引
            window_indices = torch.arange(window_start, window_end).long()

            probs = torch.full((len(gathered_losses),), 0.025)

            # 设置窗口内的样本的概率较大
            probs[window_indices] = 1.0  # 窗口内的样本设置为较大的概率值（可以调节大小）

            # 归一化概率
            probs = probs / probs.sum()  # 归一化概率，使总和为1

            available = int((probs > 0).sum().item())
            effective_replacement = False
            
            # 如果有效样本量小于请求样本数，使用放回采样
            if not effective_replacement and num_samples > available:
                effective_replacement = True
                logger.info(
                    f"[Dataflex] 有效样本量 {available} 小于请求数量 {num_samples}，"
                    f"已自动改为放回采样。"
                )

            # 创建随机数生成器
            gen = torch.Generator()
            gen.manual_seed(self.seed + int(step_id))
            
            # 使用torch.multinomial进行采样
            sel_tensor = torch.multinomial(probs.cpu(), num_samples=num_samples,
                                        replacement=effective_replacement, generator=gen)
            sel = sel_tensor.tolist()

            # ========= 4) 保存（只保存“被选中的 indices + 对应 metric”） =========
            metric_payload = {
                "delta_loss": [float(delta_loss[i].item()) for i in sel]
            }
            save_selection(save_path, sel, metric_payload, self.accelerator)
        else:
            sel = None
        # 广播选择的样本
        sel_list = [sel]
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(sel_list, src=0)
            sel = sel_list[0]
        else:
            sel = sel or []
        self.accelerator.wait_for_everyone()
        return sel
