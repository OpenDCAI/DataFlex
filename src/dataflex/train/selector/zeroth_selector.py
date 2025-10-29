from dataflex.core.registry import register_selector
from .base_selector import Selector, logger

import torch
from torch.nn.functional import normalize
from typing import List, Dict, Optional
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from trak.projectors import BasicProjector, CudaProjector, ProjectionType
import numpy as np
import json
import os
import glob # 用于文件查找

# NEW: IndexedDataset Wrapper
class IndexedDataset(Dataset):
    """一个包装类，用于在返回样本的同时返回其索引。"""
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        return index, self.original_dataset[index]
    
@register_selector('zeroth')
class ZerothSelector(Selector):
    def __init__(
            self,
            dataset, 
            eval_dataset,
            accelerator, 
            data_collator,
            cache_dir,
    ):
        super().__init__(dataset, accelerator, data_collator, cache_dir)
        self.eval_dataset = eval_dataset
        self.device = self.accelerator.device
        self.dtype = torch.float16
        self.zo_eps = 1e-3
        self.oracle = np.random.randint(100000000)
        self.names = []
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"ZerothSelector initialized. Oracle is {self.oracle}")
    
    def _get_number_of_params(self, model) -> int:
        """计算模型中的总参数数量，零阶方法不计算梯度。"""
        num_params = sum(p.numel() for p in model.parameters())
        if self.accelerator.is_main_process:
            logger.info(f"Total number of parameters that require gradients: {num_params}")
        return num_params
    def _get_trak_projector(self):
        """获取 TRAK projector，优先使用 CUDA 版本。"""
        try:
            import fast_jl
            num_sms = torch.cuda.get_device_properties(self.device.index).multi_processor_count
            fast_jl.project_rademacher_8(torch.zeros(8, 1_000, device=self.device), 512, 0, num_sms)
            projector = CudaProjector
            if self.accelerator.is_main_process:
                logger.info("Using CudaProjector for gradient projection.")
        except (ImportError, RuntimeError):
            projector = BasicProjector
            if self.accelerator.is_main_process:
                logger.info("CudaProjector not available. Using BasicProjector for gradient projection.")
        return projector
    def _reset_Generator(self,model):
        device = next(model.parameters()).device
        g = torch.Generator(device)
        g.manual_seed(self.oracle)
        return g

    def _efficient_perturb(self,model,scaling_factor=1):
        """给定seed,resample出来的随机变量是一样的,利用此可以不存储近似梯度"""
        #if dist.is_initialized():
        #    dist.barrier()
        with torch.no_grad():
            g = self._reset_Generator(model)
            for name,param in model.named_parameters():
                z = torch.normal(mean=0,std=1,size=param.shape,generator=g,dtype=param.dtype,device=param.device)
                delta = (self.zo_eps * scaling_factor * z.data)
                param += delta
                #if dist.get_rank() == 0 or dist.get_rank() == 7:
                #    print(dist.get_rank(),name)
    def _zo_forward(self,model,batch):
        """返回loss值"""
        with self.accelerator.no_sync(model):
            loss = model(**batch).loss
            return loss
    def _obtain_project_grads(self,model, dataset_to_use,save_dir):
        # 构造 DataLoader
        # NEW: 使用 IndexedDataset 来追踪样本的原始索引
        indexed_dataset = IndexedDataset(dataset_to_use)
        
        # NEW: 定义一个处理索引的 collator
        def indexed_collator_wrapper(features):
            indices = [f[0] for f in features]
            original_data = [f[1] for f in features]
            collated_batch = self.data_collator(original_data)
            return {'indices': torch.tensor(indices), 'batch': collated_batch}

        dataloader = DataLoader(
            indexed_dataset,
            batch_size=1, # 仍然是逐样本计算
            shuffle=False,
            num_workers=2,
            collate_fn=indexed_collator_wrapper,
        )
        dataloader = self.accelerator.prepare(dataloader)

        self.accelerator.wait_for_everyone()

        with torch.inference_mode():
            # 6) 循环计算、投影和保存 (在每个进程上独立进行)
            local_grads_to_project = []
            local_indices_to_project = []
            
            # enumerate(...) 使 batch_idx 从 0 开始
            for _, data in enumerate(tqdm(
                dataloader,
                desc=f"[Process {self.accelerator.process_index}] Calculating Gradients",
                disable=not self.accelerator.is_local_main_process, # 主进程打印进度条
                dynamic_ncols=True,
                position=self.accelerator.process_index,
            )):
                indices = data['indices']
                batch = data['batch']
                
                self._efficient_perturb(model,scaling_factor=1)
                loss1 = self._zo_forward(model,batch)
                self._efficient_perturb(model,scaling_factor=-1)
                self._efficient_perturb(model,scaling_factor=-1)
                loss2 = self._zo_forward(model,batch)
                self._efficient_perturb(model,scaling_factor=1)
                project_value = (loss1 - loss2) / (2 * self.zo_eps)

                local_grads_to_project.append(project_value)
                local_indices_to_project.append(indices)
            
            grads_tensor = torch.stack(local_grads_to_project)
            indices_tensor = torch.cat(local_indices_to_project)
            save_path = os.path.join(save_dir, f"grads-{indices_tensor.max().item()}-rank{self.accelerator.process_index}.pt")
            torch.save({'grads': grads_tensor, 'indices': indices_tensor.cpu()}, save_path)

            self.accelerator.wait_for_everyone()
    def _merge_grads(self,save_dir,num_samples):
        # 接下来 找到每个进程存储的project_value和下标，合并成大tensor
        if self.accelerator.is_main_process:
            logger.info(f"Merge project values")
            files = glob.glob(os.path.join(save_dir,"grads-*-rank*.pt"))
            if not files:
                logger.warning("No gradient files found to merge.")
                return

            # 初始化一个空的张量来存放排序后的数据
            # total_samples 是原始数据集的大小
            final_grads = torch.zeros(num_samples, dtype=torch.float16)

            for file_path in tqdm(files, desc="Merging files"):
                chunk = torch.load(file_path, map_location="cpu")
                grads_chunk = chunk['grads'].to(torch.float16)
                indices_chunk = chunk['indices']
                
                # 使用索引将数据放回正确的位置
                final_grads[indices_chunk] = grads_chunk
            output_file = os.path.join(save_dir, "all_projected_grads.pt")
            torch.save(final_grads, output_file)
            logger.info(f"Saved merged gradients (Shape: {grads_chunk.shape}) to {output_file}")
            
            # Optional: 清理分块文件
            for file_path in files:
                os.remove(file_path)
            logger.info(f"Cleaned up temporary chunk files in {save_dir}")
    def select(self,model,step_id,num_samples,**kwargs) -> List[int]:
        """
        选择得分最高的 num_samples 个样本。
        """
        now_train_save_dir = os.path.join(self.cache_dir, "train", str(step_id))
        now_eval_save_dir = os.path.join(self.cache_dir, "eval", str(step_id))
        
        self.step_id = step_id
        train_final_grads_path = os.path.join(now_train_save_dir, "all_projected_grads.pt")
        eval_final_grads_path = os.path.join(now_eval_save_dir, "all_projected_grads.pt")

        # 步骤 1: 计算训练集梯度
        if not os.path.exists(train_final_grads_path):
            os.makedirs(now_train_save_dir, exist_ok=True)
            self._obtain_project_grads(model,self.dataset,now_train_save_dir)
            self._merge_grads(now_train_save_dir,len(self.dataset))
        self.accelerator.wait_for_everyone()

        # 步骤 2: 计算验证集梯度
        if not os.path.exists(eval_final_grads_path):
            os.makedirs(now_eval_save_dir, exist_ok=True)
            # MODIFIED: 传入 eval_dataset
            self._obtain_project_grads(model,self.eval_dataset,now_eval_save_dir)
            self._merge_grads(now_eval_save_dir,len(self.eval_dataset))
        self.accelerator.wait_for_everyone()

        # 步骤 3: 主进程加载、计算分数并选择 top-k
        if self.accelerator.is_main_process:
            logger.info(f"Loading projected gradients from {train_final_grads_path}")
            train_projected_grads = torch.load(train_final_grads_path, map_location="cpu")

            logger.info(f"Loading projected gradients from {eval_final_grads_path}")
            eval_projected_grads = torch.load(eval_final_grads_path, map_location="cpu")

            train_eval_similarities = (train_projected_grads.unsqueeze(1) @ eval_projected_grads.unsqueeze(1).T).mean(dim=1)

            topk = torch.topk(train_eval_similarities, k=num_samples, largest=True)
            selected_indices = topk.indices.tolist()

            logger.info(f"Selecting top {num_samples} samples from {len(train_eval_similarities)}.")
            #selected_indices = list(range(1,769))
            with open(os.path.join(self.cache_dir, str(self.step_id) + "step_selected_indices.json"), "w") as f:
                json.dump({"selected_indices": selected_indices}, f)
        else:
            selected_indices = None

        # 步骤 4: 广播选择的索引
        obj_list = [selected_indices]
        if dist.is_initialized():
            dist.broadcast_object_list(obj_list, src=0)
        selected_indices = obj_list[0]

        return selected_indices

    def random_select(self, num_samples: int, replacement: bool = False) -> List[int]:
        """
        随机选择样本，作为 warmup 或 baseline.
        """
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

        obj_list = [full_indices]
        if dist.is_initialized():
            dist.broadcast_object_list(obj_list, src=0)
        
        return obj_list[0]