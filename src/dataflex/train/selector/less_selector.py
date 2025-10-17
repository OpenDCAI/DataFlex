from dataflex.core.registry import register_selector
from dataflex.utils.selector_io import load_cached_selection, save_selection
from .base_selector import Selector
from dataflex.utils.logging import logger
import torch
from torch.nn.functional import normalize
from typing import List, Dict, Optional
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from trak.projectors import BasicProjector, CudaProjector, ProjectionType
import json
import os
import glob # 用于文件查找

# NEW: IndexedDataset Wrapper
class IndexedDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        return index, self.original_dataset[index]

@register_selector('less')
class LessSelector(Selector):
    def __init__(self, 
                 dataset, 
                 eval_dataset,
                 accelerator, 
                 data_collator,
                 cache_dir,
                 gradient_type: str = "adam",
                 proj_dim: int = 8192,
                 save_interval: int = 16,
                 seed: int = 42):
        """
        初始化 LessSelector.
        """
        super().__init__(dataset, accelerator, data_collator, cache_dir)

        self.eval_dataset = eval_dataset
        self.gradient_type = gradient_type
        self.proj_dim = proj_dim
        self.save_interval = save_interval
        self.seed = seed
        
        self.device = self.accelerator.device
        self.dtype = torch.float16

        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"LessSelector initialized. Projected gradients will be saved in {self.cache_dir}")

    def _get_number_of_params(self, model) -> int:
        """计算模型中需要梯度的参数数量。"""
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if self.accelerator.is_main_process:
            logger.info(f"Total number of parameters that require gradients: {num_params}")
        return num_params

    def _prepare_optimizer_state(self, model, optimizer_state: Dict) -> (torch.Tensor, torch.Tensor):
        """从优化器状态字典中准备 Adam 的一阶和二阶矩估计。"""
        avg_list, avg_sq_list = [], []
        for param in model.parameters():
            if param.requires_grad:
                avg_list.append(optimizer_state[param]["exp_avg"].view(-1))
                avg_sq_list.append(optimizer_state[param]["exp_avg_sq"].view(-1))

        avg = torch.cat(avg_list).to(self.device)
        avg_sq = torch.cat(avg_sq_list).to(self.device)
        return avg, avg_sq

    def _obtain_gradients(self, model, batch, gradient_type, m: Optional[torch.Tensor] = None, v: Optional[torch.Tensor] = None) -> torch.Tensor:
        """根据指定的类型计算单个样本的梯度向量。"""
        with self.accelerator.no_sync(model):
            loss = model(**batch).loss
            self.accelerator.backward(loss)

        vectorized_grads = torch.cat(
            [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
        )

        if gradient_type == "adam":
            if m is None or v is None:
                raise ValueError("Adam optimizer states (m, v) must be provided for 'adam' gradient type.")
            beta1, beta2, eps = 0.9, 0.999, 1e-08
            updated_avg = beta1 * m + (1 - beta1) * vectorized_grads
            updated_avg_sq = beta2 * v + (1 - beta2) * vectorized_grads ** 2
            final_grads = updated_avg / torch.sqrt(updated_avg_sq + eps)
        elif gradient_type == "sgd":
            final_grads = vectorized_grads
        else:
            assert False, f"Unknown gradient type: {gradient_type}"
        
        model.zero_grad()
        return final_grads

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

    def _get_max_saved_index(self, save_dir) -> int:
        """
        MODIFIED: 获取已保存的最大样本数（而不是 chunk 索引），用于断点续传。
        我们通过查看最后一个文件的文件名来推断。
        """
        prefix = "grads"
        if not os.path.exists(save_dir):
            return -1
        # We only need to check this on the main process
        if not self.accelerator.is_main_process:
            return -1
            
        files = [f for f in os.listdir(save_dir) if f.startswith(prefix) and f.endswith(".pt")]
        if not files:
            return -1
        
        # 文件名格式: grads-{count}-rank{rank}.pt
        indices = [int(f.split('.')[0].split('-')[1]) for f in files]
        return max(indices) if indices else -1

    # MODIFIED: 重写核心逻辑
    def _collect_and_save_projected_gradients(self, model, save_dir, dataset_to_use, gradient_type, optimizer_state: Optional[Dict] = None):
        """
        核心函数：每个进程独立计算梯度、投影，并保存带有索引的分块文件。
        """
        # 1) 初始化 Projector (每个进程都需要一个)
        num_params = self._get_number_of_params(model)
        projector_class = self._get_trak_projector()
        projector = projector_class(
            grad_dim=num_params,
            proj_dim=self.proj_dim,
            seed=self.seed,
            proj_type=ProjectionType.rademacher,
            max_batch_size=8,
            block_size=128,
            device=self.device,
            dtype=self.dtype,
        )

        # 2) 准备 Adam 状态 (如果需要)
        m, v = None, None
        if self.gradient_type == "adam":
            if optimizer_state is None:
                raise ValueError("optimizer_state must be provided for 'adam' gradient type.")
            m, v = self._prepare_optimizer_state(model, optimizer_state)
        
        # 3) 构造 DataLoader
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

        # 4) 设置保存间隔
        # MODIFIED: 这是每个进程的本地保存间隔
        save_interval = self.save_interval # 每个进程每处理save_interval个样本就映射并保存一次

        # 5) 断点续传
        max_index = self._get_max_saved_index(save_dir=save_dir)
        start_count = max_index + 1
        if self.accelerator.is_main_process and start_count > 1:
            logger.info(f"Resuming from sample index {start_count}.")
        
        # 等待主进程完成检查
        self.accelerator.wait_for_everyone()

        # 6) 循环计算、投影和保存 (在每个进程上独立进行)
        local_grads_to_project = []
        local_indices_to_project = []
        
        total_samples_in_loader = len(dataloader)
        # enumerate(..., 1) 使 batch_idx 从 1 开始
        for batch_idx, data in enumerate(tqdm(
            dataloader,
            desc=f"[Process {self.accelerator.process_index}] Calculating Gradients",
            disable=not self.accelerator.is_local_main_process, # 主进程打印进度条
            dynamic_ncols=True,
            position=self.accelerator.process_index,
        ), 1):
            
            # 断点续传逻辑，这里的 'count' 应该是全局样本索引
            # DistributedSampler 会自动处理分片，我们只需跳过批次即可
            # 注意: 简单的跳过可能不精确，但对于重启来说是可接受的
            # dataloader.sampler.set_epoch(batch_idx) # 如果需要精确重启，可能需要更复杂的sampler状态管理
            # 这里我们简化处理，假设从头开始或不跳过
  
            indices = data['indices']
            batch = data['batch']

            vectorized_grads = self._obtain_gradients(model, batch, gradient_type, m, v)
            local_grads_to_project.append(vectorized_grads)
            local_indices_to_project.append(indices)

            # 达到保存间隔或处理完所有样本
            if batch_idx % save_interval == 0 or batch_idx == total_samples_in_loader:
                if local_grads_to_project:
                    grads_tensor = torch.stack(local_grads_to_project).to(self.dtype)
                    indices_tensor = torch.cat(local_indices_to_project)
                    
                    # 在当前进程上进行投影
                    projected = projector.project(grads_tensor, model_id=0).cpu()

                    # 保存投影梯度和对应的索引
                    # 文件名包含 batch_idx 和 rank 以确保唯一性
                    #  252 to 255     605 to 511 
                    save_path = os.path.join(save_dir, f"grads-{indices_tensor.max().item()}-rank{self.accelerator.process_index}.pt")
                    torch.save({'grads': projected, 'indices': indices_tensor.cpu()}, save_path)
                    
                    # 清空列表以备下一批
                    local_grads_to_project = []
                    local_indices_to_project = []
        
        # 等待所有进程完成文件写入
        self.accelerator.wait_for_everyone()


    # MODIFIED: 重写合并逻辑
    def _merge_and_normalize_info(self, save_dir, total_samples):
        """
        在主进程上合并所有分块文件，根据索引重建顺序，然后归一化。
        """
        if self.accelerator.is_main_process:
            logger.info(f"Merging and normalizing projected gradients from {save_dir}")
            
            # 使用 glob 查找所有 rank 保存的文件
            files = glob.glob(os.path.join(save_dir, "grads-*-rank*.pt"))
            if not files:
                logger.warning("No gradient files found to merge.")
                return

            # 初始化一个空的张量来存放排序后的数据
            # total_samples 是原始数据集的大小
            final_grads = torch.zeros(total_samples, self.proj_dim, dtype=torch.float32)

            for file_path in tqdm(files, desc="Merging files"):
                chunk = torch.load(file_path, map_location="cpu")
                grads_chunk = chunk['grads'].to(torch.float32)
                indices_chunk = chunk['indices']
                
                # 使用索引将数据放回正确的位置
                final_grads[indices_chunk] = grads_chunk
            
            # 归一化整个张量
            normalized_data = normalize(final_grads, dim=1)
            
            output_file = os.path.join(save_dir, "all_projected_grads.pt")
            torch.save(normalized_data, output_file)
            logger.info(f"Saved merged and normalized gradients (Shape: {normalized_data.shape}) to {output_file}")
            
            # Optional: 清理分块文件
            for file_path in files:
                os.remove(file_path)
            logger.info(f"Cleaned up temporary chunk files in {save_dir}")

    def select(self, model, step_id: int, num_samples: int, **kwargs) -> List[int]:
        """
        选择得分最高的 num_samples 个样本。
        """

        # 有无存储的step顺序
        os.makedirs(self.cache_dir, exist_ok=True)
        save_path = os.path.join(self.cache_dir, f"step_{step_id}.json")
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

        now_train_save_dir = os.path.join(self.cache_dir, "train", str(step_id))
        now_eval_save_dir = os.path.join(self.cache_dir, "eval", str(step_id))
        
        self.step_id = step_id
        train_final_grads_path = os.path.join(now_train_save_dir, "all_projected_grads.pt")
        eval_final_grads_path = os.path.join(now_eval_save_dir, "all_projected_grads.pt")

        # 步骤 1: 计算训练集梯度
        if not os.path.exists(train_final_grads_path):
            os.makedirs(now_train_save_dir, exist_ok=True)
            optimizer_state = kwargs.get('optimizer_state', None) 
            self._collect_and_save_projected_gradients(model, now_train_save_dir, self.dataset, self.gradient_type, optimizer_state)
            self._merge_and_normalize_info(now_train_save_dir, len(self.dataset))
        
        self.accelerator.wait_for_everyone()

        # 步骤 2: 计算验证集梯度
        if not os.path.exists(eval_final_grads_path):
            os.makedirs(now_eval_save_dir, exist_ok=True)
            # MODIFIED: 传入 eval_dataset
            self._collect_and_save_projected_gradients(model, now_eval_save_dir, self.eval_dataset, "sgd", None)
            self._merge_and_normalize_info(now_eval_save_dir, len(self.eval_dataset))
        
        self.accelerator.wait_for_everyone()

        # 步骤 3: 主进程加载、计算分数并选择 top-k
        if self.accelerator.is_main_process:
            logger.info(f"Loading projected gradients from {train_final_grads_path}")
            train_projected_grads = torch.load(train_final_grads_path, map_location="cpu")

            logger.info(f"Loading projected gradients from {eval_final_grads_path}")
            eval_projected_grads = torch.load(eval_final_grads_path, map_location="cpu")

            train_eval_similarities = (train_projected_grads @ eval_projected_grads.T).mean(dim=1)
            topk = torch.topk(train_eval_similarities, k=num_samples, largest=True)
            selected_indices = topk.indices.tolist()

            logger.info(f"Selecting top {num_samples} samples from {len(train_eval_similarities)}.")
        
            # ========= 4) 保存（只保存“被选中的 indices + 对应 metric”） =========
            metric_payload = {
                "train_eval_similarity": [float(train_eval_similarities[i].item()) for i in selected_indices]
            }
            save_selection(save_path, selected_indices, metric_payload, self.accelerator)
        else:
            selected_indices = None

        # 步骤 4: 广播选择的索引
        obj_list = [selected_indices]
        if dist.is_initialized():
            dist.broadcast_object_list(obj_list, src=0)
        selected_indices = obj_list[0]

        return selected_indices