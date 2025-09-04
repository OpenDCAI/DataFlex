import torch
from torch.nn.functional import normalize
from typing import List, Dict, Optional
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import DataLoader
from trak.projectors import BasicProjector, CudaProjector, ProjectionType
import json
import os

import logging
import sys
logging.basicConfig(level=logging.INFO)
handler = logging.StreamHandler(sys.stdout)
logger = logging.getLogger(__name__)
logger.addHandler(handler)

class LessSelector:
    def __init__(self, 
                 dataset, 
                 eval_dataset,
                 accelerator, 
                 data_collator,
                 save_dir: str = "/data/malu/DataFlex-Preview/saves/less/",
                 gradient_type: str = "adam",
                 proj_dim: int = 4096,
                 seed: int = 42):
        """
        初始化 LessSelector.

        Args:
            dataset: 完整的数据集.
            accelerator: Hugging Face Accelerate 的 accelerator 实例.
            data_collator: 用于 DataLoader 的 data collator.
            save_dir (str): 用于保存计算出的梯度投影的目录.
            gradient_type (str, optional): 梯度类型，可选 "adam", "sgd", "sign". 默认为 "adam".
            proj_dim (int, optional): 梯度投影的目标维度. 默认为 4096.
            seed (int, optional): 随机种子. 默认为 42.
        """
        self.dataset = dataset
        self.eval_dataset = eval_dataset
        self.accelerator = accelerator
        self.data_collator = data_collator
        self.save_dir = save_dir
        self.gradient_type = gradient_type
        self.proj_dim = proj_dim
        self.seed = seed
        
        self.device = self.accelerator.device
        self.dtype = torch.float16 # 使用 float16 以节省内存

        os.makedirs(self.save_dir, exist_ok=True)
        logger.info(f"LessSelector initialized. Projected gradients will be saved in {self.save_dir}")

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

    def _get_number_of_params(self, model) -> int:
        """计算模型中需要梯度的参数数量。"""
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if self.accelerator.is_main_process:
            logger.info(f"Total number of parameters that require gradients: {num_params}")
        return num_params

    def _prepare_optimizer_state(self, model, optimizer_state: Dict) -> (torch.Tensor, torch.Tensor):
        """从优化器状态字典中准备 Adam 的一阶和二阶矩估计。"""
        # DDP 下，optimizer_state 的 key 是 full_param
        avg_list, avg_sq_list = [], []
        for param in model.parameters():
            if param.requires_grad:
                avg_list.append(optimizer_state[param]["exp_avg"].view(-1))
                avg_sq_list.append(optimizer_state[param]["exp_avg_sq"].view(-1))

        avg = torch.cat(avg_list).to(self.device)
        avg_sq = torch.cat(avg_sq_list).to(self.device)
        return avg, avg_sq

    def _obtain_gradients(self, model, batch, m: Optional[torch.Tensor] = None, v: Optional[torch.Tensor] = None) -> torch.Tensor:
        """根据指定的类型计算单个样本的梯度向量。"""
        # 在 no_sync 上下文中计算梯度，因为我们是逐样本处理，不需要 DDP 同步
        with self.accelerator.no_sync(model):
            loss = model(**batch).loss
            self.accelerator.backward(loss)

        vectorized_grads = torch.cat(
            [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
        )

        if self.gradient_type == "adam":
            if m is None or v is None:
                raise ValueError("Adam optimizer states (m, v) must be provided for 'adam' gradient type.")
            beta1, beta2, eps = 0.9, 0.999, 1e-08
            updated_avg = beta1 * m + (1 - beta1) * vectorized_grads
            updated_avg_sq = beta2 * v + (1 - beta2) * vectorized_grads ** 2
            final_grads = updated_avg / torch.sqrt(updated_avg_sq + eps)
        elif self.gradient_type == "sign":
            final_grads = torch.sign(vectorized_grads)
        else: # "sgd"
            final_grads = vectorized_grads
        
        model.zero_grad()
        return final_grads

    def _get_trak_projector(self):
        """获取 TRAK projector，优先使用 CUDA 版本。"""
        try:
            # 尝试导入和使用 CudaProjector
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
        """获取已保存的 chunk 的最大索引，用于断点续传。"""
        prefix = "grads"

        if not os.path.exists(save_dir):
            return -1
        files = [f for f in os.listdir(save_dir) if f.startswith(prefix) and f.endswith(".pt")]
        if not files:
            return -1
        indices = [int(f.split(".")[0].split("-")[1]) for f in files]
        return max(indices) if indices else -1

    def _collect_and_save_projected_gradients(self, model, save_dir, optimizer_state: Optional[Dict] = None):
        """
        核心函数：在所有 GPU 上计算梯度，然后在主进程上投影并分块保存。
        """
        if self.accelerator.is_main_process:
            logger.info(f"Starting gradient collection for {len(self.dataset)} samples.")
        
        # 1) 初始化 Projector
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
            logger.info("Adam optimizer states prepared.")

        # 3) 构造 DataLoader
        # 使用 batch_size=1 确保我们为每个样本计算独立的梯度
        dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            collate_fn=self.data_collator,
        )
        dataloader = self.accelerator.prepare(dataloader)

        # 4) 设置保存和投影的间隔
        project_interval = 8  # 每处理32个样本投影一次
        save_interval = 320    # 每处理320个样本（10次投影）保存一次

        # 5) 断点续传
        max_index = self._get_max_saved_index(save_dir=save_dir)
        start_count = max_index + 1
        if self.accelerator.is_main_process and start_count > 1:
            logger.info(f"Resuming from sample index {start_count}.")

        # 6) 循环计算、收集、投影和保存
        local_grads_to_project = []
        projected_grads_to_save = []
        
        # 使用 enumerate(dataloader, 1) 使 count 从 1 开始
        for count, batch in enumerate(tqdm(
            dataloader,
            desc="[LessSelector] Calculating Gradients",
            disable=not self.accelerator.is_main_process,
            dynamic_ncols=True,
            initial=start_count
        ), 1):
            if count < start_count:
                continue

            # 每个进程计算自己分到的样本的梯度
            vectorized_grads = self._obtain_gradients(model, batch, m, v)
            
            # 收集所有进程的梯度到主进程
            # gather 操作会阻塞，直到所有进程都到达
            gathered_grads_list = self.accelerator.gather(vectorized_grads.unsqueeze(0))

            if self.accelerator.is_main_process:
                # gathered_grads_list 的形状是 (num_processes, 1, grad_dim)
                # 我们将其合并到待投影列表中
                local_grads_to_project.append(gathered_grads_list.squeeze(1))

                # 达到投影间隔
                if count % project_interval == 0 or count == len(dataloader):
                    if local_grads_to_project:
                        grads_tensor = torch.cat(local_grads_to_project).to(self.dtype)
                        projected = projector.project(grads_tensor, model_id=0)
                        projected_grads_to_save.append(projected.cpu())
                        local_grads_to_project = []

                # 达到保存间隔
                if count % save_interval == 0 or count == len(dataloader):
                    if projected_grads_to_save:
                        save_tensor = torch.cat(projected_grads_to_save)
                        save_path = os.path.join(save_dir, f"grads-{count}.pt")
                        torch.save(save_tensor, save_path)
                        logger.info(f"Saved {save_tensor.shape[0]} projected gradients to {save_path}")
                        projected_grads_to_save = []
        
        # 等待所有进程完成
        self.accelerator.wait_for_everyone()

    def _merge_and_normalize_info(self, save_dir):
        """合并所有分块的梯度投影文件并进行归一化。"""
        if self.accelerator.is_main_process:
            logger.info(f"Merging and normalizing projected gradients from {save_dir}")
            files = [f for f in os.listdir(save_dir) if f.startswith("grads-") and f.endswith(".pt")]
            if not files:
                logger.warning("No gradient files found to merge.")
                return
            
            # 按数字顺序排序
            files.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
            
            merged_data = []
            for file in files:
                data = torch.load(os.path.join(save_dir, file))
                # 归一化每个样本的投影梯度向量
                normalized_data = normalize(data.to(torch.float32), dim=1)
                merged_data.append(normalized_data)
            
            merged_data = torch.cat(merged_data, dim=0)
            
            output_file = os.path.join(save_dir, "all_projected_grads.pt")
            torch.save(merged_data, output_file)
            logger.info(f"Saved merged and normalized gradients (Shape: {merged_data.shape}) to {output_file}")

    def select(self, model, step_id: int, num_samples: int, optimizer_state: Optional[Dict] = None) -> List[int]:
        """
        选择得分最高的 num_samples 个样本。
        如果预计算的梯度不存在，则会先进行计算。

        Args:
            model: 用于计算梯度的模型.
            num_samples (int): 要选择的样本数量.
            optimizer_state (Optional[Dict]): Adam 优化器的状态字典，当 gradient_type='adam' 时必须提供.

        Returns:
            List[int]: 被选中的样本的索引列表.
        """
        now_train_save_dir = os.path.join(self.save_dir, "train", str(step_id))
        now_eval_save_dir = os.path.join(self.save_dir, "eval", str(step_id))
        
        self.step_id = step_id
        train_final_grads_path = os.path.join(now_train_save_dir, "all_projected_grads.pt")
        eval_final_grads_path = os.path.join(now_eval_save_dir, "all_projected_grads.pt")

        # 步骤 1: 检查是否存在最终的训练集的梯度文件，如果不存在则计算训练集对应的梯度
        if not os.path.exists(train_final_grads_path):
            os.makedirs(now_train_save_dir, exist_ok=True)
            self._collect_and_save_projected_gradients(model, now_train_save_dir, optimizer_state)
            self._merge_and_normalize_info(now_train_save_dir)
        
        # 确保所有进程都已完成文件写入
        self.accelerator.wait_for_everyone()

        # 步骤 2: 检查是否存在最终的验证集的梯度文件，如果不存在则计算验证集对应的梯度
        if not os.path.exists(eval_final_grads_path):
            os.makedirs(now_eval_save_dir, exist_ok=True)
            self._collect_and_save_projected_gradients(model, now_eval_save_dir, optimizer_state)
            self._merge_and_normalize_info(now_eval_save_dir)
        
        self.accelerator.wait_for_everyone()

        # 步骤 2: 主进程加载数据、计算分数并选择 top-k
        if self.accelerator.is_main_process:
            logger.info(f"Loading projected gradients from {train_final_grads_path}")
            train_projected_grads = torch.load(train_final_grads_path, map_location="cpu")

            logger.info(f"Loading projected gradients from {eval_final_grads_path}")
            eval_projected_grads = torch.load(eval_final_grads_path, map_location="cpu")

            # 计算每个train_grad对所有eval_grad的平均相似度
            train_eval_similarities = (train_projected_grads @ eval_projected_grads.T).mean(dim=1)
            topk = torch.topk(train_eval_similarities, k=num_samples, largest=True)
            selected_indices = topk.indices.tolist()

            logger.info(f"Selecting top {num_samples} samples from {len(train_eval_similarities)}.")
        else:
            selected_indices = None

        # 步骤 3: 将选择的索引广播到所有进程
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