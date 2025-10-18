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
import glob # 

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch.nn.functional as F

# NEW: IndexedDataset Wrapper
class IndexedDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        return index, self.original_dataset[index]


DEFAULT_POLICY_PROMPT = (
    "以下是一个指令与可选输入，请根据它们生成自然语言回答。\n"
    "### 指令:\n{instruction}\n"
    "### 输入:\n{input}\n"
    "### 回答:"
)

DEFAULT_REWARD_PROMPT = (
    "你是一名奖励模型，请阅读指令与模型回答并给出0到1之间的得分。\n"
    "请只输出一个数字。\n"
    "### 指令:\n{instruction}\n"
    "### 输入:\n{input}\n"
    "### 模型回答:\n{prediction}\n"
    "请给出分值:"
)


@register_selector('nice_rm')
class NICERMSelector(Selector):
    def __init__(self, 
                 dataset,
                 eval_dataset,
                 accelerator,
                 data_collator,
                 cache_dir,
                 policy_model_path: str,
                 reward_model_path: str,
                 gradient_type: str = "adam",
                 proj_dim: int = 8192,
                 seed: int = 42,
                 mc_samples: int = 4,
                 max_new_tokens: int = 512,
                 generation_temperature: float = 0.7,
                 prompt_template: Optional[str] = None,
                 reward_prompt_template: Optional[str] = None,
                 max_prompt_length: int = 4096):
        """初始化 NICERMSelector，加载所需的本地模型并保存超参数。"""
        super().__init__(dataset, accelerator, data_collator, cache_dir)

        self.eval_dataset = eval_dataset
        self.gradient_type = gradient_type
        self.proj_dim = proj_dim
        self.seed = seed
        self.mc_samples = mc_samples
        self.max_new_tokens = max_new_tokens
        self.generation_temperature = generation_temperature
        self.prompt_template = prompt_template or DEFAULT_POLICY_PROMPT
        self.reward_prompt_template = reward_prompt_template or DEFAULT_REWARD_PROMPT
        self.max_prompt_length = max_prompt_length
        
        self.device = self.accelerator.device
        self.dtype = torch.float16

        self.policy_model_path = policy_model_path
        self.reward_model_path = reward_model_path

        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"NICERMSelector initialized. Projected gradients will be saved in {self.cache_dir}")

        self._load_local_models()
        logger.info(f"NICERMSelector The policy model has been loaded.: {policy_model_path}")
        logger.info(f"NICERMSelector The reward model has been loaded.: {reward_model_path}")

    def _load_local_models(self):
        """从本地路径加载策略模型与评审模型。"""
        self.policy_tokenizer = AutoTokenizer.from_pretrained(self.policy_model_path, trust_remote_code=True)
        if self.policy_tokenizer.pad_token_id is None:
            self.policy_tokenizer.pad_token_id = self.policy_tokenizer.eos_token_id
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            self.policy_model_path,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.policy_model.eval()
        self.policy_model.requires_grad_(False)

        self.reward_tokenizer = AutoTokenizer.from_pretrained(self.reward_model_path, trust_remote_code=True)
        if self.reward_tokenizer.pad_token_id is None:
            self.reward_tokenizer.pad_token_id = self.reward_tokenizer.eos_token_id
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            self.reward_model_path,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.reward_model.eval()
        self.reward_model.requires_grad_(False)

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

    def _obtain_gradients(self, model, batch, m: Optional[torch.Tensor] = None, v: Optional[torch.Tensor] = None) -> torch.Tensor:
        """根据指定的类型计算单个样本的梯度向量。"""
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
        """获取已经保存的最大样本索引，方便断点续传。"""
        prefix = "grads"
        if not os.path.exists(save_dir):
            return -1
        if not self.accelerator.is_main_process:
            return -1

        files = [f for f in os.listdir(save_dir) if f.startswith(prefix) and f.endswith(".pt")]
        if not files:
            return -1
        # 文件名格式: grads-{count}-rank{rank}.pt
        indices = [int(f.split('.')[0].split('-')[1]) for f in files]
        return max(indices) if indices else -1
    
    def _format_generation_prompt(self, example: Dict) -> str:
        """根据指令样本构造生成模型提示词。"""
        instruction = example.get("instruction", "").strip()
        input_text = example.get("input", "").strip() or "No additional input"
        prompt = self.prompt_template.format(
            instruction=instruction,
            input=input_text,
        )
        return prompt

    def _format_reward_prompt(self, example: Dict, prediction: str) -> str:
        """构造reward模型使用的提示词。"""
        instruction = example.get("instruction", "").strip()
        input_text = example.get("input", "").strip() or "No additional input"
        prompt = self.reward_prompt_template.format(
            instruction=instruction,
            input=input_text,
            prediction=prediction.strip() or "No Answer"
        )
        return prompt

    def _generate_response(self, example: Dict, sample_seed: Optional[int] = None) -> Dict:
        """利用策略模型进行多次蒙特卡洛采样，返回生成文本及相关张量。"""
        prompt = self._format_generation_prompt(example)
        inputs = self.policy_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_prompt_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        cpu_state = None
        cuda_state = None
        if sample_seed is not None:
            cpu_state = torch.random.get_rng_state()
            if torch.cuda.is_available():
                cuda_state = torch.cuda.get_rng_state_all()
            torch.manual_seed(sample_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sample_seed)

        with torch.no_grad():
            generated_ids = self.policy_model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.generation_temperature,
                do_sample=True,
                pad_token_id=self.policy_tokenizer.pad_token_id,
                eos_token_id=self.policy_tokenizer.eos_token_id,
            )

        if sample_seed is not None:
            torch.random.set_rng_state(cpu_state)
            if torch.cuda.is_available() and cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)

        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[:, prompt_length:]
        generated_text = self.policy_tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

        attention_mask = generated_ids.ne(self.policy_tokenizer.pad_token_id).long()

        return {
            "prompt": prompt,
            "input_ids": generated_ids,
            "attention_mask": attention_mask,
            "prompt_length": prompt_length,
            "prediction": generated_text,
        }

    def _score_with_classifier(self, model, tokenizer, prompt: str) -> float:
        """统一将reward模型的 logits 映射到 [0, 1] 分数"""
        if model is None or tokenizer is None:
            return 0.0
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_prompt_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        if logits.ndim == 1:
            score = torch.sigmoid(logits).mean().item()
        elif logits.shape[-1] == 1:
            score = torch.sigmoid(logits.squeeze(-1)).mean().item()
        else:
            probs = F.softmax(logits, dim=-1)
            score = probs[..., -1].mean().item()
        return float(score)

    def _compute_reward(self, example: Dict, prediction: str) -> float:
        """reward模型给出奖励得分"""
        reward_prompt = self._format_reward_prompt(example, prediction)
        reward = self._score_with_classifier(self.reward_model, self.reward_tokenizer, reward_prompt)
        return reward

    def _compute_rl_gradient(self,
                             model,
                             sample_info: Dict,
                             reward: float,
                             grad_dim: int) -> torch.Tensor:
        """根据蒙特卡洛策略梯度公式计算验证集梯度。"""
        model_device = next(model.parameters()).device
        if reward == 0.0:
            return torch.zeros(grad_dim, device=model_device)

        input_ids = sample_info["input_ids"].to(model_device)
        attention_mask = sample_info["attention_mask"].to(model_device)
        labels = input_ids.clone()
        labels[:, :sample_info["prompt_length"]] = -100
        token_count = (labels != -100).sum()
        if token_count.item() == 0:
            return torch.zeros(grad_dim, device=model_device)

        reward_tensor = torch.tensor(reward, dtype=torch.float32, device=model_device)

        with self.accelerator.no_sync(model):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            nll_loss = outputs.loss * token_count
            loss = nll_loss * reward_tensor
            self.accelerator.backward(loss)

        vectorized_grads = torch.cat(
            [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
        ).to(model_device)

        model.zero_grad()
        return vectorized_grads

    def _collect_and_save_projected_gradients(self,
                                              model,
                                              save_dir,
                                              dataset_to_use,
                                              optimizer_state: Optional[Dict] = None,
                                              rl_mode: bool = False):
        """收集梯度、执行投影并保存，支持 RL 验证模式。"""

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
                raise ValueError("When using the Adam optimizer, you must provide optimizer_state.")
            m, v = self._prepare_optimizer_state(model, optimizer_state)

        # 3) 构造 DataLoader，使用 IndexedDataset 来追踪样本的原始索引
        indexed_dataset = IndexedDataset(dataset_to_use)

        # 定义一个处理索引的 collator
        if rl_mode:
            def indexed_collator_wrapper(features):
                indices = [f[0] for f in features]
                original_data = [f[1] for f in features]
                return {'indices': torch.tensor(indices), 'examples': original_data}
        else:
            def indexed_collator_wrapper(features):
                indices = [f[0] for f in features]
                original_data = [f[1] for f in features]
                collated_batch = self.data_collator(original_data)
                return {'indices': torch.tensor(indices), 'batch': collated_batch}

        dataloader = DataLoader(
            indexed_dataset,
            batch_size=1,   # 仍然是逐样本计算
            shuffle=False,
            num_workers=2,
            collate_fn=indexed_collator_wrapper,
        )
        dataloader = self.accelerator.prepare(dataloader)

        # 4) 设置保存间隔，每个进程每处理save_interval个样本就映射并保存一次
        save_interval = 64

        # 5) 断点续传
        max_index = self._get_max_saved_index(save_dir=save_dir)
        start_count = max_index + 1
        if self.accelerator.is_main_process and start_count > 1:
            logger.info(f"NICERMSelector: Resuming from sample index {start_count}.")

        # 等待主进程完成检查
        self.accelerator.wait_for_everyone()

        # 6) 循环计算、投影和保存 (在每个进程上独立进行)
        local_grads_to_project = []
        local_indices_to_project = []

        total_samples_in_loader = len(dataloader)
        # enumerate(..., 1) 使 batch_idx 从 1 开始
        for batch_idx, data in enumerate(tqdm(
            dataloader,
            desc=f"[Process {self.accelerator.process_index}] Collecting Gradients",
            disable=not self.accelerator.is_local_main_process,   # 主进程打印进度条
            dynamic_ncols=True,
            position=self.accelerator.process_index,
        ), 1):
            
            # 断点续传逻辑，这里的 'count' 应该是全局样本索引
            # DistributedSampler 会自动处理分片，我们只需跳过批次即可
            # 注意: 简单的跳过可能不精确，但对于重启来说是可接受的
            # dataloader.sampler.set_epoch(batch_idx) # 如果需要精确重启，可能需要更复杂的sampler状态管理
            # 这里我们简化处理，假设从头开始或不跳过

            indices = data['indices']

            if rl_mode:
                example = data['examples'][0]
                mc_gradients = []
                num_mc = max(1, self.mc_samples)
                base_seed = self.seed + indices[0].item() * 997
                for mc_id in range(num_mc):
                    sample_seed = base_seed + mc_id
                    sample_info = self._generate_response(example, sample_seed=sample_seed)
                    reward = self._compute_reward(example, sample_info['prediction'])
                    grad_vector = self._compute_rl_gradient(model, sample_info, reward, num_params)
                    mc_gradients.append(grad_vector)
                if mc_gradients:
                    vectorized_grads = torch.stack(mc_gradients, dim=0).mean(dim=0)
                else:
                    vectorized_grads = torch.zeros(num_params, device=self.device)
            else:
                batch = data['batch']
                vectorized_grads = self._obtain_gradients(model, batch, m, v)

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
        """主流程，与 LESS 类似但加入 RL 验证梯度。"""
        
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

        optimizer_state = kwargs.get('optimizer_state', None)

        # 步骤 1: 计算训练集梯度
        if not os.path.exists(train_final_grads_path):
            os.makedirs(now_train_save_dir, exist_ok=True)
            self._collect_and_save_projected_gradients(model, now_train_save_dir, self.dataset, optimizer_state, rl_mode=False)
            self._merge_and_normalize_info(now_train_save_dir, len(self.dataset))

        self.accelerator.wait_for_everyone()

         # 步骤 2: 计算验证集梯度
        if not os.path.exists(eval_final_grads_path):
            os.makedirs(now_eval_save_dir, exist_ok=True)
            self._collect_and_save_projected_gradients(model, now_eval_save_dir, self.eval_dataset, optimizer_state, rl_mode=True)
            self._merge_and_normalize_info(now_eval_save_dir, len(self.eval_dataset))

        self.accelerator.wait_for_everyone()

        # 步骤 3: 主进程加载、计算分数并选择 top-k
        if self.accelerator.is_main_process:
            logger.info(f"Loading Training Data projected gradients from {train_final_grads_path}")
            train_projected_grads = torch.load(train_final_grads_path, map_location="cpu")

            logger.info(f"Loading Valid Data projected gradients from {eval_final_grads_path}")
            eval_projected_grads = torch.load(eval_final_grads_path, map_location="cpu")

            train_eval_similarities = (train_projected_grads @ eval_projected_grads.T).mean(dim=1)
            topk = torch.topk(train_eval_similarities, k=num_samples, largest=True)
            selected_indices = topk.indices.tolist()

            logger.info(f"Selecting top {num_samples} samples from {len(train_eval_similarities)} .")

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