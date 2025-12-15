from dataflex.core.registry import register_selector
from dataflex.utils.selector_io import load_cached_selection, save_selection
from .base_selector import Selector
from dataflex.utils.logging import logger
import torch
from torch.nn.functional import normalize
from typing import List, Dict, Optional, Any
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from trak.projectors import BasicProjector, CudaProjector, ProjectionType
import os
import glob 
import re
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None


class LocalVLLMRewardServing:
    """基于 vLLM 的本地奖励模型推理，支持多卡并行。"""

    def __init__(self,
                 hf_model_name_or_path: str,
                 hf_cache_dir: Optional[str] = None,
                 hf_local_dir: Optional[str] = None,
                 vllm_tensor_parallel_size: int = 1,
                 vllm_temperature: float = 0.0,
                 vllm_top_p: float = 0.9,
                 vllm_max_tokens: int = 512,
                 vllm_top_k: int = 40,
                 vllm_repetition_penalty: float = 1.0,
                 vllm_seed: Optional[int] = None,
                 vllm_max_model_len: Optional[int] = None,
                 vllm_gpu_memory_utilization: float = 0.9):
        if not hf_model_name_or_path:
            raise ValueError("'hf_model_name_or_path' is required for local_vllm reward backend.")

        self.hf_model_name_or_path = hf_model_name_or_path
        self.hf_cache_dir = hf_cache_dir
        self.hf_local_dir = hf_local_dir
        self.vllm_tensor_parallel_size = vllm_tensor_parallel_size
        self.vllm_temperature = vllm_temperature
        self.vllm_top_p = vllm_top_p
        self.vllm_max_tokens = vllm_max_tokens
        self.vllm_top_k = vllm_top_k
        self.vllm_repetition_penalty = vllm_repetition_penalty
        self.vllm_seed = vllm_seed
        self.vllm_max_model_len = vllm_max_model_len
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization

        self._sampling_kwargs = dict(
            temperature=self.vllm_temperature,
            top_p=self.vllm_top_p,
            max_tokens=self.vllm_max_tokens,
            top_k=self.vllm_top_k,
            repetition_penalty=self.vllm_repetition_penalty,
            seed=self.vllm_seed,
        )
        self._llm = None
        self._base_sampling_params = None

    def _ensure_backend(self):
        if self._llm is not None:
            return

        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:  
            raise ImportError("Please install vLLM first, e.g., pip install vllm.\n") from exc

        model_path = self.hf_model_name_or_path
        if not os.path.exists(model_path):
            if snapshot_download is None:
                raise ImportError("Huggingface_hub is not installed, unable to download remote models.")
            model_path = snapshot_download(
                repo_id=self.hf_model_name_or_path,
                cache_dir=self.hf_cache_dir,
                local_dir=self.hf_local_dir,
            )

        os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')

        self._llm = LLM(
            model=model_path,
            tensor_parallel_size=self.vllm_tensor_parallel_size,
            max_model_len=self.vllm_max_model_len,
            gpu_memory_utilization=self.vllm_gpu_memory_utilization,
        )
        self._base_sampling_params = SamplingParams(**self._sampling_kwargs)

    def generate_from_input(self,
                             user_inputs: List[str],
                             system_prompt: str) -> List[str]:
        self._ensure_backend()
        sampling_params = self._base_sampling_params
        prompts = []
        system_prefix = (system_prompt.strip() + "\n\n") if system_prompt else ""
        for prompt in user_inputs:
            prompt = prompt or ""
            prompts.append(system_prefix + prompt)

        outputs = self._llm.generate(prompts, sampling_params)
        return [result.outputs[0].text if result.outputs else "" for result in outputs]


class APIRewardServing:
    """通过 HTTP 接口调用远程奖励模型。"""

    def __init__(self,
                 api_url: str,
                 api_key: str,
                 model_name: str = "gpt-4o",
                 max_workers: int = 8,
                 max_retries: int = 5,
                 temperature: float = 0.0,
                 request_timeout: int = 60):
        if not api_url:
            raise ValueError("'api_url' is required for api reward backend.")
        if not api_key:
            raise ValueError("'api_key' is required for api reward backend.")

        if api_key in os.environ:
            self.api_key = os.environ[api_key]
        else:
            self.api_key = api_key

        self.api_url = api_url
        self.model_name = model_name
        self.max_workers = max(1, max_workers)
        self.max_retries = max(1, max_retries)
        self.temperature = temperature
        self.request_timeout = request_timeout

    def _post(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=self.request_timeout,
                )
            except requests.RequestException as exc:
                logger.error(f"API reward request error: {exc}")
                time.sleep(2 ** attempt)
                continue

            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError:
                    logger.error("Failed to decode JSON from API response")
                    return None

            logger.error(f"API reward request failed with status {response.status_code}: {response.text}")
            time.sleep(2 ** attempt)

        return None

    def _format_payload(self, prompt: str, system_prompt: str):
        messages = [
            {"role": "system", "content": system_prompt or "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ]

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
        }
        return payload

    def _parse_response(self, response: Dict[str, Any]) -> str:
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return ""

    def generate_from_input(self,
                             user_inputs: List[str],
                             system_prompt: str) -> List[str]:
        prompts = user_inputs or []
        results = ["" for _ in prompts]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {}
            for idx, prompt in enumerate(prompts):
                payload = self._format_payload(prompt or "", system_prompt)
                future = executor.submit(self._post, payload)
                future_to_idx[future] = idx

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                response = future.result()
                if response:
                    results[idx] = self._parse_response(response)

        return results

class IndexedDataset(Dataset):
    """索引包装，确保样本索引在缓存时保持一致。"""
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        return index, self.original_dataset[index]

DEFAULT_GENERATION_PROMPT = (
    "Below is an instruction that describes a task, paired with an optional input that provides further context. "
    "Write a response that appropriately completes the request.\n"
    "### Instruction:\n{instruction}\n"
    "### Input:\n{input}\n"
    "### Response:"
)

# 默认奖励模型系统提示
DEFAULT_REWARD_SYSTEM_PROMPT = "You are a reward model. Return only a floating point score between 0 and 1."

DEFAULT_REWARD_PROMPT_WITH_REF = (
    "You are a reward model. Your task is to evaluate an AI's response based on a given instruction, input, and a reference answer. "
    "Provide a single score between 0.0 (worst) and 1.0 (best).\n\n"
    "A score of 1.0 means the **Candidate** response is correct, helpful, and perfectly aligned with the **Reference** answer.\n"
    "A score of 0.0 means the response is incorrect, unhelpful, or completely misaligned.\n\n"
    "### Instruction:\n{instruction}\n"
    "### Input:\n{input}\n"
    "### Reference:\n{reference}\n"
    "### Candidate:\n{prediction}\n"
    "Score:"
)

DEFAULT_REWARD_PROMPT_NO_REF = (
    "You are a reward model. Your task is to evaluate an AI's response based on a given instruction and input. "
    "Provide a single score between 0.0 (worst) and 1.0 (best).\n\n"
    "A score of 1.0 means the **Candidate** response is correct, helpful, and completely safe.\n"
    "A score of 0.0 means the response is incorrect, unhelpful, or unsafe.\n\n"
    "### Instruction:\n{instruction}\n"
    "### Input:\n{input}\n"
    "### Candidate:\n{prediction}\n"
    "Score:"
)

@register_selector('nice')
class NICESelector(Selector):
    def __init__(self,
                 dataset,
                 eval_dataset,
                 accelerator,
                 data_collator,
                 cache_dir,
                 reward_model_backend: str = "local_vllm",
                 reward_backend_params: Optional[Dict[str, Any]] = None,
                 gradient_type: str = "adam",
                 proj_dim: int = 8192,
                 save_interval: int = 16,
                 seed: int = 42,
                 mc_samples: int = 4,
                 max_new_tokens: int = 512,
                 generation_temperature: float = 0.7,
                 prompt_template: Optional[str] = None,
                 reward_prompt_with_ref: Optional[str] = None,
                 reward_prompt_without_ref: Optional[str] = None,
                 max_prompt_length: int = 4096):
        """初始化 NICE 选择器，加载策略与奖励模型。"""
        super().__init__(dataset, accelerator, data_collator, cache_dir)

        self.eval_dataset = eval_dataset
        self.gradient_type = gradient_type
        self.proj_dim = proj_dim
        self.save_interval = save_interval
        self.seed = seed
        self.mc_samples = mc_samples
        self.max_new_tokens = max_new_tokens
        self.generation_temperature = generation_temperature
        self.prompt_template = prompt_template or DEFAULT_GENERATION_PROMPT
        self.reward_prompt_with_ref = reward_prompt_with_ref or DEFAULT_REWARD_PROMPT_WITH_REF
        self.reward_prompt_without_ref = reward_prompt_without_ref or DEFAULT_REWARD_PROMPT_NO_REF
        self.max_prompt_length = max_prompt_length

        self.device = self.accelerator.device
        self.dtype = torch.float16

        self.reward_model_backend = (reward_model_backend or "local_vllm").lower()
        self.reward_backend_params = dict(reward_backend_params or {})
        self.reward_system_prompt = DEFAULT_REWARD_SYSTEM_PROMPT

        self.reward_serving = None

        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info("NICESelector initialized, setting up reward backend...")

        self._setup_reward_backend()

    def _setup_reward_backend(self):
        """根据配置加载奖励模型或服务。"""
        backend = self.reward_model_backend
        raw_params = self.reward_backend_params or {}
        if isinstance(raw_params, dict) and backend in raw_params and isinstance(raw_params[backend], dict):
            params = dict(raw_params[backend])
        else:
            params = dict(raw_params)
        if backend == "local_vllm":
            model_path = params.get("hf_model_name_or_path")
            if not model_path and isinstance(raw_params, dict):
                model_path = raw_params.get("hf_model_name_or_path")
            if not model_path:
                raise ValueError(
                    "'hf_model_name_or_path' must be provided for 'local_vllm' reward backend."
                )
            params["hf_model_name_or_path"] = model_path
            self.reward_serving = LocalVLLMRewardServing(**params)
            logger.info("Initialized local vLLM reward backend")
            return

        if backend == "api":
            if "api_url" not in params or "api_key" not in params:
                raise ValueError("'api_url' and 'api_key' must be provided in reward_backend_params for 'api' backend.")
            self.reward_serving = APIRewardServing(**params)
            logger.info("Initialized API reward backend")
            return

        raise ValueError(f"Unknown reward backend: {self.reward_model_backend}")

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

    def _obtain_gradients(self, model, batch, gradient_type: str, *, m: Optional[torch.Tensor] = None, v: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        获取已保存的最大样本索引，方便断点续传。
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

    def _normalize_example(self, example: Dict) -> Dict:
        """统一 Alpaca 与 ShareGPT 格式为包含 instruction/input/output 的结构。"""
        if any(k in example for k in ("instruction", "input", "output")):
            instruction = str(example.get("instruction") or "").strip()
            input_text = str(example.get("input") or "").strip()
            output = str(example.get("output") or "").strip()
            return {"instruction": instruction, "input": input_text, "output": output}

        conversations = example.get("conversations") or example.get("messages")
        if isinstance(conversations, list):
            question_parts = []
            answer_parts = []
            for message in conversations:
                if not isinstance(message, dict):
                    continue
                role = message.get("from") or message.get("role")
                if role is None:
                    continue
                role = str(role).lower()
                content = message.get("value") or message.get("content") or ""
                content = str(content).strip()
                if not content:
                    continue
                if role in {"human", "user"}:
                    question_parts.append(content)
                elif role in {"gpt", "assistant"}:
                    answer_parts.append(content)
                elif role == "system":
                    question_parts.append(content)

            question_text = "\n\n".join(question_parts).strip()
            answer_text = "\n\n".join(answer_parts).strip()

            if question_text:
                if len(question_parts) > 1:
                    instruction = question_parts[0]
                    input_text = question_text
                else:
                    instruction = question_text
                    input_text = ""
            else:
                instruction = ""
                input_text = ""

            return {"instruction": instruction, "input": input_text, "output": answer_text}

        # fallback: treat raw text as instruction
        raw_text = str(example).strip()
        return {"instruction": raw_text, "input": "", "output": ""}

    def _format_generation_prompt(self, example: Dict) -> str:
        """构造策略模型提示词"""
        example = self._normalize_example(example)
        instruction = example.get("instruction", "").strip()
        input_text = example.get("input", "").strip() or "No additional input."
        prompt = self.prompt_template.format(
            instruction=instruction,
            input=input_text,
        )
        return prompt

    def _format_reward_prompt(self, example: Dict, prediction: str) -> str:
        """根据是否存在参考答案动态生成奖励模型提示词。"""
        example = self._normalize_example(example)
        instruction = example.get("instruction", "").strip()
        input_text = example.get("input", "").strip() or "No additional input."
        reference = example.get("output", "").strip()
        if reference:
            prompt = self.reward_prompt_with_ref.format(
                instruction=instruction,
                input=input_text,
                reference=reference,
                prediction=prediction.strip() or "No response.",
            )
        else:
            prompt = self.reward_prompt_without_ref.format(
                instruction=instruction,
                input=input_text,
                prediction=prediction.strip() or "No response.",
            )
        return prompt

    def _generate_response(self,
                           model,
                           tokenizer,
                           example: Dict,
                           sample_seed: Optional[int] = None) -> Dict:
        """调用策略模型生成回答并保留生成细节，支持蒙特卡洛采样。"""
        prompt = self._format_generation_prompt(example)
        inputs = tokenizer(
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
            was_training = model.training
            model.eval()
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.generation_temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            if was_training:
                model.train()

        if sample_seed is not None:
            torch.random.set_rng_state(cpu_state)
            if torch.cuda.is_available() and cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)

        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[:, prompt_length:]
        generated_text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

        attention_mask = generated_ids.ne(tokenizer.pad_token_id).long()

        return {
            "prompt": prompt,
            "input_ids": generated_ids,
            "attention_mask": attention_mask,
            "prompt_length": prompt_length,
            "prediction": generated_text,
        }

    def _score_with_serving(self, prompt: str) -> float:
        """使用 DataFlow 提供的服务型后端打分。"""
        if self.reward_serving is None:
            return 0.0
        try:
            responses = self.reward_serving.generate_from_input(
                [prompt],
                system_prompt=self.reward_system_prompt,
            )
        except Exception as exc:
            if self.accelerator.is_main_process:
                logger.error(f"Reward backend generation failed: {exc}")
            return 0.0

        text = ""
        if responses:
            text = responses[0] or ""
        return self._extract_score_from_text(text)

    def _extract_score_from_text(self, text: str) -> float:
        """从文本中提取 [0, 1] 范围内的分数。"""
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if match:
            try:
                value = float(match.group())
            except ValueError:
                value = 0.0
            return float(min(max(value, 0.0), 1.0))

        if self.accelerator.is_main_process:
            logger.warning("Failed to parse reward score from backend output: %s", text)
        return 0.0

    def _compute_reward(self, example: Dict, prediction: str) -> float:
        """奖励模型既支持带参考答案也支持无参考答案的打分。"""
        reward_prompt = self._format_reward_prompt(example, prediction)
        reward = self._score_with_serving(reward_prompt)
        return reward

    def _compute_rl_gradient(self,
                             model,
                             sample_info: Dict,
                             reward: float,
                             grad_dim: int) -> torch.Tensor:
        """根据策略梯度公式计算验证集梯度，直接回传序列对数似然。"""
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

    # 核心逻辑
    def _collect_and_save_projected_gradients(self,
                                              model,
                                              save_dir,
                                              dataset_to_use,
                                              optimizer_state: Optional[Dict] = None,
                                              rl_mode: bool = False,
                                              tokenizer=None):
        """统一采集梯度、执行投影并保存，rl_mode 控制是否启用蒙特卡洛采样。"""
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
            batch_size=1, # 仍然是逐样本计算
            shuffle=False,
            num_workers=2,
            collate_fn=indexed_collator_wrapper,
        )
        dataloader = self.accelerator.prepare(dataloader)

        # 4) 设置保存间隔
        save_interval = self.save_interval

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
        model_device = next(model.parameters()).device

        # enumerate(..., 1) 使 batch_idx 从 1 开始
        generation_model = None
        if rl_mode:
            generation_model = self.accelerator.unwrap_model(model)

        for batch_idx, data in enumerate(tqdm(
            dataloader,
            desc=f"[Process {self.accelerator.process_index}] Calculating Gradients",
            disable=not self.accelerator.is_local_main_process, # 主进程打印进度条
            dynamic_ncols=True,
            position=self.accelerator.process_index,
        ), 1):
            indices = data['indices']
            
            # 断点续传逻辑，这里的 'count' 应该是全局样本索引
            # DistributedSampler 会自动处理分片，我们只需跳过批次即可
            # 注意: 简单的跳过可能不精确，但对于重启来说是可接受的
            # dataloader.sampler.set_epoch(batch_idx) # 如果需要精确重启，可能需要更复杂的sampler状态管理
            # 这里我们简化处理，假设从头开始或不跳过
  
            if rl_mode:
                example = data['examples'][0]
                if tokenizer is None:
                    raise ValueError("Tokenizer must be provided for NICE selector in RL mode.")
                mc_gradients = []
                num_mc = max(1, self.mc_samples)
                base_seed = self.seed + indices[0].item() * 997
                for mc_id in range(num_mc):
                    sample_seed = base_seed + mc_id
                    sample_info = self._generate_response(generation_model, tokenizer, example, sample_seed=sample_seed)
                    reward = self._compute_reward(example, sample_info['prediction'])
                    grad_vector = self._compute_rl_gradient(model, sample_info, reward, num_params)
                    mc_gradients.append(grad_vector)
                if mc_gradients:
                    vectorized_grads = torch.stack(mc_gradients, dim=0).mean(dim=0)
                else:
                    vectorized_grads = torch.zeros(num_params, device=model_device)
            else:
                batch = data['batch']
                vectorized_grads = self._obtain_gradients(
                    model,
                    batch,
                    gradient_type=self.gradient_type,
                    m=m,
                    v=v,
                )

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


    # 合并
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
        
        optimizer_state = kwargs.get('optimizer_state', None)
        tokenizer = kwargs.get('tokenizer', None)
        if tokenizer is None and hasattr(self.data_collator, 'tokenizer'):
            tokenizer = getattr(self.data_collator, 'tokenizer')
        if tokenizer is not None and tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided to the NICE selector to run in RL mode.")

        # 步骤 1: 计算训练集梯度
        if not os.path.exists(train_final_grads_path):
            os.makedirs(now_train_save_dir, exist_ok=True)
            self._collect_and_save_projected_gradients(model, now_train_save_dir, self.dataset, optimizer_state, rl_mode=False)
            self._merge_and_normalize_info(now_train_save_dir, len(self.dataset))
        
        self.accelerator.wait_for_everyone()

        # 步骤 2: 计算验证集梯度
        if not os.path.exists(eval_final_grads_path):
            os.makedirs(now_eval_save_dir, exist_ok=True)
            self._collect_and_save_projected_gradients(
                model,
                now_eval_save_dir,
                self.eval_dataset,
                optimizer_state,
                rl_mode=True,
                tokenizer=tokenizer,
            )    
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