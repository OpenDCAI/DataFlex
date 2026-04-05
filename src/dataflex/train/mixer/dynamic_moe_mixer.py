import json
import os
import time
import torch
import torch.distributed as dist
import numpy as np

from dataflex.core.registry import register_mixer
from dataflex.utils.logging import logger
from .base_mixer import Mixer

from transformers.trainer_pt_utils import nested_detach


@register_mixer("dynamic_moe")
class DynamicMoEMixer(Mixer):
    def __init__(self, mixture_manager, eta: float = 10.0, c: float = 0.05,
                 collect_steps: int = 10, eval_samples: int = 1000,
                 output_dir: str = None, accelerator=None):
        """
        Dynamic MoE Mixer implementing Algorithm 1 from the paper.

        Args:
            mixture_manager: MixedProportionManager instance.
            eta (float): Update step size (learning rate for weights).
            c (float): Smoothing parameter to prevent extreme weights.
            collect_steps (int): Number of batches to sample per domain to estimate gate loads.
            eval_samples (int): Number of samples per domain reserved for gate load evaluation.
                Paper Section 5.1: "we sample 20K instances for training, and 1K instances
                for gate load evaluation in the sampling weight adjustment."
                These samples are split from the training set and excluded from training
                to ensure unbiased gate load estimation.
        """
        super().__init__(mixture_manager)
        self.eta = eta
        self.c = c
        self.collect_steps = collect_steps
        self.output_dir = output_dir
        self.accelerator = accelerator

        # Initialize weights uniformly or based on manager's current state
        # Algorithm 1 requires w_{t-1}, initialize it uniformly as per paper
        self.current_weights = np.ones(len(self.mixture_manager.names)) / len(self.mixture_manager.names)
        self._gate_load_seed = 42

        # ── Use independent eval datasets for gate load evaluation ──────────
        # Paper Section 5.1: "1K instances for gate load evaluation in the
        # sampling weight adjustment."
        #
        # Priority:
        #   1. Pre-loaded independent eval datasets (from mixer_eval_dataset config)
        #      → Official MoE-SFT dev/ split, loaded in loader.py
        #   2. Fallback: split eval_samples from each training domain
        self.eval_datasets = {}

        pre_loaded = getattr(self.mixture_manager, 'mixer_eval_datasets', None)
        if pre_loaded:
            # Use official independent eval datasets (no data leakage)
            for name in self.mixture_manager.names:
                if name in pre_loaded:
                    self.eval_datasets[name] = pre_loaded[name]
                    logger.info(
                        f"[DynamicMoEMixer] Domain '{name}': using independent eval dataset "
                        f"({len(pre_loaded[name])} samples)"
                    )
                else:
                    # Fallback for domains without independent eval
                    dataset = self.mixture_manager.sources[name]
                    n = len(dataset)
                    n_eval = min(eval_samples, n // 2)
                    if n_eval <= 0:
                        logger.warning(f"[DynamicMoEMixer] Domain '{name}' has {n} samples, too few to split eval set.")
                        self.eval_datasets[name] = dataset
                        continue
                    eval_rng = np.random.RandomState(seed=0)
                    all_indices = eval_rng.permutation(n)
                    eval_indices = sorted(all_indices[:n_eval].tolist())
                    train_indices = sorted(all_indices[n_eval:].tolist())
                    self.eval_datasets[name] = dataset.select(eval_indices)
                    self.mixture_manager.sources[name] = dataset.select(train_indices)
                    logger.info(
                        f"[DynamicMoEMixer] Domain '{name}': no independent eval, "
                        f"split {n_eval} from training (train: {len(train_indices)})"
                    )
            logger.info("[DynamicMoEMixer] Using independent eval datasets for gate load evaluation.")
        else:
            # Fallback: split eval sets from training sets (Paper Section 5.1)
            eval_rng = np.random.RandomState(seed=0)  # fixed seed for reproducible split
            for name in self.mixture_manager.names:
                dataset = self.mixture_manager.sources[name]
                n = len(dataset)
                n_eval = min(eval_samples, n // 2)  # never take more than half

                if n_eval <= 0:
                    logger.warning(f"[DynamicMoEMixer] Domain '{name}' has {n} samples, too few to split eval set.")
                    self.eval_datasets[name] = dataset
                    continue

                # Shuffle indices with fixed seed, take first n_eval as eval
                all_indices = eval_rng.permutation(n)
                eval_indices = sorted(all_indices[:n_eval].tolist())
                train_indices = sorted(all_indices[n_eval:].tolist())

                self.eval_datasets[name] = dataset.select(eval_indices)
                self.mixture_manager.sources[name] = dataset.select(train_indices)

                logger.info(
                    f"[DynamicMoEMixer] Domain '{name}': split {n_eval} eval samples "
                    f"(train: {len(train_indices)}, eval: {n_eval})"
                )
            logger.info("[DynamicMoEMixer] No independent eval datasets found, split from training data.")

    def _collect_gate_loads(self, model, data_collator, batch_size=4):
        """
        Collects aggregated gate loads for each domain by running inference on a few batches.

        In distributed training (DeepSpeed ZeRO-2/3), the unwrapped model is used for
        inference to avoid triggering NCCL collective operations that could cause deadlocks
        when different ranks have inconsistent forward pass counts.

        Returns:
            np.ndarray: Matrix of shape [num_domains, num_experts] containing normalized gate loads.
        """
        torch.cuda.empty_cache()

        # Unwrap the model to avoid DeepSpeed engine triggering NCCL collectives
        if hasattr(model, "module"):
            real_model = model.module
        else:
            real_model = model

        real_model.eval()

        device = next(real_model.parameters()).device

        domain_names = self.mixture_manager.names
        raw_loads_list = []

        num_experts = getattr(real_model.config, "num_experts", None)

        # Use a fixed seed that is the same across all ranks so that all ranks
        # sample exactly the same indices and execute the same number of forward passes.
        rng = np.random.RandomState(self._gate_load_seed)
        self._gate_load_seed += 1

        with torch.no_grad():
            for name in domain_names:
                # Use independent eval dataset instead of training set (Paper Section 5.1)
                dataset = self.eval_datasets[name]

                domain_load_sum = None

                if len(dataset) == 0:
                    if num_experts is None:
                        num_experts = 8
                    raw_loads_list.append(np.ones(num_experts) / num_experts)
                    continue

                # Randomly sample indices for estimation (num_samples = collect_steps * batch_size)
                num_samples = min(len(dataset), self.collect_steps * batch_size)
                if num_samples == 0:
                    if num_experts is None:
                        num_experts = 8
                    raw_loads_list.append(np.ones(num_experts) / num_experts)
                    continue

                indices = rng.choice(len(dataset), num_samples, replace=False)

                # Process in micro-batches to control memory
                for step in range(0, num_samples, batch_size):
                    batch_indices = indices[step : step + batch_size]
                    samples = [dataset[int(idx)] for idx in batch_indices]

                    batch = data_collator(samples)
                    batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

                    # Use the unwrapped model to avoid DeepSpeed collective operations
                    outputs = real_model(**batch)

                    # Extract gate_load
                    gate_load = None
                    if hasattr(outputs, "gate_load") and outputs.gate_load is not None:
                        gate_load = outputs.gate_load
                        gate_load = gate_load[-1]
                    if gate_load is not None:
                        gate_load = nested_detach(gate_load)

                        # Reduce to 1D [num_experts] for domain-level aggregation.
                        if gate_load.dim() > 1:
                            gate_load = gate_load.sum(dim=0)

                        # Move gate_load to CPU immediately to prevent GPU memory accumulation
                        gate_load_cpu = gate_load.float().cpu()

                        if domain_load_sum is None:
                            domain_load_sum = torch.zeros_like(gate_load_cpu)
                            if num_experts is None:
                                num_experts = gate_load_cpu.shape[0]

                        domain_load_sum += gate_load_cpu
                    else:
                        if step == 0:
                            logger.warning(f"[DynamicMoEMixer] Model output does not contain 'gate_load'. Using uniform load.")
                        if num_experts is None: num_experts = 8 # Fallback
                        if domain_load_sum is None:
                            domain_load_sum = torch.ones(num_experts, dtype=torch.float32)
                        else:
                            domain_load_sum += torch.ones(num_experts, dtype=torch.float32)

                    # Delete intermediate tensors explicitly to free GPU memory
                    del outputs
                    del batch
                    del gate_load

                # Cleanup cache after processing one domain to avoid OOM
                torch.cuda.empty_cache()

                # Normalize per domain: L1 normalization (aligned with official MoE-SFT)
                # Official code (_parse_name2load): _load = val / val.sum()
                # Converts gate load to probability distribution before similarity computation
                if domain_load_sum is not None and domain_load_sum.sum() > 0:
                    # domain_load_sum is already on CPU
                    load_np = domain_load_sum.numpy()
                    l1_sum = load_np.sum()
                    if l1_sum > 0:
                        normalized_load = load_np / l1_sum
                    else:
                        normalized_load = load_np
                else:
                    if num_experts is None: num_experts = 8
                    normalized_load = np.ones(num_experts) / num_experts

                raw_loads_list.append(normalized_load)

        real_model.train()

        result = np.stack(raw_loads_list)

        # Broadcast result from rank 0 to all ranks to ensure consistency
        if dist.is_initialized():
            result_tensor = torch.from_numpy(result).float().to(device)
            dist.broadcast(result_tensor, src=0)
            result = result_tensor.cpu().numpy()

        return result 

    def mix(self, model, step_id: int, **kwargs) -> np.ndarray:
        """
        Implements Algorithm 1: DynamicSampling
        """

        data_collator = kwargs.get('data_collator')
        if data_collator is None:
            logger.warning("[DynamicMoEMixer] data_collator not found in kwargs, cannot collect gate loads. Returning current weights.")
            return self.current_weights

        domain_names = self.mixture_manager.names

        # 2. Collect Normalized Gate Loads (O_hat)
        # Shape: [|D|, N] where |D| is num datasets, N is num experts
        O_hat = self._collect_gate_loads(model, data_collator)

        # ── Detailed diagnostic logging ──────────────────────────────────────
        np.set_printoptions(precision=6, suppress=True)
        logger.info(f"[DynamicMoEMixer] ═══ Step {step_id} Diagnostic ═══")
        logger.info(f"[DynamicMoEMixer] Domain order: {domain_names}")
        for i, name in enumerate(domain_names):
            logger.info(f"[DynamicMoEMixer] O_hat[{name}] (L1-normalized gate load): {O_hat[i]}")

        # 3. Calculate L2 distance across datasets (aligned with official MoE-SFT)
        l2_dist_matrix = np.linalg.norm(O_hat[:, np.newaxis] - O_hat, axis=2)  # [|D|, |D|]

        logger.info(f"[DynamicMoEMixer] L2 Distance Matrix:")
        for i, name_i in enumerate(domain_names):
            row_str = "  ".join(f"{name_j}={l2_dist_matrix[i,j]:.6f}" for j, name_j in enumerate(domain_names))
            logger.info(f"[DynamicMoEMixer]   {name_i}: {row_str}")

        # 4. Delta_i = mean_j(||load_i - load_j||_2) — includes self-distance=0 (official behavior)
        Delta = l2_dist_matrix.mean(axis=1)  # [|D|]

        delta_detail = ", ".join(f"{name}={Delta[i]:.6f}" for i, name in enumerate(domain_names))
        logger.info(f"[DynamicMoEMixer] Delta (mean L2 dist): {delta_detail}")

        # 5. Calculate updated sampling weights (Algorithm 1 lines 6-9)
        log_w_prev = np.log(self.current_weights + 1e-10)
        logits = log_w_prev + self.eta * Delta

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        alpha = exp_logits / exp_logits.sum()

        # Smoothing: w'_t <- (1 - c) * alpha + c / |D|
        num_datasets = len(self.current_weights)
        w_prime = (1 - self.c) * alpha + self.c / num_datasets

        # Normalize: w_t <- w'_t / sum(w'_t)
        new_weights = w_prime / w_prime.sum()

        old_w_str = ", ".join(f"{name}={self.current_weights[i]:.6f}" for i, name in enumerate(domain_names))
        new_w_str = ", ".join(f"{name}={new_weights[i]:.6f}" for i, name in enumerate(domain_names))
        logger.info(f"[DynamicMoEMixer] Old weights: {old_w_str}")
        logger.info(f"[DynamicMoEMixer] New weights: {new_w_str}")
        logger.info(f"[DynamicMoEMixer] ═══ End Step {step_id} ═══")

        self.current_weights = new_weights

        self.save_weights_to_jsonl(step_id, O_hat, Delta, alpha, new_weights)

        return new_weights

    def save_weights_to_jsonl(self, step_id: int, O_hat: np.ndarray,
                               Delta: np.ndarray, alpha: np.ndarray,
                               new_weights: np.ndarray):
        """update weights and save to jsonl file (only main process write)"""
        if self.accelerator is not None and not self.accelerator.is_main_process:
            return

        if self.output_dir is None:
            return

        try:
            os.makedirs(self.output_dir, exist_ok=True)
            weights_file = os.path.join(self.output_dir, "dynamic_moe_weights.jsonl")

            domain_names = self.mixture_manager.names
            log_entry = {
                "step": step_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "domain_names": domain_names,
                "gate_loads": {name: O_hat[i].tolist() for i, name in enumerate(domain_names)},
                "delta": {name: float(Delta[i]) for i, name in enumerate(domain_names)},
                "alpha": {name: float(alpha[i]) for i, name in enumerate(domain_names)},
                "new_weights": {name: float(new_weights[i]) for i, name in enumerate(domain_names)},
                "eta": self.eta,
                "c": self.c,
            }

            with open(weights_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            logger.info(f"[DynamicMoEMixer] Saved weights to {weights_file} at step {step_id}")

        except Exception as e:
            logger.warning(f"[DynamicMoEMixer] Failed to save weights: {e}")