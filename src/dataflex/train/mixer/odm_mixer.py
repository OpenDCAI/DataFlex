from dataflex.core.registry import register_mixer
from dataflex.utils.logging import logger
from .base_mixer import Mixer

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os
from contextlib import nullcontext

@register_mixer("odm")
class ODMMixer(Mixer):
    def __init__(self, mixture_manager, alpha=0.9, warmup_steps=1000, num_eval_samples=1000, 
                 eval_batch_size=8, output_dir=None, accelerator=None, data_collator=None,
                 dataset=None, initial_proportions=None, **kwargs):
        """
        Initialize Online Data Mixing (ODM) Mixer based on Multi-Armed Bandits (Exp3).
        
        ODM uses the Exp3 algorithm to dynamically adjust domain weights during training
        by using training loss as a reward signal. This is more efficient than DoReMi as it
        doesn't require a reference model.
        
        Based on the official implementation from:
        https://github.com/alon-albalak/online-data-mixing
        
        Args:
            mixture_manager: The mixture manager object
            alpha: Smoothing factor for moving average of rewards (default: 0.9, as in official code)
            warmup_steps: Number of warmup steps with uniform sampling (default: 1000)
            num_eval_samples: Number of samples to evaluate per domain (default: 1000)
            eval_batch_size: Batch size for evaluation (default: 8)
            output_dir: Output directory for saving weight logs (default: None)
            accelerator: Accelerator object for distributed training
            data_collator: Data collator for batching
            datatset: Training dataset
            initial_proportions: Initial domain proportions. If None, uses uniform distribution.
            reward_scaling: Scaling factor for rewards in Exp3 (default: 10.0, higher = more aggressive reweighting)
        """
        super().__init__(mixture_manager)
        self.alpha = float(alpha)
        self.warmup_steps = int(warmup_steps)
        self.num_eval_samples = int(num_eval_samples)
        self.eval_batch_size = int(eval_batch_size)
        self.output_dir = output_dir
        self.accelerator = accelerator
        self.data_collator = data_collator
        self.dataset = dataset
        
        # Initialize domain settings
        self.k = len(self.mixture_manager.names)  # Number of domains
        
        # Initialize proportions
        if initial_proportions is not None:
            initial_proportions = np.array(initial_proportions, dtype=float)
            if len(initial_proportions) == self.k:
                # Normalize to ensure they sum to 1
                self.initial_proportions = initial_proportions / np.sum(initial_proportions)
                logger.info(f"[ODMMixer] Using provided initial proportions: {self.initial_proportions}")
            else:
                logger.warning(f"[ODMMixer] Initial proportions length ({len(initial_proportions)}) "
                             f"doesn't match number of domains ({self.k}). Using uniform distribution.")
                self.initial_proportions = np.ones(self.k) / self.k
        elif hasattr(self.mixture_manager, 'initial_proportions') and self.mixture_manager.initial_proportions is not None:
            # Use from mixture_manager
            init_props = np.array(self.mixture_manager.initial_proportions, dtype=float)
            if len(init_props) == self.k:
                self.initial_proportions = init_props / np.sum(init_props)
                logger.info(f"[ODMMixer] Using initial proportions from mixture_manager: {self.initial_proportions}")
            else:
                self.initial_proportions = np.ones(self.k) / self.k
        else:
            # Default to uniform distribution
            self.initial_proportions = np.ones(self.k) / self.k
            logger.info(f"[ODMMixer] Using uniform initial distribution: {self.initial_proportions}")
        
        # Current domain proportions (policy π)
        self.domain_weights = self.initial_proportions.copy()
        
        # Initialize cumulative estimated rewards (as in official implementation)
        # These are cumulative importance-weighted rewards that grow over time
        self.cumulative_estimated_rewards = np.zeros(self.k)
        
        # Track the current exploration rate
        self.exploration_rate = 1.0 / self.k
        
        # Track step counter
        self.step_counter = 0
        
        logger.info(f"[ODMMixer] Initialized with:")
        logger.info(f"  K (num_domains) = {self.k}")
        logger.info(f"  Domain names = {self.mixture_manager.names}")
        logger.info(f"  alpha (moving avg) = {self.alpha}")
        logger.info(f"  warmup_steps = {self.warmup_steps}")
        logger.info(f"  num_eval_samples = {self.num_eval_samples}")
        logger.info(f"  eval_batch_size = {self.eval_batch_size}")
        logger.info(f"  initial_proportions = {self.initial_proportions}")
    
    def _compute_exploration_rate(self, t):
        """
        Compute the exploration rate at step t using decaying exploration.
        
        ε_t = min{1/K, sqrt(ln(K) / (K * t))}
        
        Args:
            t: Current training step (1-indexed)
            
        Returns:
            Exploration rate for step t
        """
        if t <= 0:
            return 1.0 / self.k
        
        decay_term = np.sqrt(np.log(self.k) / (self.k * t))
        exploration_rate = min(1.0 / self.k, decay_term)
        
        return exploration_rate
    
    def _update_policy(self, step_t):
        """
        Update the sampling policy using Exp3 algorithm (official ODM implementation).
        
        Following the official code:
        - scaling_factor = (1 - K*ε_t) / Σ_j exp(ε_{t-1} * R̂_j)
        - w_i = exp(ε_{t-1} * R̂_i) * scaling_factor + ε_t
        - π_t(D_i) = w_i / Σ_j w_j
        
        Args:
            step_t: Current training step
        """
        # CRITICAL: First check if cumulative_estimated_rewards are valid
        # This prevents cascading numerical errors
        if not np.all(np.isfinite(self.cumulative_estimated_rewards)):
            logger.warning(f"[ODMMixer] cumulative_estimated_rewards contain NaN/inf: {self.cumulative_estimated_rewards}, resetting to 0")
            self.cumulative_estimated_rewards[:] = 0.0
        
        # Store previous exploration rate (used in weight calculation)
        prev_eps = self.exploration_rate
        
        # Compute exploration rate for this step
        self.exploration_rate = self._compute_exploration_rate(step_t)
        
        # Calculate scaling factor following official implementation
        # total_estimated_rewards = Σ exp(prev_eps * R̂_cumulative_j)
        # Following official: line 397
        x = prev_eps * self.cumulative_estimated_rewards
        # Clip to prevent overflow in exp()
        x = np.clip(x, -50.0, 50.0)
        exp_scaled_rewards = np.exp(x)
        total_estimated_rewards = np.sum(exp_scaled_rewards)

        # Safety check: if total_estimated_rewards is invalid, reset to initial proportions
        if not np.isfinite(total_estimated_rewards) or total_estimated_rewards <= 0:
            logger.warning(f"[ODMMixer] total_estimated_rewards invalid ({total_estimated_rewards}), reset to initial proportions")
            self.domain_weights = self.initial_proportions.copy()
            self.cumulative_estimated_rewards[:] = 0.0
            return

        # scaling_factor = (1 - K*ε_t) / Σ exp(prev_eps * R̂_j)
        # Following official: line 398
        numerator = 1 - self.k * self.exploration_rate
        # Ensure numerator is non-negative (should always be true when ε ≤ 1/K)
        if numerator < 0:
            logger.warning(f"[ODMMixer] numerator negative ({numerator}), using 0")
            numerator = 0.0
            
        scaling_factor = numerator / total_estimated_rewards
        
        # Update weights: w_i = exp(prev_eps * R̂_cumulative_i) * scaling_factor + ε_t
        # Following official: line 404
        self.domain_weights = exp_scaled_rewards * scaling_factor + self.exploration_rate

        # Robustness check for domain_weights
        if not np.all(np.isfinite(self.domain_weights)):
            logger.warning(f"[ODMMixer] domain_weights has NaN/inf: {self.domain_weights}, reset to initial_proportions")
            self.domain_weights = self.initial_proportions.copy()
            self.cumulative_estimated_rewards[:] = 0.0  # Also reset rewards to prevent future issues
        else:
            # Normalize to get probabilities
            weight_sum = np.sum(self.domain_weights)
            if weight_sum > 0:
                self.domain_weights = self.domain_weights / weight_sum
            else:
                logger.warning(f"[ODMMixer] All domain weights sum to zero, resetting to initial proportions")
                self.domain_weights = self.initial_proportions.copy()
                # logger.warning("[ODMMixer] All domain weights sum to zero, resetting to uniform")
                # self.domain_weights = np.ones(self.k) / self.k

            # Clip very small probabilities to avoid numerical issues
            self.domain_weights = np.maximum(self.domain_weights, 1e-8)
            self.domain_weights = self.domain_weights / np.sum(self.domain_weights)
    
    def _compute_pertoken_loss(self, model, batch):
        """
        Compute per-token loss for a batch (adapted from DoremiMixer).
        
        Args:
            model: The model to evaluate
            batch: The batch to process
        
        Returns:
            avg_loss: Scalar average loss for the batch
            num_tokens: Number of valid tokens
        """
        with torch.no_grad():
            # Ensure model is in eval mode
            was_training = model.training
            model.eval()
            
            # Check batch contents
            if 'input_ids' not in batch:
                if was_training:
                    model.train()
                return 0.0, 0
            
            input_ids = batch['input_ids']
            
            # Prepare labels for causal LM (PT stage)
            labels = batch.get('labels')
            if labels is None:
                labels = input_ids.clone()
            
            # Prepare attention_mask
            attention_mask = batch.get('attention_mask')
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            
            try:
                # Prepare batch for model
                model_batch = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels,
                    'return_dict': True
                }
                
                # Forward pass
                outputs = model(**model_batch)
                
                # Get loss from model output
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    loss = outputs.loss
                    # Count valid tokens (excluding padding)
                    num_tokens = (labels != -100).sum().item()
                    
                    if num_tokens == 0:
                        if was_training:
                            model.train()
                        return 0.0, 0
                    
                    avg_loss = loss.item()
                    
                    # Restore training state
                    if was_training:
                        model.train()
                    
                    return avg_loss, num_tokens
                
                # Fallback: manual loss computation
                logits = outputs.logits
                
                # Shift for causal LM: predict next token
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Count valid tokens
                valid_mask = (shift_labels != -100)
                num_tokens = valid_mask.sum().item()
                
                if num_tokens == 0:
                    if was_training:
                        model.train()
                    return 0.0, 0
                
                # Compute cross-entropy loss
                vocab_size = shift_logits.size(-1)
                loss = F.cross_entropy(
                    shift_logits.view(-1, vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction='sum'
                )
                
                avg_loss = (loss / num_tokens).item()
                
                # Restore training state
                if was_training:
                    model.train()
                
                return avg_loss, num_tokens
                    
            except Exception as e:
                # Restore training state on error
                if was_training:
                    model.train()
                    
                logger.error(f"[ODMMixer] Error in _compute_pertoken_loss: {e}")
                import traceback
                logger.error(f"[ODMMixer] Traceback: {traceback.format_exc()}")
                return 0.0, 0
    
    def _evaluate_domain_loss(self, model, domain_id):
        """
        Evaluate loss for a specific domain by sampling batches.
        
        Args:
            model: The current training model
            domain_id: Index of the domain to evaluate
            
        Returns:
            avg_loss: Average loss for this domain
        """
        # Get the domain name and corresponding dataset
        domain_name = self.mixture_manager.names[domain_id]
        
        # Use processed dataset instead of raw sources
        if hasattr(self.mixture_manager, 'per_source') and domain_name in self.mixture_manager.per_source:
            domain_dataset = self.mixture_manager.per_source[domain_name]
        elif hasattr(self.mixture_manager, 'sources') and domain_name in self.mixture_manager.sources:
            domain_dataset = self.mixture_manager.sources[domain_name]
        else:
            logger.error(f"[ODMMixer] Cannot find dataset for domain '{domain_name}'")
            return 0.0
        
        # Sample a subset if dataset is large
        dataset_size = len(domain_dataset)
        num_samples = min(self.num_eval_samples, dataset_size)
        
        if num_samples == 0:
            logger.warning(f"[ODMMixer] Domain '{domain_name}' has no samples, returning 0 loss")
            return 0.0
        
        # Create random indices for sampling
        if num_samples == dataset_size:
            subset = domain_dataset
        else:
            indices = np.random.choice(dataset_size, num_samples, replace=False)
            subset = Subset(domain_dataset, indices)
        
        # Create dataloader
        try:
            if self.data_collator is None:
                from transformers import default_data_collator
                collate_fn = default_data_collator
            else:
                collate_fn = self.data_collator
            
            dataloader = DataLoader(
                subset,
                batch_size=self.eval_batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                drop_last=True,
            )
        except Exception as e:
            logger.error(f"[ODMMixer] Failed to create dataloader for domain '{domain_name}': {e}")
            return 0.0
        
        # Collect losses
        total_loss = 0.0
        total_tokens = 0
        
        model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                try:
                    # Move batch to device
                    device = next(model.parameters()).device
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    # Compute loss
                    batch_loss, batch_tokens = self._compute_pertoken_loss(model, batch)
                    
                    if batch_tokens == 0:
                        continue
                    
                    # Accumulate weighted by number of tokens
                    total_loss += batch_loss * batch_tokens
                    total_tokens += batch_tokens
                        
                except Exception as e:
                    logger.warning(f"[ODMMixer] Error processing batch {batch_idx} for domain '{domain_name}': {e}")
                    continue
        
        # Compute average loss
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            logger.info(f"[ODMMixer] Domain '{domain_name}' - avg_loss={avg_loss:.4f} ({total_tokens} tokens)")
        else:
            avg_loss = 0.0
            logger.warning(f"[ODMMixer] Domain '{domain_name}' - No valid tokens processed!")
        
        return avg_loss
    
    def _update_rewards(self, domain_losses):
        """
        Update the cumulative estimated rewards using importance weighting (official ODM implementation).

        Following official code (line 383-386):
        1. Scale down reward: reward = reward/10
        2. Update cumulative reward with importance weighting:
           cumulative_reward[i] += reward / probability[i]

        Args:
            domain_losses: Array of current losses for each domain
        """
        # Validate input
        if not np.all(np.isfinite(domain_losses)):
            logger.warning(f"[ODMMixer] domain_losses contain NaN/inf: {domain_losses}, skipping update")
            return

        for i in range(self.k):
            loss = domain_losses[i]

            # Validate loss
            if not np.isfinite(loss):
                logger.warning(f"[ODMMixer] Invalid loss for domain {i}: {loss}, skipping")
                continue

            # Scale down reward so that cumulative reward doesn't explode
            # Following official: line 383
            reward = loss / 10.0

            # Validate reward before importance weighting
            if not np.isfinite(reward):
                logger.warning(f"[ODMMixer] Invalid reward for domain {i}: {reward}, skipping")
                continue

            # Get current probability (domain weight) for importance weighting
            prob = self.domain_weights[i]
            
            # Ensure probability is not too small to avoid division by very small numbers
            if prob < 1e-8:
                logger.warning(f"[ODMMixer] Probability too small for domain {i}: {prob}, using 1e-8")
                prob = 1e-8

            # Update cumulative estimated reward with importance weighting
            # Following official: line 386
            importance_weighted_reward = reward / prob
            
            # Validate before adding to cumulative
            if not np.isfinite(importance_weighted_reward):
                logger.warning(f"[ODMMixer] Invalid importance_weighted_reward for domain {i}: {importance_weighted_reward}, skipping")
                continue
            
            # Accumulate the importance-weighted reward
            new_cumulative = self.cumulative_estimated_rewards[i] + importance_weighted_reward
            
            # Validate and clip to prevent extreme growth
            if not np.isfinite(new_cumulative):
                logger.warning(f"[ODMMixer] Invalid new_cumulative for domain {i}: {new_cumulative}, skipping update")
                continue
            
            # Clip to reasonable range to prevent numerical overflow in exp() later
            # Since we use exp(prev_eps * cumulative), and prev_eps ≈ 0.005-0.14,
            # cumulative should stay < 500 to keep exp argument < 70
            self.cumulative_estimated_rewards[i] = np.clip(new_cumulative, -500.0, 500.0)

        # Final validation of all cumulative rewards
        if not np.all(np.isfinite(self.cumulative_estimated_rewards)):
            logger.error(f"[ODMMixer] cumulative_estimated_rewards invalid after update: {self.cumulative_estimated_rewards}")
            logger.error(f"[ODMMixer] Resetting all cumulative rewards to 0")
            self.cumulative_estimated_rewards[:] = 0.0
    
    def mix(self, model, step_id: int, **kwargs) -> np.ndarray:
        """
        Compute new domain weights using ODM (Online Data Mixing) algorithm.
        
        The ODM algorithm:
        1. During warmup: use initial proportions (uniform or provided)
        2. After warmup:
           a. Evaluate current loss for each domain
           b. Update estimated rewards using importance weighting and moving average
           c. Compute exploration rate (decaying over time)
           d. Update policy using Exp3 (mix of Gibbs and uniform)
        3. Return normalized weights
        
        Args:
            model: Current model being trained
            step_id: Current training step
            **kwargs: Additional arguments (can include output_dir for logging)
            
        Returns:
            np.ndarray: Updated domain proportions (normalized)
        """
        # Get output_dir from kwargs if provided
        output_dir = kwargs.get('output_dir', self.output_dir)
        
        self.step_counter = step_id
        
        # During warmup, use initial proportions
        if step_id <= self.warmup_steps:
            logger.info(f"[ODMMixer] Step {step_id} (warmup): Using initial proportions")
            for i, name in enumerate(self.mixture_manager.names):
                logger.info(f"  {name}: {self.initial_proportions[i]:.4f}")
            return self.initial_proportions.copy()
        
        # After warmup: use ODM algorithm
        logger.info(f"[ODMMixer] Step {step_id}: Updating domain weights with ODM")
        
        # Step 1: Evaluate current losses for each domain
        logger.info(f"[ODMMixer] Evaluating {self.k} domains...")
        domain_losses = []
        
        for domain_id in range(self.k):
            domain_name = self.mixture_manager.names[domain_id]
            
            try:
                avg_loss = self._evaluate_domain_loss(model, domain_id)
                # Clip to avoid extreme values
                avg_loss = max(0.0, avg_loss)
                domain_losses.append(avg_loss)
                
                logger.info(f"[ODMMixer] Domain '{domain_name}': loss={avg_loss:.4f}")
                
            except Exception as e:
                logger.warning(f"[ODMMixer] Failed to evaluate domain '{domain_name}': {e}. "
                             f"Using default loss value.")
                # If evaluation fails, use a default loss value
                # Use the mean of successfully evaluated losses, or 1.0 as fallback
                if len(domain_losses) > 0:
                    approx_loss = np.mean(domain_losses)
                else:
                    approx_loss = 1.0
                domain_losses.append(approx_loss)
        
        domain_losses = np.array(domain_losses)
        
        # Validate domain losses
        if not np.all(np.isfinite(domain_losses)):
            logger.error(f"[ODMMixer] Domain losses contain NaN/inf: {domain_losses}")
            logger.error(f"[ODMMixer] Falling back to initial proportions")
            return self.initial_proportions.copy()
        
        # Log the domain losses for debugging
        logger.info(f"[ODMMixer] Domain losses: min={np.min(domain_losses):.4f}, "
                   f"max={np.max(domain_losses):.4f}, mean={np.mean(domain_losses):.4f}")
        
        # Step 2: Update cumulative estimated rewards based on domain losses
        self._update_rewards(domain_losses)
        logger.info(f"[ODMMixer] Updated cumulative estimated rewards: {self.cumulative_estimated_rewards}")
        
        # Step 3: Update policy using Exp3
        self._update_policy(step_id - self.warmup_steps)  # Use step relative to warmup end
        
        # Final validation of domain weights
        if np.any(np.isnan(self.domain_weights)) or np.any(np.isinf(self.domain_weights)):
            logger.error(f"[ODMMixer] Final domain weights contain NaN or Inf: {self.domain_weights}")
            logger.error(f"[ODMMixer] Falling back to initial proportions")
            return self.initial_proportions.copy()

        # Ensure weights sum to 1 and are non-negative
        self.domain_weights = np.maximum(self.domain_weights, 0)
        self.domain_weights = self.domain_weights / np.sum(self.domain_weights)

        # Log updated weights
        logger.info(f"[ODMMixer] Exploration rate ε_t = {self.exploration_rate:.6f}")
        logger.info(f"[ODMMixer] Updated domain weights:")
        for i, name in enumerate(self.mixture_manager.names):
            logger.info(f"  {name}: {self.domain_weights[i]:.4f} "
                       f"(loss: {domain_losses[i]:.4f}, cumulative_reward: {self.cumulative_estimated_rewards[i]:.4f})")
        
        # Save weights to JSONL file for tracking
        if output_dir is not None:
            self.save_weights_to_jsonl(output_dir, step_id)
        
        return self.domain_weights.copy()
    
    def save_weights_to_jsonl(self, output_dir, step_id):
        """
        Save current domain weights to a JSONL file (append mode).
        This logs each weight update as a new line in the same file.
        Only the main process saves to avoid duplicate entries in distributed training.
        
        Args:
            output_dir: Directory to save the JSONL file
            step_id: Current training step
        """
        # Only save on main process to avoid duplicate entries
        if self.accelerator is not None and not self.accelerator.is_main_process:
            return
        
        try:
            import json
            import time
            
            if output_dir is None:
                logger.warning(f"[ODMMixer] No output_dir provided, skipping weight logging")
                return
            
            os.makedirs(output_dir, exist_ok=True)
            weights_file = os.path.join(output_dir, "odm_weights.jsonl")
            
            # Prepare log entry with detailed information
            log_entry = {
                "step": step_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "domain_names": self.mixture_manager.names,
                "domain_weights": self.domain_weights.tolist(),
                "cumulative_estimated_rewards": self.cumulative_estimated_rewards.tolist(),
                "exploration_rate": float(self.exploration_rate),
                "alpha": self.alpha,
                "warmup_steps": self.warmup_steps,
                "is_warmup": step_id <= self.warmup_steps
            }
            
            # Append to JSONL file
            with open(weights_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
            logger.info(f"[ODMMixer] Logged domain weights to {weights_file} at step {step_id}")
            
        except Exception as e:
            logger.warning(f"[ODMMixer] Failed to log domain weights to JSONL: {e}")
    
    def save_weights(self, output_dir, step_id):
        """Save current domain weights to file (for compatibility)."""
        try:
            import json
            weights_file = os.path.join(output_dir, f"odm_weights_step_{step_id}.json")
            
            weights_data = {
                "step": step_id,
                "domain_names": self.mixture_manager.names,
                "domain_weights": self.domain_weights.tolist(),
                "cumulative_estimated_rewards": self.cumulative_estimated_rewards.tolist(),
                "exploration_rate": float(self.exploration_rate),
                "alpha": self.alpha,
                "warmup_steps": self.warmup_steps
            }
            
            os.makedirs(output_dir, exist_ok=True)
            with open(weights_file, 'w') as f:
                json.dump(weights_data, f, indent=2)
                
            logger.info(f"[ODMMixer] Saved domain weights to {weights_file}")
            
        except Exception as e:
            logger.warning(f"[ODMMixer] Failed to save domain weights: {e}")
    
    def get_current_weights(self):
        """Get the current domain weights."""
        return self.domain_weights.copy()
    
    def get_cumulative_estimated_rewards(self):
        """Get the current cumulative estimated rewards."""
        return self.cumulative_estimated_rewards.copy()
