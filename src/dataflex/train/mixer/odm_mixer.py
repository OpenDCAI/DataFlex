from dataflex.core.registry import register_mixer
from dataflex.utils.logging import logger
from .base_mixer import Mixer

import numpy as np
import torch
import os

@register_mixer("odm")
class ODMMixer(Mixer):
    def __init__(self, mixture_manager, alpha=0.90, warmup_steps=2000, output_dir=None, 
                 accelerator=None, initial_proportions=None, **kwargs):
        """
        Initialize Online Data Mixing (ODM) Mixer based on Multi-Armed Bandits (Exp3).
        
        ODM uses Exp3 with Moving Average to dynamically adjust domain weights during training.
        Key formula: R_hat_t = alpha * R_hat_{t-1} + (1-alpha) * (reward_t / pi_{t-1})
        
        Args:
            mixture_manager: The mixture manager object
            alpha: Smoothing factor for moving average
            warmup_steps: Number of warmup steps with initial proportions
            output_dir: Output directory for saving weight logs
            accelerator: Accelerator object for distributed training
            initial_proportions: Initial domain proportions. If None, uses uniform distribution.
            reward_scale: Scale factor for loss values to amplify signal
            min_exploration_rate: Minimum exploration rate to maintain algorithm sensitivity
        """
        super().__init__(mixture_manager)
        self.alpha = float(alpha)
        self.warmup_steps = int(warmup_steps)
        self.output_dir = output_dir
        self.accelerator = accelerator
        
        # Reward scaling to amplify loss signal for better differentiation
        self.reward_scale = float(kwargs.get('reward_scale', 15.0))
        
        # Minimum exploration rate to prevent algorithm from "freezing" in late stages
        self.min_exploration_rate = float(kwargs.get('min_exploration_rate', 0.03))
        
        # Number of domains
        self.k = len(self.mixture_manager.names)
        
        # Initialize proportions
        if initial_proportions is not None:
            initial_proportions = np.array(initial_proportions, dtype=float)
            if len(initial_proportions) == self.k:
                self.initial_proportions = initial_proportions / np.sum(initial_proportions)
                logger.info(f"[ODMMixer] Using provided initial proportions: {self.initial_proportions}")
            else:
                logger.warning(f"[ODMMixer] Initial proportions mismatch. Using uniform distribution.")
                self.initial_proportions = np.ones(self.k) / self.k
        elif hasattr(self.mixture_manager, 'initial_proportions') and self.mixture_manager.initial_proportions is not None:
            init_props = np.array(self.mixture_manager.initial_proportions, dtype=float)
            if len(init_props) == self.k:
                self.initial_proportions = init_props / np.sum(init_props)
                logger.info(f"[ODMMixer] Using initial proportions from mixture_manager: {self.initial_proportions}")
            else:
                self.initial_proportions = np.ones(self.k) / self.k
        else:
            self.initial_proportions = np.ones(self.k) / self.k
            logger.info(f"[ODMMixer] Using uniform initial distribution")
        
        # Current domain sampling policy π_t
        self.domain_weights = self.initial_proportions.copy()
        
        # Estimated rewards (moving average, NOT cumulative sum)
        self.estimated_rewards = np.zeros(self.k)
        
        # Exploration rate ε_t
        self.exploration_rate = 1.0 / self.k
        
        # Track previous batch info for reward computation
        self.prev_batch_info = None
        
        logger.info(f"[ODMMixer] Initialized: K={self.k}, alpha={self.alpha}, warmup={self.warmup_steps}")
        logger.info(f"[ODMMixer] Scaling: reward_scale={self.reward_scale}, min_eps={self.min_exploration_rate}")
        logger.info(f"[ODMMixer] Domain names: {self.mixture_manager.names}")
    
    def _compute_exploration_rate(self, t):
        """
        Compute exploration rate: ε_t = max{min_eps, min{1/K, sqrt(ln(K) / (K * t))}}
        The min_eps floor ensures algorithm maintains sensitivity in late stages.
        """
        if t <= 0:
            return 1.0 / self.k
        decay_term = np.sqrt(np.log(self.k) / (self.k * t))
        calculated_rate = min(1.0 / self.k, decay_term)
        # Apply floor to maintain minimum exploration (annealing with lower bound)
        return max(self.min_exploration_rate, calculated_rate)
    
    def _update_policy(self, step_t):
        """
        Update sampling policy using Exp3 algorithm.
        π_t(D_i) ∝ exp(ε_{t-1} * R̂_i) * (1 - K*ε_t) + ε_t
        """
        # Validate estimated_rewards
        if not np.all(np.isfinite(self.estimated_rewards)):
            logger.warning(f"[ODMMixer] Invalid estimated_rewards, resetting to 0")
            self.estimated_rewards[:] = 0.0
        
        # Store previous exploration rate
        prev_eps = self.exploration_rate
        
        # Compute new exploration rate
        self.exploration_rate = self._compute_exploration_rate(step_t)
        
        # Compute exp(prev_eps * R̂_i) for each domain
        x = prev_eps * self.estimated_rewards
        x = x - np.max(x)
        x = np.clip(x, -5.0, 0.0)  # Prevent overflow
        exp_scaled_rewards = np.exp(x)
        total_exp = np.sum(exp_scaled_rewards)

        # Safety check
        if not np.isfinite(total_exp) or total_exp <= 0:
            logger.warning(f"[ODMMixer] Invalid exp sum, resetting to initial proportions")
            self.domain_weights = self.initial_proportions.copy()
            self.estimated_rewards[:] = 0.0
            return

        # scaling_factor = (1 - K*ε_t) / Σ exp(...)
        numerator = max(0.0, 1.0 - self.k * self.exploration_rate)
        scaling_factor = numerator / total_exp
        
        # w_i = exp(...) * scaling_factor + ε_t
        self.domain_weights = exp_scaled_rewards * scaling_factor + self.exploration_rate

        # Normalize
        if np.all(np.isfinite(self.domain_weights)) and np.sum(self.domain_weights) > 0:
            self.domain_weights = self.domain_weights / np.sum(self.domain_weights)
            self.domain_weights = np.maximum(self.domain_weights, 1e-8)
            self.domain_weights = self.domain_weights / np.sum(self.domain_weights)
        else:
            logger.warning(f"[ODMMixer] Invalid weights after update, resetting")
            self.domain_weights = self.initial_proportions.copy()
            self.estimated_rewards[:] = 0.0
    
    def _update_reward_from_batch(self, batch_loss, domain_id):
        """
        Update estimated reward using moving average (ODM core formula).
        R̂_t = alpha * R̂_{t-1} + (1-alpha) * (reward / π_{t-1})
        
        This is the KEY difference from standard Exp3: using moving average instead of cumulative sum.
        """
        if not np.isfinite(batch_loss):
            logger.warning(f"[ODMMixer] Invalid batch_loss: {batch_loss}, skipping update")
            return
        
        # Scale loss to amplify signal for Exp3 algorithm
        # Current loss range: ~3-12, scaled to ~150-600 for better differentiation
        reward = batch_loss * self.reward_scale
        
        # Get previous policy probability for importance weighting
        prob = max(self.domain_weights[domain_id], 1e-8)
        
        # Importance-weighted reward
        importance_weighted_reward = reward / prob
        
        if not np.isfinite(importance_weighted_reward):
            logger.warning(f"[ODMMixer] Invalid importance_weighted_reward, skipping")
            return
        
        # CRITICAL: Moving average update (not cumulative sum!)
        # This is the core innovation of ODM over standard Exp3
        old_estimate = self.estimated_rewards[domain_id]
        new_estimate = self.alpha * old_estimate + (1.0 - self.alpha) * importance_weighted_reward
        
        # Validate and clip for numerical stability
        if np.isfinite(new_estimate):
            self.estimated_rewards[domain_id] = np.clip(new_estimate, -10000.0, 10000.0)
        else:
            logger.warning(f"[ODMMixer] Invalid new_estimate for domain {domain_id}, keeping old value")
    
    def mix(self, model, step_id: int, **kwargs) -> np.ndarray:
        """
        Compute new domain weights using ODM algorithm.
        
        Args:
            model: Current model being trained
            step_id: Current training step
            **kwargs: Must contain training batch info for reward update
            
        Returns:
            np.ndarray: Updated domain proportions
        """
        output_dir = kwargs.get('output_dir', self.output_dir)
        
        # Warmup phase: use initial proportions
        if step_id <= self.warmup_steps:
            logger.info(f"[ODMMixer] Step {step_id} (warmup): Using initial proportions")
            return self.initial_proportions.copy()
        
        logger.info(f"[ODMMixer] Step {step_id}: Updating domain weights with ODM")
        
        # Update reward from previous training batch
        if self.prev_batch_info is not None:
            prev_loss = self.prev_batch_info.get('loss')
            prev_domain_id = self.prev_batch_info.get('domain_id')
            
            if prev_loss is not None and prev_domain_id is not None:
                # Update reward using moving average (ODM's key innovation)
                self._update_reward_from_batch(prev_loss, prev_domain_id)
                domain_name = self.mixture_manager.names[prev_domain_id]
                logger.info(f"[ODMMixer] Updated reward for '{domain_name}': "
                           f"loss={prev_loss:.4f}, R̂={self.estimated_rewards[prev_domain_id]:.4f}")
        
        # Update policy using Exp3
        self._update_policy(step_id - self.warmup_steps)
        
        # Validate and normalize
        if np.any(~np.isfinite(self.domain_weights)):
            logger.error(f"[ODMMixer] Invalid domain weights, falling back to initial proportions")
            return self.initial_proportions.copy()

        self.domain_weights = np.maximum(self.domain_weights, 0)
        self.domain_weights = self.domain_weights / np.sum(self.domain_weights)

        # Log results
        logger.info(f"[ODMMixer] ε_t={self.exploration_rate:.6f}")
        logger.info(f"[ODMMixer] Updated domain weights:")
        for i, name in enumerate(self.mixture_manager.names):
            logger.info(f"  {name}: {self.domain_weights[i]:.4f} (R̂={self.estimated_rewards[i]:.4f})")
        
        # Save weights
        if output_dir is not None:
            self.save_weights_to_jsonl(output_dir, step_id)
        
        return self.domain_weights.copy()
    
    def update_batch_info(self, batch_loss, domain_id):
        """
        Store batch information for next reward update.
        Should be called from trainer after each training step.
        
        Args:
            batch_loss: Training loss from the batch
            domain_id: Domain ID of the sampled batch
        """
        self.prev_batch_info = {
            'loss': batch_loss,
            'domain_id': domain_id
        }
    
    def save_weights_to_jsonl(self, output_dir, step_id):
        """Save domain weights to JSONL file (append mode, one line per update)."""
        if self.accelerator is not None and not self.accelerator.is_main_process:
            return
        
        try:
            import json
            import time
            
            if output_dir is None:
                return
            
            os.makedirs(output_dir, exist_ok=True)
            weights_file = os.path.join(output_dir, "odm_weights.jsonl")
            
            log_entry = {
                "step": step_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "domain_names": self.mixture_manager.names,
                "domain_weights": self.domain_weights.tolist(),
                "estimated_rewards": self.estimated_rewards.tolist(),
                "exploration_rate": float(self.exploration_rate),
                "alpha": self.alpha,
            }
            
            with open(weights_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            logger.warning(f"[ODMMixer] Failed to log weights: {e}")
    
    def get_current_weights(self):
        """Get current domain weights."""
        return self.domain_weights.copy()
