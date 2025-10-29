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

@register_mixer("doremi")
class DoremiMixer(Mixer):
    def __init__(self, mixture_manager, reference_model_path=None, reweight_eta=1.0, reweight_eps=1e-3, 
                 num_eval_samples=1000, eval_batch_size=8, accelerator=None, data_collator=None,
                 dataset=None, **kwargs):
        """
        Initialize DoReMi Mixer.
        
        Args:
            mixture_manager: The mixture manager object
            reference_model_path: Path to the reference model checkpoint
            reweight_eta: Learning rate for domain weight updates (default: 1.0)
            reweight_eps: Smoothing parameter for domain weights (default: 1e-3)
            num_eval_samples: Number of samples to evaluate per domain (default: 1000)
            eval_batch_size: Batch size for evaluation (default: 8)
            accelerator: Accelerator object for distributed training
            data_collator: Data collator for batching
            dataset: Training dataset
        """
        super().__init__(mixture_manager)
        self.reference_model_path = reference_model_path
        # Ensure parameters are numeric types
        self.reweight_eta = float(reweight_eta)
        self.reweight_eps = float(reweight_eps)
        self.num_eval_samples = int(num_eval_samples)
        self.eval_batch_size = int(eval_batch_size)
        self.accelerator = accelerator
        self.data_collator = data_collator
        self.dataset = dataset
        
        # Initialize domain weights from mixture_manager's initial proportions
        k = len(self.mixture_manager.names)
        if hasattr(self.mixture_manager, 'initial_proportions') and self.mixture_manager.initial_proportions is not None:
            # Use the initial proportions from mixture_manager
            init_proportions = np.array(self.mixture_manager.initial_proportions)
            if len(init_proportions) == k:
                self.domain_weights = init_proportions.copy()
                logger.info(f"[DoremiMixer] Using initial proportions from mixture_manager: {self.domain_weights}")
            else:
                logger.warning(f"[DoremiMixer] Initial proportions length ({len(init_proportions)}) doesn't match "
                             f"number of domains ({k}). Using uniform distribution.")
                self.domain_weights = np.ones(k) / k
        else:
            # Fallback to uniform distribution
            self.domain_weights = np.ones(k) / k
            logger.info(f"[DoremiMixer] No initial proportions found, using uniform distribution: {self.domain_weights}")
        
        # Initialize per-domain scores (used for tracking excess loss)
        # Use log vocabulary size as initial value (similar to DoReMi)
        self.perdomain_scores = np.ones(k) * np.log(50000)  # Approximate vocabulary size
        
        # Load reference model
        self.reference_model = None
        self.reference_model_loaded = False
        
        # Check for DeepSpeed ZeRO-3 which requires special handling
        self.using_deepspeed_zero3 = False
        if accelerator is not None and hasattr(accelerator, 'state'):
            if hasattr(accelerator.state, 'deepspeed_plugin'):
                ds_plugin = accelerator.state.deepspeed_plugin
                if ds_plugin is not None and hasattr(ds_plugin, 'zero_stage'):
                    if ds_plugin.zero_stage == 3:
                        self.using_deepspeed_zero3 = True
                        logger.warning(
                            "[DoremiMixer] DeepSpeed ZeRO-3 detected! "
                            "For best results, consider using:\n"
                            "  1. ZeRO-2 instead of ZeRO-3 (change deepspeed config)\n"
                            "  2. LoRA instead of full finetuning (set finetuning_type: lora)\n"
                            "Proceeding with workaround but evaluation may be slow."
                        )
        
        if reference_model_path is None:
            logger.warning(f"[DoremiMixer] No reference model path provided. Will use proxy model loss only.")
        
        logger.info(f"[DoremiMixer] Initialized with eta={self.reweight_eta} (type: {type(self.reweight_eta)}), "
                   f"eps={self.reweight_eps} (type: {type(self.reweight_eps)}), "
                   f"num_domains={k}, domain_names={self.mixture_manager.names}, "
                   f"reference_model_path={reference_model_path}, "
                   f"using_deepspeed_zero3={self.using_deepspeed_zero3}")
    
    def _load_reference_model(self, proxy_model):
        """Lazy loading of reference model."""
        if self.reference_model_loaded:
            return
        
        if self.reference_model_path is None or not os.path.exists(self.reference_model_path):
            if self.reference_model_path is not None:
                logger.warning(f"[DoremiMixer] Reference model path not found: {self.reference_model_path}. "
                              f"Using proxy model loss only.")
            self.reference_model_loaded = True
            return
        
        try:
            from transformers import AutoModelForCausalLM
            
            logger.info(f"[DoremiMixer] Loading reference model from {self.reference_model_path}")
            self.reference_model = AutoModelForCausalLM.from_pretrained(
                self.reference_model_path,
                torch_dtype=proxy_model.dtype if hasattr(proxy_model, 'dtype') else torch.float32,
                trust_remote_code=True  # For Qwen models
            )
            
            # Move to the same device as proxy model
            device = next(proxy_model.parameters()).device
            self.reference_model = self.reference_model.to(device)
            
            # Set to eval mode and freeze parameters
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad = False
            
            # Log model info
            num_params = sum(p.numel() for p in self.reference_model.parameters())
            logger.info(f"[DoremiMixer] Reference model loaded successfully with {num_params:,} parameters")
            logger.info(f"[DoremiMixer] Reference model device: {device}, dtype: {self.reference_model.dtype}")
            
            self.reference_model_loaded = True
            
        except Exception as e:
            logger.error(f"[DoremiMixer] Failed to load reference model: {e}")
            import traceback
            logger.error(f"[DoremiMixer] Traceback: {traceback.format_exc()}")
            self.reference_model = None
            self.reference_model_loaded = True
    
    def _prepare_model_for_eval(self, model, log_info=False):
        """
        Prepare model for evaluation, handling DeepSpeed ZeRO-3 parameter gathering.
        
        Args:
            model: The model to prepare
            log_info: Whether to log information (only log once per domain evaluation)
        
        Returns:
            (eval_model, context_manager): Model to use and context manager for parameter gathering
        """
        # For ZeRO-3, we need to gather parameters before evaluation
        if self.using_deepspeed_zero3:
            try:
                import deepspeed
                # Get the unwrapped model
                if hasattr(model, 'module'):
                    base_model = model.module
                else:
                    base_model = model
                
                # Create context manager for gathering all parameters
                # modifier_rank=None means all ranks get full parameters
                params_to_gather = [p for p in base_model.parameters() if hasattr(p, 'ds_id')]
                
                if params_to_gather:
                    if log_info:
                        logger.info(f"[DoremiMixer] Using DeepSpeed parameter gathering for {len(params_to_gather)} parameters")
                    context = deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=None)
                    return base_model, context
                else:
                    if log_info:
                        logger.warning(f"[DoremiMixer] No DeepSpeed parameters found, using model as-is")
                    return base_model, nullcontext()
                    
            except Exception as e:
                logger.error(f"[DoremiMixer] Failed to setup DeepSpeed parameter gathering: {e}")
                # Fallback to using model as-is
                return model, nullcontext()
        
        # For non-ZeRO-3, just unwrap the model
        if hasattr(model, 'module'):
            return model.module, nullcontext()
        
        return model, nullcontext()
    
    def _compute_pertoken_loss(self, model, batch, log_info=False):
        """
        Compute per-token loss for a batch.
        Handles DeepSpeed ZeRO-3 by using unwrapped model for evaluation.
        
        Args:
            model: The model to evaluate
            batch: The batch to process
            log_info: Whether to log information (only for first batch)
        
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
                raise ValueError("No input_ids found in batch")
            
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
                # Prepare model for evaluation (handles DeepSpeed ZeRO-3)
                eval_model, param_context = self._prepare_model_for_eval(model, log_info=log_info)
                
                # Prepare batch for model
                model_batch = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels,
                    'return_dict': True
                }
                
                # Forward pass with parameter gathering context
                with param_context:
                    outputs = eval_model(**model_batch)
                    
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
                    
                logger.error(f"[DoremiMixer] Error in _compute_pertoken_loss: {e}")
                logger.error(f"[DoremiMixer] Batch info - input_ids shape: {input_ids.shape}")
                import traceback
                logger.error(f"[DoremiMixer] Traceback: {traceback.format_exc()}")
                raise
    
    def _evaluate_domain_losses(self, proxy_model, domain_id):
        """
        Evaluate excess loss for a specific domain.
        
        Args:
            proxy_model: The current training model
            domain_id: Index of the domain to evaluate
            
        Returns:
            avg_excess_loss: Average excess loss for this domain
        """
        # Get the domain name and corresponding dataset
        domain_name = self.mixture_manager.names[domain_id]
        # Use processed dataset instead of raw sources
        if hasattr(self.mixture_manager, 'per_source') and domain_name in self.mixture_manager.per_source:
            domain_dataset = self.mixture_manager.per_source[domain_name]
        elif hasattr(self.mixture_manager, 'sources') and domain_name in self.mixture_manager.sources:
            domain_dataset = self.mixture_manager.sources[domain_name]
        else:
            logger.error(f"[DoremiMixer] Cannot find dataset for domain '{domain_name}'")
            return 0.0
        
        # Sample a subset if dataset is large
        dataset_size = len(domain_dataset)
        num_samples = min(self.num_eval_samples, dataset_size)
        
        if num_samples == 0:
            logger.warning(f"[DoremiMixer] Domain '{domain_name}' has no samples, returning 0 excess loss")
            return 0.0
        
        # Create random indices for sampling
        if num_samples == dataset_size:
            # Use all samples if dataset is small
            subset = domain_dataset
        else:
            # Sample without replacement
            indices = np.random.choice(dataset_size, num_samples, replace=False)
            subset = Subset(domain_dataset, indices)
        
        # Create dataloader with error handling
        try:
            # If no data_collator provided, use a simple one
            if self.data_collator is None:
                logger.warning(f"[DoremiMixer] No data_collator provided, using default")
                from transformers import default_data_collator
                collate_fn = default_data_collator
            else:
                collate_fn = self.data_collator
            
            dataloader = DataLoader(
                subset,
                batch_size=self.eval_batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                drop_last=True,  # Drop incomplete batches to avoid dimension issues
            )
        except Exception as e:
            logger.error(f"[DoremiMixer] Failed to create dataloader for domain '{domain_name}': {e}")
            return 0.0
        
        # Collect losses
        excess_losses = []
        total_tokens = 0
        
        proxy_model.eval()
        
        # Log info once at the start
        logger.info(f"[DoremiMixer] Evaluating domain '{domain_name}' with {len(dataloader)} batches")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                try:
                    # Validate batch structure
                    from transformers.tokenization_utils_base import BatchEncoding
                    if not isinstance(batch, (dict, BatchEncoding)):
                        logger.warning(f"[DoremiMixer] Invalid batch type for domain '{domain_name}', batch {batch_idx}")
                        continue
                    
                    if 'input_ids' not in batch:
                        logger.warning(f"[DoremiMixer] No input_ids in batch for domain '{domain_name}', batch {batch_idx}")
                        continue
                    
                    # Check batch dimensions
                    input_ids = batch['input_ids']
                    if len(input_ids.shape) != 2 or input_ids.shape[0] == 0:
                        logger.warning(f"[DoremiMixer] Invalid input_ids shape for domain '{domain_name}', batch {batch_idx}")
                        continue
                    
                    # Move batch to device
                    device = next(proxy_model.parameters()).device
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    # Compute proxy model loss (only log for first batch)
                    proxy_loss, proxy_tokens = self._compute_pertoken_loss(proxy_model, batch, log_info=(batch_idx == 0))
                    
                    if proxy_tokens == 0:
                        continue  # Skip batches with no valid tokens
                    
                    # Compute reference model loss if available (only log for first batch)
                    if self.reference_model is not None:
                        ref_loss, ref_tokens = self._compute_pertoken_loss(self.reference_model, batch, log_info=(batch_idx == 0))
                        
                        # Debug logging for first few batches
                        if batch_idx < 3:
                            logger.info(f"[DoremiMixer] Batch {batch_idx} - Proxy loss: {proxy_loss:.4f}, "
                                       f"Ref loss: {ref_loss:.4f}, Tokens: {proxy_tokens}")
                        
                        # Compute excess loss (difference in average losses)
                        # Following DoReMi: clip at 0 (non-negative loss)
                        excess_loss = max(0.0, proxy_loss - ref_loss)
                    else:
                        # If no reference model, just use proxy loss
                        excess_loss = proxy_loss
                        if batch_idx < 3:
                            logger.info(f"[DoremiMixer] Batch {batch_idx} - Proxy loss: {proxy_loss:.4f}, "
                                       f"Tokens: {proxy_tokens} (no reference model)")
                    
                    # Accumulate weighted by number of tokens
                    batch_excess_loss = excess_loss * proxy_tokens
                    excess_losses.append(batch_excess_loss)
                    total_tokens += proxy_tokens
                        
                except Exception as e:
                    logger.warning(f"[DoremiMixer] Error processing batch {batch_idx} for domain '{domain_name}': {e}")
                    continue
        
        # Compute average excess loss
        if total_tokens > 0:
            avg_excess_loss = sum(excess_losses) / total_tokens
            logger.info(f"[DoremiMixer] Domain '{domain_name}' - Processed {len(excess_losses)} batches, "
                       f"{total_tokens} tokens, avg_excess_loss={avg_excess_loss:.4f}")
        else:
            avg_excess_loss = 0.0
            logger.warning(f"[DoremiMixer] Domain '{domain_name}' - No valid tokens processed!")
        
        return avg_excess_loss
    
    def _update_domain_weights(self, perdomain_scores):
        """
        Update domain weights using the DoReMi v1 algorithm.
        
        Args:
            perdomain_scores: Array of per-domain excess losses
            
        Returns:
            Updated domain weights (normalized)
        """
        k = len(self.domain_weights)
        
        logger.info(f"[DoremiMixer] Updating weights: eta={self.reweight_eta} (type: {type(self.reweight_eta)}), "
                   f"eps={self.reweight_eps} (type: {type(self.reweight_eps)}), k={k}")
        logger.info(f"[DoremiMixer] Input scores: {perdomain_scores}")
        logger.info(f"[DoremiMixer] Current weights: {self.domain_weights}")
        
        # Exponentiated gradient ascent update
        log_new_weights = np.log(self.domain_weights) + self.reweight_eta * perdomain_scores
        
        # Normalize in log space
        log_new_weights = log_new_weights - np.log(np.sum(np.exp(log_new_weights)))
        
        # Convert back from log space and apply smoothing
        new_weights = (1 - self.reweight_eps) * np.exp(log_new_weights) + self.reweight_eps / k
        
        # Final normalization (should already be normalized, but ensure numerical stability)
        new_weights = new_weights / new_weights.sum()
        
        return new_weights
    
    def mix(self, model, step_id: int, **kwargs) -> np.ndarray:
        """
        Compute new domain weights using DoReMi algorithm.
        
        The DoReMi algorithm:
        1. Compute excess loss for each domain: excess_loss = proxy_loss - reference_loss
        2. Update domain weights via exponentiated gradient ascent
        3. Return normalized weights
        
        Args:
            model: Current proxy model being trained
            step_id: Current training step
            **kwargs: Additional arguments
            
        Returns:
            np.ndarray: Updated domain proportions (normalized)
        """
        logger.info(f"[DoremiMixer] Starting domain weight update at step {step_id}")
        
        # Load reference model if not already loaded
        self._load_reference_model(model)
        
        # If reference model loading failed, use proxy model loss only
        if self.reference_model is None:
            logger.info(f"[DoremiMixer] No reference model available, using proxy model loss only for domain reweighting")
        
        # Evaluate excess loss for each domain
        k = len(self.mixture_manager.names)
        perdomain_scores = []
        
        logger.info(f"[DoremiMixer] Evaluating {k} domains...")
        
        for domain_id in range(k):
            domain_name = self.mixture_manager.names[domain_id]
            
            try:
                avg_excess_loss = self._evaluate_domain_losses(model, domain_id)
                # Clip to avoid extreme values
                avg_excess_loss = max(0.0, avg_excess_loss)
                perdomain_scores.append(avg_excess_loss)
                
                logger.info(f"[DoremiMixer] Domain '{domain_name}': excess_loss={avg_excess_loss:.4f}")
                
            except Exception as e:
                logger.warning(f"[DoremiMixer] Failed to evaluate domain '{domain_name}': {e}. "
                             f"Using previous score.")
                # Use previous score if evaluation fails
                perdomain_scores.append(self.perdomain_scores[domain_id])
        
        # Update stored scores
        self.perdomain_scores = np.array(perdomain_scores)
        
        # Update domain weights
        self.domain_weights = self._update_domain_weights(self.perdomain_scores)
        
        # Log updated weights
        logger.info(f"[DoremiMixer] Step {step_id} Updated domain weights:")
        for i, name in enumerate(self.mixture_manager.names):
            logger.info(f"  {name}: {self.domain_weights[i]:.4f} (score: {self.perdomain_scores[i]:.4f})")
        
        return self.domain_weights
    
    def save_weights(self, output_dir, step_id):
        """Save current domain weights to file."""
        try:
            import json
            weights_file = os.path.join(output_dir, f"doremi_weights_step_{step_id}.json")
            
            weights_data = {
                "step": step_id,
                "domain_names": self.mixture_manager.names,
                "domain_weights": self.domain_weights.tolist(),
                "perdomain_scores": self.perdomain_scores.tolist(),
                "reweight_eta": self.reweight_eta,
                "reweight_eps": self.reweight_eps
            }
            
            os.makedirs(output_dir, exist_ok=True)
            with open(weights_file, 'w') as f:
                json.dump(weights_data, f, indent=2)
                
            logger.info(f"[DoremiMixer] Saved domain weights to {weights_file}")
            
        except Exception as e:
            logger.warning(f"[DoremiMixer] Failed to save domain weights: {e}")
    
    def load_weights(self, weights_file):
        """Load domain weights from file."""
        try:
            import json
            with open(weights_file, 'r') as f:
                weights_data = json.load(f)
            
            self.domain_weights = np.array(weights_data["domain_weights"])
            self.perdomain_scores = np.array(weights_data["perdomain_scores"])
            
            logger.info(f"[DoremiMixer] Loaded domain weights from {weights_file}")
            return True
            
        except Exception as e:
            logger.warning(f"[DoremiMixer] Failed to load domain weights: {e}")
            return False

