from dataflex.core.registry import register_mixer
from dataflex.utils.logging import logger
from .base_mixer import Mixer
import numpy as np
import torch
import os
from contextlib import nullcontext

@register_mixer("doremi")
class DoremiMixer(Mixer):
    def __init__(self, mixture_manager, reference_model_path=None, reweight_eta=1.0, reweight_eps=1e-3, 
                 accelerator=None, output_dir=None, dataset=None, data_collator=None, **kwargs):
        super().__init__(mixture_manager)
        self.reference_model_path = reference_model_path
        self.reweight_eta = float(reweight_eta)
        self.reweight_eps = float(reweight_eps)
        self.accelerator = accelerator
        self.output_dir = output_dir
        self.dataset = dataset
        self.data_collator = data_collator
        
        k = len(self.mixture_manager.names)
        self.domain_weights = np.ones(k) / k  # Algorithm 1 line 26
        self.perdomain_scores = np.zeros(k)
        self.weight_history = []
        self.step_history = []
        
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
        
        logger.info(f"[DoremiMixer] Init: k={k}, η={self.reweight_eta}, c={self.reweight_eps}")
    
    def _load_reference_model(self, proxy_model):
        if self.reference_model_loaded:
            return
        
        if self.reference_model_path is None or not os.path.exists(self.reference_model_path):
            self.reference_model_loaded = True
            return
        
        try:
            from transformers import AutoModelForCausalLM
            self.reference_model = AutoModelForCausalLM.from_pretrained(
                self.reference_model_path,
                torch_dtype=proxy_model.dtype if hasattr(proxy_model, 'dtype') else torch.float32,
                trust_remote_code=True
            )
            device = next(proxy_model.parameters()).device
            self.reference_model = self.reference_model.to(device)
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
            self.reference_model = None
            self.reference_model_loaded = True
    
    def _prepare_model_for_eval(self, model, log_info=False):
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
    
    def _compute_per_token_loss(self, model, batch):
        # Algorithm 1 line 31: compute ℓ_{θ,j}(x) for each token j
        with torch.no_grad():
            was_training = model.training
            model.eval()
            
            input_ids = batch['input_ids']
            labels = batch.get('labels')
            attention_mask = batch.get('attention_mask', torch.ones_like(input_ids))
            
            try:
                # Prepare model for evaluation (handles DeepSpeed ZeRO-3)
                eval_model, param_context = self._prepare_model_for_eval(model, log_info=False)
                
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
                logits = outputs.logits
                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                vocab_size = shift_logits.size(-1)
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
                per_token_loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1)).view(shift_labels.size())
                
                valid_mask = (shift_labels != -100)
                
                if was_training:
                    model.train()
                
                return per_token_loss, valid_mask
            except Exception as e:
                if was_training:
                    model.train()
                logger.error(f"[DoremiMixer] Error in _compute_per_token_loss: {e}")
                raise
    
    def compute_batch_excess_losses(self, proxy_model, batch, domain_ids):
        # Algorithm 1 line 30-31: λ_t[i] ← (1/Σ|x|) · Σ Σ_j max{ℓ_θ,j - ℓ_ref,j, 0}
        k = len(self.mixture_manager.names)
        perdomain_scores = np.zeros(k)
        domain_has_data = np.zeros(k, dtype=bool)
        
        if not self.reference_model_loaded:
            self._load_reference_model(proxy_model)
        
        with torch.no_grad():
            device = next(proxy_model.parameters()).device
            batch = {key: val.to(device) if isinstance(val, torch.Tensor) else val for key, val in batch.items()}
            domain_ids = domain_ids.to(device)
            
            proxy_token_losses, valid_mask = self._compute_per_token_loss(proxy_model, batch)
            
            if self.reference_model is not None:
                ref_token_losses, _ = self._compute_per_token_loss(self.reference_model, batch)
                # CRITICAL: clip at token level THEN average
                excess_token_losses = torch.clamp(proxy_token_losses - ref_token_losses, min=0.0)
            else:
                excess_token_losses = proxy_token_losses
            
            for domain_id in range(k):
                domain_mask = (domain_ids == domain_id)
                if domain_mask.sum() > 0:
                    domain_mask_expanded = domain_mask.unsqueeze(1).expand_as(excess_token_losses)
                    domain_valid_mask = valid_mask & domain_mask_expanded
                    
                    domain_excess_sum = (excess_token_losses * domain_valid_mask.float()).sum().item()
                    domain_token_count = domain_valid_mask.sum().item()
                    
                    if domain_token_count > 0:
                        domain_has_data[domain_id] = True
                        perdomain_scores[domain_id] = domain_excess_sum / domain_token_count
        
        return perdomain_scores, domain_has_data
    
    
    def _update_domain_weights(self, perdomain_scores, domain_has_data):
        # Algorithm 1 lines 32-33
        k = len(self.domain_weights)
        # alpha_prime = self.domain_weights * np.exp(self.reweight_eta * perdomain_scores)
        scores = perdomain_scores.copy()
        valid = domain_has_data
        if valid.any():
            mu = scores[valid].mean()
            scores[valid] = scores[valid] - mu
        scores = np.clip(scores, -5.0, 5.0)
        alpha_prime = self.domain_weights * np.exp(self.reweight_eta * scores)
        
        u = 1.0 / k
        new_weights = (1 - self.reweight_eps) * (alpha_prime / alpha_prime.sum()) + self.reweight_eps * u
        return new_weights / new_weights.sum()
    
    def mix(self, model, step_id: int, **kwargs) -> np.ndarray:
        # Algorithm 1: Use current training batch to compute λ_t, update α_t, store for averaging
        batch = kwargs.get('batch')
        domain_ids = kwargs.get('domain_ids')
        output_dir = kwargs.get('output_dir', self.output_dir)
        
        if batch is None or domain_ids is None:
            raise ValueError(
                "[DoremiMixer] Algorithm 1 requires current training batch. "
                "batch and domain_ids must be provided to mix() method."
            )
        
        if not self.reference_model_loaded:
            self._load_reference_model(model)
        
        # Algorithm 1 line 30-31: Compute λ_t
        perdomain_scores, domain_has_data = self.compute_batch_excess_losses(model, batch, domain_ids)
        self.perdomain_scores = perdomain_scores
        
        # Algorithm 1 lines 32-33: Update α_t
        self.domain_weights = self._update_domain_weights(perdomain_scores, domain_has_data)
        
        # Algorithm 1 line 36: Store for averaging
        self.weight_history.append(self.domain_weights.copy())
        self.step_history.append(step_id)
        
        logger.info(f"[DoremiMixer] Step {step_id} - α_t: {self.domain_weights}, λ_t: {perdomain_scores}")
        
        if output_dir:
            self.save_weights_to_jsonl(output_dir, step_id)
        
        # Algorithm 1: Return uniform weights for sampling (u = 1/k)
        # The domain_weights α_t will be used for loss reweighting in training_step
        k = len(self.domain_weights)
        uniform_weights = np.ones(k) / k
        logger.info(f"[DoremiMixer] Returning uniform weights for sampling: {uniform_weights}")
        return uniform_weights
    
    def get_current_doremi_weights(self) -> np.ndarray:
        return self.domain_weights.copy()
    
    def save_weights_to_jsonl(self, output_dir, step_id):
        if self.accelerator is not None and not self.accelerator.is_main_process:
            return
        
        try:
            import json, time
            os.makedirs(output_dir, exist_ok=True)
            weights_file = os.path.join(output_dir, "doremi_weights.jsonl")
            
            log_entry = {
                "step": step_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "domain_names": self.mixture_manager.names,
                "domain_weights": self.domain_weights.tolist(),
                "perdomain_scores": self.perdomain_scores.tolist(),
            }
            
            with open(weights_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.warning(f"[DoremiMixer] Failed to log weights: {e}")
    
