from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Union
from torch import nn
import torch


class Weighter(ABC):
    """
    Abstract base class for data weighting, defining the basic interface and common functionality.
    """
    
    def __init__(self, **kwargs):
        """
        Base class constructor
        
        Args:
            **kwargs: Subclass-specific parameters
        """
        # Subclasses can define common initialization logic here
        pass

    def prepare(self, ctx, model):
        """Refresh optional component state before the training forward."""
        return None
    
    def _per_sample_loss_from_logits(self, logits, labels, ignore_index: int = -100):
        """
        Calculate the per-sample loss from logits and labels
        
        Args:
            logits: Model output logits
            labels: True labels
            ignore_index: Ignored label index
            
        Returns:
            torch.Tensor: The per-sample loss (B,)
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        num_active = (shift_labels != ignore_index).sum(dim=1)  # (B,)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        tok_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1).long()
        )
        return tok_loss.view(shift_logits.size(0), -1).sum(dim=1) / torch.clamp(num_active, min=1)
    
    @abstractmethod
    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        *,
        ctx: Any = None,
        model: nn.Module | None = None,
        inputs: dict[str, Union[torch.Tensor, Any]] | None = None,
    ) -> torch.Tensor:
        """
        Core weighting method, subclasses must implement this method
        
        Args:
            losses: Per-sample loss on this card (B,)
            ctx: Trainer context, can get global_step information
            model: Current model
            inputs: Input data
            
        Returns:
            torch.Tensor: The weighted total loss (scalar)
        """
        pass
    
    def training_step(self, ctx, model, inputs, num_items_in_batch=None, use_weighter=False):
        """
        Execute the training step, including forward propagation, loss calculation, weighting, and backpropagation
        
        Args:
            ctx: Trainer context
            model: Model
            inputs: Input data
            num_items_in_batch: Number of samples in the batch
            use_weighter: Whether to use the weighter
            
        Returns:
            The loss value for this step
        """
        from dataflex.utils.logging import logger
        from transformers.utils import is_apex_available
        from accelerate.utils import DistributedType

        if use_weighter:
            self.prepare(component_context(ctx, model=model, batch=inputs), model)
        
        model.train()
        if hasattr(ctx.optimizer, "train") and callable(ctx.optimizer.train):
            ctx.optimizer.train()

        inputs = ctx._prepare_inputs(inputs)

        # Save a copy of labels (to prevent them from being popped in some implementations)
        labels_for_weighter = inputs.get("labels", None)

        with ctx.compute_loss_context_manager():
            # Key: get the outputs
            loss, outputs = ctx.compute_loss(
                model, inputs, num_items_in_batch=num_items_in_batch, return_outputs=True
            )

        # Whether the model's scalar loss really got replaced by our own per-sample one.
        # If it did, the model's per-token normalization no longer applies and this
        # method has to do the gradient accumulation scaling itself.
        reweighted = False

        if use_weighter:
            # 1) If compute_loss already returned a (B,) vector, use it directly
            if torch.is_tensor(loss) and loss.dim() == 1:
                per_sample = loss
            else:
                # 2) Otherwise, calculate the per-sample loss from logits and labels (no need for second forward pass)
                logits = getattr(outputs, "logits", None) if outputs is not None else None
                labels = inputs.get("labels", None)
                if labels is None:
                    labels = labels_for_weighter
                per_sample = None
                if logits is not None and labels is not None:
                    per_sample = self._per_sample_loss_from_logits(logits, labels)

            if per_sample is not None:
                # Log only from the main process
                if ctx.args.local_rank in [-1, 0]:
                    ps = per_sample.detach().float().cpu().view(-1)[0]
                    logger.info(f"[Dataflex] Before weighting per-sample (first sample): {ps}")
                # Distributed weighting
                with ctx.compute_loss_context_manager():
                    loss = self.get_weighted_loss(per_sample, ctx=ctx, model=model, inputs=inputs)
                reweighted = True
                if ctx.args.local_rank in [-1, 0]:
                    logger.info(f"[Dataflex] After weighting (first sample): {float(loss.detach().cpu())}")
            else:
                if ctx.args.local_rank in [-1, 0]:
                    logger.info("[Dataflex] Could not form per-sample losses; fallback to scalar loss (no reweight).")

        del inputs

        if ctx.args.torch_empty_cache_steps is not None and ctx.state.global_step % ctx.args.torch_empty_cache_steps == 0:
            ctx._empty_cache()

        kwargs = {}
        if ctx.args.n_gpu > 1:
            loss = loss.mean()

        if getattr(ctx, "use_apex", False):
            if is_apex_available():
                from apex import amp
                with amp.scale_loss(loss, ctx.optimizer) as scaled_loss:
                    scaled_loss.backward()
        else:
            # Gradient accumulation scaling, aligned with upstream Trainer.training_step.
            #
            # Upstream's rule: skip the division when the model already normalized by
            # num_items_in_batch, otherwise divide by the group's *actual* micro-batch
            # count. This used to divide unconditionally, and by the static
            # args.gradient_accumulation_steps, so an already-normalized loss got
            # divided twice and a short final group got divided too much.
            #
            # Reweighting is a third case: the per-sample loss is recomputed here from
            # logits, so the model's normalization does not apply and we must scale.
            if reweighted or (
                (not getattr(ctx, "model_accepts_loss_kwargs", False) or num_items_in_batch is None)
                and getattr(ctx, "compute_loss_func", None) is None
            ):
                loss = loss / getattr(
                    ctx, "current_gradient_accumulation_steps", ctx.args.gradient_accumulation_steps
                )
            if ctx.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False
            ctx.accelerator.backward(loss, **kwargs)

        return loss.detach()


def component_context(ctx, *, model=None, batch=None):
    if hasattr(ctx, "step_id"):
        return ctx
    # Lazy import: importing LEGO also registers the component families.
    from dataflex.train.lego.context import PlanContext

    return PlanContext.from_trainer(ctx, model=model, batch=batch)


@contextmanager
def scoring_mode(model):
    """Restore even mixed submodule training modes after scoring or an error."""
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, training in modes:
            module.training = training


class IndexedWeightLifecycle:
    """Periodic full-dataset refresh, followed by lookup in processed-row space."""

    def set_score_source(self, source):
        """LEGO supplies raw scores; standalone weighters retain their scorer."""
        self._score_source = source

    def _source_scores(self, metric):
        source = getattr(self, "_score_source", None)
        if source is None:
            return None
        scores = torch.as_tensor(source(range(len(self.dataset)), metric=metric), dtype=torch.float32)
        if scores.shape != (len(self.dataset),):
            raise ValueError("Score source does not match the global weight table")
        return scores

    def prepare(self, ctx, model):
        ctx = component_context(ctx, model=model)
        if self.refresh_interval <= 0:
            raise ValueError("refresh_interval must be positive")
        step = ctx.step_id
        if self.W is None or (step != self._last_refresh_step and step % self.refresh_interval == 0):
            if not len(self.dataset):
                raise ValueError("Cannot refresh weights on an empty training dataset")
            self._uses_sample_ids = "idx" in self.dataset[0]
            with scoring_mode(model), ctx.accelerator.autocast():
                self._refresh(ctx, model, step)
            self._last_refresh_step = step

    def _batch_weights(self, inputs, losses, key_fn):
        if self.W is None:
            raise RuntimeError("Weights are uninitialized; call prepare() before the training forward")
        if "idx" in inputs:
            indices = torch.as_tensor(inputs["idx"], dtype=torch.long).detach().cpu().view(-1)
            if len(indices) != losses.numel() or (indices < 0).any() or (indices >= len(self.W)).any():
                raise ValueError("Batch idx does not match the global weight table")
            return self.W[indices].to(device=losses.device, dtype=losses.dtype)
        # Compatibility for standalone WeightTrainer datasets without row IDs.
        if getattr(self, "_uses_sample_ids", False):
            raise ValueError("Batch idx is required for weights scored in global sample-ID space")
        return torch.tensor(
            [self._weight_map.get(k, 1.0) for k in key_fn(inputs)], dtype=losses.dtype, device=losses.device
        )

    def _get_length_order(self):
        if self._order is None:
            # Works on both HF datasets and DomainLabeledDataset/Subset, without
            # forwarding HF .map() through a wrapper and losing ID alignment.
            lengths = [len(self.dataset[i]["input_ids"]) for i in range(len(self.dataset))]
            self._order = torch.tensor(lengths).argsort(descending=True, stable=True).tolist()
        return self._order

    def state_dict(self):
        return {
            "weights": self.W,
            "last_refresh_step": self._last_refresh_step,
            "weight_map": self._weight_map,
            "uses_sample_ids": getattr(self, "_uses_sample_ids", False),
        }

    def load_state_dict(self, state):
        weights = state["weights"]
        if weights is not None:
            if weights.shape != (len(self.dataset),) or not torch.isfinite(weights).all():
                raise ValueError("Checkpoint weights do not match the training dataset")
            weights = weights.detach().cpu().clone()
        self.W = weights
        self._weight_map = dict(state["weight_map"])
        self._last_refresh_step = int(state["last_refresh_step"])
        self._uses_sample_ids = bool(state["uses_sample_ids"])
