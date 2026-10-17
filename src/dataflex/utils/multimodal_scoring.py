"""Answer-only scoring and shared model/device state helpers."""

from collections.abc import Mapping
from contextlib import contextmanager
from inspect import unwrap

import torch
import torch.nn.functional as F


def move_to_device(value, device):
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(move_to_device(item, device) for item in value)
    return value.to(device) if torch.is_tensor(value) else value


@contextmanager
def scoring_state(model, checkpoint_llama=False):
    """Restore every module's mode and RNG; autograd.grad never writes .grad."""
    modes = [(module, module.training) for module in model.modules()]
    rope_state = [(module, module.rope_deltas) for module in model.modules() if hasattr(module, "rope_deltas")]
    devices = sorted({p.device.index for p in model.parameters() if p.device.type == "cuda"})
    with torch.random.fork_rng(devices=devices):
        try:
            model.eval()
            if checkpoint_llama:
                from transformers.models.llama.modeling_llama import LlamaDecoderLayer

                # LlamaDecoderLayer uses this flag for checkpointing; attention
                # and LoRA dropout are child modules and remain in eval mode.
                # Non-reentrant checkpointing supports autograd.grad.
                for module in model.modules():
                    if isinstance(module, LlamaDecoderLayer) and getattr(module, "gradient_checkpointing", False):
                        function = unwrap(module._gradient_checkpointing_func)
                        if getattr(function, "keywords", {}).get("use_reentrant", True):
                            continue  # retain ordinary eval scoring for older training configurations
                        module.training = True
            yield
        finally:
            for module, training in modes:
                module.training = training
            for module, value in rope_state:
                module.rope_deltas = value


def answer_statistics(logits, labels, token_chunk_size=128):
    """Differentiable mean CE plus token-mean EL2N and predictive entropy.

    Only labels != -100 are scored after the causal shift. Unsupported expanded
    vision-token layouts fail instead of silently aligning logits incorrectly.
    """
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels must have aligned [batch, sequence] dimensions")
    if logits.shape[0] != 1:
        raise ValueError("per-example gradient scoring requires batch size 1")
    active = labels[0, 1:] != -100
    targets = labels[0, 1:][active].long().to(logits.device)
    positions = torch.nonzero(active.to(logits.device), as_tuple=True)[0]
    if not len(targets):
        raise ValueError("sample has no supervised answer tokens after causal shift")
    losses, el2n, entropy = [], [], []
    for start in range(0, len(targets), token_chunk_size):
        pred = logits[0, positions[start : start + token_chunk_size]].float()
        target = targets[start : start + token_chunk_size]
        losses.append(F.cross_entropy(pred, target, reduction="sum"))
        with torch.no_grad():
            log_prob = pred.log_softmax(-1)
            prob = log_prob.exp()
            target_prob = prob.gather(1, target[:, None]).squeeze(1)
            error = (prob.square().sum(-1) - 2 * target_prob + 1).clamp_min(0).sqrt()
            el2n.append(error.sum())
            entropy.append(-(prob * log_prob).sum())
    count = len(targets)
    return {
        "loss": torch.stack(losses).sum() / count,
        "el2n": float(torch.stack(el2n).sum() / count),
        "entropy": float(torch.stack(entropy).sum() / count),
        "answer_tokens": count,
    }


def assert_same_answers(first, second):
    first = torch.as_tensor(first).flatten()
    second = torch.as_tensor(second).flatten()
    if not torch.equal(first[first != -100].cpu(), second[second != -100].cpu()):
        raise ValueError("paired conditions do not supervise exactly the same response tokens")
