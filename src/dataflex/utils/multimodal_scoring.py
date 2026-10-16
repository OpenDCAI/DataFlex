"""Answer-only scoring and explicit Qwen-VL/LLaVA image ablation adapters."""

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


def qwen_text_only_sample(sample, tokenizer):
    """Remove Qwen image spans before re-collation; never send orphan image tokens.

    This adapter is intentionally restricted to image-only Qwen-VL examples.
    Other templates need an explicit adapter, not zeroed pixels or images=None.
    """
    if not sample.get("images") or sample.get("videos") or sample.get("audios"):
        raise ValueError("IG requires an image-only example; text/video/audio examples are unsupported")
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    vocab = tokenizer.get_vocab()
    tokens = ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>"]
    if not all(token in vocab for token in tokens):
        raise ValueError("IG currently supports Qwen-VL image spans; supply a model-specific ablation adapter")
    start_id, end_id, image_id = [vocab[token] for token in tokens]
    ids, labels = list(sample["input_ids"]), list(sample["labels"])
    keep, inside, image_seen, spans = [], False, False, 0
    for i, token in enumerate(ids):
        if token == start_id:
            if inside:
                raise ValueError("nested vision spans")
            inside, image_seen = True, False
        elif token == end_id:
            if not inside or not image_seen:
                raise ValueError("malformed or non-image vision span")
            inside = False
            spans += 1
        elif inside:
            image_seen |= token == image_id
        else:
            if token == image_id:
                raise ValueError("image token outside vision span")
            keep.append(i)
        if inside or token in (start_id, end_id):
            if labels[i] != -100:
                raise ValueError("vision tokens must not be supervised")
    if inside or spans != len(sample["images"]):
        raise ValueError("image count and vision spans do not match")
    result = {"input_ids": [ids[i] for i in keep], "labels": [labels[i] for i in keep], "images": []}
    if "attention_mask" in sample:
        result["attention_mask"] = [sample["attention_mask"][i] for i in keep]
    assert_same_answers(labels, result["labels"])
    return result


def image_text_only_sample(sample, tokenizer, model_type=None):
    """Remove image placeholders using the model's actual token layout.

    LLaVA-1.5 expands each image into a consecutive run of <image> tokens.
    These are removed entirely, preserving all response labels. Other LLaVA
    families are deliberately not inferred from the shared placeholder name.
    """
    if model_type != "llava":
        return qwen_text_only_sample(sample, tokenizer)
    if not sample.get("images") or sample.get("videos") or sample.get("audios"):
        raise ValueError("IG requires an image-only LLaVA example")
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    image_id = tokenizer.get_vocab().get("<image>")
    if image_id is None:
        raise ValueError("LLaVA IG requires the registered <image> token")
    ids, labels = list(sample["input_ids"]), list(sample["labels"])
    keep, spans, previous_image = [], 0, False
    for i, token in enumerate(ids):
        is_image = token == image_id
        if is_image:
            if labels[i] != -100:
                raise ValueError("vision tokens must not be supervised")
            spans += not previous_image
        else:
            keep.append(i)
        previous_image = is_image
    if spans != len(sample["images"]):
        raise ValueError("image count and LLaVA token spans do not match")
    result = {"input_ids": [ids[i] for i in keep], "labels": [labels[i] for i in keep], "images": []}
    if "attention_mask" in sample:
        result["attention_mask"] = [sample["attention_mask"][i] for i in keep]
    assert_same_answers(labels, result["labels"])
    return result
