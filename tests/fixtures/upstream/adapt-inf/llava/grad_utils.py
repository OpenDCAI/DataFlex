"""
Adapted from https://github.com/princeton-nlp/LESS/blob/main/less/data_selection/collect_grad_reps.py#L90
"""

import json
import os, sys, re
from hashlib import md5
from typing import Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
# from functorch import grad, make_functional_with_buffers, vmap
from peft import PeftModel
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn.functional import normalize
from tqdm import tqdm
from trak.projectors import BasicProjector, CudaProjector, ProjectionType
from transformers import RobertaModel


def get_shifted_logits_labels(batch, outputs):

    batch_size = batch['input_ids'].shape[0]
    target_length = batch['input_ids'].shape[1]
    outputs['logits'] = outputs['logits'][:, -target_length:, :]

    # print(batch['input_ids'].shape, outputs['logits'].shape)

    # Shift so that tokens < n predict n
    if batch['attention_mask'] is not None:
        shift_attention_mask = batch['attention_mask'][..., 1:]
        if batch_size == 1:
            # print(outputs['logits'].shape, shift_attention_mask.shape)
            shift_logits = outputs['logits'][..., :-1, :][shift_attention_mask.to(outputs['logits'].device) != 0].contiguous()
            shift_labels = batch['labels'][..., 1:][shift_attention_mask.to(batch['labels'].device) != 0].contiguous()
            # print('attention', shift_attention_mask[0, :], 'logits', shift_logits[0], 'labels', shift_labels[0])
        else:
            shift_logits = [outputs['logits'][i, :-1, :][shift_attention_mask[i].to(outputs['logits'].device) != 0].contiguous() for i in range(batch_size)]
            shift_labels = [batch['labels'][i, 1:][shift_attention_mask[i].to(outputs['logits'].device) != 0].contiguous() for i in range(batch_size)]
            # print('attention', shift_attention_mask[0, :], 'logits', shift_logits[0][0], 'labels', shift_labels[0])

    else:
        shift_logits = outputs['logits'][..., :-1, :].contiguous()
        shift_labels = batch['labels'][..., 1:].contiguous()

    return shift_logits, shift_labels


def calculate_cross_entropy_loss(labels, logits):

    vocab_size = logits.shape[-1]
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss


def prepare_batch(batch, device=torch.device("cuda:0")):
    """ Move the batch to the device. """
    for key in batch:
        if key == 'images':
            batch[key] = batch[key].half().to(device)
        batch[key] = batch[key].to(device)


def get_max_saved_index(output_dir: str, prefix="reps") -> int:
    """ 
    Retrieve the highest index for which the data (either representation or gradients) has been stored. 

    Args:
        output_dir (str): The output directory.
        prefix (str, optional): The prefix of the files, [reps | grads]. Defaults to "reps".

    Returns:
        int: The maximum representation index, or -1 if no index is found.
    """

    files = [file for file in os.listdir(
        output_dir) if file.startswith(prefix)]
    index = [int(file.split(".")[0].split("-")[1])
             for file in files]  # e.g., output_dir/reps-100.pt
    return max(index) if len(index) > 0 else -1


def get_output(model,
               weights: Iterable[Tensor],
               buffers: Iterable[Tensor],
               input_ids=None,
               attention_mask=None,
               labels=None,
               ) -> Tensor:
    logits = model(weights, buffers, *(input_ids.unsqueeze(0),
                   attention_mask.unsqueeze(0))).logits
    labels = labels.unsqueeze(0)
    loss_fct = F.cross_entropy
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))
    return loss


def get_trak_projector(device: torch.device):
    """ Get trak projectors (see https://github.com/MadryLab/trak for details) """
    try:
        num_sms = torch.cuda.get_device_properties(
            device.index).multi_processor_count
        import fast_jl

        # test run to catch at init time if projection goes through
        fast_jl.project_rademacher_8(torch.zeros(
            8, 1_000, device=device), 512, 0, num_sms)
        projector = CudaProjector
        print("Using CudaProjector")
    except:
        projector = BasicProjector
        print("Using BasicProjector")
    return projector


def get_number_of_params(model, per_layer=False): # TODO: better way of calculating parameters per layer
    """ Make sure that only lora parameters require gradients in peft models. """
    if isinstance(model, PeftModel):
        names = [n for n, p in model.named_parameters(
        ) if p.requires_grad and "lora" not in n]
        assert len(names) == 0
    num_params = sum([p.numel()
                     for p in model.parameters() if p.requires_grad])
    total_params = sum([p.numel()
                     for p in model.parameters()])
    
    print(f"Total number of parameters: {total_params}")
    print(f"Total number of parameters that require gradients: {num_params}")

    if per_layer:
        num_params_per_layer = sum([p.numel()
                     for n, p in model.named_parameters() if p.requires_grad and 'layers.0.' in n])
        print(f"Total number of parameters that require gradients per layer: {num_params_per_layer}")
        return num_params_per_layer
    else:
        return num_params


def print_params_req_grad(model):
    """ Enlist the parameters that require grads and print their size. """
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n, ': ', p.shape)


def get_layer_dim_from_param_name(param_name):

    result = re.search(r"(\d+)", param_name)
    if result is not None:
        assert len(result.groups()) == 1, "Found more than one match for layer dimension in param name"
        return result[0]
    else:
        return None


def obtain_gradients(model, layer_dim):
    """ obtain gradients. """
    # loss = model(**batch).loss
    # loss.backward()
    vectorized_grads = {}
    layer_dim = [str(layer) for layer in layer_dim]
    for layer in layer_dim:
        vectorized_grads[int(layer)] = torch.cat(
            [p.grad.view(-1) for n, p in model.named_parameters() if p.grad is not None and get_layer_dim_from_param_name(n) == layer]).detach().clone()
    
    # vectorized_grads = torch.cat(
    #     [p.grad.view(-1) for p in model.parameters() if p.grad is not None])
    return vectorized_grads


def obtain_sign_gradients(model, layer_dim):
    """ obtain gradients with sign. """
    # loss = model(**batch).loss
    # loss.backward()

    vectorized_grad_signs = {}
    layer_dim = [str(layer) for layer in layer_dim]
    for layer in layer_dim:
        vectorized_grad_signs[int(layer)] = torch.cat(
            [torch.sign(p.grad).view(-1) for n, p in model.named_parameters() if p.grad is not None and get_layer_dim_from_param_name(n) == layer]).detach().clone()

    # Instead of concatenating the gradients, concatenate their signs
    # vectorized_grad_signs = torch.cat(
    #     [torch.sign(p.grad).view(-1) for p in model.parameters() if p.grad is not None])

    return vectorized_grad_signs


def obtain_gradients_with_adam(model, avg, avg_sq):
    """ obtain gradients with adam optimizer states. """
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-08

    # loss = model(**batch).loss
    # loss.backward()

    vectorized_grad_adam = {}
    layer_dim = [str(layer) for layer in layer_dim]
    for layer in layer_dim:
        vectorized_grads = torch.cat(
            [p.grad.view(-1) for n, p in model.named_parameters() if p.grad is not None and get_layer_dim_from_param_name(n) == layer])

        updated_avg = beta1 * avg + (1 - beta1) * vectorized_grads
        updated_avg_sq = beta2 * avg_sq + (1 - beta2) * vectorized_grads ** 2
        vectorized_grads = updated_avg / torch.sqrt(updated_avg_sq + eps)
        vectorized_grad_adam[int(layer)] = vectorized_grads.detach().clone()

    return vectorized_grads


def obtain_perplexity_score(shift_logits, shift_labels):
# def obtain_perplexity_score(outputs, eval_batch_size, shift_logits=None, shift_labels=None):
    # if eval_batch_size == 1:
    #     return torch.exp(outputs.loss).detach().cpu().item()
    # else:
    assert shift_logits is not None and shift_labels is not None
    indices = torch.where(shift_labels != -100)[0]
    shift_labels = shift_labels[indices]
    shift_logits = shift_logits[indices]
    loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none')

    # Calculate perplexity per sample
    if shift_labels.size(0) == 0:
        return torch.exp(loss.mean())
    else:
        sample_perplexities = torch.exp(loss.view(shift_labels.size(0), -1).mean(dim=1)).mean().squeeze()
        # print('ppl', indices.shape, shift_logits.shape, shift_labels.shape, loss.shape, sample_perplexities.shape)
        return sample_perplexities


def obtain_el2n_score(shift_logits, shift_labels, eval_batch_size):
    # L2 Norm of the error vector, averaged over all tokens to adapt for sequences (see https://arxiv.org/pdf/2403.09559)

    # print(type(shift_logits), type(shift_labels))
    # print([label.shape for label in shift_labels], [logit.shape for logit in shift_logits])
    # print(shift_logits.shape)
    # vocab_size = shift_logits.shape[-1]
    # indices = torch.where(shift_labels != -100)[0]
    # print(indices.shape)
    # probs = torch.nn.Softmax(dim=-1)(shift_logits)[indices]
    # print(probs.shape)
    # # print(probs)
    # # print(shift_labels)
    
    # label_onehot = torch.nn.functional.one_hot(shift_labels[indices], num_classes=vocab_size).to(probs.device)
    
    if eval_batch_size == 1:
        assert type(shift_logits) != list
        vocab_size = shift_logits.shape[-1]
        indices = torch.where(shift_labels != -100)[0]
        probs = torch.nn.Softmax(dim=-1)(shift_logits)[indices]
        # print('Probabilities', probs, probs.sum(dim=-1))
        label_onehot = torch.nn.functional.one_hot(shift_labels[indices], num_classes=vocab_size).to(probs.device)
        # print(probs.shape, label_onehot.shape)
        # _ = label_onehot - probs
        # l2_values = torch.nn.functional.mse_loss(label_onehot, probs, reduction='none')
        # print(label_onehot, probs)
        l2_values = torch.pow(label_onehot-probs, 2)
        # print(l2_values)
        l2_values = l2_values.sum(dim=1)
        # print(l2_values)
        # print(l2_values.shape)
        # print(l2_values)
        l2_values = torch.sqrt(l2_values)
        # print(l2_values)
        el2n_score = torch.mean(l2_values).detach().cpu()
        # print('el2n', el2n_score.shape, el2n_score)
        # print(el2n_score)
    else:
        assert type(shift_logits) == list
        # print([logits.shape for logits in shift_logits], [labels.shape for labels in shift_labels])
        vocab_size = shift_logits[0].shape[-1]
        indices = [torch.where(labels != -100)[0] for labels in shift_labels]
        probs = [torch.nn.Softmax(dim=-1)(logits)[indic] for logits, indic in zip(shift_logits, indices)]
        
        label_onehot = [torch.nn.functional.one_hot(labels[indic], num_classes=vocab_size).to(probs[0].device) for labels, indic in zip(shift_labels, indices)]
        # print('Probabilities', [prob for prob in probs], [prob.sum(dim=-1) for prob in probs])
        # print([prob.shape for prob in probs], [onehot.shape for onehot in label_onehot])
        l2_values = [torch.pow(label_onehot_i-probs_i, 2) for label_onehot_i, probs_i in zip(label_onehot, probs)]
        print(probs[0][0], probs[0][0].sum(), l2_values[0][0], l2_values[0][0].sum())
        # print(l2_values)
        l2_values = [val.sum(dim=1) for val in l2_values]
        # print(l2_values)
        l2_values = [torch.sqrt(val) for val in l2_values]
        # print(l2_values)
        el2n_score = [torch.mean(val).detach().cpu().item() for val in l2_values]
        print(el2n_score)
    
    return el2n_score

def obtain_entropy_score(outputs, eval_batch_size=None):
    # Entropy of the probability vector

    probs = torch.nn.Softmax(dim=-1)(outputs.logits)
    entropy = -1 * probs * torch.log(probs + 1e-10)
    # print(probs.shape, entropy.shape)
    entropy = torch.mean(torch.sum(entropy, dim=-1), dim=-1).detach().cpu().squeeze()
    # print('entropy', entropy.shape, entropy)
    return entropy


def obtain_grand_score(gradients, eval_batch_size=None):
    # L2-Norm of the gradient vector; we consider the gradients that we get (see https://arxiv.org/pdf/2403.09559)
    grand = torch.linalg.norm(gradients, dim=-1, ord=2).detach().cpu()
    # print('grand', grand.shape, grand)
    return grand

def obtain_ig_score(model, batch, ppl_with_image, eval_batch_size): #TODO: Fix

    if eval_batch_size == 1:
        batch['images'] = torch.zeros_like(batch['images'])
        # print(batch.keys())
        with torch.inference_mode():
            loss_no_image = model(**batch).loss
        ppl_wo_image = torch.exp(loss_no_image)
        ig_score = ppl_wo_image/ppl_with_image
        # print('ig_score', ig_score.shape, ig_score)
        return ig_score
    else:
        raise NotImplementedError


def obtain_learnability_score():
    raise NotImplementedError


def prepare_optimizer_state(model, optimizer_state, device):
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    avg = torch.cat([optimizer_state[n]["exp_avg"].view(-1) for n in names])
    avg_sq = torch.cat([optimizer_state[n]["exp_avg_sq"].view(-1)
                       for n in names])
    avg = avg.to(device)
    avg_sq = avg_sq.to(device)
    return avg, avg_sq


def save_metrics(scores, output_dir, count):

    if len(scores) == 0:
        return
    
    keys = scores[0].keys()
    scores = {k: torch.stack([s[k] for s in scores]) for k in keys} #TODO: fix for multi-dimensiona;
    outfile = os.path.join(output_dir, f"scores-{count}.pt")
    torch.save(scores, outfile)
    print(
        f"Saving scores to {outfile}", flush=True
    )


def collect_grads(dataloader,
                  model,
                  output_dir,
                  proj_dim: List[int] = [8192],
                  layer_dim: List[int] = [0],
                  adam_optimizer_state: Optional[dict] = None,
                  gradient_type: str = "adam",
                  max_samples: Optional[int] = None,
                  metrics: Optional[List[str]] = None,
                  zero_order: bool = False):
    """
    Collects gradients from the model during evaluation and saves them to disk.

    Args:
        dataloader (torch.utils.data.DataLoader): The data loader for evaluation dataset.
        model (torch.nn.Module): The model from which gradients will be collected.
        output_dir (str): The directory where the gradients will be saved.
        proj_dim List[int]: The dimensions of the target projectors. Each dimension will be saved in a separate folder.
        layer_dim List[int]: The dimensions of the layers from which grads are extracted. Each layer will be saved in a separate folder.
        gradient_type (str): The type of gradients to collect. [adam | sign | sgd]
        adam_optimizer_state (dict): The optimizer state of adam optimizers. If None, the gradients will be collected without considering Adam optimization states. 
        max_samples (int, optional): The maximum number of samples to collect. Defaults to None.
    """


    model_id = 0  # model_id is used to draft the random seed for the projectors
    block_size = 128  # fixed block size for the projectors
    projector_batch_size = 16  # batch size for the projectors
    torch.random.manual_seed(0)  # set the random seed for torch

    project_interval = 16  # project every 16 batches
    save_interval = 160  # save every 160 batches

    def _project(current_full_grads, projected_grads):
        current_full_grads = torch.stack(current_full_grads).to(torch.float16)
        for i, projector in enumerate(projectors):
            for j, layer in enumerate(layer_dim):
                current_projected_grads = projector.project(
                    current_full_grads[:, j, :], model_id=model_id)
                projected_grads[proj_dim[i]][layer].append(current_projected_grads.cpu())

    def _save(projected_grads, output_dirs):
        for dim in proj_dim:
            for layer in layer_dim:
                if len(projected_grads[dim][layer]) == 0:
                    continue
                projected_grads[dim][layer] = torch.cat(projected_grads[dim][layer])

                output_dir = output_dirs[dim][layer]
                outfile = os.path.join(output_dir, f"grads-{count}.pt")
                torch.save(projected_grads[dim][layer], outfile)
                print(
                    f"Saving {outfile}, {projected_grads[dim][layer].shape}", flush=True)
                projected_grads[dim][layer] = []

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # prepare optimization states
    if gradient_type == "adam":
        assert adam_optimizer_state is not None
        # first and second moment estimates
        m, v = prepare_optimizer_state(model, adam_optimizer_state, device)

    projector = get_trak_projector(device)
    number_of_params = get_number_of_params(model, per_layer=True)
    # print_params_req_grad(model)

    # initialize a projector for each target projector dimension
    projectors = []
    for dim in proj_dim:
        proj = projector(grad_dim=number_of_params,
                         proj_dim=dim,
                         seed=0,
                         proj_type=ProjectionType.rademacher,
                         device=device,
                         dtype=dtype,
                         block_size=block_size,
                         max_batch_size=projector_batch_size)
        projectors.append(proj)

    count = 0

    # set up a output directory for each projector dimension and layer dimension
    if zero_order:
        grad_output_dir = os.path.join(output_dir, "zerograds")
    else:
        grad_output_dir = os.path.join(output_dir, 'grads')
    grad_output_dirs = {}
    for dim in proj_dim:
        grad_output_dirs[dim] = {}
        for layer in layer_dim:
            output_dir_per_dim_per_layer = os.path.join(grad_output_dir, f"dim{dim}", f"layer{layer}")
            grad_output_dirs[dim][layer] = output_dir_per_dim_per_layer
            os.makedirs(output_dir_per_dim_per_layer, exist_ok=True)

    if metrics is not None:
        scores_output_dir = os.path.join(output_dir, 'scores')
        os.makedirs(scores_output_dir, exist_ok=True)

    # max index for each dimension
    max_index = min(min(get_max_saved_index(
        grad_output_dirs[dim][layer], "grads") for layer in layer_dim) for dim in proj_dim)
    if metrics is not None:
        max_index = min(get_max_saved_index(scores_output_dir, "scores"), max_index)
    # max_index = 19520
    # stop_index = 19680

    # projected_gradients
    full_grads = []  # full gradients
    projected_grads = {dim: {layer: [] for layer in layer_dim} for dim in proj_dim}  # projected gradients

    all_scores = []

    for batch in tqdm(dataloader, total=len(dataloader)):
        prepare_batch(batch)

        count += 1

        if count <= max_index:
            print("skipping count", count, "/", max_index)
            continue

        # print(batch['labels'].shape, batch['input_ids'].shape, batch['attention_mask'].shape)

        if zero_order:
            vectorized_grads = zo_step(model, batch, layer_dim)
        else:
            # perform backpropagation # TODO: fwd-fwd algorithm
            outputs = model(**batch)
            # print(list(batch.keys()))
            # print(list(outputs.__dict__.keys()))
            loss = outputs.loss
            loss.backward()

            # print(outputs['logits'].shape)

            # get gradient vectors
            if gradient_type == "adam":
                if count == 1:
                    print("Using Adam gradients")
                vectorized_grads = obtain_gradients_with_adam(model, m, v, layer_dim)
            elif gradient_type == "sign":
                if count == 1:
                    print("Using Sign gradients")
                vectorized_grads = obtain_sign_gradients(model, layer_dim)
            else:
                if count == 1:
                    print("Using SGD gradients")
                vectorized_grads = obtain_gradients(model, layer_dim)

        # add the gradients to the full_grads
        # full_grads.append(vectorized_grads)
        full_grads.append(torch.stack([v for v in vectorized_grads.values()]))
        model.zero_grad()

        score = {} #TODO: implement for multiple eval batch size
        # get importance scores
        eval_batch_size = batch['labels'].shape[0]
        if metrics is not None:
            # if 'ppl' in metrics:
            #     score['ppl'] = obtain_perplexity_score(outputs, eval_batch_size, batch['labels'])
            #     print('PPL: ', eval_batch_size, batch['labels'].shape, score['ppl'])
            if 'entropy' in metrics:
                score['entropy'] = obtain_entropy_score(outputs, eval_batch_size)
                # print('Entropy: ', score['entropy'])
            if 'grand' in metrics:
                score['grand'] = obtain_grand_score(torch.cat([v for v in vectorized_grads.values()]), eval_batch_size)
                # print('GraNd: ', score['grand'])
            
            shift_logits, shift_labels = get_shifted_logits_labels(batch, outputs)
                
            if 'el2n' in metrics:
                # print(list(batch.keys()))
                # score['el2n'] = obtain_el2n_score(outputs, eval_batch_size, shift_logits, shift_labels)
                score['el2n'] = obtain_el2n_score(shift_logits, shift_labels, eval_batch_size)
                # print('EL2N: ', score['el2n'])
            if 'ppl' in metrics:
                # score['ppl'] = obtain_perplexity_score(outputs, eval_batch_size, shift_logits, shift_labels)
                score['ppl'] = obtain_perplexity_score(shift_logits, shift_labels)
                # print('PPL: ', eval_batch_size, batch['labels'].shape, score['ppl'])

            if 'ig' in metrics and 'images' in batch:
                # print(batch['images'])
                score['ig'] = obtain_ig_score(model, batch, score['ppl'], eval_batch_size)
                # print('IG: ', score['ig'])
            all_scores.append(score.copy())


        if count % project_interval == 0:
            _project(full_grads, projected_grads)
            full_grads = []

        if count % save_interval == 0:
            _save(projected_grads, grad_output_dirs)
            if metrics is not None:
                save_metrics(all_scores, scores_output_dir, count)
                all_scores = []
        

        if max_samples is not None and count == max_samples:
            break

    if len(full_grads) > 0:
        _project(full_grads, projected_grads)
        full_grads = []
    _save(projected_grads, grad_output_dirs)
    if metrics is not None and len(all_scores):
        save_metrics(all_scores, scores_output_dir, count)

    torch.cuda.empty_cache()
    for dim in proj_dim:
        for layer in layer_dim:
            merge_and_normalize_info(grad_output_dirs[dim][layer], prefix="grads")
            merge_info(grad_output_dirs[dim][layer], prefix="grads")
    if metrics is not None:
        merge_info(scores_output_dir, "scores")

    print("Finished")


def collect_grads_and_reps(dataloader,
                  model,
                  output_dir,
                  proj_dim: List[int] = [8192],
                  layer_dim: List[int] = [0],
                  adam_optimizer_state: Optional[dict] = None,
                  gradient_type: str = "adam",
                  max_samples: Optional[int] = None,
                  metrics: Optional[List[str]] = None):
    """
    Collects gradients and representations from the model during evaluation and saves them to disk.

    Args:
        dataloader (torch.utils.data.DataLoader): The data loader for evaluation dataset.
        model (torch.nn.Module): The model from which gradients will be collected.
        output_dir (str): The directory where the gradients will be saved.
        proj_dim List[int]: The dimensions of the target projectors. Each dimension will be saved in a separate folder.
        layer_dim List[int]: The dimensions of the layers from which grads are extracted. Each layer will be saved in a separate folder.
        gradient_type (str): The type of gradients to collect. [adam | sign | sgd]
        adam_optimizer_state (dict): The optimizer state of adam optimizers. If None, the gradients will be collected without considering Adam optimization states. 
        max_samples (int, optional): The maximum number of samples to collect. Defaults to None.
    """


    model_id = 0  # model_id is used to draft the random seed for the projectors
    block_size = 128  # fixed block size for the projectors
    projector_batch_size = 16  # batch size for the projectors
    torch.random.manual_seed(0)  # set the random seed for torch

    project_interval = 16  # project every 16 batches
    save_interval = 80  # save every 160 batches

    def _project(current_full_grads, projected_grads):
        current_full_grads = torch.stack(current_full_grads).to(torch.float16)
        for i, projector in enumerate(projectors):
            for j, layer in enumerate(layer_dim):
                current_projected_grads = projector.project(
                    current_full_grads[:, j, :], model_id=model_id)
                projected_grads[proj_dim[i]][layer].append(current_projected_grads.cpu())

    def _save_grads(projected_grads, output_dirs):
        for dim in proj_dim:
            for layer in layer_dim:
                if len(projected_grads[dim][layer]) == 0:
                    continue
                projected_grads[dim][layer] = torch.cat(projected_grads[dim][layer])

                output_dir = output_dirs[dim][layer]
                outfile = os.path.join(output_dir, f"grads-{count}.pt")
                torch.save(projected_grads[dim][layer], outfile)
                print(
                    f"Saving {outfile}, {projected_grads[dim][layer].shape}", flush=True)
                projected_grads[dim][layer] = []

    def _save_reps(reps, output_dirs):
        for layer in layer_dim:
            if len(reps) == 0:
                continue
            layer_reps = torch.cat([rep[layer] for rep in reps])
            output_dir = output_dirs[layer]
            outfile = os.path.join(output_dir, f"reps-{count}.pt")
            torch.save(layer_reps, outfile)
            print(
                f"Saving {outfile}, {layer_reps.shape}", flush=True
            )

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # prepare optimization states
    if gradient_type == "adam":
        assert adam_optimizer_state is not None
        # first and second moment estimates
        m, v = prepare_optimizer_state(model, adam_optimizer_state, device)

    projector = get_trak_projector(device)
    number_of_params = get_number_of_params(model, per_layer=True)
    # print_params_req_grad(model)

    # initialize a projector for each target projector dimension
    projectors = []
    for dim in proj_dim:
        proj = projector(grad_dim=number_of_params,
                         proj_dim=dim,
                         seed=0,
                         proj_type=ProjectionType.rademacher,
                         device=device,
                         dtype=dtype,
                         block_size=block_size,
                         max_batch_size=projector_batch_size)
        projectors.append(proj)

    count = 0

    # set up a output directory for each projector dimension and layer dimension for gradients
    grad_output_dir = os.path.join(output_dir, 'grads')
    grad_output_dirs = {}
    for dim in proj_dim:
        grad_output_dirs[dim] = {}
        for layer in layer_dim:
            output_dir_per_dim_per_layer = os.path.join(grad_output_dir, f"dim{dim}", f"layer{layer}")
            grad_output_dirs[dim][layer] = output_dir_per_dim_per_layer
            os.makedirs(output_dir_per_dim_per_layer, exist_ok=True)
    
    rep_output_dir = os.path.join(output_dir, 'reps')
    rep_output_dirs = {}
    for layer in layer_dim:
        rep_output_dirs[layer] = os.path.join(rep_output_dir, f"layer{layer}")
        os.makedirs(rep_output_dirs[layer], exist_ok=True)

    if metrics is not None:
        scores_output_dir = os.path.join(output_dir, 'scores')
        os.makedirs(scores_output_dir, exist_ok=True)

    # max index for each dimension
    max_index = min(min(get_max_saved_index(
        grad_output_dirs[dim][layer], "grads") for layer in layer_dim) for dim in proj_dim)
    max_index = min(min(get_max_saved_index(rep_output_dirs[layer], "reps") for layer in layer_dim), max_index)
    if metrics is not None:
        max_index = min(get_max_saved_index(scores_output_dir, "scores"), max_index)
    # max_index = 19520
    # stop_index = 19680

    # projected_gradients
    full_grads = []  # full gradients
    full_reps = []
    projected_grads = {dim: {layer: [] for layer in layer_dim} for dim in proj_dim}  # projected gradients

    all_scores = []
    for batch in tqdm(dataloader, total=len(dataloader)):
        prepare_batch(batch)

        count += 1

        if count <= max_index:
            print("skipping count", count)
            continue

        # print(batch['labels'].shape, batch['input_ids'].shape, batch['attention_mask'].shape)
        # perform backpropagation # TODO: fwd-fwd algorithm
        # outputs = model(**batch)
        outputs = model(input_ids=batch['input_ids'], 
                        labels=batch['labels'], 
                        images=batch['images'], 
                        attention_mask=batch['attention_mask'],
                        output_hidden_states=True)
        
        # print(list(batch.keys()))
        # print(list(outputs.__dict__.keys()))
        loss = outputs.loss
        loss.backward()

        # print(outputs['logits'].shape)

        # get gradient vectors
        if gradient_type == "adam":
            if count == 1:
                print("Using Adam gradients")
            vectorized_grads = obtain_gradients_with_adam(model, m, v, layer_dim)
        elif gradient_type == "sign":
            if count == 1:
                print("Using Sign gradients")
            vectorized_grads = obtain_sign_gradients(model, layer_dim)
        else:
            if count == 1:
                print("Using SGD gradients")
            vectorized_grads = obtain_gradients(model, layer_dim)

        # add the gradients to the full_grads
        # full_grads.append(vectorized_grads)
        full_grads.append(torch.stack([v for v in vectorized_grads.values()]))
        model.zero_grad()

        # get representations
        reps = {layer: outputs.hidden_states[layer].mean(dim=1).detach().cpu() for layer in layer_dim} #TODO: fix for multiple batch size
        full_reps.append(reps)

        score = {} #TODO: implement for multiple eval batch size
        # get importance scores
        eval_batch_size = batch['labels'].shape[0]
        if metrics is not None:
            # if 'ppl' in metrics:
            #     score['ppl'] = obtain_perplexity_score(outputs, eval_batch_size, batch['labels'])
            #     print('PPL: ', eval_batch_size, batch['labels'].shape, score['ppl'])
            if 'entropy' in metrics:
                score['entropy'] = obtain_entropy_score(outputs, eval_batch_size)
                # print('Entropy: ', score['entropy'])
            if 'grand' in metrics:
                score['grand'] = obtain_grand_score(torch.cat([v for v in vectorized_grads.values()]), eval_batch_size)
                # print('GraNd: ', score['grand'])
            
            shift_logits, shift_labels = get_shifted_logits_labels(batch, outputs)
                
            if 'el2n' in metrics:
                # print(list(batch.keys()))
                # score['el2n'] = obtain_el2n_score(outputs, eval_batch_size, shift_logits, shift_labels)
                score['el2n'] = obtain_el2n_score(shift_logits, shift_labels, eval_batch_size)
                # print('EL2N: ', score['el2n'])
            if 'ppl' in metrics:
                # score['ppl'] = obtain_perplexity_score(outputs, eval_batch_size, shift_logits, shift_labels)
                score['ppl'] = obtain_perplexity_score(shift_logits, shift_labels)
                # print('PPL: ', eval_batch_size, batch['labels'].shape, score['ppl'])

            if 'ig' in metrics and 'images' in batch:
                # print(batch['images'])
                score['ig'] = obtain_ig_score(model, batch, score['ppl'], eval_batch_size)
                # print('IG: ', score['ig'])
            all_scores.append(score.copy())


        if count % project_interval == 0:
            _project(full_grads, projected_grads)
            full_grads = []

        if count % save_interval == 0:
            _save_grads(projected_grads, grad_output_dirs)
            _save_reps(full_reps, rep_output_dirs)
            full_reps = []
            if metrics is not None:
                save_metrics(all_scores, scores_output_dir, count)
                all_scores = []

        if max_samples is not None and count == max_samples:
            break


    if len(full_grads) > 0:
        _project(full_grads, projected_grads)
        full_grads = []

    _save_grads(projected_grads, grad_output_dirs)
    _save_reps(full_reps, rep_output_dirs)
    if metrics is not None and len(all_scores):
        save_metrics(all_scores, scores_output_dir, count)

    torch.cuda.empty_cache()
    
    for dim in proj_dim:
        for layer in layer_dim:
            merge_and_normalize_info(grad_output_dirs[dim][layer], prefix="grads")
            merge_info(grad_output_dirs[dim][layer], prefix="grads")
    
    for layer in layer_dim:
        merge_info(rep_output_dirs[layer], "reps")
    
    if metrics is not None:
        merge_info(scores_output_dir, "scores")

    print("Finished")


def merge_and_normalize_info(output_dir: str, prefix="reps"):
    """ Merge and normalize the representations and gradients into a single file. """
    info = os.listdir(output_dir)
    info = [file for file in info if file.startswith(prefix)]
    # Sort the files in ascending order
    info.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
    merged_data = []
    for file in info:
        data = torch.load(os.path.join(output_dir, file))
        normalized_data = normalize(data, dim=1)
        merged_data.append(normalized_data)
    merged_data = torch.cat(merged_data, dim=0)

    output_file = os.path.join(output_dir, f"all_orig.pt")
    torch.save(merged_data, output_file)
    print(
        f"Saving the normalized {prefix} (Shape: {merged_data.shape}) to {output_file}.")


def merge_info(output_dir: str, prefix="reps"):
    """ Merge the representations and gradients into a single file without normalization. """
    info = os.listdir(output_dir)
    info = [file for file in info if file.startswith(prefix)]
    # Sort the files in ascending order
    info.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
    merged_data = []
    for file in info:
        data = torch.load(os.path.join(output_dir, file))
        merged_data.append(data)

    if prefix == "scores":
        keys = merged_data[0].keys()
        merged_data = {k: torch.cat([d[k] for d in merged_data]) for k in keys}
    else:
        merged_data = torch.cat(merged_data, dim=0)

    if prefix == 'scores':
        output_file = os.path.join(output_dir, f"all_scores.pt")
        print(f"Saving the {prefix} to {output_file}.")
    else:
        output_file = os.path.join(output_dir, f"all_unormalized.pt")
        print(
        f"Saving the unnormalized {prefix} (Shape: {merged_data.shape}) to {output_file}.")
    torch.save(merged_data, output_file)


def collect_reps(dataloader: torch.utils.data.DataLoader,
                 model: torch.nn.Module,
                 output_dir: str,
                 max_samples: Optional[int] = None):
    """
    Collects representations from a dataloader using a given model and saves them to the output directory.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader containing the input data.
        model (torch.nn.Module): The model used to compute the representations.
        output_dir (str): The directory where the representations will be saved.
        max_samples (int, optional): The maximum number of samples to collect. Defaults to None.
    """

    all_reps = []
    count = 0
    save_interval = 160  # save every 160 batches

    device = next(model.parameters()).device  # only works for single gpu
    output_dir = os.path.join(output_dir, 'reps')
    max_index = get_max_saved_index(output_dir, "reps")

    for batch in tqdm(dataloader):
        prepare_batch(batch)
        count += 1
        if count <= max_index:
            print("skipping count", count)
            continue

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.inference_mode():
            if isinstance(model, RobertaModel):
                reps = model(input_ids=input_ids,
                             attention_mask=attention_mask, output_hidden_states=True, return_dict=True).pooler_output
            else:
                hidden_states = model(input_ids,
                                      labels=input_ids,
                                      attention_mask=attention_mask,
                                      output_hidden_states=True).hidden_states
                ids = torch.arange(len(input_ids), device=input_ids.device)
                # pos = attention_mask.sum(dim=1) - 1
                # reps = hidden_states[-1][ids, pos]
                reps = hidden_states[-1]

            all_reps.append(reps.cpu())
            if count % save_interval == 0:
                all_reps = torch.cat(all_reps)
                outfile = os.path.join(output_dir, f"reps-{count}.pt")
                torch.save(all_reps, outfile)
                all_reps = []
                print(f"Saving {outfile}")

            if max_samples is not None and count >= max_samples:
                break

    if len(all_reps) > 0:
        all_reps = torch.cat(all_reps)
        outfile = os.path.join(output_dir, f"reps-{count}.pt")
        torch.save(all_reps, outfile)
        print(f"Saving {outfile}")

    torch.cuda.empty_cache()
    merge_and_normalize_info(output_dir, prefix="reps")

    print("Finished")


def get_loss(dataloader: torch.utils.data.DataLoader,
             model: torch.nn.Module,
             output_dir: str,):
    """ Get the loss of the model on the given dataset. """
    total_loss = 0
    total_tokens = 0
    for batch in tqdm(dataloader):
        prepare_batch(batch)
        num_token = (batch["labels"] != -100).sum()
        with torch.inference_mode():
            loss = model(**batch).loss * num_token
        total_loss += loss.item()
        total_tokens += num_token.item()

    print(f"Loss: {total_loss / total_tokens}")
    result = {"num_tokens": total_tokens, "loss": (
        total_loss / total_tokens)}
    with open(os.path.join(output_dir, "loss.txt"), "w") as f:
        f.write(json.dumps(result, indent=4))


def zo_perturb_parameters(named_parameters_to_optim, random_seed, scaling_factor=1):
    """
    Perturb the parameters with random vector z.
    Input: 
    - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
    - scaling_factor: theta = theta + scaling_factor * z * eps
    """

    # Set the random seed to ensure that we sample the same z for perturbation/update
    torch.manual_seed(random_seed)
    zo_eps = 1e-3
    
    for name, param in named_parameters_to_optim:
        z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
        param.data = param.data + scaling_factor * z * zo_eps


def zo_step(model, inputs, layer_dim):
    """
    Estimate gradient by MeZO. Return the loss from f(theta + z)
    """
    
    zo_eps = 1e-3

    layer_dim = [str(n) for n in layer_dim]
    vectorized_grads = {}

    # What parameters to optimize 
    named_parameters_to_optim = {layer: [] for layer in layer_dim}
    for name, param in model.named_parameters():
        layer = get_layer_dim_from_param_name(name)
        if param.requires_grad and layer in layer_dim:
            named_parameters_to_optim[layer].append((name, param))

    for layer, layer_params_to_optim in named_parameters_to_optim.items():

        # print(layer, [n for n, _ in layer_params_to_optim])

        # Sample the random seed for sampling z
        zo_random_seed = np.random.randint(1000000000)

        # First function evaluation
        zo_perturb_parameters(layer_params_to_optim, zo_random_seed, scaling_factor=1)
        loss1 = zo_forward(model, inputs)

        # Second function evaluation
        zo_perturb_parameters(layer_params_to_optim, zo_random_seed, scaling_factor=-2)
        loss2 = zo_forward(model, inputs)

        # Reset model back to its parameters at start of step
        zo_perturb_parameters(layer_params_to_optim, zo_random_seed, scaling_factor=1)

        projected_grad = ((loss1 - loss2) / (2 * zo_eps)).item()
        grads = zo_gradient(layer_params_to_optim, projected_grad, zo_random_seed)
        vectorized_grads[int(layer)] = grads.clone()

    return vectorized_grads


def zo_forward(model, inputs):
    """
    Get (no gradient) loss from the model. Dropout is turned off too.
    """
    model.eval()
    with torch.inference_mode():
        loss = model(**inputs).loss
    return loss.detach()


def zo_gradient(named_parameters_to_optim, projected_grad, zo_random_seed):
    """
    Update the parameters with the estimated gradients.
    """

    # Reset the random seed for sampling zs
    torch.manual_seed(zo_random_seed)

    grads = []
    for name, param in named_parameters_to_optim:

        # Resample z
        z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
        gradient = projected_grad * z
        # if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
        #     param.data = param.data - self._get_learning_rate() * (self.projected_grad * z + args.weight_decay * param.data)
        # else:
        #     param.data = param.data - self._get_learning_rate() * (self.projected_grad * z)
        grads.append(gradient.view(-1))

    return torch.cat(grads, dim=0)
