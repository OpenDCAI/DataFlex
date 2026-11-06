import torch
import math
import os
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from dataflex.train.data.loader import split_batch
from dataflex.core.registry import register_selector
from .base_selector import Selector
from dataflex.utils.logging import logger
from dataflex.utils.loss_utils import UNSCORED, per_sample_loss_from_outputs
from dataflex.utils.selector_io import load_cached_selection, save_selection

def sigmoid(x, k):
    return 1 / (1 + np.exp(-k * (x - 0.5)))

# Compute the window position based on the current update times and the total update times.
def calculate_window_position(current_update_times, update_times, dataset_len, window_size=0.2, k=10):
    scaled_iteration = current_update_times / update_times

    delta = sigmoid(scaled_iteration, k) * (dataset_len - window_size * dataset_len)
    
    # Compute the window start and end position.
    window_start = delta
    window_end = delta + window_size * dataset_len
    return int(window_start), int(window_end)


class IndexedDataset(Dataset):
    def __init__(self, original_dataset, indices=None):
        self.dataset = original_dataset
        self.indices = list(indices) if indices is not None else None

    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.dataset)

    def __getitem__(self, i):
        index = self.indices[i] if self.indices is not None else i
        data = self.dataset[index]
        return {"idx": index, **data}

@register_selector('delta_loss')
class DeltaLossSelector(Selector):
    def __init__(
        self,
        dataset,
        accelerator,
        data_collator,
        cache_dir,
        window_size: float = 0.2,         # Window size, default 20%
        score_batch_size: int = 1,        # Samples per scoring forward pass
        score_num_workers: int = 2,       # DataLoader workers for scoring
    ):
        super().__init__(dataset, accelerator, data_collator, cache_dir)
        self.seed = 42
        self.window_size = window_size
        self.first_time = True
        self.path_to_initial_losses = None
        self._initial_losses = None
        self.score_batch_size = int(score_batch_size)
        self.score_num_workers = int(score_num_workers)

    def _compute_losses(self, positions, model, step_id: int):
        """Per-sample loss for `positions`, one value per position.

        Same change as in LossSelector: reading `model(...).loss` pinned scoring
        to `batch_size=1` because that value is a batch mean. Going through
        logits+labels makes the result independent of how the pool is batched.
        """
        positions = list(positions)
        if not positions:
            return []

        dataloader = DataLoader(
            IndexedDataset(self.dataset, positions),
            batch_size=self.score_batch_size,
            shuffle=False,
            num_workers=self.score_num_workers,
            collate_fn=self.data_collator,
        )
        dataloader = self.accelerator.prepare(dataloader)

        if self.accelerator.is_main_process:
            logger.info(
                f"[Dataflex] Calculating loss for {len(positions)} samples using "
                f"{self.accelerator.num_processes} GPUs (batch_size={self.score_batch_size})"
            )
        local_losses, local_indices = [], []
        for batch in tqdm(
            dataloader,
            desc=f"[Selector step {step_id}]",
            disable=not self.accelerator.is_main_process,
            dynamic_ncols=True,
        ):
            idx = batch["idx"]
            if not torch.is_tensor(idx):
                idx = torch.tensor(idx, dtype=torch.long, device=self.accelerator.device)
            idx = idx.view(-1).to(dtype=torch.long)

            with torch.no_grad():
                model_inputs, _ = split_batch(batch)
                outputs = model(**model_inputs)
                loss = per_sample_loss_from_outputs(outputs, model_inputs).view(-1)

            local_losses.append(loss.float())
            local_indices.append(idx)

        if local_losses:
            local_losses = torch.cat(local_losses, dim=0)
            local_indices = torch.cat(local_indices, dim=0)
        else:
            local_losses = torch.zeros(0, device=self.accelerator.device)
            local_indices = torch.zeros(0, dtype=torch.long, device=self.accelerator.device)

        all_losses = self.accelerator.gather(local_losses).detach().cpu().tolist()
        all_indices = self.accelerator.gather(local_indices).detach().cpu().tolist()

        # gather pads the last batch, so the same idx can come back more than
        # once; keeping the first occurrence makes the result independent of the
        # world size. Non-finite values are left out so that a duplicate
        # carrying a real number can still fill the index in, and anything still
        # missing afterwards reads as UNSCORED.
        by_index = {}
        for value, i in zip(all_losses, all_indices):
            i = int(i)
            if i not in by_index and math.isfinite(value):
                by_index[i] = float(value)

        if self.accelerator.is_main_process:
            logger.info(f"[Dataflex] Loss calculation finished")
        return [by_index.get(p, UNSCORED) for p in positions]

    def _dense_baseline(self, positions, scores):
        """Baseline laid out over the whole dataset, UNSCORED where missing.

        Kept dense so the on-disk format stays what it was (a list of length
        `len(dataset)` under `metric["loss"]`) and so it can be looked up by
        dataset index later, even if a pool restricted what was scored.
        """
        dense = np.full(len(self.dataset), UNSCORED, dtype=np.float64)
        for p, v in zip(positions, scores):
            dense[p] = v
        return dense

    def state_dict(self):
        return {"first_time": self.first_time,
                "initial_losses": (torch.as_tensor(self._initial_losses).clone()
                                   if self._initial_losses is not None else None)}

    def load_state_dict(self, state):
        baseline = state["initial_losses"]
        if baseline is not None and baseline.shape != (len(self.dataset),):
            raise ValueError("DeltaLoss baseline does not match the dataset")
        if not state["first_time"] and baseline is None:
            raise ValueError("DeltaLoss checkpoint is missing its initial baseline")
        self.first_time = bool(state["first_time"])
        self._initial_losses = baseline.cpu().numpy().copy() if baseline is not None else None
        self.path_to_initial_losses = None  # Resume does not depend on old cache files.

    def select(self, model, step_id: int, num_samples: int, **kwargs):
        model.eval()
        os.makedirs(self.cache_dir, exist_ok=True)
        save_path = os.path.join(self.cache_dir, f"step_{step_id}.json")

        positions = self.candidate_positions()
        n = len(positions)

        # ========= First call to select, calculate and save initial_losses =========
        if self.first_time == True:
            self.first_time = False
            self.path_to_initial_losses = save_path
            # Read and broadcast in main
            if getattr(self, "reuse_selection_cache", True) and os.path.exists(save_path):
                if self.accelerator.is_main_process:
                    cached_indices, metrics = load_cached_selection(save_path)
                    self._initial_losses = np.asarray(metrics["loss"], dtype=np.float64)
                else:
                    cached_indices = None
                cached_indices_list = [cached_indices]
                if dist.is_available() and dist.is_initialized():
                    dist.broadcast_object_list(cached_indices_list, src=0)
                    cached_indices = cached_indices_list[0]
                else:
                    cached_indices = cached_indices or []
                return cached_indices

            logger.info(f"[Dataflex] Calculating initial losses...")
            scores = self._scores_for(positions, model, step_id)
            self._initial_losses = self._dense_baseline(positions, scores)

            if self.accelerator.is_main_process:
                logger.info(f"[Dataflex] Got initial_losses. Return random warmup selection.")
                gen = torch.Generator()
                gen.manual_seed(self.seed)
                if num_samples > n:
                    raise ValueError(
                        f"Cannot sample {num_samples} without replacement from {n} samples"
                    )
                picked = torch.randperm(n, generator=gen)[:num_samples].tolist()
                sel = [positions[i] for i in picked]
                metric_payload = {"loss": self._dense_baseline(positions, scores).tolist()}
                # Only save in main
                save_selection(save_path, sel, metric_payload, self.accelerator)

            else:
                sel = None

            obj = [sel]
            if dist.is_available() and dist.is_initialized():
                dist.broadcast_object_list(obj, src=0)
                sel = obj[0]
            else:
                sel = sel or []

            return sel
        
        # ========= Subsequent call to select, select samples based on delta_loss =========
        
        if getattr(self, "reuse_selection_cache", True) and os.path.exists(save_path):
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

        logger.info(f"[Dataflex] Calculating current losses...")
        current_scores = self._scores_for(positions, model, step_id)

        # ========= Delta Loss selection =========
        if self.accelerator.is_main_process:
            logger.info(f"[Dataflex] Loading initial losses from {self.path_to_initial_losses}")

            if self._initial_losses is None:
                _, metrics = load_cached_selection(self.path_to_initial_losses)
                self._initial_losses = np.asarray(metrics["loss"], dtype=np.float64)
            baseline = self._initial_losses.copy()
            # Caches written before the sentinel was unified hold +inf for
            # entries that were never scored, and an infinite loss is not a
            # usable measurement either way. Normalising them here keeps both
            # file formats on the one code path below.
            baseline[~np.isfinite(baseline)] = UNSCORED

            logger.info(f"[Dataflex] Selecting samples based on delta loss.")

            # Calculate delta loss, only compare on candidate pool
            initial = torch.tensor(baseline[np.asarray(positions, dtype=np.int64)], dtype=torch.float32)
            current = torch.tensor(current_scores, dtype=torch.float32)
            delta_loss = initial - current
            # An unknown score on either side leaves the difference unknown, and
            # those samples sort last so the window never reaches them.
            delta_loss = torch.nan_to_num(delta_loss, nan=float("-inf"))

            # Sort indices
            sorted_indices = torch.argsort(delta_loss, descending=True)

            # Calculate sliding window position
            window_start, window_end = calculate_window_position(kwargs["current_update_times"]-1, kwargs["update_times"]-1, n, window_size=self.window_size)
            invalid_position = (delta_loss[sorted_indices] < 0).nonzero()
            if len(invalid_position) > 0:
                invalid_position = invalid_position[0].item()  # Get the first position less than 0
                window_end = min(window_end, invalid_position)  # Set window right endpoint
            
            # Output selection times, window position
            logger.info(f"[Dataflex] Step {step_id}, Update {kwargs['current_update_times']-1}/{kwargs['update_times']-1}, Window position: [{window_start}, {window_end})")
            # Output the largest and smallest five delta loss and their indices in the window
            window_delta_loss = delta_loss[sorted_indices][window_start:window_end]
            if len(window_delta_loss) > 0:
                logger.info(f"[Dataflex] Window delta loss stats:")
                logger.info(f"  Max 5: {window_delta_loss[:5].cpu().numpy()}")
                logger.info(f"  Min 5: {window_delta_loss[-5:].cpu().numpy()}")
            probs = torch.full((len(delta_loss),), 0.025, device=delta_loss.device)

            # Set the probability of samples in the window to be larger
            selected = sorted_indices[window_start:window_end]
            probs[selected] = 1.0

            # Normalize probability
            probs = probs / probs.sum()  # Normalize probability, make the sum to be 1

            available = int((probs > 0).sum().item())
            effective_replacement = False
            
            # If the effective sample size is less than the requested sample size, use replacement sampling
            if not effective_replacement and num_samples > available:
                effective_replacement = True
                logger.info(
                    f"[Dataflex] Effective sample size {available} is less than the requested number {num_samples},"
                    f"automatically changed to replacement sampling."
                )

            # Create random number generator
            gen = torch.Generator()
            gen.manual_seed(self.seed + int(step_id))
            
            # Use torch.multinomial for sampling
            picked = torch.multinomial(probs.cpu(), num_samples=num_samples,
                                       replacement=effective_replacement, generator=gen).tolist()
            # multinomial gives the indices in the candidate pool, map back to the dataset indices
            sel = [positions[i] for i in picked]

            # ========= Save (only save "selected indices + corresponding metric") =========
            metric_payload = {
                "delta_loss": [float(delta_loss[i].item()) for i in picked]
            }
            save_selection(save_path, sel, metric_payload, self.accelerator)
        else:
            sel = None
        # Broadcast selected samples
        sel_list = [sel]
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(sel_list, src=0)
            sel = sel_list[0]
        else:
            sel = sel or []
        self.accelerator.wait_for_everyone()
        return sel
