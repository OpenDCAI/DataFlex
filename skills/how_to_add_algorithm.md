# How to Add a New Algorithm to DataFlex

This guide explains the DataFlex architecture and walks you through adding a new **Selector**, **Mixer**, or **Weighter**.

## Architecture Overview

DataFlex extends [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) via runtime monkey-patching. When you run `dataflex-cli train config.yaml`, the CLI (`cli.py`) does the following:

1. **Patches hparams** — replaces LlamaFactory's `FinetuningArguments` and `DataArguments` with DataFlex versions that add fields like `train_type`, `component_name`, `warmup_step`, etc.
2. **Patches trainer** — based on `train_type`, replaces LlamaFactory's `CustomSeq2SeqTrainer` / `CustomTrainer` with one of `SelectTrainer`, `MixTrainer`, or `WeightTrainer`.
3. **Patches dataset loader** (only for `dynamic_mix`) — replaces `get_dataset` to support multi-source mixture sampling.
4. **Runs training** — calls `llamafactory.train.tuner.run_exp()` which now uses the patched classes.

```
dataflex-cli train config.yaml
       │
       ▼
   cli.main()
       │
       ├── patch_finetune_params()    # inject DynamicFinetuningArguments
       ├── patch_trainer(train_type)  # inject SelectTrainer / MixTrainer / WeightTrainer
       ├── patch_get_dataset()        # (dynamic_mix only)
       │
       ▼
   llamafactory.train.tuner.run_exp()  # standard LlamaFactory flow with patched classes
```

## Directory Structure

```
src/dataflex/
├── __init__.py                  # exports __version__
├── version.py                   # single source of truth for version
├── cli.py                       # CLI entry point, monkey-patching logic
├── launcher.py                  # torchrun distributed launcher
│
├── core/
│   └── registry.py              # Registry, register_selector/mixer/weighter
│
├── configs/
│   └── components.yaml          # default component parameter presets
│
├── train/
│   ├── data/
│   │   └── loader.py            # patched get_dataset for multi-source mixing
│   ├── dataset/
│   │   └── mixed_proportion_manager.py  # multi-source dataset mixing
│   ├── hparams/
│   │   ├── dynamic_params.py    # DynamicFinetuningArguments
│   │   └── dynamic_data_params.py  # extended DataArguments
│   ├── selector/
│   │   ├── base_selector.py     # Selector ABC
│   │   ├── random_selector.py   # example: RandomSelector
│   │   ├── loss_selector.py     # LossSelector
│   │   ├── less_selector.py     # LESSSelector
│   │   └── ...                  # other selectors
│   ├── mixer/
│   │   ├── base_mixer.py        # Mixer ABC
│   │   ├── random_mixer.py      # RandomMixer
│   │   ├── doremi_mixer.py      # DoremiMixer
│   │   └── ...                  # other mixers
│   ├── weighter/
│   │   ├── base_weighter.py     # Weighter ABC
│   │   ├── loss_weighter.py     # LossWeighter
│   │   └── ...                  # other weighters
│   └── trainer/
│       ├── select_trainer.py    # SelectTrainer
│       ├── mix_trainer.py       # MixTrainer
│       └── weight_trainer.py    # WeightTrainer
│
├── offline_selector/            # standalone offline preprocessing scripts
│   ├── offline_tsds_selector.py
│   └── offline_near_selector.py
│
└── utils/
    ├── load_component.py        # loads component params from YAML
    ├── logging.py
    └── selector_io.py
```

## Registry System

All components are managed through a central `Registry` in `core/registry.py`:

```python
REGISTRY = Registry()

def register_selector(name): return REGISTRY.register("selector", name)
def register_mixer(name):    return REGISTRY.register("mixer", name)
def register_weighter(name): return REGISTRY.register("weighter", name)
```

When a trainer instantiates a component, it calls:

```python
self.selector = REGISTRY.build("selector", name, runtime=runtime, cfg=sel_params)
```

The `build` method:
1. Merges `cfg` (from `components.yaml`) and `runtime` (from trainer), with **runtime values taking priority**.
2. Inspects the class `__init__` signature to filter out unknown kwargs.
3. Instantiates the class with only the accepted parameters.

This means your component's `__init__` only receives parameters it declares — you don't need to handle unknown kwargs.

## Base Classes

### Selector

```python
# train/selector/base_selector.py
class Selector(ABC):
    def __init__(self, dataset, accelerator, data_collator, cache_dir):
        ...

    def warmup(self, num_samples, replacement) -> List[int]:
        """Default warmup: random sample indices (distributed-aware)."""
        ...

    @abstractmethod
    def select(self, model, step_id: int, num_samples: int, **kwargs) -> List[int]:
        """Return indices of selected samples for this update step."""
        ...
```

The `**kwargs` in `select` may include: `optimizer_state`, `scheduler_state`, `current_update_times`, `update_times`, `tokenizer`.

### Mixer

```python
# train/mixer/base_mixer.py
class Mixer(ABC):
    def __init__(self, mixture_manager):
        ...

    @abstractmethod
    def mix(self, model, step_id: int, **kwargs) -> np.ndarray:
        """Return updated domain proportions as a numpy array."""
        ...
```

The `**kwargs` in `mix` may include: `batch`, `domain_ids`, `data_collator`, `dataset`.

### Weighter

```python
# train/weighter/base_weighter.py
class Weighter(ABC):
    def __init__(self, **kwargs):
        ...

    @abstractmethod
    def get_weighted_loss(self, losses, *, ctx, model, inputs) -> torch.Tensor:
        """Given per-sample losses (B,), return a weighted scalar loss."""
        ...

    def training_step(self, ctx, model, inputs, num_items_in_batch, use_weighter):
        """Full training step with forward, loss computation, weighting, and backward."""
        ...
```

The base `Weighter` provides a `training_step` that handles the full forward-backward loop. Override `get_weighted_loss` to define your weighting strategy; the `training_step` will call it automatically when `use_weighter=True` (i.e., after warmup).

## Step-by-Step: Adding a New Selector

We use **Selector** as the example; the process is identical for Mixer and Weighter.

### Step 1: Create the implementation file

Create `src/dataflex/train/selector/my_selector.py`:

```python
import torch
import torch.distributed as dist
from dataflex.core.registry import register_selector
from dataflex.utils.logging import logger
from .base_selector import Selector


@register_selector("my_method")
class MySelector(Selector):
    def __init__(self, dataset, accelerator, data_collator, cache_dir, my_param: float = 0.5):
        super().__init__(dataset, accelerator, data_collator, cache_dir)
        self.my_param = my_param

    def select(self, model, step_id: int, num_samples: int, **kwargs) -> list[int]:
        if self.accelerator.is_main_process:
            # Your selection logic here
            selected = list(range(min(num_samples, len(self.dataset))))
            logger.info(f"[MySelector] Selected {len(selected)} samples at step {step_id}")
        else:
            selected = None

        # Broadcast to all ranks
        obj = [selected]
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(obj, src=0)
            selected = obj[0]
        else:
            selected = selected or []

        return selected
```

Key points:
- Use `@register_selector("my_method")` to register with the registry.
- Accept `dataset`, `accelerator`, `data_collator`, `cache_dir` from runtime. Additional params (like `my_param`) come from `components.yaml`.
- Perform selection on `is_main_process` only, then broadcast to all ranks for distributed training.

### Step 2: Register the import

Add an import in `src/dataflex/train/selector/__init__.py`:

```python
from .my_selector import *
```

### Step 3: Add configuration

Add a block in `src/dataflex/configs/components.yaml`:

```yaml
selectors:
  my_method:
    name: my_method
    params:
      cache_dir: ../dataflex_saves/my_method_output
      my_param: 0.8
```

### Step 4: Create a training config

Create a YAML config (e.g., `examples/train_lora/selectors/my_method.yaml`):

```yaml
### model
model_name_or_path: meta-llama/Llama-3.1-8B
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all

### dataset
dataset: alpaca_en_demo
template: llama3

### output
output_dir: ../dataflex_saves/Llama-3.1-8B/my_method

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
learning_rate: 1.0e-4
num_train_epochs: 1.0
bf16: true

### DataFlex
train_type: dynamic_select
components_cfg_file: src/dataflex/configs/components.yaml
component_name: my_method
warmup_step: 10
update_step: 10
update_times: 2
```

### Step 5: Run it

```bash
dataflex-cli train examples/train_lora/selectors/my_method.yaml
```

## Adding a Mixer or Weighter

The process is the same pattern:

**Mixer:**
- Subclass `Mixer`, implement `mix()`, use `@register_mixer("name")`
- Add import in `train/mixer/__init__.py`
- Add config under `mixers:` in `components.yaml`
- Set `train_type: dynamic_mix` in training YAML

**Weighter:**
- Subclass `Weighter`, implement `get_weighted_loss()`, use `@register_weighter("name")`
- Add import in `train/weighter/__init__.py`
- Add config under `weighters:` in `components.yaml`
- Set `train_type: dynamic_weight` in training YAML

## Training Pipeline Details

### SelectTrainer

```
_inner_training_loop:
  1. warmup: selector.warmup(warmup_samples) → random indices → build dataloader
  2. train for warmup_step steps
  3. at warmup_step and every update_step thereafter:
     - selector.select(model, step_id, num_samples) → new indices
     - rebuild dataloader with Subset(train_dataset, new_indices)
  4. repeat for update_times updates
  total_steps = warmup_step + update_step * update_times
```

### MixTrainer

```
_inner_training_loop:
  if static_mix:
    - mixture_manager.rebuild(total_samples) → fixed mixed dataset
    - train for train_step steps (no updates)
  else:
    1. warmup: mixture_manager.rebuild(warmup_samples) → initial mixed dataset
    2. train for warmup_step steps
    3. at warmup_step and every update_step thereafter:
       - mixer.mix(model, step_id) → new proportions
       - mixture_manager.set_proportions(probs)
       - mixture_manager.rebuild(update_samples) → new mixed dataset
       - rebuild dataloader
    4. repeat for update_times updates (or until training ends if update_times=-1)
```

### WeightTrainer

```
_inner_training_loop:
  1. use standard dataloader (no subset)
  2. each step: weighter.training_step(ctx, model, inputs, ..., use_weighter)
     - use_weighter = False during warmup (step < warmup_step)
     - use_weighter = True after warmup → applies get_weighted_loss to per-sample losses
  total_steps = train_step
```
