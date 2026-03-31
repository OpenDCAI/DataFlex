# How to Use DataFlex

DataFlex is a data-centric training system built on top of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). It supports **dynamic data selection**, **dynamic data mixture**, and **dynamic data reweighting** during LLM training.

## Installation

```bash
git clone https://github.com/OpenDCAI/DataFlex.git
cd DataFlex
pip install -e .
pip install llamafactory==0.9.3
```

## CLI Commands

DataFlex provides a single CLI entry point:

```bash
# Check version
dataflex-cli version

# Run training
dataflex-cli train <config.yaml> [key=value overrides ...]
```

OmegaConf-style overrides can be appended after the YAML path:

```bash
dataflex-cli train examples/train_lora/selectors/less.yaml learning_rate=5e-5 warmup_step=20
```

For multi-GPU training, use `FORCE_TORCHRUN`:

```bash
FORCE_TORCHRUN=1 dataflex-cli train examples/train_lora/selectors/less.yaml
```

## YAML Configuration

A DataFlex training config is a standard LlamaFactory YAML with additional DataFlex-specific fields.

### Standard LlamaFactory Fields

| Section | Key Fields |
|---------|------------|
| Model | `model_name_or_path`, `trust_remote_code` |
| Method | `stage` (pt/sft), `do_train`, `finetuning_type` (lora/freeze/full), `lora_rank`, `lora_alpha`, `lora_target` |
| Dataset | `dataset`, `template`, `cutoff_len`, `overwrite_cache`, `preprocessing_num_workers` |
| Output | `output_dir`, `logging_steps`, `save_steps`, `overwrite_output_dir` |
| Training | `per_device_train_batch_size`, `gradient_accumulation_steps`, `learning_rate`, `num_train_epochs`, `lr_scheduler_type`, `bf16` |
| DeepSpeed | `deepspeed` (path to ds config JSON) |

### DataFlex-Specific Fields

| Field | Type | Description |
|-------|------|-------------|
| `train_type` | str | Training mode. One of: `dynamic_select`, `dynamic_mix`, `dynamic_weight`, `static` |
| `components_cfg_file` | str | Path to the components config YAML (default: `src/dataflex/configs/components.yaml`) |
| `component_name` | str | Which algorithm to use, matching a key in `components_cfg_file` |
| `warmup_step` | int | Number of warmup steps before dynamic behavior kicks in |
| `update_step` | int | Interval (in steps) between dynamic updates |
| `update_times` | int | Total number of dynamic updates. Use `-1` for continuous updates until training ends |
| `static_mix` | bool | If `true` with `dynamic_mix`, use fixed proportions (no dynamic updates). Used in DoReMi Step 1 & 3 |
| `train_step` | int | Total training steps. Required for `dynamic_weight` and `dynamic_mix` with `static_mix: true` |

### Data Mixture Fields (for `dynamic_mix` only)

| Field | Type | Description |
|-------|------|-------------|
| `mixture_sample_rule` | str | Sampling rule: `mixture` (proportional), `stratified` (by dataset size), `uniform` |
| `init_mixture_proportions` | list[float] | Initial proportions for each source dataset, e.g. `[0.5, 0.5]` |

## Training Modes

### 1. Dynamic Data Selection (`dynamic_select`)

Dynamically selects a subset of training samples at regular intervals based on model state.

```yaml
train_type: dynamic_select
components_cfg_file: src/dataflex/configs/components.yaml
component_name: less       # choices: less, nice, loss, delta_loss, tsds, near, random, custom
warmup_step: 10
update_step: 10
update_times: 2
```

**How it works:**
1. Warmup phase: train on randomly sampled data for `warmup_step` steps.
2. At `warmup_step` and every `update_step` steps: pause training, run the selector to pick new samples, rebuild the dataloader.
3. Total steps = `warmup_step + update_step * update_times`.

**Example:**
```bash
dataflex-cli train examples/train_lora/selectors/less.yaml
```

### 2. Dynamic Data Mixture (`dynamic_mix`)

Dynamically adjusts the proportions of data from multiple source datasets. Your `dataset` field should list multiple datasets separated by commas (e.g., `dataset: wiki_demo,c4_demo`).

#### Dynamic mixture (DoReMi Step 2, ODM):

```yaml
train_type: dynamic_mix
components_cfg_file: src/dataflex/configs/components.yaml
component_name: doremi     # choices: doremi, odm, random, static
mixture_sample_rule: mixture
init_mixture_proportions: [0.5, 0.5]
warmup_step: 100
update_step: 200
update_times: 3
```

#### Static mixture (DoReMi Step 1 & 3):

```yaml
train_type: dynamic_mix
components_cfg_file: src/dataflex/configs/components.yaml
component_name: static
static_mix: true
mixture_sample_rule: mixture
init_mixture_proportions: [0.5, 0.5]
train_step: 1000
```

**Example:**
```bash
dataflex-cli train examples/train_lora/mixers/doremi_step2_dynamic_qwen_pt_lora.yaml
```

### 3. Dynamic Data Reweighting (`dynamic_weight`)

Dynamically adjusts per-sample loss weights during backpropagation based on sample characteristics.

```yaml
train_type: dynamic_weight
components_cfg_file: src/dataflex/configs/components.yaml
component_name: loss       # choices: loss, custom
warmup_step: 100
train_step: 500
```

**How it works:**
1. Standard training for `warmup_step` steps (no reweighting).
2. After warmup: each training step computes per-sample losses and applies the weighting strategy.
3. Total steps = `train_step`.

**Example:**
```bash
dataflex-cli train examples/train_lora/weighters/loss.yaml
```

## Component Configuration (`components.yaml`)

The `components.yaml` file defines algorithm-specific parameters. It has three top-level sections:

```yaml
selectors:
  algorithm_name:
    name: algorithm_name
    params:
      param1: value1
      param2: value2

mixers:
  algorithm_name:
    name: algorithm_name
    params:
      ...

weighters:
  algorithm_name:
    name: algorithm_name
    params:
      ...
```

You select which algorithm to use via `component_name` in your training YAML.

## Supported Algorithms

### Data Selection

| Algorithm | `component_name` | Category | Description |
|-----------|-----------------|----------|-------------|
| LESS | `less` | Gradient-based | Selects samples based on gradient similarity to validation set |
| NICE | `nice` | Gradient-based | Neural network-based importance sampling with reward model |
| Loss | `loss` | Loss-based | Selects samples based on current training loss |
| Delta Loss | `delta_loss` | Loss-based | Selects based on loss change over a sliding window |
| TSDS | `tsds` | Distribution-based | Task-specific data selection using pre-computed probabilities |
| NEAR | `near` | Distribution-based | Nearest-neighbor based selection using pre-computed indices |
| Random | `random` | Random | Uniform random sampling |
| Custom | `custom` | Custom | Template for user-defined selection logic |

### Data Mixture

| Algorithm | `component_name` | Category | Description |
|-----------|-----------------|----------|-------------|
| DoReMi | `doremi` | Offline | Domain reweighting with minimax optimization (3-step pipeline) |
| ODM | `odm` | Online | Online data mixing using Exp3 multi-armed bandit |
| Static | `static` | Fixed | Fixed proportions throughout training |
| Random | `random` | Random | Random domain proportions |

### Data Reweighting

| Algorithm | `component_name` | Category | Description |
|-----------|-----------------|----------|-------------|
| Loss Reweighting | `loss` | Loss-based | Strategies: `linupper`, `uniform`, `quadratic`, `extremes` |
| Custom | `custom` | Custom | Template for user-defined weighting logic |

## Offline Preprocessing

Some selectors require offline preprocessing before training:

### TSDS (Task-Specific Data Selection)

Generates sampling probabilities based on embedding similarity between candidate and target data.

```bash
python src/dataflex/offline_selector/offline_tsds_selector.py
```

Produces `tsds_probs.npy` — set the path in `components.yaml` under `selectors.tsds.params.probs_path`.

### NEAR (Nearest-Neighbor Selection)

Computes nearest-neighbor indices between candidate and query datasets.

```bash
python src/dataflex/offline_selector/offline_near_selector.py
```

Produces `top_indices.npy` — set the path in `components.yaml` under `selectors.near.params.indices_path`.

## DoReMi Multi-Step Workflow

DoReMi requires a 3-step pipeline:

1. **Step 1** — Train a reference model with static uniform/given proportions:
   ```bash
   dataflex-cli train examples/train_full/mixers/doremi_step1_static_qwen_pt_full.yaml
   ```

2. **Step 2** — Train a proxy model with dynamic DoReMi mixing, using the Step 1 checkpoint as reference:
   ```bash
   dataflex-cli train examples/train_full/mixers/doremi_step2_dynamic_qwen_pt_full.yaml
   ```
   This outputs optimized domain weights.

3. **Step 3** — Train the final model with static proportions set to the optimized weights from Step 2:
   ```bash
   dataflex-cli train examples/train_full/mixers/doremi_step3_static_qwen_pt_full.yaml
   ```

## Example Configurations

All example configs are in the `examples/` directory:

```
examples/
├── train_lora/
│   ├── selectors/     # LESS, NICE, Loss, Delta Loss, TSDS, NEAR, Random, Custom
│   ├── mixers/        # DoReMi Step 2 (LoRA), Random
│   └── weighters/     # Loss, Custom
├── train_full/
│   └── mixers/        # DoReMi Steps 1-3 (full), ODM (full)
├── test/              # minimal smoke-test configs
├── merge_lora/        # LoRA merge configs (use llamafactory-cli export)
├── deepspeed/         # DeepSpeed ZeRO configs
└── accelerate/        # FSDP configs
```

## Compatibility with LlamaFactory

DataFlex is fully compatible with LlamaFactory. Any standard LlamaFactory YAML works with `dataflex-cli train` — if `train_type` is not specified or set to `static`, DataFlex uses the default LlamaFactory trainer with no modifications.

For operations like model export/merge, continue using `llamafactory-cli`:

```bash
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```
