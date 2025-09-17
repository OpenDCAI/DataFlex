
# DataFlex-Preview

**DataFlex** is an advanced dynamic training framework built on top of [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory).  
It intelligently schedules data during training, supporting **dynamic sample selection**, **domain ratio adjustment**, and **dynamic weighting**, aiming to improve both training efficiency and final model performance.  

DataFlex integrates seamlessly with LlamaFactory, offering researchers and developers more flexible and powerful training control.

---

## ✨ Features

- **Dynamic Select Trainer**: Dynamically selects training samples according to a given strategy (e.g., focus on “hard” samples).  
- **Dynamic Mix Trainer**: Dynamically adjusts the ratio of data from different domains during training.  
- **Dynamic Weight Trainer**: Dynamically adjusts sample weights during backpropagation to emphasize data preferred by the model.  
- **Full compatibility with LlamaFactory**, drop-in replacement.  

---

## 🚀 Installation

```bash
git clone https://github.com/OpenDCAI/DataFlex-Preview.git
cd DataFlex-Preview
pip install -e .
pip install llamafactory
```

---

## 📌 Usage Example

The launch command is similar to [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory).
Below is an example using [LESS](https://arxiv.org/abs/2402.04333) :

```bash
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 dataflex-cli train examples/train_lora/less.yaml
```

Unlike vanilla LlamaFactory, your `.yaml` config file must also include **DataFlex-specific parameters**.

---

## 🔑 Select Trainer Configuration

```yaml
# Select Trainer parameters
train_type: dynamic_select    # [dynamic_select, dynamic_mix, dynamic_weighting, static]
component_name: loss          # selection strategy, e.g. loss / less
components_cfg_file: src/dataflex/configs/components.yaml
warmup_step: 200              # warmup steps before the first selection
update_step: 1000             # trigger selection every N steps
update_times: 2               # number of times selection will be performed
```

**Parameter Details**:

* `train_type`: Training mode (dynamic select / dynamic mix / dynamic weighting / static).
* `component_name`: Selector strategy (e.g., loss or less).
* `components_cfg_file`: Path to the selector’s config file.
* `warmup_step`: Warmup steps before the first selection.
* `update_step`: Frequency of selector activation.
* `update_times`: Total number of times the selector will run.

---

## 📚 Core Concepts

* **Trainer**: Defines the training flow (selection, mixing, weighting).
* **Selector**: Encapsulates sample selection strategies.
* **Components**: Modular parameter definitions in config files.

---

## 🤝 Contributing

We welcome contributions of new trainers and selectors!
Please ensure code formatting is consistent with the existing style before submitting a PR.

```
```
