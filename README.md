
# DataFlex

<div align="center">

[![Documents](https://img.shields.io/badge/Documents-Click_here-brightgreen?logo=read-the-docs)](https://OpenDCAI.github.io/DataFlex-Doc/)
[![](https://img.shields.io/github/license/OpenDCAI/DataFlex)](https://github.com/OpenDCAI/DataFlex/blob/main/LICENSE)
[![](https://img.shields.io/github/stars/OpenDCAI/DataFlex?style=social)](https://github.com/OpenDCAI/DataFlex)
[![](https://img.shields.io/github/contributors/OpenDCAI/DataFlex)](https://github.com/OpenDCAI/DataFlex/graphs/contributors)
[![](https://img.shields.io/github/repo-size/OpenDCAI/DataFlex?color=green)](https://github.com/OpenDCAI/DataFlex)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/OpenDCAI/DataFlex)

<!-- [![](https://img.shields.io/github/last-commit/OpenDCAI/DataFlex)](https://github.com/OpenDCAI/DataFlex/commits/main/) -->
<!--[![](https://img.shields.io/github/issues-raw/OpenDCAI/DataFlex)](https://github.com/OpenDCAI/DataFlex/issues) -->

🎉 If you like our project, please give us a star ⭐ on GitHub for the latest update.

[简体中文](./README-zh.md) | English

</div>

## 📰 1. News
- [2025-12-23] 🎉 We’re excited to announce the first Data-Centric Training System DataFlex, is now released! Stay tuned for future updates.


## 🔍 2. Overview
<img src="https://github.com/user-attachments/assets/093bfc8e-f450-4048-ad22-456edfdc00d9">

**DataFlex** is an advanced dynamic training framework built on top of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).  
It intelligently schedules data during training, supporting **dynamic sample selection**, **domain ratio adjustment**, and **dynamic weighting**, aiming to improve both training efficiency and final model performance.  

DataFlex integrates seamlessly with LlamaFactory, offering researchers and developers more flexible and powerful training control, for goals and design philosophy, please refer to [Dataflex-Doc](https://opendcai.github.io/DataFlex-Doc/).

- **Dynamic Select Trainer**: Dynamically selects training samples according to a given strategy (e.g., focus on “hard” samples).  
- **Dynamic Mix Trainer**: Dynamically adjusts the ratio of data from different domains during training.  
- **Dynamic Weight Trainer**: Dynamically adjusts sample weights during backpropagation to emphasize data preferred by the model.  
- **Full compatibility with LlamaFactory**, drop-in replacement.  

## 📌 3. Quick Start

Please use the following commands for environment setup and installation👇

```bash
git clone https://github.com/OpenDCAI/DataFlex.git
cd DataFlex
pip install -e .
pip install llamafactory==0.9.3
```

The launch command is similar to [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory).
Below is an example using [LESS](https://arxiv.org/abs/2402.04333) :

```bash
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 dataflex-cli train examples/train_lora/selectors/less.yaml
```

Unlike vanilla LlamaFactory, your `.yaml` config file must also include **DataFlex-specific parameters**, for details please refer to [Dataflex-Doc](https://opendcai.github.io/DataFlex-Doc/).


## 📚 4. Experimental Results

### Data Selector Results
The data selection algorithm outperforms the random selector baseline on the MMLU benchmark.

<p align="center">
  <img src="https://github.com/user-attachments/assets/afa8f232-a338-48e4-8bb2-47a79dde008b" width="49%">
  <img src="https://github.com/user-attachments/assets/c4e382f5-10ca-4cce-9f31-467b23032916" width="49%">
</p>

### Data Mixture Results
The data mixture algorithm also outperforms baselines on the MMLU benchmark.
<div align="center">

| Dataset | Baseline | DoReMi | ODM |
|:------:|:--------:|:------:|:---:|
|  ALL   |  25.27   | 25.84  | 26.04 |

</div>


## 🤝 5. Acknowledgements
We thank [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for offering an efficient and user-friendly framework for large model fine-tuning, which greatly facilitated rapid iteration in our training and experimentation workflows.  
Our gratitude extends to all contributors in the open-source community—their efforts collectively drive the development of DataFlex.

## 🤝 6. Community & Support

We welcome contributions of new trainers and selectors!
Please ensure code formatting is consistent with the existing style before submitting a PR.

We also welcome you to join the [DataFlex](https://github.com/OpenDCAI/DataFlex) and [Dataflow](https://github.com/OpenDCAI/DataFlow) open-source community to ask questions, share ideas, and collaborate with other developers!

•	📮 [GitHub Issues](../../issues): Report bugs or suggest features
 
•	🔧 [GitHub Pull Requests](../../pulls): Contribute code improvements

•	💬 Join our community groups to connect with us and other contributors!
 
<div align="center">
  <img src="https://github.com/user-attachments/assets/c04cc04c-f1f4-49b0-9758-56d9d8d37c4a" width="60%">
</div>
