
# DataFlex
<p align="center">
  <img src="https://github.com/user-attachments/assets/d250853b-2a03-43b0-bde3-19bf9e7142fd" width="30%">
</p>

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
- [2026-03-17] We now support gradient computation under DeepSpeed ZeRO-3, enabling training and analysis of larger-scale models.
- [2025-12-23] 🎉 We’re excited to announce the first Data-Centric Training System DataFlex, is now released! Stay tuned for future updates.


## 🔍 2. Overview
<img src="https://github.com/user-attachments/assets/093bfc8e-f450-4048-ad22-456edfdc00d9">

**DataFlex** is an advanced dynamic training framework built on top of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).  
It intelligently schedules training data during optimization and integrates several difficult-to-reproduce repositories into a unified framework. The system provides reproducible implementations of **Data Selection**, **Data Mixture**, and **Data Reweighting**, thereby improving both experimental reproducibility and final model performance.

DataFlex integrates seamlessly with LLaMA-Factory, offering researchers and developers more flexible and powerful training control. For goals and design philosophy, please refer to [DataFlex-Doc](https://opendcai.github.io/DataFlex-Doc/).
We summarize repositories related to Data Selection, Data Mixture, and Data Reweighting.
❌ indicates that no official repository is available;
✅ indicates that an official repository is available;
⚠️ indicates that an official repository exists but contains issues.

- **Data Selection**: Dynamically selects training samples according to a given strategy (e.g., focus on “hard” samples). The data selection algorithms are summarized as follows:

<div align="center">

| Method | Category | Requires Model-in-the-Loop? | Official Repo |
|:------:|:--------:|:---------------------------:|:-------------:|
| **LESS** | Gradient-Based | ✅ Yes | ⚠️[official code](https://github.com/princeton-nlp/LESS) |
| **NICE** | Gradient-Based | ✅ Yes | ⚠️[official code](https://github.com/JTWang2000/NICE) |
| **Loss** | Loss-Based | ✅ Yes | ❌ |
| **Delta Loss** | Loss-Based | ✅ Yes | ❌ |
| **NEAR** | Data Distribution-Based | ❌ No | ❌ |
| **TSDS** | Data Distribution-Based | ❌ No | ✅[official code](https://github.com/ZifanL/TSDS) |
| **Static** | No Selection | ❌ No | ❌ |
| **Random** | Random Sampling | ❌ No | ❌ |

</div>


- **Data Mixture**: Dynamically adjusts the ratio of data from different domains during training. The data mixture algorithms are summarized as follows:

<div align="center">

| Method | Category | Requires Model-in-the-Loop? | Official Repo |
|:------:|:--------:|:---------------------------:|:-------------:|
| **DOREMI** | Offline Mixture | ✅ Yes | ⚠️[official code](https://github.com/sangmichaelxie/doremi) |
| **ODM** | Online Mixture | ✅ Yes | ❌ |

</div>

- **Data Reweighting**: Dynamically adjusts sample weights during backpropagation to emphasize data preferred by the model. The data reweighting algorithms are summarized as follows:

<div align="center">

| Method | Category | Requires Model-in-the-Loop? | Official Repo |
|:------:|:--------:|:---------------------------:|:-------------:|
| **Loss Reweighting** | Loss-Based | ✅ Yes | ❌ |

</div>

- **Full compatibility with LLaMA-Factory**, drop-in replacement.  

## 📌 3. Quick Start

Please use the following commands for environment setup and installation👇

```bash
git clone https://github.com/OpenDCAI/DataFlex.git
cd DataFlex
pip install -e .
pip install llamafactory==0.9.3
```

The launch command is similar to [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).
Below is an example using [LESS](https://arxiv.org/abs/2402.04333) :

```bash
dataflex-cli train examples/train_lora/selectors/less.yaml
```

Unlike vanilla LLaMA-Factory, your `.yaml` config file must also include **DataFlex-specific parameters**. For details, please refer to [DataFlex-Doc](https://opendcai.github.io/DataFlex-Doc/).


## 📚 4. Experimental Results
Using DataFlex can improve performance over the default LLaMA-Factory training.

### Data Selector & Reweightor Results
We use a subset of [Open-Hermes-2.5](https://huggingface.co/datasets/OpenDCAI/DataFlex-selector-openhermes-10w) as the training dataset. The data selection algorithms and data reweighting algorithm outperform the random selector baseline on the [MMLU benchmark](https://huggingface.co/datasets/OpenDCAI/dataflex-selector-MMLUSubset-test) subset relevant to the training dataset. For the Less and Nice algorithm, we set the validation set as the [MMLU-Validation-Set](https://huggingface.co/datasets/OpenDCAI/dataflex-selector-MMLUSubset-valid-cot), using a GPT-5-generated trajectory.

<p align="center">
  <img src="https://github.com/user-attachments/assets/7c00d51b-e0eb-41c0-970f-0c8ab5112fa0" width="49%">
  <img src="https://github.com/user-attachments/assets/589d1c58-ee91-49c4-b4fd-670aee8e0945" width="49%">
</p>

### Data Mixture Results
We use subsets of [SlimPajama-627B](https://huggingface.co/datasets/cerebras/SlimPajama-627B) for data mixture. The data mixture algorithms outperform the baseline (default data mixture) on MMLU accuracy while also achieving lower perplexity across different data domains.

<div align="center">

| | Acc ↑ | | Perplexity (PPL) ↓ | | | | | |
|:------:|:--------:|:------:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Method** | **MMLU** | **ALL** | **CC** | **C4** | **SE** | **Wiki** | **GitHub** | **ArXiv** | **Book** |
| | | | **Slim-Pajama-6B** | | | | | |
| Baseline | 25.27 | 4.217 | 4.278 | 4.532 | 3.402 | **3.546** | **2.640** | 3.508 | 4.778 |
| DoReMi | 25.84 | **4.134** | **4.108** | **4.358** | 3.788 | 3.997 | 3.420 | 3.413 | 4.661 |
| ODM | **26.04** | 4.244 | 4.326 | 4.555 | **3.243** | 3.699 | 2.704 | **2.904** | **4.613** |
| | | | **Slim-Pajama-30B** | | | | | |
| Baseline | 25.51 | 3.584 | 3.723 | 3.505 | 2.850 | 3.215 | 3.163 | 4.540 | 5.329 |
| DoReMi | **25.97** | 3.562 | 3.731 | **3.503** | 2.706 | 2.985 | 2.973 | 4.441 | 5.214 |
| ODM | 25.63 | **3.429** | **3.598** | 3.519 | **2.382** | **2.713** | **2.255** | **3.487** | **4.746** |

</div>


## 🤝 5. Acknowledgements
We thank [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for offering an efficient and user-friendly framework for large model fine-tuning, which greatly facilitated rapid iteration in our training and experimentation workflows.  
We thank Zhongguancun Academy for their API and GPU support.
Our gratitude extends to all contributors in the open-source community—their efforts collectively drive the development of DataFlex.

## 📜 6. Citation

If you use DataFlex in your research, feel free to give us a cite.
```bibtex
@misc{dataflex2026,
  title={DataFlex},
  author={{OpenDCAI}},
  year={2026},
  howpublished={\url{https://github.com/OpenDCAI/DataFlex}},
  note={GitHub repository}
}
```

## 🤝 7. Community & Support

We welcome contributions of new trainers and selectors!
Please ensure code formatting is consistent with the existing style before submitting a PR.

We also welcome you to join the [DataFlex](https://github.com/OpenDCAI/DataFlex) and [DataFlow](https://github.com/OpenDCAI/DataFlow) open-source community to ask questions, share ideas, and collaborate with other developers!

•	📮 [GitHub Issues](../../issues): Report bugs or suggest features
 
•	🔧 [GitHub Pull Requests](../../pulls): Contribute code improvements

•	💬 Join our community groups to connect with us and other contributors!
 
<div align="center">
  <img src="https://github.com/user-attachments/assets/c04cc04c-f1f4-49b0-9758-56d9d8d37c4a" width="60%">
</div>
