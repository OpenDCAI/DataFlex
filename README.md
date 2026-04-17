
# DataFlex

<p align="center">
 Data Select · Mix · Reweight — Right in the LLM Training Loop
  <img src="https://github.com/user-attachments/assets/12b542ed-3cd9-43a9-acf0-8ebcfe564ecd" width="90%">
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
- [2026-04-04] 🎉 Our [technical report](https://huggingface.co/papers/2603.26164) ranked #1 on the Hugging Face Daily Papers leaderboard for that day.
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
| **ODM** | Online Mixture | ✅ Yes | ⚠️[official code](https://github.com/alon-albalak/online-data-mixing) |

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
pip install dataflex
```

Or install from source for development:

```bash
git clone https://github.com/OpenDCAI/DataFlex.git
cd DataFlex
pip install -e .
```

> **Note:** Python 3.11+ is recommended. The core dependencies (including `llamafactory`) will be installed automatically. If you are using Python 3.10, you need to install a compatible version of `llamafactory` manually.

The launch command is similar to [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).
Below is an example using [LESS](https://arxiv.org/abs/2402.04333) :

```bash
dataflex-cli train examples/train_lora/selectors/less.yaml
```

Unlike vanilla LLaMA-Factory, your `.yaml` config file must also include **DataFlex-specific parameters**. For details, please refer to [DataFlex-Doc](https://opendcai.github.io/DataFlex-Doc/).

## 📖 Skills

- [How to Use DataFlex](skills/how_to_use.md) — Installation, CLI commands, YAML configuration, training modes, and supported algorithms.
- [How to Add a New Algorithm](skills/how_to_add_algorithm.md) — Architecture overview, registry system, base class interfaces, and step-by-step guide for adding selectors/mixers/weighters.

## 📚 4. Experimental Results
Using DataFlex can improve performance over the default LLaMA-Factory training.

### Data Selector & Reweightor Results
We use a subset of [Open-Hermes-2.5](https://huggingface.co/datasets/OpenDCAI/DataFlex-selector-openhermes-10w) as the training dataset. The data selection algorithms and data reweighting algorithm outperform the random selector baseline on the [MMLU benchmark](https://huggingface.co/datasets/OpenDCAI/dataflex-selector-MMLUSubset-test) subset relevant to the training dataset. For the Less and Nice algorithm, we set the validation set as the [MMLU-Validation-Set](https://huggingface.co/datasets/OpenDCAI/dataflex-selector-MMLUSubset-valid-cot), using a GPT-5-generated trajectory.

<p align="center">
  <img src="https://github.com/user-attachments/assets/817c7dd7-79cf-4c70-b683-32b5b4c1722b" width="49%">
  <img src="https://github.com/user-attachments/assets/ed5495db-1c5c-4dfd-a0cd-a941000ab33d" width="49%">
</p>

### Data Mixture Results
We use subsets of [SlimPajama-627B](https://huggingface.co/datasets/cerebras/SlimPajama-627B) for data mixture. The data mixture algorithms outperform the baseline (default data mixture) on MMLU accuracy while also achieving lower perplexity across different data domains.

<div align="center">

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="1">Acc ↑</th>
      <th colspan="8">Perplexity (PPL) ↓</th>
    </tr>
    <tr>
      <th>MMLU</th>
      <th>ALL</th>
      <th>CC</th>
      <th>C4</th>
      <th>SE</th>
      <th>Wiki</th>
      <th>GitHub</th>
      <th>ArXiv</th>
      <th>Book</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="10"><b>Slim-Pajama-6B</b></td>
    </tr>
    <tr>
      <td>Baseline</td>
      <td>25.27</td>
      <td>4.217</td>
      <td>4.278</td>
      <td>4.532</td>
      <td>3.402</td>
      <td><b>3.546</b></td>
      <td><b>2.640</b></td>
      <td>3.508</td>
      <td>4.778</td>
    </tr>
    <tr>
      <td>DoReMi</td>
      <td>25.84</td>
      <td><b>4.134</b></td>
      <td><b>4.108</b></td>
      <td><b>4.358</b></td>
      <td>3.788</td>
      <td>3.997</td>
      <td>3.420</td>
      <td>3.413</td>
      <td>4.661</td>
    </tr>
    <tr>
      <td>ODM</td>
      <td><b>26.04</b></td>
      <td>4.244</td>
      <td>4.326</td>
      <td>4.555</td>
      <td><b>3.243</b></td>
      <td>3.699</td>
      <td>2.704</td>
      <td><b>2.904</b></td>
      <td><b>4.613</b></td>
    </tr>
    <tr>
      <td colspan="10"><b>Slim-Pajama-30B</b></td>
    </tr>
    <tr>
      <td>Baseline</td>
      <td>25.51</td>
      <td>3.584</td>
      <td>3.723</td>
      <td>3.505</td>
      <td>2.850</td>
      <td>3.215</td>
      <td>3.163</td>
      <td>4.540</td>
      <td>5.329</td>
    </tr>
    <tr>
      <td>DoReMi</td>
      <td><b>25.97</b></td>
      <td>3.562</td>
      <td>3.731</td>
      <td><b>3.503</b></td>
      <td>2.706</td>
      <td>2.985</td>
      <td>2.973</td>
      <td>4.441</td>
      <td>5.214</td>
    </tr>
    <tr>
      <td>ODM</td>
      <td>25.63</td>
      <td><b>3.429</b></td>
      <td><b>3.598</b></td>
      <td>3.519</td>
      <td><b>2.382</b></td>
      <td><b>2.713</b></td>
      <td><b>2.255</b></td>
      <td><b>3.487</b></td>
      <td><b>4.746</b></td>
    </tr>
  </tbody>
</table>

</div>

## 🧩 5. Ecosystem
DataFlex focuses on data scheduling during training. For a complete pipeline starting from raw data, it pairs well with [DataFlow](https://github.com/OpenDCAI/DataFlow):
<div align="center">
  <img src="https://github.com/user-attachments/assets/7459da1b-86ec-40cc-873b-c54f10f8c291" width="90%">
</div>

[DataFlow](https://github.com/OpenDCAI/DataFlow) converts raw files into LLM training data through composable operator pipelines — document parsing, knowledge cleaning, QA / CoT synthesis, and training format conversion. The output JSON can be fed directly into DataFlex.
The two projects are independent with no code dependency, connected only by standard data formats. DataFlex accepts training data from any source — DataFlow, manual annotation, HuggingFace datasets, or custom processing scripts.

## 🤝 6. Acknowledgements
We thank [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for offering an efficient and user-friendly framework for large model fine-tuning, which greatly facilitated rapid iteration in our training and experimentation workflows.  
We thank Zhongguancun Academy for their API and GPU support.
Our gratitude extends to all contributors in the open-source community—their efforts collectively drive the development of DataFlex.

## 📜 7. Citation

If you use DataFlex in your research, feel free to give us a cite.
```bibtex
@article{liang2026dataflex,
  title={DataFlex: A Unified Framework for Data-Centric Dynamic Training of Large Language Models},
  author={Liang, Hao and Zhao, Zhengyang and Qiang, Meiyi and Chen, Mingrui and Ma, Lu and Yu, Rongyi and Feng, Hengyi and Sun, Shixuan and Meng, Zimo and Ma, Xiaochen and others},
  journal={arXiv preprint arXiv:2603.26164},
  year={2026}
}

@article{liang2026towards,
  title={Towards Next-Generation LLM Training: From the Data-Centric Perspective},
  author={Liang, Hao and Zhao, Zhengyang and Han, Zhaoyang and Qiang, Meiyi and Ma, Xiaochen and Zeng, Bohan and Cai, Qifeng and Li, Zhiyu and Tang, Linpeng and Zhang, Wentao and others},
  journal={arXiv preprint arXiv:2603.14712},
  year={2026}
}
```

## 🤝 8. Community & Support

We welcome contributions of new trainers and selectors!
Please ensure code formatting is consistent with the existing style before submitting a PR.

We also welcome you to join the [DataFlex](https://github.com/OpenDCAI/DataFlex) and [DataFlow](https://github.com/OpenDCAI/DataFlow) open-source community to ask questions, share ideas, and collaborate with other developers!

•	📮 [GitHub Issues](../../issues): Report bugs or suggest features
 
•	🔧 [GitHub Pull Requests](../../pulls): Contribute code improvements

•	💬 Join our community groups to connect with us and other contributors!
 
<div align="center">
  <img src="https://github.com/user-attachments/assets/4af98244-47a8-46e2-a6c7-366fb2d99681" width="70%">
</div>
