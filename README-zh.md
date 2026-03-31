
# DataFlex
<p align="center">
  <img src="https://github.com/user-attachments/assets/d250853b-2a03-43b0-bde3-19bf9e7142fd" width="30%">
</p>

<div align="center">

[![Documents](https://img.shields.io/badge/官方文档-单击此处-brightgreen?logo=read-the-docs)](https://OpenDCAI.github.io/DataFlex-Doc/)
[![](https://img.shields.io/github/license/OpenDCAI/DataFlex)](https://github.com/OpenDCAI/DataFlex/blob/main/LICENSE)
[![](https://img.shields.io/github/stars/OpenDCAI/DataFlex?style=social)](https://github.com/OpenDCAI/DataFlex)
[![](https://img.shields.io/github/contributors/OpenDCAI/DataFlex)](https://github.com/OpenDCAI/DataFlex/graphs/contributors)
[![](https://img.shields.io/github/repo-size/OpenDCAI/DataFlex?color=green)](https://github.com/OpenDCAI/DataFlex)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/OpenDCAI/DataFlex)

<!-- [![](https://img.shields.io/github/last-commit/OpenDCAI/DataFlex)](https://github.com/OpenDCAI/DataFlex/commits/main/) -->
<!--[![](https://img.shields.io/github/issues-raw/OpenDCAI/DataFlex)](https://github.com/OpenDCAI/DataFlex/issues) -->

🎉 如果你认可我们的项目，欢迎在 GitHub 上点个 ⭐ Star，关注项目最新进展。

简体中文 | [English](./README.md)

</div>

## 📰 1. 新闻
* [2026-03-17] 我们现已支持在 DeepSpeed ZeRO-3 下进行梯度计算，从而支持更大规模模型的训练与分析。
* [2025-12-23] 🎉 我们很高兴地宣布首个 **数据中心训练系统 DataFlex** 正式发布！敬请期待后续更新。

## 🔍 2. 概述

<img src="https://github.com/user-attachments/assets/1fdb62e4-1143-4866-afd2-c1067ad25ae8">

**DataFlex** 是一个构建在 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 之上的高级动态训练框架。
它能够在训练过程中智能地调度数据，支持 **动态样本选择**、**领域比例调整** 以及 **动态加权**，旨在同时提升训练效率与最终模型性能。

DataFlex 与 LLaMA-Factory 无缝集成，为研究人员和开发者提供更灵活、更强大的训练控制能力。关于目标与设计理念，请参考 [DataFlex-Doc](https://opendcai.github.io/DataFlex-Doc/)。

我们汇总了与数据选择、数据混合和数据重加权相关的仓库。
❌ 表示没有官方仓库；
✅ 表示有官方仓库；
⚠️ 表示有官方仓库但存在问题。

- **Dynamic Select Trainer（动态数据选择训练器）**：  
  根据给定策略在训练过程中**动态选择训练样本**（例如，优先关注“困难样本”）。支持的数据选择算法总结如下：
<div align="center">
  
  | 方法 | 类别 | 是否需要模型参与 | 官方仓库 |
  |:----:|:----:|:-------------------------------------:|:-------------:|
  | **LESS** | 基于梯度 | ✅ 是 | ⚠️[official code](https://github.com/princeton-nlp/LESS) |
  | **NICE** | 基于梯度 | ✅ 是 | ⚠️[official code](https://github.com/JTWang2000/NICE) |
  | **Loss** | 基于损失 | ✅ 是 | ❌ |
  | **Delta Loss** | 基于损失 | ✅ 是 | ❌ |
  | **NEAR** | 基于数据分布 | ❌ 否 | ❌ |
  | **TSDS** | 基于数据分布 | ❌ 否 | ✅[official code](https://github.com/ZifanL/TSDS) |
  | **Static** | 无数据选择 | ❌ 否 | ❌ |
  | **Random** | 随机采样 | ❌ 否 | ❌ |
  
</div>

- **Dynamic Mix Trainer（动态数据混合训练器）**：  
  在训练过程中**动态调整来自不同数据域的数据比例**。支持的数据混合算法总结如下：
<div align="center">
  
  | 方法 | 类别 | 是否需要模型参与 | 官方仓库 |
  |:----:|:----:|:-------------------------------------:|:-------------:|
  | **DOREMI** | 离线混合 | ✅ 是 | ⚠️[official code](https://github.com/sangmichaelxie/doremi) |
  | **ODM** | 在线混合 | ✅ 是 | ⚠️[official code](https://github.com/alon-albalak/online-data-mixing) |
</div>

- **Dynamic Weight Trainer（动态样本加权训练器）**：  
  在反向传播过程中**动态调整样本权重**，以强调模型更偏好的数据。支持的数据重加权算法总结如下：
  
<div align="center">
  
  | 方法 | 类别 | 是否需要模型参与 | 官方仓库 |
  |:----:|:----:|:-------------------------------------:|:-------------:|
  | **Loss Reweighting** | 基于损失 | ✅ 是 | ❌ |
</div>
* **与 LLaMA-Factory 完全兼容**，可作为即插即用的替代方案。

## 📌 3. 快速开始

请使用以下命令进行环境配置与安装👇

```bash
git clone https://github.com/OpenDCAI/DataFlex.git
cd DataFlex
pip install -e .
pip install llamafactory==0.9.3
```

启动命令与 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 类似。
下面给出一个使用 [LESS](https://arxiv.org/abs/2402.04333) 的示例：

```bash
dataflex-cli train examples/train_lora/selectors/less.yaml
```

与原生 LLaMA-Factory 不同的是，你的 `.yaml` 配置文件中还必须包含 **DataFlex 特有的参数**，具体请参考 [DataFlex-Doc](https://opendcai.github.io/DataFlex-Doc/)。

## 📖 Skills

- [如何使用 DataFlex](skills/how_to_use.md) — 安装、CLI 命令、YAML 配置、训练模式、已支持算法一览。
- [如何添加新算法](skills/how_to_add_algorithm.md) — 架构概览、Registry 机制、基类接口、添加 Selector/Mixer/Weighter 的完整步骤。

## 📚 4. 实验结果
使用 DataFlex 可以在默认 LLaMA-Factory 训练的基础上提升模型性能。

### 数据选择与加权实验结果
我们使用 Open-Hermes-2.5 的一个子集作为训练数据集。实验结果表明，相较于随机选择（random selector）基线，所采用的数据选择算法和数据重加权算法在与训练数据集相关的 MMLU 基准测试子集上均取得了更优的性能。对于 LESS 和 NICE 算法，我们将 MMLU-Validation-Set 作为验证集，并使用由 GPT-5 生成的推理轨迹（trajectory）进行验证。

<p align="center">
  <img src="https://github.com/user-attachments/assets/817c7dd7-79cf-4c70-b683-32b5b4c1722b" width="49%">
  <img src="https://github.com/user-attachments/assets/ed5495db-1c5c-4dfd-a0cd-a941000ab33d" width="49%">
</p>

### 数据配比实验结果
我们使用 [SlimPajama-627B](https://huggingface.co/datasets/cerebras/SlimPajama-627B) 的子集进行数据配比实验。数据配比算法在 MMLU 准确率上超过了基线方法，同时在不同数据域上也取得了更低的困惑度（PPL）。

<div align="center">

<table>
  <thead>
    <tr>
      <th rowspan="2">方法</th>
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


## 🤝 5. 致谢

我们感谢 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 提供了高效且易用的大模型微调框架，极大地促进了我们在训练与实验中的快速迭代。
感谢中关村学院提供的 API 和 GPU 支持。
同时也感谢所有开源社区的贡献者——正是你们的努力共同推动了 DataFlex 的发展。

## 📜 6. 引用

如果您在研究中使用了 DataFlex，欢迎引用我们的项目。
```bibtex
@ARTICLE{2026arXiv260326164L,
       author = {{Liang}, Hao and {Zhao}, Zhengyang and {Qiang}, Meiyi and {Chen}, Mingrui and {Ma}, Lu and {Yu}, Rongyi and {Feng}, Hengyi and {Sun}, Shixuan and {Meng}, Zimo and {Ma}, Xiaochen and {Yang}, Xuanlin and {Cai}, Qifeng and {An}, Ruichuan and {Zeng}, Bohan and {Hao Wong}, Zhen and {Shen}, Chengyu and {He}, Runming and {Han}, Zhaoyang and {Zheng}, Yaowei and {Fu}, Fangcheng and {He}, Conghui and {Cui}, Bin and {Li}, Zhiyu and {E}, Weinan and {Zhang}, Wentao},
        title = "{DataFlex: A Unified Framework for Data-Centric Dynamic Training of Large Language Models}",
      journal = {arXiv e-prints},
     keywords = {Machine Learning, Computation and Language},
         year = 2026,
        month = mar,
          eid = {arXiv:2603.26164},
        pages = {arXiv:2603.26164},
archivePrefix = {arXiv},
       eprint = {2603.26164},
 primaryClass = {stat.ML},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026arXiv260326164L},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@article{liang2026towards,
  title={Towards Next-Generation LLM Training: From the Data-Centric Perspective},
  author={Liang, Hao and Zhao, Zhengyang and Han, Zhaoyang and Qiang, Meiyi and Ma, Xiaochen and Zeng, Bohan and Cai, Qifeng and Li, Zhiyu and Tang, Linpeng and Zhang, Wentao and others},
  journal={arXiv preprint arXiv:2603.14712},
  year={2026}
}
```

## 🤝 7. 社区与支持

我们欢迎贡献新的 trainers 和 selectors！
在提交 PR 之前，请确保代码风格与现有代码保持一致。

我们也欢迎你加入 [DataFlex](https://github.com/OpenDCAI/DataFlex) 与 [DataFlow](https://github.com/OpenDCAI/DataFlow) 开源社区，提出问题、分享想法，并与其他开发者协作！

•	📮 [GitHub Issues](../../issues)：报告 Bug 或提出新功能建议

•	🔧 [GitHub Pull Requests](../../pulls)：贡献代码改进

•	💬 加入我们的社区群组，与我们及其他贡献者交流！

<div align="center">
  <img src="https://github.com/user-attachments/assets/c04cc04c-f1f4-49b0-9758-56d9d8d37c4a" width="60%">
</div>
