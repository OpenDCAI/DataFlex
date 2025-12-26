
# DataFlex
<p align="center">
  <img src="https://github.com/user-attachments/assets/d250853b-2a03-43b0-bde3-19bf9e7142fd" width="30%">
</p>

<div align="center">

[![Documents](https://img.shields.io/badge/官方文档-单击此处-brightgreen?logo=read-the-docs)](https://OpenDCAI.github.io/DataFlex-Doc/)
[![](https://img.shields.io/github/license/OpenDCAI/DataFlex)](https://github.com/OpenDCAI/DataFlex/blob/main/LICENSE)
[![](https://img.shields.io/github/stars/OpenDCAI/DataFlex?style=social)](https://github.com/OpenDCAI/DataFlex)
[![](https://img.shields.io/github/issues-raw/OpenDCAI/DataFlex)](https://github.com/OpenDCAI/DataFlex/issues)
[![](https://img.shields.io/github/contributors/OpenDCAI/DataFlex)](https://github.com/OpenDCAI/DataFlex/graphs/contributors)
[![](https://img.shields.io/github/repo-size/OpenDCAI/DataFlex?color=green)](https://github.com/OpenDCAI/DataFlex)

<!-- [![](https://img.shields.io/github/last-commit/OpenDCAI/DataFlex)](https://github.com/OpenDCAI/DataFlex/commits/main/) -->

🎉 如果你认可我们的项目，欢迎在 GitHub 上点个 ⭐ Star，关注项目最新进展。

简体中文 | [English](./README.md)

</div>

## 📰 1. 新闻

* [2025-12-23] 🎉 我们很高兴地宣布首个 **数据中心训练系统 DataFlex** 正式发布！敬请期待后续更新。

## 🔍 2. 概述

<img src="https://github.com/user-attachments/assets/1fdb62e4-1143-4866-afd2-c1067ad25ae8">

**DataFlex** 是一个构建在 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 之上的高级动态训练框架。
它能够在训练过程中智能地调度数据，支持 **动态样本选择**、**领域比例调整** 以及 **动态加权**，旨在同时提升训练效率与最终模型性能。

DataFlex 与 LLaMA-Factory 无缝集成，为研究人员和开发者提供更灵活、更强大的训练控制能力。关于目标与设计理念，请参考 [DataFlex-Doc](https://opendcai.github.io/DataFlex-Doc/)。

- **Dynamic Select Trainer（动态数据选择训练器）**：  
  根据给定策略在训练过程中**动态选择训练样本**（例如，优先关注“困难样本”）。支持的数据选择算法总结如下：
<div align="center">
  
  | 方法 | 类别 | 是否需要模型参与（Model-in-the-Loop） |
  |:----:|:----:|:-------------------------------------:|
  | **LESS** | 基于梯度 | 是 |
  | **NICE** | 基于梯度 | 是 |
  | **Loss** | 基于损失 | 是 |
  | **Delta Loss** | 基于损失 | 是 |
  | **NEAR** | 基于数据分布 | 否 |
  | **TSDS** | 基于数据分布 | 否 |
  | **Static** | 无数据选择 | 否 |
  | **Random** | 随机采样 | 否 |
  
</div>

- **Dynamic Mix Trainer（动态数据混合训练器）**：  
  在训练过程中**动态调整来自不同数据域的数据比例**。支持的数据混合算法总结如下：
<div align="center">
  
  | 方法 | 类别 | 是否需要模型参与（Model-in-the-Loop） |
  |:----:|:----:|:-------------------------------------:|
  | **DOREMI** | 离线混合 | 是 |
  | **ODM** | 在线混合 | 是 |
</div>

- **Dynamic Weight Trainer（动态样本加权训练器）**：  
  在反向传播过程中**动态调整样本权重**，以强调模型更偏好的数据。支持的数据重加权算法总结如下：
  
<div align="center">
  
  | 方法 | 类别 | 是否需要模型参与（Model-in-the-Loop） |
  |:----:|:----:|:-------------------------------------:|
  | **Loss Reweighting** | 基于损失 | 是 |
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
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 dataflex-cli train examples/train_lora/selectors/less.yaml
```

与原生 LLaMA-Factory 不同的是，你的 `.yaml` 配置文件中还必须包含 **DataFlex 特有的参数**，具体请参考 [DataFlex-Doc](https://opendcai.github.io/DataFlex-Doc/)。

## 📚 4. 实验结果
### 数据选择与加权实验结果
我们使用 Open-Hermes-2.5 的一个子集作为训练数据集。实验结果表明，相较于随机选择（random selector）基线，所采用的数据选择算法和数据重加权算法在与训练数据集相关的 MMLU 基准测试子集上均取得了更优的性能。对于 LESS 和 NICE 算法，我们将 MMLU-Validation-Set 作为验证集，并使用由 GPT-5 生成的推理轨迹（trajectory）进行验证。

<p align="center">
  <img src="https://github.com/user-attachments/assets/7c00d51b-e0eb-41c0-970f-0c8ab5112fa0" width="49%">
  <img src="https://github.com/user-attachments/assets/589d1c58-ee91-49c4-b4fd-670aee8e0945" width="49%">
</p>

### 数据配比实验结果
我们使用SlimPajama-627B的子集的3B子集进行数据配比。数据配比算法能够在MMLU数据集上超过baseline方法。
<div align="center">

| Dataset | Baseline | DoReMi | ODM |
|:------:|:--------:|:------:|:---:|
|  ALL   |  25.27   | 25.84  | 26.04 |

</div>


## 🤝 5. 致谢

我们感谢 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 提供了高效且易用的大模型微调框架，极大地促进了我们在训练与实验中的快速迭代。
同时也感谢所有开源社区的贡献者——正是你们的努力共同推动了 DataFlex 的发展。

## 🤝 6. 社区与支持

我们欢迎贡献新的 trainers 和 selectors！
在提交 PR 之前，请确保代码风格与现有代码保持一致。

我们也欢迎你加入 [DataFlex](https://github.com/OpenDCAI/DataFlex) 与 [DataFlow](https://github.com/OpenDCAI/DataFlow) 开源社区，提出问题、分享想法，并与其他开发者协作！

•	📮 [GitHub Issues](../../issues)：报告 Bug 或提出新功能建议

•	🔧 [GitHub Pull Requests](../../pulls)：贡献代码改进

•	💬 加入我们的社区群组，与我们及其他贡献者交流！

<div align="center">
  <img src="https://github.com/user-attachments/assets/c04cc04c-f1f4-49b0-9758-56d9d8d37c4a" width="60%">
</div>
