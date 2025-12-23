
# DataFlex

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

<img src="https://github.com/user-attachments/assets/935c2537-8cde-44ae-a8e1-c6ec30695810">

**DataFlex** 是一个构建在 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 之上的高级动态训练框架。
它能够在训练过程中智能地调度数据，支持 **动态样本选择**、**领域比例调整** 以及 **动态加权**，旨在同时提升训练效率与最终模型性能。

DataFlex 与 LlamaFactory 无缝集成，为研究人员和开发者提供更灵活、更强大的训练控制能力。关于目标与设计理念，请参考 [Dataflex-Doc](https://opendcai.github.io/DataFlex-Doc/)。

* **动态选择 Trainer**：根据给定策略动态选择训练样本（例如聚焦“困难”样本）。
* **动态混合 Trainer**：在训练过程中动态调整来自不同领域的数据比例。
* **动态加权 Trainer**：在反向传播过程中动态调整样本权重，以强调模型更偏好的数据。
* **与 LlamaFactory 完全兼容**，可作为即插即用的替代方案。

## 📌 3. 快速开始

请使用以下命令进行环境配置与安装👇

```bash
git clone https://github.com/OpenDCAI/DataFlex.git
cd DataFlex
pip install -e .
pip install llamafactory==0.9.3
```

启动命令与 [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) 类似。
下面给出一个使用 [LESS](https://arxiv.org/abs/2402.04333) 的示例：

```bash
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 dataflex-cli train examples/train_lora/selectors/less.yaml
```

与原生 LlamaFactory 不同的是，你的 `.yaml` 配置文件中还必须包含 **DataFlex 特有的参数**，具体请参考 [Dataflex-Doc](https://opendcai.github.io/DataFlex-Doc/)。

## 📚 4. 实验结果
### Data Selector Results
我们的算法能够超过random selector算法。
<div align="center">

  <div style="display: inline-block; width: 49%; text-align: center;">
    <img src="https://github.com/user-attachments/assets/afa8f232-a338-48e4-8bb2-47a79dde008b"
         alt="ICML 2025 Certificate" width="95%"><br>
    <sub><em>LLaMA3.2-3B Results</em></sub>
  </div>

  <div style="display: inline-block; width: 49%; text-align: center;">
    <img src="https://github.com/user-attachments/assets/c4e382f5-10ca-4cce-9f31-467b23032916"
         alt="LIC 2025 Certificate" width="95%"><br>
    <sub><em>Mistral-7B Results</em></sub>
  </div>

</div>

## 🤝 5. 致谢

我们感谢 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 提供了高效且易用的大模型微调框架，极大地促进了我们在训练与实验中的快速迭代。
同时也感谢所有开源社区的贡献者——正是你们的努力共同推动了 DataFlex 的发展。

## 🤝 6. 社区与支持

我们欢迎贡献新的 trainers 和 selectors！
在提交 PR 之前，请确保代码风格与现有代码保持一致。

我们也欢迎你加入 [DataFlex](https://github.com/OpenDCAI/DataFlex) 与 [Dataflow](https://github.com/OpenDCAI/DataFlow) 开源社区，提出问题、分享想法，并与其他开发者协作！

•	📮 [GitHub Issues](../../issues)：报告 Bug 或提出新功能建议

•	🔧 [GitHub Pull Requests](../../pulls)：贡献代码改进

•	💬 加入我们的社区群组，与我们及其他贡献者交流！

<div align="center">
  <img src="https://github.com/user-attachments/assets/c04cc04c-f1f4-49b0-9758-56d9d8d37c4a" width="60%">
</div>
