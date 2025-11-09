
# DataFlex

<div align="center">
<img src="https://github.com/user-attachments/assets/935c2537-8cde-44ae-a8e1-c6ec30695810">

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

# DataFlex

**DataFlex** 是一个基于 [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) 构建的高级动态训练框架。
它能够在训练过程中智能调度数据，支持 **动态样本选择**、**领域比例调整** 和 **动态加权**，旨在同时提升训练效率和最终模型表现。

DataFlex 可与 LlamaFactory 无缝集成，为研究人员和开发者提供更灵活、更强大的训练控制能力。

---

## ✨ 功能特性

* **Dynamic Select Trainer**：根据给定策略动态选择训练样本（例如，聚焦“困难”样本）。
* **Dynamic Mix Trainer**：在训练过程中动态调整不同领域数据的比例。
* **Dynamic Weight Trainer**：在反向传播时动态调整样本权重，以强化模型更偏好的数据。
* **与 LlamaFactory 完全兼容**，可直接替换使用。

---

## 🚀 安装方式

```bash
git clone https://github.com/OpenDCAI/DataFlex.git
cd DataFlex
pip install llamafactory==0.9.3
pip install -e .
```

---

## 📌 使用示例

启动命令与 [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) 类似。
以下是使用 [LESS](https://arxiv.org/abs/2402.04333) 的示例：

```bash
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 dataflex-cli train examples/train_lora/less.yaml
```

与原版 LlamaFactory 不同的是，你的 `.yaml` 配置文件必须包含 **DataFlex 特有参数**。

---

## 🔑 Select Trainer 配置示例

```yaml
# Select Trainer 参数
train_type: dynamic_select    # [dynamic_select, dynamic_mix, dynamic_weighting, static]
component_name: loss          # 选择策略，例如 loss / less
components_cfg_file: src/dataflex/configs/components.yaml
warmup_step: 200              # 在第一次选择前的预热步数
update_step: 1000             # 每隔 N 步触发一次选择
update_times: 2               # 选择操作的执行次数
```

**参数说明**：

* `train_type`：训练模式（dynamic select / dynamic mix / dynamic weighting / static）。
* `component_name`：选择器策略（例如 loss 或 less）。
* `components_cfg_file`：选择器配置文件路径。
* `warmup_step`：在第一次选择前的预热步数。
* `update_step`：选择器的触发频率。
* `update_times`：选择器执行的总次数。

---

## 📚 核心概念

* **Trainer**：定义训练流程（选择、混合、加权）。
* **Selector**：封装样本选择策略。
* **Components**：配置文件中的模块化参数定义。

---

## 🤝 贡献指南

欢迎贡献新的 Trainer 和 Selector！
在提交 PR 之前，请确保代码格式与现有风格保持一致。
