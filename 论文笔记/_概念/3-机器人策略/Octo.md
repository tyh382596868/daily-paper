---
type: concept
aliases: [Octo, Octo Policy]
---

# Octo

## 定义

由 Berkeley、Stanford 等机构联合发布的**通用机器人 Transformer 策略**，2024 年开源，可以视为开放版的 [[RT-2]]。Octo 基于 Open X-Embodiment 数据集预训练，输出 [[Action Chunking|动作块]]，支持语言、目标图像、机器人本体等多模态条件。

## 核心思路

1. 用 [[Transformer]] 把语言、图像、本体状态统一 token 化
2. 头部用[[扩散模型]]做连续动作回归（不像 [[RT-2]] / [[OpenVLA]] 用离散动作 token）
3. 在 80 万条多机器人轨迹上预训练，下游可零样本或少样本微调

## 与其他 VLA 的对比

| 方法 | 输出形式 | Backbone | 开源 |
|------|---------|---------|------|
| [[RT-2]] | 离散动作 token | PaLI-X | 否 |
| [[OpenVLA]] | 离散动作 token | LLaMA-2 | 是 |
| **Octo** | **扩散动作** | Transformer | **是** |

## 已知局限

- 视觉表征较弱（无 LLM），长程任务规划能力弱
- 与 [[OpenVLA]]、[[RT-2]] 一样属于[[反应式策略]]，对[[动态操作]]场景失败

## 代表论文 / 工作

- Octo (2024)
- [[AHEAD]]: 把 Octo 作为基线之一，证明 latent 预测对动态场景的提升

## 相关概念

- [[VLA]]
- [[OpenVLA]]
- [[Action Chunking]]
- [[反应式策略]]
