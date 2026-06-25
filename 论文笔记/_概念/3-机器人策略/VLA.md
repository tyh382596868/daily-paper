---
type: concept
aliases: [Vision-Language-Action Model, 视觉语言动作模型]
---

# VLA

## 定义
将预训练的视觉语言模型（VLM）扩展为机器人控制策略，使其能够根据视觉观测和语言指令直接输出机器人动作。

## 数学形式
$$a_t = \pi_\theta(o_{0:t}, l)$$
其中 $o$ 为视觉观测序列，$l$ 为语言指令，$a_t$ 为时刻 $t$ 的动作。

## 核心要点
1. 继承 VLM 的大规模预训练知识（视觉理解 + 语言推理）
2. 通过在机器人数据集上 fine-tune，输出连续/离散动作
3. 代表工作包括 OpenVLA、π0、RT-2 等

## 代表工作
- [[OpenVLA]]: 开源 VLA，基于 Llama 骨干
- [[RLDX-1]]: 加入 RL 后训练和物理信号流
- [[MolmoAct2]]: 部署导向 VLA，支持低成本硬件
- [[ACNet]]: 针对 VLA 推理延迟的轻量级异步控制适配器

## 相关概念
- [[VLM]]
- [[OpenVLA]]
- [[LoRA]]
