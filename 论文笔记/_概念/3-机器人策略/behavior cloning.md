---
type: concept
aliases: [行为克隆, BC, imitation learning]
---

# Behavior Cloning

## 定义
最基础的模仿学习方法：将专家演示作为监督信号，通过最小化策略输出与专家动作之间的差异来学习策略，本质是监督学习。

## 数学形式
$$\mathcal{L}_{\text{BC}} = \mathbb{E}_{(s,a) \sim \mathcal{D}_{\text{expert}}} \left[ \| \pi_\theta(s) - a \|^2 \right]$$

## 核心要点
1. 假设演示数据 i.i.d.，导致训练分布与执行时访问的状态分布不匹配（compounding errors）
2. 简单高效，是 VLA 预训练的主要范式
3. **Filtered BC**：只在成功演示上训练，丢弃失败轨迹；可减少噪声但同时丢弃有用的子轨迹信息
4. 在长时域任务中，BC 的 compounding errors 问题更严重

## 代表工作
- [[FlowPRO]]：将标准 BC 和 filtered BC 作为对比基线
- [[ForesightFlow]]：将混合质量经验（成功/失败）统一利用，超越 filtered BC

## 相关概念
- [[DAgger]]（BC 的在线改进版本）
- [[VLA 后训练]]（后训练场景下的 BC 变体）
- [[Compounding Errors]]（BC 的核心缺陷）
- [[filtered BC]]（BC 的过滤变体）
