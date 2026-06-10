---
type: concept
aliases: [Dexterous Retargeting, 灵巧手重定向, 手势重定向, dex_retargeting]
---

# Dex-Retargeting

## 定义

Dex-Retargeting 是一类将人手姿态（通常以 [[MANO]] 参数化）映射到不同机器人末端执行器关节角度的技术，通过可微分逆运动学（IK）优化实现语义一致的跨本体动作对齐。

## 数学形式

给定人手 MANO 姿态 $\theta^{\text{human}} \in \mathbb{R}^{189}$（21 关节的 6D 旋转），求机器人关节角 $q^{\text{robot}}$：

$$
q^* = \arg\min_{q} \sum_{j \in \mathcal{K}} \| \text{FK}^{\text{robot}}(q)_j - T(\text{FK}^{\text{human}}(\theta)_j) \|^2 + \lambda \cdot \mathcal{R}(q)
$$

其中 $\mathcal{K}$ 为关键点集合（指尖、指节），$T$ 为坐标系对齐变换，$\mathcal{R}$ 为关节限位正则项。

## 核心要点

1. **Keyvector 对齐**: 使用指尖/指节关键向量（而非绝对位置）对齐人手与机器人手，提升跨形态适应性
2. **可微分 IK**: 支持梯度传播，可端到端优化映射参数
3. **关节限位约束**: 施加机器人物理关节限位约束，生成机器人可执行的配置
4. **工具**: `dex-retargeting`（dexsuite/dex-retargeting）是常用开源库

## 代表工作

- [[LAD]]: 用 Dex-Retargeting 生成 Faive/mimic 手与夹爪的配对动作数据
- [[MANO]]: 人手参数化模型，Dex-Retargeting 的输入来源

## 相关概念

- [[MANO]]
- [[Cross-Embodiment Learning]]
- [[Faive Hand]]
- [[mimic Hand]]
