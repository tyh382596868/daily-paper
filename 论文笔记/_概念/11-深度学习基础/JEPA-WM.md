---
type: concept
aliases: [JEPA 世界模型, Joint-Embedding Predictive World Model]
---

# JEPA-WM

## 定义
一类**在嵌入空间内做预测与规划**的世界模型：用冻结的自监督视觉编码器（如 [[DINOv2]]/[[DINOv3]]/[[V-JEPA]]）把观测映射到 token 嵌入，再用可训练的预测器在嵌入空间预测下一帧嵌入，**完全不重建像素、不预测奖励**。

## 数学形式
$$
\mathcal{L} = \mathbb{E}\,\big\| P_\theta(E_\phi(o_{t-w:t}), A_\theta(a_{t-w:t})) - E_\phi(o_{t+1}) \big\|^2_2
$$

规划阶段用 [[MPC]]/[[CEM]] 在动作空间搜索使预测终态嵌入接近目标嵌入的动作序列。

## 核心要点
1. **frozen encoder**：避免端到端学习导致的 [[表征坍塌]]
2. **嵌入空间预测**：比像素重建（[[DreamerV3]]）计算更省、对视觉细节更鲁棒
3. **无 reward/value head**：纯 self-supervised，靠规划而非策略学习
4. **依赖编码器质量**：[[DINO]] 系（细粒度物体分割）显著优于 [[V-JEPA]] 系视频编码器
5. **多步 rollout 训练**：缓解自回归 [[Compounding Errors]]

## 代表工作
- [[DINO-WM]]: 首个用 [[DINOv2]] 做 JEPA-WM 的工作
- [[JEPA-WM]] (Terver et al. 2025): 系统消融，给出 SOTA 配置
- V-JEPA-2-AC: Meta 提出的 video-JEPA 加 action conditioning 版本
- [[PLDM]]: 早期潜空间 planning 工作

## 相关概念
- [[JEPA]]
- [[潜空间世界模型]]
- [[MPC]]
- [[CEM]]
- [[AdaLN]]
