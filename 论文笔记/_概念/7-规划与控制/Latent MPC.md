---
type: concept
aliases: [Latent MPC, Latent-Space MPC, 潜空间模型预测控制]
---

# Latent MPC

## 定义

在 [[World Model]] 的 **latent 空间**而非像素 / 状态空间中进行 [[MPC]] 滚动优化的范式：候选动作序列在 latent 动力学下展开 $H$ 步，用一个**终端代价**（通常是预测终端 latent 与目标 latent 的欧氏距离）对候选打分，[[CEM]] / [[Nevergrad]] 等采样优化器选最优。

## 数学形式

$$
\min_{\mathbf{a}_{0:H-1}} \; c_{\text{term}}\big(F_\theta(f_\theta(o_t), \mathbf{a}_{0:H-1}), \, \mathbf{z}_g\big)
$$

其中 $f_\theta$ 是冻结 encoder，$F_\theta$ 是冻结 latent dynamics。

## 核心要点

1. **接口干净**：只需要 latent encoder + dynamics + 候选采样器 + **终端代价** 四个组件
2. **速度优势**: latent 维度 << 像素，CEM 评分极快（[[LeWM]] 比 [[DINO-WM]] 快 48×）
3. **终端代价是瓶颈**: [[TRM]] 论文证明欧氏距离会把任务信号埋在 <1% 子空间里
4. **不需要奖励信号**: 只需目标 latent，适合 goal-conditioned 设定

## 代表工作

- [[LeWM]]: ViT-Tiny encoder + Transformer predictor + CEM
- [[PLDM]]: 端到端像素 JEPA + 终端 latent MSE
- [[DINO-WM]]: 冻结 DINOv2 encoder 的 foundation-style latent MPC
- [[TRM]]: 把欧氏终端代价换成时间间隔监督的成对头

## 相关概念

- [[MPC]]
- [[CEM]]
- [[World Model]]
- [[Terminal Proximity Cost]]
- [[Hybrid Terminal Cost]]
