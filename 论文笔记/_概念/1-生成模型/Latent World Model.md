---
type: concept
aliases: [Latent World Model, Latent WM, 潜空间世界模型]
---

# Latent World Model

## 定义

Latent World Model（潜空间世界模型）指**不在像素空间预测**、而是把观测先编码到低维 latent 表示再在该空间建模动力学的一类[[世界模型]]。相比像素重建型 WM（如 [[DreamerV3]] 的解码分支），Latent WM 直接对 $\hat{z}_{t+1} = f_\theta(z_t, a_t)$ 建模，避免像素重建的浪费且大幅加速规划。

## 数学形式

$$
z_t = \mathrm{enc}_\phi(o_t), \qquad \hat{z}_{t+1} = f_\theta(z_t, a_t), \qquad \mathcal{L} = \big\| \hat{z}_{t+1} - \mathrm{sg}(\mathrm{enc}_\phi(o_{t+1})) \big\|^2
$$

其中 $\mathrm{enc}_\phi$ 可以是冻结的预训练 encoder（如 [[DINOv2]]）或端到端联合学习，$\mathrm{sg}$ 是 stop-gradient。

## 核心要点

1. **规划成本低**：在 latent 空间做 [[CEM]] / [[MPPI]] rollout 比像素空间快 1-2 个数量级
2. **避免重建偏置**：不再为像素细节支付建模代价，专注 task-relevant 表征
3. **两条路线**：
   - **Foundation-based**: 冻结预训练 encoder（[[DINO-WM]]、[[V-JEPA]] 衍生）
   - **End-to-end**: 联合学习 encoder + predictor（[[JEPA]]、[[PLDM]]、LeWM）
4. **核心难题**：[[表征坍塌]] —— 无监督下 encoder 容易学到平凡解，需要 stop-gradient / EMA / 对比损失防止

## 代表工作

- [[DINO-WM]]: 冻结 [[DINOv2]] 的 foundation-based latent WM
- [[PLDM]]: 端到端像素 [[JEPA]] WM，靠 7 项损失维持稳定
- [[V-JEPA]]: 视频自监督 JEPA，可作 WM 的预训练 encoder
- [[StableWM]]: 提供统一基础设施评测各类 Latent WM
- [[DreamerV3]] / [[TD-MPC]]: 任务特定的 latent dynamics（含价值/奖励头）

## 相关概念

- [[世界模型]]
- [[JEPA]]
- [[表征坍塌]]
- [[模型预测控制|MPC]]
- [[CEM]]
