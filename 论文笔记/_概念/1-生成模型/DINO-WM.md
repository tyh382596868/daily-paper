---
type: concept
aliases: [DINO-WM, DINO World Model]
---

# DINO-WM

## 定义

DINO-WM (Zhou et al., 2024) 是一类 **foundation-based 世界模型**：冻结预训练的 [[DINOv2]] encoder 作为视觉 backbone，仅训练一个 predictor 在 latent 空间预测下一帧。回避了 [[JEPA]] 类自监督的[[表征坍塌]]问题，但放弃端到端学习，且每帧 token 数高（~200）导致规划昂贵。

## 数学形式

$$
\mathcal{L} = \|\text{pred}_\phi(\text{enc}_{\text{DINOv2}}(\mathbf{o}_t), \mathbf{a}_t) - \text{enc}_{\text{DINOv2}}(\mathbf{o}_{t+1})\|^2
$$

其中 $\text{enc}_{\text{DINOv2}}$ 在训练中冻结。

## 核心要点

1. **冻结 backbone**：靠预训练的高质量 encoder 提供"已学好"的 latent
2. **不端到端**：encoder 不能从任务数据微调，对域外/新视觉风格适应差
3. **Token 多 → 规划慢**：~200 tokens × horizon × CEM samples 让 [[模型预测控制|MPC]] 计算量很大
4. **对比 LeWM**：在 Push-T / Reacher 等 2D 任务上被 LeWM 反超且 ~50× 慢

## 代表工作

- Zhou et al., 2024: DINO-WM 原始论文
- [[LeWM]]: 把 DINO-WM 作为 foundation-based 基线对比

## 相关概念

- [[DINOv2]]
- [[世界模型]]
- [[JEPA]]
- [[CEM]]
