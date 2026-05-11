---
type: concept
aliases: [PLDM, Pixel Latent Dynamics Model]
---

# PLDM

## 定义

PLDM (Sobal et al., 2025) 是在 LeWM 之前**唯一**的端到端像素级 [[JEPA]] 世界模型：联合训练 encoder + predictor 在 latent 空间预测下一帧，用 [[VICReg]] + 多项辅助损失（共 7 项、6+ 超参）防止[[表征坍塌]]。训练稳定性差、调参困难，在小批量上常常崩溃。

## 数学形式

$$
\mathcal{L}_{\text{PLDM}} = \alpha \mathcal{L}_{\text{pred}} + \beta \mathcal{L}_{\text{var}} + \gamma \mathcal{L}_{\text{cov}} + \zeta \mathcal{L}_{\text{inv}} + \nu \mathcal{L}_{\text{aux1}} + \mu \mathcal{L}_{\text{aux2}} + \dots
$$

最优系数（[[LeWM]] 论文 Table 2 网格搜索）：α=18, β=12, γ=0.2, ζ=0.7, ν=0, μ=0。

## 核心要点

1. **首个端到端像素 JEPA**：但靠堆叠损失维持稳定，无理论保证
2. **超参噩梦**：6+ 损失系数互相耦合，对学习率 / batch 高度敏感
3. **训练曲线非单调**：常需早停或多次重启
4. **被 LeWM 全面超越**：单超参、单一原则的反坍塌 + 性能更强 + 速度持平

## 代表工作

- Sobal et al., 2025: PLDM 原始论文
- [[LeWM]]: 主要对比基线，用 [[SIGReg]] 替代 [[VICReg]] 多项损失

## 相关概念

- [[JEPA]]
- [[VICReg]]
- [[世界模型]]
- [[表征坍塌]]
