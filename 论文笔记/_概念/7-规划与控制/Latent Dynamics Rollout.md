---
type: concept
aliases: [Latent Dynamics Rollout, Latent Rollout, 潜空间展开]
---

# Latent Dynamics Rollout

## 定义

[[Latent MPC]] 中，把候选动作序列 $\mathbf{a}_{0:H-1}$ 送进**冻结的 latent dynamics 模型** $F_\theta$，从初始 latent $\mathbf{z}_t$ 自回归预测得到终端 latent $\hat{\mathbf{z}}_{t+H}$ 的过程。

## 数学形式

$$
\hat{\mathbf{z}}_{t+H} = F_\theta(\mathbf{z}_t, \mathbf{a}_{0:H-1})
$$

或写成自回归递推:

$$
\hat{\mathbf{z}}_{k+1} = F_\theta(\hat{\mathbf{z}}_k, a_k), \quad \hat{\mathbf{z}}_0 = \mathbf{z}_t
$$

## 核心要点

1. **完全在 latent 空间**: 不解码到像素，速度极快
2. **误差累积**: 自回归预测 $H$ 步后误差会复合放大，限制了长 horizon 规划
3. **典型 horizon**: [[LeWM]] / [[PLDM]] 用 $H \in [10, 50]$
4. **冻结假设**: TRM 之类的接口层修复**不动** $F_\theta$，只换终端代价

## 代表工作

- [[LeWM]]: 6 层 16 头 Transformer predictor + AdaLN 动作注入
- [[PLDM]]: 端到端预测器
- [[DreamerV3]]: RSSM (Recurrent State-Space Model)

## 相关概念

- [[Latent MPC]]
- [[World Model]]
- [[MPC]]
