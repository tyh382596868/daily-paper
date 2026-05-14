---
type: concept
aliases: [Drive-JEPA]
---

# Drive-JEPA

## 定义

Drive-JEPA 是把 [[JEPA]] / [[V-JEPA]] 自监督表征学习范式迁移到 [[自动驾驶]] 端到端规划的 perception-free 方法。采用 world prediction 与 action generation **并行**分支的 [[World Action Model|WAM]] 结构，是 [[DAWN]] 的主要对比基线。

## 核心要点

1. **Perception-free**: 不显式输出 BEV 检测/分割，直接从 latent 到轨迹
2. **并行式 WAM**: world 与 action 共享 backbone 但**未递归互修**——这是 [[WAIM]] 论文指出的缺陷
3. **基准成绩**:
   - NAVSIM v1 perception-free PDMS **89.0**（DAWN 89.1）
   - NAVSIM v2 EPDMS **87.8**（仍优于 DAWN 的 83.2）

## 与 [[DAWN]] 的对比

| 维度 | Drive-JEPA | DAWN |
|------|-----------|------|
| 耦合方式 | 并行 | 递归交互 |
| Rollout | 隐式 | 显式短 latent rollout |
| 推理迭代轮数 | 1 | $K=4$ |
| NAVSIM v1 PDMS | 89.0 | **89.1** |
| NAVSIM v1 TTC | 95.5 | **96.0** |

## 相关概念

- [[JEPA]]
- [[V-JEPA]]
- [[World Action Model]]
- [[WAIM]]
- [[自动驾驶]]
