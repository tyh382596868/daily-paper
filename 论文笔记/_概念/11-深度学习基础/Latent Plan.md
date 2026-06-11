---
type: concept
aliases: [Latent Planning, Visual Latent Plan, 潜在规划]
---

# Latent Plan

## 定义

将高层任务信息（目标状态、演示视频、语言指令等）压缩为紧凑的低维向量表示，供下游策略解码为动作序列的中间表示；是"规划-执行"解耦架构的核心。

## 数学形式

$$
z = f_{\text{enc}}(V_{\text{demo}}, \{o^k_t\}, l), \quad \hat{a}_{t:t+H} = f_{\text{act}}(z)
$$

## 核心要点

1. **信息瓶颈**: 潜在向量 $z$ 需要捕获任务执行所需的全部语义，压缩冗余信息
2. **可解码性**: $z$ 应同时支持动作解码和辅助监督（如轨迹预测）
3. **跨体共享**: 良好的潜在规划应在不同执行机体间具有可迁移性
4. **训练解耦**: 训练时可用丰富辅助信号（轨迹、状态）监督 $z$，推理时只需 $z \to a$

## 代表工作

- [[SeeTraceAct]]: Visual Latent Plan，通过轨迹预测监督潜在规划的空间定位质量
- [[JEPA-WM]]: 基于 JEPA 的潜在世界模型
- [[Dream-exe]]: 梦境执行中的潜在规划

## 相关概念

- [[Demo-Conditioned VLA]]
- [[Visibility-Aware Trace Loss]]
- [[Action Chunking]]
