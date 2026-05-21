---
type: concept
aliases: [Physion Benchmark, 物理推理基准]
---

# Physion

## 定义

**物理推理 (physical reasoning) 视频基准**，包含多种物理场景（Drape 布料、碰撞、滑动、容器等），用于评估模型对真实物理规律的预测与推理能力。

## 核心要点

1. **多种物理子任务**: Drape（布料形变）、Collide、Roll、Drop 等
2. **可形变 + 刚体混合**: 覆盖软体动力学
3. **视频形式**: 输入图像序列、预测未来或回答物理问题
4. **常用 Drape 子集**: 评估世界模型对 cloth 动力学的处理能力

## 代表工作

- Bear et al. (2021) Physion 原论文
- [[OrbiSim]]: 在 Physion Drape 上验证可形变物体仿真能力

## 相关概念

- [[世界模型]]
- [[AdaManip]]
- [[Embodied AI]]
