---
type: concept
aliases: [VICON, Vision In-Context Operator Networks]
---

# VICON（Vision In-Context Operator Networks）

## 定义

VICON（Vision In-Context Operator Networks）将[[In-Context Operator Learning|情境算子学习]]扩展到视觉观测领域，通过图像输入的情境提示来学习多物理场（multi-physics）流体动力学的预测算子，无需针对新物理参数重新训练。

## 核心要点

1. 输入为图像（而非数值或坐标），故名"Vision"情境算子网络
2. 用少量参考图像-结果对作为 prompt，适应不同物理参数的流体场预测
3. V2T-ICON（[[VICX]] 的核心组件）是 VICON 在机器人领域的延伸

## 代表工作

- VICON 原论文（arXiv:2411.16063）: 多物理流体预测
- [[V2T-ICON]] / [[VICX]]: 将视觉情境算子网络用于机器人视频-状态映射

## 相关概念

- [[In-Context Operator Learning]]
- [[In-Context Operator Network]]
- [[视频生成模型]]
