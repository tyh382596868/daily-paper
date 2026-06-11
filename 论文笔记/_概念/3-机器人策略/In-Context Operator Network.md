---
type: concept
aliases: [情境算子网络, In-Context Operator Network]
---

# In-Context Operator Network（情境算子网络）

## 定义

情境算子网络是将[[In-Context Operator Learning|情境算子学习]]落地为神经网络的实现形式：以少量检索到的输入-输出对为情境提示，通过 Transformer 等架构实现无参数更新的任务适应推理。

## 核心要点

1. 情境样本通过注意力机制被网络"阅读"，隐式定义当前推理所需的函数映射
2. 推理时网络权重冻结，仅依靠情境提示实现跨任务/跨分布适应
3. 相比元学习（MAML 等），无需额外梯度步骤，推理速度更快

## 代表工作

- [[V2T-ICON]]: Video-to-Trajectory 情境算子网络，将视频帧映射为机器人状态
- [[VICX]]: 使用 V2T-ICON 实现跨任务机器人操作泛化

## 相关概念

- [[In-Context Operator Learning]]
- [[In-Context Learning]]
- [[Transformer]]
