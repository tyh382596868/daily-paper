---
type: concept
aliases: [MSAT, Multi-Stream Action Transformer, 多流动作 Transformer]
---

# Multi-Stream Action Transformer (MSAT)

## 定义
MSAT 是 RLDX-1 提出的动作模型架构，为异构感知输入（视觉语言上下文、本体感觉/动作、物理信号）设计独立的专用流，并通过联合自注意力实现跨流信息融合，同时保留各模态专用表示。

## 数学形式

三流输入：
- **认知流**: $\mathbf{c}_t$（VLM 提取的认知特征）
- **动作流**: $[\mathbf{q}_t,\, \mathbf{a}^\tau_{t:t+H}]$（本体感觉 + 含噪动作）
- **物理流**: $\mathbf{p}^\tau_{t+1:t+L}$（触觉/力矩信号，可选）

联合自注意力（跨流交互）：

$$
[\mathbf{c}',\, \mathbf{a}',\, \mathbf{p}'] = \text{JointSelfAttention}([\mathbf{c},\, \mathbf{a},\, \mathbf{p}])
$$

## 核心要点

1. **模态专用流**: 各流使用独立的投影层处理不同维度和语义的输入，避免跨模态特征污染
2. **联合自注意力**: 拼接所有流的 token 后做统一自注意力，允许自由的跨流信息交互
3. **物理流可选**: 无触觉/力矩传感器时物理流可省略，模型退化为标准双流（认知+动作）
4. **异构尺度处理**: 认知流（高维语义）与动作流（低维连续）的维度差异通过独立投影层对齐

## 代表工作

- [[RLDX-1]]: 提出 MSAT 架构，用于仿人灵巧操纵

## 相关概念

- [[联合自注意力|Joint Self-Attention]]
- [[Flow Matching]]
- [[Action Chunking]]
- [[VLA]]
