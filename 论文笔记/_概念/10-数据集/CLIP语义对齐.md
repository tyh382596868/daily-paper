---
type: concept
aliases: [CLIP Semantic Alignment, CLIP Score for Video]
---

# CLIP语义对齐

## 定义

CLIP 语义对齐是通过计算视频帧的视觉 CLIP 嵌入与对应文本描述的文本 CLIP 嵌入之间的余弦相似度，评估视频内容与动作字幕语义一致性的方法。

## 数学形式

$$
S_{\mathrm{clip}} = \frac{\sum_{i=1}^{T}\sum_{j=1}^{m_i}\mathrm{sim}(\mathrm{CLIP}_v(f_{i,j}), \mathrm{CLIP}_t(p_i))}{\sum_{i=1}^{T}m_i}
$$

$$
\widetilde{S}_{\mathrm{clip}} = \mathrm{clip}\!\left(\frac{S_{\mathrm{clip}} - \tau_{\min}}{\tau_{\max} - \tau_{\min}},\ 0,\ 1\right)
$$

## 核心要点

1. **帧级计算**：对每帧独立计算与对应字幕的相似度，避免时序压缩导致的信息损失。
2. **归一化处理**：通过边界值 $\tau_{\min}, \tau_{\max}$ 将原始相似度映射到 [0,1]，消除 CLIP 绝对值分布的影响。
3. **辅助角色**：在 [[交互保真度评分]] 中权重仅 λ=0.1，作为 [[MLLM评分]] 的客观补充。

## 代表工作

- [[WorldOlympiad]]: 用于交互赛道的 CLIP 语义对齐子分

## 相关概念

- [[CLIP]]
- [[交互保真度评分]]
- [[MLLM评分]]
