---
type: concept
aliases: [想象力模块, 前瞻预测模块, imagination module]
---

# Imagination Module（想象力模块）

## 定义

MemoryVLA++ 中提出的前瞻性时序建模组件：在 [[Perceptual-Cognitive Memory Bank]] 的历史检索基础上，通过轻量解码器预测未来 $n$ 步的工作记忆 token，并以辅助对齐损失训练，使 [[VLA]] 模型同时具备"回顾过去"和"预见未来"的时序建模能力。

## 数学形式

**前瞻预测**:

$$\hat{W}_{t+1}, \ldots, \hat{W}_{t+n} = \text{ImagineDecoder}(\hat{W}_t, l)$$

**想象对齐损失**:

$$\mathcal{L}_{\text{imag}} = \sum_{k=1}^{n} \left\| \hat{W}_{t+k} - \text{sg}(W_{t+k}) \right\|^2_2$$

其中 $\text{sg}(\cdot)$ 为 stop-gradient 算子。

## 核心要点

1. **前瞻补充回顾**：与 PCMB 的历史检索互补，形成双向时序建模（Memory + Imagination）
2. **训练时监督**：利用 demonstration 中的未来帧作为监督信号，训练想象 token 与真实未来对齐
3. **推理时辅助决策**：预测的未来 token 与 PCMB 检索结果拼接，共同条件化动作专家
4. **认知科学启发**：模拟人类的"心理模拟"（Mental Simulation）机制

## 代表工作

- [[MemoryVLA++]]（arXiv 2606.09827）: 首次在 VLA 中引入 Imagination Module

## 相关概念

- [[Perceptual-Cognitive Memory Bank]]
- [[MemoryVLA]]
- [[非马尔可夫任务]]
- [[VLA]]
