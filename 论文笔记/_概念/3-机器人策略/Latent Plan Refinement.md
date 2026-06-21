---
type: concept
aliases: [潜在计划精炼, 潜空间动作计划迭代精炼]
---

# Latent Plan Refinement

## 定义
在 VLM 或策略网络的潜在空间中，通过多轮迭代对动作计划表示进行精炼的机制，每轮利用外部信号（如世界模型的未来预测）更新计划表示，无需解码为像素或文字。

## 数学形式

$$
z^{(k)} = z^{(k-1)} + \alpha_k \cdot \Delta z^{(k)}, \quad k = 1, \ldots, K
$$

经 K 轮精炼后：

$$
a_{t:t+H} = \pi_{\mathrm{head}}\!\left(z^{(K)}\right)
$$

## 核心要点
1. **保留低延迟**: 精炼在潜空间完成，动作解码仍为一次性并行推理
2. **避免显式推理开销**: 不依赖文字 CoT token 或像素级子目标重建
3. **可组合性**: 可与 RL（如 CRG-PRL）结合进一步优化精炼轨迹

## 代表工作
- [[PearlVLA]]: 提出 Progressive Embodied Action-Plan Refinement，K=4 轮 + RefineNet

## 相关概念
- [[VLM Latent Space Deliberation]]
- [[RefineNet (PearlVLA)]]
- [[Action Chunking]]
