---
type: concept
aliases: [Chameleon Model, Chameleon 7B]
---

# Chameleon

## 定义

Chameleon 是 Meta 提出的原生多模态基础模型，将文本和图像统一在单一的 token 序列框架下，使用 VQ-GAN 将图像离散化为 token，与文本 token 混合输入 Transformer。

## 数学形式

图像离散化（VQ-GAN 编码）：

$$
z_q = \text{VQ-GAN}_{\text{enc}}(I), \quad z_q \in \{1, \ldots, V\}^{N_{\text{img}}}
$$

其中 $V$ 为码本大小，$N_{\text{img}}$ 为图像 token 数。

## 核心要点

1. **统一 token 化**: 文本 BPE token 和图像 VQ-GAN token 共享同一词表和 Transformer 序列，实现真正的混合模态输入
2. **Causal LM 架构**: 32 层 Transformer，标准自回归训练，支持文本/图像的任意混合生成
3. **多模态理解与生成**: 可同时进行图像理解（问答）和图像生成任务
4. **作为骨干网络**: 在 OA-WAM 等机器人策略中用作冻结骨干，通过 LoRA 微调适配下游任务

## 代表工作

- [[OA-WAM]]: 以 Chameleon 7B 为冻结骨干，添加槽适配器和动作/世界预测头

## 相关概念

- [[VLA]]
- [[World Action Model]]
- [[Flow Matching]]
