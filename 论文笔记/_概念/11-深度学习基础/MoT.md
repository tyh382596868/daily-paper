---
type: concept
aliases: [Mixture-of-Transformers, Mixture of Transformers]
---

# MoT

## 定义

Mixture-of-Transformers — 类似 [[MoE]] 但在 Transformer 层级做模态特化的架构。多个"专家 Transformer"各自处理一种模态（视频/动作/语言），通过 layer-wise 交叉注意力或共享自注意力交换信息。

## 数学形式

$$
(h^v_{\ell+1},\ h^a_{\ell+1}) = F^{mix}_\ell(h^v_\ell,\ h^a_\ell;\ o_t,\ l)
$$

其中 $h^v, h^a$ 是不同专家的隐状态，$F^{mix}_\ell$ 是层内融合算子（cross attention / joint self-attention）。

## 核心要点

1. 比单 backbone 联合训练更能保留模态特化的优化路径
2. 比完全解耦的两阶段架构有更强的跨模态信息交换
3. 在机器人世界模型中常用于"视频专家 + 动作专家"的耦合
4. 与 [[MoE]] 的区别：MoE 是 sparse routing；MoT 是 dense expert per modality

## 代表工作

- GE-Act, Motus, LingBot-VA, BagelVLA, Fast-WAM, LDA-1B, FRAPPE, DiT4DiT — [[RobotWM-Survey]] Section 3.4

## 相关概念

- [[MoE]]
- [[扩散变换器]]
- [[JointSelfAttention]]
- [[RobotWM-Survey]]
