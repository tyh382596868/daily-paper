---
type: concept
aliases: [KL Distillation, KL散度蒸馏, multi-teacher KL distillation]
---

# KL 蒸馏

## 定义
通过最小化学生策略与教师策略输出分布之间的 KL 散度，将多个教师模型的知识迁移到单一学生模型的训练方法。

## 数学形式
$$\mathcal{L}_{\text{KL}} = \sum_i w_i \cdot D_{\text{KL}}(\pi_{\text{teacher}_i}(\cdot | s) \| \pi_{\text{student}}(\cdot | s))$$

其中 $w_i$ 为各教师的权重（可由 context-conditioned gating 决定），$\pi_{\text{teacher}_i}$ 为第 $i$ 个教师策略的动作分布。

## 核心要点
1. 相比硬标签监督，KL 蒸馏保留了教师的完整概率分布（软标签），信息更丰富
2. 多教师场景下，可用 mixture-of-experts gating 动态分配不同教师的权重
3. 显式近端约束（proximal regularizer）可防止学生策略偏离参考分布过远
4. 常用于仿人机器人控制中融合步态、操作、跌倒恢复等不同专家策略

## 代表工作
- [[HANDOFF]]：三位互补教师（全身运动跟踪、步态、跌倒恢复）通过多教师 KL 蒸馏训练单一 MoE 控制器
- [[FlowPRO]]：近端正则化器锚定 KL 蒸馏中的隐式奖励幅值

## 相关概念
- [[MoE]]（Mixture-of-Experts，常与多教师蒸馏配合使用）
- [[Knowledge Distillation]]（一般知识蒸馏）
- [[强化学习]]（策略蒸馏的常用框架）
