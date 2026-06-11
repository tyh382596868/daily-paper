---
type: concept
aliases: [动作头, Policy Head, Action Decoder]
---

# Action Head

## 定义

VLA 或机器人策略模型中负责将特征表示（如潜在规划向量、视觉-语言特征）映射到具体动作输出的模块；通常为 MLP、Transformer 解码器或扩散头。

## 核心要点

1. **输入**: 骨干网络输出的特征向量或潜在规划向量 $z$
2. **输出**: 连续动作（末端执行器位姿增量 + 夹爪状态）或离散化 token
3. **常见设计**:
   - MLP 回归头（直接输出）
   - [[Action Chunking]] 头（一次输出 $H$ 步动作块）
   - 扩散头（用扩散过程建模动作分布）
4. **与骨干解耦**: 允许冻结骨干、仅微调 Action Head 以适应新任务

## 代表工作

- [[SeeTraceAct]]: 从潜在规划 $z$ 直接回归动作块
- [[ACT (Action Chunking Transformer)]]: 最早系统化使用 Action Chunking 的方法

## 相关概念

- [[Action Chunking]]
- [[VLA]]
- [[Demo-Conditioned VLA]]
