---
type: concept
aliases: [TTS, Test-Time Scaling, 测试时扩展]
---

# Test-Time Scaling

## 定义

Test-Time Scaling（TTS）是指在推理阶段通过额外的计算（候选采样、验证、搜索）来提升模型输出质量的一类技术。在机器人策略中，典型做法是：策略采样 $K$ 个候选动作 → 用世界模型想象对应未来 → 用价值/奖励模型评分 → 选最优执行。

## 核心要点

1. **采样 + 验证**: $a^{(1)}, \dots, a^{(K)} \sim \pi(\cdot \mid s)$，逐个用 $V$ 评分
2. **想象式验证**: 借助 [[World Action Model|世界模型]] 把候选动作"展开"成未来状态，再评分
3. **训练免费**: 不改训练流程，只在推理时增加计算
4. **K-延迟权衡**: K 越大效果越好但延迟线性增长

## 公式

$$
a^\star = \arg\max_{k \in \{1,\dots,K\}} V\big(f_{\text{wm}}(s, a^{(k)})\big)
$$

## 局限性

- 收益边际递减（如 [[WLA]] 在 LIBERO 上 TTS 仅 +0.3 pt）
- 增加 K 倍的世界模型推理开销，对实时控制不友好
- 依赖价值模型的准确性

## 代表工作

- [[WLA]]: 用 World Expert 想象未来 + Value Model 评分
- [[π0]]: 多候选动作 + 离线奖励排序
- 大模型领域: best-of-N、self-consistency、tree search

## 相关概念

- [[World Action Model]]
- [[Value Model]]
- [[Action Chunking]]
