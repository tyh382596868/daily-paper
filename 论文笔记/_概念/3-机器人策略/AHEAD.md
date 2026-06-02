---
type: concept
aliases: [AHEAD, Anticipatory Horizon Extrapolation with Adaptive Dynamics]
---

# AHEAD

## 定义

CMU 2026 年提出的[[VLA]]增强方法，全称 **Anticipatory Horizon Extrapolation with Adaptive Dynamics**，在冻结的 [[OpenVLA]] 7B backbone 上加挂仅 4.9M 参数的[[潜空间世界模型]]，把决策时刻的 patch token 向前 rollout 若干步，让 VLA 在"预测出来的未来观测"上做决策，解决[[动态操作]]中的滞后问题。

## 核心组件

1. **语言-运动联合显著性**: [[CLIP]] 语言 saliency + [[RAFT]] 光流幅值，把预测限制在 5-15% 显著 token
2. **运动学条件 Flow Matching**: 用速度/加速度作为 [[Flow Matching]] 条件，物理先验显式编码
3. **自适应预测时域**: 用 ensemble disagreement 估计不确定度，超阈值停止外推
4. **冻结 VLA**: 不更新 backbone，只训 4.9M 世界模型 + 显著性投影头

## 关键指标

- 20 个仿真动态场景：79-97% 成功率（基线 31-58%）
- 真机抛物拦截 / 滚球抓取：从 0/30 提升到 19-30/30
- 端到端延迟 ~158 ms，在 200 ms 实时预算内

## 代表工作

- [[AHEAD]]（论文笔记）: 2026 arXiv 2606.02486

## 相关概念

- [[OpenVLA]]: backbone
- [[潜空间世界模型]]
- [[Flow Matching]]
- [[RAFT]]
- [[动态操作]]
- [[潜在空间预测]]
- [[显著性掩码]]
- [[自适应预测时域]]
