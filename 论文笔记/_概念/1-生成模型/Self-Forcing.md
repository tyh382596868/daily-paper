---
type: concept
aliases: [Self-Forcing Training, On-Policy Distillation]
---

# Self-Forcing

## 定义
autoregressive 视频扩散模型的在线训练范式：模型用自己当前的生成结果作为下一段预测的条件，对应反向散度（reverse KL）优化，等价于 on-policy 策略蒸馏（DMD 风格）。

## 核心要点
1. **对比 Teacher-Forcing**：Teacher-Forcing 用 ground-truth 历史帧条件化（offline，正向散度）；Self-Forcing 用模型自己生成的历史帧（online，反向散度）
2. 自蒸馏特性：避免 exposure bias（推理时没有 GT 可用），提升 AR 生成连贯性
3. 缺点：on-policy 采样计算代价高，且容易 mode collapse（反向 KL 的特性）
4. 在 [[Causal-rCM]] 框架里：Self-Forcing = DMD 风格精修阶段，Teacher-Forcing 提供初始化

## 数学形式
$$\mathcal{L}_\text{SF} = \mathbb{E}_{x \sim p_\theta}\left[\text{score}(x)\right]$$
（反向 KL 方向：$D_\text{KL}(p_\theta \| p_\text{data})$）

## 代表工作
- [[Causal-rCM]]: TF+SF 互补蒸馏框架

## 相关概念
- [[Teacher Forcing]]
- [[DMD]]
- [[Consistency Model]]
