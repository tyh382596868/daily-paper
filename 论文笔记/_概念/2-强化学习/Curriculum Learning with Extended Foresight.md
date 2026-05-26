---
type: concept
aliases: [CLEF, Curriculum Learning with Extended Foresight, 扩展视野课程学习]
---

# Curriculum Learning with Extended Foresight (CLEF)

## 定义

CLEF 是 [[X-Foresight]] 提出的两阶段 [[课程学习]] 策略，用于在长视野（21 秒）下训练 [[Chunk-Wise Autoregression|chunk-wise 自回归]] 世界模型：先用相邻 chunk（stride = 1 s）训出短视野基础能力，再把跨 chunk stride 拉到 3 s 做"chunk-wise longer foresight"。

## 核心要点

1. **两阶段**:
   - **初始阶段**: chunk 间 stride = 1 s（相邻、连续）
   - **延展阶段**: chunk 间 stride = 3 s（非相邻、跨越大时间间隔）
2. **非对称性**: 动作预测保持密集时间分辨率（即时控制），视觉观测预测则跨越较大间隔（捕捉长视野动力学）
3. **避免直接长视野发散**: 不走课程的话，从头训 H=21、stride=3 s 容易发散
4. **训练量不增**: stride 拉大不意味着 token 总数增加，因此扩展视野的计算代价可控
5. **与 [[Temporal Importance Sampling|TIS]] 互补**: CLEF 控时间结构，TIS 控样本分布

## 数学形式

设 chunk 长度 $K$、stride $s$、视野 $H$。原始序列采样模式：

$$
\{c_0, c_1, \ldots, c_{H-1}\} \quad \text{with } c_i = x_{i \cdot s \cdot K : (i \cdot s + 1) \cdot K}
$$

初始阶段 $s=1$，延展阶段 $s=3$。

## 代表工作

- [[X-Foresight]]: 首次提出，使 H=21 (21 秒) 训练可行；ablation 显示带来 collision 率 ~3% 下降

## 相关概念

- [[课程学习]]
- [[Chunk-Wise Autoregression]]
- [[Temporal Importance Sampling]]
