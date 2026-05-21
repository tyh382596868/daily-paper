---
type: concept
aliases: [误差累积, Error Compounding, Compounding Prediction Error]
---

# Compounding Errors

## 定义
自回归预测/控制中，每一步的小误差被下一步当作输入并继续放大，导致 $H$ 步后的总误差**指数级增长**的现象。

## 数学形式
若预测器在状态空间是 [[Lipschitz 常数|Lipschitz]] 连续的（Lipschitz 常数 $\Lambda$），且单步误差上界为 $\delta$，则 $H$ 步开环预测的总误差满足：
$$
\|\hat{z}_{t+H} - z_{t+H}\|\ \le\ \sum_{k=1}^{H}\Lambda^{H-k}\delta = \frac{\Lambda^H - 1}{\Lambda - 1}\,\delta
$$

## 核心要点
1. **训练-推理分布不匹配** 是根本原因（见 [[Teacher Forcing]]）
2. 缓解手段：
   - **多步 rollout 训练**（K-step loss）：直接在训练时暴露累积误差，作为 distributed scheduled sampling
   - **闭环 MPC**：频繁重规划，避免开环长 horizon
   - **降低 [[Lipschitz 常数]]**：架构正则、谱归一化
3. **准确率-鲁棒性 trade-off**：K-step 训练减小 $\Lambda_K$ 但增大单步 $\delta_K$

## 代表工作
- [[JEPA-WM]]: Appendix D 形式化分析 + K-step 训练实验
- [[DreamerV3]]: imagination rollout
- [[DINO-WM]]: 也用多步 rollout 训练

## 相关概念
- [[Teacher Forcing]]
- [[Lipschitz 常数]]
- [[JEPA-WM]]
- [[MPC]]
