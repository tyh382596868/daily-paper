---
type: concept
aliases: [Rectified Flow, RF, 整流流]
---

# Rectified Flow

## 定义

Rectified Flow (Liu et al., 2022) 是一种把扩散过程**线性化**的生成范式：沿数据样本 $y_0$ 到高斯噪声 $y_1$ 的**直线插值** $y_t = (1-t) y_0 + t y_1$ 构造训练对，学习一个速度场 $v_\theta$ 预测直线方向 $y_1 - y_0$。推理时反向积分即可生成数据，理论上比 DDPM 步数更少且采样路径更直。

## 数学形式

**插值轨迹**:
$$
y_t = (1-t) \cdot y_0 + t \cdot y_1, \quad y_0 \sim p_{data}, \; y_1 \sim \mathcal{N}(0, I), \; t \sim \mathcal{U}(0, 1)
$$

**速度匹配损失**:
$$
\mathcal{L}_{velocity}(\theta) = \mathbb{E}_{y_0, y_1, t, c} \left[ \left\| v_\theta(y_t, t, c) - (y_1 - y_0) \right\|_2^2 \right]
$$

**推理（反向 ODE）**:
$$
\frac{dy_t}{dt} = v_\theta(y_t, t, c), \quad y_1 \sim \mathcal{N}(0, I), \; t: 1 \to 0
$$

## 核心要点

1. **直线路径**: 训练目标是常值速度 $(y_1 - y_0)$，理论上推理只需少量 ODE 步
2. **与 [[Flow Matching]] 等价**: 是 Flow Matching 的一个特例（线性概率路径）
3. **与扩散等价**: 在某些参数化下等价于 v-prediction，但训练更稳定
4. **重整化（Re-Flow）**: 可以用训好的模型再生成数据对，进一步把路径"拉直"
5. **生产部署主流**: SD3 / FLUX / X-World / Cosmos 等都用 RF

## 代表工作

- Liu et al., "Flow Straight and Fast" (ICLR 2023)
- [[X-Foresight]] Vision Renderer / [[X-World]]: 用 RF 训练 [[DiT]] 视频渲染
- SD3 / FLUX.1: 高质量图像生成
- [[FlowMatching]] / [[Flow Matching]]: 更一般的概率路径框架

## 相关概念

- [[Flow Matching]]
- [[扩散模型]]
- [[DiT]]
- [[X-World]]
