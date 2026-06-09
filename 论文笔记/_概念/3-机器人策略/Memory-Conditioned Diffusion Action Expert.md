---
type: concept
aliases: [记忆条件扩散动作专家, memory-conditioned action expert]
---

# Memory-Conditioned Diffusion Action Expert（记忆条件扩散动作专家）

## 定义

MemoryVLA 系列中的动作生成模块：以记忆增强（和想象增强）的工作记忆 token 为条件，通过扩散变换器（DiT）经 $T$ 步去噪生成时序感知的动作块（7-DoF 末端执行器轨迹序列）。

## 数学形式

$$a_{t:t+k} = \text{DiT-Denoise}(z_T, c=[\hat{W}_t; C_t], T)$$

其中 $\hat{W}_t$ 为记忆/想象增强后的工作记忆，$C_t$ 为认知 token，$z_T$ 为初始高斯噪声。

## 核心要点

1. **扩散生成**：使用 [[DiT]] 做 DDPM/DDIM 去噪，而非 autoregressive 生成
2. **双路条件**：感知分支（$\hat{W}_t$）提供细节，认知分支（$C_t$）提供语义理解
3. **Action Chunking 输出**：一次预测 $k$ 步动作块，减少复合误差
4. **7-DoF 格式**：$(\Delta x, \Delta y, \Delta z, \Delta r_x, \Delta r_y, \Delta r_z)$ + 夹爪状态

## 代表工作

- [[MemoryVLA]]（ICLR 2026）
- [[MemoryVLA++]]（arXiv 2606.09827）

## 相关概念

- [[Diffusion Policy]]
- [[DiT]]
- [[Action Chunking]]
- [[Perceptual-Cognitive Memory Bank]]
