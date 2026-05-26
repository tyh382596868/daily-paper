---
type: concept
aliases: [V-4D-A, Vision-to-4D-to-Action, V-4D-A 范式]
---

# V-4D-A 范式 (Vision-to-4D-to-Action)

## 定义

机器人策略学习的一种新范式：先从视觉输入构建带时间维度的 **4D 动态表征**（如 [[Gaussian Action Field|GAF]]），再从该表征导出动作。区别于：

- **V-A**：视觉直接到动作，无显式 3D
- **V-3D-A**：视觉到静态 3D 再到动作
- **V-4D-A**：视觉到动态 4D 再到动作

## 数学形式

$$
\pi: \text{Image}_t \xrightarrow{\text{4D Reconstruction}} \{\mu_t, \Delta\mu_{t \to t+\Delta t}, f_t\} \xrightarrow{\text{Flow → Action}} a_{t:t+k}
$$

## 核心要点

1. **加入时间维度**：相比静态 3D，4D 表征显式建模场景如何演化
2. **动作来自场景流**：动作不是黑盒回归，而是从可解释的位移场抽取
3. **几何约束动作**：动作的合理性由 3D 几何一致性保证
4. **双重监督信号**：当前帧重建 + 未来帧预测同时反传

## 代表工作

- [[GAF]]: 该范式的首次系统化提出，使用 [[Gaussian Action Field|GAF]] 作为 4D 表征

## 相关概念

- [[Gaussian Action Field]]
- [[3D Gaussian Splatting]]
- [[Diffusion Policy]]
- [[Act3D]]
- [[ManiGaussian]]
- [[世界模型]]
