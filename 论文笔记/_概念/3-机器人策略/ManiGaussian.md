---
type: concept
aliases: [ManiGaussian, Mani-Gaussian]
---

# ManiGaussian

## 定义

一种用 [[3D Gaussian Splatting|3D 高斯]] 做机器人操作的代表工作：以 3D 高斯作为世界模型表征，但动作建模与重建解耦——重建网络输出 3D 高斯场，动作策略独立从高斯特征预测动作。

## 数学形式

$$
\pi: \text{Image} \to \text{3DGS} \to a
$$

重建与动作两个网络分别训练。

## 核心要点

1. **3DGS 世界模型**：用高斯泼溅做场景表征
2. **重建-动作解耦**：3D 几何与动作生成是两个独立模块
3. **静态高斯**：未建模高斯的时序运动属性
4. **被 [[GAF]] 用作主要基线**：在场景重建和成功率上对比

## 代表工作

- ManiGaussian: 提出该方法
- [[GAF]] 在 RLBench 9 任务上比 ManiGaussian 平均 +10.3% 成功率，重建 PSNR +11.5 dB

## 相关概念

- [[3D Gaussian Splatting]]
- [[Gaussian Action Field]]
- [[V-4D-A 范式]]
- [[世界模型]]
