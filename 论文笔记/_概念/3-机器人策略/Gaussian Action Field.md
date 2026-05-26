---
type: concept
aliases: [GAF, Gaussian Action Field, 高斯动作场]
---

# Gaussian Action Field (GAF)

## 定义

把 [[3D Gaussian Splatting|3D 高斯]] 表征扩展为 4D：在每个高斯原有的位置 $\mu$、外观 $\{c, \sigma, r, s\}$ 之外，再附加一个**可学习的位移属性** $\Delta\mu$，让单一表征同时承载"当前几何 + 外观 + 短期运动"。

## 数学形式

$$
\mathcal{F}_\Theta : \{g(x), t\} \mapsto \{\mu, \Delta\mu, f\}
$$

其中 $g(x)$ 是高斯索引，$\mu$ 是当前位置，$\Delta\mu$ 是 $t$ 到 $t+\Delta t$ 的位移，$f = \{c, \sigma, r, s\}$ 是外观参数。

## 核心要点

1. **4D 表征**：从静态 3DGS 升级为带时间维度的动态高斯场
2. **位移驱动动作**：夹爪相关高斯的位移直接通过 [[ICP]] 抽取为刚体变换序列，作为初始动作
3. **双帧渲染监督**：用 $\mu$ 渲染当前帧、用 $\mu + \Delta\mu$ 渲染未来帧，让位移属性靠图像监督学到
4. **几何 + 动作统一**：场景重建、未来预测、动作生成共用同一组高斯参数

## 代表工作

- [[GAF]]: 提出该表征，用 V-4D-A 范式做机器人操作

## 相关概念

- [[3D Gaussian Splatting]]
- [[V-4D-A 范式]]
- [[动作-视觉对齐去噪]]
- [[世界模型]]
