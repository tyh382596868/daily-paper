---
type: concept
aliases: [Object-Centric, 物体中心表示, 对象中心表征, OCR]
---

# Object-Centric Representation

## 定义

一种将场景分解为**离散物体单元**的表征范式：每个物体由一组与之**绑定**的潜变量（属性/几何/位姿/外观）描述，且物体之间共享统一的处理结构。区别于像素级或场景全局表征——它显式编码"什么是物体"以及物体之间的可组合关系。

## 核心特点

1. **可分解性**: 场景 = 物体集合 $\\{o_1, o_2, \dots, o_N\\}$，每个物体独立编码
2. **对称性**: 物体顺序不影响整体语义，常用 [[Transformer|集合-自注意力]] 处理
3. **可组合泛化**: 训练时见过的物体类型可重新组合成新场景
4. **动力学解耦**: 每个物体可被赋予独立刚体/铰链动力学，自然支持多物体物理预测

## 典型形式

- **离散槽位 (Slot Attention)**: 每个 slot 通过竞争式注意力对应一个物体区域
- **以 mask 为先验**: 利用 SAM / 实例分割提供物体 ID，再编码物体特征
- **物体锚点 (Object Anchors)**: 在物体规范坐标系中放置一组点/Gaussian/锚点，整体被 SE(3) 变换控制（MRO-GWM 的做法）

## 数学描述

物体 $i$ 的世界态由规范帧表示 $g_i^{can}$ 与 SE(3) 位姿 $\xi_i$ 组合：

$$
g_i^{world} = \xi_i \cdot g_i^{can}, \quad \xi_i \in \mathrm{SE}(3)
$$

物体间交互通过相对位姿差 $\xi_i^{-1}\xi_j$ 或 k-近邻注意力实现。

## 优点 vs. 全局表征

| 维度 | Object-Centric | Global (单 latent) |
|------|---------------|--------------------|
| 多物体可扩展性 | 强（按物体数线性增长） | 弱（信息瓶颈） |
| 物理可解释 | 高（显式刚体变换） | 低 |
| 数据效率 | 高（共享物体模板） | 低 |
| 渲染细节 | 看物体表征丰富度 | 全局难独立编辑 |

## 代表工作

- **Slot Attention** (Locatello 2020): 离散槽位的经典实现
- **MRO-GWM** (2026): 物体中心 + Gaussian + 规范帧 + SE(3) 世界模型
- **CompNeRF / CompNerfDyn**: 组合式 NeRF 表征多物体动力学
- **DreamerV3-Object**: 物体中心的潜空间世界模型变体

## 关联概念

- [[3D Gaussian Splatting]]: 可在物体规范帧中放置高斯，组成 Object-Centric Gaussian
- [[Action-Conditioned World Model]]: 物体中心表征常作为多物体动力学预测的基底
- [[Spatial Consistency]]: 物体内部一致性 + 物体间相对位姿构成总场景
