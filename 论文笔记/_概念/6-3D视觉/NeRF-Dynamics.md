---
type: concept
aliases: [Dynamic NeRF, 4D NeRF, NeRF Dynamics, 动态 NeRF]
---

# NeRF-Dynamics

## 定义

**NeRF-Dynamics** 指一类将经典 [[Neural Implicit Representation|NeRF]] 静态隐式辐射场扩展到**含时演化**的方法集合：场景表征从 $F_\theta(x,d) \to (c, \sigma)$ 扩展为 $F_\theta(x, d, t)$ 或 $F_\theta(x, d, a)$（动作条件），用于建模刚体运动、可变形物体、铰链关节、流体等动态过程。

## 主要分支

### 1. 时变 NeRF (Time-Conditioned)
直接把时间 $t$ 作为输入：
$$
(c, \sigma) = F_\theta(x, d, t)
$$
缺点：每段动态需重新训练，无法泛化到新轨迹。

### 2. 形变场 NeRF (Deformable / D-NeRF)
学习一个形变场 $\mathcal{D}(x, t) \to \Delta x$ 把当前坐标映射回规范空间：
$$
(c, \sigma) = F_\theta(\mathcal{D}(x, t) + x, d)
$$

### 3. 物体中心动力学 NeRF (Object-Centric Dynamic NeRF)
- 每个物体在规范帧中单独编码 NeRF
- 物体间用 SE(3) 刚体变换组合
- 代表：**CompNeRFDyn** (MRO-GWM 论文提到的最近基线)

### 4. 动作条件 NeRF
学习 $F_\theta(x, d, a_{1:T})$，输入动作序列预测未来场景；常与 [[Action-Conditioned World Model]] 结合。

## 与 [[3D Gaussian Splatting]] 动态版本的关系

| 维度 | NeRF-Dynamics | 4D Gaussian / Dynamic GS |
|------|---------------|--------------------------|
| 表征 | 隐式 MLP | 显式 Gaussians |
| 渲染速度 | 慢（体积渲染） | 快（光栅化） |
| 形变描述 | 形变场或 latent code | SE(3) 直接作用于 Gaussian 中心/旋转 |
| 多物体扩展 | 较复杂 | 天然适配（按物体分组高斯） |

MRO-GWM 等近期工作正是用 **Gaussian 取代 NeRF** 作为动态场景的基础元件，保留物体中心 + 规范帧的思想，但获得显著的渲染/训练加速。

## 关键挑战

1. **观测稀疏**: 动态场景下每个时刻视图少，单目动态重建欠约束
2. **物理一致性**: 纯数据驱动易出现非物理形变；需加刚体/接触先验
3. **长时序预测**: 隐式表征预测多步后误差累积，需世界模型架构包裹
4. **物体间交互**: 接触、碰撞难显式建模

## 代表工作

- **D-NeRF** (Pumarola 2021): 形变场分解
- **HyperNeRF** (Park 2021): 高维拓扑变化
- **CompNeRFDyn**: 多物体动作条件动力学
- **MRO-GWM** (2026): 用 Gaussian 替代 NeRF 的物体中心动力学世界模型

## 关联概念

- [[Neural Implicit Representation]]: NeRF 基础
- [[3D Gaussian Splatting]]: 显式替代方案
- [[Object-Centric Representation]]: 多物体动力学的常用基底
- [[Action-Conditioned World Model]]: NeRF-Dynamics 作为状态空间的世界模型骨架
