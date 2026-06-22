---
type: concept
aliases: [几何融合多视图3D重建, 几何感知重建]
---

# GeoFuse-MV3D

## 定义

GeoFuse-MV3D 是 GeneralVLA-2 中提出的多视图 3D 重建分支，通过将外部几何先验（如 VGGT）与遮罩验证、外观仿射校准、软视觉外壳和逐轴精细化相结合，保守地修正 [[3D Gaussian Splatting]] 表示，提升机器人操作所需的重建精度。

## 数学形式

遮罩一致性分数（验证几何可信度）：

$$
s(p) = \frac{1}{\max(|V(p)|, 1)} \sum_{i \in V(p)} M_i(\pi_i(p))
$$

置信度加权残差融合：

$$
x_{out}^j = x_A^j + \alpha w^j (x_B^j - x_A^j), \quad w^j = \mathrm{clip}(s(x_A^j) s(x_B^j), 0, 1)
$$

## 核心要点

1. **两条互补几何路径**: 来源 A（外部几何先验+外观仿射校准）和来源 B（无先验逐轴补偿）提供正交修正
2. **保守融合**: 对低遮罩支持度点施加向内收缩而非删除，避免损害操作关键几何
3. **输入协议不变**: 与 MV-SAM3D 基线完全相同的输入格式，零侵入集成

## 代表工作

- [[GeneralVLA2]]: 提出 GeoFuse-MV3D 的论文，在 GSO-30 上 CD 降低 2.20%，PSNR 提升 2.36%

## 相关概念

- [[3D Gaussian Splatting]]
- [[VGGT]]
- [[Governed KnowledgeBank]]
