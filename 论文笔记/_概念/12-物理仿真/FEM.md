---
type: concept
aliases: [Finite Element Method, 有限元法, FEA, Finite Element Analysis]
---

# FEM（有限元法）

## 定义
有限元法（FEM / FEA）是一种数值求解偏微分方程的方法，将连续介质离散化为有限数量的单元，用于分析固体、流体、热传导等物理问题，在机器人触觉仿真中用于建模接触变形。

## 数学形式
对弹性体变形问题，求位移场 $u$ 使弱形式成立：

$$\int_\Omega \sigma(u) : \varepsilon(v) \, d\Omega = \int_\Gamma t \cdot v \, d\Gamma \quad \forall v$$

离散化后得到线性系统：$K u = f$，其中 $K$ 为刚度矩阵。

## 核心要点
1. 将连续体划分为 mesh（三角形/四面体单元），在每个单元内插值
2. 比解析解精确，适合复杂几何形状（如 GelSight 凝胶变形）
3. 商用软件：ABAQUS（TactSpace 使用）、COMSOL、Ansys
4. 在触觉传感器仿真中用于生成 tactile map → sim-to-real transfer

## 代表工作
- [[TactSpace]]: 用 ABAQUS FEM 建模 GelSight 触觉变形，结合 InfoNCE 做 sim-to-real

## 相关概念
- [[TactSpace]]
- [[Tactile Sensing]]
- [[Sim-to-Real]]
- [[InfoNCE]]
