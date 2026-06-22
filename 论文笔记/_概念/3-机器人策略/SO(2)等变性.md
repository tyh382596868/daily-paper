---
type: concept
aliases: [SO2 equivariance, 旋转等变性, 平面旋转等变]
---

# SO(2) 等变性

## 定义

若函数 $f: X \to Y$ 满足 $f(\rho_{\mathrm{in}}(g) \cdot x) = \rho_{\mathrm{out}}(g) \cdot f(x)$ 对所有 $g \in G$ 成立，则称 $f$ 关于群 $G$ **等变**。SO(2) 等变性特指二维平面旋转群 $G = \mathrm{SO}(2)$ 或其循环子群 $C_u$。

## 数学形式

$$
f(\rho_{\mathrm{in}}(g) \cdot x) = \rho_{\mathrm{out}}(g) \cdot f(x), \quad \forall g \in G
$$

其中 $\rho_{\mathrm{in}}, \rho_{\mathrm{out}}$ 为群表示，描述输入输出空间中群元素的作用方式。

## 核心要点

1. **不可约表示**：SO(2) 的不可约表示为 $\rho_\omega$（频率 $\omega$），$\omega=0$ 为不变标量，$\omega=1$ 为旋转 2D 向量
2. **正则表示** $\rho_{\mathrm{reg}}$：将群元素的作用表示为循环置换，是等变层的常用中间表示
3. **等变 vs 不变**：不变是等变的特例（$\rho_{\mathrm{out}}$ 为平凡表示 $\rho_0$）
4. **等变的好处**：旋转相关的观测和动作自动对应，无需额外数据覆盖所有旋转朝向

## 代表工作

- [[EquiVLA]]: 首个端到端 SO(2) 等变 VLA 框架，通过 EquiPerceptor + EquiActor 注入旋转等变性

## 相关概念

- [[Frame Averaging]]
- [[EquiPerceptor]]
- [[EquiActor]]
- [[VLA]]
