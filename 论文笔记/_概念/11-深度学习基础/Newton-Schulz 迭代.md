---
type: concept
aliases: [NS Iteration, Newton-Schulz Iteration, NS 多项式迭代]
---

# Newton-Schulz 迭代

## 定义

不显式计算 [[奇异值分解|SVD]] 而通过一个奇次多项式迭代近似[[矩阵符号函数]] $\mathrm{msign}(M) = UV^\top$ 的数值方法,只需矩阵乘法,GPU 高效。

## 数学形式

输入预归一化 $X_0 = M / (\|M\|_F + \varepsilon)$,迭代:

$$
X_{i+1} = a X_i + b X_i X_i^\top X_i + c X_i (X_i^\top X_i)^2
$$

等价于对奇异值施加多项式:

$$
\sigma \mapsto f(\sigma; a, b, c) = a\sigma + b\sigma^3 + c\sigma^5
$$

经 $k$ 步迭代,得到的 $X_k$ 近似 $\mathrm{msign}(M)$。

## 核心要点

1. **不变奇异向量**: 多项式只作用在奇异值上, $U, V$ 不变。
2. **多项式可设计**:
   - **全通** (Muon): $(a,b,c) = (3.4445, -4.7750, 2.0315)$,把 $[0,1]$ 全部拉到 1。
   - **高通** (Pion): 用 Promotion + Suppression 让大奇异值收敛 1, 小奇异值压向 0。
   - **低通** (反向消融): 让大值衰减,理论上禁止使用,实验直接发散。
3. **步数预算**: $k = 5$ 是 Muon/Pion 默认值,平衡精度与速度。
4. **数值稳定**: $\|X_0\| \le 1$ 是收敛前提,故输入须 Frobenius 归一化。

## 代表工作

- [[Muon]]: 用 NS 迭代近似 msign 实现 GPU 高效谱白化
- [[Pion]]: 把 NS 多项式重设计为高通滤波器

## 相关概念

- [[矩阵符号函数]]
- [[谱白化]]
- [[奇异值分解]]
- [[高通 NS 迭代]]
