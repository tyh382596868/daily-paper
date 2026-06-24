---
type: concept
aliases: [STL, Signal Temporal Logic, 信号时序逻辑]
---

# STL

## 定义
一种形式化规范语言，用于描述连续信号随时间演化需满足的时序约束。广泛用于 cyber-physical system 的安全验证，近年引入机器人策略约束。

## 数学形式
STL 语法（部分）：
$$\phi ::= p \mid \neg\phi \mid \phi_1 \wedge \phi_2 \mid \mathbf{G}_{[a,b]}\phi \mid \mathbf{F}_{[a,b]}\phi \mid \phi_1 \mathbf{U}_{[a,b]}\phi_2$$

量化鲁棒性（robustness degree）$\rho(\phi, s, t)$ 可微，支持梯度引导。

## 核心要点
1. **时间界限运算符**：$\mathbf{G}_{[a,b]}$（Always）、$\mathbf{F}_{[a,b]}$（Eventually）、$\mathbf{U}_{[a,b]}$（Until）
2. **量化鲁棒性**：不只是 True/False，输出满足程度，可作为可微损失
3. 与扩散策略结合：在去噪过程中用 STL 鲁棒性引导采样方向

## 代表工作
- [[TL-DiffPolicy]]: 将 STL 引入扩散策略的推理时引导

## 相关概念
- [[策略]] — 被约束的对象
- [[Diffusion Policy]] — 结合的策略架构
