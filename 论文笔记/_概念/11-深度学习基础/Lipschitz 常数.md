---
type: concept
aliases: [Lipschitz Constant, Lipschitz 连续性]
---

# Lipschitz 常数

## 定义
若函数 $f: \mathcal{X} \to \mathcal{Y}$ 在度量空间上满足：
$$
\|f(x_1) - f(x_2)\| \le \Lambda \cdot \|x_1 - x_2\|,\quad \forall x_1, x_2 \in \mathcal{X}
$$
则称 $f$ 是 **$\Lambda$-Lipschitz 连续**的，$\Lambda$ 称为 Lipschitz 常数。

## 核心要点
1. **衡量函数对输入扰动的敏感度**：$\Lambda$ 越小，输出对输入越鲁棒
2. **误差传播**：自回归系统的 [[Compounding Errors]] 由 $\Lambda^H$ 主导
3. **神经网络的 Lipschitz 上界**：可通过谱归一化、权重正则约束
4. **应用**:
   - 世界模型稳定性分析（[[JEPA-WM]]）
   - GAN 训练稳定（Wasserstein GAN）
   - 鲁棒性证书（adversarial training）

## 代表工作
- [[JEPA-WM]]: Appendix D 用 $\Lambda_K$ 分析多步训练的鲁棒性 trade-off

## 相关概念
- [[Compounding Errors]]
- [[Teacher Forcing]]
- [[JEPA-WM]]
