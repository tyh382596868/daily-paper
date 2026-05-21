---
type: concept
aliases: [Product of Experts, PoE, 专家乘积, 对比专家乘积, PoCE, Product of Contrastive Experts]
---

# Product of Experts

## 定义
Product of Experts (PoE) 是一种把多个专家分布相乘并归一化来组合模型的方法——只有所有专家都认可的区域才获得高概率，等价于「所有专家投票一致」。

## 数学形式
朴素 PoE：

$$
p(\mathbf{x}) \propto p_{\text{query}}(\mathbf{x})\prod_{i} p_i(\mathbf{x})
$$

对比专家乘积 (PoCE)，引入对比系数 $\alpha_i>1$：

$$
\tilde{p}_i(\mathbf{x}) \propto p_i(\mathbf{x})^{\alpha_i}\,\bar{p}_i(\mathbf{x})^{1-\alpha_i}
$$

## 核心要点
1. 分布相乘聚焦于所有专家的高置信交集，支持推理时组合、无需联合重训。
2. **朴素 PoE 的问题**：当多个专家共享虚假模式时会过度锐化共识区域、放大虚假峰，导致方差收缩与模式坍塌。
3. **对比专家乘积 (PoCE)**：用对比系数把条件分布相对其无条件基线 $\bar{p}_i$ 放大，抑制虚假共识模式而不损失多样性（[[CoME]] 提出，配 Proposition 1 理论保证）。
4. 取对数后乘积变求和、幂次变系数，因此对[[Score Function|得分函数]]/扩散模型天然适配。

## 代表工作
- [[CoME]]: 提出对比专家乘积 (PoCE) 组合多个记忆专家

## 相关概念
- [[Score Function]]
- [[记忆专家]]
- [[Mixture of Contrastive Experts]]
- [[MoE]]
