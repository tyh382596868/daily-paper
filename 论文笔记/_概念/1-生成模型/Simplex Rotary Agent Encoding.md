---
type: concept
aliases: [Simplex RoPE, 单纯形旋转位置编码, Simplex Rotary Encoding]
---

# Simplex Rotary Agent Encoding

## 定义

把多 agent 的身份编码扩展到 [[RoPE]] 体系中：在 3D RoPE 的 (时间, 空间) 三轴上额外加入一个 **agent 轴**，每个 agent 的 agent-band 旋转相位由**正单纯形顶点**给出，使所有 agent 两两等距、置换等价、parameter-free，且无需学习 per-slot 嵌入。

## 数学形式

给定单纯形池大小 $V \le d_p/2 + 1$，构造单位长度顶点：

$$
\mathbf{s}_v = \sqrt{\frac{V}{V-1}}\, \mathbf{Q} \left( \mathbf{e}_v - \frac{1}{V}\mathbf{1} \right) \in \mathbb{R}^{d_p / 2}
$$

满足等距性 $\|\mathbf{s}_v - \mathbf{s}_{v'}\|_2^2 = \frac{2V}{V-1}, \forall v \ne v'$。

agent $p$ 的 agent-band 旋转角为 $\theta_p = \alpha\, \mathbf{s}_{\pi(p)}$，完整 4D RoPE 算子：

$$
\mathbf{R}_{\text{simp-4D}}(t, p, h, w) = \text{diag}\bigl(\mathbf{R}_t(t), \mathbf{R}_{\text{simp}}(\pi(p)), \mathbf{R}_h(h), \mathbf{R}_w(w)\bigr)
$$

## 核心要点

1. **置换对称**：所有 agent 顶点两两等距，不同 agent 对的旋转距离相同，没有"特殊槽位"。
2. **参数无关**：完全由单纯形几何构造，不需要任何额外可学习参数。
3. **agent-count 可扩展**：训练时随机激活 $P \le V$ 个顶点，推理时启用未用顶点即可加 agent，不需要改架构、不需要重训。
4. **对比**：
    - 1D 标量 $\theta_p = p\omega$：把 agent 放在直线上，距离取决于 $|p-q|$，破坏对称。
    - 学习的 per-slot ID 嵌入：绑定固定 roster，破坏对称且不可扩展。
5. **对已有 DiT 的迁移**：按 [[ReRoPE]] 思路从时间 band 低频端"借" $d_p$ 维，不破坏空间 / 高频时间。
6. 复 RoPE 空间下，小 $\alpha$ 近似仍保持等距：$\|\Phi_p - \Phi_q\|_2^2 \approx \alpha^2 \cdot \frac{2V}{V-1}$。

## 代表工作

- [[Gamma-World]]: 首次提出，并在 Cosmos-Predict2.5-2B 上 $V=4$ 实现 2-agent 训练、4-agent 零样本生成

## 相关概念

- [[RoPE]]
- [[ReRoPE]]
- [[Sparse Hub Attention]]
- [[DiT]]
- [[World Model]]
