---
type: concept
aliases: [Pion Optimizer, 高通 Muon]
---

# Pion

## 定义

[[Muon]] 的后训练改进版,用两阶段"Promotion + Suppression"高通 [[Newton-Schulz 迭代]]替换 Muon 的全谱白化,在保持单步开销不变的前提下实现"保留主奇异方向 + 抑制小奇异方向(噪声)"。

## 数学形式

5 步 NS 迭代拆分为 $k_p$ 步 Promotion + $k_s = 5 - k_p$ 步 Suppression。

**Promotion**:
$$
f_p(\sigma) = 1.875\,\sigma - 1.25\,\sigma^3 + 0.375\,\sigma^5,\quad f_p'(\sigma) = 1.875(1-\sigma^2)^2 \ge 0
$$

**Suppression**:
$$
f_s(\sigma) = 2.5\,\sigma^3 - 1.5\,\sigma^5,\quad f_s'(0) = 0
$$

组合后整体效果: 大奇异值锚定 1, 小奇异值压向 0 -> 高通滤波。

## 核心要点

1. **诊断驱动**: VLA 动作头梯度低[[有效秩]]、RLVR 梯度低[[梯度信噪比]],均要求滤掉小奇异方向。
2. **Promotion 单调上抬**: 保留序关系,为后续抑制步做准备。
3. **Suppression $f_s'(0) = 0$**: 噪声方向二次以上抑制,这是高通滤波核心。
4. **Per-Head 模式**: 把注意力 $W \in \mathbb{R}^{d \times d}$ reshape 成 $H$ 个 head 独立做高通,保护预训练 head 异质性。
5. **零额外开销**: 与 Muon 等开销 (同样 5 步 NS),只是换系数。
6. **结果**: LIBERO Object 1500 步 100%, 真实机器人 85.6%, RLVR 8 设置全胜 [[AdamW]] 而 Muon 全坍塌。

## 代表工作

- [[Pion]] (本论文 *Rethinking Muon Beyond Pretraining*, 2026): 提出该优化器

## 相关概念

- [[Muon]]
- [[Newton-Schulz 迭代]]
- [[高通 NS 迭代]]
- [[谱白化]]
- [[有效秩]]
- [[梯度信噪比]]
