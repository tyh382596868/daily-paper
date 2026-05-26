---
type: concept
aliases: [High-pass Newton-Schulz, High-pass NS Iteration, 高通 Newton-Schulz]
---

# 高通 NS 迭代

## 定义

把 [[Newton-Schulz 迭代]]的奇次多项式系数从 [[Muon]] 默认的"全通"重设计为"高通"——大奇异值收敛到 1, 小奇异值被压向 0, 等价于在奇异值频谱上做高通滤波。是 [[Pion]] 优化器的核心机制。

## 数学形式

5 步 NS 拆为 $k_p$ 步 Promotion + $k_s = 5 - k_p$ 步 Suppression。

**Promotion** (单调上抬,保序):

$$
f_p(\sigma) = 1.875\,\sigma - 1.25\,\sigma^3 + 0.375\,\sigma^5
$$

**Suppression** (锚定大值,压低小值):

$$
f_s(\sigma) = 2.5\,\sigma^3 - 1.5\,\sigma^5,\quad f_s'(0) = 0
$$

## 核心要点

1. **设计目标**:
   - $f_p$ 单调增、 $f_p(0)=0$、 $f_p(1)=1$ -> "提升"信号但保留序关系;
   - $f_s$ 在 0 附近导数为 0 -> 二次以上压制噪声方向;
   - $f_s(1)=1$ -> 大奇异值锚定。
2. **应用场景**: 低[[有效秩]] (VLA 动作头) / 低[[梯度信噪比]] (RLVR 后训练)。
3. **反向消融失败**: 把高通换成低通直接发散 -> 高通方向才是必要的。
4. **零开销**: 仍 5 步 NS,只是换系数,与 [[Muon]] 等开销。

## 代表工作

- [[Pion]]: 提出 Promotion+Suppression 两阶段高通设计

## 相关概念

- [[Newton-Schulz 迭代]]
- [[Pion]]
- [[Muon]]
- [[谱白化]]
- [[有效秩]]
- [[梯度信噪比]]
