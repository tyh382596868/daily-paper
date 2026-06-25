---
type: concept
aliases: [Jacobian-Vector Product, 雅可比向量积, 正向模式自动微分, JVP]
---

# JVP（Jacobian-Vector Product）

## 定义

JVP（Jacobian-Vector Product）是正向模式自动微分（Forward-Mode AD）的核心操作：给定函数 $f: \mathbb{R}^n \to \mathbb{R}^m$ 和切线向量 $v \in \mathbb{R}^n$，计算：

$$
\text{JVP}(f, x, v) = J_f(x) \cdot v
$$

其中 $J_f(x)$ 是 $f$ 在 $x$ 处的 Jacobian 矩阵，但实际计算时**不显式构造 Jacobian**，而是沿 $v$ 方向做方向导数。

## 与反向模式 AD（VJP）的对比

| 特性 | JVP（正向模式） | VJP（反向模式 / Backprop） |
|------|---------------|------------------------|
| 计算 | $J \cdot v$（Jacobian 左乘向量） | $u^\top \cdot J$（Jacobian 右乘行向量） |
| 适用场景 | $n \ll m$（少输入多输出） | $m \ll n$（多输入少输出，典型 DL） |
| 显存 | $O(n)$ 额外切线传播 | $O(\text{激活显存})$ 用于反向 |
| 典型用途 | 切线传播、连续时间 CM | 梯度计算 |

## 在扩散蒸馏中的应用（Causal-rCM）

[[sCM]]/[[MeanFlow]] 的连续时间一致性损失需要计算沿教师 ODE 轨迹的切线：

$$
\mathbf{g} = \frac{\mathrm{d}\mathbf{f}_{\theta^-}(\mathbf{x}_t, t)}{\mathrm{d}t}
$$

这正是 $\mathbf{f}_{\theta^-}$ 对时间 $t$ 的 JVP。在 [[Causal-rCM]] 中：
- 需要通过 **TF 自定义掩码** 的 packed attention 做 JVP
- 朴素实现（unfused attention）会导致显存爆炸（需存储完整 Jacobian）
- 解决方案：扩展 [[FlashAttention-2]] 支持 JVP，将稀疏 TF 掩码表达为 admissible query-key 区间，在 fused kernel 内传播切线

## 工程挑战

**FSDP2 × JVP 兼容性**：PyTorch FSDP2 参数分片与 JVP 的 forward pass 存在冲突（all-gather 触发点不同）。Causal-rCM 采用 `FSDP2(JVP)` 设计——先分片参数再做 JVP，避免内存冲突。

**Ulysses CP × JVP**：Context Parallelism 对 token 做 all-to-all 分片，JVP 在局部 token 上计算，无额外通信开销。

## PyTorch API

```python
from torch.func import jvp

# f: 函数, x: 输入, v: 切线向量
output, tangent = jvp(f, (x,), (v,))
```

## 代表工作

- [[Causal-rCM]]: 首个在自定义掩码 FlashAttention-2 上实现 JVP 的自回归视频扩散蒸馏框架
- [[sCM]]: 连续时间 CM 需要 JVP 计算 ODE 切线

## 相关概念

- [[FlashAttention-2]]
- [[sCM]]
- [[Consistency Model]]
- [[MeanFlow]]
