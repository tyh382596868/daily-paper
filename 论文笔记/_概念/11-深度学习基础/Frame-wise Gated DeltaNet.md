---
type: concept
aliases: [Frame-wise GDN, 帧级 GDN, 帧级门控 DeltaNet]
---

# Frame-wise Gated DeltaNet

## 定义
把 token 级 [[Gated DeltaNet]] 扫描升级为**潜帧级扫描**的线性循环单元，每帧维护一个紧凑的 $D \times D$ 状态矩阵，是 [[SANA-WM]] 提出的长视频建模骨干。

## 数学形式

$$
\begin{aligned}
S_t &= S_{t-1} M_t + U_t \\
M_t &= \gamma_t (I - \hat{K}_t \beta_t \hat{K}_t^\top) \\
U_t &= V_t \beta_t \hat{K}_t^\top \\
O_t &= S_t \hat{Q}_t
\end{aligned}
$$

配合空间稳定缩放 $\hat{K}_t = \bar{K}_t / \sqrt{D \cdot S}$，保证 $\|M_t\|_2 \le \gamma_t \le 1$。

## 核心要点

1. **帧级扫描**: 把整帧空间 token 视作同步事件，循环只在时间维进行，长度从 $T \cdot S$ 缩减到 $T$
2. **衰减门 $\gamma_t \in (0, 1]$**: 允许遗忘老帧
3. **更新门 $\beta_{t,s} \in [0, 1]$**: [[Delta 规则|delta 修正]] 强度
4. **空间稳定**: $1/\sqrt{D \cdot S}$ 缩放使状态范数不随空间 token 数 $S$ 爆炸
5. **常数显存**: 推理时不维护 KV cache，60 秒视频仍 ~6.5 GB
6. **配合 Triton 内核**: 自定义 GDN scan / gate 内核获得 1.5-2.0× 加速

## 代表工作
- [[SANA-WM]]: 首次把 GDN 用于分钟级视频世界模型，15 GDN + 5 softmax 交错

## 相关概念
- [[Delta 规则]]
- [[线性注意力]]
- [[空间稳定的键归一化]]
- [[Context-Parallel 训练]]
- [[softmax 注意力]]
