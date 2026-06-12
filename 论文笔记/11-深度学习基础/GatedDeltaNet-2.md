---
title: "Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention"
method_name: "GatedDeltaNet-2"
authors: [Ali Hatamizadeh, Yejin Choi, Jan Kautz]
year: 2026
venue: arXiv
tags: [linear-attention, sequence-modeling, delta-rule, recurrent-state, long-context, world-model-backbone]
zotero_collection: 11-深度学习基础
image_source: online
arxiv_html: https://arxiv.org/html/2605.22791v1
created: 2026-05-24
---

# 论文笔记：Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | NVIDIA |
| 日期 | May 2026 |
| 项目主页 | N/A |
| 代码仓库 | [NVlabs/GatedDeltaNet-2](https://github.com/NVlabs/GatedDeltaNet-2) |
| 对比基线 | [[Gated DeltaNet]] / KDA (Kimi Delta Attention) / [[Mamba]]-2 / Mamba-3 (SISO & MIMO) / [[Transformer]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.22791) / [PDF](https://arxiv.org/pdf/2605.22791) / [Code](https://github.com/NVlabs/GatedDeltaNet-2) |

---

## 一句话总结

> Gated DeltaNet-2 把 [[Gated DeltaNet|GDN]] 与 KDA 中"擦除-写入"耦合的标量 [[Delta 规则|delta]] 门拆成 **channel-wise 擦除门 $\bm{b}_t$ + channel-wise 写入门 $\mathbf{w}_t$**，在 1.3B 规模上把长上下文检索（NIAH 多键检索）的精度抬到 [[线性注意力]] 家族的新 SOTA。

---

## 核心贡献

1. **解耦 erase / write 的 Gated Delta Rule-2**: 用独立的 channel-wise [[擦除门]] $\bm{b}_t$ 和 [[写入门]] $\mathbf{w}_t$ 分别控制 key 轴擦除强度与 value 轴写入强度，自然包含 [[Gated DeltaNet]] 与 KDA 作为标量退化的特例。
2. **吸收通道衰减的 [[WY 算法|WY chunkwise]] 训练算法**: 把 channel-wise 衰减 $\mathbf{D}_t$ 通过键归一化吸收为非对称的"erase factor"，在保持 $\mathcal{O}(LC d_k d_v)$ 复杂度的同时支持 chunk 并行训练。
3. **Gate-aware 反向传播**: 标量门时代可以"先做矩阵乘再乘标量"的反向技巧失效，论文显式把 $\bm{b}_t, \mathbf{w}_t, \bm{\alpha}_t$ 三路门塞进点积位置，保证梯度数值正确。
4. **1.3B 规模全面 SOTA**: 在 [[FineWeb-Edu]] 100B token 上训练，Wiki PPL、LAMBADA、Common-sense、RULER NIAH、真实世界 QA 检索全面优于 [[Mamba|Mamba-2]]、[[Gated DeltaNet]]、KDA、Mamba-3。

---

## 问题背景

### 要解决的问题

[[线性注意力]] 通过把状态压缩进固定大小的 $d_k \times d_v$ 矩阵 $\mathbf{S}_t$ 来换取常数显存与线性复杂度，但代价是 **"如何编辑这个被压缩的记忆而不破坏已有关联"** 变成了核心瓶颈。在 [[Delta 规则|delta 规则]] 视角下，每一步需要做两件事：

1. **Erase**：用 key $\bm{k}_t$ 把当前状态里跟它有关的部分擦掉一部分（避免冲突写入）
2. **Write**：把新的 value $\bm{v}_t$ 以加性形式注入

### 现有方法的局限

- **[[Gated DeltaNet]]**：用单一标量 $\beta_t \in [0,1]$ 控制 erase 与 write 强度——擦多少 = 写多少，强行绑定
- **KDA (Kimi Delta Attention)**：加了 channel-wise decay $\bm{\alpha}_t$，遗忘可以按通道精细化，但 delta 部分仍是标量 $\beta_t$
- 两者的根本约束：**erase 作用于 key 轴 ($d_k$)，write 作用于 value 轴 ($d_v$)，它们维度都不同，本不应该被同一个标量绑定**

### 本文的动机

既然 erase 和 write 在张量维度上就是不同的轴，那就给它们各自一个通道级门控——$\bm{b}_t \in [0,1]^{d_k}$ 控制每个 key 维度上的擦除强度，$\mathbf{w}_t \in [0,1]^{d_v}$ 控制每个 value 维度上的写入强度。这样既不增加额外的状态大小，又能让模型在 key/value 两个轴上独立调节"记忆编辑粒度"。

---

## 方法详解

### 模型架构

Gated DeltaNet-2 作为 [[Transformer]]-style 残差块里的 **recurrent token mixer**，每个 block 内部由以下分支构成：

- **输入**：token 序列 $\bm{x}_{1:L}$，每步通过线性投影 + 短因果卷积 + [[SiLU]] 得到 $\{\bm{q}_t, \bm{k}_t, \bm{v}_t\}$
- **稳定化**：$\bm{q}_t, \bm{k}_t$ 做 L2 归一化（保 $\|M_t\|_2 \le 1$）
- **三个独立门控分支**：
  - **衰减门** $\bm{\alpha}_t \in (0,1]^{d_k}$（继承 KDA 的 channel-wise decay）
  - **擦除门** $\bm{b}_t \in [0,1]^{d_k}$（**本文新增**）
  - **写入门** $\mathbf{w}_t \in [0,1]^{d_v}$（**本文新增**）
- **核心循环**：执行 [[Gated Delta Rule-2]] 更新状态 $\mathbf{S}_t$，再读出 $\bm{o}_t = \mathbf{S}_t^\top \bm{q}_t$
- **输出**：[[RMSNorm]] → SiLU 输出门 → 线性投影回模型维度

### 核心模块

#### 模块 1: Gated Delta Rule-2（核心循环）

**设计动机**：把 [[Delta 规则]] 里的标量 $\beta_t$ 拆成两个不同维度的向量门，让 erase 和 write 解耦。

**具体实现**：定义局部擦除向量 $\bm{e}_t = \bm{b}_t \odot \bm{k}_t$ 和局部写入向量 $\bm{z}_t = \mathbf{w}_t \odot \bm{v}_t$，先做 decay 再做擦写：

1. 先按 channel-wise decay 把上一时刻状态衰减：$\bar{\mathbf{S}}_t = \mathbf{D}_t \mathbf{S}_{t-1}$
2. 计算"读回"：$\bm{r}_t = \bar{\mathbf{S}}_t^\top \bm{e}_t$
3. 残差注入：$\mathbf{S}_t = \bar{\mathbf{S}}_t + \bm{k}_t (\bm{z}_t - \bm{r}_t)^\top$

这等价于矩阵形式的 Householder-like 更新（详见公式 1）。

#### 模块 2: Chunkwise Parallel Training

**设计动机**：纯循环训练受限于 GPU 的 sequence-level 串行，必须把序列切 chunk（$C=64$）后做 chunk-内并行 + chunk-间 scan。

**具体实现**：
- 把通道衰减 $\mathbf{D}_t$ **吸收**进归一化键 $\bar{\bm{k}}_t$ 和归一化擦除 $\bar{\bm{e}}_t$，让 chunk-内的递推保持下三角结构
- 形成辅助矩阵 $\mathbf{A} = (\mathbf{I} + \mathbf{T})^{-1}$（[[WY 算法|WY 形式]] 的逆），其中 $\mathbf{T}$ 是 chunk-内严格下三角的 $(\bar{\mathbf{K}} \bar{\mathbf{E}}^\top)$
- 通过两个辅助变量 $\mathbf{Y} = \mathbf{A} \bar{\mathbf{E}}$（erase 侧）和 $\mathbf{U} = \mathbf{A} \mathbf{Z}$（write 侧）做 chunk-并行计算
- 复杂度：$\mathcal{O}(L C d_k d_v + L d_k d_v)$，与 [[Gated DeltaNet]] 同阶

#### 模块 3: Gate-Aware Backward Pass

**设计动机**：标量门下可以把 $\beta_t$ 当系数从矩阵乘里提出来，但 channel-wise 门必须留在点积位置上。

**具体实现**：在反向累加 $d\mathbf{A}$ 时显式写出：

- $d\mathbf{A} \mathrel{+}= d\mathbf{U} \cdot (\mathbf{W} \odot \mathbf{V})^\top$（write-side）
- $d\mathbf{A} \mathrel{+}= d\mathbf{Y} \cdot (\bm{\gamma} \odot (\mathbf{B} \odot \mathbf{K}))^\top$（erase-side）

这是相对 [[Gated DeltaNet]] 反向的"主要数学改动"。

#### 模块 4: Hybrid Block 设计

把 recurrent mixer 与 [[滑动窗口注意力|Sliding-Window Attention (SWA)]] 交错堆叠：
- **Recurrent-only**: Gated DeltaNet-2 + MLP × $N$ 层
- **Hybrid**: (Gated DeltaNet-2 + MLP + SWA + MLP) × $N$ 层，SWA 窗口 2K
- 设计哲学：recurrent 负责压缩长历史到常数状态，SWA 负责局部精确检索

---

## 关键公式

### 公式 1: [[Gated Delta Rule-2|Gated Delta Rule-2 状态更新]]（核心）

$$
\mathbf{S}_t = \bigl(\mathbf{I} - \bm{k}_t (\bm{b}_t \odot \bm{k}_t)^\top\bigr) \mathbf{D}_t \mathbf{S}_{t-1} + \bm{k}_t (\mathbf{w}_t \odot \bm{v}_t)^\top
$$

**含义**：先用 channel-wise decay $\mathbf{D}_t$ 软遗忘老状态，再用 channel-wise erase $\bm{b}_t$ 擦除与当前 key 关联的部分，最后用 channel-wise write $\mathbf{w}_t$ 加性注入新 value。

**符号说明**：
- $\mathbf{S}_t \in \mathbb{R}^{d_k \times d_v}$：第 $t$ 步的紧凑循环状态矩阵
- $\bm{k}_t \in \mathbb{R}^{d_k}, \bm{v}_t \in \mathbb{R}^{d_v}$：当前 key / value 向量（已 L2 归一化）
- $\bm{b}_t \in [0,1]^{d_k}$：**channel-wise 擦除门**（key 轴）
- $\mathbf{w}_t \in [0,1]^{d_v}$：**channel-wise 写入门**（value 轴）
- $\mathbf{D}_t = \mathrm{Diag}(\bm{\alpha}_t)$：channel-wise 衰减矩阵，$\bm{\alpha}_t \in (0,1]^{d_k}$
- $\odot$：[[Hadamard 积]]（逐元素乘）

### 公式 2: [[Delta 规则]] 三步残差展开（推导视角）

$$
\begin{aligned}
\bar{\mathbf{S}}_t &= \mathbf{D}_t \mathbf{S}_{t-1} \\
\bm{r}_t &= \bar{\mathbf{S}}_t^\top \bm{e}_t \\
\mathbf{S}_t &= \bar{\mathbf{S}}_t + \bm{k}_t (\bm{z}_t - \bm{r}_t)^\top
\end{aligned}
$$

**含义**：把 Gated Delta Rule-2 拆成"先衰减、再读回旧值、再写残差"三个直观步骤；这是公式 1 的可计算形式。

**符号说明**：
- $\bm{e}_t = \bm{b}_t \odot \bm{k}_t$：局部擦除向量
- $\bm{z}_t = \mathbf{w}_t \odot \bm{v}_t$：局部写入向量
- $\bar{\mathbf{S}}_t$：衰减后但未擦写的中间状态
- $\bm{r}_t$：从中间状态里"读回"的旧 value 估计

### 公式 3: [[在线梯度下降|快速权重在线优化目标]]

$$
\mathbf{S}_t = \mathop{\mathrm{arg\,min}}_{\mathbf{S}} \bm{L}_t(\mathbf{S}), \quad \bm{L}_t(\mathbf{S}) = \|\mathbf{S} - \bar{\mathbf{S}}_t\|_F^2 - 2\langle \mathbf{S}^\top \bm{k}_t,\ \bm{z}_t - \bar{\mathbf{S}}_t^\top \bm{e}_t\rangle
$$

**含义**：Gated Delta Rule-2 等价于在局部联想记忆上做一步在线梯度下降——一阶项是"记住 $(\bm{k}_t, \bm{z}_t)$"，二阶约束是"别偏离上一步状态太远"。这给了"为什么 erase/write 应该解耦"的理论解释：两个项分别作用于不同子空间。

**符号说明**：
- $\|\cdot\|_F$：Frobenius 范数
- $\langle \cdot, \cdot \rangle$：向量内积
- 一阶项 $-2\langle \mathbf{S}^\top \bm{k}_t, \bm{z}_t - \bar{\mathbf{S}}_t^\top \bm{e}_t\rangle$：把读回误差 $\bm{z}_t - \bm{r}_t$ 当作目标残差

### 公式 4: 退化为已有方法的特例

$$
\bm{b}_t = \beta_t \mathbf{1}_{d_k}, \quad \mathbf{w}_t = \beta_t \mathbf{1}_{d_v} \implies \text{KDA}
$$
$$
\bm{b}_t = \beta_t \mathbf{1}_{d_k}, \quad \mathbf{w}_t = \beta_t \mathbf{1}_{d_v}, \quad \bm{\alpha}_t = \alpha_t \mathbf{1}_{d_k} \implies \text{Gated DeltaNet}
$$

**含义**：把两个 channel-wise 门折叠成标量，就回到 KDA；再把衰减也折叠成标量，就回到 [[Gated DeltaNet]]。说明本方法是严格的泛化。

**符号说明**：$\mathbf{1}_{d}$ 表示 $d$ 维全 1 向量。

### 公式 5: 门控参数化方式

$$
\bm{b}_t = \sigma(\mathbf{W}_b \bm{x}_t), \quad \mathbf{w}_t = \sigma(\mathbf{W}_w \bm{x}_t)
$$
$$
\bm{g}_t = -\exp(\mathbf{a}) \odot \mathrm{softplus}(\mathbf{W}_f \bm{x}_t + \bm{\delta}), \quad \bm{\alpha}_t = \exp(\bm{g}_t)
$$

**含义**：两个新门用线性投影 + [[sigmoid]] 产生（保证 $[0,1]$ 范围），衰减门用 [[Softplus|softplus]]-based 参数化保证 $\bm{\alpha}_t \in (0,1]$。

**符号说明**：
- $\sigma$：[[sigmoid]] 激活
- $\mathbf{W}_b \in \mathbb{R}^{d_k \times d_{\text{model}}}$, $\mathbf{W}_w \in \mathbb{R}^{d_v \times d_{\text{model}}}$：门控投影矩阵
- $\mathbf{a}, \bm{\delta}$：每通道可学偏置
- $\bm{g}_t$：log-decay 张量（论文里 $\bm{g}$ 专指此项，与输出门不同符号）

### 公式 6: [[Chunkwise Parallel|Chunkwise 归一化递推]]

$$
\hat{\mathbf{S}}_r = \bigl(\mathbf{I} - \bar{\bm{k}}_r \bar{\bm{e}}_r^\top\bigr) \hat{\mathbf{S}}_{r-1} + \bar{\bm{k}}_r \bm{z}_r^\top
$$

其中 $\bar{\bm{k}}_r, \bar{\bm{e}}_r$ 是把 channel-wise decay 吸收进键/擦除向量后的归一化形式。

**含义**：通过把 decay $\mathbf{D}_t$ 吸收到键-擦除两个向量里，chunk 内递推保持纯 Householder 形式，从而能用 WY 表示一次性并行求解。

**符号说明**：
- $r$：chunk 内的局部索引
- $\hat{\mathbf{S}}_r$：归一化坐标系下的中间状态

### 公式 7: [[WY 算法|WY 辅助矩阵构造]]

$$
\mathbf{A} = (\mathbf{I} + \mathbf{T})^{-1}, \quad \mathbf{Y} = \mathbf{A} \bar{\mathbf{E}}, \quad \mathbf{U} = \mathbf{A} \mathbf{Z}
$$

**含义**：构造 chunk 内的辅助矩阵，把递推变成"一次三角求逆 + 两个矩阵乘"，让 chunk 内训练完全并行。

**符号说明**：
- $\mathbf{T}$：chunk 内严格下三角 $(\bar{\mathbf{K}} \bar{\mathbf{E}}^\top)$
- $\mathbf{Y}, \mathbf{U}$：erase / write 侧的辅助矩阵

### 公式 8: Gate-aware 反向累加

$$
d\mathbf{A} \mathrel{+}= d\mathbf{U} (\mathbf{W} \odot \mathbf{V})^\top
$$
$$
d\mathbf{A} \mathrel{+}= d\mathbf{Y} (\bm{\gamma} \odot (\mathbf{B} \odot \mathbf{K}))^\top
$$

**含义**：channel-wise 门不能像标量门那样在反向里被"提出来"，必须在点积位置上保留。这是相对 GDN 反向的核心数学差异。

**符号说明**：
- $d\mathbf{A}, d\mathbf{U}, d\mathbf{Y}$：辅助矩阵的梯度
- $\bm{\gamma}$：累积 decay
- $\mathbf{B}, \mathbf{W}, \mathbf{K}, \mathbf{V}$：chunk 内的门 / 键 / 值矩阵

---

## 关键图表

### Figure 1: Hybrid 架构 + Block 设计

![[GatedDeltaNet-2_fig1_architecture.svg]]

**说明**：左侧为 Hybrid 架构图——一个重复 cell 包含 (Gated DeltaNet-2 mixer + MLP + SWA + MLP)，共 $N$ 层堆叠。右侧为 Gated DeltaNet-2 token mixer 的 block 设计：输入 $\bm{x}_t$ 经线性投影 + 短因果卷积 + SiLU 产生 $\{\bm{q}_t, \bm{k}_t, \bm{v}_t\}$，其中 $\bm{q}_t, \bm{k}_t$ 做 L2 归一化；三个独立分支分别产出 channel-wise 衰减 $\bm{\alpha}_t$、擦除门 $\bm{b}_t$、写入门 $\mathbf{w}_t$；recurrent 输出经 [[RMSNorm]] + SiLU 输出门 + 线性投影回模型维度。注意：论文中 $\bm{g}$ 专指公式 5 的 log-decay 张量，不是输出门。

### Figure 2: H100 训练吞吐对比

![[GatedDeltaNet-2_fig2_throughput.svg]]

**说明**：1.3B Hybrid 模型在单块 H100 上的训练吞吐随序列长度变化曲线。
- **Gated DeltaNet-2**：序列长度从 2K 涨到 16K 时，吞吐从 38.0 Kt/s 平滑下降到 36.1 Kt/s，几乎水平
- **Transformer**：吞吐随序列长度急剧下降（二次复杂度的代价）
- **KDA / [[Gated DeltaNet]]**：与 Gated DeltaNet-2 同档，本文的额外门控只带来很小的常数开销

结论：channel-wise 门控的精度提升几乎是"免费午餐"。

### Table 2: 语言建模 + 常识推理（1.3B / 100B tokens）

| Model | Wiki PPL ↓ | LMB PPL ↓ | LMB acc ↑ | PIQA | Hella. | Wino. | ARC-e | ARC-c | OBQA | SIQA | BoolQ | Avg ↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Recurrent** | | | | | | | | | | | | |
| Mamba-2 | 16.79 | 12.38 | 45.24 | 72.58 | 55.51 | 55.33 | 70.68 | 35.26 | 31.00 | 40.63 | 60.19 | 51.82 |
| Gated DeltaNet | 16.40 | 11.89 | 49.62 | 72.31 | 56.50 | 56.75 | 68.81 | 35.15 | 30.20 | 40.53 | 58.78 | 52.07 |
| KDA | 16.81 | 11.68 | 48.13 | 72.09 | 55.75 | 55.72 | 70.83 | 35.92 | 30.40 | 40.99 | 60.67 | 52.28 |
| Mamba-3 (SISO) | 16.30 | 12.99 | 45.06 | 72.31 | 55.58 | 56.20 | 70.45 | 34.56 | 31.00 | 41.76 | 55.90 | 51.42 |
| Mamba-3 (MIMO) | 16.45 | 11.66 | 47.82 | 72.36 | 56.49 | 55.78 | 72.38 | 38.07 | 30.00 | 40.89 | 57.74 | 52.39 |
| **GatedDeltaNet-2** | **15.90** | **11.41** | **48.09** | **72.80** | **56.84** | **57.85** | **72.43** | **38.23** | **31.60** | **40.58** | **59.54** | **53.11** |
| **Hybrid** | | | | | | | | | | | | |
| Transformer | 19.22 | 13.72 | 48.32 | 70.21 | 56.12 | 55.85 | 69.23 | 33.84 | 25.00 | 39.74 | 59.42 | 50.86 |
| Mamba-2 | 17.46 | 11.29 | 48.05 | 71.47 | 57.52 | 56.17 | 70.50 | 34.73 | 29.80 | 40.35 | 59.31 | 51.99 |
| Gated DeltaNet | 16.00 | 10.82 | 48.71 | 70.06 | 57.50 | 56.83 | 70.41 | 35.15 | 30.60 | 40.97 | 60.00 | 52.25 |
| KDA | 16.01 | 10.66 | 49.21 | 71.06 | 56.89 | 57.77 | 71.59 | 35.07 | 30.00 | 40.53 | 62.03 | 52.68 |
| Mamba-3 (SISO) | 15.54 | 10.65 | 49.19 | 71.01 | 58.75 | 57.30 | 70.54 | 36.35 | 32.00 | 41.20 | 57.86 | 52.69 |
| Mamba-3 (MIMO) | 15.81 | 10.92 | 49.82 | 71.98 | 58.19 | 57.06 | 70.54 | 38.48 | 29.40 | 40.99 | 57.98 | 52.72 |
| **GatedDeltaNet-2** | **15.62** | **10.43** | **50.90** | **72.20** | **58.46** | **58.56** | **71.89** | **36.69** | **33.00** | **41.50** | **62.57** | **53.97** |

**说明**：Recurrent 设置下 GDN-2 的 Avg 53.11 比次优的 KDA（52.28）高 0.83，比 [[Gated DeltaNet]](52.07) 高 1.04；Hybrid 设置下 53.97 vs Mamba-3 MIMO 的 52.72，gap 1.25。Wiki PPL 在两种设置下都是全场最低（15.90 / 15.62）。

### Table 3: RULER S-NIAH 合成检索（最能体现长上下文）

只列出关键长度（4K / 8K）和 multi-key 任务，完整表见原文：

| Model | S-NIAH-1 @8K | S-NIAH-2 @4K | S-NIAH-2 @8K | S-NIAH-3 @2K | S-NIAH-3 @4K | MK-NIAH-1 @4K |
|---|---|---|---|---|---|---|
| **Recurrent** | | | | | | |
| Mamba-2 | 55.8 | 62.6 | 21.0 | 38.6 | 14.4 | 21.4 |
| Gated DeltaNet | 97.6 | 87.2 | 32.0 | 54.2 | 60.6 | 27.8 |
| KDA | 70.6 | 89.0 | 30.6 | 63.2 | 26.2 | 28.0 |
| Mamba-3 (SISO) | 27.8 | 59.4 | 25.2 | 35.6 | 12.2 | 20.2 |
| Mamba-3 (MIMO) | 35.6 | 64.2 | 27.2 | 72.4 | 29.2 | 18.0 |
| **GatedDeltaNet-2** | **97.8** | **93.0** | **39.2** | **89.8** | **31.8** | **37.8** |
| **Hybrid** | | | | | | |
| Transformer | 0.0 | 44.2 | 0.0 | 94.8 | 37.0 | 38.2 |
| KDA | 26.2 | 56.0 | 23.0 | 93.4 | 51.6 | 40.4 |
| Mamba-3 (MIMO) | 22.8 | 53.0 | 27.8 | 98.4 | 54.6 | 46.6 |
| **GatedDeltaNet-2** | **27.4** | **57.9** | **29.2** | **99.0** | **55.6** | **48.0** |

**说明**：
- **Recurrent S-NIAH-3 @2K**：89.8 vs KDA 63.2 vs Mamba-3 MIMO 72.4——大幅领先
- **Recurrent MK-NIAH-1 @4K**（多键，最考验记忆编辑）：37.8 vs KDA 28.0 vs Gated DeltaNet 27.8——领先 +9.8 个百分点
- **Hybrid 8K**：纯 Transformer 在 8K context 上 S-NIAH-1 / S-NIAH-2 全崩到 0%（训练长度 4K 外推失败），而 GDN-2 hybrid 仍保持 27.4 / 29.2

### Table 4: 真实世界检索任务（2K context）

| Model | SWDE | SQD | FDA | TQA | NQ | DROP | Avg ↑ |
|---|---|---|---|---|---|---|---|
| **Recurrent** | | | | | | | |
| Mamba-2 | 17.24 | 32.38 | 14.53 | 58.35 | 18.91 | 19.60 | 26.84 |
| Gated DeltaNet | 17.90 | 32.67 | 18.52 | 59.60 | 20.16 | 19.69 | 28.09 |
| Mamba-3 (SISO) | 17.62 | 35.07 | 11.08 | 58.89 | 18.18 | 21.32 | 27.03 |
| KDA | 22.49 | 35.10 | 14.90 | 58.12 | 19.58 | 21.80 | 28.67 |
| Mamba-3 (MIMO) | 16.68 | 36.65 | 17.44 | 59.06 | 19.16 | 21.08 | 28.35 |
| **GatedDeltaNet-2** | **23.65** | **36.75** | **19.98** | **61.37** | **19.64** | **17.87** | **29.88** |
| **Hybrid** | | | | | | | |
| Transformer | 32.21 | 38.67 | 54.78 | 58.09 | 22.49 | 22.18 | 38.07 |
| Mamba-2 | 34.67 | 40.74 | 52.31 | 60.13 | 25.91 | 24.68 | 39.74 |
| Gated DeltaNet | 33.18 | 42.28 | 50.86 | 60.60 | 25.78 | 21.95 | 39.11 |
| Mamba-3 (SISO) | 35.30 | 46.42 | 54.95 | 59.54 | 25.91 | 23.96 | 41.01 |
| KDA | 39.83 | 40.10 | 53.59 | 59.89 | 25.27 | 22.18 | 40.14 |
| Mamba-3 (MIMO) | 32.33 | 44.70 | 55.31 | 59.00 | 26.26 | 23.08 | 40.11 |
| **GatedDeltaNet-2** | **41.96** | **44.70** | **54.68** | **62.38** | **26.31** | **23.67** | **42.28** |

**说明**：Recurrent 设置 29.88 vs KDA 28.67（+1.21）；Hybrid 设置 42.28 vs Mamba-3 SISO 41.01（+1.27）。SWDE（结构化文档抽取）涨幅最大（+1.16 recurrent / +2.13 hybrid），说明 channel-wise 编辑对"多键值检索"类任务收益最大。

### Table 5: 消融实验——门结构 + 擦除范围（Recurrent-Only）

| Variant | Wiki PPL ↓ | LMB PPL ↓ | Common avg ↑ | S-NIAH-2 @4K | S-NIAH-3 @2K | MK-NIAH-1 @4K | Recall avg ↑ |
|---|---|---|---|---|---|---|---|
| **Channel structure** | | | | | | | |
| w-only (标量 $\bm{b}_t$, channel $\mathbf{w}_t$) | 16.55 | 11.62 | 52.45 | 90.6 | 71.4 | 30.6 | 28.92 |
| b-only (channel $\bm{b}_t$, 标量 $\mathbf{w}_t$) | 16.12 | 11.50 | 52.79 | 92.1 | 84.6 | 35.2 | 29.51 |
| **Erase range** | | | | | | | |
| **Full (both channel, $\bm{b}_t \in [0,1]$)** | **15.90** | **11.41** | **53.11** | **93.0** | **89.8** | **37.8** | **29.88** |
| Expanded ($\bm{b}_t \in [0,2]$) | 15.95 | 11.44 | 53.04 | 93.1 | 89.4 | 37.6 | 29.81 |

**关键发现**：
1. **b-only > w-only**：channel-wise erase gate 比 channel-wise write gate 收益更大（Recall avg 29.51 vs 28.92），说明擦除粒度是主要瓶颈
2. **Full > b-only**：把 write gate 也通道化能再提 0.37（29.88 vs 29.51），尤其在 S-NIAH-3 @2K（89.8 vs 84.6 = +5.2）
3. **扩大擦除范围到 [0,2]**：在 1.3B 规模上几乎无增益（29.88 vs 29.81），保持 [0,1] 即可

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|---|---|---|---|
| [[FineWeb-Edu]] | 100B tokens | 高质量 web 教育语料 | 预训练 |
| Wikitext-103 | - | 长文本困惑度 | 评测 |
| [[LAMBADA]] | - | 句末预测 | 评测 |
| Common-sense suite | PIQA / HellaSwag / Wino. / ARC-e/c / OBQA / SIQA / BoolQ | 常识推理 | Zero-shot |
| RULER S-NIAH-{1,2,3} | 长度 1K-8K | 单键 needle-in-haystack | 长上下文检索 |
| RULER MK-NIAH-1 | 长度 1K-4K | 多键 needle-in-haystack | 抗干扰检索 |
| SWDE / SQuAD / FDA / TriviaQA / NQ / DROP | 2K context | 真实世界 QA 与抽取 | 真实检索 |

### 实现细节

- **模型规模**：1.3B 参数
- **预训练 token 数**：100B
- **训练序列长度**：4K
- **SWA 窗口**（Hybrid 时）：2K
- **优化器**：[[AdamW]]，lr = $4 \times 10^{-4}$，weight decay = 0.1
- **梯度裁剪**：1.0
- **Warm-up**：1B tokens
- **Chunk size**：$C = 64$
- **硬件**：单卡 H100 测吞吐；多卡训练（具体数量论文未明说）

### 可视化结果

- Figure 2 显示 GDN-2 在 16K context 训练吞吐仅比 2K 时下降 5%（38.0 → 36.1 Kt/s）
- Transformer 在相同条件下吞吐曲线急剧塌缩，验证了线性注意力的扩展优势

---

## 批判性思考

### 优点

1. **理论 framing 干净**：从 fast-weight / 在线优化视角解释了"erase 与 write 应该解耦"，比"加门效果更好"的纯实证立论更有说服力
2. **向后兼容**：可以直接 drop-in 替换 KDA / [[Gated DeltaNet]] 的代码，工程迁移成本低
3. **跨设置一致领先**：在 PPL、common-sense、合成 NIAH、真实 QA 四个口径上同时领先，不是"挑指标"
4. **吞吐几乎不损失**：channel-wise 门控的额外开销在 H100 上 < 5%，相比精度提升非常划算
5. **消融做了关键 cut**：b-only / w-only 直接验证了 erase 通道化是主因

### 局限性

1. **scale 单一**：只在 1.3B / 100B tokens 验证，更大规模（7B+）是否仍 outperform Mamba-3 没说
2. **跨 task transfer 没测**：language modeling SOTA 不等于做 VLA / 世界模型 backbone 时也 SOTA，需要后续工作接力
3. **erase range 扩展失败有点意外**：[0,2] 应该理论上更灵活，但增益几乎为零——可能是 1.3B 还不够大，或者 sigmoid 的尾巴在训练时被压平了
4. **decoupling 的代价没充分讨论**：两套独立门控等于增加了参数和优化变量，是否带来训练不稳定？论文没给 loss curve

### 潜在改进方向

1. **接 world model / VLA**：把 [[Frame-wise Gated DeltaNet]] 那条线（[[SANA-WM]] 用的帧级 GDN）升级成 frame-wise Gated DeltaNet-2，做长视频 / 长 horizon memory
2. **erase-aware 数据 curriculum**：既然 erase gate 是关键，可以专门构造"需要频繁覆盖记忆"的训练任务
3. **跨头共享 vs 独立**：当前 grouped value heads 下 $\bm{b}, \bm{g}$ 是跨 head 复用的，$\mathbf{w}$ 独立——这个共享策略是否最优？
4. **更精细的 chunkwise 切分**：channel-wise decay 吸收技术允许动态 chunk 大小，可以做长度自适应

### 可复现性评估

- [x] 代码开源（[NVlabs/GatedDeltaNet-2](https://github.com/NVlabs/GatedDeltaNet-2)）
- [ ] 预训练权重（论文未明确说，需查 GitHub）
- [x] 训练细节完整（hyperparameter、chunk size、token budget 都给了）
- [x] 数据集可获取（FineWeb-Edu 公开）

---

## 关联笔记

### 基于

- [[Gated DeltaNet]]: 直接前作，本文是它的 channel-wise 扩展
- [[Delta 规则]]: 状态更新的理论基础
- [[线性注意力]]: 整个家族的根
- KDA (Kimi Delta Attention): channel-wise decay 的前驱，本文补全了 channel-wise gate

### 对比

- [[Mamba]] / Mamba-2 / Mamba-3 (SISO & MIMO): 同档线性序列模型主要对手
- [[Transformer]]: 长上下文外推上的 baseline，8K 之后崩盘
- [[滑动窗口注意力|SWA]]: Hybrid 时的局部检索分量

### 方法相关

- [[WY 算法]]: chunkwise parallel 的核心数据结构
- [[RMSNorm]] / [[SiLU]]: block 内的标准化与激活
- [[Hadamard 积]]: channel-wise gate 的逐元素乘
- [[在线梯度下降]]: fast-weight 视角的理论锚点
- [[Hybrid KV Cache]]: hybrid 架构的设计哲学
- [[chunk-causal]]: chunk-内并行的因果约束

### 下游应用 / 同生态

- [[Frame-wise Gated DeltaNet]]: 把 GDN 扫描升级到帧级的视频 world model 路线（[[SANA-WM]]）
- [[潜空间世界模型]]: 长 horizon memory 模块的候选 backbone
- [[VLA]] 中的 memory module：[[MemoryVLA]]、[[VLA-JEPA]] 等若替换 mixer 可能拿到 long-horizon 提升

---

## 速查卡片

> [!summary] Gated DeltaNet-2
> - **核心**: 把 [[Gated DeltaNet]] 的标量 delta 门拆成 channel-wise 擦除门 $\bm{b}_t$ + channel-wise 写入门 $\mathbf{w}_t$
> - **方法**: Gated Delta Rule-2 + WY chunkwise 训练 + gate-aware backward
> - **结果**: 1.3B / 100B tokens 上 Wiki PPL 15.90 (recurrent) / 15.62 (hybrid)，RULER MK-NIAH-1 @4K 37.8 vs KDA 28.0
> - **代码**: https://github.com/NVlabs/GatedDeltaNet-2

---

*笔记创建时间: 2026-05-24*
