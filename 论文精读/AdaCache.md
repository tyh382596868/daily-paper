---
type: deep-reading
title: "Adaptive Caching for Faster Video Generation with Diffusion Transformers"
method_name: "AdaCache"
authors: [Kumara Kahatapitiya, Haozhe Liu, Sen He, Ding Liu, Menglin Jia, Michael S. Ryoo, Tian Xie]
year: 2024
venue: arXiv
arxiv: https://arxiv.org/abs/2411.02397
tags: [video-diffusion, inference-acceleration, caching, diffusion-transformer, training-free, motion-regularization]
image_source: pending
source: arxiv-html
created: 2026-06-22
---

# 精读：Adaptive Caching for Faster Video Generation with Diffusion Transformers

> **一句话**：不同视频内容复杂度不同，AdaCache 用内容自适应的残差缓存调度，让每个视频按自身复杂度决定重算频率，达到 4.7× 推理加速而不损画质。

---

## TL;DR（30 秒）

- **解决什么问题**：视频 DiT 推理太慢——每个去噪步都要对所有 Transformer 层做全量计算，即使相邻步之间特征几乎没变化。
- **核心 idea**：测量相邻去噪步的特征"变化速率"，自适应地决定当前步缓存上一步残差还是重新算——静态帧的视频可以复用更多，运动剧烈的视频多算几步。
- **为什么有效**：视频 DiT 的 Temporal Self-Attention 输出在相邻去噪步之间变化极其平滑，直接复用误差极小；而不同视频的平滑程度差异巨大，统一调度必然顾此失彼。
- **最强证据**：Open-Sora 720p-2s 视频生成，VBench 从 84.16 仅降至 83.40（-0.76），但推理速度提升 **4.7×**（419s → 89s on A100）。
- **代码/主页**：https://adacache-dit.github.io/

---

## 0. 阅读前置（先懂这些再往下）

读懂本文你需要先理解：

- [[DDPM]] / [[Diffusion Model]] — 扩散模型的去噪过程：推理时要跑 T 步去噪，每步都调一次神经网络。AdaCache 的加速正是建立在"能不能跳过某些层的计算"这个问题上。
- [[DiT]] / [[Diffusion Transformer]] — 把 U-Net 换成 Transformer 做扩散模型的骨架；视频 DiT 在帧序列的 token 上做时序+空间 attention。
- [[Spatio-Temporal Attention]] — 视频 DiT 的关键组件：在时间维度上做 self-attention（STA）来建模帧间一致性，这正是 AdaCache 主要缓存的对象。
- [[VBench]] — 视频生成的多维度评测 benchmark，涵盖动态程度、美学、语义一致性等，是本文主要质量指标。
- [[TeaCache]] — 同类工作：也是基于特征变化检测的 DiT 残差缓存，但用固定阈值、不区分视频内容；理解它的局限是理解 AdaCache 切入点的最快路径。

> 如果上面有你不熟的，先点进概念笔记补一下，否则 §3 会读得很吃力。

---

## 1. 问题是什么 & 为什么难

### 1.1 任务设定

- **输入**：文本提示（text prompt），可选条件（如参考帧）
- **输出**：完整视频（多帧连续图像），如 Open-Sora 生成 720p × 2s 视频
- **评价指标**：VBench（综合质量，越高越好）、PSNR/SSIM/LPIPS（与 baseline 输出的像素级相似度）、端到端推理延迟和 FLOPs

### 1.2 朴素做法为什么不行

**问题核心：视频 DiT 推理极慢。**

以 Open-Sora 720p-2s 为例，100 步去噪 × 每步的全量 Transformer 计算 = 419 秒/视频（单 A100）。工业级部署完全不可接受。

**朴素加速尝试一：均匀固定缓存（如 PAB）**

Pyramid Attention Broadcast（PAB）按固定周期缓存 attention 输出。问题：

1. 对所有视频用同一个缓存周期——静态场景（微博客厅）和高速运动（体育追逐）的 feature 变化速率可能差 3–5 倍，统一周期要么对前者浪费算力，要么对后者伤画质。
2. 对所有扩散步用同一调度——去噪早期（大噪声）和后期（精细细节）的 feature 变化速率也截然不同，早期变化快、后期趋于稳定。

实验结果：Open-Sora-Plan 上 PAB-fast 的 VBench 从 80.39 暴跌到 71.81（-8.58），非常严重的质量损失。

**朴素加速尝试二：步数蒸馏（Consistency Distillation 等）**

需要重训练，换一个 baseline 就要重做一次，工程成本高；且与缓存加速正交，本文聚焦推理端 training-free 方案。

### 1.3 本文的切入点

**观察**：不同视频生成任务之间，DiT Temporal Self-Attention 残差的"帧间变化速率"差异极大（论文 Figure 2 的直方图直观展示了这一点）。

**切入点**：既然不同视频的 feature 变化速率不同，那就让**每个视频自己量测自己的变化速率**，再根据这个速率动态决定本步是复用缓存还是重新计算。这样既不浪费容易的视频的算力，也不牺牲复杂视频的画质。

---

## 2. 核心思想（用直觉讲透）

**一句话核心**：用"feature 变化速率"做实时测速仪，给每个视频的每个去噪步贴上合适的"缓存时长标签"。

**直觉/类比**：

想象你在看一部电影——有些场景（城市延时摄影）帧与帧之间几乎静止，你可以每隔 10 帧才认真看一帧，其余直接用上一帧代替，完全不影响理解；而有些场景（足球射门特写）每帧都有关键变化，你必须帧帧都看。

AdaCache 就是这个"智能观众"：

1. 每隔几步，测一下"上一次缓存的特征"和"现在算出来的特征"差了多少（这就是 $c_t^l$，变化速率）。
2. 查一张预先定好的"速率→缓存时长"对照表（codebook），得出"下 $\tau$ 步可以放心复用这个缓存"。
3. 在这 $\tau$ 步里，不再重新计算这一层的 Temporal Self-Attention，直接把上次结果贴进去，省掉大量算力。

**为什么这能解决 §1 的难点**：

- 相比 PAB 的"统一调度"：每个视频的 $c_t^l$ 分布不同，AdaCache 的 $\tau$ 自然也不同 → 静态视频缓存更久，运动视频缓存更短，都在自己的最优点上。
- 相比去噪步均匀调度：去噪早期 $c_t^l$ 大（feature 变化快）→ $\tau$ 小（频繁重算）；后期 $c_t^l$ 小 → $\tau$ 大（更多复用）。这符合扩散过程的物理规律。

---

## 3. 方法逐层拆解

### 3.1 总览：数据流

AdaCache 作为推理时插件，插在标准 Video DiT 的每个 Transformer block 内部：

**文本条件 $c$ + 噪声视频 $x_T$** → *去噪循环（T 步）*：
- 每步 $t$，对每层 $l$：
  - **层前**：查询缓存状态 $k_t^l$（已缓存几步了？）
  - 若 $k_t^l < \tau_t^l$：直接复用缓存残差 $\hat{p}_t^l$（跳过 STA 计算）
  - 若 $k_t^l = \tau_t^l$：重新计算 STA，得到 $p_t^l$；更新 $c_t^l$；查 codebook 得新 $\tau_t^l$；重置 $k=0$
- 其余层（Cross-Attention、MLP）正常计算

→ 清洁视频 $x_0$

### 3.2 模块 A：残差缓存（Residual Caching）

- **是什么**：在视频 DiT 的 Temporal Self-Attention（STA）层，把该层输出（残差 $p_t^l$）存入缓存，后续步直接复用。
- **为什么缓存 STA 而不是 CA 或 MLP**：
  - STA 负责帧间时序一致性建模，其输出在相邻去噪步间变化最小（ablation 验证）。
  - Cross-Attention（文本条件）和 MLP 的输出变化更显著，复用误差更大。
  - 实验：缓存 STA 残差（VBench 83.40）> 缓存 MLP 残差（明显下降）。
- **怎么实现**：维护一个 per-layer 缓存 $\hat{p}^l$ 和计数器 $k^l$；重算时更新缓存，复用时直接加到特征上。
- **去掉会怎样**：退化为 baseline（无加速）。

### 3.3 模块 B：内容自适应缓存调度（Content-Adaptive Schedule）

- **是什么**：实时估算当前层的"feature 变化速率" $c_t^l$，查 codebook 映射到对应缓存时长 $\tau_t^l$。
- **为什么这样设计**：不同视频、不同去噪步的 feature 变化速率差异大（Figure 2 的直方图跨越多个数量级），固定阈值必然为某些情况过度或不足。内容自适应调度让系统自我适配，无需人工分类视频。
- **怎么实现**：
  - 计算 $c_t^l$（见 §4 公式推导）
  - 查 codebook（一个手工设定的阈值→缓存步数映射表，见下）：

  **AdaCache-fast codebook（Open-Sora 100步）：**

  | 变化速率阈值 $c$ | 对应缓存时长 $\tau$ |
  |:---:|:---:|
  | < 0.03 | 12 步 |
  | < 0.05 | 10 步 |
  | < 0.07 | 8 步 |
  | < 0.09 | 6 步 |
  | < 0.11 | 4 步 |
  | ≥ 0.11 | 3 步 |

  变化越慢 → $\tau$ 越大 → 缓存更久 → 更省算力。

- **去掉会怎样**：退化为均匀固定缓存（如 PAB 的方案），在高运动或高步数场景显著掉质量。

### 3.4 模块 C：运动正则化（Motion Regularization, MoReg）

- **是什么**：在计算 $c_t^l$ 时乘一个运动感知权重，让运动剧烈的视频/区域少缓存。
- **为什么需要它**：极高缓存率下（如 4.7× 配置），纯粹基于特征距离的调度可能在运动剧烈时仍给出过长 $\tau$，导致时序不一致（鬼影、模糊）。MoReg 用运动量作为额外惩罚项，拉低这些情况的 $\tau$。
- **怎么实现**：见 §4 公式推导；用 STA 残差的帧间差分估计运动量 $m_t^l$ 和运动梯度 $mg_t^l$，乘到 $c_t^l$ 上。
- **去掉会怎样**：Open-Sora-Plan 上 VBench 从 79.30 降至 75.83（-3.47），说明 150 步的长调度下运动对齐非常关键。

> 为什么不用光流？光流需要额外网络、训练、推理开销，违背 training-free 原则。MoReg 直接从已有的 STA 残差提取运动信息，零额外参数。

---

## 4. 关键公式逐步推导

### 公式 1-3：[[Spatio-Temporal Attention|标准 DiT Block 前向过程]]

**直觉先行**：每个 Transformer block 依次做三件事：时序 self-attention（STA）、文本交叉 attention（CA）、前馈 MLP，每步以残差方式叠加。

$$
\begin{aligned}
p_t^l &= \mathrm{STA}(f_t^l) \\
\tilde{f}_t^l &= f_t^l + p_t^l \\
q_t^l &= \mathrm{CA}(\tilde{f}_t^l) \\
\bar{f}_t^l &= \tilde{f}_t^l + q_t^l \\
r_t^l &= \mathrm{MLP}(\bar{f}_t^l) \\
f_t^{l+1} &= \bar{f}_t^l + r_t^l
\end{aligned}
$$

**逐符号解释**：
- $t$：去噪时间步索引（从 $T$ 到 $0$）
- $l$：Transformer 层索引
- $f_t^l$：第 $l$ 层在第 $t$ 步的输入特征（视频 token 序列）
- $p_t^l$：STA 的残差输出，**AdaCache 缓存的对象**
- $q_t^l$：CA 的残差输出（文本-视觉交互）
- $r_t^l$：MLP 残差输出

**由来/推导**：这是标准 Post-LN/Pre-LN Transformer 的残差连接形式，AdaCache 不修改这个结构，只是在某些步把 $p_t^l$ 替换成缓存值。

**自检**：当 $p_t^l \equiv 0$ 时，STA 退化为直通（skip），相当于该层只保留 CA 和 MLP，这是缓存失效的极端情况。

---

### 公式 4：[[Cache Update|特征变化速率（距离度量）]]

**直觉先行**：如果我们每隔 $k$ 步重算一次，那么"这 $k$ 步里 feature 平均每步变化了多少"就是变化速率——数值越小，说明 feature 越稳定，可以缓存更久。

$$
c_t^l = \frac{\|p_t^l - p_{t+k}^l\|}{k}
$$

**逐符号解释**：
- $c_t^l$：第 $l$ 层在第 $t$ 步测量的变化速率（scalar）
- $p_t^l$：当前步重算得到的 STA 残差
- $p_{t+k}^l$：上一次重算（$k$ 步之前）缓存的 STA 残差
- $k$：自上次重算起已经过去的步数（即当前缓存时长）
- $\|\cdot\|$：L1 或 L2 范数（ablation 显示两者相差不大，L1 更快）

**由来/推导**：这是离散时间导数的定义：$\frac{\Delta f}{\Delta t}$。用历史两点的差分除以时间间隔，近似"变化的平均速率"。这里的"时间"是去噪步索引，而非物理时间。

**自检**：
- 极端情况 $k=1$：每步重算，$c_t^l$ 总是精确的即时变化速率。
- 极端情况 $k \to \infty$：$c_t^l$ 可能严重低估真实速率（因为中间若干步的变化被"稀释"了），这是为什么 codebook 要限制最大 $\tau$ 的原因。

---

### 公式 5-6：[[Adaptive Caching Schedule|自适应缓存调度决策]]

**直觉先行**：拿到变化速率 $c_t^l$，查表得到下次允许缓存多久 $\tau_t^l$；然后在接下来的每步判断："离上次重算过去了多久？还没到期就复用，到期了就重算并更新。"

$$
\tau_t^l = \mathrm{codebook}(c_t^l)
$$

$$
p_{t-k}^l = \begin{cases}
p_t^l & \text{if } k < \tau_t^l \quad \text{（复用缓存）} \\
\mathrm{STA}(f_{t-k}^l) & \text{if } k = \tau_t^l \quad \text{（重新计算）}
\end{cases}
$$

**逐符号解释**：
- $\mathrm{codebook}(\cdot)$：分段阈值函数（见 §3.3 的表格），将 $c_t^l$ 映射到整数 $\tau \in \{3,4,6,8,10,12\}$
- $k$：自当前缓存建立起已过去的步数（计数器）
- $t-k$：当前正在计算的去噪步

**由来**：这是 AdaCache 的核心决策逻辑。Codebook 的阈值是在 Open-Sora 验证集上通过网格搜索得到的，不同 baseline/分辨率可能需要重调。

**自检**：当 $c_t^l$ 极小（完全静止视频），$\tau = 12$，100 步里只需重算约 8 次，FLOPs 降至约 1/12。当 $c_t^l$ 很大，$\tau = 3$，大约每 3 步重算一次，FLOPs 降至约 1/3。

---

### 公式 7-8：[[Motion Score|运动量估计（MoReg 第一步）]]

**直觉先行**：用 STA 残差 $p_t^l$ 内部的帧间差分来衡量视频运动量——相邻帧之间 feature 变化越大，运动越剧烈。同时计算运动量的"梯度"（运动在加速还是减速），避免遮挡突变等情况被低估。

$$
m_t^l = \| p_{t,\, i:N}^l - p_{t,\, 0:N-i}^l \|
$$

$$
mg_t^l = \frac{m_t^l - m_{t+k}^l}{k}
$$

**逐符号解释**：
- $p_{t,\, i:N}^l$：第 $t$ 步残差中第 $i$ 到第 $N$ 帧的 token（帧切片）
- $p_{t,\, 0:N-i}^l$：第 $0$ 到第 $N-i$ 帧的 token
- $N$：视频总帧数；$i$：帧间距（取 1 帧或几帧）
- $m_t^l$：当前步的运动量（标量）
- $mg_t^l$：运动梯度，衡量"运动在加速还是减速"

**由来**：这是光流思想的零成本近似：光流衡量像素运动，这里用 STA 残差的帧差代替。STA 残差包含了时序交互信息，其帧差近似反映了视频运动程度，且完全不需要额外网络。

**自检**：静止视频 → $m_t^l \approx 0$，$mg_t^l \approx 0$；高速运动 → $m_t^l$ 大，若运动持续加速则 $mg_t^l > 0$，若减速则 $mg_t^l < 0$。

---

### 公式 9：[[Motion Regularization|运动正则化距离度量]]

**直觉先行**：把运动量"乘"进变化速率里——运动越剧烈，$c_t^l$ 被放大，查 codebook 得到更小的 $\tau$（更频繁重算），分配更多算力。

$$
c_t^l \leftarrow c_t^l \cdot (m_t^l + mg_t^l)
$$

**逐符号解释**：
- 左侧 $c_t^l$：被 MoReg 调整后的新变化速率，直接用于查 codebook
- $m_t^l + mg_t^l$：综合运动量 + 运动趋势的标量权重；运动越大、加速越快，权重越高

**由来**：简单的逐元素乘法——把物理运动量作为调制因子注入到 AdaCache 的调度信号里。这是一种典型的"正则化"：在原始信号上加约束，防止极端情况（高运动）下缓存过度。

**自检**：
- 若 $m_t^l = 0$（静止），$c_t^l$ 变为 0，查 codebook 得最大 $\tau$（最省算力）——与预期完全一致。
- 若 $mg_t^l < 0$（运动正在减速），权重减小，允许缓存稍久一些——合理，因为运动要停下来了。

---

## 5. 关键图表精读

### Figure 1：定性对比——Open-Sora 720p-2s

> 🖼️ **Figure 1: Qualitative comparison on Open-Sora 720p-2s** — 图片暂缺（原图见 [arXiv HTML](https://arxiv.org/html/2411.02397v1)）

**这张图在说什么**：展示 baseline（419s/视频）vs PAB-fast（~100s）vs AdaCache-fast（89s，**4.7×**）的生成帧。视觉质量上 AdaCache 的帧与 baseline 难以区分，而 PAB-fast 出现了明显的时序抖动和细节模糊。这是全文最直接的"眼见为实"证据。

---

### Figure 2：特征距离分布直方图

> 🖼️ **Figure 2: Feature distance distributions across different videos and diffusion steps** — 图片暂缺（原图见 [arXiv HTML](https://arxiv.org/html/2411.02397v1)）

**这张图在说什么**：核心观察图。横轴是特征距离 $c_t^l$，纵轴是频率；

- **左图（跨视频）**：不同视频的 $c_t^l$ 分布差异极大（直方图峰值位置相差 3–5 倍）。这直接证明了"不同视频需要不同调度"的假设。
- **右图（同视频跨去噪步）**：同一个视频，早期去噪步（大噪声）$c_t^l$ 集中在高值区，后期步集中在低值区。这解释了为什么固定周期缓存在早期会伤画质，在后期又浪费算力。

这张图是整篇论文动机的"指纹"，没有它，后面的方法就缺乏支撑。

---

### Figure 3（项目主页）：方法总览图

> 🖼️ **Figure 3: AdaCache overview** — [adacache-dit.github.io](https://adacache-dit.github.io/clarity/images/adacache.png)

**这张图在说什么**：展示标准 DiT Block 中 AdaCache 的插入位置：STA 层前加一个"缓存查询/更新"分支，CA 和 MLP 保持原样。同时展示 codebook 查询逻辑和 MoReg 调制路径。理解这张图等于理解了整个实现。

---

### Figure 4：方法-数据流细节

> 🖼️ **Figure 4: Detailed data flow of AdaCache within a DiT block** — 图片暂缺（原图见 [arXiv HTML](https://arxiv.org/html/2411.02397v1)）

**这张图在说什么**：展示在具体的一个去噪步 $t$ 内，缓存决策是如何做出的——距离 $c_t^l$ 的计算位置、$\tau$ 的更新时机、以及计数器 $k$ 的递增逻辑。

---

### Figure 5：用户研究结果

> 🖼️ **Figure 5: User study results** — 图片暂缺（原图见 [arXiv HTML](https://arxiv.org/html/2411.02397v1)）

**这张图在说什么**：A/B 盲测，36 名用户，1800 次评判。

- 对比 AdaCache vs PAB（同等延迟下）：用户 **70%** 倾向 AdaCache，说明在主观感知上的优势也很显著。
- 对比 AdaCache vs baseline（不加速）：**41%** 的评判认为两者无差异，说明 AdaCache 在很多时候已经"无损"。

这组数据比 VBench 数字更有说服力：分位数指标可能被平滑，主观评测对时序抖动等问题更敏感。

---

### Figure 6：质量-延迟权衡曲线

> 🖼️ **Figure 6: Quality-latency trade-off curves** — 图片暂缺（原图见 [arXiv HTML](https://arxiv.org/html/2411.02397v1)）

**这张图在说什么**：横轴延迟，纵轴 VBench；每条曲线是一个方法（AdaCache、PAB、T-GATE）。关键观察：

- AdaCache 的曲线**斜率更平缓**——随着延迟降低，质量下降更慢（抗压）。
- PAB 的曲线在低延迟端"断崖式"下跌（尤其 Open-Sora-Plan 上 71.81 那个点）。
- 这证明 AdaCache 的设计点不只在一个特定配置上，而是在整个速度-质量空间里都优于 SOTA。

---

### Table 1：全量对比实验

**Open-Sora（480p-2s，30 步）**

| 方法 | VBench (%) | PSNR | LPIPS | SSIM | FLOPs (T) | 延迟 (s) | 加速比 |
|------|-----------|------|-------|------|-----------|---------|-------|
| Baseline | 79.22 | — | — | — | 3230.24 | 54.02 | 1.00× |
| ΔΔ-DiT | 78.21 | 11.91 | 0.5692 | 0.4811 | 3166.47 | — | — |
| T-GATE | 77.61 | 15.50 | 0.3495 | 0.6760 | 2818.40 | 49.11 | 1.10× |
| PAB-fast | 76.95 | 23.58 | 0.1743 | 0.8220 | 2558.25 | 40.23 | 1.34× |
| PAB-slow | 78.51 | 27.04 | 0.0925 | 0.8847 | 2657.70 | 44.93 | 1.20× |
| **AdaCache-fast** | **79.39** | 24.92 | 0.0981 | 0.8375 | 1331.97 | 24.16 | **2.24×** |
| AdaCache-fast+MoReg | 79.48 | 25.78 | 0.0867 | 0.8530 | 1383.66 | 25.71 | 2.10× |
| AdaCache-slow | 79.66 | 29.97 | 0.0456 | 0.9085 | 2195.50 | 37.01 | 1.46× |

**Open-Sora-Plan（512×512-2.7s，150 步）**

| 方法 | VBench (%) | PSNR | LPIPS | SSIM | FLOPs (T) | 延迟 (s) | 加速比 |
|------|-----------|------|-------|------|-----------|---------|-------|
| Baseline | 80.39 | — | — | — | 12032.40 | 129.67 | 1.00× |
| ΔΔ-DiT | 77.55 | 13.85 | 0.5388 | 0.3736 | 12027.72 | — | — |
| T-GATE | 80.15 | 18.32 | 0.3066 | 0.6219 | 10663.32 | 113.75 | 1.14× |
| PAB-fast | 71.81 | 15.47 | 0.5499 | 0.4717 | 8551.26 | 89.56 | 1.45× |
| PAB-slow | 80.30 | 18.80 | 0.3059 | 0.6550 | 9276.57 | 98.50 | 1.32× |
| **AdaCache-fast** | 75.83 | 13.53 | 0.5465 | 0.4309 | 3283.60 | 35.04 | **3.70×** |
| **AdaCache-fast+MoReg** | **79.30** | 17.69 | 0.3745 | 0.6147 | 3473.68 | 36.77 | 3.53× |
| AdaCache-slow | 80.50 | 22.98 | 0.1737 | 0.7910 | 4983.30 | 58.88 | 2.20× |

**Latte（512×512-2s，50 步）**

| 方法 | VBench (%) | PSNR | LPIPS | SSIM | FLOPs (T) | 延迟 (s) | 加速比 |
|------|-----------|------|-------|------|-----------|---------|-------|
| Baseline | 77.40 | — | — | — | 3439.47 | 32.45 | 1.00× |
| ΔΔ-DiT | 52.00 | 8.65 | 0.8513 | 0.1078 | 3437.33 | — | — |
| T-GATE | 75.42 | 19.55 | 0.2612 | 0.6927 | 3059.02 | 29.23 | 1.11× |
| PAB-fast | 73.13 | 17.16 | 0.3903 | 0.6421 | 2576.77 | 24.33 | 1.33× |
| PAB-slow | 76.32 | 19.71 | 0.2699 | 0.7014 | 2767.22 | 26.20 | 1.24× |
| **AdaCache-fast** | 76.26 | 17.70 | 0.3522 | 0.6659 | 1010.33 | 11.85 | **2.74×** |
| AdaCache-fast+MoReg | 76.47 | 18.16 | 0.3222 | 0.6832 | 1187.31 | 13.20 | 2.46× |
| AdaCache-slow | 77.07 | 22.78 | 0.1737 | 0.8030 | 2023.65 | 20.35 | 1.59× |

**怎么读**：最该看的是 AdaCache-fast+MoReg 这一行——在与 PAB-fast 相当甚至更高的 VBench 下，FLOPs 降幅更大、延迟更低。尤其 Open-Sora-Plan：PAB-fast 损了 8.58 VBench 才换来 1.45×，AdaCache+MoReg 损了 1.09 VBench 就换来 3.53×。

---

### Table 2：消融 — Open-Sora 720p-2s

| 配置 | VBench (%) | 延迟 (s) | 加速比 |
|------|-----------|---------|-------|
| Baseline | 84.16 | 419.60 | 1.0× |
| AdaCache（无 MoReg） | 83.40 | 89.53 | **4.7×** |
| + MoReg（完整版） | **83.50** | 93.50 | 4.5× |
| + MoReg（无梯度项 $mg$） | 83.36 | 89.01 | 4.7× |
| + MoReg（多步平均版） | 83.42 | 95.65 | 4.4× |

**怎么读**：MoReg 以 4s 延迟代价（+4%）换来 0.1 VBench 提升，且梯度项 $mg_t^l$ 是有贡献的（去掉后 83.36 < 83.50）。多步平均版开销最大但提升最少，说明当前帧的运动梯度比历史平均更有效。

---

### Table 3：消融 — 各设计选择（Open-Sora 720p-2s）

**距离度量选择：**

| 度量 | VBench (%) | 延迟 (s) | 加速比 |
|------|-----------|---------|-------|
| L1 | 83.40 | 89.53 | 4.7× |
| L2 | 83.50 | 92.70 | 4.5× |
| Cosine | 83.19 | 86.74 | 4.8× |

**缓存层位置（Mid-layer vs 其他）：**
- Mid-layer 最优，多层叠加带来的边际收益极小且增加开销。

**残差类型：**
- Temporal Attention（STA）残差 > MLP 残差（后者出现明显质量下降）

---

## 6. 实验：它到底证明了什么

### 主结果

核心结论：**AdaCache 在三个独立 baseline（Open-Sora、Open-Sora-Plan、Latte）上均超越所有对比方法，在 FLOPs 和延迟维度上优势尤为突出。**

关键数字：
- Open-Sora 720p-2s：**4.7×** 加速，VBench 仅 -0.76（83.40 vs 84.16）
- Open-Sora-Plan：**3.70×**（AdaCache-fast），加上 MoReg 后 VBench 恢复至 79.30（baseline 80.39），而 PAB-fast 只有 71.81
- Latte：**2.74×**，VBench 76.26（baseline 77.40），PAB-fast 只有 73.13

**多 GPU 加成**（1→8 A100）：
- Open-Sora 480p-2s：2.24× → 6.94×
- Open-Sora-Plan：3.70× → 9.22×

这说明 AdaCache 与并行化正交，且在多卡场景下通信开销相对计算开销的占比更小，缓存节省的计算更稀有，加速比进一步放大。

### 消融

**最关键组件**：内容自适应调度（公式 5-6 的 codebook 设计）是核心——若退化为固定 $\tau$，在 Open-Sora-Plan 这类长步数场景下质量断崖式下跌（这是 PAB 的问题，也是本文最主要打败的对手）。

**MoReg 的贡献是有条件的**：步数越多（Open-Sora-Plan 150步 > Open-Sora 100步 > Latte 50步），MoReg 的作用越显著。这符合直觉：更长的缓存时长窗口里，运动不一致积累的误差更多。

**梯度项 $mg_t^l$ 有效**：对比"有梯度/无梯度"消融，加入运动趋势信息能稳定改善 VBench（83.50 vs 83.36）。

### 局限实验 / 反例

- **Codebook 需要分 baseline/分辨率手工调**：论文提到 fast/slow 两套 codebook 是针对 Open-Sora 调的，Open-Sora-Plan（150步）和 Latte（50步）需要重调，但作者没有给出通用调参公式，只说"按照相似流程"。
- **Open-Sora-Plan AdaCache-fast（无 MoReg）质量较低**（75.83 vs 80.39），说明在步数极多时仅靠距离调度不够，MoReg 是必需的而非可选的。
- **ΔΔ-DiT 几乎没有加速效果**（FLOPs 几乎不变），但质量损失明显，是本文对比中最弱的 baseline，表明简单差分方法在视频 DiT 上无效。

---

## 7. 复现思路（动手向）

### 训练流程 / 伪代码

```python
# AdaCache 推理伪代码
# 初始化：每层 l 的缓存残差 cache[l] = None，计数器 counter[l] = 0

for t in denoising_steps:        # t 从 T 到 0
    for l in transformer_layers:
        f = input_features[l]
        
        if counter[l] == tau[l] or cache[l] is None:
            # 重新计算 STA
            p = STA(f)
            
            # 更新变化速率
            if cache[l] is not None:
                c = norm(p - cache[l]) / counter[l]  # 公式 4
                
                # MoReg（可选）
                if use_moreg:
                    m = motion_score(p)               # 公式 7
                    mg = (m - prev_m[l]) / counter[l]  # 公式 8
                    c = c * (m + mg)                   # 公式 9
                
                tau[l] = codebook_lookup(c)            # 公式 5
            
            cache[l] = p          # 更新缓存
            counter[l] = 0        # 重置计数器
        else:
            p = cache[l]          # 复用缓存（公式 6）
        
        counter[l] += 1
        
        # 后续 CA、MLP 正常计算
        f = f + p
        f = f + CA(f)
        f = f + MLP(f)
        features[l+1] = f
```

### 关键超参

- **Codebook 阈值设定**（AdaCache-fast，Open-Sora 100步）：`{0.03:12, 0.05:10, 0.07:8, 0.09:6, 0.11:4, 1.0:3}`
- **Codebook 对应 slow 模式**：阈值相同，但最大 $\tau$ 更小（如 8/6/4/2/1），质量更好、速度更慢
- **MoReg 帧偏移 $i$**：取 1（逐帧差分）
- **距离范数**：L1 或 L2（相差不大，L1 更快）

### 数据与算力

- **评测设备**：单 A100 80G（延迟测量），多卡 1/2/4/8 A100（多卡消融）
- **质量评测**：VBench 需要生成 ~900+ 视频，建议用多卡并行
- **无需训练**：纯推理端修改，接入一个新 baseline 只需重调 codebook 阈值（约 1–2 小时搜索）

### 容易踩的坑

1. **Codebook 不可直接跨 baseline 迁移**：Open-Sora 100步的 codebook 不适用于 Latte 50步，阈值需要按比例重调。
2. **计数器重置时机**：重算发生在 `counter == tau`，之后 counter 要重置为 0 而非 1，否则会跳过下一步应有的计算。
3. **MoReg 中 prev_m 的初始化**：第一次重算时 prev_m 未定义，要做好边界检查（通常第一步不使用 MoReg 加权）。
4. **多 GPU 下缓存同步**：各 GPU 持有不同 token 分片时，motion_score 需要 all-reduce 才能得到全局运动量。论文提到这是额外通信开销，但实验上可接受。

---

## 8. 批判与延伸

### 真创新 vs 包装

**真正新的点有两个**：

1. **每视频的内容自适应调度**：这在计算机视觉里并不罕见（自适应计算早有先例），但在视频生成 DiT 推理加速里是首个系统论证"不同视频需要不同调度"的工作，Figure 2 的分布分析有说服力。
2. **MoReg 的设计**：用 STA 残差做零参数运动估计，把运动量反注入缓存调度，这个"闭环"设计有工程美感。

**偏包装的部分**：基础的残差缓存思路（TeaCache、PAB 等）早已有之，AdaCache 的主要改进是把"固定调度"升级为"自适应调度"，没有引入新的架构组件。从技术深度看，这是一篇扎实的工程论文，而非提出了全新范式的理论工作。

### 站不住的 claim / 评估盲区

1. **Codebook 的可迁移性未充分验证**：论文在三个 baseline 上都给出了不同 fast/slow 配置，但没有说明调参成本，让人担忧对私有模型的应用难度。
2. **PSNR/SSIM 指标的解读需谨慎**：AdaCache-fast 的 PSNR（Open-Sora 480p：24.92）低于 AdaCache-slow（29.97），但 VBench 反而比 slow 低（79.39 vs 79.66）——这说明 VBench 和像素级相似度之间存在弱相关，可能存在"VBench 高但与 baseline 输出不像"的情况。
3. **测速环境单一**：所有延迟测量在 A100 上，V100、H100 或消费级 GPU（RTX 4090）的速度比可能不同，因为缓存访问延迟和计算带宽比不同。
4. **长视频未测试**：所有实验是 2–2.7 秒短视频；在 10 秒以上的长视频里，运动模式的时序多变性可能让 codebook 失配。

### 可以接着做什么

1. **学习化 codebook**：目前 codebook 是手工调的分段函数。可以训一个轻量 MLP，直接从 $c_t^l$ 预测最优 $\tau$，在不增加太多开销的情况下泛化到任意 baseline 和分辨率。
2. **跨层协同调度**：当前每层独立决策；若相邻层的 $c_t^l$ 呈现强相关（论文隐含提到"mid-layer 够用"），可以学一个层级聚合的调度，进一步减少调度开销。
3. **结合步数蒸馏**：AdaCache 减少每步计算，蒸馏减少总步数，两者正交。可以探索两者联合的 Pareto 前沿，比如 20 步蒸馏模型 + AdaCache 4× = 理论上近 80× 加速。
4. **对高动态视频的细粒度调度**：MoReg 是全视频层面的，对局部高运动（前景快速运动但背景静止）可能不够精细。可以做 token 级或空间区域级的缓存决策，进一步提升高动态视频的质量上限。

---

## 9. 常见疑问 Q&A

> **Q：为什么只缓存 STA 残差，而不缓存 Cross-Attention 或 MLP？**
>
> A：消融（Table 3）直接验证了这一点——缓存 MLP 残差 VBench 明显下降，而缓存 STA 表现最好。直觉上：STA 建模帧间时序一致性，其输出在相邻去噪步间变化本就最慢（"时间的惯性"）；CA 依赖文本条件，变化由文本-视觉对齐决定，更难预测；MLP 是非线性变换，残差更难复用。

> **Q：Codebook 是怎么"训"出来的？需要大量实验吗？**
>
> A：Codebook 不是"训"出来的，是在验证集上网格搜索的（一组固定的视频生成 prompts）。阈值选 6 档（0.03/0.05/0.07/0.09/0.11/∞），每档试几个整数 $\tau$ 值，挑 VBench 和延迟权衡最优的组合。成本相对有限（几十次推理），但确实需要针对每个 baseline/步数设置重做一遍。

> **Q：MoReg 和直接用光流的区别是什么？为什么不用光流会更好？**
>
> A：光流需要额外网络（如 RAFT），增加推理延迟和部署依赖，违背 training-free 原则。MoReg 用 STA 残差本身做帧差——这是一种"零成本"的运动感知，因为 $p_t^l$ 本来就要计算。代价是：STA 残差是高维语义特征层面的帧差，不等于像素运动，对遮挡、光照变化等情况的响应可能与真实运动不完全一致。在实验中这个近似已经足够好。

> **Q：AdaCache 的加速比为什么在 Open-Sora-Plan（3.70×）比 Open-Sora（2.24×）更高？**
>
> A：Open-Sora-Plan 用 150 步去噪，步数更多 → 更多机会缓存 → 绝对节省的步数更多。同时，更长的去噪轨迹中"平稳期"（$c_t^l$ 小的步）占比更高，AdaCache 可以分配更长的 $\tau$。直觉：跑马拉松比 100 米更容易节省体力——可以在平路上惯性前进更久。

> **Q：多 GPU 下加速比为什么反而更大（从 3.70× 到 9.22×）？**
>
> A：多 GPU 时，每卡持有部分 token，通信开销（all-reduce）是固定的。当有缓存时，计算量减少但通信量不减少，所以通信开销在总时间里占比升高；当没缓存时，计算主导，通信占比低。AdaCache 跳过的是**计算**，不跳通信，所以多 GPU 下每跳过一次计算节省的比例（相对总时间）更大，累积起来加速比更高。

> **Q：AdaCache 能用于图像生成 DiT（如 FLUX）吗？**
>
> A：技术上可以，但图像 DiT 只有空间维度，没有时序 STA。缓存对象会变成空间 self-attention 残差，变化规律与视频不同——图像去噪步之间的 feature 变化模式可能没有视频那么平滑。作者没有测试图像 DiT，但这是一个自然的延伸方向（相关工作 TeaCache 已经在做这件事）。

---

## 关联

- **基于**：[[DDPM]] / [[Diffusion Model]] — 扩散模型去噪框架，AdaCache 是其推理加速插件
- **基于**：[[DiT]] — Video DiT 的 Transformer block 结构，AdaCache 在其 STA 层插入缓存
- **对比**：[[TeaCache]] — 同类残差缓存方法，AdaCache 的主要改进是从固定调度升级为内容自适应调度
- **对比**：PAB（Pyramid Attention Broadcast）— 固定周期 attention 缓存，在高步数 baseline 上质量损失严重
- **应用于**：[[Sora]] / Open-Sora / Open-Sora-Plan / Latte — 验证的 Video DiT baseline
- **评测**：[[VBench]] — 主要质量指标
- **概念**：[[Spatio-Temporal Attention]]、[[Motion Regularization]]、[[Adaptive Caching Schedule]]、[[Cache Update]]

---

## 速查卡

> [!summary] AdaCache
> - **核心**：内容自适应残差缓存——用 feature 变化速率 + 运动量动态决定每个视频每层的缓存时长
> - **方法**：测量 STA 残差距离 $c_t^l$，查 codebook 得 $\tau_t^l$，不满期就复用，到期才重算；MoReg 用帧差运动量调制 $c_t^l$
> - **结果**：Open-Sora 720p 4.7×加速，VBench 仅 -0.76；多卡叠加可达 9.22×
> - **一句话评价**：扎实的工程推理加速论文，创新深度适中但实用价值极高，是视频生成落地必读的基础加速工作

---

*精读创建时间：2026-06-22*
