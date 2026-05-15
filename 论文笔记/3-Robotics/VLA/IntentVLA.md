---
title: "IntentVLA: Short-Horizon Intent Modeling for Aliased Robot Manipulation"
method_name: "IntentVLA"
authors: [Shijie Lian, Bin Yu, Xiaopeng Lin, Zhaolong Shen, Laurence Tianruo Yang, Yurun Jin, Haishan Liu, Changti Wu, Hang Yuan, Cong Huang, Kai Chen]
year: 2026
venue: arXiv
tags: [vla, robot-manipulation, short-horizon-intent, observation-aliasing, flow-matching, history-conditioning]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.14712v1
created: 2026-05-15
---

# 论文笔记：IntentVLA: Short-Horizon Intent Modeling for Aliased Robot Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | ZGC EmbodyAI（中关村具身智能） |
| 日期 | May 2026 |
| 项目主页 | https://github.com/ZGC-EmbodyAI/IntentVLA |
| 对比基线 | [[π₀]] / [[π₀.₅]] / [[GR00T N1.7]] / [[MemoryVLA]] / [[VLA-JEPA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.14712) / [HTML](https://arxiv.org/html/2605.14712v1) / [Code](https://github.com/ZGC-EmbodyAI/IntentVLA) |

---

## 一句话总结

> 用冻结的 [[VGGT]] 编码最近 $K=16$ 帧视觉历史抽取 [[短期意图表征|short-horizon intent]]，通过 [[门控交叉注意力|Gated Cross-Attention]] 注入 [[Qwen3-VL]] 主干，在 [[AliasBench]] 上把 [[观测混叠|observation aliasing]] 场景的成功率从 9.0% 推到 45.8%。

---

## 核心贡献

1. **问题刻画 — [[观测混叠]]**: 首次系统化定义机器人模仿学习中的 "观测混叠"：相似帧 $o_t^{(1)} \approx o_t^{(2)}$ 却需要执行不同的动作块 $a_t^{(1)} \neq a_t^{(2)}$，揭示当前 [[VLA]] 只看当前帧会在 [[部分可观测过程|partial observability]] 下退化为均值动作。
2. **[[AliasBench]] 基准**: 在 [[RoboTwin]] 2 上构造 12 任务、4 类（[[Back-and-Forth 任务族]]、[[Crossing-Path 任务族]]、[[Bimanual 任务族]]、[[Multi-Goal 任务族]]）专门隔离混叠的 benchmark，并给出基于视觉嵌入检索的混叠诊断。
3. **IntentVLA 框架**: 用冻结的 [[VGGT]] 抽取 camera token + 4 register token 作为 [[短期意图表征]]，经 [[门控交叉注意力]] 融入 [[Qwen3-VL]] 当前上下文，最后用基于 [[DiT]] 的 [[Flow Matching|条件流匹配]] 动作头解码 [[Action Chunking|动作块]]。
4. **[[Inter-Chunk Consistency|ICC-L2]] 指标 + 强结果**: 提出 inter-chunk consistency 度量重规划时的动作抖动，IntentVLA 把均值 ICC-L2 降低 17.6%，90 分位降低 21.7%；在 [[SimplerEnv]]、[[LIBERO]]、[[RoboCasa]] 上全面超越前 SOTA。

---

## 问题背景

### 要解决的问题

机器人模仿数据天然 **多模态（multimodal）**：人类示教者带着不同的短期意图执行任务，因此**相似的视觉-语言观测**会被不同的动作块跟随。论文用一个煎面包的例子说明：手里拿着面包的同一画面下，下一步可能是「放进煎锅」也可能是「送回盘子」，取决于当前阶段。

### 现有方法的局限

- 标准 [[VLA]]（[[π₀]]、[[π₀.₅]]、[[OpenVLA]]、[[GR00T N1.7]]）只用 $(o_t, \ell)$ 预测动作块 $\tau_t$，相当于在 latent intent $z_t$ 上做边缘化（见公式 1）。
- 朴素地把最近若干帧拼到上下文里（"history-as-context"）虽然能微弱改善，但帧间冗余高、计算量大、且容易过拟合到无关纹理变化。
- 之前的"意图建模"方法（DIAL、ACoT-VLA、MINT、MAIN-VLA、VFP）仍以**当前帧**为主，无法解决"同一帧不同意图"的根本问题。

### 本文的动机

作者把单帧策略写成对 latent intent 的边缘化（公式 1），观察到只要补一段**短期视觉历史** $h_t$（最近 $K$ 帧），后验 $p(z_t \mid o_t, \ell, h_t)$ 就会塌缩到当前 episode 真正承诺的意图（公式 2）。因此关键是用一个**强 3D-aware 历史编码器**（[[VGGT]]）提取紧凑的意图证据，再以**轻量的门控融合**注入主干，避免破坏预训练的 [[VLM]] 表征。

---

## 方法详解

### 模型架构

<!-- 论文中 IntentVLA 的整体管线 -->

IntentVLA 采用 **历史条件化 + 流匹配动作头** 架构：

- **输入**:
  - 当前帧 $o_t \in \mathbb{R}^{H \times W \times 3}$（仅头部摄像头）
  - 语言指令 $\ell$
  - 历史窗口 $h_t^K = o_{t-K:t-1}$（默认 $K = 16$）
- **当前上下文编码器**: [[Qwen3-VL]] 4B，把 $(o_t, \ell)$ 编码成 token 矩阵 $F_t \in \mathbb{R}^{N \times d}$
- **历史编码器**: 冻结的 [[VGGT|VGGT-1B]]，每帧只保留 **1 个 camera token + 4 个 register token**，避免引入 patch token 噪声；用可学投影映射到 $\mathbb{R}^{d_h}$
- **融合模块**: [[门控交叉注意力]]，以 $F_t$ 为 query、历史 token $\tilde{U}_t$ 为 key/value，并附加 pool 出的 [[历史证据 Token|history-evidence token]] $e_t^{\text{tok}}$
- **动作头**: 基于 [[DiT]] 的 [[Flow Matching|条件流匹配]] 头，沿用 [[GR00T N1.7]] 的解码结构
- **输出**: [[Action Chunking|动作块]] $\tau_t \in \mathbb{R}^{H \times d_a}$
- **总参数**: ≈ 5B（Qwen3-VL 4B + VGGT 1B frozen + 小量融合 / 头部参数）

### 核心模块

#### 模块 1: 基于 [[VGGT]] 的历史意图编码器

**设计动机**: [[VGGT]] 是为多视图 3D 重建预训练的 transformer，其 camera/register token 天然编码了**视角变化与几何关系**，正好对应"我刚才从哪儿过来"这种短期意图证据；冻结它可以避免在小机器人数据上过拟合。

**具体实现**:
- 对窗口 $h_t^K$ 内每帧抽取 1 个 camera token + 4 个 register token（共 $5K$ 个 token），完全丢弃高维 patch token
- 经可学线性投影对齐到 Qwen3-VL 隐维度 $d$，得到 $\tilde{U}_t \in \mathbb{R}^{5K \times d}$
- 同时 pool 出一个全局历史摘要 $\bar{e}_t = \text{Pool}(U_t) \in \mathbb{R}^{d_h}$，作为额外的 evidence token

#### 模块 2: [[门控交叉注意力]] 融合

**设计动机**: 直接把历史 token 拼到上下文会污染 [[Qwen3-VL]] 的预训练分布；门控注意力允许网络**自适应**地决定要不要听历史。

**具体实现**:
- $F_t' = F_t + \sigma(\alpha) \cdot \text{MHA}(\text{Q}=\text{LN}(F_t), \text{K}=\tilde{U}_t, \text{V}=\tilde{U}_t)$
- $\alpha$ 是**可学标量门**，初始化在 sigmoid(0)=0.5 附近，但实际训练后会自适应缩放
- 最终条件上下文 $C_t = [F_t'; e_t^{\text{tok}}]$ 同时携带"被历史调制过的当前 token"和"显式历史摘要"

#### 模块 3: [[Flow Matching|条件流匹配]] 动作头

**设计动机**: 复用 [[GR00T N1.7]] 的 [[DiT]] 动作头，已经过大规模机器人数据预训练，IntentVLA 只换条件信号 $C_t$。

**具体实现**:
- 训练时采样 flow time $s \sim p(s)$，从噪声 $\epsilon \sim \mathcal{N}(0, I)$ 插值得 $X_s = (1-s)\epsilon + s \cdot \tau_t$
- 用 DiT 预测速度场 $\hat{V}_\theta(X_s, s \mid C_t)$ 回归目标速度 $\tau_t - \epsilon$
- 推理时用 Euler ODE 积分还原动作块

---

## 关键公式

### 公式 1: [[部分可观测过程|帧条件策略的边缘化]]

$$
p_\theta(\tau_t \mid o_t, \ell) = \int p_\theta(\tau_t \mid o_t, \ell, z_t)\, p(z_t \mid o_t, \ell)\, dz_t
$$

**含义**: 标准 [[VLA]] 在 latent intent $z_t$ 上做边缘化；当 $p(z_t \mid o_t, \ell)$ 多模态时，模型输出趋向多峰均值，导致 [[Inter-Chunk Consistency|ICC]] 抖动。

**符号说明**:
- $\tau_t = (a_t, \ldots, a_{t+H-1}) \in \mathbb{R}^{H \times d_a}$: 长度 $H$ 的动作块
- $o_t$: 当前视觉观测
- $\ell$: 语言指令
- $z_t$: 隐式短期意图变量（episode 级承诺）

### 公式 2: [[历史条件化策略]]

$$
p_\theta(\tau_t \mid o_t, \ell, h_t) = \int p_\theta(\tau_t \mid o_t, \ell, h_t, z_t)\, p(z_t \mid o_t, \ell, h_t)\, dz_t
$$

**含义**: 加入历史窗口 $h_t$ 后，后验 $p(z_t \mid o_t, \ell, h_t)$ 通常退化为狄拉克（dirac）峰，从而消除多模态边缘化带来的均值动作问题。

**符号说明**:
- $h_t = h_t^K = o_{t-K:t-1}$: 最近 $K$ 帧视觉历史
- 其余符号同公式 1

### 公式 3: 学习到的[[短期意图表征]]

$$
m_t = f_\phi(o_t, \ell, h_t^K)
$$

**含义**: 实际实现中并不显式建模 $z_t$，而是用编码器 $f_\phi$ 把 $(o_t, \ell, h_t^K)$ 映射为意图表征 $m_t$，再作为动作头的条件。

**符号说明**:
- $f_\phi$: 由 [[Qwen3-VL]] + [[VGGT]] + 融合模块构成的复合编码器
- $\phi$: 可学参数（VGGT 部分冻结）

### 公式 4: [[Qwen3-VL]] 当前上下文编码

$$
F_t = q_\psi(o_t, \ell) \in \mathbb{R}^{N \times d}
$$

**符号说明**:
- $q_\psi$: [[Qwen3-VL]] backbone
- $N$: 视觉 + 文本 token 总数
- $d$: 隐藏维度

### 公式 5: [[VGGT]] 历史编码

$$
U_t = g_\phi(h_t^K) \in \mathbb{R}^{M \times d_h}, \quad \bar{e}_t = \text{Pool}(U_t)
$$

**符号说明**:
- $g_\phi$: 冻结的 VGGT + 可学投影
- $M = 5K$: 每帧 5 个 token，共 $5K$ 个（默认 $M = 80$）
- $d_h$: 历史 token 隐藏维度
- $\bar{e}_t$: 全局池化得到的历史摘要

### 公式 6: [[门控交叉注意力]] 融合

$$
F_t' = F_t + \sigma(\alpha) \cdot \text{MHA}\!\left(Q = \text{LN}(F_t),\; K = \tilde{U}_t,\; V = \tilde{U}_t \right)
$$

**含义**: 让当前 token $F_t$ 通过多头交叉注意力主动"查询"历史 token；可学标量 $\alpha$ 经 sigmoid 后控制融合强度。

**符号说明**:
- $\sigma(\alpha)$: sigmoid 门，$\alpha \in \mathbb{R}$ 可学
- LN: LayerNorm
- $\tilde{U}_t$: 投影后的历史 token

### 公式 7: 最终条件上下文

$$
C_t = [F_t';\; e_t^{\text{tok}}]
$$

**含义**: 融合后的当前 token 序列与 history-evidence token 拼接，作为动作头的条件信号。

### 公式 8: [[Flow Matching|条件流匹配]] 损失

$$
\mathcal{L}_{\text{flow}} = \mathbb{E}\!\left[\, \big\|\, \hat{V}_\theta(X_s, s \mid C_t) - (\tau_t - \epsilon)\, \big\|_2^2 \,\right]
$$

其中 $X_s = (1-s)\epsilon + s \cdot \tau_t$，$\epsilon \sim \mathcal{N}(0, I)$，$s \sim p(s)$。

**含义**: 标准 [[Flow Matching|流匹配]] 训练目标，预测从噪声到动作块的条件速度场；唯一的"意图监督"完全通过 $C_t$ 间接学习，不需要显式 intent label。

**符号说明**:
- $\hat{V}_\theta$: [[DiT]] 动作头预测的速度场
- $X_s$: flow time $s$ 处的插值样本
- $\epsilon$: 高斯噪声
- $s$: flow time，$p(s)$ 通常取 logit-normal

### 公式 9: [[Inter-Chunk Consistency|ICC-L2]] 度量

$$
\text{ICC}_t = \frac{1}{H - r} \sum_{j = r}^{H-1} \big\| \hat{a}_{t+j}^{(t)} - \hat{a}_{t+j}^{(t+r)} \big\|_2^2
$$

**含义**: 衡量重规划带来的动作抖动：把 $t$ 时刻预测的 chunk 与 $t + r$ 时刻重规划后的 chunk 在公共时间窗口上做 L2 距离平均。值越低代表策略越自洽，越不易因换 chunk 引发突变。

**符号说明**:
- $\hat{a}_{t+j}^{(t)}$: 在 $t$ 时刻预测的、对应未来 $t+j$ 步的动作
- $r$: 重规划间隔（论文实验中扫描多个 $r$）
- $H$: chunk 长度

### 公式 10: 模式切换概率（理论诊断）

$$
P_{\text{switch}}(t, r) = 1 - \sum_z p_t(z)\, p_{t+r}(z)
$$

**含义**: 把意图后验在 $t$ 和 $t + r$ 时的相似度转成"切换概率"。在帧条件策略下 $P_{\text{switch}}$ 会随 $r$ 增长而陡升，IntentVLA 由于 $p(z \mid o, \ell, h)$ 接近 dirac，$P_{\text{switch}}$ 显著被压制。

**符号说明**:
- $p_t(z)$: $t$ 时刻的意图后验
- $z$: 离散化的意图索引（理论诊断用）

---

## 关键图表

### Figure 1: Bread-Cooking 的[[观测混叠]]示意

![Figure 1](https://arxiv.org/html/2605.14712v1/x1.png)

**说明**: 用"煎面包"任务展示同一视觉状态下两种可能的下一步——"放入煎锅"或"放回盘子"。仅看当前帧无法决定该走哪一支，必须依赖最近几帧才能锁定 episode 级承诺。这是后续所有方法论的动机起点。

### Figure 2: 四类混叠任务族示例

![Figure 2](https://arxiv.org/html/2605.14712v1/x2.png)

**说明**: 展示 [[AliasBench]] 中四类典型任务的混叠模式：[[Back-and-Forth 任务族|Back-and-Forth]]（Move Phone, 同一物体反复来回）、[[Crossing-Path 任务族|Crossing-Path]]（Cook Bread, 路径源点决定终点）、[[Bimanual 任务族|Bimanual]]（Hand Over Roller, 对称交接）、[[Multi-Goal 任务族|Multi-Goal]]（瞬时线索选目标）。

### Figure 3: 混叠诊断 — 邻居异意图率 + cosine 距离分布

![Figure 3](https://arxiv.org/html/2605.14712v1/x3.png)

**说明**: 用视觉嵌入做 top-5 retrieval，左图显示约 50% 的最近邻来自**不同意图**的轨迹；右图给出"同意图" vs "异意图"对的 cosine 距离 gap 中位数 < $3 \times 10^{-3}$，证明仅靠当前帧的视觉嵌入根本无法把不同意图分开。

### Figure 4: IntentVLA 总体架构

![Figure 4](https://arxiv.org/html/2605.14712v1/x4.png)

**说明**: 左路 [[Qwen3-VL]] 处理 $(o_t, \ell)$ 得到当前 token $F_t$，右路冻结 [[VGGT]] 对 $h_t^K$ 提取 camera + register token，经线性投影成 $\tilde{U}_t$；中间用[[门控交叉注意力]] 把历史信息注入 $F_t$，再附加 history-evidence token $e_t^{\text{tok}}$ 形成条件 $C_t$，最后送进 [[DiT]] [[Flow Matching|流匹配]] 动作头输出 [[Action Chunking|动作块]]。

### Figure 5: [[Inter-Chunk Consistency|ICC-L2]] 改善曲线

![Figure 5](https://arxiv.org/html/2605.14712v1/x5.png)

**说明**: 横轴为重规划间隔 $r$、纵轴为 ICC-L2。IntentVLA 在所有混叠窗口下都把 ICC 压到 baseline 之下，均值降低 17.6%、90 分位降低 21.7%，直观说明历史条件化显著抑制 chunk 间动作跳变。

### Table 1: [[AliasBench]] 主结果

| 方法 | Back-and-Forth | Crossing-Path | Bimanual | Multi-Goal | Avg. |
|------|----------------|---------------|----------|------------|------|
| Qwen3-VL-GR00T（frame-only） | 6.0 | 15.7 | 5.5 | 8.7 | 9.0 |
| + last 4 frames | 7.3 | 19.3 | 2.5 | 11.0 | 10.4 |
| + 4 sampled from 16 | 31.8 | 47.3 | 6.0 | 18.7 | 28.1 |
| **IntentVLA** | **49.3** | **74.7** | **17.0** | **31.3** | **45.8** |

**说明**: 朴素拼最近 4 帧几乎无效（10.4%），稀疏采样 4/16 帧已经能拿到 28.1%，证明历史确有用；但 IntentVLA 通过 VGGT + 门控融合把均值再提到 45.8%，相比 frame-only 提升 **+36.8 个百分点**。

### Table 2: [[SimplerEnv]] (WidowX) 结果

| 方法 | Spoon | Carrot | Blocks | Eggplant | Avg. |
|------|-------|--------|--------|----------|------|
| Octo-Small | 41.7 | 8.2 | 0.0 | 56.7 | 26.7 |
| [[π₀]] | 29.2 | 62.5 | 29.2 | 91.6 | 53.1 |
| [[π₀.₅]] | 49.3 | 64.7 | 44.7 | 69.7 | 57.1 |
| Qwen3-VL-GR00T | 83.0 | 59.4 | 18.8 | 100.0 | 65.3 |
| **IntentVLA** | **70.8** | **66.7** | **54.2** | **100.0** | **72.9** |

**说明**: 在通用混叠程度较低的 SimplerEnv 上 IntentVLA 仍把均值从 65.3% 推到 72.9%，超越之前 SOTA 4.7 个百分点；Blocks 任务受益最大（+35.4），Spoon 由于本身混叠低反而略降（-12.2）。

### Table 3: [[LIBERO]] (Avg@500, 4 suites)

| 方法 | Spatial | Object | Goal | Long | Avg |
|------|---------|--------|------|------|-----|
| [[π₀.₅]] | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| [[VLA-JEPA]] | 94.8 | 99.6 | 95.8 | 94.0 | 96.1 |
| Qwen3-VL-GR00T | 97.8 | 98.8 | 97.4 | 92.0 | 96.5 |
| **IntentVLA** | **99.3** | **99.7** | **98.1** | **97.4** | **98.6** |

**说明**: LIBERO-Long suite（最长时序、最依赖历史）增益最大：92.0 → 97.4（+5.4），与作者"短期历史压缩长程意图"的解释一致。

### Table 4: [[RoboCasa]]-GR1 24 任务汇总

| 方法 | 平均成功率 |
|------|------------|
| MemoryVLA | 49.8 |
| TwinBrainVLA | 54.6 |
| **IntentVLA** | **57.0** |

代表性 task：PnP Bottle To Cabinet 76.0（best），PnP Can To Drawer 88.0（best）。

**说明**: 在 NVIDIA PhysicalAI RoboCasa-GR00T-X-Embodiment-Sim 基准上 24 个任务平均胜出 2.4 个百分点。

### Table 5: [[SimplerEnv]] 上的消融实验

| 变体 | Spoon | Carrot | Blocks | Eggplant | Avg |
|------|-------|--------|--------|----------|-----|
| Frame-only Qwen3-VL-GR00T | 83.0 | 59.4 | 18.8 | 100.0 | 65.3 |
| VGGT 仅当前帧 | 72.5 | 61.5 | 30.2 | 94.8 | 64.8 |
| History fusion，无 intent token | 67.7 | 65.6 | 49.0 | 95.8 | 69.5 |
| **IntentVLA（完整）** | **70.8** | **66.7** | **54.2** | **100.0** | **72.9** |

**关键发现**: (1) 单独把 backbone 换成 VGGT 没用 (64.8)；(2) 只要把历史 token 用门控注意力融进来就能 +4.2 (69.5)；(3) 再补上 history-evidence token 再 +3.4 (72.9)。说明**历史的价值来自融合方式，而非编码器本身**。

### Tables 6-10（Appendix）

附录里列出 [[AliasBench]] 12 个任务的详细定义：

- Table 7：Back-and-Forth 4 任务
- Table 8：Crossing-Path 3 任务
- Table 9：Bimanual 2 任务
- Table 10：Multi-Goal 3 任务

每个任务给出场景、阶段切换条件、混叠源、成功判据。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[AliasBench]] | 12 任务 × 100 demo | 在 [[RoboTwin]] 2 上自构，专门隔离混叠 | 训练 + 测试 |
| [[SimplerEnv]] (WidowX) | BridgeDataV2 子集 | 4 任务标准 manipulation | 测试 |
| [[LIBERO]] | 4 suites | Spatial/Object/Goal/Long | 测试 |
| [[RoboCasa]]-GR1 | 24 任务 | PhysicalAI Robotics-GR00T-X-Embodiment-Sim | 测试 |

### 实现细节

- **Backbone**: [[Qwen3-VL]] 4B（当前帧/语言）+ [[VGGT|VGGT-1B]] 冻结（历史）
- **动作头**: [[DiT]] + [[Flow Matching|条件流匹配]]，复用 [[GR00T N1.7]] 结构
- **优化器**: AdamW，lr $1 \times 10^{-5}$，cosine 退火
- **训练**: DeepSpeed ZeRO-2，梯度裁剪 norm = 1.0，无梯度累积
- **Batch Size**: 16 per device × 16 GPU = 256
- **训练步数**: 30,000
- **历史窗口** $K$: 16 帧（仅头部摄像头）
- **重规划间隔** $r$: ICC 评估时扫描多个值
- **硬件**: 16 × NVIDIA H100

### 可视化结果

定性 rollout 显示：在 Cook Bread 这种"路径来源决定下一步"的任务上，frame-only 基线在等价手部姿态处会随机选择"放锅 / 放盘"，而 IntentVLA 一致地沿着初始 episode 选定的子目标执行；在 Bimanual Hand Over Roller 任务上同样维持手部分工不抖动。

---

## 批判性思考

### 优点

1. **问题定义清晰**: "观测混叠"是一个早就在策略学习里隐性存在却没人正面命名的问题，论文用 [[AliasBench]] + 视觉检索诊断把它定量化。
2. **方法极简**: 不需要显式 intent label，不动 [[Qwen3-VL]] 主干，不动 [[GR00T N1.7]] 动作头，只加一个冻结编码器 + 一个门控交叉注意力，工程改动极小。
3. **token 选取很巧**: 只取 [[VGGT]] 的 camera token + 4 register token，避免引入 patch token 噪声，同时显式编码了视角变化，这恰好是"我从哪儿来"的信息。
4. **新指标 [[Inter-Chunk Consistency|ICC-L2]] 有独立价值**: 这个度量与最终成功率正交，未来可作为评估任何 chunk-based VLA 的稳定性指标。

### 局限性

1. **只解决短期意图**: 窗口固定 16 帧，对依赖几十秒以前事件的任务（如先按按钮再回来取物）依然无能为力，需要和 [[MemoryVLA]] / [[关键帧记忆库]] 这类长期记忆机制结合。
2. **仅仿真评估**: 全部实验在 [[RoboTwin]] / [[SimplerEnv]] / [[LIBERO]] / [[RoboCasa]] 仿真上，没有真机；摄像头抖动、光照变化下 [[VGGT]] 的几何先验是否仍稳健没验证。
3. **VGGT 冻结的代价**: 冻结避免过拟合，但也意味着无法针对特定 embodiment 调整意图特征；如果场景视角与 VGGT 预训练域差距大可能退化。
4. **门控强度无可解释性**: 论文没给出 $\sigma(\alpha)$ 的最终值，无法判断历史在不同任务上的真实贡献度。
5. **Spoon 任务略降**: 在混叠程度本来就很低的任务上，加历史反而拖累 2 个点左右，说明方法对"什么时候该用历史"还是默认全用，缺乏自适应。

### 潜在改进方向

1. **自适应窗口长度**: 用 mode-switching 概率（公式 10）触发动态选择 $K$，仅在需要时调用 VGGT。
2. **与长期记忆融合**: 用 IntentVLA 作为"反应层"，[[MemoryVLA]] / [[MemER]] 提供长期 anchor，组成 hybrid memory VLA。
3. **可学历史选择**: 把 4 sampled from 16 这种稀疏化思路也学进来，用 attention sparsity 替代固定窗口。
4. **真机验证**: 至少在 [[π₀.₅]] / [[GR00T N1.7]] 已开源的真机平台上跑同样的混叠任务。

### 可复现性评估

- [x] 代码开源（https://github.com/ZGC-EmbodyAI/IntentVLA）
- [ ] 预训练模型（README 待确认）
- [x] 训练细节完整（GPU 数、lr、step 数、batch 都给了）
- [x] 数据集可获取（[[RoboTwin]] / [[SimplerEnv]] / [[LIBERO]] / [[RoboCasa]] 均公开）

---

## 关联笔记

### 基于

- [[Qwen3-VL]]: 当前观测 + 语言的主干 encoder
- [[VGGT]]: 冻结的 3D-aware 历史视觉编码器
- [[GR00T N1.7]]: DiT + Flow Matching 动作头骨架，IntentVLA 直接沿用
- [[Flow Matching]]: 动作生成的训练目标

### 对比

- [[π₀]] / [[π₀.₅]]: 帧条件 VLA 的代表，被 IntentVLA 在所有四个 benchmark 上超越
- [[MemoryVLA]]: 同样关注历史，但用 token-level working memory；IntentVLA 用更紧凑的 register token
- [[VLA-JEPA]]: 用 JEPA 目标增强历史表征，与 IntentVLA 思路互补
- [[OpenVLA]] / [[CoT-VLA]] / [[MolmoAct2]]: 其他 VLA baseline

### 方法相关

- [[观测混叠]]: 论文核心问题定义
- [[短期意图表征]]: 论文要学的中间表示
- [[门控交叉注意力]]: 历史融合机制
- [[历史证据 Token]]: 附加的全局历史摘要
- [[Inter-Chunk Consistency]]: 论文提出的稳定性指标
- [[历史条件化策略]]: 公式 2 给出的策略形式
- [[Action Chunking|动作块]]: 输出形式
- [[DiT]]: 动作头骨架
- [[部分可观测过程]]: 理论框架

### 基准/数据相关

- [[AliasBench]]: 论文新建的 12 任务混叠 benchmark
- [[Back-and-Forth 任务族]] / [[Crossing-Path 任务族]] / [[Bimanual 任务族]] / [[Multi-Goal 任务族]]: 四类混叠模式
- [[RoboTwin]] / [[SimplerEnv]] / [[LIBERO]] / [[RoboCasa]]: 评估平台

---

## 速查卡片

> [!summary] IntentVLA
> - **核心**: 用冻结 [[VGGT]] 编码 $K=16$ 帧历史，经[[门控交叉注意力]]注入 [[Qwen3-VL]] 主干，解决 [[观测混叠]]
> - **方法**: Qwen3-VL (current) + frozen VGGT (history) + gated cross-attention + DiT flow-matching action head
> - **结果**: AliasBench 9.0 → 45.8（+36.8），SimplerEnv 65.3 → 72.9，LIBERO 96.5 → 98.6，RoboCasa 54.6 → 57.0
> - **指标创新**: [[Inter-Chunk Consistency|ICC-L2]] 均值 -17.6%，90 分位 -21.7%
> - **代码**: https://github.com/ZGC-EmbodyAI/IntentVLA

---

*笔记创建时间: 2026-05-15*
