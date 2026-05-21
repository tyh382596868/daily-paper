---
title: "Composition of Memory Experts for Diffusion World Models"
method_name: "CoME"
authors: [Sebastian Stapf, Pablo Acuaviva Huertos, Aram Davtyan, Paolo Favaro]
year: 2026
venue: ICLR 2026
tags: [world-model, video-diffusion, memory, product-of-experts, test-time-training, long-context]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.18813v1
created: 2026-05-20
---

# 论文笔记：Composition of Memory Experts for Diffusion World Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | University of Bern |
| 日期 | May 2026 |
| 项目主页 | 未公布 |
| 对比基线 | [[DiT]] (Full Attention) / Chunked SSM ([[Mamba]]) / NWM |
| 链接 | [arXiv](https://arxiv.org/abs/2605.18813) / Code: 未公布 |

---

## 一句话总结

> 把视频[[世界模型]]的「记忆」拆成短期 / 长期 / 空间三种专属专家，用 **Product of Contrastive Experts** 在测试时组合，达到与全注意力 Transformer 相当的一致性，却只需其 1/12 的训练开销。

---

## 核心贡献

1. **多记忆专家分解**: 不再用单一架构承载全部历史，而是把记忆拆成 [[短期记忆专家]]（局部动态）、[[长期记忆专家]]（情节记忆）、[[空间记忆专家]]（几何一致性）三类专属 [[记忆专家]]，分别针对各自最擅长的时间尺度优化。
2. **Product of Contrastive Experts (PoCE)**: 提出 [[Product of Contrastive Experts]] 公式，在 [[Product of Experts]] 组合中引入对比项，抑制「虚假模式」（spurious modes）而**不收缩核形状、不损失多样性**，并给出理论命题（KDE 不相交支撑下的混合权重重加权）。
3. **基于扩散的外部长期记忆**: [[长期记忆专家]] 用 [[测试时训练|test-time finetuning]]（[[LoRA]] 适配器）把长程历史写入权重，避免把上下文长度直接拉长带来的二次注意力开销，可扩展到 480+ 帧而无饱和。

---

## 问题背景

### 要解决的问题

视频[[世界模型]]需要在生成未来帧时保持与**过去观测**的一致性（场景回访、循环路径、长程记忆），但同时要能高效扩展到长上下文。

### 现有方法的局限

- **Transformer 架构**：高保真，但注意力开销随上下文长度二次增长，长上下文不可行。
- **循环 / 状态空间模型（[[Mamba]] 等）**：可高效扩展，但把历史压缩进固定状态，丢失细节、保真度下降。
- 二者构成「保真度 vs. 效率」的根本权衡，单一架构难以两全。

### 本文的动机

不同记忆需求（局部动态、情节回忆、空间一致性）本质不同，应由**不同的专属专家**处理；[[扩散模型]]天然支持在测试时通过 [[Product of Experts]] 把多个条件分布相乘组合，无需重新训练即可灵活拼装记忆能力。难点在于：朴素 PoE 会引入虚假模式，朴素 tempering 又会过度锐化、损失多样性 —— PoCE 正是为此设计。

---

## 方法详解

### 模型架构

CoME 采用 **多专家扩散组合（Composition of Memory Experts）** 架构。视频序列记为 $\mathbf{x}\in\mathbb{R}^{T\times 3\times H\times W}$，过去帧集合记为 $\mathcal{M}$。

- **输入**: 不同时间尺度的上下文 —— 通用上下文 $c$、短期上下文 $c_{ST}$（10–100 帧）、长期上下文 $c_{LT}$（100–1000 帧）、空间信号 $\mathcal{S}$（相机位姿 / 点图）。
- **Backbone**: 预训练 Image-to-Video（I2V）[[扩散变换器|DiT]]，3 帧输入、17 帧预测。
- **核心模块**: 三类 [[记忆专家]] + [[Product of Contrastive Experts]] 组合层。
- **输出**: 与全部历史证据一致的未来视频帧分布 $p_{\text{CoME}}(\mathbf{x}\mid c,c_{ST},c_{LT},\mathcal{S})$。

> 架构关系（用结构化列表描述，非流程图）：
> - 基础先验 $p_\theta(\mathbf{x}\mid c)$ —— 预训练 I2V 模型，提供生成保真度
> - 短期记忆专家 $p_\phi(\mathbf{x}\mid c_{ST})$ —— 滑窗注意力捕捉局部动态
> - 长期记忆专家 $p_{\psi(c_{LT})}(\mathbf{x}\mid c)$ —— [[LoRA]] 权重写入情节记忆
> - 空间记忆专家 $p_\lambda(\mathbf{x}\mid\mathcal{S})$ —— 几何一致性先验
> - 四者各自配一个无条件「对比基线」，按 [[Product of Contrastive Experts|PoCE]] 公式相乘组合

### 核心模块

#### 模块1: 短期记忆专家 (Short-Term Memory, STM)

**设计动机**: 利用 [[扩散模型]]在中等扩展上下文上的高保真能力捕捉**局部动态与细节**。

**具体实现**:
- 用带[[滑动窗口注意力]]的 [[DiT]]（DiT-S），chunk size 20，在第 2、6 层插入全注意力块。
- 输入 33 帧、输出 17 帧；上下文窗口 $c_{ST}$ 约 10–100 帧。
- 专家分布记为 $p_\phi(\mathbf{x}\mid c_{ST})$，对比基线为无条件版本 $\bar{p}_\phi(\mathbf{x})=p_\phi(\mathbf{x}\mid\varnothing)$。
- 局限：受有限上下文窗口约束，无法记住更久远的历史。

#### 模块2: 长期记忆专家 (Long-Term Memory, LTM)

**设计动机**: 通过 [[测试时训练|test-time weight finetuning]] 把情节知识写入**模型权重**，而非拉长输入上下文，从而绕开二次注意力开销。

**具体实现**:
- 双重条件通道：① 标准上下文条件（输入帧 $c$）；② 权重条件 —— 在长程历史 $c_{LT}$（100–1000 帧）上微调得到的权重 $\psi(c_{LT})$。
- 微调用 [[LoRA]] 适配器（典型 rank 64，约 7% 参数），冻结主干：
  - LoRA 提供**隐式正则**，防止灾难性遗忘；
  - 降低计算开销；
  - 保留原模型的泛化能力。
- 专家分布记为 $p_{\psi(c_{LT})}(\mathbf{x}\mid c)$，对比基线为 $\bar{p}_\psi(\mathbf{x})=p_{\psi(\varnothing)}(\mathbf{x}\mid\varnothing)$。

#### 模块3: 空间长期记忆专家 (Spatial LTM, SLTM)

**设计动机**: 在循环路径或视觉相似地点，单靠 LTM 仍会因视觉线索歧义而漂移；需要引入显式几何信号消歧。

**具体实现**:
- 把上下文 $\mathcal{M}$ 扩展出空间信号 $\mathcal{S}=\text{Enc}_\lambda(\mathcal{M})$，其中 $\text{Enc}_\lambda(\cdot)$ 可以是 [[SLAM]] 算法或学习式编码器。
- 空间先验记为 $p_\lambda(\mathbf{x}\mid\mathcal{S})$，条件为绝对相机位姿；输入投影层的 patch size 加倍以容纳空间通道。
- 编码器把相对位姿 + 帧估计映射为绝对位姿，借此消歧地点、减少长程漂移。
- 对比基线为丢弃空间先验的 $p_\lambda(\mathbf{x}\mid\varnothing)$。

---

## 关键公式

### 公式1: [[去噪扩散概率模型|DDPM 训练目标]]

$$

\mathcal{L}(\theta)=\mathbb{E}_{\mathbf{x}_0,\,t,\,\epsilon\sim\mathcal{N}(0,\mathbf{I})}\left[\;\|\epsilon-\epsilon_\theta(\mathbf{x}_t,t)\|^2\;\right]

$$

**含义**: 训练噪声预测网络 $\epsilon_\theta$ 拟合前向加噪过程中注入的噪声；隐式学到分数函数 $\nabla_{\mathbf{x}_t}\log p_\theta(\mathbf{x}_t)\approx-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(\mathbf{x}_t,t)$。LTM 的测试时微调也用此损失。

**符号说明**:
- $\mathbf{x}_0$: 原始干净帧；$\mathbf{x}_t$: 第 $t$ 步加噪后的帧
- $\epsilon$: 标准高斯噪声；$\epsilon_\theta$: 噪声预测网络
- $t$: 扩散步；$\beta_t$: 噪声调度参数；$\bar{\alpha}_t=\prod_{s=1}^t(1-\beta_s)$: 累积保留系数

### 公式2: [[去噪扩散概率模型|逆过程采样]]（Langevin 式更新）

$$

\mathbf{x}_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\epsilon_\theta(\mathbf{x}_t,t)\right)+\sigma_t\,\eta,\qquad \eta\sim\mathcal{N}(0,\mathbf{I})

$$

**含义**: 反向去噪逐步从纯噪声采样出视频帧；CoME 在此采样步中把各专家的分数（噪声估计）按 PoCE 权重相加。

**符号说明**:
- $\alpha_t=1-\beta_t$；$\sigma_t$: 由噪声调度导出的采样方差系数
- $\eta$: 每步注入的随机噪声

### 公式3: [[Product of Experts]] 组合

$$

p(\mathbf{x})\propto p_{\text{query}}(\mathbf{x})\prod_i p_i(\mathbf{x})

$$

**含义**: 把多个专家分布相乘，得到「同时满足所有证据源」的联合分布；使得选择性、内容寻址式生成无需重训练即可实现。

**符号说明**:
- $p_i(\mathbf{x})$: 第 $i$ 个专家的分布
- $p_{\text{query}}(\mathbf{x})$: 查询 / 基础分布

### 公式4: [[Product of Contrastive Experts]] (PoCE)

$$

p(\mathbf{x}\mid\mathcal{M})\propto\prod_{i=1}^{K}\tilde{p}_i(\mathbf{x}),\qquad \tilde{p}_i(\mathbf{x})\propto p_i(\mathbf{x})^{\alpha_i}\,\bar{p}_i(\mathbf{x})^{1-\alpha_i}

$$

**含义**: 本文核心创新。每个专家用「条件分布 $p_i$」与「无条件基线 $\bar{p}_i$」做对比加权（$\alpha_i>1$），再相乘组合。相比朴素 tempering $p_i(\mathbf{x})^\alpha$（对高斯会把方差缩成 $\frac{1}{\alpha}\Sigma$、过度锐化损失多样性），PoCE 改为**重加权混合分量**而不扭曲核形状。

**符号说明**:
- $\tilde{p}_i$: 第 $i$ 个对比专家；$\bar{p}_i$: 第 $i$ 个专家的无条件基线
- $\alpha_i>1$: 对比系数（contrast coefficient），越大模式分离越强
- $\mathcal{M}$: 全部过去帧上下文；$K$: 专家数

### 公式5: PoCE 的 KDE 重加权（Proposition 1）

在 [[核密度估计|KDE]] 设定下，设 $p_i(\mathbf{x})=\sum_{k=1}^M\pi_k^i h_k(\mathbf{x})$、$\bar{p}_i(\mathbf{x})=\sum_{k=1}^M\omega_k^i h_k(\mathbf{x})$，且核 $\{h_k\}$ **支撑不相交**、$\sum_k\pi_k^i=\sum_k\omega_k^i=1$、$\int h_k\,d\mathbf{x}=1$，则：

$$

\tilde{p}_i(\mathbf{x})\propto\sum_{k=1}^M(\pi_k^i)^{\alpha_i}(\omega_k^i)^{1-\alpha_i}\,h_k(\mathbf{x})

$$

**含义**: 理论证明 PoCE 只改变各核（模式）的**权重**，不改变核形状本身。特别地，当基线取均匀 $\omega_k^i=\tfrac{1}{M}$ 时 $(\omega_k^i)^{1-\alpha_i}$ 成常数被归一化掉，得到 $\tilde{p}_i(\mathbf{x})\propto\sum_k(\pi_k^i)^{\alpha_i}h_k(\mathbf{x})$。定义对比比 $\rho_k=\pi_k^i/\omega_k^i$：$\alpha_i>1$ 时，$\rho_k>1$ 的真实模式被放大，$\rho_k<1$ 的虚假模式被压制。

**符号说明**:
- $\pi_k^i$: 条件分布在第 $k$ 个核上的混合权重；$\omega_k^i$: 基线分布对应权重
- $h_k$: 第 $k$ 个核函数；$M$: 核数量
- $\rho_k=\pi_k^i/\omega_k^i$: 对比比，衡量条件相对基线对该模式的强调程度

### 公式6: CoME 完整组合

$$

\begin{aligned}
p_{\text{CoME}}(\mathbf{x}\mid c,c_{ST},c_{LT},\mathcal{S})\propto\;
& \Big[\,p_\theta(\mathbf{x}\mid\varnothing)^{1-\alpha_0}\,p_\theta(\mathbf{x}\mid c)^{\alpha_0}\,\Big]\\
\times\;& \Big[\,p_\phi(\mathbf{x}\mid\varnothing)^{1-\alpha_1}\,p_\phi(\mathbf{x}\mid c_{ST})^{\alpha_1}\,\Big]\\
\times\;& \Big[\,p_{\psi(\varnothing)}(\mathbf{x}\mid c)^{1-\alpha_2}\,p_{\psi(c_{LT})}(\mathbf{x}\mid c)^{\alpha_2}\,\Big]\\
\times\;& \Big[\,p_\lambda(\mathbf{x}\mid\varnothing)^{1-\alpha_3}\,p_\lambda(\mathbf{x}\mid\mathcal{S})^{\alpha_3}\,\Big]
\end{aligned}

$$

**含义**: 把四个专家（基础先验 $p_\theta$、短期 $p_\phi$、长期 $p_{\psi}$、空间 $p_\lambda$）各自的「条件 / 无条件」对比对相乘，得到 CoME 的最终生成分布。每个专家有独立的平衡系数 $\alpha_i\ge1$。

**符号说明**:
- $\alpha_0,\alpha_1,\alpha_2,\alpha_3\ge1$: 四个专家对的平衡系数（需按数据集调）
- $p_\theta$: 预训练基础 I2V 模型；$p_\phi$: STM；$p_{\psi(c_{LT})}$: LTM（权重微调）；$p_\lambda$: 空间先验

---

## 关键图表

### Figure 1: Overview / 系统概览

![Figure 1](https://arxiv.org/html/2605.18813v1/sections/image.png)

**说明**: 记忆增强扩散[[世界模型]]总览。短期动态、长期情节记忆、空间一致性三类专属专家通过 [[Product of Experts]] 组合，生成与过去观测一致的未来帧。

### Figure 2: Context Length Scaling / 上下文长度扩展

![Figure 2](https://arxiv.org/html/2605.18813v1/sections/resub/context4.png)

**说明**: Memory Maze 上 [[感知图像相似度|LPIPS]]（越低越好）随上下文长度的变化。上下文越长重建越忠实，直到 480 帧仍**无饱和迹象**，说明 LTM 可有效扩展到长上下文。

### Figure 3: Continuous Stream Evaluation / 连续流式评估

![Figure 3](https://arxiv.org/html/2605.18813v1/sections/resub/long_gt2.png)

**说明**: Memory Maze 上以 10 段 × 20 帧的方式连续评估。CoME 增量式记忆，随迭代持续保持一致性；STM 单独使用早期改善后趋于平台，STM+LTM 则**持续改善**，体现两者的复合增益。

### Figure 4: RealEstate10K Recall / 场景回访定性结果

![Figure 4](https://arxiv.org/html/2605.18813v1/x1.png)

**说明**: RealEstate10K 上先做六段前向 rollout 再反转相机轨迹。与无长期记忆的基础模型不同，CoME 在两个例子中都能**正确回忆起初始帧**，保证场景一致性。

### Figure 5: Mixture of Contrastive Experts / 对比专家示意

![Figure 5](https://arxiv.org/html/2605.18813v1/sections/resub/toy_example.png)

**说明**: [[Product of Contrastive Experts|PoCE]] 的玩具示例。(a) 单个专家建模为高斯混合，模式按几何衰减；(b) 对比后的单个专家，模式权重被均匀化；(c) 普通 PoE vs. 指数放缩 PoE vs. PoCE —— PoCE 在压制不一致模式（最右峰）的同时保留主导的左核形状。

### Table 1: 不同记忆配置对比 (Memory Maze)

| Method | LPIPS ↓ | SSIM ↑ | PSNR ↑ |
|--------|---------|--------|--------|
| Base | 0.209 | 0.771 | 19.16 |
| + STM | 0.156 | 0.820 | 21.29 |
| + LTM | 0.171 | 0.805 | 19.98 |
| + SLTM | 0.150 | 0.833 | 20.65 |
| + STM+LTM | 0.114 | 0.862 | 22.32 |
| **CoME (all)** | **0.097** | **0.892** | **23.07** |
| Sliding window | 0.183 | 0.753 | 19.02 |
| SSM (Mamba) | 0.158 | 0.828 | 20.62 |
| Full attention | 0.113 | 0.859 | 22.78 |

**说明**: CoME（全部专家）LPIPS 0.097，相比 Base 0.209 改善约 54%，且**优于全注意力 Transformer**（0.113），却只需其约 1/12 的训练开销。每帧 2 步梯度更新。

### Table 2: RECON 基准上的规划精度

| Method | ATE ↓ | RPE ↓ |
|--------|-------|-------|
| GNM | 1.87 | 0.73 |
| NOMAD | 1.93 | 0.52 |
| NWM | 1.13 | 0.35 |
| CoME - STM | 1.05 | 0.32 |
| CoME - LTM | 1.10 | 0.32 |
| NWM+STM | 0.98 | 0.30 |
| NWM+LTM | 1.07 | 0.33 |
| **CoME (combined)** | **0.96** | **0.28** |

**说明**: 100 条采样轨迹评估。[[相机位姿评估指标|ATE]]（绝对轨迹误差，衡量全局偏差）与 RPE（相对位姿误差，衡量局部一致性）上 CoME 均最优，相比 NWM 分别降低约 15% / 20%。

### Table 3: RealEstate10K 回访评估

| Method | LPIPS ↓ | PSNR ↑ | SSIM ↑ |
|--------|---------|--------|--------|
| Base | 0.405 | 19.8 | 0.794 |
| **CoME** | **0.359** | **21.3** | **0.83** |
| HG-v | 0.414 | 19.2 | 0.764 |
| HG-t | 0.400 | 19.7 | 0.788 |

**说明**: 协议为 3 段前向 rollout 后接 3 段反向 rollout。CoME 能准确重建反向轨迹，LPIPS 较 Base 改善约 11%。

### Table 4: 对比专家消融

| 配置 | w/o Contrastive (LPIPS) | w/ Contrastive (LPIPS) |
|------|-------------------------|------------------------|
| Base | 0.203 | 0.200 |
| STM | 0.175 | 0.156 |
| LTM | 0.188 | 0.171 |
| SLTM | 0.178 | 0.150 |
| STM+LTM | 0.170 | 0.114 |
| All experts | 0.192 | 0.097 |

**关键发现**: 朴素 [[Product of Experts|PoE]]（不带对比项）在多专家组合时几乎失效（All experts 仅 0.192）；引入 [[Product of Contrastive Experts|对比加权]]后 All experts 降到 0.097 —— 对比公式是稳定提升的**必要条件**。

### Table 5: LTM 适配容量与上下文长度

| 上下文帧数 | Rank 8 (1%) | Rank 32 (4%) | Rank 64 (7%) | Rank 256 (25%) | Full (100%) |
|------------|-------------|--------------|--------------|----------------|-------------|
| 50 帧 | 0.193 | 0.175 | 0.161 | 0.162 | 0.221 |
| 150 帧 | 0.161 | 0.169 | 0.158 | 0.142 | 0.188 |
| 450 帧 | 0.146 | 0.128 | 0.124 | 0.118 | 0.125 |

**关键发现**: 指标为 LPIPS，列为 [[LoRA]] rank 占模型参数比例。Rank 8 欠拟合；Rank 64 在中短上下文是甜点；全量微调（Full）在数据多样性不足时反而**过拟合**。上下文越长（450 帧）效果越好，验证 LTM 的长程扩展性。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| Memory Maze | 30K 轨迹 × 1000 帧 | 3D 迷宫导航，含绝对/相对位姿，循环路径丰富 | 训练/测试（主战场） |
| RECON | 5K+ 真实户外导航轨迹 | Clearpath Jackal 机器人，RGB/立体/热成像/LiDAR/GPS/IMU | 规划评估 |
| RealEstate10K | 室内房产视频，256×256 | 含精确 3D 相机位姿，跨房间视角 | 回访/记忆测试 |
| DMLab-40K | 40K 动作条件视频，64×64 × 300 帧 | 程序生成 7×7 随机迷宫，每集布局静态 | 确定性环境测试 |
| Minecraft-200K | 200K 游戏视频，256×256 × 300 帧 | 离散动作（前/左/右），程序生成沼泽地形 | 动态地形测试 |
| Memory Cards（新提出） | 100K 序列 × 250 帧（9:1 划分） | 4×4 网格 8 个物体，翻牌覆盖/揭示机制 | 离散记忆/回忆测试 |

### 实现细节

- **Backbone**: 预训练 I2V [[DiT]]（3 帧输入 / 17 帧预测）。
- **STM**: DiT-S + 滑窗注意力，chunk size 20，第 2、6 层全注意力；33 帧输入 / 17 帧输出。
- **LTM**: 冻结主干 + [[LoRA]]（典型 rank 64），用[[去噪扩散概率模型|扩散损失]]做[[测试时训练]]，每个上下文帧 2 步梯度。
- **训练**: Memory Maze 训 150K 步；全注意力需约 60× STM 的算力、12× 基础模型算力，且收敛需 3× 步数、训练不稳定需特殊调度。
- **推理开销**: 流式评估每帧 2 步记忆化；LoRA rank 64 带来约 4× 开销，rank 16 约 2×。

### 评测指标

- **重建质量**: [[感知图像相似度|LPIPS]] ↓、[[结构相似性|SSIM]] ↑、[[PSNR]] ↑。
- **导航/规划**: [[相机位姿评估指标|ATE]] ↓（全局轨迹误差）、RPE ↓（相对位姿误差）。
- **离散回忆**: 逐 tile MSE（阈值 $2\times10^{-5}$）→ 回忆准确率。

### 可视化结果

- **连续流式（Fig. 3）**: STM 单独早期改善后平台化；STM+LTM 随迭代持续改善，LTM 成功整合新观测。
- **回访（Fig. 4 / Table 3）**: CoME 能精确回忆起初始帧；基础模型只能生成「看似合理但不一致」的帧。
- **离散记忆（Memory Cards）**: CoME 20 步记忆化后回忆准确率 79.2%，基础 SD 仅 13%（接近随机）。
- **确定性环境（DMLab）**: CoME LPIPS 0.456 vs. 基线 0.558，改善约 18%。

---

## 批判性思考

### 优点

1. **理论扎实**: PoCE 给出 KDE 不相交支撑下的重加权命题，清楚解释了「为何朴素 tempering 损害多样性、对比加权不损害」，把 PoE 在扩散世界模型中的失效原因讲透。
2. **效率优势显著**: 用多个小专家 + 测试时 LoRA 微调，达到甚至超过全注意力 Transformer 的一致性，训练算力仅约 1/12，且训练稳定。
3. **即插即用**: 各专家可在测试时任意组合，无需联合重训练；Table 1 的逐专家消融清晰展示了每个专家的边际贡献。
4. **评测全面**: 覆盖 6 个数据集（含自建 Memory Cards），重建、规划、回访、离散记忆多维度验证。

### 局限性

1. **在线计算开销**: LTM 的测试时微调在线适配阶段带来 2–4× 开销，对实时部署是负担。
2. **多样性折损**: 对比公式在压制虚假模式的同时也会削弱「次要但合理」的模式（diversity trade-off），生成多样性受限。
3. **空间信号依赖**: SLTM 需要可靠的位姿/SLAM 信号，在缺乏纹理特征的环境中受限。
4. **超参敏感**: 各专家对比系数 $\alpha_i$ 需逐数据集调，泛化部署成本高。

### 潜在改进方向

1. 让 LTM 显式建模时间依赖，而非仅靠权重隐式编码情节记忆。
2. 引入更丰富、更彻底的空间条件形式，降低对外部 SLAM 的依赖。
3. 自适应地学习对比系数 $\alpha_i$，减少逐数据集调参。

### 可复现性评估

- [ ] 代码开源（截至笔记日期未公布）
- [ ] 预训练模型
- [x] 训练细节完整（含算力对比、LoRA rank、步数）
- [x] 数据集可获取（多为公开数据集，Memory Cards 为自建）

---

## 关联笔记

### 基于

- [[Product of Experts]]: PoCE 是其对比式改进
- [[去噪扩散概率模型]]: 各专家均为条件扩散模型
- [[LoRA]]: LTM 测试时权重微调的具体手段
- [[DiT]]: 各专家的骨干网络

### 对比

- [[Mamba]] / [[状态空间模型]]: 论文将其作为「高效但压缩历史」的基线
- 全注意力 Transformer: 高保真但二次开销的对照上界

### 方法相关

- [[Product of Contrastive Experts]]: 核心组合公式
- [[记忆专家]]: 短期/长期/空间三类专家的统称
- [[短期记忆专家]]: 局部动态
- [[长期记忆专家]]: 情节记忆 + 测试时微调
- [[空间记忆专家]]: 几何一致性
- [[测试时训练]]: LTM 的关键机制
- [[世界模型]]: 任务大类

### 硬件/数据相关

- [[感知图像相似度]] / [[结构相似性]] / [[PSNR]]: 重建质量指标
- [[相机位姿评估指标]]: 规划精度指标

---

## 速查卡片

> [!summary] Composition of Memory Experts for Diffusion World Models
> - **核心**: 把视频世界模型的记忆拆成短期/长期/空间三类专属专家，用对比式 Product of Experts 在测试时组合
> - **方法**: PoCE（对比专家乘积，理论上只重加权模式不锐化核） + LoRA 测试时微调写入长期记忆 + 空间位姿先验
> - **结果**: Memory Maze LPIPS 0.097（优于全注意力 0.113），训练算力约 1/12；规划 ATE/RPE 较 NWM 降 15%/20%；上下文可扩展到 480+ 帧无饱和
> - **代码**: 未公布（ICLR 2026）

---

*笔记创建时间: 2026-05-20*
