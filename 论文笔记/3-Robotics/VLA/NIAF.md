---
title: "Neural Implicit Action Fields: From Discrete Waypoints to Continuous Functions for Vision-Language-Action Models"
method_name: "NIAF"
authors: [Haoyun Liu, Jianzhuang Zhao, Xinyuan Chang, Tianle Shi, Chuanzhang Meng, Jiayuan Tan, Feng Xiong, Tong Lin, Dongjie Huo, Mu Xu, SongLin Dong, Zhiheng Ma, Yihong Gong, Sheng Zhong]
year: 2026
venue: ICML 2026
tags: [vision-language-action, implicit-neural-representation, siren, hypernetwork, action-representation, impedance-control, robot-manipulation]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2603.01766v2
created: 2026-05-28
---

# 论文笔记：Neural Implicit Action Fields — From Discrete Waypoints to Continuous Functions for VLAs

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 西安交大 / 哈工大 / 中科院深圳先进院 等（合作团队） |
| 日期 | March 2026（v2 May 2026），ICML 2026 接收 |
| 项目主页 | 暂未公开 |
| 对比基线 | [[Pi05\|π₀.₅]], [[OpenVLA-OFT]], [[UniVLA]], [[BEAST]], [[FLOWER]], [[FAST]] |
| 链接 | [arXiv](https://arxiv.org/abs/2603.01766) / [arXiv HTML](https://arxiv.org/html/2603.01766v2) |

---

## 一句话总结

> 把 [[VLA]] 的动作表示从"离散 waypoint"重写为"连续函数"，让 [[MLLM]] 充当 [[SIREN]] 的**频谱调制器**，从而获得 $C^\infty$ 光滑、任意时间分辨率、可解析求导的动作流。

---

## 核心贡献

1. **动作场范式 ([[Neural Implicit Action Field]])**：首次将 VLA 的动作输出由"离散 waypoint 序列 $\{a_1, ..., a_K\}$"重写为"连续函数 $\mathcal{A}: [-1, 1] \to \mathbb{R}^d$"，分辨率与采样率彻底解耦。
2. **分组超调制 ([[Grouped Hyper-Modulation]])**：MLLM 输出一组 token，分组对 [[SIREN]] 各层的频率（权重）与相位（偏置）做独立调制，等价于一个轻量 [[Hypernetwork]]。
3. **解析速度/加加速度监督**：由于 [[SIREN]] 的 $\sin$ 激活保持 $C^\infty$，速度损失 $\mathcal{L}_{vel}$ 与 [[Jerk]] 正则 $\mathcal{L}_{jerk}$ 可直接对网络求导得到，无需有限差分。
4. **可即插即用**：在 [[Florence-2]]、[[Qwen3-VL]]、[[Pi05\|π₀.₅]] 三种主干上均替换原动作头，[[CALVIN]] / [[LIBERO]] SOTA，真机任务也获益。

---

## 问题背景

### 要解决的问题

[[VLA]] 模型当前主流的动作输出形式（离散 token 化 + [[Action Chunking]]）与机器人物理控制系统的**连续时间**本质存在根本错配，导致：

1. 采样率绑死：训练用 10/50 Hz，推理切到 200 Hz 要重新训练；
2. 速度只能用有限差分估计，离散噪声会污染 [[Impedance Control]] 的 D 项；
3. 块边界的 [[Inter-Chunk Consistency|块间不一致]] 会产生 [[Jerk|jerk]] 尖峰。

### 现有方法的局限

| 路线 | 代表 | 问题 |
|------|------|------|
| 离散 token 自回归 | [[OpenVLA]], [[RT-2]] | 频率低、token 误差累积 |
| Action Chunking | [[ACT (Action Chunking Transformer)\|ACT]], [[Diffusion Policy]] | 块边界抖动、分辨率固定 |
| 参数化轨迹 | [[BEAST]], [[FAST]] | B-spline 段数固定，高频细节受限 |
| 流模型 / 扩散 | [[Pi05\|π₀.₅]], [[FLOWER]] | 多步采样昂贵、输出仍是离散 waypoint |

### 本文的动机

把 "动作生成" 看成 "**给定语言+观测，求解一条连续动作曲线**"。如果让网络的输出本身就是一个**对时间 $\tau$ 可任意求导**的隐函数，那么：

- 任意采样率：query $\tau_k = -1 + 2k/(K-1)$，$K$ 任选；
- 解析速度/加速度：对 $\tau$ 直接反向传播；
- 没有显式分块：自然连续，无块间抖动。

[[SIREN]] 的 $C^\infty$ 性质与 [[Hypernetwork]] 的条件化能力为此提供了天然解决方案。

---

## 方法详解

### 模型架构

NIAF 采用 **MLLM 频谱调制 + SIREN 隐式动作场** 架构：

- **输入**：语言指令 $l$ + 多视角观测 $o_t$ + 本体感觉 $s_t$
- **Backbone**：[[Florence-2]] Large / [[Qwen3-VL]] / [[Pi05\|π₀.₅]] 三选一（即插即用）
- **Query Tokens**：可学习嵌入 $\mathbf{E}_{qry} \in \mathbb{R}^{Q \times d}$，$Q = L \times (G+1)$
- **调制头**：每层 SIREN 对应 $G$ 个频率 token（产生 $\gamma^{(\ell)}$）+ $1$ 个相位 token（产生 $\beta^{(\ell)}$）
- **动作场**：$L=3$ 层 [[SIREN]]，隐藏维 256，$\omega_0 = 30$
- **输出**：连续函数 $\mathcal{A}(\tau)$，$\tau \in [-1, 1]$

### 核心模块

#### 模块 1：分组超调制 [[Grouped Hyper-Modulation]]

**设计动机**：标准 [[Hypernetwork]] 一次性生成所有 SIREN 权重，参数量极大且优化困难。NIAF 按 SIREN 层 + 频率/相位拆成 $L \times (G+1)$ 个**小 token**，每个 token 经一个轻 MLP 生成对应的局部调制系数。

**具体实现**：
- 把 MLLM 解码器的 query 输出 $\mathbf{Z} \in \mathbb{R}^{Q \times d}$ 切成 $L$ 块；
- 每块前 $G$ 个 token 经 MLP $\psi_{\gamma_g}$ 拼接成频率调制向量 $\gamma^{(\ell)}$；
- 第 $G+1$ 个 token 经 $\psi_\beta$ 得到相位向量 $\beta^{(\ell)}$；
- 等价于让 [[MLLM]] 充当**频谱意义上的轻量 Hypernetwork**。

#### 模块 2：[[SIREN|Sinusoidal Representation Network]] 动作场

**设计动机**：要让网络输出 $\mathcal{A}(\tau)$ 是 $C^\infty$ 的，且任意阶导数仍是同类网络可表达——正弦激活是天然选择（[[SIREN]] 原始论文已证明）。

**具体实现**：
- 输入：归一化时间 $\tau \in [-1, 1]$；
- 隐层：$L = 3$ 层 $\sin$ 激活，频率因子 $\omega_0 = 30$；
- 调制：第 $\ell$ 层权重/偏置由 MLLM 输出 $(\gamma^{(\ell)}, \beta^{(\ell)})$ 进行**乘性 + 加性**扰动；
- 输出：线性层映射到动作空间 $\mathbb{R}^d$。

#### 模块 3：解析高阶导数 + Jerk 正则

**设计动机**：[[Impedance Control]] 控制律 $u = K_p(\Phi - a) + K_d(\dot\Phi - v)$ 中的速度项，传统方法靠 $\dot\Phi \approx (\Phi_{k+1} - \Phi_k) / \Delta t$ 估计，离散噪声会让 D 项变成放大器；NIAF 直接对 $\tau$ 反传得到解析 $\nabla_\tau \Phi$，速度无噪。

**具体实现**：
- 训练时同时监督 $\Phi$ 与 $\dot\Phi$；
- 加入 [[Jerk]] 正则 $\mathcal{L}_{jerk} = \|\nabla^3_\tau \Phi\|^2$，鼓励高阶光滑；
- 推理时上采样到 200 步，速度曲线几乎无锯齿（见 Figure 9）。

---

## 关键公式

<!-- 公式标题使用 [[概念|名称]] 格式链接到概念库 -->

### 公式 1: [[Neural Implicit Action Field|动作场定义]]

$$
\mathcal{A}: [-1, 1] \to \mathbb{R}^d, \quad \tau \mapsto \mathcal{A}(\tau)
$$

**含义**：把整段动作 chunk 抽象成时间域 $[-1,1]$ 上的一个连续函数，$\tau$ 是规范化时间。

**符号说明**：
- $\tau$：规范化时间，$\tau = -1$ 对应 chunk 起点，$\tau = 1$ 对应终点；
- $d$：单步动作维度（如 7 维末端位姿 + gripper）；
- $\mathcal{A}$：被 [[SIREN]] 参数化的隐式函数。

### 公式 2: 规范化时间采样

$$
\tau_k = -1 + \frac{2k}{K - 1}, \quad k = 0, 1, \ldots, K-1
$$

**含义**：在 $[-1, 1]$ 上均匀采样 $K$ 个点。$K$ 在训练时固定（如 10/50），推理时可任意（如 200），**分辨率与训练完全解耦**。

### 公式 3: [[SIREN]] 前向传播（已调制）

$$
\begin{aligned}
\mathbf{h}^{(0)} &= \tau \\
\mathbf{h}^{(\ell)} &= \sin\!\big(\omega_0 (\hat{\mathbf{W}}^{(\ell)} \mathbf{h}^{(\ell-1)} + \hat{\mathbf{b}}^{(\ell)})\big), \quad \ell = 1, \ldots, L \\
\mathcal{A}(\tau) &= \mathbf{W}_{out} \mathbf{h}^{(L)} + \mathbf{b}_{out}
\end{aligned}
$$

**含义**：以 $\tau$ 为唯一输入的 $L$ 层正弦激活 MLP；正弦激活保证 $C^\infty$ 可导。

**符号说明**：
- $\omega_0 = 30$：频率因子（[[SIREN]] 标准设定）；
- $\hat{\mathbf{W}}^{(\ell)}, \hat{\mathbf{b}}^{(\ell)}$：由 MLLM 调制后的权重与偏置；
- $L = 3$：隐层数。

### 公式 4: [[Grouped Hyper-Modulation|调制规则]]

$$
\hat{\mathbf{W}}^{(\ell)} = \mathbf{W}^{(\ell)} \odot (1 + \gamma^{(\ell)}), \qquad \hat{\mathbf{b}}^{(\ell)} = \mathbf{b}^{(\ell)} + \beta^{(\ell)}
$$

**含义**：MLLM 输出**乘性频率调制** $\gamma^{(\ell)}$ 与**加性相位调制** $\beta^{(\ell)}$，作用在 SIREN 各层。

**符号说明**：
- $\odot$：逐元素乘；
- $\gamma, \beta$：来自下方公式 5 的拼接 / 映射；
- 默认参数 $\mathbf{W}^{(\ell)}, \mathbf{b}^{(\ell)}$ 是与任务无关的"运动先验"。

### 公式 5: MLLM Token → 调制向量

$$
\gamma^{(\ell)} = \mathrm{Concat}\!\big(\psi_{\gamma_1}(\mathbf{Z}_{\ell,1}), \ldots, \psi_{\gamma_G}(\mathbf{Z}_{\ell,G})\big), \quad \beta^{(\ell)} = \psi_\beta(\mathbf{Z}_{\ell, \mathrm{bias}})
$$

**含义**：每层 $G$ 个 token 各自经独立 MLP $\psi_{\gamma_g}$，输出分组拼接得 $\gamma^{(\ell)}$；额外一个 bias token 经 $\psi_\beta$ 得 $\beta^{(\ell)}$。

### 公式 6: 位置监督

$$
\mathcal{L}_{pos} = \frac{1}{K} \sum_{k=1}^{K} \big\|\Phi(\tau_k) - a_{gt, k}\big\|_2^2
$$

**含义**：在采样时间点上做标准 MSE，匹配真值动作。

### 公式 7: 解析速度监督

$$
\mathcal{L}_{vel} = \frac{1}{K} \sum_{k=1}^{K} \big\|\tfrac{2}{T} \nabla_\tau \Phi(\tau_k) - v_{gt, k}\big\|_2^2
$$

**含义**：对 $\tau$ 解析求导得到速度（链式法则中 $\frac{d\tau}{dt} = \frac{2}{T}$），与真值速度对齐。

**符号说明**：
- $T$：chunk 实际时长（秒），用于把 $\tau$-时域换算回物理时域；
- $\nabla_\tau \Phi$：[[Autograd]] 直接反传得到的解析导数。

### 公式 8: [[Jerk]] 正则

$$
\mathcal{L}_{jerk} = \frac{1}{K} \sum_{k=1}^{K} \Big\|\big(\tfrac{2}{T}\big)^3 \nabla^3_\tau \Phi(\tau_k)\Big\|_2^2
$$

**含义**：直接对网络求三阶导，惩罚加加速度，提高轨迹平滑度。

### 公式 9: 总损失

$$
\mathcal{L}_{total} = \mathcal{L}_{pos} + \lambda_{vel} \mathcal{L}_{vel} + \lambda_{jerk} \mathcal{L}_{jerk}
$$

**含义**：位置、速度、加加速度三项联合优化；论文中 $\lambda_{vel} = 1.0$，$\lambda_{jerk} \in [10^{-4}, 10^{-3}]$。

### 公式 10: 推理期 [[Impedance Control]] 律

$$
\mathbf{u}_{cmd} = \mathbf{K}_p \big(\Phi(\tau) - \mathbf{a}_{curr}\big) + \mathbf{K}_d \Big(\tfrac{2}{T} \nabla_\tau \Phi(\tau) - \mathbf{v}_{curr}\Big)
$$

**含义**：把解析位置与解析速度直接喂给阻抗控制器，D 项不再受有限差分噪声污染。

**符号说明**：
- $\mathbf{K}_p, \mathbf{K}_d$：阻抗增益矩阵；
- $\mathbf{a}_{curr}, \mathbf{v}_{curr}$：当前实际位姿、速度（来自机器人状态）。

---

## 关键图表

### Figure 1: From Discrete Waypoints to Continuous Functions

![Figure 1](https://arxiv.org/html/2603.01766v2/x1.png)

**说明**：左图展示主流 VLA 学到的是**离散 waypoint** 序列，被绑定到固定采样率；右图展示 NIAF 把同样的轨迹建模成**连续函数** $\mathcal{A}(\tau)$，可任意时间分辨率 query。

### Figure 2: NIAF 整体架构

![Figure 2](https://arxiv.org/html/2603.01766v2/x2.png)

**说明**：[[MLLM]]（[[Florence-2]] / [[Qwen3-VL]] / [[Pi05]]）的 decoder 末端 $Q$ 个可学习 query token 与视觉、语言进行 cross-attention，得到的 latent 经分组 MLP 调制 $L$ 层 [[SIREN]] 的频率与相位，构成隐式动作场 $\mathcal{A}(\tau)$。

### Figure 3: Action Representation 横向对比

![Figure 3](https://arxiv.org/html/2603.01766v2/x3.png)

**说明**：在 CALVIN ABC→D 与 LIBERO-Long 上，NIAF 的连续函数表示显著超过 token 离散化、B-spline、Flow Matching 等所有现有动作表示。

### Figure 4: Florence-2 主干真机对比

![Figure 4](https://arxiv.org/html/2603.01766v2/x4.png)

**说明**：在 Item Placement / Cup Stacking 真机任务上，把 Florence-2 默认动作头换成 NIAF 即可显著提升成功率。

### Figure 5: π₀.₅ 主干可移植性

![Figure 5](https://arxiv.org/html/2603.01766v2/x5.png)

**说明**：换到 [[Pi05]] 主干上，NIAF 在 Shape Insertion / Towel Folding 这类高精度任务依然有效，验证"动作场"作为通用动作头的可移植性。

### Figure 6: 真机综合结果

![Figure 6](https://arxiv.org/html/2603.01766v2/x6.png)

**说明**：跨 4 个真机任务的成功率柱状图，NIAF 平均超越各主干默认动作头 10–20 个点。

### Figure 7: 控制动力学对比

![Figure 7](https://arxiv.org/html/2603.01766v2/x7.png)

**说明**：顶行关节位置跟踪曲线，底行速度/加速度 profile。NIAF 的速度/加速度曲线明显更平滑，块边界无尖峰。

### Figure 8: 真机评测关键帧

![Figure 8](https://arxiv.org/html/2603.01766v2/x8.png)

**说明**：水杯搬运、积木堆叠等任务的关键帧 rollout，可视化 NIAF 在长程任务上的稳定性。

### Figure 9: 任意采样率

![Figure 9](https://arxiv.org/html/2603.01766v2/x5.png)

**说明**：将动作 chunk 从训练时的 50 步直接上采样到 200 步，NIAF 的速度曲线波动显著降低，证明分辨率独立的优势。

### Figure 10: Water Cup Transport 控制对比

![Figure 10](https://arxiv.org/html/2603.01766v2/x6.png)

**说明**：水杯搬运任务里 NIAF 与基线的实测速度/加速度曲线，离散方法在 chunk 切换处出现明显尖峰。

### Figure 11: SIREN vs. B-spline 拟合

![Figure 11](https://arxiv.org/html/2603.01766v2/x7.png)

**说明**：同一段真值轨迹，[[SIREN]] 比 B-spline 在高频细节上拟合误差更低。

### Figure 12: SIREN + Jerk 正则 vs. B-spline

![Figure 12](https://arxiv.org/html/2603.01766v2/x8.png)

**说明**：加入 $\mathcal{L}_{jerk}$ 之后 SIREN 在保留细节的同时显著降低 jerk，B-spline 则需要增加段数才能逼近，但段数过多又会过拟合噪声。

### Table 1: [[CALVIN]] Benchmark 结果

| Method | Setting | Avg Length (1–5) |
|--------|---------|------------------|
| UP-VLA | ABC→D | 3.83 |
| RoboVLMs | ABC→D | 4.04 |
| [[UniVLA]] | ABC→D | 4.20 |
| [[BEAST]] | ABC→D | 4.28 |
| [[FLOWER]] | ABC→D | 4.39 |
| **NIAF (ours)** | **ABC→D** | **4.47** |
| **NIAF (ours)** | **ABCD→D** | **4.66** |

**说明**：NIAF 在 CALVIN 两个设定上均达到 SOTA，平均链长接近上限 5.0。

### Table 2: [[LIBERO]] Benchmark 结果

| Method | Spatial | Object | Goal | Long | Avg |
|--------|---------|--------|------|------|-----|
| [[π₀]] | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| [[FAST]] | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| [[OpenVLA-OFT]] | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| [[FLOWER]] | 97.8 | 99.3 | 96.7 | 94.0 | 97.0 |
| [[BEAST]] | 97.3 | 99.0 | 97.1 | 94.0 | 96.9 |
| **NIAF (ours)** | **98.7** | **100.0** | **97.4** | **95.5** | **97.9** |

**说明**：NIAF 在 LIBERO 4 个套件上全面领先，LIBERO-Object 满分、Long 任务 95.5%。

### Table 3: [[Qwen3-VL]] 主干扩展性

| Method | Spatial | Object | Goal | Long | Avg |
|--------|---------|--------|------|------|-----|
| FAST | 95.2 | 96.1 | 92.0 | 71.7 | 88.8 |
| OFT | 97.1 | 98.5 | 96.8 | 93.8 | 96.6 |
| PI | 96.8 | 99.0 | 95.6 | 88.4 | 95.0 |
| GR00T | 97.0 | 98.7 | 96.4 | 91.0 | 95.8 |
| **Qwen3-VL-NIAF** | **98.3** | **99.8** | **97.5** | **95.0** | **97.7** |

**说明**：换成更大主干后 NIAF 仍是最强动作头，Long 任务相比 PI 提升 6.6 个点。

### Table 4: 仿真训练超参数

| 项目 | 值 |
|------|----|
| Chunk length | 10 |
| Optimizer | AdamW, $\beta=(0.9, 0.95)$ |
| Learning rate | 1e-5 ~ 2e-5 |
| Batch size | 32 |
| GPU | 2–4 张 |
| Steps | 20k |

### Table 5: 真机训练超参数

| 项目 | 值 |
|------|----|
| Chunk length | 50 |
| Learning rate | 1e-5 ~ 2.5e-5 |
| Epoch / Steps | 30 epoch 或 30k steps |
| GPU | 4–8 张 |
| Demonstrations | 50–100 / 任务 |

### Table 6: 消融实验（CALVIN ABC→D, Avg Length）

| 配置 | 取值 | Avg Length | 说明 |
|------|------|------------|------|
| Chunk size | 5 / **10** / 15 / 20 | 4.21 / **4.47** / 4.39 / 4.30 | $H=10$ 是甜蜜点 |
| Weight Groups | 16 / 32 / **64** | 4.30 / 4.38 / **4.47** | 分组越多越好但收益递减 |
| Activation | ReLU / **Sine** | 3.81 / **4.47** | Sine 远胜 ReLU |
| SIREN Depth | 2 / **3** / 4 | 4.32 / **4.47** | 3 层最优，更深容易过拟合 |

**关键发现**：Sine 激活带来约 +0.66 的 Avg Length 提升，是性能的主要来源。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[CALVIN]] | 24k+ 演示 | 长程语言条件操作 | ABC→D / ABCD→D 训练+评估 |
| [[LIBERO]] | 130 任务 × N 演示 | 4 套件分别测空间/物体/目标/长程 | 训练+评估 |
| 真机 (单臂) | 50–100 / 任务 | Item Placement / Cup Stacking / Shape Insertion / Water Cup Transport | 训练+评估 |

### 实现细节

- **Backbone**：[[Florence-2]] Large（默认）/ [[Qwen3-VL]] / [[Pi05]]；
- **优化器**：AdamW, betas=(0.9, 0.95)；
- **Batch Size**：32（仿真）/ 16（真机）；
- **训练步数**：仿真 20k step；真机 30 epoch；
- **硬件**：仿真 2–4 GPU；真机 4–8 GPU；
- **关键超参**：$\omega_0 = 30, L = 3, G = 64, H = 10$（仿真）/ $50$（真机）。

### 可视化结果

- Figure 9 显示动作 chunk 上采样到 4× 时 NIAF 速度曲线几乎无锯齿，离散方法波动显著；
- Figure 11/12 显示 [[SIREN]] 比 B-spline 在高频细节与 jerk 之间更易取得平衡；
- 真机水杯搬运任务里，NIAF 末端速度方差降低约 60%，水面晃动可观察到明显减弱。

---

## 实验结果

### 仿真主线

- **[[CALVIN]] ABC→D**：4.47（SOTA，超 BEAST 0.19、FLOWER 0.08）；
- **[[CALVIN]] ABCD→D**：4.66（接近 5.0 上限）；
- **[[LIBERO]] 平均**：97.9%（SOTA），LIBERO-Object 100%、Long 95.5%；
- **[[Qwen3-VL]] 主干**：97.7% 平均，Long 95.0%（相对 π/FAST 提升 6+ 点）。

### 真机评估

| 任务 | 主干 | 成功率 |
|------|------|--------|
| Item Placement | [[Florence-2]] | **90%** |
| Cup Stacking | [[Florence-2]] | **80%** |
| Shape Insertion | [[Pi05]] | 显著优于 π₀.₅ 默认头 |
| Water Cup Transport | [[Florence-2]] | 水面波动 ↓60% |
| Towel Folding | [[Pi05]] | 显著优于 π₀.₅ 默认头 |

### 控制动力学

- 块边界 jerk 尖峰几乎消失（Figure 7、Figure 10）；
- 训练 50 步动作 chunk，推理直接 query 200 步而无需重训，仍保持平滑（Figure 9）；
- 速度/加速度由解析求导获取，[[Impedance Control]] D 项噪声显著降低。

### 关键 takeaway

| 现象 | 解释 |
|------|------|
| Sine vs ReLU +0.66 Avg Length | $C^\infty$ 性质对动作场必不可少 |
| 64 groups > 32 > 16 | 频谱细粒度调制可让动作更精细 |
| Long 任务收益最大 | 连续函数避免了 chunk 间累积误差 |
| 真机水面波动 ↓60% | 解析速度直接利好阻抗控制 |

---

## 批判性思考

### 优点

1. **范式新颖**：第一次把"动作表示"从离散序列彻底切到连续函数，思路简洁优美；
2. **物理友好**：直接对接阻抗控制器，不再需要数值微分；
3. **跨主干通用**：作为"动作头"在 Florence-2/Qwen3-VL/π₀.₅ 上都能即插即用。

### 局限性

1. **整段 chunk 一次生成**：仍是块级，不是真正逐步在线（没解决长程历史依赖问题，[[AR-VLA]] 在这点上更彻底）；
2. **SIREN 表达能力依赖 $\omega_0$**：不同任务最优 $\omega_0$ 可能不同，论文未深挖；
3. **没有显式时序对齐机制**：若任务时长在测试时大幅变化，$\tau \in [-1,1]$ 的语义可能漂移；
4. **代码 / 项目页尚未公开**：复现门槛较高。

### 潜在改进方向

1. **结合 [[AR-VLA]] 的自回归动作专家**：让每个 chunk 的 $\mathcal{A}(\tau)$ 进一步以历史 chunk 为条件；
2. **自适应 $\omega_0$**：让 MLLM 同时输出频率因子，对高速 / 低速任务自适应；
3. **多尺度动作场**：分层 SIREN，粗尺度管整体轨迹、细尺度补高频；
4. **直接在 latent 空间做动作场**：避免末端动作维度对网络宽度的限制。

### 可复现性评估

- [ ] 代码开源（未释出）
- [ ] 预训练模型（未释出）
- [x] 训练细节较完整
- [x] 数据集均为公开 benchmark

---

## 关联笔记

### 基于

- [[SIREN]]：动作场骨架直接来自 SIREN 的 $\sin$ 激活体系
- [[Hypernetwork]]：分组超调制本质是轻量 hypernet
- [[Neural Implicit Representation]]：把 NeRF/SIREN 那套搬到动作域

### 对比

- [[BEAST]]：用 B-spline 参数化轨迹（NIAF 的主要对比之一）
- [[FAST]]：DCT 频域离散化动作 token（频域思路相似但仍离散）
- [[Pi05]]：流匹配连续动作头（NIAF 直接替换它的动作头）
- [[OpenVLA-OFT]]：动作头优化（NIAF 进一步把离散化彻底去掉）

### 方法相关

- [[SIREN]]：核心 backbone
- [[Grouped Hyper-Modulation]]：核心调制机制
- [[Neural Implicit Action Field]]：本文提出的概念
- [[Impedance Control]]：直接受益方
- [[Jerk]]：核心评估指标
- [[Action Chunking]]：本文要"超越"的旧范式

### 硬件/数据相关

- [[CALVIN]]：主要 benchmark
- [[LIBERO]]：第二个主要 benchmark
- [[Florence-2]] / [[Qwen3-VL]] / [[Pi05]]：三种受测主干

---

## 速查卡片

> [!summary] Neural Implicit Action Fields (NIAF)
> - **核心**：把 VLA 的离散 waypoint 输出换成对 $\tau$ 的连续函数 $\mathcal{A}(\tau)$
> - **方法**：[[MLLM]] 输出分组调制 token → 调制 [[SIREN]] 各层频率/相位 → 解析求导监督 $\dot\Phi, \dddot\Phi$
> - **结果**：[[CALVIN]] 4.47/4.66 + [[LIBERO]] 97.9% + 真机水面波动 ↓60%
> - **代码**：暂未公开

---

*笔记创建时间: 2026-05-28*
