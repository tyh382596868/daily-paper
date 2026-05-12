---
title: "Learning Visual Feature-Based World Models via Residual Latent Action"
method_name: "RLA-WM"
aliases: [RLA-WM, Residual Latent Action, RLA]
authors: [Xinyu Zhang, Zhengtong Xu, Yutian Tao, Yeping Wang, Yu She, Abdeslam Boularias]
year: 2026
venue: arXiv
tags: [world-model, latent-action, flow-matching, dino-features, robot-learning, visual-rl]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.07079v1
created: 2026-05-12
---

# 论文笔记：Learning Visual Feature-Based World Models via Residual Latent Action

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Rutgers University、Purdue University、University of Wisconsin-Madison |
| 日期 | May 2026 |
| 项目主页 | https://mlzxy.github.io/rla-wm |
| 对比基线 | [[DINO-WM]]、[[RAE]]、FM-WM、[[Vid2World]]、[[AdaWorld]]、[[UniVLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.07079) / Code: 见项目主页 |

---

## 一句话总结

> 把 [[DINO]] token 的帧间残差压缩成 64 维左右的紧凑潜变量「残差潜在动作（RLA）」，再用 [[Flow Matching|流匹配]] 在这个低维空间里建[[世界模型]]，预测又快又准，还衍生出"从无动作视频学策略"和"在世界模型里跑视觉 RL"两个应用。

---

## 核心贡献

1. **残差潜在动作（[[残差潜在动作|RLA]]）**: 用一个轻量自编码器把 $s_{t+h}-s_t$ 这个 DINO token 残差编码成紧凑潜变量 $z$（$|z|\approx 64$）。RLA 具有三个涌现性质——**预测充分性**（单次前向解码即可还原 $s_{t+h}$）、**可泛化性**（在有限 ManiSkill 数据上训练即可迁移到新机器人/新交互）、**时间拓扑性**（噪声与 RLA 之间线性插值能得到时间上的中间帧）。
2. **RLA 世界模型（[[RLA-WM]]）**: 用[[Flow Matching|流匹配]]在紧凑 RLA 空间里生成 $z$，而不是直接回归高维 DINO token $s_{t+h}$。规避了直接回归导致的模糊/坍塌问题，FLOPs 介于直接回归（[[DINO-WM]]）与视频扩散（[[Vid2World]]）之间，但预测质量全面领先。
3. **两个下游应用**: (a) **极简世界动作模型（WAM）**——给 [[ResNet|ResNet-18]] BC 策略加一个线性头预测 RLA，从而能利用 95% 没有动作标签的视频；(b) **世界模型内视觉 RL（WMRL）**——用 [[LoRA]] 适配预训练 BC 策略，提出"视频对齐奖励（VAR）"，完全在 RLA-WM 里用 [[PPO]] 训练。

---

## 问题背景

### 要解决的问题

如何用离线视频学一个**视觉特征空间的世界模型** $f_{dyn}: (s_t, a_{t:t+h}) \mapsto s_{t+h}$，既要预测准确（物理保真、不幻觉），又要计算高效（能支撑规划/RL 的大量 rollout）。

### 现有方法的局限

- **像素空间预测**（视频扩散，如 [[Vid2World]]）：视觉锐利但容易**幻觉**、物理状态偏离现实，且推理昂贵。
- **直接回归 DINO token**（[[DINO-WM]] 家族、[[V-JEPA]]）：把高维特征当回归目标，在复杂 3D 任务里产生**模糊和表征坍塌**——DINO token 维度高达约 100 万维（512×512 图像），比 Stable Diffusion VAE 潜变量（约 1.6 万维）还高一个量级，是典型的"维度灾难"。
- **在 DINO token 空间直接做扩散/流匹配**（RAE、FM-WM）：维度太高导致生成质量差、FLOPs 巨大。
- **已有潜在动作方法**（[[AdaWorld]]、[[UniVLA]]）：要么作为模仿学习的代理标签、要么作为视频扩散的条件，潜变量本身**不具备预测充分性**——从 $z$ 重建 $s_{t+h}$ 严重模糊。

### 本文的动机

与其预测高维特征本身，不如学一个**更低维的"转移表征"**：DINO token 在相邻帧之间变化很小（局部），残差 $s_{t+h}-s_t$ 的有效信息量远小于 $s_{t+h}$，可以压成紧凑潜变量。这样世界模型只需要在低维空间生成，再用一个轻量解码器一步还原特征，兼顾准确与高效。

---

## 方法详解

### 模型架构

RLA-WM 由三部分组成（见 Figure 1、2）：

- **输入**: 当前帧的 [[DINO]] patch token $s_t \in \mathbb{R}^{L \times C}$（$L = HW/P^2$），机器人动作块 $a_{t:t+h}$（padding 到最大 horizon 15）
- **RLA 自编码器**: 12 层[[自注意力]]、16 头、通道 1024；编码器 $f_{enc}$ 把残差 $s_{t+h}-s_t$ 压成 RLA $z$（32 query × 64 dim = 2048，论文示意图用到 $|z|=64$ 仍能高保真重建），解码器 $f_{dec}$ 从 $(z, s_t)$ 单次前向还原 $\hat{s}_{t+h}$
- **条件网络**: 8 层注意力，把 $s_t$ 与 MLP 嵌入的动作、外加 32 个可学习 query 拼接，经[[自注意力]]产生固定的 condition tokens
- **流网络**: 8 层注意力，64 个 token（condition + 噪声潜变量），预测速度场 $\hat{v}$，把高斯噪声 $z_0=\epsilon$ 迭代变换为 RLA $z_1$
- **解码出 RGB**（仅用于可视化/RL）: 一个预训练 [[UNet]] 把 DINO token 解码回像素

### 核心模块

#### 模块 1: RLA 自编码器（[[残差潜在动作]]）

**设计动机**: 让世界模型在低维空间工作，同时保留"从 $z$ 一步还原未来特征"的能力。

**具体实现**:
- 用任务无关的离线视频对 $(s_t, s_{t+h})$ 训练；编码器吃残差 $s_{t+h}-s_t$，解码器吃 $(z, s_t)$
- 损失 = L1 + MSE（两者权重各 1.0），无需任何对比/正则项
- 涌现性质 A —— **预测充分性**：即便 $|z|$ 小到 64，$f_{dec}(z, s_t)$ 也能高保真还原 1024×1024 维的 $s_{t+h}$，远好于 [[AdaWorld]]（$|z|=2048$）和 [[UniVLA]]（$|z|=256$）（Figure A1）
- 涌现性质 B —— **时间拓扑性**：把 $z$ 归一化为 $\bar z$，则 $(\epsilon+\bar z)/2$ 解码后近似 $s_{t+h/2}$（Figure A2）
- 涌现性质 C —— **跨本体泛化**：Panda → XArm+Robotiq 这种训练时没见过的交互，重建仍然清晰（Figure A3）

#### 模块 2: RLA 世界模型 —— 在 RLA 空间做[[Flow Matching|流匹配]]

**设计动机**: RLA 是连续低维表征，自然适合用流匹配生成；条件固定后只在 64 维空间迭代，比在 DINO token 空间做扩散便宜得多。

**具体实现**:
- 条件 = 动作嵌入 + $s_t$ + 可学习 query，经条件网络得到固定 condition tokens
- 训练：采 $\tau$、构造噪声潜变量 $z_\tau = \tau z + (1-\tau)\epsilon$，流网络预测 $\hat v$，用 MSE 对齐真值速度 $v^*=z-\epsilon$ —— **不需要任何 feature/image 重建损失**
- 推理：从 $z_0=\epsilon$ 出发跑 30 步 Euler ODE 得 $z_1$，再 $f_{dec}(z_1, s_t)\to\hat s_{t+h}$

#### 模块 3: 极简世界动作模型（[[World Action Model|WAM]]，从无动作视频学策略）

**设计动机**: 大量演示视频没有本体状态/动作标签；与其强迫策略耦合 DINO 或视频生成 backbone，不如让一个普通 BC 策略**额外预测 RLA**作为辅助监督。

**具体实现**（Figure 4）:
- [[ResNet|ResNet-18]] BC + 两个线性头：一个出动作 $a$，一个出 RLA $\hat z$
- RLA 目标由从任务无关视频学好的 $f_{enc}$ 从 $(s_t, s_{t+h})$ 抽取
- 训练时每个 batch 从有标注（5%）和无标注（95%）视频等量采样；无标注样本只回传 RLA 头的梯度

#### 模块 4: 世界模型内视觉 RL（WMRL + 视频对齐奖励 VAR）

**设计动机**: 在真机/仿真里跑 RL 昂贵；既然 RLA-WM 能可靠预测，就让策略**完全在世界模型里交互**。

**具体实现**（Figure 5）:
- 策略 = 预训练 BC-ResNet + [[LoRA]] 适配器 + 残差动作头（输出 delta 动作和高斯 log std）
- rollout：策略出动作块 $a_{t:t+h}$ → RLA-WM 预测 $\hat s_{t+h}$ → UNet 解码 RGB 给下一步；初始 $\hat s_0$ 从随机抽的离线演示首帧 $s_0$ 重置
- 奖励 —— **视频对齐奖励（VAR）**：因为神经 rollout 与参考视频时间同步，VAR 直接定义为 $\hat s_t$ 与参考 $s_t$（或终态 $s_T$）的负 DINO L1 距离
- 优化：[[PPO]]（$\gamma=0.9$，GAE $\lambda=0.95$），300 步 rollout，112 个向量化环境，单次试验 3 小时

---

## 关键公式

### 公式1: [[残差潜在动作|RLA 编解码]]

$$
z = f_{enc}(s_{t+h} - s_t), \qquad \hat{s}_{t+h} = f_{dec}(z,\, s_t)
$$

**含义**: 编码器把帧间 DINO token 残差压成紧凑潜变量 $z$；解码器结合当前帧 $s_t$ 单次前向还原未来特征。

**符号说明**:
- $s_t, s_{t+h} \in \mathbb{R}^{L\times C}$: 当前/未来帧的 [[DINO]] patch token，$L=HW/P^2$
- $z$: 残差潜在动作，$|z|=2048$（32 query × 64 dim），示意图中 $|z|=64$ 亦可
- $h$: 时间跨度（action horizon，最大 15）

### 公式2: RLA 自编码器重建损失

$$
\mathcal{L}_{AE} = \lVert s_{t+h} - \hat{s}_{t+h} \rVert_1 + \lVert s_{t+h} - \hat{s}_{t+h} \rVert_2^2
$$

**含义**: L1 + MSE 联合监督特征重建，无对比/正则项；权重各 1.0。

**符号说明**:
- $\lVert\cdot\rVert_1$: L1 距离（保细节、抗模糊）
- $\lVert\cdot\rVert_2^2$: MSE（稳定优化）

### 公式3: [[Flow Matching|流匹配]] 噪声插值与速度目标

$$
z_\tau = \tau z + (1-\tau)\epsilon,\quad \epsilon \sim \mathcal{N}(0, I),\qquad v^{*} = z - \epsilon
$$

**含义**: 在噪声 $\epsilon$ 与目标 RLA $z$ 之间线性插值构造训练样本，真值速度场是常向量 $z-\epsilon$。

**符号说明**:
- $\tau \in [0,1]$: 流匹配时间步
- $\epsilon$: 标准高斯噪声，也是 ODE 起点 $z_0$
- $v^*$: 监督流网络的真值速度

### 公式4: 流网络训练损失

$$
\mathcal{L}_{WM} = \mathbb{E}_{\tau,\,\epsilon,\,(s_t, a_{t:t+h}, z)}\;\bigl\lVert \hat{v}\bigl(z_\tau, \tau \mid c(s_t, a_{t:t+h})\bigr) - v^{*} \bigr\rVert_2^2
$$

**含义**: 用 MSE 让流网络在固定条件 $c$ 下预测速度场——整个世界模型训练**不含 feature/image 重建损失**。

**符号说明**:
- $\hat v$: 流网络预测的速度
- $c(s_t, a_{t:t+h})$: 条件网络输出的固定 condition tokens

### 公式5: 推理 ODE 积分（Euler）

$$
z_{\tau+\Delta\tau} = z_\tau + \Delta\tau\, \hat{v}\bigl(z_\tau, \tau \mid c\bigr), \qquad \tau: 0 \to 1
$$

**含义**: 从高斯噪声 $z_0=\epsilon$ 出发，30 步 Euler 迭代得到 RLA $z_1$，再 $f_{dec}(z_1, s_t)\to\hat s_{t+h}$。

**符号说明**:
- $\Delta\tau = 1/30$: 步长（30 步 Euler）
- $z_1$: 最终 RLA，喂给解码器

### 公式6: 视频对齐奖励（VAR）

$$
r_t = -\lVert \hat{s}_t - s_t \rVert_1, \qquad r_T = -\lVert \hat{s}_T - s_T \rVert_1
$$

**含义**: 因为世界模型 rollout 与参考演示视频时间同步，奖励直接是预测特征与参考帧 DINO token 的负 L1 距离（终态用 $s_T$）。

**符号说明**:
- $\hat s_t$: RLA-WM 在第 $t$ 步预测的 DINO token
- $s_t$: 同一时刻参考演示视频的 DINO token
- $T$: episode 终止步

### 公式7: 时间拓扑性（RLA 插值）

$$
\bar{z} = (z - \mu)/\sigma, \qquad f_{dec}\bigl((\epsilon + \bar{z})/2 \cdot \sigma + \mu,\; s_t\bigr) \approx s_{t+h/2}
$$

**含义**: 把 RLA 归一化后与噪声做线性插值再反归一化解码，得到时间上的中间帧——RLA 空间内蕴含时间进度结构（Figure A2）。

**符号说明**:
- $\mu, \sigma$: 从数据估计的 RLA 均值/标准差
- $s_{t+h/2}$: 时间在 $t$ 与 $t+h$ 中点的特征

---

## 关键图表

> 图片来自 arXiv HTML（https://arxiv.org/html/2605.07079v1/）。

### Figure 1: Overview / 框架总览

![Figure 1a](https://arxiv.org/html/2605.07079v1/x1.png)
![Figure 1b](https://arxiv.org/html/2605.07079v1/x2.png)
![Figure 1c](https://arxiv.org/html/2605.07079v1/x3.png)

**说明**: 引入 [[残差潜在动作|RLA]]：把 DINO token 残差 $s_{t+h}-s_t$ 压成紧凑潜变量 $z$；RLA 是预测充分、可泛化、含时间进度的。[[RLA-WM]] 从离线视频学习，预测 $z$ 而非直接预测 $s_{t+h}$，准确且比 SOTA 特征/视频扩散世界模型更高效。衍生两个应用：从无动作视频学策略、完全在 RLA-WM 内做视觉 RL。

### Figure 2: RLA World Model 架构

![Figure 2](https://arxiv.org/html/2605.07079v1/x4.png)

**说明**: 动作 $a_{t:t+h}$ 经 MLP 嵌入，与 DINO token $s_t$ 和可学习 query 拼接，过[[自注意力]]层得到 condition tokens；[[Flow Matching|流匹配]]阶段条件固定，与噪声潜变量 $z_\tau$（起点 $z_0=\epsilon$）拼接，流网络预测速度 $\hat v$ 迭代得 $z_1$；最后 $f_{dec}$ 从 $(z_1, s_t)$ 解码 $\hat s_{t+h}$。训练只用对真值速度 $v^*=z-\epsilon$ 的 MSE，无需任何特征/图像重建损失。

### Figure 3: RLA-WM 定性对比

![Figure 3](https://arxiv.org/html/2605.07079v1/x5.png)

**说明**: 给 $t=0$ 输入帧，RLA-WM 预测的未来帧视觉质量与物理保真都接近真值。[[DINO-WM]] 在 Push-T 长 horizon 上越来越模糊、绳子状态不一致；在 DINO token 空间直接做扩散/流匹配（RAE、FM-WM）结果更差；[[Vid2World]] 帧很锐利但**幻觉严重**、物理状态偏离现实。

### Figure 4: 从无动作视频学策略（极简 WAM）

![Figure 4](https://arxiv.org/html/2605.07079v1/x6.png)

**说明**: [[ResNet]] BC 加一个线性层预测 RLA $\hat z$；RLA 目标由从任务无关视频学好的 $f_{enc}$ 从 $(s_t, s_{t+h})$ 抽取。把 BC 策略变成极简[[World Action Model|世界动作模型]]，能利用没有本体状态/动作标签的视频，且不强迫策略耦合 DINO 或视频生成 backbone。

### Figure 5: 世界模型内视觉 RL（WMRL + VAR）

![Figure 5](https://arxiv.org/html/2605.07079v1/x7.png)

**说明**: 预训练 ResNet BC 策略 + [[LoRA]] 适配器 + 残差动作头（出 delta 动作和高斯 log std）；策略出动作块给 RLA-WM 预测 $\hat s_{t+h}$，预训练 UNet 解码成 RGB 供下一步；$\hat s_0$ 从随机离线演示首帧 $s_0$ 重置。奖励用**视频对齐奖励（VAR）**——神经 rollout 与参考视频时间同步，VAR 即 $\hat s_t$ 与 $s_t$（或终态 $s_T$）的负 DINO L1。策略用 [[PPO]] 在完全位于 RLA-WM 内部的 rollout 上优化。

### Figure 6: WMRL 性能分布（50 episode 评测）

![Figure 6](https://arxiv.org/html/2605.07079v1/x8.png)

**说明**: 选最好的 BC 模型（BC*），在 15 个独立随机种子上跑 WMRL；每个种子最佳 checkpoint 的成功率画成一个点，整体最优 checkpoint 标为 RL*。这里成功率在 50 episode（标准种子 42-91）上评测。

### Figure A1: RLA 重建质量对比

![Figure A1](https://arxiv.org/html/2605.07079v1/x9.png)

**说明**: 比较从 $(s_t, s_{t+h})$ 编码出 $z$、再由 $f_{dec}(z, s_t)$ 解码 $\hat s_{t+h}$ 的质量。RLA 即使 $|z|=64$ 也能精确重建（鉴于 $s_{t+h}$ 是 1024×1024 维，相当惊人）；[[AdaWorld]]（$|z|=2048$）、[[UniVLA]]（$|z|=256$）则严重模糊失真。说明 RLA 捕获了动力学的预测充分信息，单次前向即可解码未来 token。

### Figure A2: RLA 时间拓扑性

![Figure A2](https://arxiv.org/html/2605.07079v1/x10.png)

**说明**: 把 RLA $z$ 归一化为 $\bar z$，在高斯噪声 $\epsilon$ 与 $\bar z$ 之间插值再反归一化解码——$(\epsilon+\bar z)/2$ 解码后近似 $s_{t+h/2}$，说明 RLA 潜空间内蕴含时间进度。直接在 DINO token 间插值效果差。

### Figure A3: RLA 跨任务/跨本体泛化

![Figure A3](https://arxiv.org/html/2605.07079v1/x11.png)

**说明**: 把预训练 RLA 自编码器用到完全没见过的设置（Panda 换成带 Robotiq 夹爪的 XArm 做 Pull Cube with Tool），重建仍然高保真。这种跨本体泛化只靠有限的 ManiSkill 数据训练即可获得，无需大规模视频预训练。

### Figure A4: 任务与数据集总览

![Figure A4a](https://arxiv.org/html/2605.07079v1/x12.png)
![Figure A4b](https://arxiv.org/html/2605.07079v1/x13.png)

**说明**: ManiSkill 五个任务、三个机器人——Panda：Pull Cube、Pull Cube with Tool；UR10（圆柱末端）：Roll Ball、Push T；XArm（Robotiq 夹爪）：Poke Cube。IWS（ALOHA 双臂）：Rope Routing、Box Packing、Push T。

### Figure A5: WMRL 性能分布（1500 episode 评测）

![Figure A5](https://arxiv.org/html/2605.07079v1/x14.png)

**说明**: 补充 Table 3，展示 15 个独立种子的性能分布；每个点是该种子最佳 checkpoint 的成功率，在每种子 1500 episode（种子 1-1500）上评测。

### Figure A6–A10: RLA-WM 更多定性对比

![Figure A6](https://arxiv.org/html/2605.07079v1/x15.png)
![Figure A7](https://arxiv.org/html/2605.07079v1/x16.png)
![Figure A8](https://arxiv.org/html/2605.07079v1/x17.png)
![Figure A9](https://arxiv.org/html/2605.07079v1/x18.png)
![Figure A10](https://arxiv.org/html/2605.07079v1/x19.png)

**说明**: 附录中 RLA-WM 与各基线的额外定性预测对比，结论与 Figure 3 一致。

### Table 1: 预测质量对比（ManiSkill / IWS）

| Model | ManiSkill LPIPS ↓ | ManiSkill SSIM ↑ | ManiSkill DINO L1 ↓ | IWS LPIPS ↓ | IWS SSIM ↑ | IWS DINO L1 ↓ | FLOPs ↓ |
|-------|------:|------:|------:|------:|------:|------:|------:|
| [[DINO-WM]] | 0.156 | 0.865 | 0.078 | 0.223 | 0.825 | 0.058 | 2.1T |
| [[RAE]] | 0.324 | 0.717 | 0.143 | 0.550 | 0.625 | 0.159 | 14.3T |
| FM-WM | 0.127 | 0.890 | 0.063 | 0.360 | 0.741 | 0.119 | 14.3T |
| [[Vid2World]] | 0.199 | 0.705 | 0.084 | 0.388 | 0.710 | 0.139 | 1.1P |
| **RLA-WM** | **0.071** | **0.931** | **0.030** | **0.196** | **0.847** | **0.053** | **3.5T** |

**说明**: RLA-WM 在所有指标上全面领先；FLOPs 介于直接回归（[[DINO-WM]] 2.1T）和 DINO 空间扩散（14.3T）/视频扩散（[[Vid2World]] 1.1 PFLOPs）之间。直接在 DINO token 空间做扩散（RAE）反而最差，印证了"维度灾难"。

### Table 2: 从无动作视频学策略（成功率）

| Method | PushT | Roll | Pull | Pull Tool | Poke | Avg SR ↑ |
|--------|------:|------:|------:|------:|------:|------:|
| BC-ResNet | 3.6% | 42.0% | 33.6% | 7.6% | 49.2% | 27.2% |
| DINO CLS | 7.6% | 39.6% | 40.4% | 4.4% | 44.8% | 27.4% |
| [[UniVLA]] | 6.0% | 37.6% | 42.8% | 7.2% | 50.0% | 28.7% |
| [[AdaWorld]] | 9.2% | 38.4% | 48.4% | 10.8% | 61.6% | 33.7% |
| **RLA** | **15.2%** | **43.8%** | 43.6% | **12.0%** | **63.6%** | **35.6%** |

**说明**: 95% 视频无动作标签时，给 BC 策略加 RLA 辅助头，平均成功率比纯 BC +8.4%、比 [[AdaWorld]] +1.9%；在最难的 PushT 上提升最明显（3.6% → 15.2%）。

### Table 3: 世界模型内视觉 RL（大规模 1500 seed 评测，成功率）

| Method | XArm: Poke Cube | UR10e: Roll Ball | UR10e: PushT | Panda: Pull Cube | Panda: Pull Cube Tool | Avg |
|------|------:|------:|------:|------:|------:|------:|
| BC-ResNet | 89.9% | 65.5% | 17.2% | **84.5%** | **41.1%** | 59.6% |
| **RLA-WMRL** | **95.9%** | **73.1%** | **20.7%** | 74.1% | 39.9% | **60.7%** |

**说明**: 列分组——XArm（Poke Cube）、UR10e（Roll Ball、PushT）、Panda（Pull Cube、Pull Cube Tool）。WMRL 在 XArm 和 UR10e 上有统计显著提升（Poke Cube +6%、Roll Ball +7.6%、PushT +3.5%），五任务平均 +1.1%（不用任何额外数据、只用额外计算）；但在 Panda 的两个 Pull 任务上略降（Pull Cube 84.5%→74.1%、Pull Cube Tool 41.1%→39.9%），附录 A.3 归因于 Panda 运动学复杂、动作多样性有限。论文将此视为世界模型 RL 走向严格评估标准的"初步但扎实的一步"。

### 实现细节速查（Section A.2）

| 组件 | 配置 |
|------|------|
| RLA 自编码器 | 12 层自注意力 / 16 头 / 通道 1024；输入 512×512；batch 128；lr 1e-4；100k 步；损失 L1+MSE（各 1.0）；$\lvert z\rvert=2048$（32 query × 64） |
| 条件网络 | 8 层注意力，32 query token |
| 流网络 | 8 层注意力，64 token；最大 action horizon 15；batch 64；lr 1e-4；100k 步；推理 30 步 Euler；训练 3 天 / 4× A6000 |
| WMRL | 所有 linear/conv 层挂 LoRA；PPO（$\gamma=0.9$, GAE $\lambda=0.95$）；lr 1e-4；112 向量化环境；单次试验 3 小时 |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| ManiSkill | 3 机器人（Panda / XArm+Robotiq / UR10e）、5 任务，每任务 1500 episode | 单臂操作仿真，含跨本体 | 预测质量、WAM、WMRL |
| IWS（ALOHA） | 3 任务（Rope Routing / Box Packing / Push T），每任务 600+ 演示 | 双臂、含可变形物体（绳子） | 预测质量 |

评测指标：[[感知图像相似度|LPIPS]]、[[结构相似性|SSIM]]、对 DINO token 的 L1 距离、FLOPs。

### 实现细节

- **Backbone**: 视觉编码器用预训练 [[DINO]]（patch token）；RLA-WM 全为[[自注意力]] Transformer；RGB 解码用预训练 [[UNet]]；下游策略用 [[ResNet|ResNet-18]]
- **优化器/学习率**: Adam，lr $10^{-4}$（自编码器、流网络、WMRL 均如此）
- **Batch Size**: 自编码器 128，流网络 64
- **训练步数**: 自编码器 100k 步、流网络 100k 步
- **硬件**: 4× NVIDIA A6000（世界模型训练约 3 天）；WMRL 单次试验约 3 小时

### 可视化结果

RLA-WM 长 horizon 预测保持物理保真，不像 [[DINO-WM]] 那样越来越模糊、也不像 [[Vid2World]] 那样视觉锐利但幻觉漂移；RLA 自编码器在 $|z|=64$ 下仍能高保真重建，且能零样本迁移到训练时没见过的机器人本体（Figure A1/A3）。

---

## 批判性思考

### 优点

1. **抓住了关键 insight**：DINO token 的帧间残差信息量远小于 token 本身，"预测转移而非预测特征"在维度上是数量级的节省，且实验上同时拿到更准 + 更快。
2. **RLA 的三个涌现性质有说服力**：预测充分性（$|z|=64$ 重建 1024×1024 维特征）、时间拓扑性（噪声-RLA 插值得中间帧）、跨本体泛化（不靠大规模视频预训练），都做了直接可视化验证。
3. **训练目标极简**：世界模型只用对真值速度的 MSE，不需要 feature/image 重建损失，工程上干净。
4. **两个下游应用是真实用例**：从 95% 无动作标签视频学策略、完全在世界模型里跑 PPO（VAR 利用 rollout 与参考视频时间同步这一巧思），都展示了 RLA-WM 的实用价值。

### 局限性

1. **背景运动浪费表征容量**：workspace 随机化带来的任务无关视觉变化会占用 RLA 容量；作者提出"3D 视角无关 RLA"作为未来方向。
2. **记忆/部分可观**：单对帧无法编码遮挡/物体消失等情形，需要扩展到多帧条件。
3. **只预测视觉状态**：当前模型不预测本体感知（proprioception），未来应加 proprioceptive 输出头。
4. **规模刻意小**：只在 ManiSkill/IWS 上评测以隔离方法收益与数据规模收益，互联网规模适配尚未验证。
5. **Panda 机器人 WMRL 失败**：附录归因于运动学/视角/数据多样性，说明方法对某些本体仍不稳健；Table 3 中 Pull Cube/Pull Tool 上 WMRL 也略低于 BC。

### 潜在改进方向

1. **3D / 视角无关 RLA**：用几何对齐的特征作为残差基底，剔除背景/视角扰动。
2. **多帧条件 + 记忆**：把条件从单帧 $s_t$ 扩展为短历史，处理遮挡和长程依赖。
3. **联合预测本体状态**：在 RLA 解码端加 proprio head，让世界模型可用于状态空间规划。
4. **更强的奖励设计**：VAR 仅基于特征 L1，可探索结合任务进度/对比奖励减少对参考视频的依赖。

### 可复现性评估

- [x] 代码开源（项目主页 https://mlzxy.github.io/rla-wm 提供）
- [ ] 预训练模型（未明确）
- [x] 训练细节完整（附录 A.2 给了完整超参与硬件）
- [x] 数据集可获取（ManiSkill 公开；IWS 需确认）

---

## 关联笔记

### 基于

- [[DINO]] / [[DINOv2]]: RLA 直接构建在 DINO patch token 残差之上
- [[Flow Matching]]: RLA-WM 在紧凑潜空间用流匹配生成 $z$
- [[世界模型]] / [[World Model]]: 本文是视觉特征空间世界模型的一种新范式

### 对比

- [[DINO-WM]]: 直接回归 DINO token 的基线，长 horizon 模糊
- [[RAE]]: 在 DINO token 空间直接做扩散的基线（最差）
- [[Vid2World]]: 像素空间视频扩散世界模型，锐利但幻觉
- [[AdaWorld]] / [[UniVLA]]: 已有潜在动作方法，潜变量不具预测充分性

### 方法相关

- [[残差潜在动作]]: 本文核心表征
- [[RLA-WM]]: 本文核心世界模型（即本笔记主体）
- [[World Action Model]]: 下游应用——从无动作视频学策略
- [[LoRA]]: WMRL 中适配预训练 BC 策略
- [[PPO]]: WMRL 的 RL 算法
- [[UNet]]: DINO token → RGB 的解码器
- [[ResNet]]: 下游 BC 策略 backbone
- [[自注意力]]: RLA 自编码器与流网络的基础结构

### 硬件/数据相关

- [[ManiSkill]]: 主要评测仿真平台（3 机器人 5 任务）
- [[Push-T]]: 评测任务之一（ManiSkill 与 IWS 都有）
- [[感知图像相似度|LPIPS]] / [[结构相似性|SSIM]]: 预测质量指标

---

## 速查卡片

> [!summary] Learning Visual Feature-Based World Models via Residual Latent Action
> - **核心**: 把 DINO token 帧间残差压成 ~64 维的"残差潜在动作（RLA）"，世界模型只在这个低维空间用流匹配生成
> - **方法**: RLA 自编码器（L1+MSE，单次前向解码）+ RLA-WM（条件网络 + 流匹配，只用速度 MSE）；衍生极简 WAM（从无动作视频学策略）和 WMRL（用 VAR 在世界模型里跑 PPO）
> - **结果**: ManiSkill/IWS 预测质量全面领先（LPIPS 0.071 vs DINO-WM 0.156），FLOPs 3.5T 介于回归与扩散之间；无动作视频学策略平均成功率 +8.4%，WMRL 平均 +1.1%（Panda 失败）
> - **代码**: https://mlzxy.github.io/rla-wm

---

*笔记创建时间: 2026-05-12*
