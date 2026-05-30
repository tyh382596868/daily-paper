---
title: "minWM: A Full-Stack Open-Source Framework for Real-Time Interactive Video World Models"
method_name: "minWM"
authors: [Min Zhao, Hongzhou Zhu, Bokai Yan, Zihan Zhou, Yimin Chen, Wenqiang Sun, Kaiwen Zheng, Guande He, Xiao Yang, Chongxuan Li, Fan Bao, Jun Zhu]
year: 2026
venue: arXiv
tags: [world-model, video-diffusion, interactive-world-model, camera-control, autoregressive, diffusion-distillation, real-time]
zotero_collection: 1-生成模型
image_source: online
arxiv_html: https://arxiv.org/html/2605.30263v1
created: 2026-05-30
---

# 论文笔记：minWM: A Full-Stack Open-Source Framework for Real-Time Interactive Video World Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 生数科技（Shengshu AI）/ 清华大学 |
| 日期 | May 2026 (arXiv:2605.30263v1, 提交于 2026-05-28) |
| 项目主页 | https://github.com/shengshu-ai/minWM |
| 代码库 | https://github.com/shengshu-ai/minWM |
| 对比基线 | [[Wan2.1]]-T2V-1.3B、HY1.5-TI2V-8B（HunyuanVideo 1.5）|
| 链接 | [arXiv](https://arxiv.org/abs/2605.30263) / [HTML](https://arxiv.org/html/2605.30263v1) / [Code](https://github.com/shengshu-ai/minWM) |

---

## 一句话总结

> 一套端到端开源流水线，把双向 [[视频扩散模型|T2V/TI2V]] 基座转成**相机可控、少步自回归**的[[交互式世界模型]]，首帧延迟降低 200× 以上。

---

## 核心贡献

1. **全栈开源 pipeline**: 覆盖从相机条件数据构造、[[相机可控微调]]、[[自回归扩散训练]]、[[少步蒸馏]] 到流式低延迟推理的完整链路，代码与权重全部公开。
2. **架构无关**: 同一套配方分别成功落地到 1.3B 的 [[Wan2.1]]-T2V 和 8B 的 HY1.5-TI2V 两种主流视频基座，证明方法可迁移。
3. **首帧延迟 223–236× 加速**: 通过三阶段 [[Causal Forcing]] / [[Causal Forcing++]] 蒸馏，在保持相机可控性的前提下，把多步双向扩散打成 4-step 因果自回归生成。
4. **实用 Ablation**: 给出训练数据来源（[[SpatialVID-HQ|SpatialVid]] 失败 vs [[DL3DV]] 重建 + [[WorldPlay]] 合成成功）、训练步数、批量大小三个关键超参的失败/成功阈值，方便复现者避坑。

---

## 问题背景

### 要解决的问题

把现有强力但**离线、双向、慢**的 [[视频扩散模型]] 改造成能够：

- **响应相机动作**（用户给定 trajectory 即时改变视角）；
- **因果滚动**（不依赖未来帧，可无限续帧）；
- **低延迟**（达到实时交互所需的 sub-second 首帧），

这样的[[交互式世界模型]]。

### 现有方法的局限

| 现状 | 问题 |
|------|------|
| 双向 [[文本驱动视频生成]] | 强生成质量但单 batch 推理动辄分钟级，不能交互 |
| 普通 AR 视频生成 | 因果但仍需多步去噪，延迟不达标 |
| 既有蒸馏（如 DMD/CD） | 只针对非条件、非 AR 模型，没人系统化做"相机条件 + 因果 + 少步" |
| 训练数据 | [[SpatialVID-HQ|SpatialVid]] 这种"用感知模型反推相机"的数据集，pose 含噪，直接训会失败 |

### 本文的动机

作者认为，把现有零散技术（[[PRoPE 相机嵌入]]、[[Causal Forcing]]、[[DMD]]、[[Consistency Distillation]]）**系统化串成一条 pipeline**，比再发明一种新算法更有工程价值。重点不在算法本身的新颖性，而在：

1. 把"基础模型 → 相机条件 → 自回归 → 少步 → 流式推理"这条工程线打通；
2. 给出对实际训练成败有决定性影响的数据 / 超参实践经验。

---

## 方法详解

### 总体流程

minWM 的 pipeline 由四个串行模块组成，每个模块接收前一模块的产物：

- **数据构造**: 利用 [[DL3DV]] 做 3D 重建 + 按指定轨迹重渲染，或用 [[OpenVid]] 图像 + [[WorldPlay]] 生成"伪真值相机轨迹"的视频；
- **相机可控微调**: 在双向 [[DiT]] 基座（[[Wan2.1]] 或 HY1.5）上注入 [[PRoPE 相机嵌入]]，得到 camera-controllable bidirectional model；
- **AR 扩散训练（Stage 1）**: 用 [[Teacher Forcing]] + 因果注意力掩码，把双向模型转成多步自回归；
- **少步蒸馏（Stage 2–3）**: 两条可选支线
  - Stage 2A: [[Causal ODE Initialization]]（需离线 ODE 数据）
  - Stage 2B: [[Causal Consistency Distillation]]（在线、省存储）
  - Stage 3: [[Asymmetric DMD]] 与双向 teacher 做分布对齐；
- **流式推理**: chunk size = 4 latent frames，4 步去噪，单 A800 上首帧延迟 < 1.2 s。

### 模块 1: 相机条件注入（PRoPE）

**设计动机**: 相机参数是 4×4 的投影矩阵，直接拼接到 token 上既丢失几何信息又破坏 [[RoPE]] 的相对位置假设。论文采用 **[[PRoPE 相机嵌入]]**（Projective RoPE），用提升的投影矩阵作为 self-attention 中的旋转/缩放算子。

**具体实现**:

- 每帧的相机表示为 lifted projective matrix $\widetilde{P}_i$（见 [[#公式 1: PRoPE 提升投影矩阵|公式1]]）；
- 在 self-attention 中以 block-diagonal 形式作用到 $Q,K,V$ 上（见 [[#公式 2: PRoPE Attention|公式2]]）；
- 保持原始 RoPE 结构兼容，**仅引入相机条件，不改变 attention 形状**。

### 模块 2: AR 化 — Stage 1 教师强迫

把双向 [[DiT]] 用因果注意力掩码 + [[Teacher Forcing]] 训练，使模型在每个 latent 时间步上只能看到过去 chunk。此阶段模型已具备自回归能力，但仍需多步采样（高延迟）。

### 模块 3: 少步蒸馏

两条互补路径（论文称为 [[Causal Forcing]] vs [[Causal Forcing++]]）：

#### Stage 2A — [[Causal ODE Initialization]]（Causal Forcing）

离线用教师模型跑出多步 ODE 轨迹作为监督，少步学生回归到 clean 帧。见 [[#公式 3: Causal ODE 回归损失|公式3]]。

- 优点：监督信号干净；
- 缺点：需要离线生成 + 存储大量 ODE 轨迹，存储 / 时间成本高。

#### Stage 2B — [[Causal Consistency Distillation]]（Causal Forcing++）

在线版本，不需要离线 ODE 数据，让学生在两个邻近 timestep 上做 endpoint 一致（见 [[#公式 4: Causal Consistency Distillation 损失|公式4]]）。

- 优点：无需离线数据；训练 pipeline 更简洁；
- 缺点：早期不稳定，需要 warm-up。

#### Stage 3 — [[Asymmetric DMD]]

对齐学生分布与双向教师的高质量分布（[[DMD]] 的 score-divergence 变体），见 [[#公式 5: Asymmetric DMD 梯度|公式5]]。

- "Asymmetric" 指 $s_\text{real}$（保留双向 teacher）与 $s_\text{fake}$（用学生本身 fine-tune 的 fake 网络）结构不对称；
- 该 stage 步数少（HY1.5 仅 500 步、Wan2.1 仅 200 步），属于"轻量对齐"。

### 模块 4: 相机可控蒸馏

关键 trick：**所有蒸馏阶段都把相机条件视为输入并冻结**。学生模型在自滚（self-rollout）时用自己生成的历史帧，但**相机条件来自外部用户指令**；两个 teacher（real / fake score 网络）均接收同样的相机条件。这样保证蒸馏后相机可控性不退化。

### 模块 5: 流式推理

- **Chunk 大小**: 4 个 latent frame；
- **蒸馏步数**: 4 步去噪；
- **硬件**: 单卡 A800；
- 推理时按 chunk 自回归滚动，KV cache 复用历史 chunk。

---

## 关键公式

### 公式 1: [[PRoPE 相机嵌入|PRoPE 提升投影矩阵]]

$$
\widetilde{P}_{i}=\begin{bmatrix}[K_{i}\;0]\,T_{i}^{cw}\\ e_{4}^{\top}\end{bmatrix}\in\mathbb{R}^{4\times 4},\qquad e_{4}=(0,0,0,1)^{\top}
$$

**含义**: 把第 $i$ 帧的相机内参 $K_i$ 和世界→相机外参 $T_i^{cw}$ 拼成 4×4 的 lifted projective matrix，作为 PRoPE 在 attention 中的旋转/缩放算子。

**符号说明**:

- $K_i \in \mathbb{R}^{3\times 3}$: 第 $i$ 帧的相机内参矩阵；
- $T_i^{cw} \in SE(3)$: 第 $i$ 帧的世界到相机的位姿（4×4 齐次变换）；
- $[K_i\;0]$: 把内参右侧补 0 列形成 3×4 投影矩阵；
- $e_4 = (0,0,0,1)^\top$: 补齐到 4×4 的最后一行；
- $\widetilde{P}_i \in \mathbb{R}^{4\times 4}$: 最终用于 attention 的投影矩阵。

### 公式 2: [[PRoPE 相机嵌入|PRoPE Attention]]

$$
\mathrm{Attn}_{\mathrm{PRoPE}}(Q,K,V)=D^{\mathrm{PRoPE}}\odot\mathrm{Attn}\!\left((D^{\mathrm{PRoPE}})^{\top}\odot Q,\,(D^{\mathrm{PRoPE}})^{-1}\odot K,\,(D^{\mathrm{PRoPE}})^{-1}\odot V\right)
$$

**含义**: 在标准 self-attention 上叠加由 $\widetilde{P}_i$ 构成的 block-diagonal 算子 $D^\text{PRoPE}$，使 attention 显式感知每个 token 所属帧的相机几何关系。

**符号说明**:

- $D^\text{PRoPE}$: 由 $\{\widetilde{P}_i\}$ 组装的 block-diagonal 算子；
- $\odot$: 此处表示 PRoPE 的张量级旋转/缩放作用（类似 [[RoPE]] 的相位旋转）；
- $(D^\text{PRoPE})^{-1}$ 作用到 $K,V$，$D^\text{PRoPE}$ 作用到 $Q$ 与输出，结构对称，保留 attention 不变性；
- 标准 Attn 内部仍是普通 softmax(Q K^T / √d) V。

### 公式 3: [[Causal ODE Initialization|Causal ODE 回归损失]]

$$
\theta^{*}=\arg\min_{\theta}\mathbb{E}_{\boldsymbol{x}_{\mathrm{gt}}^{<i},\,t,\,i,\,\boldsymbol{x}_{t}^{i}}\left[\,\|G_{\theta}(\boldsymbol{x}_{t}^{i},\boldsymbol{x}_{\mathrm{gt}}^{<i},t)-\boldsymbol{x}_{0}^{i}\|^{2}\,\right]
$$

**含义**: Stage 2A 的目标，学生网络 $G_\theta$ 把"含噪的当前帧 + 真值历史帧"映射回 clean 帧 $\boldsymbol{x}_0^i$，相当于把多步 ODE 解蒸馏成单/少步直接预测。

**符号说明**:

- $\boldsymbol{x}_\text{gt}^{<i}$: 历史真值帧（teacher forcing 给的 ground-truth chunk）；
- $\boldsymbol{x}_t^i$: 第 $i$ 帧在 timestep $t$ 的含噪状态；
- $\boldsymbol{x}_0^i$: 第 $i$ 帧的 clean 真值；
- $G_\theta$: 少步学生生成器；
- $t \sim \mathcal{U}$, $i$: 在 timestep 与 chunk index 上联合采样。

### 公式 4: [[Causal Consistency Distillation|Causal Consistency Distillation 损失]]

$$
\theta^{*}=\arg\min_{\theta}\mathbb{E}_{\boldsymbol{x}_{\text{gt}},\,\boldsymbol{\epsilon},\,t,\,i}\Big[w(t)\,d\big(G_{\theta}(\boldsymbol{x}_{t}^{i},\boldsymbol{x}_{\text{gt}}^{<i},t),\;G_{\theta^{-}}(\hat{\boldsymbol{x}}^{i}_{t-\Delta t},\boldsymbol{x}_{\text{gt}}^{<i},t-\Delta t)\big)\Big]
$$

**含义**: Stage 2B 的目标，学生在 timestep $t$ 与 $t-\Delta t$ 上的预测应一致（[[Consistency Distillation]] 的因果版），完全在线，无需离线 ODE 数据。

**符号说明**:

- $\theta^{-}$: 学生的 EMA 版本（target 网络）；
- $\hat{\boldsymbol{x}}^i_{t-\Delta t}$: 用 $G_{\theta^-}$ 一步从 $\boldsymbol{x}_t^i$ 推到 $t-\Delta t$ 的中间状态；
- $w(t)$: 时间步加权函数；
- $d(\cdot, \cdot)$: 距离度量（如 LPIPS 或 L2）；
- $\Delta t$: 一致性的步长间隔。

### 公式 5: [[Asymmetric DMD|Asymmetric DMD 梯度]]

$$
\nabla_{\theta}\mathbb{E}_{t}\big[D_{\mathrm{KL}}\big(p_{\theta,t}(\tilde{\boldsymbol{x}}_{t})\,\Vert\,p_{\text{data},t}(\tilde{\boldsymbol{x}}_{t})\big)\big]=-\mathbb{E}_{\tilde{\boldsymbol{x}},t,\tilde{\boldsymbol{x}}_{t}}\Big[\big(s_{\text{real}}(\tilde{\boldsymbol{x}}_{t},t)-s_{\text{fake}}(\tilde{\boldsymbol{x}}_{t},t)\big)\frac{\partial\tilde{\boldsymbol{x}}}{\partial\theta}\Big]
$$

**含义**: Stage 3 用 [[DMD]] 的 score-divergence 梯度，把学生分布 $p_\theta$ 拉向 ($\to$) 真实数据分布 $p_\text{data}$；"Asymmetric" 指 $s_\text{real}$ 来自冻结的双向 teacher，$s_\text{fake}$ 来自动态 fine-tune 的 fake score 网络，结构上不对称。

**符号说明**:

- $\tilde{\boldsymbol{x}}$: 学生生成的样本；
- $\tilde{\boldsymbol{x}}_t$: 对生成样本加 $t$ 步噪后的扰动状态；
- $s_\text{real}, s_\text{fake}$: 真实分布与"假"分布的 score 函数（$\nabla_x \log p$）；
- $\partial \tilde{\boldsymbol{x}}/\partial \theta$: 通过学生参数的可微采样路径；
- $D_\text{KL}$: 在每个 noise level $t$ 上的 KL 散度。

---

## 关键图表

### Figure 1: minWM Pipeline Overview / 总览

![Figure 1](https://arxiv.org/html/2605.30263v1/x1.png)

**说明**: minWM 的全栈流水线。从左到右依次为：数据构造（[[DL3DV]] 重建 / [[WorldPlay]] 合成）→ [[相机可控微调]]（[[PRoPE 相机嵌入]]）→ AR 训练（[[Teacher Forcing]] + 因果掩码）→ 少步蒸馏（[[Causal Forcing]] 三阶段）→ 流式低延迟推理。

### Figure 2: Camera-Controllable Generation Samples / 相机控制生成样例

![Figure 2](https://arxiv.org/html/2605.30263v1/x2.png)

**说明**: 蒸馏后 few-step AR 模型在**不同相机动作**（前进 / 旋转 / 平移）下的生成结果。证明蒸馏算法成功保留了 base model 的相机可控性，没有因为压缩到 4 步而退化。

### Figure 3: Training Data Ablation / 训练数据消融

![Figure 3a](https://arxiv.org/html/2605.30263v1/x3.png)

![Figure 3b](https://arxiv.org/html/2605.30263v1/x4.png)

![Figure 3c](https://arxiv.org/html/2605.30263v1/x5.png)

**说明**: 训练数据对相机可控性的影响。

- (a) 直接用 [[SpatialVID-HQ|SpatialVid]]（感知反推 pose）训练 → 控制失败；
- (b) 用 [[DL3DV]] 重建 + 按指定轨迹重渲染 → 控制成功；
- (c) 用 [[OpenVid]] 图像 + [[WorldPlay]] 合成视频 → 控制成功。

**核心结论**: **相机注释精确度 > 数据规模**。感知估计的伪标签 pose 即使量大也不如几何重建 / 规则合成的小数据。

### Figure 4: Training Steps Ablation (HY1.5) / 训练步数消融

![Figure 4a](https://arxiv.org/html/2605.30263v1/x6.png)

![Figure 4b](https://arxiv.org/html/2605.30263v1/x7.png)

![Figure 4c](https://arxiv.org/html/2605.30263v1/x8.png)

**说明**: HY1.5 上相机可控性的"出现"阶段。

- 1–2K steps: 模型无视相机条件；
- ~5K steps: 控制能力开始出现但不稳；
- 8K steps: 强且稳定的可控性。

→ **相机可控性是一种 emergent capability**，存在明确门槛。

### Figure 5: Batch Size Ablation (Wan2.1) / 批量大小消融

![Figure 5a](https://arxiv.org/html/2605.30263v1/x9.png)

![Figure 5b](https://arxiv.org/html/2605.30263v1/x10.png)

![Figure 5c](https://arxiv.org/html/2605.30263v1/x11.png)

**说明**: Wan2.1 上 batch size 对训练稳定性的影响。

- bs < 4: 通常学不到可控性；
- bs = 8: 大幅改善但不稳定；
- bs ≥ 16: 稳定训练，可控性高。

→ 相机条件训练**强依赖足够大的 batch** 来稳定梯度。

### Table 1: First-Frame Latency Comparison / 首帧延迟对比

| Base Model | 类型 | First-Frame Latency (s, A800) | Speedup |
|------------|------|-------------------------------|---------|
| HY1.5 | Multi-step bidirectional | 1.041 | 1.00× |
| HY1.5 | Multi-step AR | 1.014 | 9.52× |
| HY1.5 | Few-step AR (4 step) | 3.446 | 223.75× |
| Wan2.1 | Multi-step bidirectional | 269.055 | 1.00× |
| Wan2.1 | Multi-step AR | 28.651 | 9.39× |
| Wan2.1 | Few-step AR (4 step) | 1.137 | 236.64× |

**说明**: 首帧延迟在单 A800 上测量（不含 VAE 时间）。

- Wan2.1 上 few-step AR 从 269s 降到 1.137s，达到**亚秒级交互**；
- "Speedup" 列的标定方式与原始延迟列表面值不完全自洽（HY1.5 行的延迟随 speedup 反而上升），疑似论文 v1 表格存在标注错位，**建议复读 GitHub 公布的最新数据**；
- 核心结论是趋势：**bidirectional → AR → few-step AR 两阶段各贡献约 10× 加速**。

---

## 实验结果

### 数据集

| 数据集 | 用途 | 特点 |
|--------|------|------|
| [[SpatialVID-HQ\|SpatialVid]] | 失败对照 | 感知反推 pose，噪声大 → 训不出可控性 |
| [[DL3DV]] | 主训练 | 3D 场景重建 + 按指定轨迹重渲染，pose 精确 |
| [[OpenVid]] + [[WorldPlay]] | 补充训练 | 用 [[WorldPlay]] 在 OpenVid 图像上生成"指定轨迹"视频 |

### 实现细节（统一在 480×832 分辨率、77 帧训练）

**HY1.5-TI2V-8B 训练配置**

- Batch size: 32；Learning rate: $1\times 10^{-5}$；
- Bidirectional camera-fine-tune: 8K steps；
- Stage 1 (AR teacher forcing): 4K steps；
- Stage 2 (Causal ODE / Consistency): 1.5K steps；
- Stage 3 (Asymmetric DMD): 500 steps。

**Wan2.1-T2V-1.3B 训练配置**

- Batch size: 32；Learning rate: $2\times 10^{-6}$；
- Bidirectional: 5K steps；Stage 1: 4K steps；Stage 2: 2K steps；Stage 3: 200 steps。

**推理配置**

- Chunk size: 4 latent frames；
- 蒸馏后采样步数: 4；
- 硬件: 单 A800 GPU。

### 可视化结果

定性结果（Figure 2）显示蒸馏后模型能够：

- 在同一文本提示下，根据**前进 / 后退 / 旋转**不同相机轨迹生成对应视角的视频；
- 长序列滚动（>77 帧）不崩溃；
- 速度从 269s 量级降到 1s 量级，但视觉一致性可见。

---

## 批判性思考

### 优点

1. **工程价值大**: 把分散在多篇论文里的相机条件 / AR 蒸馏 / DMD 技术系统化串成可复现 pipeline，并完整开源。
2. **架构无关性验证完备**: 同一配方在两种 size、两种基座（T2V 与 TI2V）上都跑通，说明方法本身有迁移性。
3. **Ablation 实用**: 训练数据 / 步数 / batch size 三个 ablation 直接对应"我自己复现时会踩什么坑"，对工程师极友好。
4. **数据收集思路新**: 显式承认"感知反推 pose"路线不可行，转向"3D 重建重渲染"或"合成数据" — 给后续相机条件视频生成工作指了一条明确路。

### 局限性

1. **缺少与同期工作的定量对比**: 没有跟 CausVid、Self Forcing 等同期相机控制视频世界模型直接比 FVD / 相机控制精度等指标。
2. **首帧延迟表存在标注不一致**: Table 1 中 HY1.5 行的 latency 与 speedup 不自洽（multi-step AR 1.014s 标 9.52× 但 bidirectional 是 1.041s），需要核对 v2 / GitHub。
3. **没有 KV cache 细节**: 论文反复强调"流式推理"，但实际 KV 复用、显存管理、长序列退化等关键工程细节未展开。
4. **算法层面无新意**: PRoPE、Causal Forcing、DMD 都是引用 2026 年其他论文的方法，本文是 integration 而非 invention。
5. **可控性主要是相机 trajectory**: 没扩展到物体级动作 / 文本条件实时切换，距 [[交互式世界模型]] 的"用户给任意动作"还有距离。

### 潜在改进方向

1. 加入物体级动作条件（如 [[Action-Conditioned World Model]] 路线）做完整 embodied 控制；
2. 引入 [[Forcing-KV]] 等混合 KV cache 压缩，进一步降低长序列推理显存；
3. 在 [[OmniWorld]] 或 [[WBench]] 等评测上给定量分数；
4. 把 [[Causal Forcing++]] 的在线蒸馏推广到更激进的 1-step 设置。

### 可复现性评估

- [x] 代码开源（GitHub: shengshu-ai/minWM）
- [x] 权重承诺开源（基于已开源的 Wan2.1 / HunyuanVideo 基座）
- [x] 训练细节完整（batch size / lr / step 数都给了）
- [x] 训练数据来源公开（DL3DV、OpenVid 均开源；WorldPlay 也是开源工具）

---

## 关联笔记

### 基于

- [[Wan2.1]]: 1.3B T2V 基座之一；
- [[Diffusion Forcing]]: AR 扩散训练的通用范式；
- [[DMD]]: Stage 3 分布匹配蒸馏的核心算法；
- [[Consistency Distillation]]: Stage 2B 的理论基础；
- [[PRoPE 相机嵌入]]: 相机条件注入方法（引用自 Li et al. 2026）。

### 对比

- [[Causal Forcing]]: 本文 Stage 2A 选择的蒸馏路径；
- [[Causal Forcing++]]: 本文 Stage 2B 选择的蒸馏路径；
- [[HWM]]: 同期"物理原生"路线的世界模型立场论文，对比维度（视觉生成式 vs 物理结构化）；
- [[EA-WM]]: 以视频扩散为基座的另一类机器人世界模型。

### 方法相关

- [[交互式世界模型]]: 本文目标范式；
- [[视频扩散模型]]: 基座类型；
- [[Asymmetric DMD]]: 本文 Stage 3 核心技术；
- [[Causal ODE Initialization]]: Stage 2A 技术；
- [[Causal Consistency Distillation]]: Stage 2B 技术；
- [[相机可控微调]]: Stage 0 核心；
- [[少步蒸馏]]: 整体目标技术。

### 数据/工具相关

- [[DL3DV]]: 主训练数据；
- [[OpenVid]]: 补充图像源；
- [[WorldPlay]]: 合成训练视频的工具；
- [[SpatialVID-HQ]]: 失败对照数据。

---

## 速查卡片

> [!summary] minWM (2026.05)
> - **核心**: 全栈开源把 T2V/TI2V 基座蒸馏成相机可控、4-step 自回归的[[交互式世界模型]]；
> - **方法**: PRoPE 相机注入 + 三阶段 Causal Forcing(++)（AR 训练 → ODE/CD → Asymmetric DMD）；
> - **结果**: 首帧延迟 269s → 1.1s（Wan2.1，236× 加速），相机可控性保留；
> - **经验**: 数据精度 > 数据规模；可控性是 emergent capability，需 ≥ 5K 步 + bs ≥ 16；
> - **代码**: https://github.com/shengshu-ai/minWM

---

*笔记创建时间: 2026-05-30*
