---
title: "Causal-rCM: A Unified Teacher-Forcing and Self-Forcing Open Recipe for Autoregressive Diffusion Distillation in Streaming Video Generation and Interactive World Models"
method_name: "Causal-rCM"
authors: [Kaiwen Zheng, Guande He, Min Zhao, Jintao Zhang, Huayu Chen, Jianfei Chen, Chen-Hsuan Lin, Ming-Yu Liu, Jun Zhu, Qianli Ma]
year: 2026
venue: arXiv
tags: [diffusion-distillation, autoregressive-video, consistency-model, streaming-video, world-model]
zotero_collection: 1-生成模型
image_source: online
arxiv_html: https://arxiv.org/html/2606.25473
created: 2026-06-25
---

# 论文笔记：Causal-rCM

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Tsinghua University / UT Austin / NVIDIA |
| 日期 | June 2026 |
| 项目主页 | — |
| 对比基线 | [[Self-Forcing]]、[[Causal Forcing]]、AnyFlow |
| 链接 | [arXiv](https://arxiv.org/abs/2606.25473) / [Code](https://github.com/NVlabs/rcm) |

---

## 一句话总结

> 将 [[rCM]] 的「正向CM + 反向DMD」互补范式推广到自回归视频扩散，提出融合 Teacher-Forcing 和 Self-Forcing 的统一蒸馏配方，以 1-2 步采样达到 84.63 VBench-T2V SOTA。

---

## 核心贡献

1. **统一 TF+SF 蒸馏框架**: 证明 Teacher-Forcing [[Consistency Model]] 与 Self-Forcing [[DMD]] 在自回归视频场景下互补，离线 TF 目标提供 mode-covering 初始化，在线 SF 目标进行分布对齐
2. **首个因果连续时间 CM 实现**: 引入 TF-sCM/[[MeanFlow]] 并实现支持自定义掩码的 FlashAttention-2 [[JVP]] 核，收敛速度比离散时间 CM 快 **10×**
3. **生产级基础设施配方**: 统一支持 TF/DF/SF 因果范式，融合 FSDP2、Context Parallelism、Selective Activation Checkpointing，并应用于 Cosmos 3 交互式世界模型

---

## 问题背景

### 要解决的问题

自回归视频扩散（如 [[Self-Forcing]]、[[Causal Forcing]]）存在 **训练-推理 Gap**：训练时用 clean context（teacher-forcing），推理时用 noisy/generated context（self-forcing），导致误差积累和分布漂移。同时，少步采样（1-4步）下生成质量仍落后于多步基线（50步）。

### 现有方法的局限

- **[[Self-Forcing]]**: 纯在线 SF-DMD 初始化不稳定，需大量迭代才能收敛
- **[[Causal Forcing]]**: 纯 Teacher-Forcing 一致性蒸馏，不做在线对齐，质量上限受限
- **TF-dCM（离散时间 CM）**: 离散步长导致梯度估计粗糙，收敛慢

### 本文的动机

[[rCM]] 在图像生成中证明了 CM（正向散度，mode-covering）+ DMD（反向散度，mode-seeking）的互补性。Causal-rCM 将同样的思路扩展到因果自回归视频场景：用 TF-sCM 做稳定的离线初始化，再用 SF-DMD 做在线分布精修。

---

## 方法详解

### 总体框架

Causal-rCM 分三阶段训练：

1. **Stage 1-2**: TF/DF 预热（蒸馏 clean-context 条件下的视频预测能力）
2. **Stage 3a**: TF-sCM（连续时间一致性蒸馏，快速初始化）
3. **Stage 3b**: SF-DMD（自强迫分布匹配蒸馏，在线精修）

基于 [[Wan2.1]]-1.3B 的 T2V 模型作为教师，Transformer 骨干 + [[3D Causal VAE]] 编码。

### 核心模块

#### 模块1: Teacher-Forcing Packed Causal Forward

**设计动机**: 在一次前向传播中同时处理 clean context 帧和 noisy target 帧，利用 [[因果注意力掩码]] 约束信息流。

**具体实现**:
- 输入拼接：`[x₀^clean, x_t^noisy]`，时间步标注 `[0^clean, t^noisy]`
- 自定义 TF 注意力掩码 $M_{\text{TF}}$：noisy 块只能 attend 到 clean 历史，clean 块正常双向自注意力
- 输出：取 `_noisy` 对应位置的速度预测 $\mathbf{v}_\theta$

#### 模块2: TF-sCM（连续时间因果一致性模型）

**设计动机**: 避免 TF-dCM 的离散步长估计误差，利用 [[JVP]]（Jacobian-vector product）计算沿 ODE 轨迹的切线方向，实现连续时间监督。

**具体实现**:
- 基础预测函数 $\mathbf{F}_\theta$（与 [[sCM]]/[[MeanFlow]] 兼容）
- 沿教师 ODE 轨迹计算切线：需要通过 TF-packed 前向传播的 JVP
- 自定义 [[FlashAttention-2]] JVP 核：将稀疏 TF 掩码表达为 admissible query-key 区间，避免显存爆炸（不显式物化密集掩码）
- 切线归一化稳定训练

**RF 原生公式优于 TrigFlow 包装**：在因果设置下直接用 Rectified Flow 公式（速度场 $v$）推导 TF-sCM，比套 TrigFlow 得到更平滑的输出。

#### 模块3: SF-DMD（自强迫分布匹配蒸馏）

**设计动机**: 弥合训练-推理 Gap，让模型在自生成 context 上进行在线对齐。

**具体实现**:
- 自回归 rollout：逐帧/块生成，用 KV Cache 缓存 clean token 的 KV 状态（`Append` 操作）
- 仅最后一步 $t_1 \to 0$ 保留梯度，中间步 stop-gradient
- DMD 损失用自适应归一化防止梯度爆炸
- SF-DMD 训练循环遍历不同步数调度表，确保各去噪区间都被覆盖为可微最终步

#### 模块4: Noisy Context 推理加速

**设计动机**: 避免 clean context 编码的额外 forward pass，减少延迟。

**具体实现**:
- 推理时直接复用最后一步去噪的 KV Cache 作为 context
- 残留噪声提供低通滤波效果，抑制误差积累
- 将有效 NFE 从 $N+1$ 降至 $N$（每块节省 1 次 forward）

---

## 关键公式

### 公式1: [[Consistency Model|离散时间 CM 损失]] (dCM)

$$
\mathcal{L}_{\text{dCM}}(\theta) = \mathbb{E}_{\mathbf{x}_0 \sim p_{\text{data}}, \epsilon, t}\left[ w(t)\, d\!\left(\mathbf{f}_\theta(\mathbf{x}_t, t),\; \mathbf{f}_{\theta^-}(\hat{\mathbf{x}}_{t-\Delta t}, t-\Delta t)\right) \right]
$$

**含义**: 要求模型对同一 ODE 轨迹相邻两步的预测一致，$\mathbf{f}_{\theta^-}$ 为 EMA 教师

**符号说明**:
- $w(t)$: 时间步权重函数
- $d(\cdot, \cdot)$: 距离度量（如 $\ell_2$）
- $\hat{\mathbf{x}}_{t-\Delta t}$: 由教师 ODE solver 从 $\mathbf{x}_t$ 推进一步得到的样本
- $\theta^-$: EMA 教师参数

### 公式2: [[sCM|连续时间 CM 损失]] (sCM)

$$
\mathcal{L}_{\text{sCM}}(\theta) = \mathbb{E}\!\left[\left\| \mathbf{F}_\theta - \mathbf{F}_{\theta^-} - \frac{\mathbf{g}}{\|\mathbf{g}\|_2^2 + c} \right\|_2^2\right]
$$

**含义**: 用连续时间切线 $\mathbf{g} = w(t)\,\mathrm{d}\mathbf{f}_{\theta^-}/\mathrm{d}t$ 代替离散步长，精确监督 ODE 轨迹方向

**符号说明**:
- $\mathbf{F}_\theta$: 连续时间基础预测（由 $\mathbf{f}_\theta$ 导出）
- $\mathbf{g}$: 沿教师 ODE 的切线向量，通过 JVP 计算
- $c$: 归一化常数，防止除零

### 公式3: [[DMD|分布匹配蒸馏梯度]]

$$
\nabla_\theta \mathcal{L}_{\text{DMD-raw}}(\theta) = \mathbb{E}\!\left[w(t)\left(\nabla \log p_\theta^t - \nabla \log p_{\text{teacher}}^t\right)^\top \frac{\mathrm{d}\mathbf{x}_t}{\mathrm{d}\theta}\right]
$$

**含义**: 最小化学生分布与教师分布的反向 KL，通过 score 函数差计算梯度

**符号说明**:
- $p_\theta^t$: 学生在时间步 $t$ 的扩散边际分布
- $p_{\text{teacher}}^t$: 教师分布
- $\mathrm{d}\mathbf{x}_t/\mathrm{d}\theta$: 生成轨迹对参数的梯度

### 公式4: [[DMD|DMD 自适应归一化损失]]

$$
\mathcal{L}_{\text{DMD}}(\theta) = \mathbb{E}\!\left[\left\| \mathbf{x}_0^\theta - \mathrm{sg}\!\left[\mathbf{x}_0^\theta - \frac{\mathbf{f}_{\text{fake}} - \mathbf{f}_{\text{teacher}}}{\mathrm{mean}(|\cdot|)}\right] \right\|_2^2\right]
$$

**含义**: 用 stop-gradient 和自适应归一化稳定 DMD 训练

**符号说明**:
- $\mathbf{x}_0^\theta$: 学生生成的干净样本
- $\mathrm{sg}[\cdot]$: stop-gradient 操作
- $\mathbf{f}_{\text{fake}}, \mathbf{f}_{\text{teacher}}$: 分别为 fake scorer 和教师 scorer 的输出

### 公式5: [[Teacher Forcing|Teacher-Forcing 速度预测目标]] (Eq. 9)

$$
\mathcal{L}_{\text{TF}}(\theta) = \mathbb{E}\!\left[w(t)\left\| \left[\mathbf{v}_\theta\!\left([\mathbf{x}_0^{\text{clean}}, \mathbf{x}_t^{\text{noisy}}],\, [0^{\text{clean}}, t^{\text{noisy}}];\, M_{\text{TF}}\right)\right]_{\text{noisy}} - \mathbf{v} \right\|_2^2\right]
$$

**含义**: TF packed forward 的流匹配损失，只对 noisy 目标位置监督速度场

**符号说明**:
- $[\cdot]_{\text{noisy}}$: 取拼接序列中 noisy 部分对应的输出
- $M_{\text{TF}}$: 自定义 TF 因果注意力掩码
- $\mathbf{v}$: 目标速度（真实数据对应的 RF 速度）

### 公式6: [[sCM|TF-sCM 切线归一化损失]] (Eq. 18)

$$
\mathcal{L}_{\text{TF-sCM}}(\theta) = \mathbb{E}\!\left[\left\| \Delta \mathbf{v}_\theta^{\text{TF}} - \frac{w(t)\,\mathbf{h}_{\text{TF-sCM}}}{w^2(t)\|\mathbf{h}_{\text{TF-sCM}}\|_2^2 + c} \right\|_2^2\right]
$$

**含义**: TF 因果场景下的连续时间 CM 损失，$\Delta \mathbf{v}$ 为学生与 EMA 教师的速度差，$\mathbf{h}$ 为 JVP 切线

**符号说明**:
- $\Delta \mathbf{v}_\theta^{\text{TF}}$: 学生与 EMA 教师在 TF 设置下的速度预测差
- $\mathbf{h}_{\text{TF-sCM}}$: 教师 ODE 轨迹的 JVP 切线向量
- $c$: 归一化稳定常数

---

## 关键图表

### Figure 1: VBench-T2V 性能对比

![Figure 1](https://arxiv.org/html/2606.25473/x1.png)

**说明**: Causal-rCM 在 1/2/4-step 生成下的 VBench-T2V 分数，frame-wise（c1-1）和 chunk-wise（c3-3）两种设置均达到 SOTA。2-step 帧级生成达到 84.63，超越所有现有流式视频方法。

### Figure 2: 统一散度视角

![Figure 2](https://arxiv.org/html/2606.25473/x2.png)

**说明**: 从正向/反向 KL 视角统一 CM 和 DMD：TF-CM 最小化正向散度（mode-covering），SF-DMD 最小化反向散度（mode-seeking），两者在 Causal-rCM 中互补。

### Figure 3: 因果训练范式对比

![Figure 3](https://arxiv.org/html/2606.25473/x3.png)

**说明**: 并排展示 Teacher-Forcing（TF）、Diffusion-Forcing（DF）和 Self-Forcing（SF）三种因果范式。TF 使用 clean context，SF 使用自生成 context，DF 使用加噪 context。

### Figure 4: Causal-rCM 与现有方法对比

![Figure 4](https://arxiv.org/html/2606.25473/x4.png)

**说明**: 算法层面定位图，展示 Causal-rCM 如何融合多种范式，相比 [[Self-Forcing]] / [[Causal Forcing]] 的独立实现，提供了更完整的训练流程。

### Figure 5: 加速技术适配

![Figure 5](https://arxiv.org/html/2606.25473/x5.png)

**说明**: 展示如何将 SAC（Selective Activation Checkpointing）× FlexAttention、JVP × FSDP2、Ulysses CP 等基础设施组合适配到 Causal-rCM 训练框架。

### Figure 6: TF-dCM vs TF-sCM 训练曲线

![Figure 6](https://arxiv.org/html/2606.25473/x6.png)

**说明**: TF-sCM（连续时间）比 TF-dCM（离散时间）收敛快约 **10×**，验证了 JVP 切线监督的有效性。

### Figure 7a & 7b: SF-DMD 训练曲线

![Figure 7a](https://arxiv.org/html/2606.25473/x7.png)
![Figure 7b](https://arxiv.org/html/2606.25473/x8.png)

**说明**: Frame-wise（7a）和 Chunk-wise（7b）SF-DMD 各初始化策略的训练曲线。TF-dCM 初始化（TF-sCM 早停版）实现最佳质量-收敛平衡。

### Figure 9: Cosmos 3 交互式世界模型架构

![Figure 9](https://arxiv.org/html/2606.25473/x9.png)

**说明**: 双塔架构：UND（Understanding）塔处理文本/提示（因果自注意力），GEN（Generation）塔处理视觉/动作/声音（超 token 级时序因果注意力）。Action conditioning 通过 null action supertoken 对齐，统一 3D mRoPE 位置编码。

### Figure 10: 自动驾驶动作条件生成演示

![Figure 10](https://arxiv.org/html/2606.25473/x10.png)

**说明**: Cosmos 3 在自动驾驶场景下的流式动作条件生成，展示 Causal-rCM 作为交互式世界模型的实际应用。

### Table 1: 正向-反向目标互补性对比

| 框架 | 正向目标 | 反向目标 | 领域 |
|------|---------|---------|------|
| DDO | Score Distillation | — | 图像 |
| DiffusionNFT | Flow Matching | — | 图像 |
| DDRL | CM | RLHF | 图像 |
| rCM | CM | DMD | 图像/视频 |
| **Causal-rCM** | **TF-CM** | **SF-DMD** | **自回归视频** |

**说明**: 前向（mode-covering）+ 反向（mode-seeking）互补范式贯穿多个扩散蒸馏领域，Causal-rCM 是这一思路在自回归视频中的系统实现。

### Table 2: 自回归视频扩散代码库对比

| 特性 | Self-Forcing | FastVideo | FastGen | Causal-rCM |
|------|-------------|-----------|---------|------------|
| FSDP2 | — | ✓ | — | ✓ |
| Context Parallelism | — | — | — | ✓ |
| JVP 支持 | — | — | — | ✓ |
| KV Cache | ✓ | — | ✓ | ✓ |
| TF/DF/SF 全范式 | — | — | — | ✓ |

**说明**: Causal-rCM 是唯一同时支持所有因果范式和完整并行训练基础设施的开源配方。

### Table 3: 训练配置（三阶段）

| 配置项 | Stage 1-2: TF/DF | Stage 3a: TF-dCM | Stage 3a: TF-sCM | Stage 3b: SF-DMD |
|--------|-----------------|-----------------|-----------------|----------------|
| Global batch size | 256 / 64 | 32 | 32 | 64 |
| Context parallel | 1 / 8 | 4 | 4 | 4 |
| Student LR | 1e-5 | 2e-6 | 2e-6 | 2e-6 |
| EMA / Fake-score LR | — | — | — | 4e-7 |
| Optimizer β | (0.9, 0.999) | (0, 0.999) | (0, 0.999) | (0, 0.999) |
| CFG scale | — | 3.0 | 3.0 | 5.0 |
| 时间步采样 | UniformShift(5) | uniform RF grid, skip=1 | LogitNormal(μ=-0.8, σ=1.6) | UniformShift(5) |
| 训练迭代 | 30k + 30k | 10k | 1k | varies |

**关键发现**: TF-sCM 仅需 1k 步即可完成 TF-dCM 10k 步的效果，体现 10× 收敛加速。

### Table 4: 流式视频生成主要结果（VBench-T2V）

| 方法 | NFE | Total | Quality | Semantic | FPS | TF 延迟(s) | SF 延迟(s) |
|------|-----|-------|---------|---------|-----|-----------|-----------|
| **双向基线** |
| Wan2.1-1.3B (50步) | 100 | 82.78 | 83.44 | 80.13 | 0.72 | — | — |
| Wan2.1-14B (50步) | 100 | 83.35 | 83.97 | 80.88 | 0.18 | — | — |
| **Frame-wise (c1-1)** |
| Causal Forcing (4步) | 5 | 81.56 | 82.59 | 77.44 | 8.3 | 0.40 | 0.46 |
| Causal-rCM (4步) | 5 | 84.29 | 85.27 | 80.36 | 8.3 | 0.40 | 0.46 |
| **Causal-rCM (2步)** | **3** | **84.63** | **85.46** | **81.31** | **12.2** | **0.40** | **0.31** |
| Causal-rCM (2步, noisy ctx) | 2 | 83.11 | 83.55 | 81.37 | 15.9 | 0.40 | 0.23 |
| Causal-rCM (1步) | 2 | 84.63 | 85.54 | 81.01 | 15.9 | 0.40 | 0.23 |
| **Chunk-wise (c3-3)** |
| Self-Forcing (4步) | 5 | 83.76 | 84.53 | 80.68 | 17.4 | 0.57 | 0.64 |
| LongLive (4步) | 5 | 83.62 | 84.36 | 80.69 | 17.4 | 0.57 | 0.64 |
| Causal Forcing (4步) | 5 | 83.96 | 84.94 | 80.04 | 17.4 | 0.57 | 0.64 |
| AnyFlow (4步) | 5 | 84.31 | 85.15 | 80.94 | 17.4 | 0.57 | 0.64 |
| Causal-rCM (4步) | 5 | 84.37 | 85.02 | 81.73 | 17.4 | 0.57 | 0.64 |
| Causal-rCM (2步) | 3 | 84.30 | 85.04 | 81.36 | 22.2 | 0.57 | 0.49 |
| Causal-rCM (1步) | 2 | 84.01 | 84.71 | 81.22 | 25.6 | 0.57 | 0.41 |

**关键发现**: 1-2 步 Causal-rCM 已超越 50 步双向基线（+1.28 Total），15.9-25.6 FPS 实现真正实时流式生成。

### Table 5: 初始化策略消融（4-step SF-DMD）

| 初始化 | FW Total | FW Quality | FW Semantic | FW 迭代数 | CW Total | CW Quality | CW Semantic | CW 迭代数 |
|--------|---------|-----------|-----------|---------|---------|-----------|-----------|---------|
| DF | 83.11 | 83.85 | 80.16 | 2000 | 84.80 | 85.58 | 81.65 | 1500 |
| TF | 82.62 | 83.62 | 78.61 | 1000 | 84.95 | 85.82 | 81.47 | 1000 |
| DF-KD | 80.59 | 80.41 | 81.32 | 2000 | 83.61 | 84.10 | 81.68 | 1500 |
| TF-KD | 83.49 | 84.50 | 79.43 | 1250 | 83.79 | 84.41 | 81.30 | 1000 |
| TF-dCM | **84.29** | **85.27** | 80.36 | 1200 | 84.33 | 85.22 | 80.75 | 3200 |
| TF-sCM | 83.84 | 84.67 | 80.55 | 1000 | **84.37** | **85.02** | **81.73** | 1250 |

**关键发现**: TF-dCM 在 frame-wise 最优，TF-sCM 在 chunk-wise 最优；两者均显著优于 DF/TF 简单初始化，验证了 [[Consistency Model]] 初始化的必要性。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| Wan2.1 合成数据 | — | 仅用合成数据，无真实数据 | 训练 |
| VBench-T2V | 标准基准 | 覆盖 Total/Quality/Semantic 三个维度 | 评测 |

> 论文强调所有蒸馏训练仅使用 **合成数据**（教师生成），不依赖真实视频标注。

### 实现细节

- **Backbone**: [[Wan2.1]]-1.3B T2V（Transformer + 3D Causal VAE）
- **Student optimizer**: AdamW，lr=2×10⁻⁶，β=(0, 0.999)，wd=0.01
- **并行策略**: FSDP2 参数分片 + Ulysses Context Parallelism（4路）
- **显存优化**: Selective Activation Checkpointing（SAC）
- **KV Cache**: Post-RoPE 位置编码后缓存，兼容 CP 分片
- **硬件**: 多 GPU（具体数量未明确）

### 关键工程细节

**JVP × FSDP2 兼容性**：FSDP2(JVP) 设计——先用 FSDP2 分片参数再做 JVP，避免 JVP × all-gather 的内存冲突。

**SAC × FlexAttention**：只对计算密集型无状态区域做选择性重计算，与自定义掩码 FlexAttention 组合使用。

**Ulysses CP × JVP**：时空 token 展平后 all-to-all 分片，JVP 在分片后的局部 token 上计算，无通信开销增加。

---

## 批判性思考

### 优点

1. **系统性**: 将 TF/DF/SF 三种范式、连续/离散时间 CM 统一在单一框架内，是迄今最完整的自回归视频蒸馏配方
2. **工程贡献实质**: JVP FlashAttention-2 核、FSDP2(JVP) 设计等基础设施贡献有独立价值，可复用于其他因果 sCM 工作
3. **性能突破**: 1-2 步超越 50 步基线，对实时应用意义重大

### 局限性

1. **步数调度依赖 chunk 粒度**: frame-wise 偏好 1-2 步浅 rollout，chunk-wise 需要 4 步深去噪，没有统一最优调度
2. **仅合成数据训练的天花板**: 教师质量上限制约蒸馏学生，真实数据微调路径未探索
3. **Cosmos 3 部分信息不完整**: 交互世界模型应用篇幅有限，动作条件生成的量化评估缺失

### 潜在改进方向

1. **自适应步数调度**: 根据 chunk 内容复杂度动态分配去噪步数
2. **跨 chunk 长距离一致性**: 当前 noisy context trick 依赖局部 KV cache，长视频一致性待验证
3. **与 RLHF / CM-GRPO 结合**: 在 SF-DMD 基础上叠加偏好对齐

### 可复现性评估

- [x] 代码开源（https://github.com/NVlabs/rcm）
- [ ] 预训练模型（未明确）
- [x] 训练细节完整（Table 3 超参数完整）
- [x] 数据集可获取（基于公开 Wan2.1）

---

## 关联笔记

### 基于

- [[rCM]]: 核心范式来源，CM + DMD 互补原理
- [[Self-Forcing]]: SF-DMD 组件的直接前驱
- [[Causal Forcing]]: TF 蒸馏组件的直接前驱
- [[Wan2.1]]: 基础教师模型
- [[MeanFlow]]: TF-sCM 的连续时间变体之一

### 对比

- [[Causal Forcing]]: 仅 TF，无在线 SF 对齐
- [[Self-Forcing]]: 仅 SF，初始化不稳定
- AnyFlow: flow map 蒸馏路线，无 TF+SF 联合训练

### 方法相关

- [[Consistency Model]]: 核心蒸馏方法
- [[DMD]]: 反向 KL 分布匹配
- [[JVP]]: 连续时间 CM 的关键计算原语
- [[FlashAttention-2]]: 定制 JVP 核的基础
- [[因果注意力掩码]]: TF packed forward 的核心设计
- [[Teacher Forcing]]: 离线训练范式

### 硬件/数据相关

- [[3D Causal VAE]]: 视频编码器
- [[Wan2.1]]: 基础模型

---

## 速查卡片

> [!summary] Causal-rCM
> - **核心**: rCM 正向CM+反向DMD互补范式的自回归视频版本
> - **方法**: TF-sCM（连续时间因果CM，JVP加速）→ SF-DMD（在线自强迫分布对齐）
> - **结果**: 1-2步达到 84.63 VBench-T2V，超越 50步双向基线，15.9-25.6 FPS 实时流式
> - **代码**: https://github.com/NVlabs/rcm

---

*笔记创建时间: 2026-06-25*
