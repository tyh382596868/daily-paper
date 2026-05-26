---
title: "SenseNova-U1: Unifying Multimodal Understanding and Generation with NEO-unify Architecture"
method_name: "SenseNova-U1"
authors: [Haiwen Diao, Penghao Wu, Hanming Deng, Jiahao Wang, et al. (59 authors)]
year: 2026
venue: arXiv
tags: [unified-multimodal-model, native-multimodal, mixture-of-transformers, flow-matching, encoder-free, image-generation, multimodal-understanding]
zotero_collection: 1-生成模型/统一多模态
image_source: online
arxiv_html: https://arxiv.org/html/2605.12500
created: 2026-05-24
---

# 论文笔记：SenseNova-U1: Unifying Multimodal Understanding and Generation with NEO-unify Architecture

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | SenseTime（商汤） |
| 日期 | May 2026 (arXiv:2605.12500, 提交 2026-05-12) |
| 项目主页 | <https://github.com/OpenSenseNova/SenseNova-U1> |
| 模型变体 | SenseNova-U1-8B-MoT（dense）/ SenseNova-U1-A3B-MoT（sparse, ~3B 激活） |
| 对比基线 | [[BAGEL]]、[[Qwen-Image]]、[[FLUX]]、[[OmniGen2]]、Emu3.5、Lumina-DiMOO |
| 链接 | [arXiv](https://arxiv.org/abs/2605.12500) / [PDF](https://arxiv.org/pdf/2605.12500) / [Code](https://github.com/OpenSenseNova/SenseNova-U1) |

---

## 一句话总结

> 商汤把"理解"和"生成"塞进同一个 [[MoT|Mixture-of-Transformers]] 骨架，靠 [[Near-Lossless Visual Interface|近无损视觉接口]] 直接在像素 patch 上做 [[Flow Matching|流匹配]]，端到端跑通 [[Unified Multimodal Model|统一多模态]] 的"原生"路线。

---

## 核心贡献

1. **NEO-unify 架构**: 提出 [[NEO-unify]] 范式——把 understanding 和 generation 视为同一个原生 pixel-text 过程的两种 view，**不用预训练 vision encoder、不用 [[VAE]] 解码器**，靠一个轻量卷积接口 + 一个 [[MoT]] 主干完成全部工作。
2. **Near-Lossless Visual Interface**: 仅用两层卷积 + 2D 正弦位置编码完成 32× patch 压缩，生成端直接用 [[MLP]] head 预测像素 patch，绕开 deep diffusion head 和 VAE，避免下游归纳偏置污染表征。
3. **统一训练目标 + 混合注意力**: 设计 $\mathcal{L}_{\text{total}}=\lambda_1\mathcal{L}_{\text{Und}}+\lambda_2\mathcal{L}_{\text{Gen}}$ 联合优化，并在 attention 层用 [[Hybrid Attention|混合注意力]]——text 走 causal mask，image 走 bidirectional，noise/clean token 间做单向可见性控制。
4. **双规模 MoT 变体**: 8B-MoT 是 dense 对称分支；A3B-MoT 是不对称 sparse MoE（understanding 128 expert / 30B 总参，generation 32 expert / 8B 总参，top-8 路由，激活 ~3B），在 GenEval / DPG-Bench / TIIF-Bench 等多个 benchmark 上达到开源 SOTA。
5. **延伸到具身**: 在 [[VLA]] 和 [[World Model|世界模型]] 场景下也有 preliminary 表现（论文未给细节数字），暗示统一多模态可作为下一代具身基础模型 backbone 候选。

---

## 问题背景

### 要解决的问题

[[Unified Multimodal Model|统一多模态模型]] 一直在 "encoder-based" 和 "encoder-free" 两条路线之间摇摆：前者（如 [[BAGEL]]、[[OmniGen2]]、[[Qwen-Image]]）依赖独立的 ViT vision encoder + VAE，虽然性能稳定但语义和像素表征被 frozen module 限制；后者（如 Chameleon、Emu3）虽然原生但生成质量长期落后。**如何在"原生 token in / pixel out"前提下，让理解和生成相互促进，并把生成质量推到 SOTA**，是本文核心问题。

### 现有方法的局限

1. **Modular 架构包袱重**: 预训练 vision encoder（CLIP、SigLIP）和 VAE 解码器把表征压在它们自己的目标空间里，understanding 与 generation 难以共享中间层语义。
2. **MoE 路由粒度太粗**: 标准 [[MoE]] 在同一 transformer block 内做 token-level expert 选择，无法显式区分"理解流"和"生成流"——两类流量都在同一组参数上 fight。
3. **Encoder-Free 但生成弱**: Chameleon、Emu3 等离散 token 路线虽然原生，但量化 tokenizer 损失视觉细节，T2I/编辑表现一直不如 diffusion 路线。
4. **训练成本极不均衡**: 理解任务收敛快，生成任务需要长时间高分辨率训练，单一 schedule 难以兼顾。

### 本文的动机

如果把 understanding 和 generation 看作"同一个 backbone 在不同 token 类型下的两种 view"，那么：
- 把 vision encoder/VAE 全部内化进 backbone，让模型自己学最优视觉表征 → **Near-Lossless Visual Interface**
- 在 transformer 层级用 [[MoT]] 而非 [[MoE]] 做模态特化，**stream 之间完全参数解耦**，stream 之内通过 attention 共享 context → **NEO-unify MoT 主干**
- 用 [[Flow Matching|流匹配]] 取代 discrete tokenizer，保留像素细节 → **像素空间流匹配损失**

---

## 方法详解

### 模型架构

SenseNova-U1 基于 [[NEO-unify]] 范式构建，整体是一个原生 pixel-text 统一模型：

- **输入**: 任意分辨率图像 + 文本，统一切成 patch / sub-word token
- **视觉接口**: [[Near-Lossless Visual Interface|近无损视觉接口]]——两层卷积（stride 16 / stride 2）+ GELU + 2D 正弦位置编码，得到 32×32 patch token，**32× 压缩比**
- **Backbone**: [[MoT|Mixture-of-Transformers]] 主干，understanding 和 generation 两条 stream **完全参数解耦**（独立 projection / normalization / FFN），但共享同一 attention 序列空间
- **Pre-Buffer 层**: 把 raw pixel / text 映射到统一表征空间
- **Post-LLM 层**: 保留语言能力的尾部 transformer block
- **生成解码**: 直接用 [[MLP]] head 预测像素 patch，**不经过 VAE 或 diffusion head**
- **总参数**: 8B-MoT 8B dense / A3B-MoT ~38B 总参 / ~3B 激活

### 核心模块

#### 模块 1: Near-Lossless Visual Interface

**设计动机**: 取代 ViT + VAE 这种 "frozen encoder + frozen decoder" 的归纳偏置链条，让模型自己学最优视觉表征空间。

**具体实现**:
- 编码：两层 conv（stride 16 → stride 2，总 stride 32）+ GELU 激活 + [[2D 正弦位置编码]]，输出 32×32 patch token
- 解码：[[Flow Matching|流匹配]] 框架下，[[MLP]] head 直接预测每个 patch 的像素值
- 端到端可学习：representation space 不被外部模块"压死"
- 动态分辨率：训练 spans 256² → 4096²（understanding），512² → 2048²（generation）
- **动态噪声缩放** $\sigma_R(H,W) = \sigma_0\sqrt{N(H,W)/N_0}$，让 per-token noise energy 在不同分辨率下保持近似常量

#### 模块 2: Mixture-of-Transformers (MoT) Backbone

**设计动机**: 标准 [[MoE]] 是 token-level sparse routing，所有 expert 在同一参数池抢资源；[[MoT]] 在 transformer block 级别按模态/任务分流，stream 之间参数完全解耦但 context 共享。

**具体实现**:
- **Token type-aware routing**: text token 路由到 understanding stream，image token（无论生成态/理解态）路由到 generation stream
- **完全参数解耦**: 每个 stream 有独立的 projection、normalization、FFN block
- **共享 attention 序列**: 两类 stream token 在 attention 阶段在同一序列里相互可见，"perception and synthesis interact natively at every layer"
- **稀疏变体（A3B）**:
  - Understanding stream: 128 expert × ~234M ≈ 30B 总参
  - Generation stream: 32 expert × 250M ≈ 8B 总参
  - Top-8 routing → ~3B 激活参数

#### 模块 3: Hybrid Attention Pattern

**设计动机**: 文本天然是 autoregressive（causal），图像天然是 holistic（bidirectional）；统一序列里要让两种 mask 协同。

**具体实现**:
- Text row: 标准 causal mask（attend 到前面所有 text + 所有可见 image）
- Image row: 可双向 attend 到整个 image span + 前缀 text
- Noise token vs clean token: noise 可见 clean，clean 不可见 noise（保证 noise → clean 的 flow matching 方向）
- 实现侧保留 pure text block 的 causal fast path，仅在含 image token 的 block 扩展 key range，**避免不必要的 attention 开销**

#### 模块 4: Joint Training Objective

详见下文 [关键公式](#关键公式)。两类损失加权求和；Stage 3-4 用 $\lambda_1=0.1, \lambda_2=1.0$，prioritize 生成保真度。

### 训练流程（4 阶段）

| Stage | 名称 | 数据/任务 | 关键超参 |
|-------|------|----------|---------|
| **1** | Understanding Warmup | 从 NEO 初始化；attention-fusion 把 QK 参数减半；全模型 continue training | LR $2\times10^{-5}$ |
| **2** | Generation Pre-Training | 冻结 understanding stream，专训 generation：Phase I 256-1024px 120K 步 / Phase II 512-2048px 60K 步 / Phase III 编辑+交错 120K 步 | LR $2\times10^{-4}$ |
| **3** | Unified Mid-Training | 解冻全模型联合训练：33% understanding / 37% T2I / 24% editing / 6% interleaved | 84K 步，LR $2\times10^{-5}$ |
| **4** | Unified SFT | 指令对齐 | 9K 步，cosine decay $2\times10^{-5} \to 0$ |

**Post-training**:
- 强化学习：text-rendering / style / aesthetic reward，[[Classifier-Free Guidance|CFG]] 联合优化
- 蒸馏：用 [[DMD|DMD²]] 把 100-step 采样蒸馏到 8 步

### 推理基础设施

Disaggregated inference（参见 Figure 5）：
- Understanding stream 在 **LightLLM** 上跑（KV cache 友好的 LLM 推理引擎）
- Generation stream 在 **LightX2V** 上跑（针对 diffusion 优化的推理引擎）
- 两个 engine 异步通信，吃满 understanding 和 generation 不同的算力 profile

---

## 关键公式

### 公式 1: [[Unified Multimodal Model|联合训练目标]]

$$
\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{Und}} + \lambda_2 \mathcal{L}_{\text{Gen}}
$$

**含义**: 把理解 loss 和生成 loss 线性组合，权重由训练阶段决定（Stage 3-4 取 $\lambda_1=0.1, \lambda_2=1.0$，向生成保真倾斜）。

**符号说明**:
- $\mathcal{L}_{\text{Und}}$: 理解侧 next-token prediction cross-entropy
- $\mathcal{L}_{\text{Gen}}$: 生成侧 pixel-space [[Flow Matching|流匹配]] velocity MSE
- $\lambda_1, \lambda_2$: 阶段相关权重

### 公式 2: [[next-token prediction|理解侧损失]]

$$
\mathcal{L}_{\text{Und}} = -\frac{1}{N}\sum_{n=1}^{N}\log p_\theta(x_n \mid x_{<n}, \mathbf{c})
$$

**含义**: 标准自回归语言建模 negative log-likelihood，$\mathbf{c}$ 是图像上下文 condition。

**符号说明**:
- $x_n$: 第 $n$ 个 text token
- $x_{<n}$: 历史 token 序列
- $\mathbf{c}$: 图像 patch token 上下文
- $N$: 序列长度

### 公式 3: [[Flow Matching|流匹配]] 前向插值

$$
\mathbf{z}_t = t\,\mathbf{x} + (1-t)\,\sigma_R\,\boldsymbol{\epsilon}, \quad t\in[0,1]
$$

**含义**: 在数据 $\mathbf{x}$ 和高斯噪声 $\boldsymbol{\epsilon}$ 之间做线性 [[Flow Matching|插值]] 构造训练样本 $\mathbf{z}_t$；$\sigma_R$ 随分辨率动态缩放，保 per-token 噪声能量恒定。

**符号说明**:
- $\mathbf{x}$: 真实像素 patch
- $\boldsymbol{\epsilon}\sim\mathcal{N}(0,I)$: 高斯噪声
- $t\sim\mathcal{U}(0,1)$: 时间步
- $\sigma_R = \sigma_R(H,W) = \sigma_0\sqrt{N(H,W)/N_0}$: 分辨率自适应噪声尺度

### 公式 4: 速度场预测

$$
\mathbf{v}_\theta(\mathbf{z}_t, t) = \frac{\hat{\mathbf{x}}_\theta(\mathbf{z}_t, t, \mathbf{s}_t) - \mathbf{z}_t}{1 - t}
$$

**含义**: 用 $\mathbf{x}$-prediction reparametrization 把网络预测重写成速度场——既享受 $\mathbf{x}$-pred 数值稳定性，又能直接当 flow matching 的速度回归目标。

**符号说明**:
- $\hat{\mathbf{x}}_\theta$: 网络对原始像素的预测
- $\mathbf{s}_t$: 文本/图像 condition
- $\mathbf{z}_t$: 当前时间步的 noisy patch

### 公式 5: [[扩散损失|生成侧损失]]

$$
\mathcal{L}_{\text{Gen}} = \mathbb{E}_{t,\mathbf{x},\boldsymbol{\epsilon},(H,W)} \Bigl[ \bigl\| \mathbf{v}_\theta(\mathbf{z}_t,t) - \mathbf{v}^\star \bigr\|_2^2 \Bigr]
$$

**含义**: 标准 [[Flow Matching|流匹配]] velocity MSE，但期望同时对 $(H,W)$ 取——分辨率本身是训练分布一部分。

**符号说明**:
- $\mathbf{v}^\star = (\mathbf{x} - \sigma_R\boldsymbol{\epsilon})/(1-t)\cdot t + \ldots$: 目标速度场（线性插值导出）
- $\mathbf{v}_\theta$: 网络预测的速度场

### 公式 6: [[Classifier-Free Guidance|CFG]] 多条件分离

$$
\tilde{\mathbf{v}}_\theta = \mathbf{v}_\theta(\varnothing) + w_t\bigl(\mathbf{v}_\theta(\mathbf{c}_t) - \mathbf{v}_\theta(\varnothing)\bigr) + w_i\bigl(\mathbf{v}_\theta(\mathbf{c}_i) - \mathbf{v}_\theta(\varnothing)\bigr)
$$

**含义**: 把文本条件 $\mathbf{c}_t$ 和图像条件 $\mathbf{c}_i$ 拆开独立做 [[Classifier-Free Guidance|CFG]]，分别调权——对编辑、interleaved 等多条件场景至关重要。

**符号说明**:
- $\varnothing$: 无条件（dropout）
- $w_t, w_i$: 文本/图像 guidance scale
- 训练时：文本 dropout 10%，全条件 dropout 10%

### 公式 7: 动态分辨率 warmup 采样概率

$$
\hat{p}_i = p_i \cdot \mathrm{clamp}\!\left(\frac{\min(e/E_{\text{warm}},\,1) - d_i}{\delta} + 1,\; 0,\; 1\right)
$$

**含义**: RL 阶段难度分桶——困难宽高比的样本在 warmup 早期被压低权重，随训练步数线性放开。

**符号说明**:
- $p_i$: 第 $i$ 个 aspect ratio 桶的基础采样概率
- $d_i$: 难度系数
- $e/E_{\text{warm}}$: 归一化训练进度
- $\delta$: 平滑窗宽

### 公式 8: OCR 准确率 reward（文本渲染 RL）

$$
r_{\text{ocr}} = \frac{\|\mathcal{C}(\hat{T}) \cap \mathcal{C}(T^*)\|}{\|\mathcal{C}(\hat{T}) \cup \mathcal{C}(T^*)\|}
$$

**含义**: 用 IoU 衡量生成图像中 OCR 出来的字符集合与目标文本的重合度——直接奖励文本可读性。

**符号说明**:
- $\hat{T}$: 生成图像 OCR 结果
- $T^*$: 目标文本
- $\mathcal{C}(\cdot)$: 字符集合（character set）

### 公式 9: 综合 reward

$$
r = r_{\text{ocr}} + \lambda_{\text{sty}} \cdot r_{\text{sty}}
$$

**含义**: 文本渲染 reward 与风格 reward 加权——避免模型为了 OCR 准确度牺牲视觉风格。

**符号说明**:
- $r_{\text{sty}}$: style reward（来自 aesthetic / style judge 模型）
- $\lambda_{\text{sty}}$: style 权重

---

## 关键图表

### Figure 1: 信息图 / 人物生成 showcase

![Figure 1](https://arxiv.org/html/2605.12500v1/x1.png)

**说明**: SenseNova-U1-8B-MoT 在 infographics 和 human generation 两类高难任务上的样例。展示模型在文字密集场景和人物结构生成上的能力。

### Figure 2: 图像编辑 / 交错生成 showcase

![Figure 2](https://arxiv.org/html/2605.12500v1/x2.png)

**说明**: 8B-MoT 的 image editing（局部修改）和 interleaved generation（图文交错输出）能力展示。强调模型作为统一接口可以同时承接编辑指令和多模态续写。

### Figure 3: SenseNova-U1 整体架构

![Figure 3](https://arxiv.org/html/2605.12500v1/x3.png)

**说明**: 顶层架构图——左侧 encoding interface（[[Near-Lossless Visual Interface]]）+ 中部 [[MoT]] 主干 + 右侧 decoding interface。强调"无独立 vision encoder、无 VAE 解码器"的 encoder-free 设计。

### Figure 4: NEO-unify 原生 pixel-text 范式

![Figure 4](https://arxiv.org/html/2605.12500v1/x4.png)

**说明**: 详细展示 [[NEO-unify]] 内部的 stream 路由——text token 走 understanding expert pool，image patch token 走 generation expert pool，但在 attention 层共享同一序列空间，实现 "perception 和 synthesis 在每层 native 交互"。

### Figure 5: Disaggregated 推理架构

![Figure 5](https://arxiv.org/html/2605.12500v1/x5.png)

**说明**: 推理时 understanding stream 跑在 **LightLLM**（LLM-friendly 引擎，含 KV cache 优化），generation stream 跑在 **LightX2V**（diffusion-friendly 引擎，含 step 蒸馏）。两个 engine 异步协作，最大化吞吐。

### Figure 6: [[Hybrid Attention|混合注意力]] mask 模式

![Figure 6](https://arxiv.org/html/2605.12500v1/x6.png)

**说明**: 统一 prefill 阶段的注意力 mask——text 行走 causal（下三角），image 行走 bidirectional（全可见），text/image 跨行按 prefix 可见性控制。实现时保留 pure text block 的 causal fast path 不变。

### Figure 7: Understanding 数据处理流水线

![Figure 7](https://arxiv.org/html/2605.12500v1/x7.png)

**说明**: 理解侧训练数据的清洗与构造流水线，含质量过滤、去重、语义分类等步骤。

### Figure 8: 训练数据分布旭日图

![Figure 8](https://arxiv.org/html/2605.12500v1/x8.png)

**说明**: 用 sunburst chart 展示训练数据的多维分布（task / domain / language / source），强调数据多样性。

### Figure 9: Generation 数据过滤流水线

![Figure 9](https://arxiv.org/html/2605.12500v1/assets/generation_filtering_pipeline.jpg)

**说明**: 生成侧训练数据的过滤流程，含美学打分、OCR 字符密度估计、aspect ratio 分桶等。

### Table 1: SenseNova-U1 变体架构配置

| 变体 | Stream 设计 | 总参数 | 激活参数 | Expert 数 |
|------|-----------|--------|---------|----------|
| SenseNova-U1-8B-MoT | Dense 对称 | 8B+8B | 16B | - |
| SenseNova-U1-A3B-MoT | Sparse 不对称 | 30B(U) + 8B(G) | ~3B | Und 128 / Gen 32 / Top-8 |

**说明**: A3B 用稀疏路由把推理算力压到 ~3B 激活，性能却追平甚至超过 8B-MoT。

### Table 2: 4 阶段训练 recipe

| Stage | 步数 | 数据 mix | LR | 关键操作 |
|-------|-----|---------|-----|---------|
| 1 Und Warmup | - | NEO 文本+视觉 | 2e-5 | attention-fusion |
| 2 Gen Pre-Train | 120+60+120K | Phase I/II/III | 2e-4 | 冻结 und stream |
| 3 Unified Mid | 84K | 33%U / 37%T2I / 24%Edit / 6%Inter | 2e-5 | 联合优化 |
| 4 Unified SFT | 9K | 指令对齐 | 2e-5 → 0 cosine | instruction tuning |

### Table 3: 多模态理解 benchmark（部分）

| Method | MMBench-EN | MMMU | OCRBench | VSI-Bench |
|--------|-----------|------|----------|-----------|
| Qwen3VL-8B | 87.50 | 74.10 | - | 56.61 |
| Qwen3.5-9B | - | - | 89.20 | - |
| **SenseNova-U1-8B-MoT** | **90.25** | 74.78 | 82.10 | **62.66** |
| **SenseNova-U1-A3B-MoT** | - | **80.55** | - | 56.90 |

**说明**: 8B-MoT 在多个理解 benchmark 上超过 Qwen3VL-8B，A3B 在 MMMU 上达到 80.55。

### Table 5: [[GenEval]] 文生图 benchmark

| Method | Overall |
|--------|---------|
| BAGEL | 0.82 |
| Qwen-Image | 0.87 |
| Lumina-DiMOO | 0.88 |
| **SenseNova-U1-8B / A3B** | **0.91** |

**说明**: 在 GenEval 上两个变体都达到 0.91，并列开源 SOTA。

### Table 6: [[DPG-Bench]] 文生图 benchmark

| Method | Overall | Best Global |
|--------|---------|------------|
| Qwen-Image | 88.32 | - |
| **SenseNova-U1-A3B-MoT** | **88.14** | - |
| **SenseNova-U1-8B-MoT** | - | **94.19** |

### Table 7: OneIG-EN（含文本渲染）

| Method | Overall | Text |
|--------|--------|------|
| Emu3.5 | **0.564** | - |
| **SenseNova-U1-8B-MoT** | 0.549 | **0.969** |

**说明**: 整体次于 Emu3.5，但 text rendering 子项 0.969 大幅领先——证明 OCR reward + style reward 这套 RL 后训练有效。

### Table 9-10: TIIF-Bench 短/长文本指令跟随

| Method | TIIF Short | TIIF Long |
|--------|-----------|-----------|
| Emu3.5 | 89.48 | 88.18 |
| **SenseNova-U1-8B-MoT** | **89.74** | **89.17** |

### Table 11-12: 中英文本密集生成

| Method | CVTG-2K | LongText-Bench-EN |
|--------|---------|-------------------|
| Qwen-Image | 0.829 | - |
| Seedream 4.5 | - | 0.989 |
| **SenseNova-U1-8B-MoT** | **0.940** | 0.979 |

### Table 4 / 8 / 13: 文本/中文/IGen benchmark（数据见原文）

- **Table 4**（IFEval/IFBench/tau2-Bench）: 8B 在 IFEval 91.13、IFBench 67.01，与 Qwen3.5-9B 91.50 近乎持平
- **Table 8**（OneIG-ZH）: 中文生成；
- **Table 13**（IGenBench）: 交错图文 benchmark；
- 详细数字见论文原表

---

## 实验

### 5.1 主结果

**Understanding（Table 3-4）**:
- 8B-MoT 在 MMBench-EN 上 90.25，超过 Qwen3VL-8B 的 87.50
- A3B-MoT 在 MMMU 上达 80.55，MMMU-Pro 72.83
- 空间推理（VSI-Bench / MindCube-Tiny）尤其强：8B 达 62.66，A3B 在 MindCube-Tiny 上 70.86
- Agentic tau2-Bench：8B 71.70 / A3B 75.39

**Generation（Table 5-13）**:
- GenEval 0.91（两个变体并列开源 SOTA）
- DPG-Bench overall 88.14（A3B），Global 94.19（8B）
- TIIF-Bench：89.74 (short) / 89.17 (long)，开源最佳
- 中文文本密集生成 CVTG-2K：0.940，碾压 Qwen-Image 0.829
- LongText-Bench-EN：0.979，逼近闭源 Seedream 4.5 (0.989)

### 5.2 消融研究

论文给出三条核心设计的 ablation 结论（数字保密较多）：

| 配置 | 影响 |
|------|------|
| w/o Native Encoder-Free（用 ViT+VAE） | 语义/像素表征被外部模块"压死"，理解和生成都掉点 |
| w/o Unified MoT（单 stream） | 理解和生成相互打架，下游 task 表现明显下降 |
| w/o Data-Scaling Joint | 数据 scaling 效率显著弱于 modular baseline |

**关键发现**: 三个组件（encoder-free + MoT + data joint scaling）互为必要——任意拿掉一个都会破坏 NEO-unify 的协同效应。

### 5.3 可视化

- **Interleaved generation**: OpenING / VBVR-Image / Uni-MMMU / RealUnify 上展示了图文交错连续输出能力——可生成"插图教程、视觉故事、PPT、海报、漫画、简历"等组合内容
- **失败模式**: 论文未明确列出，但从摘要看主要短板在 OneIG-EN overall（次于 Emu3.5）和极长文本（次于闭源 Seedream 4.5）

### 5.4 实现细节

- **Backbone**: 自研 MoT，从 NEO 初始化
- **优化器**: AdamW（推测，未明示）
- **训练 LR**: Stage 1/3/4 用 2e-5，Stage 2 用 2e-4
- **采样**: 训练用 flow matching velocity，推理用 100→8 步 [[DMD|DMD²]] 蒸馏
- **推理**: LightLLM + LightX2V disaggregated

### 5.5 VLA / 世界模型预备结果

论文最后一段提到 "Preliminary evidence demonstrates that our models extend beyond perception and generation, performing strongly in vision-language-action (VLA) and world model (WM) scenarios"——但**未给具体数字、未给基准、未给 task 设置**。这是文章对具身社区的 teaser，需后续 follow-up 验证。

---

## 批判性思考

### 优点

1. **架构干净**: encoder-free + MoT + flow matching 三件套自洽，工程语义清晰，是同期 unified MM 工作里少见的"全栈原生"。
2. **数据-架构联合优化**: 4 阶段训练 recipe + λ 权重调度反映出对 understanding/generation 不同收敛节奏的深刻理解，不是把两个 loss 一拍脑袋加起来。
3. **生成质量真到位**: GenEval 0.91、TIIF SOTA、文本渲染 0.969——证明 encoder-free 路线在生成端也能打。
4. **稀疏 MoT 思路前瞻**: A3B 用 understanding 128 expert + generation 32 expert 不对称设计，比对称 MoE 更符合"理解比生成更需要多样化知识"的直觉。
5. **可对接具身**: dual-stream 解耦后 understanding stream 天然适合做 VLA 指令解析，generation stream 天然适合做 WM rollout——比单 backbone 模型更容易接 action head（参见 [[BAGEL]] 派生 BagelVLA 路线）。

### 局限性

1. **没有 VLA / WM 硬指标**: "preliminary" 一句话带过——既然写到摘要里，应该至少给一个 task 的数字（哪怕是 LIBERO 简单任务），不然这是营销陈述
2. **A3B 路由细节缺失**: top-8 路由的 load balancing loss、expert collapse 检测、specialization 可视化都没给——38B 总参 ~3B 激活的稀疏架构最容易出问题的就是 expert 退化
3. **OneIG-EN 落后 Emu3.5**: 0.549 vs 0.564，差距虽小但 Emu3.5 是离散 token 路线，按 NEO-unify 的"连续表征更优"逻辑这里不该输
4. **训练算力没披露**: 4 阶段总步数加起来超过 400K，但没说 batch size、GPU 数、wall-clock。复现门槛极高
5. **OCR + style reward 设计简单**: $r = r_{\text{ocr}} + \lambda_{\text{sty}} r_{\text{sty}}$ 这种线性加权很容易 reward hacking——OCR 准但风格丑、风格美但 OCR 错都没明确防御
6. **VAE-free 但还是要 noise schedule**: 虽然不用 VAE，但 $\sigma_R$ 动态缩放本质上是把 latent diffusion 的 latent scaling 问题搬到了像素空间，并没有真正解决"how to schedule noise across resolutions"

### 潜在改进方向

1. **接 action head 做 VLA 实测**: 把 generation stream 改成 action token，按 [[Diffusion Policy]] / [[π₀.₅]] 模式接 robot；理论上 NEO-unify 是当前最适合接具身的统一架构
2. **稀疏路由 + 模态特化**: A3B 可以做更深的 expert specialization 分析——是否 understanding 128 expert 学到了不同领域知识（数学/代码/对话/视觉）
3. **Reward 模型升级**: OCR + style 两个 reward 都偏低维，可以学 [[GRPO]] / [[CM-GRPO]] 路线引入 preference reward
4. **更紧的 inference engine**: LightLLM + LightX2V 异步通信是 workable 但不是最优，可以做 unified KV cache 让 understanding 和 generation 共享
5. **加 video 模态**: 当前是图文统一，加上视频后可以直接做 [[World Model]]——这是 SenseNova-U2/U3 自然演进路线

### 可复现性评估

- [x] 代码开源（GitHub: OpenSenseNova/SenseNova-U1）
- [ ] 预训练模型（截至 paper 发布未确认）
- [ ] 训练细节完整（算力、batch size 未披露）
- [ ] 数据集可获取（trained on 内部数据 + 部分公开）

---

## 关联笔记

### 基于

- [[NEO-unify]]: 本文核心架构范式
- [[MoT]]: backbone 设计的直接前驱
- [[Flow Matching]]: 生成侧损失基础
- [[MoE]]: A3B sparse 变体的对照范式

### 对比

- [[BAGEL]]: ByteDance 同方向工作，dual-stream MoE 而非 MoT
- [[OmniGen2]]: 同期 encoder-based 统一模型
- [[Qwen-Image]]: 同期 T2I 强 baseline
- [[FLUX]]: encoder-based 生成侧 baseline
- Emu3.5: 离散 token 路线对照
- Lumina-DiMOO: 同期 unified MM

### 方法相关

- [[Near-Lossless Visual Interface]]: 视觉接口（本文新概念）
- [[Hybrid Attention]]: 注意力 mask 设计（本文新概念）
- [[Classifier-Free Guidance|CFG]]: 多条件分离 guidance
- [[DMD]]: 推理 step 蒸馏
- [[GenEval]]、[[DPG-Bench]]: 生成评测

### 具身延伸（潜在）

- [[VLA]]、[[World Model]]、[[WAM-Survey]]: 论文 teaser 提到的下游方向
- [[Unified Multimodal Model]]: 概念归属

---

## 速查卡片

> [!summary] SenseNova-U1
> - **核心**: encoder-free + [[MoT]] + pixel-space [[Flow Matching]] 的原生统一多模态模型
> - **方法**: Near-Lossless Visual Interface（2 层卷积 32× 压缩）+ MoT 双 stream（参数解耦/attention 共享）+ Hybrid Attention（text causal / image bidirectional）+ 4 阶段联合训练 + DMD² 蒸馏
> - **结果**: GenEval 0.91（开源 SOTA）/ TIIF 89.74 / 文本渲染 0.969 / MMBench-EN 90.25；A3B 用 ~3B 激活逼近 8B 性能
> - **代码**: <https://github.com/OpenSenseNova/SenseNova-U1>
> - **关键创新点**: 不用预训练 vision encoder、不用 VAE 解码器，把所有视觉表征都内化进 backbone

---

*笔记创建时间: 2026-05-24*
