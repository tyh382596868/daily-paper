---
title: "Unified Vision-Language-Action Model"
method_name: "UniVLA"
authors: [Yuqi Wang, Xinghang Li, Wenxuan Wang, Junbo Zhang, Yingyan Li, Yuntao Chen, Xinlong Wang, Zhaoxiang Zhang]
year: 2025
venue: ICLR 2026
tags: [vla, autoregressive, discrete-tokens, world-model, action-tokenization, robot-manipulation, video-generation]
zotero_collection: 3-Robotics/VLA
image_source: local
arxiv_html: https://arxiv.org/html/2506.19850v1
created: 2026-06-10
---

# 论文笔记：Unified Vision-Language-Action Model

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | BAAI（北京智源人工智能研究院）、CASIA、清华大学 |
| 日期 | June 2025 |
| 项目主页 | https://robertwyq.github.io/univla.github.io |
| 对比基线 | [[pi0-FAST]], [[OpenVLA]], [[RoboVLMs]] |
| 链接 | [arXiv](https://arxiv.org/abs/2506.19850) / [Code](https://github.com/baaivision/UniVLA) |

---

## 一句话总结

> UniVLA 将视觉、语言和动作统一表示为离散 token 序列，通过自回归建模与世界模型后训练，在 CALVIN、LIBERO、SimplerEnv 等基准上刷新 SOTA，同时扩展到真实机器人和自动驾驶场景。

---

## 核心贡献

1. **原生统一多模态 VLA**: 首次将视觉、语言、动作三种模态完全统一为离散 token 在同一个 [[Autoregressive Transformer]] 中联合建模，无需额外视觉编码器。
2. **世界模型后训练**: 在下游策略学习前引入视频世界模型后训练（post-training），让模型从大规模视频数据中学习时序因果动态，显著提升长时程任务的迁移能力。
3. **FAST 动作分词器集成**: 采用 [[FAST Action Tokenizer]]（基于 DCT 的频域离散化），将连续机器人动作序列压缩为紧凑离散 token，克服了 OpenVLA 式均匀 bin 方法的精度限制。
4. **Interleaved 视频-动作训练**: 在 [[MDP]] 框架下将观测帧与动作 token 交织排列，实现视频生成、感知接地、策略学习多任务共训。
5. **SOTA 性能**: LIBERO 平均成功率 95.5%（vs. π₀-FAST 85.5%），CALVIN ABCD→D 4.63，SimplerEnv-Bridge 69.8%。

---

## 问题背景

### 要解决的问题

现有 [[VLA]] 方法（如 [[OpenVLA]]、[[π₀]]、[[RT-2]]）普遍依赖独立视觉编码器提取图像特征，再通过语言模型骨干生成动作输出。这一范式存在两个核心问题：
1. 视觉与语言模态之间的表示空间异构，限制了跨模态深度融合；
2. 无法直接利用无动作标注的大规模视频数据进行预训练，数据利用率低。

### 现有方法的局限

- **连续动作头**（如 [[Diffusion Policy]]、π₀）：动作生成能力强，但视觉-语言-动作三模态未真正统一；
- **离散 token VLA**（如 [[OpenVLA]]）：均匀 bin 分词精度不足，无法捕捉动作时序压缩结构；
- **视频生成预训练**（如 [[UniSim]]）：未与下游策略学习形成有效衔接。

### 本文的动机

将视觉、语言、动作统一表示为离散 token 后，整个 VLA 任务可被归约为**单一序列建模问题**，天然支持从互联网规模视频数据中学习时序动态，然后迁移到动作预测，形成"视频预训练 → 世界模型后训练 → 策略微调"的三阶段流水线。

---

## 方法详解

### 模型架构

![[UniVLA_fig1_overview.png]]

**UniVLA 概览图**：单一 [[Autoregressive Transformer]] 骨干（基于 [[Emu3-MoE]]），共享词表统一处理三种模态。

UniVLA 采用 **Encoder-Free Autoregressive** 架构：
- **输入**: 语言指令 $l$ + 观测 token 序列 $v_{1:T}$ + 动作 token 序列 $a_{1:T}$
- **Backbone**: Emu3-MoE（32 层 Transformer，hidden size 4096，32 heads，GQA with 8 KV heads）
- **核心模块**: [[VQ Tokenizer]] 用于视觉离散化；[[FAST Action Tokenizer]] 用于动作离散化
- **输出**: 自回归预测下一个 token（视觉 token 用于世界建模，动作 token 用于策略）
- **总参数**: ~8B（Emu3-MoE 量级）
- **词表大小**: 184,622（含视觉 codebook 32,768 个 visual token）

### 核心模块

#### 模块 1: 统一 Token 序列构建

**设计动机**: 利用 [[Discrete Tokenization]] 将三种模态统一到同一词表空间，消除模态隔阂。

**具体实现**（Interleaved 视频格式）：

单条训练样本的序列结构：

```
[BOS] 语言指令 | [BOI] 视觉frame_1 [EOI] | 动作chunk_1 | [BOI] 视觉frame_2 [EOI] | 动作chunk_2 | ... [EOS]
```

- 每帧图像通过 [[VQ-VAE]]（Emu3-VisionTokenizer，codebook 大小 32,768）编码为约 1024 个 visual token
- 每个动作块（10 步）通过 [[FAST Action Tokenizer]] 编码为固定长度离散 token 序列
- 语言指令直接使用 Emu3Tokenizer 文本 token

**帧率感知采样**：对不同数据集使用不同采样率（`calvin:10fps`、`droid:15fps`、`toto:20fps`），保证时序一致性。

#### 模块 2: FAST 动作分词器

**设计动机**: 利用 [[DCT（离散余弦变换）]] 对动作轨迹进行频域压缩，保留动作的时序相关性，比逐维度 bin 分词信息更密集。

**具体实现**:
- 输入: 连续动作序列 $a \in \mathbb{R}^{T \times D}$（$T=10$ 步，$D=7$ 维度含 gripper）
- 对每个动作维度做 DCT 变换，提取低频系数
- 通过 Byte-Pair Encoding（BPE）进一步压缩，生成紧凑 token 序列
- 词表映射: 动作 token ID $= \text{pad\_token\_id} - 1 - \text{discretized\_index}$（复用词表末尾）

#### 模块 3: 世界模型后训练

**设计动机**: 在无动作标注的大规模机器人视频上进行视频预测训练，让模型学习环境的时序因果动态，为下游策略提供更强先验。

**具体实现**:
- 数据集: OXE（RT-1、BridgeV2、DROID、TOTO 等）+ CMU、UT Austin 等开放数据集，约 13 个数据集混合
- 训练目标: 仅在视觉 token 上计算交叉熵损失（`apply_loss_on_only_vision=True`）
- 输入: 帧数 6，动作步数 5（仅视觉，不含动作标注）
- 结果权重: [world model pretraining ckpts (HuggingFace)](https://huggingface.co/Yuqi1997/UniVLA/tree/main/WORLD_MODEL_POSTTRAIN)

#### 模块 4: 策略微调（SFT）

基于世界模型后训练权重，在各任务数据集上进行 Supervised Fine-Tuning：
- 训练目标: 仅在动作 token 上计算交叉熵损失（`apply_loss_on_only_action=True`）
- 输入帧数: 2，动作步数: 10，max_position_embeddings: 1700
- FAST 分词器按数据集独立 fit（如 `fast_calvin_norm_a10_s50`）

---

## 关键公式

### 公式 1: [[Autoregressive Modeling|自回归语言模型损失]]

$$
\mathcal{L}_{AR} = - \sum_{t=1}^{N} \log P_\theta(x_t \mid x_{<t})
$$

**含义**: 对 token 序列 $x = (x_1, \ldots, x_N)$（含语言、视觉、动作 token），最大化每步条件概率的乘积。

**符号说明**:
- $x_t$: 当前步 token（可为语言/视觉/动作任意 token）
- $x_{<t}$: 历史 token 上下文
- $\theta$: Emu3-MoE 模型参数

### 公式 2: [[VQ-VAE|视觉 VQ 编码]]

$$
z_q = \text{Quantize}(z_e) = \arg\min_{e_k \in \mathcal{C}} \| z_e - e_k \|_2
$$

**含义**: 将连续图像特征 $z_e$ 映射到最近的 codebook 向量 $e_k$，得到离散 visual token。

**符号说明**:
- $z_e$: 图像编码器输出的连续特征
- $\mathcal{C} = \{e_1, \ldots, e_{32768}\}$: 视觉 codebook（大小 32,768）
- $z_q$: 量化后的离散表示

### 公式 3: [[FAST Action Tokenizer|FAST 动作离散化]]

$$
\hat{a} = \text{DCT}(a_{1:T}), \quad \hat{a}_{quant} = \text{BPE}(\text{Quantize}(\hat{a}))
$$

**含义**: 先对动作序列做 [[DCT（离散余弦变换）]]，再对频域系数量化后做 BPE 编码，得到紧凑动作 token 序列。

**符号说明**:
- $a_{1:T}$: $T$ 步连续动作序列，$T=10$，维度 $D=7$
- $\hat{a}$: DCT 系数（频域表示）
- $\hat{a}_{quant}$: 量化+BPE 后的离散动作 token

### 公式 4: [[Action Chunking|动作 token 映射]]

$$
\text{token\_id}(a) = \text{pad\_token\_id} - 1 - \text{digitize}(a, \text{bins})
$$

**含义**: 将连续动作值映射到词表末尾预留的离散 token 区域（OpenVLA 风格的简单 bin 方法，用于非 FAST 场景）。

**符号说明**:
- $\text{bins}$: 在 $[-1, 1]$ 均匀划分的 256 个 bin
- $\text{pad\_token\_id}$: 151,643（Emu3 词表的 pad token）
- 动作 token 实际占用词表末尾 257 个位置

---

## 关键图表

### Figure 1: UniVLA 系统概览

![[UniVLA_fig1_overview.png]]

**说明**: UniVLA 的整体框架。语言指令、视觉帧（经 [[VQ-VAE]] 编码）和动作（经 [[FAST Action Tokenizer]] 编码）被统一为离散 token 序列，由单一 Emu3-MoE [[Autoregressive Transformer]] 进行联合建模，支持三大任务：视觉感知接地（text-supervised）、世界建模/视频生成（vision-supervised）、策略学习（action-supervised）。

### Table 1: CALVIN ABCD→D 基准对比

| Method | Avg. Len. (↑) |
|--------|--------------|
| SuSIE | 3.27 |
| RoboFlamingo | 3.89 |
| GR-1 | 3.94 |
| Spatiotemporal Predictive Pre-Training | 4.12 |
| DynaMo | 4.15 |
| **UniVLA (video sft)** | **4.63** |
| UniVLA (5× steps) | 4.71 |

**说明**: CALVIN ABCD→D 设置，UniVLA 以 4.63 平均成功步数超越所有先前方法，5× 推理步数（180 步）可进一步提升到 4.71。

### Table 2: LIBERO 基准对比

| Method | SPATIAL | OBJECTS | GOAL | LONG (10) | AVG |
|--------|---------|---------|------|-----------|-----|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π₀-FAST | 93.4 | 94.2 | 87.6 | 66.8 | 85.5 |
| **UniVLA (img sft)** | **97.0** | **99.0** | **92.6** | **90.8** | **94.8** |
| **UniVLA (video sft)** | **95.4** | **98.8** | **93.6** | **94.0** | **95.5** |

**说明**: 在 LIBERO 四个任务套件上，UniVLA 全面超越 π₀-FAST（+10pp AVG），尤其在长时程任务（LIBERO-Long）上优势明显（94.0 vs. 66.8）。

### Figure 2: LIBERO 可视化结果

![[UniVLA_fig_libero.png]]

**说明**: LIBERO 基准上的可视化操作序列，展示 UniVLA 在不同任务（SPATIAL/OBJECTS/GOAL/LONG）中的精细操作能力。

### Table 3: SimplerEnv-Bridge (WidowX) 对比

| Method | Put Spoon | Put Carrot | Stack Block | Put Eggplant | AVG |
|--------|-----------|------------|-------------|--------------|-----|
| Octo-Base | 17.0 | 21.2 | 6.7 | 55.8 | 25.2 |
| OpenVLA | 37.5 | 34.2 | 4.2 | 55.0 | 32.7 |
| Previous SOTA | — | — | — | — | 42.7 |
| **UniVLA (video sft)** | **83.3** | **66.7** | **33.3** | **95.8** | **69.8** |

**说明**: SimplerEnv Bridge-WidowX 设置，UniVLA 将平均成功率从 42.7% 大幅提升至 69.8%（+27.1pp），尤其在 Put Spoon 和 Put Eggplant 任务上接近饱和。

### Figure 3: CALVIN 可视化结果

![[UniVLA_fig_calvin.png]]

**说明**: CALVIN 长时程任务（ABCD→D）的连续操作序列，展示 UniVLA 对 4+ 步指令的执行能力。

### Figure 4: SimplerEnv 可视化结果

![[UniVLA_fig_simpler.png]]

**说明**: SimplerEnv-Bridge WidowX 机器人上的操作可视化，包括放置勺子、胡萝卜、叠块和放茄子等任务。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| OXE (RT-1, BridgeV2, DROID 等) | 数百万帧 | 多机器人、多场景、无动作标注 | 世界模型后训练 |
| CALVIN ABCD | ~24K 演示 | 长时程桌面操作，4 个环境 | 策略微调 |
| LIBERO (4 suites) | ~8.8K 演示 | 多任务/多领域机器人学习 | 策略微调 |
| SimplerEnv-Bridge | — | 基于真实 WidowX 轨迹 | 策略微调+评估 |
| ALOHA (real) | — | 真实双臂机器人 | 实机验证 |

### 实现细节

- **Backbone**: Emu3-MoE（基于 Emu3-Base，vocab 184,622，GQA 架构）
- **视觉分词器**: Emu3-VisionTokenizer（VQ-VAE，codebook 32,768）
- **动作分词器**: FAST（DCT + BPE，`fast_calvin_norm_a10_s50` 等）
- **优化器**: AdamW（β₁=0.9，β₂=0.95，ε=1e-6）
- **学习率**: 8e-5（cosine decay，min 5e-6），warmup 50 steps
- **世界模型训练**: bs=1/GPU × 8 GPU × 2 grad_accum = bs16，30K steps，max_len=6400
- **策略微调 (CALVIN)**: bs=6/GPU × 8 GPU × 4 grad_accum = bs192，8K steps，max_len=1700
- **策略微调 (LIBERO)**: 类似设置，8K steps
- **硬件**: 8× GPU（训练），DeepSpeed ZeRO-3 + offload，Flash Attention 2，BF16
- **精度**: BF16 + TF32

### 可视化结果

世界模型后训练赋予 UniVLA 视频预测（next-frame prediction）能力，在 LIBERO-Long 等长时程任务上优势尤为明显，说明因果动态建模对复杂任务的规划有实质帮助。

---

## 批判性思考

### 优点

1. **范式简洁统一**: 单一 [[Autoregressive Transformer]] 无需额外组件即可处理三种模态，工程实现清晰。
2. **数据飞轮**: 将无标注视频（互联网规模）纳入训练体系，突破了有标注机器人数据瓶颈。
3. **SOTA 全面**: 在 CALVIN、LIBERO、SimplerEnv 三大主流 benchmark 同时刷新记录，泛化性强。
4. **代码开源**: 完整训练/评估流程公开，可复现性高。
5. **扩展性**: 框架已扩展到自动驾驶（DriveVLA-W0），证明跨域迁移潜力。

### 局限性

1. **推理速度**: 自回归 token 生成（尤其含视觉 token）在实机上的延迟问题未深入讨论，1024 个视觉 token × 多帧 = 较长序列。
2. **动作精度上限**: 离散 token 方案相比连续动作头（diffusion/flow matching）在精细操作（如微米级装配）上仍有精度上限。
3. **世界模型质量评估缺失**: 后训练阶段仅从下游任务间接评估世界模型质量，缺少直接的视频生成质量评测。
4. **高分辨率扩展性**: 当前 image_area=262144（512×512）固定分辨率，对高分辨率/多相机场景扩展受限。

### 潜在改进方向

1. 引入 [[Streaming Inference]] 流式推理，参考 FASTER 思路降低实机延迟。
2. 将视觉 token 替换为连续表示（Latent Diffusion）兼顾生成质量与推理速度。
3. 探索多相机/立体视觉输入以提升 3D 空间理解能力。
4. 扩展到 [[Loco-manipulation]] 全身控制场景。

### 可复现性评估

- [x] 代码开源（GitHub: baaivision/UniVLA）
- [x] 预训练模型（HuggingFace: Yuqi1997/UniVLA）
- [x] 训练细节完整（训练脚本全部公开）
- [x] 数据集可获取（CALVIN/LIBERO/SimplerEnv 均开放）

---

## 关联笔记

### 基于

- [[Emu3]]: 底层 VQ-VisionTokenizer 和 MoE 骨干架构
- [[FAST Action Tokenizer]]: DCT 动作频域分词器
- [[OpenVLA]]: 评估框架 RoboVLMs 部分参考

### 对比

- [[pi0-FAST]]: LIBERO 主要对比方法，UniVLA 全面超越（95.5 vs. 85.5）
- [[OpenVLA]]: 同为 token-based VLA，UniVLA 架构更统一
- [[GR-1]]: CALVIN 对比方法（4.63 vs. 3.94）

### 方法相关

- [[Autoregressive Transformer]]: 核心架构
- [[VQ-VAE]]: 视觉离散化基础
- [[FAST Action Tokenizer]]: 动作离散化
- [[DCT（离散余弦变换）]]: FAST 的核心算子
- [[Action Chunking]]: 10 步动作块预测
- [[世界模型]]: 后训练阶段核心任务
- [[MDP]]: Interleaved 训练的理论框架

### 硬件/数据相关

- [[ALOHA]]: 真实双臂机器人实验平台
- [[CALVIN]]: 长时程操作 benchmark
- [[LIBERO]]: 多任务机器人学习 benchmark
- [[SimplerEnv]]: 基于真实数据的仿真评估

---

## 速查卡片

> [!summary] UniVLA: Unified Vision-Language-Action Model
> - **核心**: 将视觉/语言/动作统一为离散 token，单一自回归模型端到端建模
> - **方法**: Emu3-MoE 骨干 + VQ 视觉分词 + FAST 动作分词 + 世界模型后训练
> - **结果**: CALVIN 4.63 / LIBERO 95.5% / SimplerEnv 69.8%，全面 SOTA
> - **代码**: https://github.com/baaivision/UniVLA

---

*笔记创建时间: 2026-06-10*
