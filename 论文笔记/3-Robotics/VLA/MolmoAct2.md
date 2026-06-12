---
title: "MolmoAct2: Action Reasoning Models for Real-world Deployment"
method_name: "MolmoAct2"
authors: [Haoquan Fang, Jiafei Duan, Donovan Clay, Sam Wang, Shuo Liu, Weikai Huang, Xiang Fan, Wei-Chuan Tsai, Shirui Chen, Yi Ru Wang, Shanli Xing, Jaemin Cho, Jae Sung Park, Ainaz Eftekhar, Peter Sushko, Karen Farley, Angad Wadhwa, Cole Harrison, Winson Han, Ying-Chun Lee, Eli VanderBilt, Rose Hendrix, Suveen Ellawela, Lucas Ngoo, Joyce Chai, Zhongzheng Ren, Ali Farhadi, Dieter Fox, Ranjay Krishna]
year: 2026
venue: arXiv cs.RO
tags: [vla, flow-matching, embodied-reasoning, action-tokenizer, bimanual-manipulation, robot-learning, open-source]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.02881
created: 2026-05-09
---

# 论文笔记：MolmoAct2: Action Reasoning Models for Real-world Deployment

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | University of Washington, Allen Institute for AI |
| 日期 | May 2026 |
| 项目主页 | [allenai.org/blog/molmoact2](https://allenai.org/blog/molmoact2) |
| 对比基线 | [[pi0.5]]、[[GR00T N1.7]]、[[π₀-DROID]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.02881) |

---

## 一句话总结

> MolmoAct2 是一个完全开源的 [[VLA|视觉-语言-动作模型]]，通过专化具身推理骨干、逐层 KV 连接和自适应深度推理，实现了超越闭源模型的真实机器人部署性能。

---

## 核心贡献

1. **Molmo2-ER 具身推理骨干**: 在 330 万具身样本上专化训练，13 个具身推理基准平均 63.8%，超越 GPT-5 和 Gemini Robotics ER-1.5
2. **OpenFAST 开源动作分词器**: 在 100 万动作序列上训练，支持 5 种机器人形态，2048 词汇量的离散动作 token
3. **三个新开源数据集**: 包含目前最大的开源双臂数据集 MolmoAct2-BimanualYAM（720 小时/3.45 万条演示）
4. **MolmoAct2-Think 自适应深度推理**: 利用时序冗余跳过静态区域的深度 token 预测，在提升性能的同时降低推理延迟
5. **逐层 KV 连接架构**: 动作专家每层直接访问对应深度 VLM 层的 KV 缓存，而非单一交叉注意力

---

## 问题背景

### 要解决的问题

现有 [[VLA|视觉-语言-动作模型]] 面临三大部署障碍：前沿模型闭源、开源模型绑定昂贵硬件、缺乏针对具身推理的专化视觉-语言骨干。

### 现有方法的局限

- [[π₀]]、[[π₀.₅]] 等高性能 VLA 模型权重不开放或训练数据不公开
- 开源模型（如 [[OpenVLA]]）通常与特定商用机器人平台深度绑定
- 通用 VLM 骨干在空间推理、自我-外视角对应等具身核心能力上存在明显短板
- 现有系统在低成本平台（SO-100/101）上的开箱即用性能极差

### 本文的动机

通过构建完全开放的生态系统（权重+代码+数据），将高性能 VLA 推进到低成本、可访问的机器人平台上，同时通过深度感知推理提升空间理解能力。

---

## 方法详解

### 模型架构

MolmoAct2 采用 **VLM + 连续动作专家（[[Flow Matching|流匹配]]）** 双模块架构：

- **输入**: 语言指令 $l$ + 多摄像头 RGB 观测 $o_t$ + 机器人关节状态 $s_t$ + 深度图
- **Backbone**: [[Molmo2]]-ER（具身推理专化的 VLM）
- **核心模块**: [[Flow Matching|流匹配]]动作专家，通过[[逐层 KV 连接|Per-Layer KV Connection]]访问 VLM 层级特征
- **动作分词**: [[OpenFAST]] 将连续轨迹转为 2048 词汇的离散 token
- **输出**: [[Action Chunking|动作块]] $a_{t:t+H}$（1 秒轨迹）

### 核心模块

#### 模块 1：Molmo2-ER — 具身推理骨干

**设计动机**: 利用[[具身推理|Embodied Reasoning]]专化训练，弥补通用 VLM 在空间感知、像素级定位方面的不足。

**训练数据（3.3M 样本）**：
- 图像具身 QA：1.33M 条
- 视频具身 QA：703K 条
- 像素级指点与目标检测：780K + 100K 条
- 多图/自我-外视角对应：700K 条
- 抽象推理：150K 条

**训练策略（两阶段"专化-复习"）**：
- Stage 1：20K 步专注于具身数据 + 8% Tulu-3 文本
- Stage 2：1.5K 步以 50% 具身 + 50% 原始 Molmo2 数据联合微调，防止灾难性遗忘

#### 模块 2：OpenFAST — 开源动作分词器

**设计动机**: 将异构机器人连续轨迹统一转化为离散 token，使语言模型自回归监督成为可能。

**具体实现**：
1. 将 1 秒动作序列做[[频域变换|Frequency-Domain Transform]]
2. 系数量化 → [[字节对编码|Byte-Pair Encoding]] 得到 2048-token 词汇表
3. 各维度用 1-99 百分位统计归一化；夹爪指令单独处理
4. 所有形态统一 padding 到 32 维连续动作

训练混合（100 万序列，5 种形态）：

| 数据集 | 比例 | 机器人 | 动作表示 |
|--------|------|--------|----------|
| MolmoAct2-BimanualYAM | 30% | YAM | 绝对关节角 |
| MolmoAct2-SO100/101 | 30% | SO-100/101 | 绝对关节角 |
| MolmoAct2-DROID | 30% | Franka | 绝对关节角 |
| Fractal | 3.33% | Google Robot | 增量末端执行器 |
| BC-Z | 3.33% | Google Robot | 增量末端执行器 |
| Bridge | 3.33% | WidowX | 增量末端执行器 |

#### 模块 3：逐层 KV 连接（Per-Layer KV Connection）

**设计动机**: 相比标准单层[[交叉注意力|Cross-Attention]]，允许动作专家在每个深度层级直接接触对应 VLM 层的视觉-语言注意力状态，提供更丰富的层级特征。

**架构**（DiT 风格动作专家块，共 36 层）：

每个专家层依次执行：
- [[自注意力|Self-Attention]]（AdaRMSNorm 调制）
- [[交叉注意力|Cross-Attention]]（以对应 VLM 层 KV 为键值）
- MLP（AdaRMSNorm 调制）

VLM KV 通过投影矩阵对齐维度后注入专家对应层（见公式 6-7）。

#### 模块 4：MolmoAct2-Think — 自适应深度推理

**设计动机**: 机器人轨迹中存在大量时序冗余，大多数帧的场景几乎不变，逐帧重新预测全部深度 token 浪费计算。

**具体实现**：
- 将深度图量化为 10×10 网格（100 个空间 code），学习 128 个深度码值
- 计算相邻帧深度嵌入的余弦相似度，低于 0.996 阈值的区域触发重新预测（公式 10）
- 通过逐层门控机制（公式 11-12）动态调整深度 token 对 VLM KV 的影响权重
- 维护深度 token 缓冲区：变化区域更新，静态区域复用

---

## 关键公式

### 公式 1: [[Flow Matching|流匹配插值]]

$$
x_t = (1-t)\varepsilon + ta, \quad u = a - \varepsilon
$$

**含义**: 在噪声 $\varepsilon$ 和真实动作 $a$ 之间定义线性插值路径，速度场 $u$ 即为两者之差。

**符号说明**:
- $t \sim \mathcal{U}(0, 1)$: 流时间步，均匀采样
- $\varepsilon \sim \mathcal{N}(0, I)$: 标准高斯噪声
- $a$: 目标动作轨迹
- $x_t$: $t$ 时刻的带噪动作
- $u$: 目标速度场

### 公式 2: [[Flow Matching|流匹配损失]]

$$
\mathcal{L}_{\text{flow}} = \mathbb{E}_{a,\varepsilon,t}\left[\|m \odot (f_\theta(x_t, t, c) - u)\|_2^2\right]
$$

**含义**: 训练动作专家 $f_\theta$ 预测从带噪动作到目标动作的速度场，$m$ 屏蔽填充时间步和维度。

**符号说明**:
- $f_\theta$: 动作专家网络
- $c$: VLM 上下文（视觉-语言特征）
- $m$: 填充掩码
- $\odot$: 逐元素乘法

### 公式 3-5: DiT 风格动作专家块

$$
h'_\ell = h_\ell + g^{sa}_\ell \cdot \text{SA}(\text{AdaRMS}^{sa}_\ell(h_\ell, t))
$$

$$
\bar{h}_\ell = h'_\ell + g^{ca}_\ell \cdot \text{CA}(\text{AdaRMS}^{ca}_\ell(h'_\ell, t),\ \tilde{K}_\ell,\ \tilde{V}_\ell)
$$

$$
h_{\ell+1} = \bar{h}_\ell + g^{ff}_\ell \cdot \text{MLP}(\text{AdaRMS}^{ff}_\ell(\bar{h}_\ell, t))
$$

**含义**: 每个动作专家层依次执行自注意力、对 VLM 第 $\ell$ 层 KV 的交叉注意力、前馈网络，均以流时间步 $t$ 为条件调制。

**符号说明**:
- $h_\ell$: 第 $\ell$ 层动作专家隐状态
- $g^{sa}_\ell, g^{ca}_\ell, g^{ff}_\ell$: AdaRMSNorm 门控系数
- $\tilde{K}_\ell, \tilde{V}_\ell$: 投影后的 VLM 第 $\ell$ 层键/值

### 公式 6: [[逐层 KV 连接|Per-Layer KV Projection]]

$$
\tilde{K}_\ell = \text{reshape}(P_K K^\text{vlm}_\ell), \quad \tilde{V}_\ell = \text{reshape}(P_V V^\text{vlm}_\ell)
$$

**含义**: 用可学习投影矩阵将 VLM 第 $\ell$ 层的 KV 缓存对齐到动作专家的交叉注意力头数和维度。

**符号说明**:
- $P_K, P_V$: 可学习线性投影，每层共享参数
- $K^\text{vlm}_\ell, V^\text{vlm}_\ell$: VLM 第 $\ell$ 层的键/值矩阵

### 公式 7: [[交叉注意力|Cross-Attention 计算]]

$$
\text{CA}(Q_\ell, \tilde{K}_\ell, \tilde{V}_\ell) = \text{softmax}\!\left(\frac{Q_\ell \tilde{K}_\ell}{\sqrt{d_h}}\right) \tilde{V}_\ell
$$

**含义**: 动作专家用自身查询向量 $Q_\ell$ 对对应 VLM 层的 KV 做缩放点积注意力。

**符号说明**:
- $Q_\ell$: 动作专家第 $\ell$ 层查询
- $d_h$: 注意力头维度

### 公式 8: [[Flow Matching|多流样本损失]]

$$
\mathcal{L}_{\text{flow}}(a, c) = \frac{1}{K} \sum_{i=1}^{K} \left\| m \odot \left(f_\theta(x_{t_i}, t_i, c) - (a - \varepsilon_i)\right) \right\|_2^2
$$

**含义**: 对同一动作块在 $K$ 个不同时间步采样，提供更密集的轨迹监督信号。后训练 $K=4$，微调 $K=8$。

**符号说明**:
- $K$: 每动作块的流样本数
- $t_i, \varepsilon_i$: 第 $i$ 个采样的时间步和噪声

### 公式 9: [[联合训练目标|Post-Training Combined Loss]]

$$
\mathcal{L}_{\text{post}} = \mathcal{L}_{\text{LM}} + \mathcal{L}_{\text{flow}}
$$

**含义**: 后训练阶段同时优化离散动作 token 的语言模型交叉熵损失和连续轨迹的流匹配损失。

### 公式 10: [[MolmoAct2-Think|自适应深度更新掩码]]

$$
m_{t,i} = \mathbb{1}\!\left[\cos(x_{t,i},\, x_{t-1,i}) < 0.996\right]
$$

$$
b_{t,i} = \begin{cases} d_{t,i}, & \text{if } m_{t,i} = 1 \\ b_{t-1,i}, & \text{if } m_{t,i} = 0 \end{cases}
$$

**含义**: 计算深度嵌入的余弦相似度，相似度低（场景变化）则重新预测深度 token，否则从缓冲区复用。

**符号说明**:
- $x_{t,i}$: 第 $t$ 帧第 $i$ 个空间格的深度嵌入
- $b_{t,i}$: 深度 token 缓冲区
- $d_{t,i}$: 新预测的深度码

### 公式 11-12: [[MolmoAct2-Think|逐层深度门控]]

$$
c_\ell = \frac{\sum_t A_t (1 - M_t) V^\text{vlm}_{\ell,t}}{\sum_t A_t (1 - M_t)}, \quad g_\ell = \sigma(w_\ell c_\ell + b_\ell)
$$

$$
\bar{K}^\text{vlm}_{\ell,t} = (1 - M_t + M_t g_\ell)\, K^\text{vlm}_{\ell,t}, \quad \bar{V}^\text{vlm}_{\ell,t} = (1 - M_t + M_t g_\ell)\, V^\text{vlm}_{\ell,t}
$$

**含义**: 以非深度 token 的注意力加权值计算层级深度上下文 $c_\ell$，通过 sigmoid 门控 $g_\ell$ 动态调制深度 token 对 VLM KV 的贡献，初始偏置 $b_\ell = -4$ 使门控偏向关闭。

**符号说明**:
- $M_t$: 深度 token 位置掩码（1 为深度 token）
- $A_t$: 注意力权重
- $g_\ell$: 第 $\ell$ 层深度门控系数 $\in (0, 1)$

---

## 关键图表

### Figure 1: MolmoAct2 系统概览

![Figure 1](https://arxiv.org/html/2605.02881v1/x5.png)

**说明**: MolmoAct2 的整体架构。输入图像观测、语言指令和机器人状态，经由 Molmo2-ER 骨干提取特征，通过[[逐层 KV 连接|Per-Layer KV Connection]]传递给[[Flow Matching|流匹配]]动作专家生成连续轨迹，同时支持 OpenFAST 离散 token 路径。系统完全开源并支持多种低成本机器人平台。

### Figure 2: 训练数据构成概览

![Figure 2](https://arxiv.org/html/2605.02881v1/x6.png)

**说明**: MolmoAct2 训练数据混合来自公开学术数据集、MolmoAct2-BimanualYAM（720 小时双臂）、MolmoAct2-SO100/101（184 小时）和 MolmoAct2-DROID（7.46 万条）。YAM/SO-100/101/DROID 各占机器人数据的 30%，另有 10% 多模态数据。

### Figure 3: BimanualYAM 数据采集装置

![Figure 3](https://arxiv.org/html/2605.02881v1/x7.png)

**说明**: 标准化的双臂 YAM 遥操作数据采集设置，覆盖家用和工厂场景，共 34,500 条演示。这是目前最大的开源双臂操控数据集。

### Figure 4: MolmoAct2 模型架构详解

![Figure 4](https://arxiv.org/html/2605.02881v1/x8.png)

**说明**: 图像观测、语言指令和机器人状态被分词后由预训练 VLA 骨干处理；动作专家通过[[逐层 KV 连接]]从每一层 VLM 获取键值对，并以流匹配方式预测目标速度场，最终输出连续动作轨迹。

### Figure 5: MolmoAct2-Think 自适应深度推理

![Figure 5](https://arxiv.org/html/2605.02881v1/figures/MAF44.png)

**说明**: MolmoAct2-Think 在 MolmoAct2 基础上增加自适应深度 token 推理。对比相邻帧深度嵌入的余弦相似度，仅对发生变化的空间格重新预测深度码，静态区域直接复用缓冲值，从而减少冗余计算并保持几何定位能力。

### Figure 6: RoboEval 基准性能对比

![Figure 6](https://arxiv.org/html/2605.02881v1/figures/MAF6.png)

**说明**: MolmoAct2-DROID 在 MolmoSpaces（仿真 Pick/Place/Open/Close）和 MolmoBot（真实操控任务）上均大幅超越 π₀.₅-DROID，平均成功率分别为 37.7% vs. 34.5%（MolmoSpaces）和 87.1%（MolmoBot）。

### Figure 7: 高效微调评测结果

![Figure 7](https://arxiv.org/html/2605.02881v1/figures/MAF5.png)

**说明**: 在 8 个真实世界任务上的综合评测，MolmoAct2 经过少量演示微调后快速适应新任务，展示了其高效微调能力。

---

### Table 1: 多模态网络数据训练语料

| 数据支柱 | Molmo2 样本 | Molmo2 权重 | Molmo2-ER 样本 | Molmo2-ER 权重 |
|---------|-----------|------------|--------------|--------------|
| 图像 QA | 2.4M | 0.115 | — | — |
| 视频 QA | 2.4M | 0.092 | — | — |
| 图像指点 | 1.1M | 0.046 | 780K | 0.11 |
| 视频指点 | 370K | 0.069 | — | — |
| 视频追踪 | 800K | 0.069 | — | — |
| 字幕/长 QA | 1.2M | 0.069 | — | — |
| 图像具身 QA | — | — | 1.33M | 0.11 |
| 图像检测 | — | — | 100K | 0.01 |
| 视频具身 QA | — | — | 703K | 0.1 |
| 多图/自我-外视角 | — | — | 700K | 0.09 |
| 抽象推理 | — | — | 150K | 0.04 |
| NLP | 980K | 0.08 | — | — |
| **小计** | **8.25M** | **0.46** | **3.26M** | **0.46** |

### Table 2: OpenFAST 分词器训练混合

| 数据集 | 比例 | 机器人 | 动作表示 |
|--------|------|--------|----------|
| MolmoAct2-BimanualYAM | 30% | YAM | 绝对关节角 |
| MolmoAct2-SO100/101 | 30% | SO-100/101 | 绝对关节角 |
| MolmoAct2-DROID | 30% | Franka | 绝对关节角 |
| Fractal | 3.33% | Google Robot | 增量末端执行器 |
| BC-Z | 3.33% | Google Robot | 增量末端执行器 |
| Bridge | 3.33% | WidowX | 增量末端执行器 |

### Table 3: 具身推理基准结果（越高越好）

| 模型 | Point-Bench | RefSpatial | RoboSpatial-Poi | Where2Place | BLINK | CV-Bench | ERQA | EmbSpatial | MindCube | RoboSpatial-VQ | SAT | OpenEQA | VSI-Bench | **平均** |
|------|------------|-----------|----------------|------------|-------|---------|------|-----------|---------|--------------|-----|--------|---------|---------|
| GR-ER 1.5 Thinking | 71.6 | 48.5 | 31.1 | 59.0 | 57.8 | 84.3 | 54.8 | 78.4 | 54.7 | 79.3 | 76.7 | 55.0 | 45.8 | 61.3 |
| GR-ER 1.5 | 73.3 | 41.8 | 25.3 | 48.0 | 65.2 | 83.6 | 47.0 | 73.4 | 47.7 | 57.0 | 62.0 | 50.5 | 39.9 | 55.0 |
| Gemini 2.5 Pro | 62.7 | 33.6 | 8.3 | 37.0 | 69.2 | 85.9 | 56.0 | 78.0 | 59.2 | 71.3 | 74.7 | 55.0 | 51.1 | 57.1 |
| GPT-5 | 43.6 | 23.5 | 19.0 | 37.0 | 71.3 | 86.1 | 59.0 | 81.5 | 58.0 | 69.3 | 86.7 | 64.4 | 52.9 | 57.9 |
| GPT-5-mini | 39.5 | 23.0 | 12.5 | 33.5 | 56.4 | 85.9 | 57.3 | 78.8 | 55.6 | 70.7 | 81.3 | 59.2 | 46.2 | 53.8 |
| Qwen3-VL-4B | 63.8 | 49.5 | 62.3 | 63.0 | 65.2 | 85.8 | 39.5 | 78.1 | 27.4 | 62.7 | 60.7 | 47.6 | 61.4 | 59.0 |
| Qwen3-VL-8B | 64.2 | 47.5 | 56.6 | 63.0 | 66.6 | 85.6 | 41.5 | 78.2 | 33.4 | 73.7 | 70.0 | 49.3 | 63.3 | 61.0 |
| LLaVA-OV-7B | 9.4 | 3.0 | 0.0 | 19.0 | 44.2 | 70.3 | 39.8 | 69.7 | 45.6 | 68.9 | 58.0 | 36.2 | 31.2 | 38.1 |
| InternVL3.5-4B | 43.0 | 21.5 | 23.0 | 53.0 | 58.7 | 80.5 | 37.0 | 72.0 | 42.6 | 61.0 | 49.3 | 44.6 | 56.5 | 49.4 |
| InternVL3.5-8B | 45.5 | 35.0 | 27.0 | 57.0 | 59.1 | 80.9 | 41.0 | 74.7 | 39.7 | 62.7 | 58.7 | 43.5 | 56.4 | 52.4 |
| Molmo2 | 76.9 | 52.5 | 13.9 | 18.0 | 50.8 | 64.4 | 46.3 | 67.0 | 37.6 | 57.9 | 50.0 | 46.9 | 26.1 | 46.8 |
| **Molmo2-ER** | **77.3** | **52.5** | **32.0** | **54.0** | **72.5** | **87.8** | **46.8** | **78.8** | **57.0** | **73.4** | **78.0** | **44.7** | **74.5** | **63.8** |

**关键发现**: Molmo2-ER 在 13 个基准中 9 个排名第一，相比基础 Molmo2 提升 17 个百分点，超越所有闭源模型包括 GPT-5。

### Table 4: MolmoSpaces 任务结果（真实机器人，开箱即用）

| 模型 | Pick | Pick & Place | Open | Close | **平均** |
|------|------|-------------|------|-------|---------|
| StereoVLA | 7.0±2.3 | — | — | — | 7.0 |
| LAP-VLA | 24.9±2.7 | 6.6±1.6 | 11.4±1.5 | 45.9±2.9 | 22.2 |
| π₀-DROID | 16.2±2.6 | 12.5±2.0 | 11.0±2.0 | 53.1±3.2 | 23.2 |
| π₀.₅-DROID | 36.4±2.9 | 13.6±2.2 | 22.7±2.6 | 65.1±3.1 | 34.5 |
| **MolmoAct2-DROID** | **43.7±3.1** | **26.7±2.8** | **9.5±2.0** | **70.8±3.0** | **37.7** |

### Table 5: 仿真保留环境结果

| 模型 | Pick MSProc | Pick Classic | Pick | Pick Rand-Cam | Pick&Place | PnP Next-To | PnP Color | **平均** |
|------|-----------|------------|------|-------------|----------|-----------|---------|---------|
| StereoVLA | 6.6±2.6 | 4.3±1.5 | 1.1±1.0 | — | — | — | — | — |
| X-VLA | 3.3±1.0 | 0.5±0.5 | 0.7±0.5 | 0.8±0.5 | 0.1/0.1 | 1.9/1.0 | 0.9/0.5 | 1.2 |
| LAP-VLA | 19.4±2.4 | 2.4±1.0 | 3.1±1.1 | 2.7±1.0 | 3.8/1.6 | 6.5/3.4 | 3.1/1.5 | 4.8 |
| π₀.₅-DROID | 18.1±2.4 | 6.4±1.5 | 7.0±1.6 | 8.0±1.9 | 11.7/7.6 | 8.2/6.2 | 10.4/6.7 | 10.0 |
| **MolmoAct2-DROID** | **35.6±3.0** | **18.9±2.6** | **20.5±2.6** | **15.4±2.4** | **15.8/7.8** | **20.9/14.4** | **17.2/8.8** | **20.6** |

### Table 6: 真实世界操控任务结果（开箱即用）

| 模型 | Apple on Plate | Pipette in Tray | Cube in Tape Roll | Knife in Box | Objects in Bowl | **平均** |
|------|--------------|----------------|-----------------|------------|----------------|---------|
| π₀.₅-DROID | 66.7% | 33.3% | 53.3% | 26.7% | 46.2% | 45.2% |
| MolmoBot | 86.7% | 53.3% | 33.3% | 40.0% | 28.6% | 48.4% |
| **MolmoAct2-DROID** | **100.0%** | **86.7%** | **93.3%** | **93.3%** | **62.0%** | **87.1%** |

**关键发现**: MolmoAct2-DROID 以 87.1% 的平均成功率大幅超越 π₀.₅（45.2%），提升近一倍，在 Apple on Plate 和 Knife in Box 任务上达到满分。

### Table 7: SO-100 低成本平台结果

| 模型 | Fork on Plate | Stack Blocks | Tissues in Basket | Pen on Notebook | Block in Box | **平均** |
|------|-------------|------------|-----------------|---------------|------------|---------|
| SmolVLA | 3.3 | 5.0 | 0.0 | 3.3 | 0.0 | 2.3 |
| π₀-SO100/101 | 30.0 | 6.7 | 20.0 | 80.0 | 90.0 | 45.3 |
| **MolmoAct2-SO100/101** | **70.0** | **20.0** | **73.3** | **86.7** | **33.3** | **56.7** |

### Table 8: LIBERO 基准结果（微调后）

| 模型 | Spatial | Object | Goal | Long | **平均** |
|------|---------|--------|------|------|---------|
| TraceVLA | 84.6% | 85.2% | 75.1% | 54.1% | 74.8% |
| OpenVLA | 84.7% | 88.4% | 79.2% | 53.7% | 76.5% |
| SpatialVLA | 88.2% | 89.9% | 78.6% | 55.5% | 78.1% |
| CoT-VLA | 87.5% | 91.6% | 87.6% | 69.0% | 83.9% |
| ThinkAct | 88.3% | 91.4% | 87.1% | 70.9% | 84.4% |
| MolmoAct-7B-D | 87.0% | 95.4% | 87.6% | 77.2% | 86.6% |
| π₀ | 96.8% | 98.8% | 95.8% | 85.2% | 94.2% |
| NORA-1.5 | 97.3% | 96.4% | 94.5% | 89.6% | 94.5% |
| GR00T N1.7 | 97.7% | 97.5% | 98.5% | 94.4% | 97.0% |
| π₀.₅ | 98.8% | 98.2% | 98.0% | 92.4% | 96.9% |
| **MolmoAct2** | **97.8%** | **100.0%** | **97.8%** | **93.2%** | **97.2%** |
| **MolmoAct2-Think** | **98.8%** | **99.8%** | **98.5%** | **95.4%** | **98.1%** |

**关键发现**: MolmoAct2-Think 以 98.1% 平均成功率超越 π₀.₅（96.9%），在 Long 任务上提升 3 个百分点，展示了深度推理对长时序任务的显著增益。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| MolmoAct2-BimanualYAM | 720 小时 / 34.5K 演示 | 最大开源双臂数据集 | 训练 |
| MolmoAct2-SO100/101 | 184 小时 / 38K 集 | 社区低成本平台 | 训练 |
| MolmoAct2-DROID | 74.6K 集 | 质量筛选的 Franka 数据 | 训练 |
| LIBERO | 4 套任务 (Spatial/Object/Goal/Long) | 标准化仿真基准 | 测试 |
| MolmoSpaces | Pick/Place/Open/Close | 真实机器人开箱即用 | 测试 |
| OpenEQA、VSI-Bench 等 | 13 个具身推理基准 | — | 测试 |

### 实现细节

- **骨干**: Molmo2-ER（Molmo2 专化版，参数量未明确披露）
- **动作专家**: 36 层 DiT 风格 Transformer
- **预训练**: 200K 步，64 张 H100，batch=128，序列长 4200 token
- **后训练**: 100K 步，$K=4$ 流样本，知识隔离（专家梯度不反传入 VLM）
- **微调**: 50-100K 步，$K=8$ 流样本，可选端到端全量训练
- **推理优化**: CUDA Graphs 固定形状流循环 + 跨注意力状态 KV 缓存复用

---

## 批判性思考

### 优点

1. **真正的完全开放**: 权重 + 训练代码 + 完整训练数据三位一体，行业罕见
2. **开箱即用性强**: 无需针对特定机器人微调即在多个平台达到可用级别
3. **架构创新有据可查**: 逐层 KV 连接的逐层消融验证了设计的有效性
4. **低成本平台友好**: 在 SO-100 上展示了 56.7% 成功率，证明可普惠性
5. **深度感知推理可解释**: 深度 token 预测提供几何定位的可视化诊断能力

### 局限性

1. **Open 任务表现回退**: MolmoSpaces 中 Open 任务（9.5%）显著低于 π₀.₅（22.7%），表明某些任务类型上仍有差距
2. **Block in Box 在 SO 上偏低**: SO-100 上 Block in Box 仅 33.3%，明显弱于 π₀ 的 90.0%
3. **计算开销未量化**: 虽有推理优化，但实际推理延迟与 π₀.₅ 的定量对比未给出
4. **消融细节较少**: 论文提到了消融研究，但数值表格未完整展示，可复现性受限

### 潜在改进方向

1. 对"Open"类任务做专项数据增强或课程学习
2. 深度推理扩展到语义层面（不只是几何），如"物体出现/消失"触发推理
3. 探索跨模型蒸馏：用 MolmoAct2-Think 为更轻量模型生成教师信号

### 可复现性评估

- [x] 代码开源（承诺发布训练代码）
- [x] 预训练模型（权重开放）
- [x] 训练细节完整（步数、GPU、采样比例均披露）
- [x] 数据集可获取（三个新数据集均开放）

---

## 关联笔记

### 基于

- [[Molmo2]]: 基础视觉-语言模型骨干
- [[Flow Matching]]: 连续动作生成的核心方法论
- [[Action Chunking]]: 预测 1 秒动作块而非单步动作

### 对比

- [[pi0.5]]: 主要对比基线，MolmoAct2-Think 在 LIBERO 上超越
- [[GR00T N1.7]]: LIBERO 强竞争对手，平均 97.0% vs. 97.2%
- [[OpenVLA]]: 早期开源 VLA，LIBERO 上落后约 20%

### 方法相关

- [[Flow Matching]]: 连续动作专家的基础
- [[逐层 KV 连接]]: 核心架构创新
- [[OpenFAST]]: 开源动作分词器
- [[具身推理]]: Molmo2-ER 的核心目标能力

### 硬件/数据相关

- [[YAM 双臂机器人]]: BimanualYAM 数据集采集平台
- [[SO-100]]: 低成本机器人平台
- [[DROID 数据集]]: 高质量 Franka 数据来源

---

## 速查卡片

> [!summary] MolmoAct2: Action Reasoning Models for Real-world Deployment
> - **核心**: 完全开源的 VLA，通过具身推理骨干 + 逐层 KV 连接 + 自适应深度推理实现顶级性能
> - **方法**: Molmo2-ER + OpenFAST + Flow Matching Action Expert + MolmoAct2-Think
> - **结果**: LIBERO 98.1%（超越 π₀.₅）；真实操控 87.1%（π₀.₅ 仅 45.2%）；具身推理 13 基准 63.8%（超 GPT-5）
> - **代码**: [allenai.org/blog/molmoact2](https://allenai.org/blog/molmoact2)

---

*笔记创建时间: 2026-05-09*
