---
title: "LaST-HD: Learning Latent Physical Reasoning from Scalable Human Data for Robot Manipulation"
method_name: "LaST-HD"
authors: [Jiaming Liu, Yinxi Wang, Chenyang Gu, Siyuan Qian, Xiangyu Mi]
year: 2026
venue: arXiv
tags: [vla, human-to-robot-transfer, latent-alignment, imitation-learning, dexterous-manipulation]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.23685v1
created: 2026-06-23
---

# 论文笔记：LaST-HD: Learning Latent Physical Reasoning from Scalable Human Data for Robot Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Peking University, CUHK, Simplexity Robotics, Aether Tech |
| 日期 | June 2026 |
| 项目主页 | [siriyep.github.io/last-hd-project-page](https://siriyep.github.io/last-hd-project-page/) |
| 对比基线 | [[π0.5]]、[[Cosmos-Policy]]、LaST0 |
| 链接 | [arXiv](https://arxiv.org/abs/2606.23685) |

---

## 一句话总结

> 通过[[潜在空间预测|潜在物理推理对齐]]，让 VLA 在人手与机器人的共享推理空间中学习，配合轻量 IMU 手套仅需 20 分钟人手数据即可在新场景达到 90% 以上准确率。

---

## 核心贡献

1. **潜在物理推理对齐（Latent Physical Reasoning Alignment）**: 用[[动作条件世界模型|动作条件化世界模型]]的中间层特征作为监督信号，强迫 VLA 的推理专家学到跨体态不变的物理动力学表征，而非单纯的动作级别模仿。
2. **OOL 手套（Out-of-Lab Glove）**: 基于 IMU 的轻量可穿戴系统（<100g），以亚毫米精度、>200 Hz 频率追踪 21 个手腕关键点，采集速度比机器人遥操作快 4-5×。
3. **混合到人手的两阶段训练范式（Mixed-to-Human Recipe）**: 第一阶段混合机器人与人手数据共训，第二阶段用 OOL 手套在线采集少量修正数据进行后训练，实现高效跨场景泛化。

---

## 问题背景

### 要解决的问题

在非结构化真实环境中开发通用机器人操作策略，需要大量机器人演示数据，而真实机器人遥操作代价高昂、难以扩展。如何利用廉价易得的人手演示数据来增强机器人策略的泛化能力？

### 现有方法的局限

- **运动学重定向（Kinematic Retargeting）**：仅对齐动作空间，忽视了人手与机器人在外观、视角上的巨大差异，在分布外场景失效。
- **纯动作级别共训**：将人手数据直接混入训练，缺乏跨体态的表征对齐，提升有限。
- **EgoMimic / H-RDT 类方法**：依赖穿戴式相机从第一人称视角学习，数据采集流程复杂，且尚未在潜在推理层面做对齐。
- **真实机器人遥操作数据**：采集慢（慢 4-5×），扩展成本高。

### 本文的动机

人类和机器人在执行物理操作时共享相同的物理规律（重力、接触力学、物体动力学）。如果能在**共享的潜在推理空间**中对齐两者的表征——而非强行对齐动作空间——那么人手演示就能有效地将物理推理能力迁移给机器人策略，同时绕开体态差异带来的运动学鸿沟。

---

## 方法详解

### 模型架构

LaST-HD 采用 **[[混合专家 Transformer|Mixture-of-Transformers（MoT）]]** [[VLA]] 架构，基于 [[Janus-Pro]] 构建：

- **输入**: 语言指令 $l$ + 三路相机观测 $o_t$（384×384 分辨率）+ 体态状态 $s_t$
- **视觉编码器**: [[SigLIP]]-Large，提取视觉特征 $f_{\text{img}} \in \mathbb{R}^{N_{\text{img}} \times d_v}$
- **VLA 主干**: DeepSeek-LLM 1.5B（24 层 decoder-only Transformer），隐藏维度 $d_l$
- **推理专家（Reasoning Expert）**: 输出潜在状态 $\mathcal{Z} \in \mathbb{R}^{N_{\text{lat}} \times d_l}$，由世界模型特征监督
- **动作专家（Action Expert）**: 基于[[流匹配（Flow Matching）|流匹配]]的解码器，输出动作块
- **总参数**: 基于 1.5B LLM

### 核心模块

#### 模块1：人手-机器人潜在对齐（Human-to-Robot Latent Alignment）

**设计动机**: 利用[[动作条件世界模型]]捕获的前向动力学特征，在体态无关的潜在空间中建立人手与机器人演示的对应关系。

**具体实现**:
1. 在混合人-机器人数据上微调一个[[动作条件世界模型|动作条件化视频世界模型]]
2. 提取世界模型最深 U-Net 层的特征作为**领域不变的物理动力学表征**
3. 将该特征投影到 LaST-HD 的潜在维度，压缩为 $N_{\text{lat}}$ 个 token，作为推理专家的监督目标 $z_t^{\text{GT}}$
4. 推理专家的输出 $\hat{z}_t$ 通过[[Cosine Similarity|余弦相似度损失]]向 $z_t^{\text{GT}}$ 对齐

#### 模块2：OOL 手套数据采集系统

**设计动机**: 利用轻量 IMU 传感器实现**实验室外（Out-of-Lab）**的可扩展人手数据采集。

**具体实现**:
- 每只手套重量 < 100g，配备 6 个 IMU 6-DoF 模块
- 追踪 20 个手部关键点 + 1 个手腕关键点，共 21 点
- 采样频率 > 200 Hz，延迟 < 10 ms，每关键点 RMS 误差亚毫米级
- 三路同步相机：头戴 ZED 2i × 1，手腕 Insta360 GO 3S × 2
- 轨迹可**重定向**到不同执行器：夹爪（fingertip 距离）、灵巧手（逆运动学）

#### 模块3：混合到人手两阶段训练范式

**第一阶段 — 混合共训（Mixed Co-training）**:
- 在人手 + 机器人**未配对**数据上训练世界模型和 LaST-HD
- 动作专家使用[[流匹配（Flow Matching）|流匹配]]损失 $\mathcal{L}_{\text{act}}$
- 推理专家由世界模型潜在目标监督，合并损失 $\mathcal{L}_{\text{loss}} = \mathcal{L}_{\text{act}} + \lambda \mathcal{L}_{\text{latent}}$

**第二阶段 — 人手在线修正（Human-Hand Online Correction / DAgger）**:
- 部署策略，识别失败倾向状态
- 用 OOL 手套采集少量修正演示（每场景约 20-60 轨迹，≈20 分钟）
- [[DAgger]] 式后训练 11-22 个 epoch，平衡回放：

$$
\mathcal{B} = \mathcal{B}_{\text{prev}} \cup \mathcal{B}_{\text{dagger}}, \quad |\mathcal{B}_{\text{prev}}| = |\mathcal{B}_{\text{dagger}}|
$$

---

## 关键公式

### 公式1：[[Cosine Similarity|潜在对齐损失]]

$$
\mathcal{L}_{\text{latent}} = \sum_{t=1}^{N_{\text{lat}}} \left[ 1 - \frac{\hat{z}_t \cdot z_t^{\text{GT}}}{\|\hat{z}_t\| \cdot \|z_t^{\text{GT}}\|} \right]
$$

**含义**: 用余弦相似度最小化推理专家输出的潜在状态与世界模型监督目标之间的角度偏差，迫使两者在方向上对齐。

**符号说明**:
- $N_{\text{lat}}$: 潜在 token 数量
- $\hat{z}_t \in \mathbb{R}^{d_l}$: 推理专家第 $t$ 个输出 token
- $z_t^{\text{GT}} \in \mathbb{R}^{d_l}$: 世界模型提供的第 $t$ 个监督目标 token

### 公式2：[[训练目标|总训练损失]]

$$
\mathcal{L}_{\text{loss}} = \mathcal{L}_{\text{act}} + \lambda \mathcal{L}_{\text{latent}}
$$

**含义**: 联合优化动作预测与潜在推理对齐两个目标，$\lambda$ 控制对齐强度。

**符号说明**:
- $\mathcal{L}_{\text{act}}$: [[流匹配（Flow Matching）|流匹配]]动作损失
- $\mathcal{L}_{\text{latent}}$: 潜在对齐余弦相似度损失
- $\lambda$: 平衡权重系数

### 公式3：[[DAgger|在线修正批次构造]]

$$
\mathcal{B} = \mathcal{B}_{\text{prev}} \cup \mathcal{B}_{\text{dagger}}, \quad |\mathcal{B}_{\text{prev}}| = |\mathcal{B}_{\text{dagger}}|
$$

**含义**: 后训练时将历史演示与新采集的 DAgger 修正数据 1:1 混合，防止灾难性遗忘。

**符号说明**:
- $\mathcal{B}_{\text{prev}}$: 原有训练数据批次
- $\mathcal{B}_{\text{dagger}}$: OOL 手套在线修正数据批次

### 公式4：视觉特征提取

$$
f_{\text{img}} \in \mathbb{R}^{N_{\text{img}} \times d_v}
$$

**含义**: [[SigLIP]]-Large 视觉编码器从输入图像中提取 $N_{\text{img}}$ 个视觉 token，每个维度为 $d_v$。

**符号说明**:
- $N_{\text{img}}$: 视觉 token 数量
- $d_v$: 视觉特征维度

### 公式5：潜在状态张量

$$
\mathcal{Z} \in \mathbb{R}^{N_{\text{lat}} \times d_l}
$$

**含义**: 推理专家输出的潜在状态序列，捕获物理推理中间表征，供动作专家作为条件。

**符号说明**:
- $N_{\text{lat}}$: 潜在 token 数量
- $d_l$: LLM 隐藏维度

---

## 关键图表

### Figure 1: 系统概览

![Figure 1](https://arxiv.org/html/2606.23685v1/x1.png)

**说明**: LaST-HD 整体系统概览。展示 OOL 手套数据采集流程、人手与机器人数据混合共训、以及通过潜在推理对齐将人类物理推理迁移到机器人策略的全流程。

### Figure 2: 框架架构详解

![Figure 2](https://arxiv.org/html/2606.23685v1/x2.png)

**说明**: LaST-HD 框架核心组件。左侧展示[[混合专家 Transformer|MoT VLA]] 的推理专家与动作专家结构；右侧展示[[动作条件世界模型]]如何生成潜在监督目标 $z_t^{\text{GT}}$，以及余弦相似度损失如何驱动跨体态潜在对齐。

### Figure 3: 消融实验与在线修正效果

![Figure 3](https://arxiv.org/html/2606.23685v1/x3.png)

**说明**: 左图展示不同潜在对齐策略的消融（动作条件化世界模型最优）；右图展示在线修正轨迹数量与成功率的关系，20 条轨迹即可在 Unseen Background 达到 100%。

### Figure 4: 机器人执行与场景展示

![Figure 4](https://arxiv.org/html/2606.23685v1/x4.png)

**说明**: 三种不同体态（Galaxea R1 Lite 双臂夹爪、Tianji Marvin 双臂夹爪、Tianji Marvin + WUJI 灵巧手）在 6 个真实任务上的执行图示，包括拧瓶盖、整理箱子、分拣水果、放置并拉链、倒水、抓取夹子。

### Figure 5: OOL 手套硬件

![Figure 5](https://arxiv.org/html/2606.23685v1/x5.png)

**说明**: OOL（Out-of-Lab）手套实物图，展示 6 个 IMU 传感器模块分布、手腕关键点追踪示意，以及配套三相机数据采集装置。

### Table 1: 域内任务成功率对比

| 方法 | Unscrew Cap | Organize Box | Sort Fruits | Put and Zip | Pour Water | Grasp Clamp | Average |
|------|-------------|--------------|-------------|-------------|------------|-------------|---------|
| [[π0.5]] | 0.70 | 0.70 | 0.85 | 0.75 | 0.30 | 0.40 | 0.62 |
| [[Cosmos-Policy]] | 0.75 | 0.50 | 0.85 | 0.60 | 0.20 | 0.20 | 0.52 |
| LaST0 | 0.80 | 0.70 | 0.75 | 0.60 | 0.40 | 0.50 | 0.63 |
| LaST-HD (Mix-HD) | 0.85 | 0.70 | 0.85 | 0.80 | 0.40 | 0.45 | 0.68 |
| **LaST-HD** | **0.85** | **0.70** | **0.95** | **0.80** | **0.60** | **0.45** | **0.73** |

**关键发现**: LaST-HD 以 0.73 平均成功率超越所有基线（π0.5: 0.62, Cosmos-Policy: 0.52）；Mix-HD（混合人手数据但无对齐）达 0.68，说明潜在对齐额外贡献约 5%。

### Table 2: 泛化成功率（零样本 vs. 加入人手修正数据）

| 方法 | Unseen Position | Unseen Object | Unseen Background |
|------|-----------------|---------------|-------------------|
| [[π0.5]] | 0.12 | 0.36 | 0.43 |
| [[Cosmos-Policy]] | 0.13 | 0.28 | 0.38 |
| LaST0 | 0.15 | 0.32 | 0.43 |
| LaST-HD (Mix-HD) | 0.15 | 0.35 | 0.43 |
| LaST0 (w/ unseen HD) | 0.33 | 0.49 | 0.58 |
| **LaST-HD (w/ unseen HD)** | **0.41** | **0.58** | **0.68** |

**关键发现**: 加入 OOL 手套数据后，LaST-HD 在三个泛化维度上均大幅领先：Unseen Position +173%，Unseen Object +61%，Unseen Background +58%（相比无人手数据的 LaST0）。

### Table 3: 消融——潜在对齐策略比较（Sort Fruits，三场景平均）

| 配置 | 平均成功率 |
|------|-----------|
| w/o Latent（纯动作） | 60% |
| SigLIP 特征作为目标 | ~65% |
| 世界模型（无动作条件化） | ~68% |
| **LaST-HD（动作条件化世界模型）** | **73%** |

**关键发现**: 动作条件化是关键——无动作条件的世界模型特征提升有限，加入动作条件后潜在目标更加物理准确，对齐效果最佳。

### Table 4: 消融——OOL 手套 vs. 其他数据来源（60 条演示，三场景平均）

| 数据来源 | 平均成功率 |
|---------|-----------|
| 裸手视觉追踪（Bare-hand） | 63% |
| 机器人遥操作（时间匹配，12 条） | 60% |
| **OOL 手套** | **73%** |
| 机器人遥操作（更多数据，60 条） | ~73% |
| 手掌朝向相机（vs. 拇食指根部） | 67% |

**关键发现**: OOL 手套在相同采集时间下比裸手视觉追踪高 10%，与更多机器人数据相当，且采集速度快 4-5×。

---

## 实验

### 数据集与任务

| 任务 | 体态 | 机器人数据 | 人手数据 |
|------|------|-----------|---------|
| Unscrew Cap | Galaxea R1 Lite | 100 条 | 50 条 |
| Organize Box | Galaxea R1 Lite | 100 条 | 50 条 |
| Sort Fruits | Galaxea R1 Lite | 100 条 | 50 条 |
| Put and Zip | Tianji Marvin | 100 条 | 50 条 |
| Pour Water | Tianji Marvin | 100 条 | 50 条 |
| Grasp Clamp | Tianji Marvin + WUJI 灵巧手 | 100 条 | 50 条 |

泛化场景额外采集：100 条机器人 + 60 条 OOL 手套数据。

### 实现细节

- **VLA 主干**: DeepSeek-LLM 1.5B
- **视觉编码器**: SigLIP-Large
- **图像分辨率**: 384×384 像素（三路相机各一）
- **动作空间**: 双臂夹爪 7-DoF × 2，灵巧手额外 26-DoF（WUJI Hand）
- **后训练轮数**: 11-22 个 epoch（DAgger 阶段）
- **评估次数**: 每任务每方法 20 次 rollout
- **硬件**: Galaxea R1 Lite, Tianji Marvin（双臂），Tianji Marvin + WUJI Hand（灵巧手）

### 关键定性发现

- 20 分钟 OOL 手套数据（≈20-60 条轨迹）在 Unseen Background 达到 100% 成功率
- Unseen Position 是最难场景（空间布局变化），60 条修正后约 80%
- 相机位置（拇食指根部 vs. 手掌）影响约 6% 成功率，说明视角对人手数据质量至关重要

---

## 批判性思考

### 优点

1. **对齐层次更高**: 在潜在推理空间对齐，而非运动学/动作级别，绕过体态形态差异，泛化性更强
2. **数据效率极高**: 仅需 20 分钟 OOL 手套数据即可大幅提升新场景成功率，工业落地成本低
3. **硬件设计务实**: <100g、<10 ms 延迟、亚毫米精度，真正可在实验室外大规模采集

### 局限性

1. **实时性问题**: 论文明确指出"潜在推理尚不实时"，限制了高频控制场景的应用
2. **数据仍需机器人**: 第一阶段仍需 100 条机器人遥操作数据/任务，尚未完全摆脱机器人数据依赖
3. **泛化场景有限**: 仅在有限的泛化维度（位置/物体/背景）上测试，更复杂任务结构（如多步骤规划）未验证

### 潜在改进方向

1. **快慢系统（Fast-Slow System）**: 作者提出的未来方向，用快速反应策略处理高频控制，潜在推理在慢速通道运行
2. **潜在空间压缩**: 减少推理 token 数量或引入稀疏潜在表征，降低推理延迟
3. **完全无机器人数据**: 探索纯人手数据驱动的训练，进一步降低数据采集门槛

### 可复现性评估

- [ ] 代码开源（项目主页有，但代码链接暂未公开）
- [ ] 预训练模型（未公开）
- [x] 训练细节完整（论文中有详细描述）
- [ ] 数据集可获取（OOL 手套数据未公开）

---

## 关联笔记

### 基于

- [[Janus-Pro]]: VLA 底座模型（VLM + 流匹配动作专家）
- [[DAgger]]: 在线修正（第二阶段训练范式的理论基础）
- [[流匹配（Flow Matching）|Flow Matching]]: 动作专家的解码方式

### 对比

- [[π0.5]]: 强基线，相同规模 VLA，无人手数据对齐机制
- [[Cosmos-Policy]]: NVIDIA 大规模预训练 VLA，无人手对齐
- LaST0: 无人手数据的 LaST-HD 消融版本

### 方法相关

- [[动作条件世界模型]]: 提供潜在对齐监督目标的核心组件
- [[潜在空间预测]]: 推理专家学习的核心表征类型
- [[Cross-Embodiment Learning]]: 跨体态数据利用的上层框架
- [[混合专家 Transformer]]: MoT 架构（推理专家 + 动作专家的分离设计）
- [[SigLIP]]: 视觉编码器

### 硬件/数据相关

- [[scalable robot data collection]]: 可扩展数据采集方法论的上下文
- [[灵巧手操作]]: Grasp Clamp 任务使用 WUJI 灵巧手

---

## 速查卡片

> [!summary] LaST-HD
> - **核心**: 在潜在推理空间对齐人手与机器人演示，用动作条件化世界模型特征监督 VLA 推理专家
> - **方法**: MoT VLA + 余弦潜在对齐损失 + OOL IMU 手套 + DAgger 在线修正
> - **结果**: 域内平均 73%（超 π0.5 11 pt），20 分钟手套数据泛化可达 90%+
> - **代码**: [项目主页](https://siriyep.github.io/last-hd-project-page/)

---

*笔记创建时间: 2026-06-23*
