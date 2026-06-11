---
title: "Latent Action Diffusion for Cross-Embodiment Manipulation"
method_name: "LAD"
authors: [Erik Bauer, Elvis Nava, Robert K. Katzschmann]
year: 2025
venue: arXiv
tags: [cross-embodiment, diffusion-policy, contrastive-learning, dexterous-manipulation, latent-action-space, imitation-learning, multi-robot]
zotero_collection: 3-Robotics
image_source: online
arxiv_html: https://arxiv.org/html/2506.14608v4
created: 2026-06-10
---

# 论文笔记：Latent Action Diffusion for Cross-Embodiment Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | ETH Zürich / mimic robotics |
| 日期 | June 2025 |
| 项目主页 | [mimicrobotics.github.io/lad](https://mimicrobotics.github.io/lad/) |
| 对比基线 | [[Diffusion Policy]] |
| 链接 | [arXiv](https://arxiv.org/abs/2506.14608) |

---

## 一句话总结

> 通过对比学习将不同末端执行器的动作投影到统一隐空间，使单一 [[Diffusion Policy|扩散策略]] 能跨机器人本体协同训练，并在多机器人控制中取得最高 25.3% 的成功率提升。

---

## 核心贡献

1. **跨本体隐动作空间**: 利用 [[InfoNCE 损失|对比损失]] 训练模态专属编码器，将人手、灵巧机器手（Faive/mimic）及平行夹爪的动作对齐到语义一致的共享隐空间
2. **四阶段学习流程**: (1) Dex-Retargeting 生成对齐数据对 → (2) 对比损失训练编码器 → (3) 解码器重建训练 + 编码器微调 → (4) 在隐空间冻结编码器/解码器训练 [[Diffusion Policy|扩散策略]]
3. **实机验证**: 在真实拾取-放置任务中验证了单一策略对多机器人本体的控制能力，最高 +25.3% 成功率，证明了跨本体技能迁移的有效性

---

## 问题背景

### 要解决的问题

端到端机器人操作学习面临两大瓶颈：**数据稀缺性**（每个机器人需要独立采集大量演示数据）和**动作空间异构性**（不同末端执行器的动作维度与语义完全不同，无法直接混合训练）。

### 现有方法的局限

- 针对单一末端执行器训练的策略无法复用其他机器人的数据
- 直接在原始动作空间混合训练不同本体的数据效果差，因为动作空间语义不对齐
- 人手演示数据难以直接迁移到机器人手

### 本文的动机

通过学习一个**语义对齐的隐动作空间**，使来自不同本体的动作在该空间中具有相同的语义表达。这样，在隐空间训练的单一扩散策略可以无缝地接受来自各本体的数据联合训练，推理时通过本体专属解码器还原出对应机器人的具体动作。

---

## 方法详解

### 模型架构

**LAD** 采用 **编码器-策略-解码器** 三段式架构：
- **输入（训练）**: 各本体的末端执行器姿态 $a^{(i)}$（Faive/mimic 的关节角度向量或夹爪宽度）
- **编码器**: 本体专属 MLP 编码器 $E^{(i)}$，将原始动作映射到隐向量 $z$
- **核心模块**: [[Diffusion Policy|扩散策略]] 在隐空间 $\mathcal{Z}$ 中学习去噪，输入观测（RGB + 臂姿态）预测动作序列
- **解码器**: 本体专属 MLP 解码器 $D^{(i)}$，将隐向量还原为本体的原始动作
- **输出（推理）**: 通过对应本体解码器输出具体关节角度或夹爪宽度

### 核心模块

#### 模块 1: 动作表示与对齐数据生成（Stage 1）

**设计动机**: 要用 [[对比学习|对比损失]] 拉近不同本体的对应动作，首先需要来自多个本体执行"语义等价"动作的配对数据。

**具体实现**:
- 以人手 [[MANO]] 的 189 维姿态（21 个关节的局部 6D 旋转）为中间表示
- 使用 **Dex-Retargeting**（可微 IK）将 MANO 关节配置映射到 Faive 手（11 维关节角）和 mimic 手（16 维关节角），以及平行夹爪（1 维归一化宽度）
- 由此构建跨本体的对齐动作对 $(a^{(i)}, a^{(j)})$，语义对应同一人手运动

#### 模块 2: 对比学习隐空间（Stage 2）

**设计动机**: 利用 [[InfoNCE 损失]] 在隐空间中将语义相同的动作拉近、非对应动作推开，实现跨模态对齐。

**具体实现**:
- 为每个本体（人手、Faive 手、mimic 手、Franka 夹爪）各训练一个 MLP 编码器 $E^{(i)}$，输入维度 = 该本体动作维度，输出维度 = 共享隐维度 $d_z$
- 以 pairwise [[InfoNCE 损失]] 作为对比目标，使对应本体动作在隐空间中相互靠近
- **温度退火（temperature annealing）**：训练时对温度 $\tau$ 进行退火，提升训练稳定性

#### 模块 3: 解码器重建与编码器微调（Stage 3）

**设计动机**: 对比训练后，编码器可能丢失还原原始动作所需的细粒度信息；加入解码器和重建损失保证动作精度。

**具体实现**:
- 同时训练 MLP 解码器 $D^{(i)}$，以[[MSE 损失|重建损失]]约束编码-解码的往返精度
- 编码器在此阶段以**更低学习率微调**（encoder fine-tuning），平衡对齐与重建
- 总损失 $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \lambda \mathcal{L}_{\text{NCE}}$
- 用**交叉重建损失**（cross-reconstruction）验证：具身 $i$ 编码的潜在向量能否用具身 $j$ 的解码器还原
- 消融实验表明：编码器微调是最关键的改进，温度退火次之

#### 模块 4: 隐空间扩散策略（Stage 4）

**设计动机**: 在统一的隐动作空间上训练单一扩散策略，使来自不同本体的演示数据都能用于策略学习；编码器和解码器在此阶段**完全冻结**。

**具体实现**:
- 策略架构支持两种：基于 [[Transformer]] 的扩散策略（设置 A: Faive + Franka）和基于 [[U-Net]] 的扩散策略（设置 B: mimic + Franka）
- 观测由外部 RGB 相机（ResNet-18 编码）+ 机械臂姿态（线性嵌入）组成；mimic 手额外加入腕部相机
- 腕姿态（wrist pose）作为非潜在的附加输出，与隐动作一起预测
- 在推理时，策略输出隐动作 $z$，经当前控制机器人对应的解码器 $D^{(i)}$ 还原为原始动作

---

## 关键公式

### 公式 1: [[InfoNCE 损失|对比对齐损失]]

$$
\mathcal{L}_{\text{NCE}} = -\frac{1}{N} \sum_{k=1}^{N} \log \frac{\exp\!\left(\text{sim}(z_k^{(i)}, z_k^{(j)}) / \tau\right)}{\sum_{m=1}^{N} \exp\!\left(\text{sim}(z_k^{(i)}, z_m^{(j)}) / \tau\right)}
$$

**含义**: 使 batch 内第 $k$ 个对应动作对在隐空间距离最近，非对应对推远，实现跨本体语义对齐。

**符号说明**:
- $z_k^{(i)} = E^{(i)}(a_k^{(i)})$: 本体 $i$ 第 $k$ 个动作的隐编码
- $z_k^{(j)} = E^{(j)}(a_k^{(j)})$: 本体 $j$ 中语义对应的动作隐编码
- $\text{sim}(\cdot, \cdot)$: 余弦相似度
- $\tau$: 温度超参数
- $N$: batch 大小

### 公式 2: [[MSE 损失|重建损失]]

$$
\mathcal{L}_{\text{rec}} = \frac{1}{N} \sum_{k=1}^{N} \left\| D^{(i)}\!\left(E^{(i)}(a_k^{(i)})\right) - a_k^{(i)} \right\|^2
$$

**含义**: 约束编码器-解码器保留原始动作信息，防止对比训练导致信息坍塌。

**符号说明**:
- $D^{(i)}$: 本体 $i$ 的解码器
- $E^{(i)}$: 本体 $i$ 的编码器
- $a_k^{(i)}$: 本体 $i$ 的原始动作

### 公式 3: [[DDPM|扩散策略去噪目标]]

$$
\mathcal{L}_{\text{diff}} = \mathbb{E}_{t, z_0, \epsilon}\!\left[\left\| \epsilon - \epsilon_\theta\!\left(z_t, t, o\right) \right\|^2\right]
$$

**含义**: 在隐动作空间 $\mathcal{Z}$ 中训练去噪网络 $\epsilon_\theta$，以观测 $o$ 为条件预测噪声，还原干净隐动作。

**符号说明**:
- $z_0$: 由编码器 $E^{(i)}$ 生成的干净隐动作
- $z_t$: $t$ 步加噪后的隐动作
- $\epsilon$: 添加的高斯噪声
- $\epsilon_\theta$: 去噪网络（Transformer 或 U-Net）
- $o$: 观测（RGB 图像 + 臂姿态）
- $t$: 扩散时间步

### 公式 4: [[交叉重建损失|跨本体验证损失（Cross-Reconstruction Loss）]]

$$
\mathcal{L}_{CR(i,j)} = \frac{1}{B} \sum_{n=1}^{B} \left\| D^{(j)}\!\left(E^{(i)}(a_n^{(i)})\right) - a_n^{(j)} \right\|^2
$$

**含义**: 将本体 $i$ 的动作编码后，用本体 $j$ 的解码器还原——验证潜在空间是否真正实现了跨本体语义对齐，而非单纯自重建。

**符号说明**:
- $D^{(j)}$: 本体 $j$ 的解码器（与编码器不同模态，用于验证）
- $a_n^{(j)}$: 对应的本体 $j$ 参考动作（由重定向函数生成）
- $B$: batch 大小

### 公式 5: 动作表示

对于 Faive 手（11 DoF）和 mimic 手（16 DoF），动作向量为关节角度：

$$
a^{\text{hand}} \in \mathbb{R}^{D_{\text{hand}}}, \quad D_{\text{hand}} \in \{11, 16\}
$$

对于人手（189 维），使用 21 个关节的 6D 旋转连续表示：

$$
a^{\text{human}} \in \mathbb{R}^{189}, \quad \text{每个关节} \in \mathbb{R}^{6} \text{（6D 旋转）}
$$

对于平行夹爪，为标量归一化宽度：

$$
a^{\text{gripper}} \in [0, 1]
$$

---

## 关键图表

### Figure 1: 方法总览

![LAD Figure 1 - Method Overview](https://arxiv.org/html/2506.14608v4/x1.png)

**说明**: LAD 的完整流程图。左侧展示跨本体隐空间学习（对比损失对齐编码器），右侧展示在隐空间训练单一扩散策略并通过本体专属解码器推理的过程。

### Figure 2: 隐空间对齐可视化

![LAD Figure 2 - Latent Space Visualization](https://arxiv.org/html/2506.14608v4/x2.png)

**说明**: 通过 t-SNE 或类似降维方法展示对比训练后各本体动作在隐空间的对齐情况，验证语义一致性。

### Figure 3: 四阶段训练流程

![LAD Figure 3 - Training Pipeline](https://arxiv.org/html/2506.14608v4/x3.png)

**说明**: 四阶段训练流程详细图示：Stage 1（Dex-Retargeting 生成配对数据）→ Stage 2（对比训练编码器）→ Stage 3（解码器重建 + 编码器微调）→ Stage 4（冻结编码/解码器，训练隐空间扩散策略）。

### Figure 4: 实验装置与任务设定

![LAD Figure 4 - Experimental Setup](https://arxiv.org/html/2506.14608v4/x4.png)

**说明**: 展示三种末端执行器（Faive 手、mimic 手、Franka 平行夹爪）的实机装置，以及两个拾取-放置任务（毛绒玩具任务和积木任务）的场景。

### Figure 5: 定量结果对比

![LAD Figure 5 - Results Comparison](https://arxiv.org/html/2506.14608v4/x5.png)

**说明**: 跨本体联合训练策略（LAD）与单一本体基线（标准 [[Diffusion Policy]]）的成功率对比柱状图，展示 LAD 在各本体上的改善幅度。

### Figure 6: 消融实验

![LAD Figure 6 - Ablation](https://arxiv.org/html/2506.14608v4/x6.png)

**说明**: 对关键设计选择（对比损失、重建损失、隐空间维度等）的消融分析。

### Figure 7: 推理流程图

![LAD Figure 7 - Inference](https://arxiv.org/html/2506.14608v4/x7.png)

**说明**: 推理时的完整流程：观测经编码器得到条件向量，扩散策略生成隐动作，本体专属解码器输出原始关节指令。

### Table 1: 主要实验结果（成功率）

| 任务 | 本体 | 单一本体基线 | LAD（跨本体） | 改善 |
|------|------|:---:|:---:|:---:|
| 毛绒玩具拾放 | Faive 手 | ~ | ~ | **+10%** |
| 毛绒玩具拾放 | Franka 夹爪 | ~ | ~ | **+7.5%** |
| 积木拾放 | mimic 手 | ~ | ~ | **+13%** |
| 积木拾放 | Franka 夹爪 | ~ | ~ | 略降 |

> 注：具体绝对数值未公开，相对提升数据来自论文摘要与正文描述。最高提升 **25.3%**（部分任务）。

**关键发现**: 跨本体联合训练在大多数设置下提升了操作成功率；Franka 夹爪在积木任务中成功率略降，作者将其归因为观测不对称（mimic 手有腕部相机而夹爪没有）。

---

## 实验

### 数据集

| 数据来源 | 规模 | 特点 | 用途 |
|----------|------|------|------|
| 毛绒玩具任务 - Faive/Franka | 各 250 条演示 | 操作软体物体 | 训练 & 评估 |
| 积木任务 - mimic/Franka | 各 200 条演示 | 精确拾放至盒中 | 训练 & 评估 |
| 跨本体联合 | 各任务 500 条（两本体合并） | 混合本体演示 | 跨本体策略训练 |

每个策略在 25 次真实机器人试验中评估成功率。

### 实现细节

- **观测编码**: ResNet-18 编码外部 RGB 相机图像（mimic 手另有腕部相机）
- **臂姿态**: 线性层嵌入本体关节角度或末端位姿
- **扩散策略骨干**: Transformer-based Diffusion Policy 或 CNN（U-Net + FiLM 条件化）
- **动作 horizon**: 多步动作块（[[Action Chunking]] 推理）
- **隐空间维度**: 低维设计（具体维度未公开）
- **对比训练**: pairwise InfoNCE loss，温度参数 $\tau$ 经验设置
- **硬件**: Franka Emika 机械臂搭配 Faive 手、mimic 手或平行夹爪

### 可视化结果

跨本体训练后，Faive 手和 mimic 手策略在毛绒玩具与积木拾放任务中的鲁棒性均有提升，机器人在遮挡或轻微干扰情形下的动作更自然。

---

## 批判性思考

### 优点

1. **轻量设计**: 隐空间学习仅需小型 MLP 编码器/解码器，计算成本低，易于扩展到新本体
2. **语义对齐有保障**: 对比损失 + 重建损失双约束，兼顾语义一致性和动作保真度
3. **实机验证充分**: 在真实物理环境（非仿真）的多种末端执行器上验证，说服力强

### 局限性

1. **观测不对称问题**: 不同本体的观测配置不同（如腕部相机），影响部分本体的跨本体迁移效果
2. **动作空间覆盖有限**: 仅验证了三种末端执行器，未涵盖双臂、全身等更复杂场景
3. **隐空间泛化未完全验证**: 未系统评估向训练时未见过的新本体迁移的能力

### 潜在改进方向

1. 引入统一的观测标准（如多视角融合）以消除本体间观测不对称
2. 结合大规模互联网人手视频数据，进一步扩大对比训练数据规模
3. 探索隐空间的分层设计，分别对齐全局任务语义和局部关节运动

### 可复现性评估

- [ ] 代码开源（项目主页未显示 GitHub 代码链接）
- [ ] 预训练模型（未公开）
- [x] 训练细节完整（论文描述了三阶段训练流程）
- [ ] 数据集可获取（专有演示数据，未公开）

---

## 关联笔记

### 基于

- [[Diffusion Policy]]: 核心策略骨干，LAD 在其隐空间版本上构建
- [[DDPM]]: 扩散模型训练目标（去噪分数匹配）
- [[MANO]]: 人手参数化模型，作为跨本体对齐的中间表示

### 对比

- [[Diffusion Policy]]: 单一本体基线，LAD 在跨本体场景下超越该基线

### 方法相关

- [[InfoNCE 损失]]: 核心对比训练目标
- [[对比学习]]: 隐空间语义对齐的理论基础
- [[Action Chunking]]: 多步动作预测机制，扩散策略推理时使用
- [[Cross-Embodiment Learning]]: 本文核心任务范式
- [[Dex-Retargeting]]: 生成跨本体对齐训练数据的工具

### 硬件/数据相关

- [[Faive Hand]]: 16 DoF 仿人机器手，ETH Zürich 开发
- [[mimic Hand]]: 16 DoF 腱驱动灵巧机器手，mimic robotics 开发
- [[Franka Emika]]: 7 DoF 机械臂平台，搭配平行夹爪作为基准本体

---

## 速查卡片

> [!summary] Latent Action Diffusion (LAD)
> - **核心**: 用对比学习对齐跨本体动作到共享隐空间，实现单策略多机器人控制
> - **方法**: 四阶段（Dex-Retargeting → 对比编码器训练 → 解码器重建+编码器微调 → 隐空间扩散策略）
> - **结果**: 真实拾放任务最高 +25.3% 成功率，支持 Faive/mimic/Franka 三种本体
> - **代码**: 暂未公开（项目主页：https://mimicrobotics.github.io/lad/）

---

*笔记创建时间: 2026-06-10*
