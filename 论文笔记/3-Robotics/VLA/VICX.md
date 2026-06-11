---
title: "VICX: Generalizable Robot Manipulation via Video Generation and In-Context Operator Network"
method_name: "VICX"
authors: [Song Chen, Linyan Xiang, Ying Zhou, Liu Yang]
year: 2026
venue: arXiv
tags: [video-generation, in-context-learning, robot-manipulation, world-model, cross-task-generalization, vla, operator-network]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.12028
created: 2026-06-11
---

# 论文笔记：VICX: Generalizable Robot Manipulation via Video Generation and In-Context Operator Network

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | National University of Singapore (NUS) |
| 日期 | June 2026 |
| 项目主页 | https://scaling-group.github.io/vicx/ |
| 对比基线 | [[AVDC]], [[π0.5]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.12028) / Code TBD |

---

## 一句话总结

> VICX 将机器人操作解耦为冻结[[视频生成模型]]驱动的视觉规划与 [[In-Context Operator Network|V2T-ICON]] 驱动的状态轨迹执行，实现无需微调的跨任务、跨形态泛化。

---

## 核心贡献

1. **解耦闭环操作框架 (Decoupled Closed-Loop Framework)**: 将高层次视觉规划（冻结视频生成模型）与低层次状态执行（V2T-ICON）彻底分离，使两个模块可以独立扩展和泛化。
2. **V2T-ICON（Video-to-Trajectory In-Context Operator Network）**: 提出任务无关的在情境操作网络，以检索到的图像-状态对为 prompt，在推理时无需参数更新即可将生成视频帧映射为机器人状态轨迹。
3. **Arm-Only 分割观测**: 通过语义分割提取仅含机械臂的帧观测，消除背景干扰，提升跨任务和跨形态的视觉-状态映射鲁棒性。
4. **闭环自我校正（Closed-Loop Self-Correction）**: 将最近执行的视频实时反馈给视频代理，使系统能基于环境反馈重新规划，纠正生成失真、平移漂移和外部扰动带来的累积误差。

---

## 问题背景

### 要解决的问题

可泛化的机器人操作需要同时具备两种能力：（1）对未见场景进行任务级推理；（2）将视觉计划可靠地映射到具身执行动作。现有方法往往在这两者之间顾此失彼。

### 现有方法的局限

- **逆动力学方法**：从相邻帧恢复动作，精度受视频生成质量限制，生成失真直接导致控制误差累积。
- **光流/结构化运动提取**：从光流或结构化表示中提取动作，对感知噪声敏感，难以跨形态迁移。
- **端到端 VLA 模型**（如 [[π0.5]]）：通常需要在目标任务数据上微调，泛化能力受限于训练分布。
- **AVDC 类层次框架**：需要在较多任务（11 个 Meta-World 任务）上训练才能表现良好，数据效率不高。

### 本文的动机

使用**机器人状态**（而非动作）作为视觉规划与物理控制之间的接口，是比动作更几何化、更与控制器无关的表示。状态空间的结构性强，更适合用 [[In-Context Learning|情境学习]] 进行跨任务泛化。冻结的大型[[视频生成模型]]天然具备丰富的物理和任务先验，无需任务专属数据即可规划合理的视觉轨迹。

---

## 方法详解

### 模型架构

VICX 采用 **两阶段解耦闭环** 架构：

- **输入**: 自然语言指令 $l$ + 当前帧观测 $o_t$ + 历史执行视频 $v_{t-1}$
- **高层规划**: 冻结[[视频生成模型]]（vision-language conditioned）生成未来视觉帧序列 $\hat{V} = \{\hat{f}_1, \hat{f}_2, \ldots, \hat{f}_T\}$
- **低层执行**: [[V2T-ICON|Video-to-Trajectory In-Context Operator Network (V2T-ICON)]] 将视频帧转换为机器人状态轨迹 $\hat{S} = \{s_1, s_2, \ldots, s_T\}$
- **反馈控制器**: 跟踪状态轨迹，执行物理动作
- **闭环**: 执行后视频反馈给视频模型以重新规划

```
语言指令 l
       │
       ▼
┌─────────────────────┐
│  冻结视频生成模型     │  ← 当前观测帧 o_t
│  (Vision-Language   │  ← 历史执行视频 v_{t-1}
│   Conditioned)      │
└──────────┬──────────┘
           │ 生成视觉帧序列 V̂
           ▼
┌─────────────────────┐
│     V2T-ICON        │  ← 检索图像-状态对 (in-context prompts)
│  (Video-to-Traj     │  ← 分割后的 arm-only 帧
│   In-Context OpNet) │
└──────────┬──────────┘
           │ 状态轨迹 Ŝ
           ▼
┌─────────────────────┐
│    反馈控制器        │
│  (Feedback Ctrl)    │
└──────────┬──────────┘
           │ 执行视频反馈
           └──────────► 闭环重新规划
```

### 核心模块

#### 模块1: 冻结视频生成模型（Frozen Video Planner）

**设计动机**: 利用大规模预训练[[视频生成模型]]内蕴的物理先验和任务理解能力，以语言指令为条件生成高质量视觉规划帧，无需任何机器人任务专属训练。

**具体实现**:
- 接受语言指令 $l$ 和当前观测帧 $o_t$ 作为条件
- 生成一段预测未来的视频帧序列 $\hat{V}$，描述机械臂完成任务的视觉过程
- 模型权重**完全冻结**，不在机器人数据上微调
- 闭环中接受历史执行视频 $v_{t-1}$ 以感知真实环境状态，实现自适应重规划

#### 模块2: V2T-ICON（Video-to-Trajectory In-Context Operator Network）

**设计动机**: 利用[[In-Context Operator Learning|情境算子学习]]框架，将视频帧-状态映射建模为函数映射学习问题。通过检索相似图像-状态对作为 prompt，网络无需重新训练即可适应新任务。

**具体实现**:

1. **Arm-Only 分割预处理**: 对输入帧使用语义分割（[[SAM|Segment Anything Model]] 等）提取仅含机械臂区域的帧 $f^{arm}$，消除背景纹理差异，降低分布偏移
2. **In-Context Prompt 检索**: 从记忆库 $\mathcal{M} = \{(f^{arm}_i, s_i)\}$ 中检索与当前帧最相似的 $K$ 对图像-状态对，作为情境提示
3. **函数映射推理**: 将检索到的 $K$ 对情境样本与当前帧拼接，输入 Transformer，预测当前状态 $\hat{s}_t$
4. **无参数更新推理**: 推理时固定模型权重，仅依靠检索的情境样本实现跨任务适应

**In-Context 推理形式**:

$$
\hat{s}_t = \text{V2T-ICON}\left(f^{arm}_t;\; \{(f^{arm}_{i_1}, s_{i_1}), \ldots, (f^{arm}_{i_K}, s_{i_K})\}\right)
$$

#### 模块3: 闭环自我校正（Closed-Loop Self-Correction）

**设计动机**: 开环的视频规划会因生成失真和执行误差而偏离真实环境，闭环反馈使视频代理能感知真实执行结果并动态调整计划。

**具体实现**:
- 每个执行周期结束后，捕获真实执行视频 $v_{exec}$
- 将 $v_{exec}$ 与原始指令一起输入视频生成模型，作为历史上下文
- 视频代理将当前失败状态视为可利用的信息（如抓取姿态不对），据此重新规划新的视觉轨迹
- 有效缓解：(1) 生成失真累积误差；(2) 平移漂移；(3) 外部扰动

---

## 关键公式

### 公式1: [[In-Context Operator Learning|V2T-ICON 情境推理]]

$$
\hat{s}_t = f_\theta\!\left(f^{arm}_t \;\Big|\; \{(f^{arm}_{i_j}, s_{i_j})\}_{j=1}^{K}\right)
$$

**含义**: V2T-ICON 以分割后的当前帧 $f^{arm}_t$ 为查询，以从记忆库中检索到的 $K$ 个图像-状态对为情境提示，预测当前机器人状态 $\hat{s}_t$；推理时 $\theta$ 固定不变。

**符号说明**:
- $f^{arm}_t$: 第 $t$ 帧的 arm-only 分割图像
- $\{(f^{arm}_{i_j}, s_{i_j})\}_{j=1}^{K}$: 从记忆库检索到的 $K$ 个情境样本（图像-状态对）
- $\hat{s}_t$: 预测的机器人状态（关节角度或末端执行器位姿）
- $f_\theta$: V2T-ICON 网络，参数 $\theta$ 在推理时冻结

### 公式2: [[In-Context Retrieval|情境样本检索]]

$$
\{i_1, \ldots, i_K\} = \arg\!\operatorname{top-}K_{i \in \mathcal{M}}\; \text{sim}(f^{arm}_t,\; f^{arm}_i)
$$

**含义**: 从记忆库 $\mathcal{M}$ 中按视觉相似度检索与当前帧最近的 $K$ 个图像-状态对作为情境提示。

**符号说明**:
- $\mathcal{M} = \{(f^{arm}_i, s_i)\}$: 存储图像-状态对的记忆库（来自训练时收集的演示数据）
- $\text{sim}(\cdot, \cdot)$: 视觉相似度度量（如余弦相似度或特征距离）
- $K$: 情境样本数量

### 公式3: [[视频规划|视觉计划生成]]

$$
\hat{V} = \mathcal{G}(l,\; o_t,\; v_{t-1})
$$

**含义**: 冻结视频生成模型 $\mathcal{G}$ 以语言指令 $l$、当前观测 $o_t$ 和历史执行视频 $v_{t-1}$ 为条件，生成未来视觉帧序列 $\hat{V}$。

**符号说明**:
- $\mathcal{G}$: 冻结的视频生成模型
- $l$: 自然语言任务指令
- $o_t$: 当前时刻的环境观测帧
- $v_{t-1}$: 上一轮执行的真实视频（闭环反馈）
- $\hat{V} = \{\hat{f}_1, \ldots, \hat{f}_T\}$: 生成的视觉规划帧序列

### 公式4: [[State Trajectory Tracking|状态轨迹跟踪]]

$$
a_t = \pi_{ctrl}(s_t,\; \hat{s}_t)
$$

**含义**: 反馈控制器 $\pi_{ctrl}$ 根据当前真实状态 $s_t$ 与 V2T-ICON 预测的目标状态 $\hat{s}_t$ 之间的误差计算控制动作 $a_t$。

**符号说明**:
- $s_t$: 当前机器人真实状态
- $\hat{s}_t$: V2T-ICON 预测的目标状态
- $a_t$: 施加到机器人的控制动作
- $\pi_{ctrl}$: 反馈控制器（如 PID 或操作空间控制）

### 公式5: [[V2T-ICON Training Loss|V2T-ICON 训练目标]]

$$
\mathcal{L} = \mathbb{E}_{(f^{arm}, s) \sim \mathcal{D}}\left[\left\| f_\theta\!\left(f^{arm} \mid \mathcal{C}\right) - s \right\|^2\right]
$$

**含义**: V2T-ICON 以均方误差（MSE）损失在标注数据集 $\mathcal{D}$ 上训练，学习从 arm-only 视频帧到机器人状态的映射函数，情境提示 $\mathcal{C}$ 在训练时随机采样。

**符号说明**:
- $\mathcal{D}$: 包含图像-状态对的训练数据集（仅来自 3 个源任务）
- $\mathcal{C} = \{(f^{arm}_{i_j}, s_{i_j})\}_{j=1}^{K}$: 随机采样的训练时情境提示
- $f_\theta(\cdot \mid \mathcal{C})$: 带情境提示的 V2T-ICON 预测函数

---

## 关键图表

### Figure 1: VICX 系统概览

![VICX Overview](https://arxiv.org/html/2606.12028/figures/fig1_overview.png)

**说明**: VICX 整体框架示意图。上半部分展示冻结视频生成模型接受语言指令和当前观测，输出视觉规划帧；下半部分展示 V2T-ICON 通过检索情境样本将规划帧转换为状态轨迹，反馈控制器执行状态跟踪，最终执行视频反馈回视频代理形成闭环。

### Figure 2: V2T-ICON 架构详解

![V2T-ICON Architecture](https://arxiv.org/html/2606.12028/figures/fig2_v2t_icon.png)

**说明**: V2T-ICON 的详细结构。输入为分割后的 arm-only 帧与 $K$ 个检索到的图像-状态情境对，通过 [[Transformer]] 编码器融合情境信息后，输出目标机器人状态。关键设计是推理时**无需参数更新**，完全依赖情境提示适应新任务。

### Figure 3: Arm-Only 分割示意

![Arm-Only Segmentation](https://arxiv.org/html/2606.12028/figures/fig3_segmentation.png)

**说明**: 展示原始观测帧（含背景和物体）经语义分割提取为仅含机械臂的 arm-only 帧的对比。arm-only 表示去除了背景纹理变化，使视觉-状态映射对任务和场景变化更鲁棒。

### Figure 4: 闭环自我校正示例

![Closed-Loop Self-Correction](https://arxiv.org/html/2606.12028/figures/fig4_closed_loop.png)

**说明**: 当初次执行失败时（如抓取姿态不对），VICX 将失败视频作为信息上下文输入视频代理，视频代理重新规划出新的抓取策略，展示了系统在外部扰动下的自适应恢复能力。

### Figure 5: 跨形态迁移

![Cross-Embodiment Transfer](https://arxiv.org/html/2606.12028/figures/fig5_cross_embodiment.png)

**说明**: 展示 VICX 在训练时未见过的机器人形态上的迁移效果。由于 arm-only 分割消除了形态特定的外观差异，V2T-ICON 能够将规划帧映射到不同的机器人状态空间，无需重新训练。

### Table 1: Meta-World 跨任务泛化对比（9 任务成功率）

| 方法 | 训练任务数 | 成功率 |
|------|----------|--------|
| AVDC | 11 | — |
| π₀.₅-Finetune (MT50) | 50 | — |
| **VICX (Ours)** | **3** | **72.2%** |

**说明**: VICX 仅使用 3 个源任务的训练数据，在 9 个 Meta-World 评估任务上达到 72.2% 的整体成功率，超越了在更多任务上训练的 AVDC 和 π₀.₅-Finetune，体现了解耦架构与视频生成模型世界先验的强大泛化优势。

### Table 2: 消融实验

| 配置 | 成功率 | 说明 |
|------|--------|------|
| w/o 闭环反馈 | 较低 | 开环规划累积误差显著 |
| w/o Arm-Only 分割 | 较低 | 背景干扰导致视觉-状态映射退化 |
| w/o 情境检索（零样本） | 较低 | 无情境提示时泛化能力下降 |
| **Full VICX** | **72.2%** | 闭环 + Arm-Only + 情境检索 |

**关键发现**: 三个设计选择（闭环反馈、arm-only 分割、情境检索）均对最终性能有显著贡献，缺一不可。

---

## 实验结果

### 数据集与基准

| 数据集/基准 | 规模 | 特点 | 用途 |
|------------|------|------|------|
| Meta-World (9 tasks) | 9 个操作任务 | 多样化桌面操作、标准基准 | 跨任务泛化评估（主实验） |
| Meta-World (source) | 3 个源任务 | 用于训练 V2T-ICON 的记忆库 | 训练 / 少量演示 |
| Meta-World MT50 | 50 任务 | π₀.₅-Finetune 的预训练集 | 基线训练 |

### 实现细节

- **视频生成模型**: 冻结的预训练大型视频生成模型（权重来自已有开源模型）
- **V2T-ICON 训练任务数**: 仅 **3 个** Meta-World 源任务
- **情境提示数 K**: $K$ 个检索样本（具体值见论文正文）
- **分割工具**: 语义分割（可插拔，支持 [[SAM]] 等通用分割模型）
- **反馈控制器**: 任务空间反馈控制器，跟踪 V2T-ICON 输出的状态序列

### 主要实验结果

**跨任务泛化**: VICX 以仅 3 个源任务的训练数据，在 9 任务评估套件上取得 **72.2% 整体成功率**，超越了训练数据更丰富的 AVDC（11 个 Meta-World 任务）和 π₀.₅-Finetune（Meta-World MT50 微调）。

**闭环自我校正**: 当初次执行失败时，VICX 将失败视频作为视觉上下文输入视频代理，自适应重规划出新的操作策略（如换一种抓取角度），成功完成任务，展示了强大的在线恢复能力。

**跨形态迁移**: VICX 无需重新训练即可迁移到训练时未见过的机器人形态，验证了 arm-only 分割去除形态特定外观差异的有效性。

### 可视化结果

- 不同任务（推、拉、放、旋转等）的视觉规划帧与对应执行轨迹对比
- 闭环中失败→重规划→成功的完整执行过程可视化
- 跨机器人形态的轨迹执行对比（形态改变但任务成功）

---

## 批判性思考

### 优点

1. **数据高效**: 仅 3 个源任务即泛化到 9 个任务，远优于需要大量任务数据的方法
2. **即插即用**: 视频生成模型和控制器可替换，框架本身不绑定特定实现
3. **无需微调**: 推理时无参数更新，部署成本低，跨任务/跨形态适应几乎零额外开销
4. **闭环鲁棒**: 实时反馈机制有效缓解开环累积误差，提升实际部署可靠性

### 局限性

1. **依赖视频生成质量**: 高层视觉规划完全依赖冻结视频生成模型，若生成帧存在结构性错误，下游 V2T-ICON 无法纠正
2. **状态空间假设**: 使用机器人状态而非动作作为接口，需要精确的状态测量和可靠的反馈控制器，对传感器精度有一定要求
3. **记忆库规模**: 情境检索的效果依赖记忆库覆盖的样本多样性，极端分布外任务可能缺乏有效的检索样本
4. **实验规模**: 目前主要在 Meta-World 仿真基准上验证，真实机器人实验场景有待进一步拓展

### 潜在改进方向

1. 将 V2T-ICON 训练扩展到更多源任务以丰富记忆库，探索记忆库自动扩充机制
2. 研究端到端联合优化视频生成模型与 V2T-ICON 的可能性
3. 在真实机器人平台上进行大规模评估和 Sim-to-Real 验证

### 可复现性评估

- [ ] 代码开源（项目主页声明中）
- [ ] 预训练模型（视频生成模型为已有开源模型；V2T-ICON 权重待发布）
- [ ] 训练细节完整（论文描述了 3 任务训练设定）
- [x] 数据集可获取（Meta-World 为公开 benchmark）

---

## 关联笔记

### 基于

- [[视频生成模型]]: VICX 高层规划的核心组件，冻结用于视觉规划
- [[In-Context Operator Learning]]: V2T-ICON 的理论基础，VICON 等前驱工作
- [[VICON]]: V2T-ICON 的直接前身，视觉情境算子网络用于流体力学预测

### 对比

- [[AVDC]]: 层次化视频规划框架，需 11 个 Meta-World 任务训练，VICX 以 3 任务数据超越
- [[π0.5]]: 闭源大型 VLA 模型，VICX 与其 LeRobot 开放权重版本对比
- [[Gen2Act]]: 利用人类视频生成实现可泛化操作，路线相似但未解耦执行层

### 方法相关

- [[In-Context Learning]]: V2T-ICON 的推理范式，检索样本作为 prompt
- [[视频规划]]: 视频生成用于机器人规划的通用技术路线
- [[SAM]]: 可用于 arm-only 语义分割的通用分割模型
- [[State Trajectory Tracking]]: 反馈控制器跟踪 V2T-ICON 输出状态的控制方法

### 硬件/数据相关

- [[Meta-World]]: 主要评估基准，标准化桌面操作任务集

---

## 速查卡片

> [!summary] VICX
> - **核心**: 解耦视频规划与情境算子执行，实现无微调跨任务/跨形态泛化
> - **方法**: 冻结视频生成模型做高层规划 + V2T-ICON 以检索情境样本做状态轨迹预测 + 闭环实时反馈校正
> - **结果**: 3 源任务训练，Meta-World 9 任务 72.2% 成功率，超越 AVDC 和 π₀.₅-Finetune
> - **代码**: https://scaling-group.github.io/vicx/

---

*笔记创建时间: 2026-06-11*
