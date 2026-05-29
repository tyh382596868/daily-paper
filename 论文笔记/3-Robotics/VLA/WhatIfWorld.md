---
title: "What-If World: A Causal Benchmark for General World Models in Embodied Scenarios"
method_name: "WhatIfWorld"
authors: [Kunlin Cai, Rui Song, Jinghuai Zhang, Kaiyuan Zhang, Pranav Bodapati, Alicia Yu, Fnu Suya, Mohammad Rostami, Jiaqi Ma, Yuan Tian]
year: 2026
venue: arXiv
tags: [benchmark, causal-evaluation, world-model, video-generation, embodied-ai, vlm-as-judge, contrastive-intervention]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.27589v1
created: 2026-05-28
---

# 论文笔记：What-If World: A Causal Benchmark for General World Models in Embodied Scenarios

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | UCLA, University of Tennessee, Amazon |
| 日期 | May 2026 |
| 项目主页 | 暂未公开 |
| 对比基线 | [[VBench]], EvalCrafter, VideoPhy, PhyGenBench, Physion++ |
| 链接 | [arXiv](https://arxiv.org/abs/2605.27589) / [HTML](https://arxiv.org/html/2605.27589v1) |

---

## 一句话总结

> 首个用"反事实对比对"评测视频 [[World Model|世界模型]] 因果敏感度的基准，揭示当前 SOTA 模型在 paired 物理一致性上普遍不到 52%。

---

## 核心贡献

1. **Contrastive Bottleneck（对比瓶颈）的揭示**: 指出现有 per-video 评测体系（如 [[VBench]]、EvalCrafter）会让"两段对 brake gently / brake hard 几乎相同的视频"同时通过——本文用 paired 评测把这种潜在失败模式暴露出来。
2. **Unified Causal Taxonomy（统一因果分类法）**: 在驾驶和机械臂操控两个 [[Embodied AI|具身]] 域上抽取 6 个共享的物理原语（surface friction / material & medium / obstacle configuration / spatial alignment / force-degree / temporal sequencing），覆盖环境域 $D_1$ 与交互域 $D_2$。
3. **APEO 评测框架**: 提出 Adherence / Physics / Environment / Outcome 四维度评分，single-video 与 paired 双模式，并用 [[VLM-as-Judge]]（Gemini 3.1 Pro）实现自动化打分，人类一致率 82.30% 接近 inter-human 的 84.03%。
4. **319 对 prompt + 9 模型 leaderboard**: 在 [[nuScenes]] 与 [[DROID|DROID 数据集]] 上构造 319 个 $(x_0, p^+, p^-)$ 三元组，评测 9 个 SOTA 模型，最高 pAvg 仅 51.7%。

---

## 问题背景

### 要解决的问题

视频生成模型被越来越多地当作 [[Action-Conditioned World Model|动作条件世界模型]] 用于 model-based planning 与策略训练，但现有评测只检查"单段视频是否物理合理"。这无法回答关键问题：**当输入动作变化时，生成的 rollout 是否按物理预期发生对应变化**？

作者用一个极具说服力的例子说明：模型可以渲染出运动学完美的"急刹车"视频，却在 prompt 改为"缓刹车"时输出几乎相同的画面——这种 rollout **无法用于动作比较**，因此无法支持因果推理或规划。

### 现有方法的局限

- [[VBench]] / EvalCrafter / VideoPhy 等单视频评测无法捕获跨视频的因果差异；
- CLEVRER、CRAFT、Physion++ 等因果视频问答聚焦"视觉感知是否理解因果"，不评测**生成模型自己产生的 rollout 是否因果一致**；
- 缺少跨 embodiment（driving + manipulation）的统一因果原语体系，导致各 benchmark 难以横向比较。

### 本文的动机

只有把"单段合理"与"对差异有响应"分开来量化，才能暴露生成式世界模型的 [[Contrastive Bottleneck|对比瓶颈]]。本文用真实数据帧作为初始状态锚点（state fixation），用最小差异的 prompt 对作为干预（contrastive intervention），用 paired VLM 评分作为度量。

---

## 方法详解

### 整体架构

[[WhatIfWorld]] 是一个三阶段评测流水线，对应论文 Figure 1 的 "what to test → how to test → how to evaluate"：

- **输入**: 真实数据集 [[nuScenes]] / [[DROID|DROID 数据集]] 的视频片段
- **核心三阶段**:
  - Stage 1: 在 6 个 [[Causal Primitive|物理原语]] 下筛选具备相应 affordance 的片段，抽取动作起始帧 $x_0$ 作为 **causal branching point**
  - Stage 2: 围绕 $x_0$ 撰写 [[Contrastive Prompt Pair|对比 prompt 对]] $(p^+, p^-)$，仅在单一物理变量上不同，且**不透露结局**
  - Stage 3: 把 $(x_0, p^+)$、$(x_0, p^-)$ 分别输入待测模型得到 $(V_+, V_-)$，再交给 [[VLM-as-Judge]] 用 APEO 四维度打分
- **输出**: 9 项分数 $\{A_s, P_s, E_s, sAvg, A_p, P_p, E_p, O_p, pAvg\}$ 构成 leaderboard

### 核心模块

#### 模块1: Unified Causal Taxonomy（统一因果分类法）

**设计动机**: 让两个 embodiment 共享同一套物理原语，使评测可以横跨驾驶与机器人，且每个原语满足三个准则——**physically fundamental（物理基本）、embodiment-shared（跨形态共享）、operationally isolable（可独立变更）**。

**具体构造**: 分为环境域 $D_1$ 与交互域 $D_2$，每域 3 个原语：

- $D_1$ 环境域：
  - [[Surface Friction|表面摩擦]]：干燥沥青 vs 冰面 / 玻璃 vs 橡胶
  - [[Material & Medium|材料与介质]]：晴天 vs 雨雪 / 刚体木块 vs 可变形海绵
  - [[Obstacle Configuration|障碍布置]]：无障碍 vs 静态障碍 / 抓取路径有无遮挡
- $D_2$ 交互域：
  - [[Spatial Alignment|空间对齐]]：横向车道偏移 / 夹爪闭合位置
  - [[Force-Degree|力度/程度]]：加减速幅度 / 推力或抓握力
  - [[Temporal Sequencing|时序顺序]]：动作相对其他 agent 的时机 / 闭合夹爪在到达目标前还是后

#### 模块2: Benchmark Construction Pipeline（基准构造流水线）

**Stage 1 — State Fixation via Real Frames**：从 [[nuScenes]]（1400+ 多模态驾驶视频）与 [[DROID|DROID 数据集]]（大规模 in-the-wild 机械臂）抽取片段，定位"动作起始前一帧" $x_0$，该帧固定了几何、材料、agent 位置、光照——保证"未来轨迹完全取决于后续动作"。

**Stage 2 — Initial Condition Anchoring**：跨两段视频共享三个常量：
- 视觉常量：单帧 $x_0$
- Prompt 常量：相机视角描述、初始场景状态（如 ego 速度等不可视信息）

**Stage 3 — Contrastive Prompt Pair Creation**：作者撰写 $p^+$ 与 $p^- = p^+[v \leftarrow v^-]$，**仅修改目标物理变量描述的那一段**，且都不剧透结局，迫使模型 simulate 而不是 retrieve。

#### 模块3: APEO 评测框架

**设计动机**: 同时在 single-video 与 paired 维度衡量"个体合理性"与"差异响应能力"，把 [[Contrastive Bottleneck|对比瓶颈]] 量化。

**四维度定义**:

- **Adherence (A)** — 动作执行：是否对目标 entity 执行了 prompt 指定动作；paired 要求两段视频动作在视觉上可区分且差异方向与 prompt 一致。
- **Physics (P)** — 物理一致性：单段是否遵守基本物理（无 teleportation、morphing、ghost force）；paired 要求干预前轨迹对齐、干预后按物理规律分叉。
- **Environment (E)** — 场景保真：背景、相机视角、非目标物体的稳定性；paired 要求两视频背景近似一致，差异可归因于干预。
- **Outcome (O)** — **paired 独有**：两视频终态可测量地不同，且差异在方向与受影响实体上均符合预期。

**评分细节**: 每个维度按 binary（0/1）由 [[VLM-as-Judge]]（Gemini 3.1 Pro）回答 primitive-conditioned 问题；single 维度对两视频取均值。

---

## 关键公式

### 公式1: [[Causal Branching Point|因果分叉点]] 上的模型 rollout

$$
(V_+, V_-) = (\mathcal{M}(x_0, p_+), \mathcal{M}(x_0, p_-))
$$

**含义**: 待测视频生成模型 $\mathcal{M}$ 在同一锚定帧 $x_0$ 与一对最小差异 prompt 上分别生成两段 rollout，构成评测单元。

**符号说明**:
- $x_0$: 动作起始前的真实帧，固定几何、材料、光照
- $p_+, p_-$: contrastive prompt pair，仅在目标物理变量上不同
- $V_+, V_-$: 模型生成的两段视频

### 公式2: [[Contrastive Prompt Pair|对比 prompt 对]] 构造

$$
p_- = p_+ [v \leftarrow v_-]
$$

**含义**: 负向 prompt 由正向 prompt 仅替换目标变量 $v$ 的描述 span 得到，其余内容（相机视角、初始状态、动作目标）保持完全一致。

**符号说明**:
- $v$: 目标物理变量（来自 6 个原语之一）
- $v_-$: 该变量的对照取值（例如 "brake hard" → "brake gently"）
- $[v \leftarrow v_-]$: 仅替换变量描述 span 的操作

### 公式3: [[APEO|sAvg 单视频综合分]]

$$
\mathrm{sAvg} = \frac{A_s + P_s + E_s}{3}
$$

**含义**: 三个 single-video 维度（Adherence / Physics / Environment，无 Outcome）取算术平均，衡量"每段视频自己是否合理"。

**符号说明**:
- $A_s, P_s, E_s$: single-video 模式下三个维度的通过率（%）

### 公式4: [[APEO|pAvg 对偶综合分（主指标）]]

$$
\mathrm{pAvg} = \frac{A_p + P_p + E_p + O_p}{4}
$$

**含义**: 四个 paired 维度取均值，是评测**因果敏感度**的主指标。一个模型若 $\mathrm{sAvg}$ 高但 $\mathrm{pAvg}$ 低，就掉进了 [[Contrastive Bottleneck|对比瓶颈]]。

**符号说明**:
- $A_p, P_p, E_p$: paired 模式下三个共享维度
- $O_p$: paired 独有的 Outcome 维度，衡量"终态分歧是否方向正确"

### 公式5: [[Contrastive Bottleneck|对比瓶颈]] 的代表性差距

$$
\Delta_{\text{bottleneck}} = P_s - P_p = 64.4\% - 12.2\% = 52.2\%
$$

**含义**: 以 HunyuanVideo-1.5 为例，单视频物理评分高达 64.4%，但 paired 物理仅 12.2%——说明模型能生成"看起来合物理"的画面，却无法在干预下产生**正确分叉**。这一 52.2 个百分点的鸿沟正是本文的核心发现之一。

---

## 关键图表

### Figure 1: What-If World Benchmark Overview / 总览

![Figure 1](https://arxiv.org/html/2605.27589v1/x1.png)

**说明**: 三阶段流水线总览——Stage 1 "what to test"（6 个物理原语的因果分类法），Stage 2 "how to test"（在 $x_0$ 锚定下生成 contrastive prompt pair），Stage 3 "how to evaluate"（APEO 四维度 + [[VLM-as-Judge]]）。

### Figure 2: Benchmark Construction Pipeline / 构造流水线

![Figure 2](https://arxiv.org/html/2605.27589v1/x2.png)

**说明**: Stage 1 在 [[nuScenes]] 与 [[DROID|DROID 数据集]] 上筛选 clip 并在动作起始点抽取 $x_0$ 作为 [[Causal Branching Point|因果分叉点]]；Stage 2 围绕 $x_0$ 撰写 $(p^+, p^-)$ 对，仅在单一物理变量上不同，且不透露 outcome。

### Figure 3: Contrastive Bottleneck Example (AD) / 驾驶场景对比示例

![Figure 3](https://arxiv.org/html/2605.27589v1/x3.png)

**说明**: 两段视频从同一帧 $x_0$ 出发，接收 Force/Degree 干预对（加速度动词不同）。模型若掉进 [[Contrastive Bottleneck|对比瓶颈]]，两段视频会高度相似——单看都合理，但**不响应 prompt 差异**。

### Figure 4: Contrastive Bottleneck Example (Robotic Arms) / 机械臂场景对比示例

![Figure 4](https://arxiv.org/html/2605.27589v1/x4.png)

**说明**: 机械臂版本，干预 Force/Pressure（接触强度），同样从同一 $x_0$ 出发。展示 paired 模式可暴露 single-video 模式看不到的物理失败。

### Table 1: APEO 评测框架

| 维度 | Single-video ($X_s$) | Paired ($X_p$) |
|------|---------------------|----------------|
| **Adherence (A)** | 视频对目标 entity 执行了 prompt 指定的动作 | 两段视频动作可视区分，且差异方向与 prompt 控制一致 |
| **Physics (P)** | 全程符合基本物理约束（无 teleport / morph / ghost force） | 干预前轨迹对齐，干预后按物理规律分叉 |
| **Environment (E)** | 背景、相机视角、非目标物体保持稳定 | 两段视频背景近似一致，可见差异可归因于干预 |
| **Outcome (O)** | — | **paired 独有**：终态可测量不同，分歧方向与受影响实体均与预期一致 |

**说明**: A/P/E 三个维度同时具备 single 与 paired 评分，O 仅 paired。综合分 $\mathrm{sAvg}=\mathrm{mean}(A_s, P_s, E_s)$；$\mathrm{pAvg}=\mathrm{mean}(A_p, P_p, E_p, O_p)$。

### Table 2: What-If World Benchmark Leaderboard（完整 9 模型分数）

| Model | $A_s$ | $P_s$ | $E_s$ | sAvg | $A_p$ | $P_p$ | $E_p$ | $O_p$ | **pAvg** | Rank |
|-------|------:|------:|------:|-----:|------:|------:|------:|------:|---------:|-----:|
| CogVideoX1.5 | 32.8 | 42.5 | 27.3 | 34.2 | 18.2 | 10.7 | 50.8 | 11.3 | **22.7** | 9 |
| Wan2.2-5B | 40.4 | 52.0 | 45.9 | 46.1 | 22.3 | 14.1 | 60.2 | 17.2 | **28.4** | 7 |
| HunyuanVideo-1.5 | 42.6 | 64.4 | 48.0 | 51.7 | 20.4 | 12.2 | 64.9 | 10.7 | **27.0** | 8 |
| Cosmos-Predict2 | 41.4 | 55.2 | 55.8 | 50.8 | 25.1 | 18.8 | 73.0 | 14.7 | **32.9** | 5 |
| Seedance 1.5 | 51.6 | 58.6 | 47.8 | 52.7 | 32.0 | 20.7 | 51.1 | 26.0 | **32.4** | 6 |
| Kling 3.0 | 54.1 | 65.5 | 66.5 | 62.0 | 34.8 | 23.8 | 74.9 | 21.0 | **38.6** | 4 |
| Seedance 2.0 | 63.0 | 74.0 | 71.5 | **69.5** | 41.7 | 28.8 | 64.9 | 32.0 | **41.8** | 3 |
| Veo 3.1 | 66.8 | 69.7 | 70.1 | 68.9 | 48.0 | 40.1 | 70.5 | 44.8 | **50.9** | 2 |
| **Grok Imagine** | **69.2** | 70.5 | 67.2 | 69.0 | **49.2** | **42.6** | 69.7 | **45.1** | **51.7** | **1** |

**说明**: 最高 pAvg 仅 51.7%（Grok Imagine），所有 9 个 SOTA 模型在因果干预上均显著失败。开源平均 27.8%，闭源平均 43.1%，差距 15.3 个百分点。**Seedance 2.0 是典型瓶颈案例**：sAvg 排第 1 但 pAvg 排第 3，说明"单段最美"并不等于"差异响应最好"。

### Table 3: Unified Causal Taxonomy（统一因果分类法）

| Domain | Primitive | Driving 实例 | Manipulation 实例 |
|--------|-----------|--------------|-------------------|
| $D_1$ 环境 | [[Surface Friction\|表面摩擦]] | 干燥沥青 vs 冰面 | 玻璃台面 vs 橡胶垫 |
| $D_1$ 环境 | [[Material & Medium\|材料与介质]] | 晴天 vs 雨雪 | 刚体木块 vs 可变形海绵 |
| $D_1$ 环境 | [[Obstacle Configuration\|障碍布置]] | 无障碍 vs 静态障碍 | 工作区清空 vs 抓取路径被挡 |
| $D_2$ 交互 | [[Spatial Alignment\|空间对齐]] | 横向车道偏移 | 夹爪闭合位置 |
| $D_2$ 交互 | [[Force-Degree\|力度/程度]] | 加减速幅度 | 推力 / 抓握力 |
| $D_2$ 交互 | [[Temporal Sequencing\|时序顺序]] | 动作相对其他 agent 的时机 | 夹爪闭合 vs 到达目标的先后 |

**说明**: 每个原语满足**物理基本 / 跨形态共享 / 可独立变更** 三个准则。

### Table 4: 按原语拆解的 Outcome 通过率（$O_p$）

| Primitive | $O_p$ (%) | 视觉显著性 |
|-----------|----------:|------------|
| [[Force-Degree\|力度/程度]] | 40.4 | 高（训练分布常见） |
| [[Spatial Alignment\|空间对齐]] | 30.9 | 中 |
| [[Material & Medium\|材料与介质]] | 23.9 | 中低 |
| [[Obstacle Configuration\|障碍布置]] | 15.5 | 低 |
| [[Temporal Sequencing\|时序顺序]] | 15.3 | 低 |
| [[Surface Friction\|表面摩擦]] | 14.2 | 最低（隐变量、轨迹差异微妙） |

**关键发现**: 模型表现强烈跟随**视觉显著性**而非**物理可解性**——可视的力度变化最容易过关，需要推断"地面有多滑"这种隐属性的则垫底。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[nuScenes]] | 1400+ 多模态视频 | 同步 camera/lidar/radar 的自动驾驶 | 抽取驾驶场景 $x_0$ |
| [[DROID\|DROID 数据集]] | 大规模 in-the-wild 机械臂 | 真实操作数据 | 抽取机械臂场景 $x_0$ |
| **What-If World** | **319 paired test instances** | 6 原语均衡覆盖 | **本文构造的评测集** |
| Human-annotated 子集 | 421 样本 | 用于 VLM 判官校准 | VLM judge 验证 |

### 实现细节

- **被测模型**: 4 个开源（HunyuanVideo-1.5 8.3B / Cosmos-Predict2 2B / CogVideoX1.5-5B / Wan2.2-5B）+ 5 个闭源（Grok Imagine / Veo 3.1 / Seedance 2.0 / Seedance 1.5 / Kling 3.0）
- **VLM Judge**: Gemini 3.1 Pro，输入 $(V_+, V_-, p^+, p^-)$，对每个维度回答 binary primitive-conditioned 问题
- **人类一致率**: VLM judge 与人类标注一致 **82.30%**，inter-human 一致 **84.03%**——VLM 评分已接近人类水平
- **零微调**: 所有模型直接以官方权重 + 默认参数评测

### 可视化结果

论文 Figure 3 / Figure 4 展示典型 contrastive bottleneck：模型在两个仅差"acceleration verb"或"contact intensity"的 prompt 下，输出几乎不可区分的视频——单看都合格，paired 视角下露馅。

---

## 批判性思考

### 优点

1. **问题诊断精准**: 把"看似合理但对差异不响应"这个隐性失败模式提炼为 [[Contrastive Bottleneck|对比瓶颈]]，并设计了可量化的 paired 指标。这是从"内容好坏"评测迈向"作为模拟器是否可用"的关键一步。
2. **跨 embodiment 的统一原语体系**: 6 个原语同时覆盖驾驶与机器人，避免了原先 benchmark 各自为政、无法比较的问题；同时也方便后续扩展到 humanoid、UAV 等场景。
3. **VLM judge 一致率扎实**: 82.3% vs 84.0%（inter-human）是相当强的可信度证据，使评测可大规模复用。
4. **leaderboard 数据公开 + 9 个 SOTA**: 给后续 world model 开发者明确的目标曲面，也明确了"哪些原语最难"。

### 局限性

1. **319 个 instance 规模偏小**: 6 原语 × 2 embodiment 下，平均每个 cell 不到 30 例，统计噪声可能较大；后续若要做超大模型间的细分排名需扩样本。
2. **仅评测视频生成而非闭环 rollout**: 真正的 world model 应支持多步 action-conditioned rollout，本文限于单次 $(x_0, p) \to V$；未涉及多步累计误差。
3. **依赖 prompt 描述动作**: 真实下游用法（VLA / 规划）通常以连续动作向量为输入，而非自然语言；prompt-based 评测与实际接口存在 gap。
4. **VLM judge 仍是单一来源**: 虽然校准过，但 Gemini 3.1 Pro 自身的物理偏见可能影响细分维度，长远应加入 multi-judge ensemble。

### 潜在改进方向

1. **多步 paired rollout 评测**: 把单次 $(x_0, p) \to V$ 升级为 $K$ 步 action-conditioned rollout，引入 [[Causal Branching Point|因果分叉点]] 序列。
2. **加入连续动作接口**: 替换或并存 prompt 输入，与 [[VLA]] / [[Diffusion Policy|扩散策略]] 评测接轨。
3. **细化 surface friction / temporal sequencing 子任务**: 这两类得分最低，需要专门设计可推断隐属性的 prompt 模板。
4. **multi-judge / physics-engine ground truth**: 用物理引擎重渲染对照轨迹，做 grounded 评价。

### 可复现性评估

- [x] 论文公开（arXiv 2605.27589）
- [ ] 代码开源（暂未提供）
- [ ] 数据集 319 paired 公开
- [x] 评测协议描述完整
- [x] VLM judge 模型与 prompt 框架可复制

---

## 关联笔记

### 基于

- [[VBench]]: 经典单视频评测，本文明确指出其 paired-blind 的局限
- [[VLM-as-Judge]]: 评测自动化的核心机制，本文复用并做人类一致率校准

### 对比

- [[WBench]]: 多轮交互式 world model 评测（导航向）；与本文的 paired contrastive 评测视角互补
- [[EA-WM]]: 具身世界模型框架；本文为其提供因果敏感度量纲
- [[JEPA-WM]]: 表征学习路径的 world model；本文揭示生成式路径的 contrastive bottleneck

### 方法相关

- [[World Model|世界模型]]: 总主题
- [[Action-Conditioned World Model|动作条件世界模型]]: 评测对象的核心定位
- [[Embodied AI|具身智能]]: 应用域
- [[Contrastive Bottleneck|对比瓶颈]]: 本文提出的核心现象
- [[Causal Branching Point|因果分叉点]]: 数据构造的关键概念
- [[Contrastive Prompt Pair|对比 prompt 对]]: 评测单元
- [[Causal Primitive|物理原语]]: 6 维度分类基础
- [[APEO]]: 四维度评分框架

### 硬件/数据相关

- [[nuScenes]]: 驾驶域帧来源
- [[DROID|DROID 数据集]]: 机械臂域帧来源

---

## 速查卡片

> [!summary] What-If World
> - **核心**: 用 paired contrastive prompt + APEO 四维度，量化视频世界模型的因果敏感度
> - **方法**: 在 $x_0$ 锚定下，构造仅差单一物理变量的 $(p^+, p^-)$，让 VLM judge 对 $(V_+, V_-)$ 打 binary 分
> - **结果**: 9 个 SOTA 模型 pAvg 最高 51.7%（Grok Imagine），开源仅 ~28%；揭示 Contrastive Bottleneck（如 HunyuanVideo $P_s=64.4\%$ 对 $P_p=12.2\%$，差 52.2pt）
> - **代码**: 暂未公开

---

*笔记创建时间: 2026-05-28*
