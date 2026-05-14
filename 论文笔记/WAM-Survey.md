---
title: "World Action Models: The Next Frontier in Embodied AI"
method_name: "WAM-Survey"
authors: [Siyin Wang, Junhao Shi, Zhaoyang Fu, Xinzhe He, Feihong Liu, Chenchen Yang, Yikang Zhou, Zhaoye Fei, Jingjing Gong, Jinlan Fu, Mike Zheng Shou, Xuanjing Huang, Xipeng Qiu, Yu-Gang Jiang]
year: 2026
venue: arXiv
tags: [survey, world-action-model, embodied-ai, vla, world-model, robotics-foundation-model, taxonomy]
zotero_collection: 3-Robotics/1-VLX
image_source: local
arxiv_html: https://arxiv.org/abs/2605.12090
created: 2026-05-14
---

# 论文笔记：World Action Models: The Next Frontier in Embodied AI

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 复旦大学 (Fudan)、上海创智学院 (Shanghai Innovation Institute)、新加坡国立大学 (NUS) |
| 日期 | May 2026 (arXiv:2605.12090) |
| 项目主页 | <https://openmoss.github.io/Awesome-WAM> |
| 代码库 | <https://github.com/OpenMOSS/Awesome-WAM> |
| 类型 | Survey (69 页, 245+ 篇被引文献) |
| 链接 | [arXiv](https://arxiv.org/abs/2605.12090) / [Awesome List](https://github.com/OpenMOSS/Awesome-WAM) |

---

## 一句话总结

> 首份系统梳理 [[World Action Model|WAM]] 领域的综述，提出 **Cascaded vs Joint** 两大架构范式，并把数据、评测和开放挑战纳入统一框架。

---

## 核心贡献

1. **形式化定义**: 用三个条件概率把 [[VLA]]、[[World Model|世界模型]]、[[World Action Model|WAM]] 解耦——$p(a|o,l)$、$p(o'|o,a)$、$p(o',a|o,l)$，并与 Video Policy / VAM / AWM 等近义概念做术语切割。
2. **二分法 + 多级子类**: 把现有 WAM 工作划分为 **Cascaded WAM**（显式管线，世界模型先生成未来再解码动作）与 **Joint WAM**（联合分布，自回归 / 扩散两条主线），并对扩散式 Joint WAM 进一步细分为 Unified-Stream 与 Multi-Stream 三种耦合方式（[[交叉注意力|Cross-Attention]] / Hidden-State / Shared Representation）。
3. **四类训练数据 + 三维评测**: 系统梳理远程操控、UMI 便携人手演示、仿真与 Egocentric 视频四大数据源；评测拆分为视觉保真度、物理常识、动作合理性三个轴。
4. **六大开放挑战**: 架构耦合、多模态物理状态、数据混合、长程规划、推理时延、评测方法学（特别是缺失的"联合 / 因果一致性"指标）。

---

## 问题背景

### 要解决的问题

[[VLA]] 模型把机器人控制建模为 $p(a|o,l)$，本质是"观测→动作"的反应式映射，缺少对物理世界演化的预测；标准 [[World Model|世界模型]] 学 $p(o'|o,a)$，可以模拟环境但不直接产生策略。**如何把"预测未来"和"生成动作"在同一个模型里统一**，正是 WAM 想回答的问题。

### 现有方法的局限

1. **VLA 是 reactive 的**: 没有 forward-dynamics 监督，泛化和零样本能力依赖纯模仿数据。
2. **World Model 离 control 太远**: 像 [[DreamerV3]]、[[V-JEPA]] 这类模型能"想象"，但还要外挂规划器 / 策略头才能动手。
3. **术语极度碎片化**: VAM、Video Policy、AWM、WAM 在文献中混用，且评测各自为政——视觉指标（PSNR/FVD）和任务成功率彼此脱节，无法衡量"想象与执行的一致性"。

### 本文的动机

WAM 应被视为 **VLA 的直接概念继承者**：把"世界"（预测物理）与"动作"（运动控制）放在同一层级、co-equal 的位置，构成下一代具身基础模型范式。综述目标是给整个 design space 画地图。

---

## 方法详解

> 注：这是一篇综述，"方法"等于"分类法 + 代表工作"。下文按论文的 Sec.2–Sec.7 顺序梳理。

### 定义体系（Sec. 2）

三类目标函数分别对应三种 paradigm：

- **[[VLA]] 损失**: $\mathcal{L}_{\mathrm{VLA}} = \mathbb{E}_{(o,l,a)\sim\mathcal{D}}\bigl[-\log p(a\mid o,l)\bigr]$
- **[[World Model|WM]] 损失**: $\mathcal{L}_{\mathrm{WM}} = \mathbb{E}_{(o,a,o')\sim\mathcal{D}}\bigl[-\log p(o'\mid o,a)\bigr]$
- **[[World Action Model|WAM]] 损失**: $\mathcal{L}_{\mathrm{WAM}} = \mathbb{E}_{(o,l,o',a)\sim\mathcal{D}}\bigl[-\log p(o',a\mid o,l)\bigr]$

WAM 必须同时满足两条核心标准：

1. **Forward Predictive Modeling**: 模型必须显式或隐式预测未来状态 $o'$（像素帧、光流、潜空间均可）。
2. **Coupled Action Generation**: 动作 $a$ 必须严格地与预测的未来 $o'$ 对齐（联合输出或 cascaded 条件）。

#### 与近义概念的边界

| 概念 | 与 WAM 的关系 |
|------|-----|
| **VAM** (Video Action Model) | WAM 的子集——把 video 当作世界状态的特例 |
| **Video Policy** | 仅借用 video 模型的 backbone 做 $p(a\mid o)$，**无 world supervision**，因此不是 WAM |
| **AWM** (Action World Model) | 功能等价；改名为 WAM 是为了把 "World" 与 "Action" 摆成 **co-equal** 而非"增强版世界模型" |

### 架构分类（Sec. 4）

#### Cascaded WAM ——管线式

形式化为 $p(o',a\mid o,l) = p(a\mid o',o,l)\,p(o'\mid o,l)$，即"先生成未来 → 再解码动作"。两条子路：

- **Explicit Planning (Sec. 4.1.1)**: 中间表示是**像素空间**（图像、视频、深度、光流、4D 点云、normal map）。
  - **Learned Action Extraction**: 用 [[Diffusion Policy|扩散 IDM]] 或学习式逆动力学回归动作。代表：[[UniPi]]、[[VLP]]、[[RoboEnvision]]、[[ThisAndThat]]、[[TesserAct]]、[[Gen2Act]]、[[Vidar]]、[[π₀]].7、SayDreamAct、Veo-Act、VAG、MVISTA-4D。
  - **Geometric Extraction**: 从生成的视觉中**几何闭式**抽出轨迹（光流→姿态变换、3D flow→末端轨迹）。代表：AVDC、[[Im2Flow2Act]]、[[Dreamitate]]、[[3DFlowAction]]、NovaFlow、LV-P、Dream2Flow、[[4DGen]]、[[RIGVid]]。
- **Implicit Planning (Sec. 4.1.2)**: 中间表示是**潜变量 / 隐藏状态**。代表：VPP、VILP、[[LaPA]]、[[VideoPolicy]]、mimic-video、[[villa-X]]、S-VAM、OmniVTA、MWM、ARDuP。

#### Joint WAM ——联合分布

未来状态和动作在**同一模型**中协同优化。两条主线：

- **Autoregressive Generation (Sec. 4.2.1)**: 把 vision+action 串成 token 序列，因果解码。
  - **Explicit-Decoupled**: 异构模态各走一个 head。代表：[[GR-1]]、[[GR-2]]、GR-MG。
  - **Unified-Discrete**: 都映射到一套离散 token 表，单一 head 解码。代表：[[CoT-VLA]]、[[WorldVLA]]、F1-VLA、RynnVLA-002。
  - **Predictive-Latent**: 不重建像素，预测 latent 因果不变量。代表：[[VLA-JEPA]]。
- **Diffusion-based Generation (Sec. 4.2.2)**: 用 [[扩散变换器|DiT]] 并行去噪，分两大类（见 Figure 6）：
  - **Unified Stream**: 单一 DiT backbone 同时去噪 future-vision token 和 action token。
    - *Explicit Future*: [[PAD]]、VideoVLA、[[UWM]]、[[Cosmos-Policy]]、[[DreamZero]]、GigaWorld-Policy、X-WAM。
    - *Implicit Future*: FLARE、FRAPPE。
  - **Multi-Stream**: 视觉与动作分支独立 DiT，通过三种方式耦合：
    - *Cross-Attention*: DUST、UD-VLA、Motus、CoVAR、LingBot-VA、LDA-1B、AdaWorldPolicy、AIM、DexWorldModel、MotuBrain。
    - *Hidden-State*: Act2Goal、DiT4DiT、Fast-WAM、WAV。
    - *Shared Representation*: UVA、PhysGen。

### 训练数据（Sec. 5）

| 数据源 | 关键属性 | 代表数据集 |
|--------|----------|------------|
| **Robot-Centric Teleoperation** | 严格对齐 state-action，sim-to-real 缝隙小，但采集昂贵 | RT-1、BridgeData V2、RH20T、[[OXE]]、[[DROID 数据集|DROID]]、RoboMIND、AgiBot World |
| **Portable Human (UMI-style)** | 用便携夹持器记录人手轨迹，便宜可扩展 | UMI、FastUMI、FastUMI-100K、RealOmin、DexUMI、UMI-on-Legs |
| **Simulation** | 提供特权信息（深度、姿态、接触力），可程序化生成 | MimicGen、[[RoboCasa]]、[[RoboTwin]]、DexMimicGen、SynGrasp-1B、InternData-M1/A1 |
| **Egocentric / Web Video** | 无动作标签但海量、多样、自然 | SSv2、EPIC-KITCHENS、Ego4D、HOI4D、EgoVid-5M、Aria Everyday、EgoDex |

数据混合的核心张力：**transfer difficulty（迁移成本）** vs **scaling difficulty（扩展成本）**，详见 Figure 7。

### 评测体系（Sec. 6）

**世界建模能力**沿三轴评测：

1. **Visual Fidelity**: PSNR、SSIM、LPIPS、DreamSim、DINO、FVD。
2. **Physical Commonsense**: VideoPhy、PhyGenBench、VBench-2.0、WorldModelBench、Physics-IQ。
3. **Action Plausibility**: WorldSimBench、IDM Turing Test。

**动作策略能力**按场景分类：

- General Manipulation: [[ManiSkill]]、CALVIN、RLBench、Meta-World、[[LIBERO]]。
- Bimanual / Humanoid: RoboTwin、BiGym、HumanoidBench。
- Mobile Manipulation: ManipulaTHOR、HomeRobot。
- Contact / Deformation: SoftGym、DaXBench、TacSL、ManiFeel。
- Real-Robot: RoboArena、RoboChallenge、Maniparena。

---

## 关键公式

### 公式1: [[VLA|VLA 训练目标]]

$$
\mathcal{L}_{\mathrm{VLA}} = \mathbb{E}_{(o,l,a)\sim\mathcal{D}}\bigl[-\log p(a\mid o,l)\bigr]
$$

**含义**: 在观测 $o$ 和语言 $l$ 上对动作 $a$ 做条件似然最大化——纯 reactive 映射，无未来预测。

**符号说明**:
- $o\in O$: 观测（视觉、proprioception 等多模态）
- $l\in L$: 语言指令
- $a\in A$: 动作
- $\mathcal{D}$: 训练数据分布

### 公式2: [[世界模型|World Model 训练目标]]

$$
\mathcal{L}_{\mathrm{WM}} = \mathbb{E}_{(o,a,o')\sim\mathcal{D}}\bigl[-\log p(o'\mid o,a)\bigr]
$$

**含义**: 给定当前状态 $o$ 和拟干预动作 $a$，最大化下一时刻观测 $o'$ 的似然，即学习 **forward dynamics**。

**符号说明**:
- $o'$: 下一时刻观测
- 其他同上

### 公式3: [[World Action Model|WAM 训练目标]]

$$
\mathcal{L}_{\mathrm{WAM}} = \mathbb{E}_{(o,l,o',a)\sim\mathcal{D}}\bigl[-\log p(o',a\mid o,l)\bigr]
$$

**含义**: 在同一目标下**联合预测**未来状态与动作，迫使模型在共享表征中内化"环境演化↔控制信号"的因果关系。这是 WAM 的**定义性方程**。

### 公式4: Cascaded WAM 因子分解

$$
p(o',a\mid o,l) \;=\; \underbrace{p(a\mid o',o,l)}_{\text{action decoder}}\;\underbrace{p(o'\mid o,l)}_{\text{world model}}
$$

**含义**: 显式把 WAM 目标拆成"世界预测 → 动作生成"两阶段。两个模块**独立训练**（separated training），各自承担不同 inductive bias：world model 不必关心机器人 kinematics，action model 不必解决 long-horizon scene prediction。

**对比**: Joint WAM 直接对左边联合分布建模，两个目标在共享网络里 **co-optimized**。

---

## 关键图表

### Figure 1: 时间演化与代表工作

![[WAM-Survey_fig1.png]]

**说明**: WAM 工作的时间线 + 分类树状视图。左分支是 **Joint WAM**（架构上把世界预测和动作生成紧耦合），进一步分裂为自回归（Autoregressive，深色）与扩散（Diffusion-based）两脉；右分支是 **Cascaded WAM**（管线式），按中间表征又分 Explicit / Implicit。可以直观看到 2024–2026 间该领域工作量呈指数增长，且 Diffusion-based Joint WAM 是当前最活跃的方向。

### Figure 2: Survey Roadmap / 综述骨架

![[WAM-Survey_fig2.png]]

**说明**: 综述结构地图。四个核心 axis ——
- **Background (Sec. 3)**: [[VLA]] + [[World Model|WM]] 各自的发展脉络与早期融合工作（如 [[CtrlWorld]]、RoboScape、DREMA、[[DreamerV3]]、[[Dreamer 4]] 等）。
- **Architecture (Sec. 4)**: Cascaded vs Joint 的完整方法树。
- **Training Data (Sec. 5)**: 四类数据源（Robot teleop / UMI / Sim / Human ego）。
- **Evaluation (Sec. 6)**: WM 能力（visual / physical / action）+ Action policy 能力（多类 benchmark）。

这张图是综述的导航中枢，文中后续每节都对照其中一个枝杈。

### Figure 3: WAM 概念定义与对比

![[WAM-Survey_fig3.png]]

**说明**: **左面板**——三个 paradigm 的输入输出对比：
- **[[VLA]]**: 输入 $O_t,L$ → 输出 $A$
- **[[World Action Model|WAM]]**: 输入 $O_t,L$ → 同时输出 $O_{t+1}$ 和 $A$
- **[[World Model|WM]]**: 输入 $O_t,A$ → 输出 $O_{t+1}$

**右面板**——概念边界 Venn 图：WAM = AWM 是最外层超集；VAM（Video Action Model）是 WAM 内、把 video 当未来表征的子集；Video Policy 部分重叠（借用 video backbone 但**不一定**做 world supervision，因此并非全部 WAM）。这张图把全文最容易混淆的术语一图说清。

### Figure 4: World Model 用于 VLA 的四种用法

![[WAM-Survey_fig4.png]]

**说明**: 这是 Sec. 3.3 "World Model for VLA" 的概览，展示 WM 作为辅助模块的四种范式：
- **(a) Imitation Learning**: WM 生成或**过滤**训练轨迹，喂给 VLA 做 IL（代表：DREMA、[[CtrlWorld]]、RoboScape）。
- **(b) Reinforcement Learning**: WM 充当 imagined env，VLA 在想象中 rollout 并做 reward-guided 策略优化（代表：[[DreamerV3]]、[[Dreamer 4]]、DayDreamer、WMPO、VLA-RFT、PhysWorld）。
- **(c) Reward Modeling**: WM 产生 reward 信号供 RL 使用（VIPER、Diffusion Reward、GenReward）。
- **(d) Policy Evaluation**: WM 充当 data-driven simulator 做虚拟评测（WorldEval、WorldGym、dWorldEval）。

$\mathcal{T}$ 表示 rollout trajectories。

### Figure 5: Cascaded WAM 三种子模式

![[WAM-Survey_fig5.png]]

**说明**: Cascaded WAM 的内部结构对比——
- **1(a) Learned Action**: world model 生成显式像素未来（"Video Gen"），后接 [[逆动力学|IDM]] 学习式抽取动作。代表 [[UniPi]]。
- **1(b) Geometric Extraction**: 同样生成像素未来，但用闭式几何（如光流求姿态变换、3D flow 求末端轨迹）抽出动作，无需训练 action head。代表 AVDC、[[Im2Flow2Act]]。
- **2(a) Latent Representation**: 中间载体不是 RGB 帧而是 **latent future**，下游 action decoder 从潜变量解码动作，避免 pixel-level 重建开销。代表 [[LaPA]]、[[villa-X]]。

### Figure 6: Diffusion-based Joint WAM 的四种耦合模式

![[WAM-Survey_fig6.png]]

**说明**: 扩散式 Joint WAM 的核心架构图（对应 Sec. 4.2.2）：
- **1(a) Unified Stream**: 单一 [[扩散变换器|DiT]] backbone 同时容纳 V (vision) 和 A (action) token，世界建模可以是 explicit（重建未来帧）或 implicit（隐式正则）。代表 [[PAD]]、[[UWM]]。
- **2(a) Multi-Stream — Cross-Attention Coupled**: 独立的 Video DiT 与 Action DiT，通过显式 [[交叉注意力|Cross-Attention]] 耦合。代表 DUST、Motus。
- **2(b) Multi-Stream — Hidden-State Coupling**: Video DiT 的中间 hidden state 直接条件化 Action DiT，无需额外 attention 模块。代表 Act2Goal、Fast-WAM。
- **2(c) Multi-Stream — Shared Representation**: V 和 A 先通过 unified encoder 融合，再分别解码——耦合发生在编码端而非解码端。代表 UVA、PhysGen。

这四张子图是理解当前 Joint WAM 设计空间最关键的图。

### Figure 7: 具身数据全景图

![[WAM-Survey_fig7.png]]

**说明**: 训练 WAM 的数据生态图，**X 轴 = Scaling Difficulty（采集成本，越右越难大规模）**，**Y 轴 = Transfer Difficulty（迁移到目标机器人的难度，越上越难）**。四象限分布：
- 右上（贵 + 难迁移但低 gap）：robot-centric teleoperation。
- 右下（贵 + 易迁移）：UMI-style 便携设备。
- 左下（廉价 + 中等迁移）：simulation（需克服 sim-to-real）。
- 左上（极廉价 + 高 gap）：internet ego video。

文中据此论证 WAM 必然走 **mixture training** 路线，并把 human video 的价值拆成三层：(1) low-level 物理先验（重力、对象恒存）、(2) mid-level 因果动力学、(3) high-level 任务逻辑。

### Table 1: Cascaded WAM 方法对比

**说明**（综合表中关键列）: 对所有 Cascaded WAM 工作按"中间表征 / Stage-1 backbone / Stage-2 model / 是否需要动作标签 / 是否 zero-shot / 评测平台"做横向比较。要点：
- 需要 **Act. Label** 的方法占多数；UMI 类方法可以 zero-shot；
- Stage-1 backbone 经历了 U-Net 扩散 → DiT → 视频基础模型（[[Wan2.2|Wan]]/[[CogVideoX]]/Veo）的迁移；
- Stage-2 从轻量 IDM 演化到 [[VLA]] 级 action expert。

### Table 2: Autoregressive Joint WAM 总览

**说明**: 对 GR-1/2/MG、[[CoT-VLA]]、[[WorldVLA]]、F1-VLA、RynnVLA-002、[[VLA-JEPA]] 等按 **参数量 / Backbone / I/O Modality / 评测**。趋势：从单一 video token AR → 加入 chain-of-thought reasoning → 走向 latent-predictive（[[VLA-JEPA]]）。

### Table 3: Diffusion Joint WAM 总览

**说明**: 覆盖 PAD、UWM、[[Cosmos-Policy]]、DreamZero、UVA、FLARE、DUST、Motus、X-WAM 等三十余种方法的参数量、backbone（CogVideoX-5B、Wan、Cosmos、自训）、I/O 模态、评测平台。Multi-Stream 架构在 2025–2026 占据主导，且参数规模普遍突破 5B。

### Table 4: Robot-Centric Teleoperation 数据集

**说明**: 列出 QT-Opt（580k traj.）、MIME、RoboNet、RoboTurk、Bridge、[[OXE]]、[[DROID 数据集|DROID]]、RH20T、AgiBot World 等的年份、规模、技能数、任务数、环境、机器人本体、采集方式、模态（RGB / P / D / A / PC / T）。可作为构建 WAM 训练 mixture 的目录索引。

### Table 5: UMI-style 数据集

**说明**: FastUMI、FastUMI-100K（100K+ traj. 多模态文本注释）、RealOmin、Hoi!、RDT2、ActiveUMI、exUMI、DexUMI、UMI-on-Legs、HoMMI、MV-UMI。UMI 路线在 2024–2025 进入了"硬件 → 大规模数据"的拐点。

### Table 6: Simulation 数据集

**说明**: MimicGen、ManiSkill2、[[RoboCasa]]、[[RoboTwin]]、DexMimicGen、TesserAct、RoboCerebra、SynGrasp-1B、InternData-M1/A1、QUARD-Auto。仿真数据正在引入 4D / 物理一致性增强（如 TesserAct）和**百万级**自动化生成（SynGrasp-1B）。

### Table 7: Human / Egocentric 数据集

**说明**: 列出 SSv2、EPIC-KITCHENS、HowTo100M、Kinetics-700、Ego4D、HOI4D、EgoVid-5M、Ego-Exo4D、ARCTIC、HoloAssist、HOT3D、TACO、Kaiwu、OAKINK2、Nymeria、EgoMimic、PH2D、Humanoid Everyday、IndEgo、HD-EPIC、UniHand、Aria Everyday、EgoDex。这部分体现"无动作标签视频→具身预训练"的范式跃迁。

### Table 8: World Modeling 评测指标

**说明**: 横跨三类——
- *Visual Fidelity*: PSNR、SSIM、LPIPS、DreamSim、DINO、FVD。
- *Physical Commonsense*: VideoPhy、PhyGenBench、VBench-2.0、WorldModelBench、Physics-IQ。
- *Action Plausibility*: WorldSimBench、IDM Turing Test。
每项给出度量公式或参考实现。

### Table 9: Action Policy 评测 Benchmark

**说明**: 按"通用 / 双臂 / 移动 / 接触变形 / Real-Robot"五类，列出每个 benchmark 的观测模态（RGB、Pose、D、S、PC、T、N）和 evaluation focus tag（G 泛化 / MT 多任务 / LH 长程 / Lang 语言 / DS 数据扩展 / S2R sim2real / De 形变）。是选 benchmark 的速查表。

---

## 实验结果

> 综述类论文无原创实验；本节记录综述中**对比分析**的核心结论。

### 架构选择的实验性发现（综述综合）

| 维度 | Cascaded | Joint |
|------|----------|-------|
| 训练效率 | 模块化，可独立预训练 / 微调 | 端到端，单一 loss |
| 推理速度 | 慢（两阶段串行） | 中（单 forward / 多步去噪） |
| 数据需求 | Stage-1 可用无动作视频 | 需要 action-labeled 或巧妙 mask |
| 泛化能力 | 受限于 stage-1 生成质量 | 共享表征通常更鲁棒 |
| 可解释性 | 高（中间像素可视化） | 低（latent / token 难解释） |

### 数据混合的关键观察

- 单纯依赖 robot teleop 训练的 WAM 在跨形态 / 跨场景 zero-shot 上失败率高；
- 加入 ego video co-training 可让模型在 **新物体、新场景**上获得 10%+ 任务成功率提升（多个工作的一致结论）；
- UMI-style 数据填补了 "便宜 + 低迁移成本" 的象限，是当前最高 ROI 的数据来源之一。

### 评测的关键空白

视觉指标和任务成功率**完全脱钩**——存在视频生成 FVD 优秀但物理不合理（物体悬浮、流体反重力）的反例，这呼吁新的 **causal consistency** 指标。

---

## 批判性思考

### 优点

1. **术语首次清算**: 把 VAM、Video Policy、AWM、WAM 这堆名词彻底厘清，配合 Figure 3 的 Venn 图直观可用，对新入门者价值极高。
2. **分类粒度合适**: Cascaded / Joint 二分→Autoregressive / Diffusion 分支→Unified-Stream / Multi-Stream 子类→Cross-Attn / Hidden-State / Shared Rep. 三种耦合，层级清晰，能容纳几乎所有 2024–2026 的新方法。
3. **数据 + 评测一并覆盖**: 不少综述只谈架构；本文把"数据-架构-评测"三轴打通，更适合做项目立项参考。
4. **开放挑战部分有原创洞察**: 不只是罗列问题，而是给出可行的研究方向（如 task-adaptive predictive fidelity、Counterfactual Consistency 指标、prediction-integrated safety），属于 actionable insight。

### 局限性

1. **公式偏少**: 全文只有目标函数级别的形式化（3 个 NLL + 1 个因子分解），没有给每类架构写出 inference-time forward 过程的具体表达式，工程读者复现门槛仍高。
2. **2026 年初截稿，遗漏快速更新**: 像 latent-predictive 方向（[[VLA-JEPA]] 之后的工作）、多模态触觉 WAM 处于雏形阶段，可能未来 6 个月就会有新分类。
3. **无定量横向对比**: 缺一个"用相同 benchmark 评测所有代表方法"的统一实验。这是综述常见短板，作者在 Sec. 7 自己也承认（Architectural Coupling 是 #1 challenge）。
4. **对 Cascaded 几何路线（光流→动作）的评估偏短**: AVDC / Im2Flow2Act / 3DFlowAction 等方向被归并讨论，但它们的几何先验在 contact-rich 场景的优势值得更深入论证。

### 潜在改进方向

1. **加一张"统一基准实验图"**: 哪怕选 3 个代表方法在 CALVIN+RH20T 上跑一遍，能极大提升综述权威性。
2. **引入 inference cost 维度**: 当前未量化各类 WAM 的 FLOPs / Latency / Memory，影响工程选型。
3. **多模态状态预测的分类拓展**: 文中提到 tactile/force 预测是 frontier，但未单独成节，可补一个 "Beyond-RGB WAM" 子类。

### 可复现性评估

- [x] GitHub awesome list 开源（持续更新，论文列表完备）
- [ ] 无统一代码（综述本身不涉及）
- [x] 训练细节通过参考各原始论文可获取
- [x] 数据集均公开或可申请

---

## 关联笔记

### 基于

- [[VLA]]: 综述定位 WAM 为 VLA 的概念继承者。
- [[World Model|世界模型]] / [[世界模型]]: 综述把 WM 视为 WAM 的另一半源流。
- [[DreamerV3]] / [[Dreamer 4]]: 经典 model-based RL 视角的世界模型，被列为 WM 历史脉络的关键节点。
- [[V-JEPA]] / [[JEPA]] / [[I-JEPA]]: latent-predictive 路线的代表，启发了 [[VLA-JEPA]]。

### 对比

- [[A Comprehensive Survey on World Models for Embodied AI]]: 另一个 World Model 角度的综述（arXiv:2510.16732），关注 WM 本身的分类（功能 / 时间 / 空间三轴），与本综述聚焦"WAM 即统一 WM+VLA"互补。
- [[A Survey on Vision-Language-Action Models for Embodied AI]] (arXiv:2405.14093): 早一年的 VLA 综述，本文 Sec. 3 沿用其 VLA 分类作为背景铺垫。

### 方法相关（代表 WAM 工作）

- [[UniPi]] / [[VLP]] / [[Vidar]]: Cascaded-Explicit 代表。
- [[LaPA]] / [[villa-X]]: Cascaded-Implicit 代表。
- [[GR-1]] / [[GR-2]] / [[WorldVLA]] / [[CoT-VLA]]: Autoregressive Joint 代表。
- [[VLA-JEPA]]: Predictive-Latent 路线代表。
- [[PAD]] / [[UWM]] / [[Cosmos-Policy]] / [[DreamZero]]: Unified-Stream Diffusion 代表。
- [[Motus]] / [[FLARE]] / [[UVA]]: Multi-Stream Diffusion 代表。
- [[Dreamer 4]]: 极端长程 imagined-rollout 的代表。

### 硬件/数据相关

- [[DROID 数据集|DROID]] / [[OXE]]: robot teleop 大规模数据。
- [[RoboCasa]] / [[RoboTwin]]: 仿真数据。
- [[Ego4D]] / [[EPIC-KITCHENS]] / [[EgoDex]]: ego-centric video（人手 / 第一人称视频数据）。
- [[UMI]]: 便携人手演示硬件鼻祖。

---

## 速查卡片

> [!summary] WAM Survey (arXiv:2605.12090)
> - **核心**: 首份系统综述 World Action Model 领域，提出 Cascaded vs Joint 二分法 + Diffusion 四类耦合。
> - **方法**: 用 $p(o',a|o,l)$ 形式化定义 WAM，对架构 / 数据 / 评测三轴做分类。
> - **数据**: 整理 4 大数据源 + 9 张数据集表。
> - **评测**: 三轴（视觉 / 物理 / 动作）+ 5 类 benchmark。
> - **开放挑战**: 架构耦合、多模态状态、数据混合、长程规划、推理时延、联合评测。
> - **代码**: <https://github.com/OpenMOSS/Awesome-WAM>

---

*笔记创建时间: 2026-05-14*
