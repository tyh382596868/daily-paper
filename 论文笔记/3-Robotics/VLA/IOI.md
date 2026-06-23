---
title: "IOI: Decoupling Kinematics and Physics for Interactive World Models"
method_name: "IOI"
authors: [Chengyu Bai, Peidong Jia, Tiecheng Guo, Yukai Wang, Rui Ma, Fangyuan Zhao, Chunkai Fan, Xiaobao Wei, Jintao Chen, Hao Wang, Ying Li, Xiaozhu Ju, Jian Tang, Shanghang Zhang]
year: 2026
venue: arXiv
tags: [interactive-world-model, video-diffusion, robot-manipulation, forward-kinematics, flow-matching]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.23296
created: 2026-06-23
---

# 论文笔记：IOI: Decoupling Kinematics and Physics for Interactive World Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Peking University 等 |
| 日期 | June 2026 |
| 项目主页 | 暂无 |
| 对比基线 | [[Ctrl-World]], [[IRASim]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.23296) |

---

## 一句话总结

> IOI 将机器人运动学（确定性）与物理交互（随机性）解耦，用 URDF 驱动的正交投影作为运动学先验注入视频扩散模型，在 RoboTwin 2.0 上以 SOTA FVD 41.23 实现时序一致的机器人世界模型。

---

## 核心贡献

1. **运动学-物理解耦范式**: 将确定性机器人运动从随机物理交互中剥离，URDF 解析计算[[正向运动学|Forward Kinematics]]，生成器专注于环境动力学建模
2. **多视角正交投影条件**: 用三视图（前/侧/顶）正交投影代替透视投影，消除相机外参标定需求，实现视角不变的几何条件
3. **MKAI 模块**: Multi-view Kinematic Aggregation and Injection，跨视图聚合运动学特征并层次化注入[[视频扩散模型|Video Diffusion Model]]，形成结构化几何引导

---

## 问题背景

### 要解决的问题

开发通用具身智能体需要可扩展、视觉真实的交互环境用于策略学习和评估。传统[[物理仿真器]]需要手工资产创建且存在显著现实差距；离线遥操作数据只提供静态轨迹，不支持交互式仿真。[[交互式世界模型|Interactive World Model]]是弥合这一鸿沟的理想方案。

### 现有方法的局限

纯数据驱动的世界模型存在两类核心失败模式：
- **动作偏移（Action Deviation）**: 合成的机器人运动因累积预测误差逐渐偏离控制指令
- **状态不合理（State Implausibility）**: 机器人几何体出现非物理形变、穿插等现象

[[IRASim]] 和 [[Ctrl-World]] 等方法尝试将动作直接条件化，但未能有效约束机器人的确定性运动部分。

### 本文的动机

机器人运动本质上是**确定性的**（由关节力学完全决定），而环境物理交互（物体碰撞、形变）才是**随机性的**。将二者解耦，用解析运动学先验处理确定部分，释放生成模型容量专注于随机部分，是更高效的混合范式。

---

## 方法详解

### 模型架构

IOI 采用**混合范式**架构：
- **输入**: 历史观测帧 $V_{t-h:t}$ + 动作序列 $A_{t+1:t+T}$ + URDF 模型 $M$
- **运动学先验提取**: [[正向运动学|Forward Kinematics]] 解析计算关节轨迹 → 三视图正交投影
- **核心模块**: [[MKAI|Multi-view Kinematic Aggregation and Injection]] 聚合多视角运动学条件
- **Backbone**: 基于[[Flow Matching]]的[[视频扩散模型|Video Diffusion Model]]（冻结预训练权重）
- **输出**: 未来视频帧 $\hat{V}_{t+1:t+T}$

### 核心模块

#### 模块 1: 运动学先验提取（Kinematic Prior Extraction，Section 4.1）

**设计动机**: 利用 URDF 模型的解析[[正向运动学]]实现确定性运动建模

**具体实现**:
- **关节空间动作**: 直接积分 $q_k = q_{k-1} + \Delta q_k$
- **笛卡尔末端执行器动作**: 先通过[[逆运动学|Inverse Kinematics]]转换 $q_k = \mathrm{IK}(a_k, q_{k-1}; M)$
- **3D 铰接几何体重建**: 用 FK 计算每个连杆变换 $\{T_{l,k}\}_{l=1}^{|L|} = \mathrm{FK}(q_k; M)$，拼合几何体 $G_k = \bigcup_{l=1}^{|L|} T_{l,k} P_l$
- **三视图渲染**: 将 $G_k$ 投影到前/侧/顶三个正交视图 $R_k^v = R(G_k; \Pi^v)$

正交投影的关键优势：无需相机外参标定，视角不变性（View-invariance）。

#### 模块 2: MKAI — Multi-view Kinematic Aggregation and Injection（Section 4.2）

**设计动机**: 将三视角运动学渲染聚合为统一表示，并以结构化方式注入扩散骨干网络

包含三个子模块：

**4.2.1 Kinematic Fusion Embedder（运动学融合嵌入器）**:
- 输入：冻结 VAE 编码的三视图潜在特征（加时间嵌入保持时序信息）
- 处理：共享融合编码器（两层 [[MLP]]）聚合三视角 Spatial-Temporal 特征 → 统一表示

**4.2.2 Alignment Embedder（对齐嵌入器）**:
- 将融合后的运动学表示 Tokenize 为扩散模型的特征空间
- 确保运动学条件与扩散模型的特征维度和分布对齐

**4.2.3 Kinematic Blocks 与注入机制**:
- 层次化地将结构化运动学条件通过**加性注入（Additive Injection）**注入扩散骨干
- 消融对比：优于 [[AdaLN]] 和[[交叉注意力|Cross-Attention]]注入方式

#### 模块 3: Kinematics-Guided Video Diffusion（Section 4.3）

**设计动机**: 在运动学条件约束下完成基于[[Flow Matching]]的视频生成

- 冻结预训练视频扩散模型权重，仅训练 MKAI 模块
- 条件输入：历史观测上下文 $z^h$ + 聚合运动学条件 $C^{\mathrm{agg}}$

---

## 关键公式

### 公式 1: [[交互式世界模型|世界模型问题定义]]

$$
p(\hat{V}_{t+1:t+T} \mid V_{t-h:t},\; A_{t+1:t+T},\; M)
$$

**含义**: 给定历史观测、未来动作序列和 URDF 模型，预测未来视频帧的条件分布

**符号说明**:
- $\hat{V}_{t+1:t+T}$: 预测的未来 $T$ 帧视频
- $V_{t-h:t}$: 过去 $h$ 帧历史观测
- $A_{t+1:t+T}$: 未来 $T$ 步动作序列
- $M$: URDF 机器人模型

### 公式 2: [[逆运动学|Inverse Kinematics 关节转换]]

$$
q_k = \mathrm{IK}(a_k,\; q_{k-1};\; M)
$$

**含义**: 对于笛卡尔末端执行器动作，通过逆运动学将动作转换为关节空间轨迹

**符号说明**:
- $q_k$: 第 $k$ 步关节角度向量
- $a_k$: 第 $k$ 步末端执行器动作（笛卡尔空间）
- $q_{k-1}$: 上一步关节角度（用于增量计算）

### 公式 3: [[正向运动学|Forward Kinematics 铰接变换]]

$$
\{T_{l,k}\}_{l=1}^{|L|} = \mathrm{FK}(q_k;\; M)
$$

$$
G_k = \bigcup_{l=1}^{|L|} T_{l,k} P_l
$$

**含义**: FK 从关节角计算每个连杆的 SE(3) 变换，拼合得到完整铰接几何体

**符号说明**:
- $T_{l,k} \in SE(3)$: 连杆 $l$ 在第 $k$ 步的变换矩阵
- $|L|$: 机器人连杆总数
- $P_l$: 连杆 $l$ 的参考点云/几何体
- $G_k$: 第 $k$ 步完整铰接几何体

### 公式 4: [[正交投影|Multi-view Orthographic Rendering]]

$$
R_k^v = R(G_k;\; \Pi^v), \quad v \in \{\text{front},\; \text{side},\; \text{top}\}
$$

**含义**: 将 3D 铰接几何体渲染为三个正交视角的 2D 图像，无需相机外参

**符号说明**:
- $\Pi^v$: 视角 $v$ 的正交投影矩阵（预定义，无需标定）
- $R(\cdot)$: 渲染函数
- $R_k^v$: 第 $k$ 步在视角 $v$ 下的运动学渲染图

### 公式 5: [[Flow Matching|Flow Matching 训练目标]]

$$
z_\tau = (1 - \tau)\,\varepsilon + \tau\, z^f
$$

$$
\mathcal{L}_{\mathrm{FM}} = \mathbb{E}\left[\left\|(z^f - \varepsilon) - v_\theta(z_\tau, \tau;\; z^h, C^{\mathrm{agg}})\right\|_2^2\right]
$$

**含义**: Flow Matching 在噪声与真实潜在变量之间插值，训练速度场网络 $v_\theta$ 以运动学条件为输入

**符号说明**:
- $\tau \sim \mathcal{U}(0,1)$: 时间步（Flow 插值参数）
- $\varepsilon \sim \mathcal{N}(0, I)$: 随机噪声
- $z^f$: 目标未来帧的 VAE 潜在编码
- $z^h$: 历史帧上下文潜在编码
- $C^{\mathrm{agg}}$: MKAI 输出的聚合运动学条件
- $v_\theta$: 参数化速度场网络

---

## 关键图表

### Figure 1: 方法动机对比图

> 🖼️ **Figure 1: IOI 方法整体说明** — ![Figure 1](https://arxiv.org/html/2606.23296v1/x1.png)

**说明**: 对比纯数据驱动方法（IRASim, Ctrl-World）与 IOI 的混合范式，展示动作偏移和状态不合理两类失败模式，以及 IOI 如何通过运动学先验解决这些问题。

### Figure 2: Pipeline Overview

![Figure 2](https://arxiv.org/html/2606.23296v1/x2.png)

**说明**: IOI 的完整处理流水线。从 URDF 模型和动作序列出发，经[[正向运动学]]解析计算关节轨迹，渲染三视角正交投影，通过 [[MKAI]] 模块聚合注入，最终驱动[[视频扩散模型]]生成时序一致的视频帧。

### Figure 3: MKAI 模块细节

![Figure 3](https://arxiv.org/html/2606.23296v1/x3.png)

**说明**: MKAI 模块内部结构。Kinematic Fusion Embedder 用共享 MLP 聚合三视角潜在特征；Alignment Embedder 对齐到扩散特征空间；Kinematic Blocks 以加性注入方式层次化植入扩散骨干网络。

### Figure 4: 仿真定性对比

![Figure 4](https://arxiv.org/html/2606.23296v1/x4.png)

**说明**: RoboTwin 2.0 上与基线方法的视觉对比。红框标注 baseline 方法的形态扭曲和物理不一致区域，黄框突出 IOI 在相同区域的稳定性表现。IOI 在机器人末端执行器形态和物体交互上明显更准确。

### Figure 5: 真实机器人泛化

![Figure 5](https://arxiv.org/html/2606.23296v1/x5.png)

**说明**: 在真实 Franka Emika 机械臂上的定性验证。样本 1 和 3 展示相同任务不同初始状态下的准确动作跟随；样本 2 展示对**未见过的机器人操作动作**的成功零样本泛化。

---

### Table 1: RoboTwin 2.0 运动学保真度与物理合理性定量对比

| 任务 | 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | FVD↓ |
|------|------|-------|-------|--------|------|
| Adjust Bottle | IRASim | 23.77 | 0.7610 | 0.1223 | 400.39 |
| | Ctrl-World | 22.67 | 0.7654 | 0.1163 | 284.53 |
| | **IOI** | **23.00** | **0.8015** | **0.1004** | **344.83** |
| Click Alarmclock | IRASim | 28.87 | 0.8589 | 0.0589 | 348.47 |
| | Ctrl-World | 28.36 | 0.8665 | 0.0619 | 244.43 |
| | **IOI** | **29.00** | **0.9122** | **0.0435** | **173.30** |
| Move Can Pot | IRASim | 29.66 | 0.8678 | 0.0496 | 236.33 |
| | Ctrl-World | 28.19 | 0.8637 | 0.0606 | 214.81 |
| | **IOI** | **30.41** | **0.9288** | **0.0310** | **89.43** |
| Place A2B Left | IRASim | 24.14 | 0.7718 | 0.1087 | 445.12 |
| | Ctrl-World | 22.72 | 0.7808 | 0.1053 | 294.24 |
| | **IOI** | **24.29** | **0.8195** | **0.0886** | **193.43** |
| Place Empty Cup | IRASim | 28.37 | 0.8287 | 0.0631 | 464.15 |
| | Ctrl-World | 28.06 | 0.8337 | 0.0624 | 327.57 |
| | **IOI** | **29.07** | **0.9038** | **0.0385** | **175.29** |
| Place Mouse Cap | IRASim | 26.63 | 0.8153 | 0.0781 | 419.34 |
| | Ctrl-World | 25.64 | 0.8184 | 0.0760 | 325.42 |
| | **IOI** | **26.83** | **0.8672** | **0.0571** | **207.33** |
| Place Object Stand | IRASim | 28.06 | 0.8581 | 0.0635 | 409.58 |
| | Ctrl-World | 26.97 | 0.8527 | 0.0655 | 271.76 |
| | **IOI** | **28.19** | **0.9033** | **0.0481** | **230.57** |
| Stack Bowls Three | IRASim | 29.00 | 0.8478 | 0.0705 | 324.85 |
| | Ctrl-World | 29.05 | 0.8606 | 0.0670 | 217.87 |
| | **IOI** | **29.59** | **0.8972** | **0.0521** | **178.08** |
| Stamp Seal | IRASim | 26.60 | 0.7987 | 0.0703 | 373.60 |
| | Ctrl-World | 24.34 | 0.8012 | 0.0847 | 319.61 |
| | **IOI** | **26.64** | **0.8829** | **0.0467** | **173.35** |
| Turn Switch | IRASim | 28.79 | 0.8442 | 0.0670 | 313.95 |
| | Ctrl-World | 26.57 | 0.8261 | 0.1088 | 233.87 |
| | **IOI** | **27.79** | **0.8914** | **0.0674** | **168.36** |
| **Overall** | IRASim | 26.81 | 0.8198 | 0.0803 | 126.20 |
| | Ctrl-World | 25.50 | 0.8192 | 0.0867 | 64.90 |
| | **IOI** | **25.73** | **0.8637** | **0.0695** | **41.23** |

**关键发现**: IOI 整体 FVD 41.23，相比 Ctrl-World 降低 36.47%，相比 IRASim 降低 67.30%；SSIM 提升尤为显著，说明运动学约束有效改善时序一致性。

### Table 3: 零样本 OOD 泛化定量评估

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | FVD↓ |
|------|-------|-------|--------|------|
| IRASim | 24.02 | 0.7740 | 0.1521 | 166.14 |
| Ctrl-World | 21.64 | 0.7672 | 0.1771 | 78.79 |
| **IOI** | **24.08** | **0.8127** | **0.1309** | **69.16** |

**关键发现**: 零样本 OOD 场景下，IOI FVD 69.16 超越 Ctrl-World 12.23%，超越 IRASim 58.40%，体现运动学先验的强泛化能力。

### Table 4: 策略评估对齐性（π₀ 策略成功率）

| 指标 | 数值 |
|------|------|
| 平均成功率差异 | +1.6 百分点 |
| 皮尔逊相关系数 | **0.9992** |

**关键发现**: IOI 作为策略评估器与真实物理仿真器高度对齐（r = 0.9992），可替代真实仿真器用于策略开发，验证了世界模型作为"数字评估环境"的可行性。

### Table 5: 动作条件机制消融研究

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | FVD↓ |
|------|-------|-------|--------|------|
| AdaLN | 24.89 | 0.8473 | 0.0812 | 56.47 |
| 交叉注意力 | 24.76 | 0.8412 | 0.0781 | 62.56 |
| **URDF+MKAI（IOI）** | **25.73** | **0.8637** | **0.0695** | **41.23** |

**关键发现**: 明确几何条件（URDF+MKAI 加性注入）相比隐式注入方式（AdaLN/Cross-Attention）实现 26.97% FVD 提升，验证了显式运动学先验的必要性。

### Table 6: MKAI 模块上界分析

| 方法 | 需要外参 | PSNR↑ | SSIM↑ | LPIPS↓ | FVD↓ |
|------|---------|-------|-------|--------|------|
| IOI-Explicit（透视投影） | 是 | 26.97 | 0.8769 | 0.0631 | 37.15 |
| **IOI-MKAI（正交投影）** | **否** | **25.73** | **0.8637** | **0.0628** | **41.23** |

**关键发现**: 无需相机外参的正交投影方案（IOI-MKAI）与需要外参的透视投影上界（IOI-Explicit）性能差距极小（FVD 仅差 4.08），极大降低了部署门槛。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| RoboTwin 2.0 | 10 个操作任务 | 仿真环境，含 URDF 标注 | 训练 + 主要测试 |
| RoboTwin 2.0 OOD | 10 个未见任务组合 | 组合泛化测试集 | 零样本泛化评估 |
| 真实 Franka Emika 数据 | 真实机器人轨迹 | 真实机械臂操作演示 | 真实世界验证 |

### 实现细节

- **Backbone**: 预训练视频扩散模型（基于 Flow Matching，冻结权重）
- **优化器**: AdamW，学习率 $5 \times 10^{-5}$
- **Batch Size**: 全局 batch size 32
- **训练步数**: 100,000 步
- **学习率调度**: 2,000 步线性 warm-up + cosine annealing 衰减
- **可训练模块**: 仅 MKAI 模块（Backbone 完全冻结）

### 可视化结果

Figure 4 展示了 RoboTwin 2.0 上的定性对比：IOI 生成的帧中机器人末端执行器形态准确，无形态扭曲，物体交互物理合理；而 IRASim 和 Ctrl-World 在长轨迹末端出现明显机器人几何变形和物理不一致现象。

---

## 批判性思考

### 优点

1. **解耦设计优雅**: 用解析方法处理确定性部分，避免让神经网络"重新学习物理定律"，思路清晰
2. **工程实用性强**: 正交投影消除相机外参依赖，大幅降低真实部署门槛（Table 6 验证性能损失微小）
3. **策略评估价值**: 皮尔逊相关 0.9992 意味着可以真正替代物理仿真器用于策略开发，不只是视觉好看

### 局限性

1. **依赖精准 URDF**: 方法强依赖 URDF 模型，对于形变体、软体、非刚性物体无法建模；URDF 获取在真实场景中也有额外成本
2. **视频质量与 FVD 解耦**: Table 1 中部分任务 PSNR 并非 IOI 最优（如 Adjust Bottle），说明运动学约束有时会牺牲像素级准确度换取时序一致性
3. **动作空间限制**: 仅支持关节空间和笛卡尔末端执行器动作，对更复杂的灵巧手（multi-finger）、柔性体任务扩展性待验证

### 潜在改进方向

1. **软运动学先验**: 对 URDF 模型误差建模，支持更宽松的几何约束（如用 3DGS 表示机器人形态）
2. **多机器人/移动底座**: 扩展到带移动底座的操作机器人（loco-manipulation），FK 链更长
3. **接触-感知物理**: 结合接触力/触觉信息，改进物体交互建模

### 可复现性评估

- [ ] 代码开源（暂未公开）
- [ ] 预训练模型（暂未公开）
- [x] 训练细节完整（AdamW, lr=5e-5, 100k steps, bs=32）
- [x] 数据集可获取（RoboTwin 2.0 公开）

---

## 关联笔记

### 基于

- [[视频扩散模型|Video Diffusion Model]]: 冻结预训练视频扩散 Backbone
- [[Flow Matching]]: 扩散训练目标
- [[正向运动学|Forward Kinematics]]: 运动学先验计算核心

### 对比

- [[IRASim]]: 早期机器人视频生成工作，FVD 126.20 vs IOI 41.23
- [[Ctrl-World]]: 更近期的竞争基线，FVD 64.90 vs IOI 41.23

### 方法相关

- [[MKAI|Multi-view Kinematic Aggregation and Injection]]: IOI 的核心注入模块
- [[正交投影]]: 三视图视角不变渲染方案
- [[逆运动学|Inverse Kinematics]]: 笛卡尔动作转关节空间
- [[交互式世界模型|Interactive World Model]]: 所属方法类别

### 硬件/数据相关

- [[RoboTwin]]: 主要评测基准（仿真环境）
- [[Franka Emika]]: 真实验证平台

---

## 速查卡片

> [!summary] IOI: Decoupling Kinematics and Physics
> - **核心**: 用 URDF 解析运动学先验（正交三视图）解耦确定性机器人运动与随机物理交互
> - **方法**: FK 提取几何条件 → 三视角渲染 → MKAI 加性注入冻结视频扩散 Backbone
> - **结果**: RoboTwin 2.0 FVD 41.23（SOTA），策略评估皮尔逊相关 0.9992，零样本 OOD FVD 69.16
> - **代码**: 暂未公开

---

*笔记创建时间: 2026-06-23*
