---
title: "HANDOFF: Humanoid Agentic Task-Space Whole-Body Control via Distilled Complementary Teachers"
method_name: "HANDOFF"
authors: [Lizhi Yang, Junheng Li, Nehar Poddar, Yiling Hou, Gio Huh, Robert Griffin, Georgia Gkioxari, Aaron D. Ames]
year: 2026
venue: arXiv
tags: [humanoid-robot, whole-body-control, knowledge-distillation, mixture-of-experts, loco-manipulation]
zotero_collection: 3-Robotics
image_source: online
arxiv_html: https://arxiv.org/html/2606.06493v1
created: 2026-06-09
---

# 论文笔记：HANDOFF: Humanoid Agentic Task-Space Whole-Body Control via Distilled Complementary Teachers

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | California Institute of Technology (Caltech) / The Institute for Human & Machine Cognition (IHMC) |
| 日期 | June 2026 |
| 项目主页 | — |
| 对比基线 | [[HOVER]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.06493v1) / Code: — |

---

## 一句话总结

> HANDOFF 是一个从三个互补教师策略蒸馏而来的单一人形机器人全身控制器，以 10 维紧凑命令接口连接任务规划与底层控制，在 Unitree G1 上同时达到 SOTA 速度跟踪和最大鲁棒操作工作空间。

---

## 核心贡献

1. **紧凑任务空间命令接口设计**: 提出 10 维显式命令向量（骨盆速度 + 高度 + 双腕目标），替代传统需要密集运动学参考的控制器接口，使规划器（VLM/传统规划/VLA）无需任何改造即可直接驱动全身控制
2. **多教师 KL 蒸馏框架**: 将三个互补专家（全身运动跟踪、运动行走、跌倒恢复）通过上下文条件化门控方案蒸馏为单一 [[MoE]] 学生策略，并提出子集感知负载均衡损失和恢复路由机制
3. **安全过滤数据 + 全身运动追踪教师**: 对运动捕捉参考数据进行安全过滤，剔除物理不可行的姿态，训练出具有大范围鲁棒操作工作空间的运动跟踪教师

---

## 问题背景

### 要解决的问题

人形机器人在真实世界部署中，任务规划层（语言指令、[[VLM]] 等）和底层全身控制器之间的**命令空间选择**（command space）极为关键。命令空间定义了上层规划和底层控制器之间的接口语言。

### 现有方法的局限

- [[HOVER]] 和 [[OmniH2O]] 暴露 3 点头手接口，但每个关键点仍需要在控制器频率下提供密集轨迹流，规划器难以实时合成
- [[ExBody2]] 等追踪上身关节角或根速度，需要密集运动学参考
- [[HOMIE]] 将上身和下身解耦，上身由外骨骼驾驶舱控制，下身独立 RL，无法端到端语言驱动
- [[FALCON]] 共同训练下身运动和上身力自适应操作器，仍依赖较重的参考格式
- 现有方案普遍不能直接将自然语言任务指令、VLM 输出或 VLA 模型输出作为控制命令

### 本文的动机

设计一个**紧凑、显式、直觉化、通用且模块化**的任务空间接口，使其同时满足：
- 足够简洁，任何规划器可直接生成
- 足够表达能力，覆盖多样化操作技能
- 天然支持全身协调（蹲取、行走操作、双手操作等）

---

## 方法详解

### 命令空间设计

HANDOFF 的核心接口是一个 **10 维命令向量**：

$$
\mathbf{c} = (v_x,\; v_y,\; \omega_z,\; h_{\text{base}},\; \mathbf{p}^P_L,\; \mathbf{p}^P_R) \in \mathbb{R}^{10}
$$

- **基座速度** $(v_x, v_y, \omega_z)$：水平面线速度 + 偏航角速度，3 维
- **基座高度** $h_{\text{base}}$：骨盆高度目标，1 维
- **双腕目标** $\mathbf{p}^P_L, \mathbf{p}^P_R$：在骨盆坐标系中表示的左右手腕目标位置，各 3 维（共 6 维）

**关键设计特性**：
- 替换了头部跟踪需求（相比 OmniH2O/HOVER 的 3 点头手接口）
- 骨盆坐标系表示使命令对全局位姿不变，规划器只需关心相对位置
- 全身协调**自然涌现**：低基座高度 + 前向腕目标 → 自动触发蹲取；非零基座速度 + 不对称腕目标 → 自动触发行走操作

### 模型架构

HANDOFF 采用 **[[MoE|混合专家模型]] + 多教师[[KL 蒸馏|知识蒸馏]]** 架构：

- **输入**: 10-D 命令 $\mathbf{c}$ + 本体感知观测 $\mathbf{o}$ + 上下文向量
- **Backbone**: 基于[[非对称 Actor-Critic]]（Asymmetric Actor-Critic）的 MoE 学生策略
- **核心模块**: 上下文条件化门控的 3 专家 MoE
- **输出**: 关节力矩目标（29-DoF Unitree G1）
- **训练框架**: rsl-rl + mjlab（基于 MuJoCo-Warp 的 Isaac Lab API）

### 核心模块

#### 模块 1: 三个互补教师策略

| 教师 | 功能 | 训练数据/目标 |
|------|------|--------------|
| 全身运动追踪教师（WBC Teacher） | 追踪骨盆系腕目标 + 身体姿态 | 安全过滤后的运动捕捉数据（AMASS 等），82 条序列，119K 帧 @ 120Hz |
| 运动行走教师（Locomotion Teacher） | 速度追踪 + 稳定行走 | 标准 [[PPO|RL]] 训练 |
| 跌倒恢复教师（Fall-Recovery Teacher） | 从跌倒状态恢复站立 | 随机初始化倒地状态 RL |

**安全过滤**：运动捕捉数据经过筛选，去除物理上不可行的姿态（关节超限、过大地面反力等），确保 WBC 教师学到安全可实现的技能。

#### 模块 2: 上下文条件化 MoE 门控蒸馏

所有三个专家在每个时间步均被评估，其动作均值输出通过门控函数软融合，避免了硬 top-k 路由引入的双峰伪影，保持策略全程可微。

**关键设计**：
- **身体-手臂分片路由（Body-Arm Slice Routing）**：
  - **身体动作**（腿部、腰部）：速度门控的 WBC 教师和运动行走教师凸组合
  - **手臂动作**：锚定到 WBC 教师
- **恢复路由**：恢复掩码 KL 项将第三个专家路由到跌倒恢复教师
- **子集感知负载均衡损失**：防止门控退化到只使用少数专家

### 系统部署：VLM 驱动的智能体规划器

HANDOFF 的 10-D 接口使其**规划器无关**（planner-agnostic）：

- **[[VLM]] 规划器**：接收自然语言任务指令 + 视觉输入，解析为原子子目标序列，每个子目标输出 10-D 命令
- **零次泛化**：无需任何任务专属数据采集或控制器微调
- **接口兼容性**：传统任务规划 / 基于 VLM 的[[具身推理|智能体规划]] / [[VLA]] 模型均可直接接入

---

## 关键公式

### 公式 1: [[KL 蒸馏|多教师 KL 蒸馏]]目标

$$
\mathcal{L}_{\text{KL}} = \sum_{i \in \{\text{wbc, loco, recovery}\}} w_i(\mathbf{c}, \mathbf{o}) \cdot D_{\text{KL}}\!\left(\pi_{\text{student}}(\cdot|\mathbf{o},\mathbf{c}) \,\|\, \pi_i(\cdot|\mathbf{o},\mathbf{c})\right)
$$

**含义**: 学生策略向多个教师策略的加权 KL 散度之和，权重 $w_i$ 由上下文条件化门控函数决定。

**符号说明**:
- $\pi_{\text{student}}$: 学生（MoE）策略的动作分布
- $\pi_i$: 第 $i$ 个教师策略的动作分布
- $w_i(\mathbf{c}, \mathbf{o})$: 与命令 $\mathbf{c}$ 和观测 $\mathbf{o}$ 相关的上下文门控权重，满足 $\sum_i w_i = 1$
- $D_{\text{KL}}$: [[KL 散度]]

### 公式 2: [[MoE|MoE 动作软融合]]

$$
a_{\text{student}} = \sum_{i=1}^{3} g_i(\mathbf{c}, \mathbf{o}) \cdot \mu_i(\mathbf{o}, \mathbf{c})
$$

**含义**: 所有专家动作均值通过软门控权重线性融合，得到最终控制动作，避免 hard top-k 路由引入双峰不稳定性。

**符号说明**:
- $g_i(\mathbf{c}, \mathbf{o})$: 上下文条件化软门控权重（满足 $\sum_i g_i = 1$，由 softmax 归一化）
- $\mu_i(\mathbf{o}, \mathbf{c})$: 第 $i$ 个专家网络输出的动作均值

### 公式 3: 工作空间可行性评估

$$
\text{Feasibility} = \frac{\left|\left\{\text{trials}: \|\mathbf{e}_{\text{wrist}}\| < 15\,\text{cm},\; \text{no-fall},\; \|\Delta\mathbf{p}_{\text{pelvis}}\| < 25\,\text{cm}\right\}\right|}{|\text{total trials}|}
$$

**含义**: 双腕目标可行率——在所有双腕目标试验中，手腕误差小于阈值、无跌倒且骨盆漂移在范围内的比例。

**符号说明**:
- $\|\mathbf{e}_{\text{wrist}}\|$: 指令腕目标与实际手腕位置的欧氏距离误差
- $\Delta\mathbf{p}_{\text{pelvis}}$: 骨盆在水平面的位移漂移量
- 15 cm：手腕误差阈值；25 cm：骨盆漂移阈值

### 公式 4: 速度追踪误差指标

$$
\text{VelTracking} = |\Delta \mathbf{v}| = \mathbb{E}\!\left[\|\mathbf{v}_{\text{cmd}} - \mathbf{v}_{\text{actual}}\|_1\right]
$$

**含义**: 指令基座速度与实际速度的平均绝对误差（MAE），用于定量评估运动行走性能。

**符号说明**:
- $\mathbf{v}_{\text{cmd}} = (v_x^{\text{cmd}}, v_y^{\text{cmd}}, \omega_z^{\text{cmd}})$: 指令基座速度向量
- $\mathbf{v}_{\text{actual}}$: 实际执行的基座速度向量

---

## 关键图表

### Figure 1: 系统概览

![HANDOFF Fig1 Overview](https://arxiv.org/html/2606.06493v1/x1.png)

**说明**: HANDOFF 整体系统图。上层 VLM 驱动的智能体规划器将自然语言任务分解为原子子目标，每个子目标输出 10-D 命令向量驱动底层全身控制器，机器人执行多步长程操作任务，无需任务特定数据或微调。

### Figure 2: 10-D 命令空间与全身协调涌现

![HANDOFF Fig2 Command Space](https://arxiv.org/html/2606.06493v1/x2.png)

**说明**: 展示 10-D 命令如何自然诱导全身协调行为。低骨盆高度 + 前向腕目标 → 蹲取动作；非零基座速度 + 不对称腕目标 → 行走操作；双侧腕目标组合 → 双手抓取。同一命令向量的不同分量组合产生不同的全身运动模式。

### Figure 3: 多教师 MoE 蒸馏架构

![HANDOFF Fig3 Architecture](https://arxiv.org/html/2606.06493v1/x3.png)

**说明**: 三个互补教师（WBC 运动追踪、运动行走、跌倒恢复）在训练时通过 [[KL 散度|KL 蒸馏]] + 上下文条件化门控监督单一学生网络。推理时只运行学生策略，接受 10-D 命令输出 29-DoF 关节控制。

### Figure 4: 双腕工作空间覆盖可视化

![HANDOFF Fig4 Workspace](https://arxiv.org/html/2606.06493v1/x4.png)

**说明**: 双腕可达工作空间凸包从骨盆坐标系三个正交视角（XY 俯视、YZ 正视、XZ 侧视）可视化。HANDOFF 在前向半空间覆盖最大，以与 SOTA baseline 相当的可行率实现了对比方法中最大的操作工作空间。

### Figure 5: 真实硬件任务演示

![HANDOFF Fig5 Hardware Demo](https://arxiv.org/html/2606.06493v1/x5.png)

**说明**: Unitree G1 + Dex1-1 灵巧手在真实场景执行的 5 类任务演示：Pick-and-Place、Pick-Transport-Place、Squat-Pick、Bimanual-Pick-and-Hand-Off、Bilateral Pick-and-Place。所有任务均由同一个 VLM 驱动的智能体规划器端到端控制，控制器无需修改。

### Table 1: 速度追踪性能对比

| 方法 | 命令空间类型 | 速度追踪 $|\Delta v|$ | 操作工作空间大小 | 规划器无关性 |
|------|------------|---------------------|----------------|------------|
| HOVER | 3 点头手接口（密集轨迹） | ≈SOTA | 中等 | 否（需流式轨迹） |
| ExBody2 | 上身关节角 | — | 较小 | 否 |
| OmniH2O | 头+双手位置（密集） | — | 中等 | 否 |
| **HANDOFF (ours)** | **10-D 紧凑命令** | **≈SOTA** | **最大** | **是** |

**说明**: HANDOFF 在速度追踪上达到 SOTA 水准，同时工作空间在所有对比方法中最大，且接口设计使其天然支持规划器无关的语言驱动。

### Table 2: 消融实验

| 配置 | 速度追踪 | 工作空间可行率 | 关键影响 |
|------|---------|-------------|---------|
| w/o 安全过滤数据 | 相近 | 显著下降 | 无安全过滤导致 WBC 教师学到不可行姿态 |
| w/o 跌倒恢复教师 | 相近 | 跌落率上升 | 缺少第三个专家使鲁棒性下降 |
| w/o 软门控（硬 top-k） | 存在抖动 | 下降 | 双峰输出引起不稳定性 |
| w/o 子集负载均衡损失 | 相近 | 轻微下降 | 专家不均衡使用 |
| **HANDOFF Full** | **最佳** | **最佳** | 所有组件均有贡献 |

**关键发现**: 安全过滤数据对工作空间可行率贡献最显著；软门控融合比硬路由更稳定，是整体性能的关键设计。

---

## 实验

### 数据集与平台

| 资源 | 规模 | 特点 | 用途 |
|------|------|------|------|
| AMASS 运动数据（安全过滤后） | 82 条序列，119K 帧 @ 120Hz | 人体运动，经安全过滤保留物理可行姿态 | WBC 教师训练参考数据 |
| Unitree G1 仿真模型 | 29-DoF | mjlab/MuJoCo-Warp，含外挂 Jetson Thor + Dex1-1 | 策略训练与评估 |
| Unitree G1 真实硬件 | 29-DoF + 双侧 Dex1-1 灵巧手 | 真实部署平台，含机载 Jetson Thor 算力 | 真机验证 |

### 实现细节

- **机器人平台**: Unitree G1（29 自由度）+ 双侧 Dex1-1 拟人灵巧抓手
- **训练框架**: [[强化学习|rsl-rl]] + mjlab（MuJoCo-Warp 加速仿真，Isaac Lab API）
- **策略架构**: [[非对称 Actor-Critic]]（Asymmetric Actor-Critic）——Critic 可访问完整特权状态，Actor 只用本体感知 + 命令
- **上层规划器**: VLM 驱动的智能体规划器，接收语言 + 视觉输入，无需额外数据采集或微调

### 真机任务

在 Unitree G1 硬件上演示的 5 类长程操作任务（均由同一 VLM 智能体规划器驱动，**控制器无修改**）：

1. **Pick-and-Place**：单手抓取物体放到目标位置
2. **Pick-Transport-Place**：抓取 + 行走搬运 + 放置
3. **Squat-Pick**：蹲低 + 抓取低处物体（需全身协调）
4. **Bimanual-Pick-and-Hand-Off**：双手抓取后内部交接换手
5. **Bilateral Pick-and-Place**：双手同时分别抓取放置

---

## 批判性思考

### 优点

1. **接口设计优雅**: 10-D 命令在信息密度和表达能力之间取得极佳平衡，使规划器复杂度降至最低，工程落地友好
2. **零适配泛化**: 同一控制器无需任何修改即可被 VLM/传统规划/VLA 模型驱动，真正实现规划器无关
3. **全身协调自然涌现**: 蹲取、行走操作等复杂协调行为作为命令的自然结果涌现，无需显式编程
4. **安全性考量到位**: 安全过滤训练数据是有价值的贡献，实验消融证明其对工作空间可行率的关键作用
5. **多教师蒸馏框架**: 优雅解决不同技能专家难以合并的问题，软门控设计避免 hard routing 的不稳定性

### 局限性

1. **腕目标仅位置、无方向**: 10-D 命令中腕目标只有 3D 位置，缺少方向分量（$\text{SE}(3)$ 旋转部分），限制精确抓取姿态控制
2. **依赖外部感知质量**: VLM 规划器需要高质量视觉输入，感知失效直接影响任务成功率；论文未系统讨论感知鲁棒性
3. **手指控制粒度有限**: 实验使用 Dex1-1 灵巧手，但控制层面停留在手腕目标，未展示精细手指级控制
4. **工作空间评估与任务成功率脱节**: 定量评估主要是工作空间覆盖率，未系统报告各类任务成功率
5. **受控实验环境**: 真机实验在室内结构化场景进行，室外/非结构化环境的泛化能力未验证

### 潜在改进方向

1. 扩展命令空间至 $\text{SE}(3)$ 腕目标（位置 + 方向），提升抓取姿态控制精度
2. 结合视觉伺服闭环（末端执行器视觉反馈），减少对精确 VLM 感知的依赖
3. 与 [[VLA]] 模型（如 [[π₀]]）深度集成，探索端到端视觉语言动作到 10-D 接口的映射
4. 扩展到更多自由度硬件（手指级控制），支持精细操作任务

### 可复现性评估

- [ ] 代码开源（论文发布时未见代码链接）
- [ ] 预训练模型（未提供）
- [x] 训练细节完整（rsl-rl + mjlab 框架有足够细节）
- [x] 数据集可获取（AMASS 公开可用）

---

## 关联笔记

### 基于

- [[HOVER]]: HANDOFF 的主要对比基线，也是多命令空间全身控制器，但需密集轨迹流
- [[OmniH2O]]: 人类到人形机器人的全身远操作，HANDOFF 参考的相关工作
- [[ExBody2]]: 上身表情化全身控制，HANDOFF 的对比基线之一
- [[FALCON]]: 力自适应人形机器人运动-操作，HANDOFF 对比的相关工作

### 对比

- [[HOVER]]: 同样解决全身控制但命令空间更密集（需要流式轨迹），无法被 VLM 直接驱动
- [[HOMIE]]: 上下身解耦方案，通过外骨骼驾驶上身，无法端到端语言驱动
- [[WholeBodyVLA]]: 从端到端 VLA 角度解决全身控制，HANDOFF 则从控制器接口设计角度切入

### 方法相关

- [[MoE]]: HANDOFF 的核心学生架构，软门控融合三专家
- [[KL 蒸馏]]: 多教师 KL 蒸馏是 HANDOFF 的训练范式
- [[KL 散度]]: 蒸馏损失的核心度量
- [[非对称 Actor-Critic]]: 训练框架，Critic 有特权信息
- [[强化学习]]: 底层训练算法（PPO 框架）

### 硬件/数据相关

- [[Unitree G1]]: 实验平台，29-DoF 人形机器人
- [[AMASS]]: 运动捕捉数据集，WBC 教师训练数据来源

---

## 速查卡片

> [!summary] HANDOFF: Humanoid Agentic Task-Space Whole-Body Control
> - **核心**: 10-D 紧凑命令接口（骨盆速度+高度+双腕目标）连接规划与控制
> - **方法**: 三教师（WBC/运动/恢复）→ KL 蒸馏 + 上下文门控 MoE 学生
> - **结果**: Unitree G1 上 SOTA 速度追踪 + 最大鲁棒操作工作空间，VLM 驱动 5 类长程任务零微调
> - **代码**: 未公开

---

*笔记创建时间: 2026-06-09*
