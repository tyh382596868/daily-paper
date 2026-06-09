---
title: "HANDOFF: Humanoid Agentic Task-Space Whole-Body Control via Distilled Complementary Teachers"
method_name: "HANDOFF"
authors: [Lizhi Yang, Junheng Li, Nehar Poddar, Yiling Hou, Gio Huh, Robert Griffin, Georgia Gkioxari, Aaron D. Ames]
year: 2026
venue: arXiv
tags: [humanoid-whole-body-control, policy-distillation, mixture-of-experts, loco-manipulation, task-space-control, agentic-planning, reinforcement-learning]
zotero_collection: 3-Robotics
image_source: online
arxiv_html: https://arxiv.org/html/2606.06493v1
created: 2026-06-09
---

# 论文笔记：HANDOFF: Humanoid Agentic Task-Space Whole-Body Control via Distilled Complementary Teachers

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | California Institute of Technology (Caltech)、The Institute for Human & Machine Cognition (IHMC) |
| 日期 | June 2026 |
| 项目主页 | 暂未公开 |
| 对比基线 | [[SONIC]]、[[FALCON]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.06493) / Code（即将开源） |

---

## 一句话总结

> HANDOFF 提出一个 10 维 task-space 命令接口，通过对三个互补专家 Teacher 进行 context 条件化 KL 蒸馏，训练单一 MoE 人形机器人 [[全身控制]] 策略，在 Unitree G1 上实现了最大操作工作空间之一。

---

## 核心贡献

1. **紧凑 10 维任务空间命令接口**：设计一个对规划器友好的接口（2D 底盘速度 + 双腕 3D 位置目标 + 2D 方向），比密集运动学参考更易被各类规划器调用，同时足够灵活支持各种操作技能。
2. **互补 Teacher 的 Context 条件化 KL 蒸馏**：将三个专门化 Teacher（全身运动跟踪、locomotion、跌倒恢复）蒸馏进单一 [[知识蒸馏|MoE 学生策略]]，body slice 按速度上下文加权融合，arm slice 锚定 WBC Teacher，跌倒恢复由离散 context 信号路由。
3. **VLM 驱动的无微调 Agentic 规划**：通过机载 VLM（Nvidia Jetson Thor 运行本地推理，可选 ChatGPT-API fallback）将自然语言指令分解为 HANDOFF 命令序列，无需任务专属数据或控制器微调。

---

## 问题背景

### 要解决的问题

人形机器人落地部署的核心瓶颈之一是任务规划与 [[全身控制]] 之间的**命令空间（Command Space）选择**：规划层需要向控制层发出什么格式的指令？

### 现有方法的局限

- 现有 [[Whole-Body Control|WBC]] 控制器（如 SONIC、FALCON）通常要求密集的运动学或空间参考（如逐帧关节角轨迹、全局坐标系末端执行器位置），规划器很难从任务语义中自动合成这些参考。
- 多 Teacher 融合策略多为简单拼接或硬切换，没有优雅地将 locomotion、操作、跌倒恢复整合进统一框架。
- 实际部署往往需要针对每个新任务进行 controller fine-tuning 或大量示范数据。

### 本文的动机

设计一个**紧凑、显式、直觉友好、可模块扩展**的命令接口（10 维），并通过 context 条件化 [[知识蒸馏]] 将多个专家 Teacher 融合进单一 [[混合专家模型|MoE]] 学生策略，使 VLM 规划器无需额外训练就能驾驭整个系统。

---

## 方法详解

### 命令空间设计（10-D Task-Space Interface）

HANDOFF 的命令向量 $\mathbf{c} \in \mathbb{R}^{10}$ 由以下分量构成：

- $v_x, v_y \in \mathbb{R}^2$：底盘平面速度（[[locomotion]] 速度目标）
- $\mathbf{p}_L, \mathbf{p}_R \in \mathbb{R}^{3} \times 2$：双腕在 **Pelvis 坐标系**中的 3D 位置目标（共 6 维）
- $\psi_L, \psi_R \in \mathbb{R}^{2}$：双腕朝向指令（共 2 维）

双腕目标作为 `RelativeFrameTask`（相对于 pelvis 坐标系），使规划器无需关注全局位姿。

### 模型架构

HANDOFF 采用 **[[混合专家模型|Mixture-of-Experts (MoE)]] 学生策略**，由三路 [[知识蒸馏|KL 蒸馏]] 项训练：

**三个 Teacher 专家**：

| Teacher | DoF | 训练数据 | 职责 |
|---------|-----|----------|------|
| WBC 运动跟踪 Teacher | 29-DoF（全身） | CoP 安全过滤后的重定向动捕片段 | 复杂全身操作运动跟踪 |
| Locomotion Teacher | 15-DoF（下半身） | 平坦地形 + 课程化手臂扰动 | 速度跟踪、稳定步态 |
| 跌倒恢复 Teacher | 29-DoF（全身） | Locomotion + 配对跌倒恢复片段 | 跌倒后自主恢复站立 |

**MoE 学生蒸馏机制**：

- **Body Slice（下半身）**：按连续速度 context $\alpha(v)$ 对 WBC Teacher 和 Locomotion Teacher 进行**凸组合 KL 蒸馏**（velocity-gated convex blend）
- **Arm Slice（手臂）**：锚定到 WBC Teacher 的 KL 项
- **跌倒恢复路由**：离散 recovery context 的 **recovery-masked KL 项**将第三个 MoE Expert 路由到跌倒恢复 Teacher

**Context 条件化**：
- Context 输入包含：底盘速度幅值（连续）、是否处于跌倒恢复状态（离散二值）
- 扩展性：新 Teacher/技能只需新增一个 Teacher head 和一个 context 通道，不改动已有 Teacher 和命令接口

### 安全数据过滤（CoP 滤波）

WBC Teacher 的训练数据来自动捕重定向，原始片段中含有动力学不可行帧。论文使用**静态质心压力（Center of Pressure, CoP）裕量**的 [[CBF（控制障碍函数）|Control Barrier Function (CBF)]] 投影进行闭形式安全过滤，剔除 CoP 超出支撑多边形的帧，再训练 WBC Teacher。

### Agentic 规划层

硬件上运行完整的机载推理栈：
- **Unitree G1**（29-DoF）+ 双侧 Dex1-1 拟人夹爪 + 头部 ZED-M 立体 RGB-D 相机
- **Nvidia Jetson Thor** 机载运行：RL 控制器 + Agentic 规划器 + 本地 VLM 推理（可选 ChatGPT-API fallback）
- VLM 感知场景 → 语言指令分解 → 输出 10-D 命令序列 → HANDOFF WBC 执行

规划器无需任务专属数据或控制器微调，体现了命令接口的**规划器无关性**。

---

## 关键公式

### 公式 1：[[知识蒸馏|Body Slice Context-Conditioned KL 蒸馏]]

$$
\mathcal{L}_{\text{body}} = \alpha(v) \cdot \text{KL}\!\left(\pi_{\theta}^{\text{body}}(\cdot|s) \,\|\, \pi_{\text{WBC}}(\cdot|s)\right) + (1 - \alpha(v)) \cdot \text{KL}\!\left(\pi_{\theta}^{\text{body}}(\cdot|s) \,\|\, \pi_{\text{loco}}(\cdot|s)\right)
$$

**含义**：下半身 MoE Expert 按速度上下文 $\alpha(v)$ 在 WBC Teacher 和 Locomotion Teacher 之间做凸组合蒸馏，速度越小越偏向 WBC Teacher（精细操作），速度越大越偏向 Locomotion Teacher（稳定步态）。

**符号说明**：
- $\alpha(v) \in [0,1]$：由底盘速度幅值 $v$ 决定的连续 gating 权重
- $\pi_{\theta}^{\text{body}}$：学生策略的下半身 Expert
- $\pi_{\text{WBC}}$：全身运动跟踪 Teacher 策略
- $\pi_{\text{loco}}$：Locomotion Teacher 策略

### 公式 2：[[知识蒸馏|Arm Slice KL 蒸馏]]

$$
\mathcal{L}_{\text{arm}} = \text{KL}\!\left(\pi_{\theta}^{\text{arm}}(\cdot|s) \,\|\, \pi_{\text{WBC}}(\cdot|s)\right)
$$

**含义**：手臂 MoE Expert 始终锚定到 WBC Teacher，确保操作精度与运动质量。

### 公式 3：[[知识蒸馏|Recovery-Masked KL 蒸馏]]

$$
\mathcal{L}_{\text{rec}} = \mathbb{1}[\text{recovery}] \cdot \text{KL}\!\left(\pi_{\theta}^{\text{rec}}(\cdot|s) \,\|\, \pi_{\text{fall-rec}}(\cdot|s)\right)
$$

**含义**：跌倒恢复 KL 项仅在机器人处于跌倒恢复状态（离散 context=1）时激活，将第三个 MoE Expert 路由到跌倒恢复 Teacher。

**符号说明**：
- $\mathbb{1}[\text{recovery}]$：离散跌倒恢复 context 指示函数
- $\pi_{\theta}^{\text{rec}}$：学生策略的跌倒恢复 Expert
- $\pi_{\text{fall-rec}}$：跌倒恢复 Teacher 策略

### 公式 4：[[CBF（控制障碍函数）|CoP 安全过滤 CBF 投影]]

$$
\mathbf{q}^* = \arg\min_{\mathbf{q}} \|\mathbf{q} - \mathbf{q}_{\text{raw}}\|^2 \quad \text{s.t.} \quad h_{\text{CoP}}(\mathbf{q}) \geq 0
$$

**含义**：对原始重定向动捕帧 $\mathbf{q}_{\text{raw}}$ 做最近可行点投影，约束为静态 CoP 裕量 $h_{\text{CoP}} \geq 0$，保证训练数据动力学可行。

---

## 关键图表

### Figure 1: 系统概览

![Figure 1 - HANDOFF Overview](https://arxiv.org/html/2606.06493v1/x1.png)

**说明**：HANDOFF 整体框架图。展示从自然语言指令出发，经 VLM Agentic 规划器分解为 10-D task-space 命令序列，驱动 MoE 学生策略（三路 Teacher 蒸馏）控制 Unitree G1 完成操作任务的完整流程。

### Figure 2: CoP 安全滤波

![Figure 2 - CoP Filtering](https://arxiv.org/html/2606.06493v1/x2.png)

**说明**：原始重定向动捕数据集中含有动力学不可行帧（CoP 超出支撑多边形），通过静态 CoP 裕量的 [[CBF（控制障碍函数）|CBF]] 闭形式投影进行安全过滤，确保 WBC Teacher 训练数据质量。

### Figure 3: 蒸馏架构

![Figure 3 - Distillation Architecture](https://arxiv.org/html/2606.06493v1/x3.png)

**说明**：三路 context 条件化 KL 蒸馏方案。Body Slice 按速度 gating 在 WBC/Locomotion Teacher 之间做凸组合；Arm Slice 锚定 WBC Teacher；Recovery Expert 由离散 context 路由到跌倒恢复 Teacher。

### Figure 4: 双腕操作工作空间对比

![Figure 4 - Workspace Comparison](https://arxiv.org/html/2606.06493v1/x4.png)

**说明**：在 pelvis 坐标系下从三个正交视角（XY 俯视、YZ 正面、XZ 侧面）展示双腕可达工作空间的凸包，仅显示前半空间。HANDOFF 覆盖了对比方法中最大的前半工作空间，可行性与 SONIC 持平，FALCON 的工作空间在侧向和顶端明显受限。

### Table 1: 速度跟踪与操作工作空间定量对比

| 方法 | 速度跟踪 (↑) | 可行工作空间体积 (↑) | 跌倒恢复 |
|------|------------|------------------|--------|
| SONIC | SOTA | 较小 | 无 |
| FALCON | 较低 | 更小（侧向/顶部受限） | 无 |
| **HANDOFF（无恢复）** | **匹配 SOTA** | **最大之一** | 无 |
| **HANDOFF（完整）** | **匹配 SOTA** | **最大之一** | **有** |

> 工作空间可行性判定条件：2000 个发现目标 + 400 个精度目标，双腕在测量窗口内始终保持在目标 15 cm 以内，策略不跌倒，pelvis 水平漂移 < 25 cm。

### Table 2: Agentic 规划硬件任务

| 任务 | 描述 |
|------|------|
| 步行取物 | 根据语言指令步行到指定位置取物 |
| 双手传递 | 从一侧手传到另一侧手 |
| 桌面整理 | 多步操作序列，VLM 自主规划步骤 |
| 跌倒恢复 | 受扰动跌倒后自主恢复站立继续任务 |

所有任务均由 VLM 驱动的 Agentic 规划器控制，无需任务专属数据或控制器微调。

---

## 实验

### 环境设置

- **仿真**：Isaac Lab（基于 [[IsaacLab]] 物理引擎），评估速度跟踪和工作空间指标
- **实机**：Unitree G1（29-DoF）
  - 末端执行器：股票 3 指手替换为双侧 **Dex1-1 拟人夹爪**
  - 感知：头部 **ZED-M 立体 RGB-D 相机**（供 VLM 和路径点投影）
  - 计算：背部 **Nvidia Jetson Thor**（机载运行全栈）

### 实现细节

- **Teacher 训练**：三个 Teacher 分别用 [[PPO]] 训练：
  - WBC Teacher：29-DoF，CoP 过滤后的重定向动捕片段
  - Locomotion Teacher：15-DoF（下半身），平坦地形 + 课程化手臂扰动
  - 跌倒恢复 Teacher：29-DoF，Locomotion + 配对跌倒恢复片段
- **学生蒸馏**：MoE 学生通过三路 context 条件化 KL 项从 Teacher 蒸馏

### 工作空间评估方法

- 2000 个**发现性目标**：随机采样探索可达边界
- 400 个**精度目标**：更细粒度的精度验证
- 可行判定：双腕 ≤15 cm 目标误差 + 不跌倒 + pelvis 水平漂移 ≤25 cm

---

## 批判性思考

### 优点

1. **接口设计优雅**：10-D 命令向量对规划器高度友好，既足够紧凑（不要求全身关节轨迹），又足够表达力（支持多样操作技能）
2. **蒸馏框架可扩展**：新技能只需添加 Teacher head + context 通道，无需修改已有组件，系统性好
3. **端到端机载部署**：Jetson Thor 运行全栈（RL 控制器 + VLM 规划），真正 zero-shot 部署无需任务微调
4. **跌倒恢复整合**：将跌倒恢复作为 Teacher 之一蒸馏进统一框架，而非单独模块，是现有同类工作的重要改进

### 局限性

1. **命令接口相对简单**：10-D 接口目前不包含接触力/阻抗控制信号，对接触丰富（contact-rich）操作任务可能不够
2. **VLM 规划器的可靠性**：依赖 VLM 感知和规划，在遮挡复杂或语义歧义场景下可能失败
3. **平坦地形假设**：Locomotion Teacher 仅在平坦地形训练，非结构化地形泛化能力未验证
4. **缺乏精细灵巧操作**：Dex1-1 夹爪相对简单，复杂灵巧手操作尚未展示

### 潜在改进方向

1. 扩展命令接口，纳入阻抗/接触力维度，支持接触丰富操作
2. 引入视觉状态估计以减少对 ZED-M 点云的依赖
3. 将课程化训练推广到非结构化地形

### 可复现性评估

- [ ] 代码开源（论文声明即将开源）
- [ ] 预训练模型（未发布）
- [ ] 训练细节完整（论文描述较详细）
- [ ] 数据集可获取（动捕数据集未公开）

---

## 关联笔记

### 基于

- [[PPO]]: Teacher 策略训练使用的 RL 算法
- [[知识蒸馏]]: 核心蒸馏框架
- [[混合专家模型]]: MoE 学生策略架构
- [[CBF（控制障碍函数）]]: CoP 安全数据滤波

### 对比

- [[SONIC]]: SOTA 速度跟踪基线，HANDOFF 在工作空间上超越
- [[FALCON]]: 力自适应 loco-manipulation，工作空间相对受限

### 方法相关

- [[全身控制]]: 核心任务
- [[Whole-Body Control]]: 人形机器人全身协调控制
- [[locomotion]]: 双足运动 Teacher 专家
- [[IsaacLab]]: 仿真训练平台

### 硬件相关

- [[Unitree G1]]: 实验机器人平台（29-DoF）

---

## 速查卡片

> [!summary] HANDOFF (arXiv 2606.06493)
> - **核心**: 10-D task-space 命令接口 + 三 Teacher KL 蒸馏 → 单一 MoE 人形全身控制策略
> - **方法**: Context 条件化 KL 蒸馏（velocity-gated body/arm/recovery 三路）
> - **结果**: Unitree G1 上最大双腕工作空间之一，速度跟踪匹配 SOTA，VLM 无微调 agentic 部署
> - **代码**: 即将开源

---

*笔记创建时间: 2026-06-09*
