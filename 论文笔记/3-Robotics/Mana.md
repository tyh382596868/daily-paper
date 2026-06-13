---
title: "Mana: Dexterous Manipulation of Articulated Tools"
method_name: "Mana"
authors: [Zhao-Heng Yin, Guanya Shi, Pieter Abbeel, C. Karen Liu]
year: 2026
venue: arXiv
tags: [dexterous-manipulation, articulated-tools, sim-to-real, diffusion-policy, point-cloud, dexterous-hand, imitation-learning, reinforcement-learning]
zotero_collection: 3-Robotics
image_source: online
arxiv_html: https://arxiv.org/html/2606.13677
created: 2026-06-13
---

# 论文笔记：Mana: Dexterous Manipulation of Articulated Tools

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | UC Berkeley、CMU、Stanford University、Amazon FAR |
| 日期 | June 2026 |
| 项目主页 | [zhaohengyin.github.io/mana](https://zhaohengyin.github.io/mana) |
| 对比基线 | [[Dex-Retargeting]]（遥操作） |
| 链接 | [arXiv](https://arxiv.org/abs/2606.13677) |

---

## 一句话总结

> Mana 是一个受计算机动画启发的、可扩展的 sim-to-real 数据生成框架，让灵巧机器人手能够零样本迁移地操作关节型工具（夹钳、钳子、晾衣夹、注射器）。

---

## 核心贡献

1. **动画驱动的数据生成系统**: 从程序化生成的抓取关键帧出发，通过运动规划 + [[PPO|强化学习]] 的粗到精流水线，自动生成高质量操作轨迹，每种工具配置耗时不到 1 分钟。
2. **零样本 Sim-to-Real 迁移**: 在四类关节型工具（夹钳、钳子、晾衣夹、注射器）上实现约 70% 的成功率，显著超越遥操作基线（~30%）。
3. **定制化硬件与感知**: 设计了扁平合规型硅胶指尖，配合[[点云]]条件[[Diffusion Policy|扩散策略]]实现鲁棒的视觉运动控制。

---

## 问题背景

### 要解决的问题

关节型工具操作（如使用夹钳夹取物体、用注射器推注液体）要求机器人同时稳定工具并施加精确的功能性驱动力。这类工具普遍薄小（厚度 0.8–1.5 cm）、关节刚度高（需施加 3–7 N 力），且接触配置的微小误差会导致任务失败。

### 现有方法的局限

- **遥操作系统**：基于位置控制，无法生成足够大且方向正确的捏合力，尤其在狭小工具表面上操作极难；人类操作员即使经过 1 小时练习，成功率仍不足 30%。
- **现有抓取规划器**（如 Lightning Grasp）：在近地面狭小抓取面场景中会发生碰撞穿透，需要额外的碰撞感知逆运动学才能可靠生成抓取姿态。
- **通用策略学习方法**：缺乏针对关节型工具"同时抓取+驱动"这一耦合动力学问题的专门建模。

### 本文的动机

受计算机动画中"关键帧→轨迹插值"思路的启发：若能先用程序化方法生成正确的功能性抓取关键帧，再通过运动规划 + RL 将其扩展为完整的操作轨迹，就能在[[IsaacLab|仿真]]中大规模生成多样化数据，进而通过[[Sim-to-Real|零样本迁移]]部署到真实机器人。

---

## 方法详解

### 整体框架

Mana 采用**粗到精（Coarse-to-Fine）**的三阶段流水线：

- **输入**: 用户在 < 1 分钟内标注工具的规范姿态和功能接触区域
- **阶段 1 — 关键帧生成**: Lightning Grasp+（碰撞感知 IK）生成功能性抓取关键帧集合
- **阶段 2 — 轨迹生成**: [[RRT-Connect]] 运动规划生成预抓取轨迹，[[PPO]] 训练短时程 RL 策略完成接触丰富的抓取和工具驱动阶段
- **输出**: 机器人用[[点云]]条件的[[Diffusion Policy|扩散策略]]执行零样本 sim-to-real 迁移

### 核心模块

#### 模块 1: Lightning Grasp+（增强版关键帧生成器）

**设计动机**: 原始 Lightning Grasp 在工具表面狭小、贴近地面场景下会产生碰撞穿透，无法生成有效抓取姿态。

**具体实现**:
- 使用接触场查询（contact field queries）在工具功能区域采样密集的接触点集合
- 新增**碰撞感知 IK 优化**：检测穿透中心 $p$ 和分离向量 $d$，在保持接触约束的同时迭代消除穿透（详见 Algorithm 1）
- 对每个候选抓取姿态在 [[IsaacLab]] 中做稳定性仿真筛选，保留高质量样本

**结果**: 覆盖功能接触区域的多样抓取关键帧集合 $\mathcal{G}$。

#### 模块 2: 轨迹生成器（Trajectory Generator）

分三个子阶段：

**① 预抓取（Pre-grasping）**:
- GPU 加速的 [[RRT-Connect]] 算法 + 路径平滑
- 预抓取姿态：将指尖沿接触法线方向向外偏移，避开工具表面

**② 抓取（Grasping）**:
- 程序化策略：将指尖沿接触法线方向向内运动
- 对困难情况（接触配置复杂）训练短时程 [[PPO]] 策略，连接预抓取关键帧到抓取关键帧

**③ 工具驱动（In-hand Tool Actuation）**:
- 建模为通用目标到达问题
- 综合奖励：$r = r_\text{tool} + w_1 r_\text{hand} + w_2 r_\text{contact}$
- 使用多样化力随机化（force randomizations）提升鲁棒性

#### 模块 3: 视觉运动策略（Visuomotor Policy）

**架构**: [[点云]]条件的[[Diffusion Policy|扩散策略]]

- **感知编码**: Perceiver-style [[Transformer]] 将 512 个点（腕部坐标系）编码为 4 个 query token，latent dim = 128，4 个注意力头，4 个 block（含 self/cross-attention + feedforward MLP）
- **扩散头**: 轻量 Transformer，6 层，2 个注意力头，latent dim = 256，预测单步动作
- **输出**: Δ6D 腕部位姿 + Δ手部关节目标
- **控制**: 腕部用差分 IK 求解器；手部用低级 PD 控制器

**鲁棒化**: 对[[点云]]施加噪声扰动和随机部件掩蔽（random part masking）

#### 模块 4: 定制化指尖设计

**设计**: 扁平合规型指尖（flattened compliant fingertip）

- 几何构造：以规范截面 $C_0$ 沿高度方向各向异性缩放挤出，取凸包
- 制造：3D 打印模具和骨架，注射硅胶铸造
- 效果：硅胶层被动形变包裹工具表面，增大有效接触面积，稳定捏合力

---

## 关键公式

### 公式 1: [[PPO|工具驱动综合奖励]]

$$
r = r_\text{tool} + w_1 r_\text{hand} + w_2 r_\text{contact}
$$

**含义**: 工具驱动阶段的综合奖励函数，权衡工具状态误差、手部姿态误差和接触质量。

**符号说明**:
- $r_\text{tool}$: 工具状态奖励
- $r_\text{hand}$: 手部姿态奖励
- $r_\text{contact}$: 接触奖励
- $w_1 = 0.1, w_2 = 0.15$: 权重系数

---

### 公式 2: [[PPO|工具状态奖励]]

$$
r_\text{tool} = \exp(-\beta_{11}\|q - \tilde{q}\|_1) + \lambda_1 \exp\!\left(-\beta_{12}\|x - \tilde{x}\|_2 - \beta_{13}\|\log(R^\top \tilde{R})^\vee\|_2\right)
$$

**含义**: 衡量工具关节角度误差（旋转/平移关节）和工具末端位姿误差（位置 + 旋转）。

**符号说明**:
- $q, \tilde{q}$: 工具当前关节角与目标关节角
- $x, \tilde{x}$: 工具末端当前位置与目标位置
- $R, \tilde{R}$: 当前与目标旋转矩阵；$(\cdot)^\vee$ 表示旋转矩阵对数映射的轴角向量
- $\beta_{11} = 10.0$（旋转关节）或 $50.0$（平移关节）；$\beta_{12} = 100.0$；$\beta_{13} = 5.0$
- $\lambda_1 = 0.3$: 位姿项权重

---

### 公式 3: [[PPO|手部姿态奖励（工具驱动阶段）]]

$$
r_\text{hand} = \exp(-\beta_{21}\|q_h - \tilde{q}_h\|_1)
$$

**含义**: 工具驱动阶段仅惩罚手部关节角度与目标的偏差，无位姿项。

**符号说明**:
- $q_h, \tilde{q}_h$: 手部当前关节角与目标关节角
- $\beta_{21} = 10.0$

---

### 公式 4: [[PPO|手部姿态奖励（抓取阶段）]]

$$
r_\text{hand} = \exp(-\beta_{21}\|q_h - \tilde{q}_h\|_1) + \lambda_2 \exp\!\left(-\beta_{22}\|x_h - \tilde{x}_h\|_2 - \beta_{23}\|\log(R^\top \tilde{R})^\vee\|_2\right)
$$

**含义**: 抓取阶段额外惩罚手部末端位姿误差，确保手部以正确姿态接近工具。

**符号说明**:
- $x_h, \tilde{x}_h$: 手部末端当前与目标位置
- $\beta_{22} = 50.0$；$\beta_{23} = 5.0$
- $\lambda_2 = 10.0$: 位姿项权重（较大，抓取阶段位姿更关键）

---

### 公式 5: [[Diffusion Policy|接触奖励]]

$$
r_\text{contact} = \sum_{i \in \text{Finger},\ j \in \text{Tool}} \mathbf{1}[f_{ij} > \varepsilon]
$$

**含义**: 统计手指与工具之间超过阈值 $\varepsilon$ 的接触力对数，鼓励建立有效多点接触。

**符号说明**:
- $f_{ij}$: 第 $i$ 根手指与工具第 $j$ 个接触点之间的接触力
- $\varepsilon$: 接触力阈值

---

### 公式 6: [[Diffusion Policy|策略训练目标]]

$$
\mathcal{L} = \mathbb{E}_{(o_i, a_i) \sim \mathcal{D},\ t \sim \mathcal{U}[0,T]} \left[\|a_i - \pi(\tilde{a}_i, o_i, t)\|^2\right]
$$

**含义**: 扩散策略的去噪训练损失，在随机扩散时间步 $t$ 预测干净动作。

**符号说明**:
- $(o_i, a_i)$: 数据集中的观测-动作对
- $\tilde{a}_i$: 在时间步 $t$ 被加噪的动作
- $\pi(\cdot)$: 策略网络（扩散模型）
- $T$: 扩散总步数

---

### 公式 7: [[Impedance Control|PD 控制律]]

$$
\tau = K_p e + K_d \dot{e}
$$

**含义**: 手部低级 PD 控制器，根据关节位置误差和速度误差计算力矩。

**符号说明**:
- $e$: 关节位置误差
- $\dot{e}$: 关节速度误差
- $K_p, K_d$: 比例/微分增益矩阵

---

### 公式 8: 指尖几何构造

$$
C_h = \{(x \cdot w_x(h),\ y \cdot w_y(h),\ h) \mid (x, y, 0) \in C_0\}
$$

$$
M = \mathrm{ConvHull}\!\left(\bigcup_{z \in [0, H]} C_z\right)
$$

**含义**: 通过对规范截面 $C_0$ 沿高度方向各向异性缩放并取凸包，构造扁平合规型指尖几何体。

**符号说明**:
- $C_0$: 规范横截面形状
- $w_x(h), w_y(h)$: 高度 $h$ 处的 $x/y$ 方向缩放系数
- $H$: 指尖总高度

---

## 关键图表

### Figure 1: 系统总览

![Figure 1](https://arxiv.org/html/2606.13677v1/x1.png)

**说明**: Mana 框架概览。系统能够抓取并操作四类关节型工具（夹钳、钳子、晾衣夹、注射器），采用零样本 sim-to-real 迁移，成功率约 70%。左侧展示抓取各类工具，右侧展示功能性使用任务。

---

### Figure 2: 关节型工具操作的物理挑战

![Figure 2](https://arxiv.org/html/2606.13677v1/x2.png)

**说明**: 三大物理挑战：① 对接触点和力配置高度敏感（微小误差导致工具滑脱）；② 抓取与驱动动力学耦合（稳定抓取与施加驱动力相互干扰）；③ 遥操作基于位置控制，无法产生足够的捏合力，尤其对小型工具。

---

### Figure 3: Mana 数据系统概览

![Figure 3](https://arxiv.org/html/2606.13677v1/x3.png)

**说明**: 粗到精的三阶段数据生成流水线。从用户标注的功能区域出发，Lightning Grasp+ 生成抓取关键帧；[[RRT-Connect]] 规划预抓取轨迹；[[PPO]] 训练 RL 策略完成接触丰富的抓取和工具驱动，最终输出用于策略学习的轨迹数据集。

---

### Figure 4: 控制器架构

![Figure 4](https://arxiv.org/html/2606.13677v1/x4.png)

**说明**: 基于[[点云]]的扩散策略架构（黄色模块）。Perceiver Transformer 将 512 个腕部坐标系点云编码为 4 个 latent token，轻量 Transformer 扩散头预测 Δ6D 腕部位姿和 Δ手部关节目标。[[Sim-to-Real]] 通过点云噪声扰动和部件掩蔽实现鲁棒化。

---

### Figure 5: 机器人硬件

![Figure 5](https://arxiv.org/html/2606.13677v1/x5.png)

**说明**: 左：定制指尖设计——扁平合规型硅胶指尖，被动形变包裹工具，提升抓握稳定性。右：部署环境——7-DoF xArm7 机械臂 + 16-DoF Allegro 手，配 Intel RealSense D435 RGB-D 相机。

---

### Figure 6: 实验对象与仿真环境

![Figure 6](https://arxiv.org/html/2606.13677v1/x6.png)

**说明**: 四类关节型工具各两个实例（共 8 个），覆盖不同尺寸、形状和关节类型（旋转/平移）。工具厚度 0.8–1.5 cm，所需驱动力 3–7 N，通过相机扫描建立仿真模型。

---

### Figure 7: 消融实验

![Figure 7](https://arxiv.org/html/2606.13677v1/x7.png)

**说明**: 任务成功率随轨迹数量和关键帧多样性的变化曲线。数据量、抓取状态多样性和力随机化三者缺一均导致显著性能下降，证明 Mana 数据生成各组件的必要性。

---

### Figure 8（附录）: Lightning Grasp+ 碰撞感知 IK

![Figure 8](https://arxiv.org/html/2606.13677v1/x8.png)

**说明**: 碰撞感知运动学优化流程。检测穿透中心 $p$ 和分离向量 $d$，通过迭代 NearestProjection + LM-step IK 在保持接触约束的同时消除碰撞穿透。

---

### Figure 9（附录）: 桌面抓取样本

![Figure 9](https://arxiv.org/html/2606.13677v1/figures/grasp_samples.png)

**说明**: Lightning Grasp+ 在四类工具上生成的桌面抓取关键帧示例，展示覆盖功能接触区域的多样抓取构型。

---

### Table 1: 主要实验结果（抓取与开/闭成功率）

| 工具 | 方法 | 抓取（实例1/2） | 开（实例1/2） | 关（实例1/2） |
|------|------|----------------|--------------|--------------|
| 夹钳（Tongs） | **Mana（ours）** | **0.8 / 0.8** | **0.8 / 0.8** | **0.7 / 0.8** |
| 夹钳（Tongs） | 遥操作 | 0.3 / 0.3 | 0.1 / 0.2 | 0.3 / 0.3 |
| 钳子（Pliers） | **Mana（ours）** | **0.7 / 0.6** | **0.7 / 0.7** | **0.7 / 0.7** |
| 晾衣夹（Clothespins） | **Mana（ours）** | **0.8 / 0.7** | **0.8 / 0.7** | **0.6 / 0.7** |
| 注射器（Syringes） | **Mana（ours）** | **0.7 / 0.5** | **0.6 / 0.6** | — |

**关键发现**: Mana 在所有工具类别上均达到约 70% 的成功率，大幅超越遥操作基线（~30%，且遥操作对晾衣夹完全失败）。

---

### Table 2: 功能性任务演示结果

| 任务 | 成功次数 / 总次数 |
|------|-----------------|
| 夹钳取物（Tong Pick） | 7 / 10 |
| 钳子剪切（Plier Cut） | 5 / 10 |
| 晾衣夹使用（Clothespin Use） | 6 / 10 |
| 注射器推注（Syringe Inject） | 5 / 10 |

**关键发现**: 端到端功能性任务成功率 50%–70%，证明零样本 sim-to-real 迁移后策略具备实际应用能力。注：细对齐阶段辅以腕部遥操作。

---

## 实验

### 实验设置

| 类别 | 内容 |
|------|------|
| 机器人平台 | 7-DoF xArm7 + 16-DoF Allegro Hand |
| 感知 | Intel RealSense D435 RGB-D 相机 + SAM 3 分割 + Fast Foundation Stereo 点云 |
| 测试对象 | 夹钳、钳子、晾衣夹、注射器各 2 个实例（共 8 个） |
| 仿真平台 | [[IsaacLab]]（4096 并行环境） |
| 基线 | (1) 开环 Mana 策略；(2) 几何重定向遥操作（1 小时练习） |

### 实现细节

- **RL 优化器**: [[PPO]]，$\gamma=0.99$，$\varepsilon=0.2$（clip），GAE $\lambda=0.95$
- **RL 并行环境**: 4096，minibatch 8192，每 8 步更新，5 个优化 epoch
- **RL 学习率**: $1\times10^{-4}$，KL 调度器（阈值 0.01）
- **策略优化器**: [[AdamW]]，lr=$1\times10^{-4}$，batch 128，weight decay 0.01
- **策略训练步数**: $2\times10^5$，余弦调度 + 1000 步 warmup，终止 lr=$1\times10^{-6}$
- **域随机化**: 动作噪声 $\pm U[-0.1, 0.1]$，PD 增益 $\times U[0.8, 1.2]$，质量 $\times U[0.8, 1.2]$，摩擦系数 $\times U[0.5, 1.5]$，外部力扰动 $\mathcal{N}(0, \sigma^2)$（$\sigma = \alpha mg$，$\alpha \sim U[0.05, 0.2]$）

### 消融实验关键发现

- 真实世界性能随**轨迹数量**和**关键帧多样性**的增加而提升，数据量是关键瓶颈
- **力随机化**对 sim-to-real 迁移至关重要，缺失则性能大幅下降
- 在功能接触区域密集采样抓取构型（探索多样接触模式）是必要条件

---

## 批判性思考

### 优点

1. **程序化数据生成避免了繁琐的人工示教**: 每种工具 < 1 分钟配置，比遥操作数据采集高效数十倍
2. **零样本 sim-to-real 成功率远超遥操作基线**: 证明动画启发的关键帧方法对接触丰富任务有效
3. **硬件-软件协同设计**: 定制硅胶指尖与扩散策略协同，规避了高精度力控传感器的需求
4. **系统覆盖完整操作链路**: 从桌面物体获取到功能性操作的全流程设计

### 局限性

1. **力量上限**: 电机力矩不足，无法操作需 >10 N 的硬工具
2. **手部尺寸**: Allegro Hand 约为人手 2 倍，限制了能抓取的工具尺寸范围
3. **遮挡下的滑脱检测**: 手指遮挡工具时接触状态难以在线估计
4. **技能链接依赖遥操作**: 细对齐阶段（将工具对准操作目标）仍需人工辅助，尚未实现全自主

### 潜在改进方向

1. 引入触觉传感器（如 [[AnySkin]]）替代基于视觉的接触估计，改善遮挡问题
2. 探索 power grasp 策略以扩大可操作工具范围
3. 学习技能链接策略，实现从物体获取到功能操作的全自主

### 可复现性评估

- [ ] 代码开源（项目主页暂无代码链接）
- [ ] 预训练模型（暂未发布）
- [x] 训练细节完整（附录 A 提供详细超参数）
- [ ] 数据集可获取（工具 CAD 模型和轨迹数据未公开）

---

## 关联笔记

### 基于

- [[Diffusion Policy]]: 核心策略学习框架
- [[PPO]]: RL 训练算法
- [[IsaacLab]]: 仿真训练平台
- [[Sim-to-Real]]: 核心迁移范式

### 对比

- [[Dex-Retargeting]]: 几何重定向遥操作基线，用于评估难度

### 方法相关

- [[点云]]: 感知输入表示
- [[RRT-Connect]]: 预抓取运动规划
- [[Transformer]]: 策略网络架构（Perceiver 编码器 + 扩散头）

### 硬件/平台相关

- [[Allegro Hand]]: 16-DoF 灵巧手平台
- [[Intel RealSense D435i]]: RGB-D 感知传感器
- [[AdamW]]: 策略训练优化器

---

## 速查卡片

> [!summary] Mana: Dexterous Manipulation of Articulated Tools
> - **核心**: 受动画启发的粗到精数据生成框架，解决关节型工具灵巧操作问题
> - **方法**: Lightning Grasp+ 关键帧 → RRT+RL 轨迹生成 → 点云扩散策略零样本迁移
> - **结果**: 四类工具约 70% 成功率，显著超越遥操作基线（~30%）
> - **代码**: [项目主页](https://zhaohengyin.github.io/mana)

---

*笔记创建时间: 2026-06-13*
