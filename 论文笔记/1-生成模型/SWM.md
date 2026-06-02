---
title: "stable-worldmodel-v1: Reproducible World Modeling Research and Evaluation"
method_name: "SWM"
authors: [Lucas Maes, Quentin Le Lidec, Dan Haramati, Nassim Massaudi, Damien Scieur, Yann LeCun, Randall Balestriero]
year: 2026
venue: arXiv
tags: [world-model, evaluation, benchmark, infrastructure, reproducibility, jepa, robustness, embodied-ai]
zotero_collection: 1-生成模型
image_source: online
arxiv_html: https://arxiv.org/html/2602.08968v2
created: 2026-06-02
---

# 论文笔记：stable-worldmodel-v1: Reproducible World Modeling Research and Evaluation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Mila & Université de Montréal、NYU、Brown University、Samsung SAIL AI Lab |
| 日期 | February 2026（v2） |
| 项目主页 | https://arxiv.org/abs/2602.08968 |
| 对比基线 | [[PLDM]]、[[DINO-WM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2602.08968) / [PDF](https://arxiv.org/pdf/2602.08968) / [HTML](https://arxiv.org/html/2602.08968v2) |
| 一作 | Lucas Maes（Mila / UdeM） |
| 通讯 | Randall Balestriero（Brown）、Yann LeCun（NYU） |

---

## 一句话总结

> SWM 是 LeCun 团队打造的[[世界模型]]研究基础设施，统一 World API、可控 [[Factors of Variation|FoV]] 环境套件与规划/评测协议，目标是让 WM 研究像写 PyTorch 一样可复现、可比较、可鲁棒性量化。

---

## 核心贡献

1. **统一 World 接口**: 把 [[Gymnasium]] 风格的环境抽象成 `swm.World` 对象，把"环境步进 + 策略调用 + 多进程并行"三件耦合的事解耦；policy 通过 `get_action(world.infos)` 接入，数据采集和评测共用同一个 step 循环，告别 [[DINO-WM]] / [[PLDM]] 那种 policy/env/eval 全部缠在一起的 monolith。
2. **可控 FoV 环境套件**: 提供 **16 个**跨 2D/3D、操作/导航/控制的环境（含 [[Push-T]]、[[TwoRoom]]、[[DMControl]]、[[OGBench]]），每个环境暴露 **6-17 个**可控的 [[Factors of Variation]]，覆盖视觉（颜色/纹理）、几何（尺寸/位置/形状）、物理（摩擦/质量/速度）三类属性。
3. **规划与评测套件**: 内置 [[CEM]]（默认）、[[MPPI]] 与梯度型规划器，统一 `PlanConfig` 接口（horizon / receding horizon / warm start）；提供 online + offline 两种 goal-conditioned 评测协议。
4. **DINO-WM 鲁棒性实证**: 用 SWM 对 [[DINO-WM]] 做零样本 FoV 扰动测试，揭示其在分布外的脆弱性 —— 在 default Push-T 上 **94%** 成功率，单一 FoV 扰动后掉到 **4-20%**，提供[[Zero-shot Robustness]]的首批量化证据。
5. **工程指标领先**: 3562 LOC、73% 测试覆盖、type checking、文档齐全，对比 [[PLDM]]（6796 LOC、0% 覆盖）与 [[DINO-WM]]（4349 LOC、0% 覆盖）。

---

## 问题背景

### 要解决的问题

[[世界模型]]（World Model, WM）研究存在严重的**实现碎片化**：每个团队都在自己造轮子，环境/数据/评测协议互不兼容，导致：

- 复现一篇 WM 论文要花 1-2 周搭环境
- 跨论文比较的数字不可信（环境实现都不一样）
- [[Zero-shot Robustness|鲁棒性]]研究无标准，每篇论文自己 hack 扰动机制

作者举的标志性例子：[[PLDM]] 和 [[DINO-WM]] 都重新实现了 TwoRoom 环境，diff 高达 **81 删除、86 新增、18 修改** —— 同一个环境的两份不可比实现。

### 现有方法的局限

- **[[Gymnasium]]** 解决了 RL 环境接口，但 WM 还需要**数据采集 + 规划 + goal-conditioned 评测**，Gym 不覆盖。
- **[[VBench]] / [[WBench]]** 是视频质量评测，跟 control 任务无关，不能测"WM 当 planner"的下游能力。
- **[[DreamerV3]] / [[TD-MPC]]** 这类任务特定的 WM 代码库高度耦合（policy / env / eval 缠在一起），几乎无法 fork 出来单独跑 baseline。
- **[[JEPA]] 系列**（[[V-JEPA]]、[[I-JEPA]]、[[PLDM]]）虽然有[[Self-Supervised Learning|自监督]]的理论统一性，但实现差异极大，缺乏共享 backbone。

### 本文的动机

借鉴 PyTorch、HuggingFace 的成功路径：**不发明新算法，提供一套大家都能用的基础设施**。具体哲学：

- "你的 codebase 不要动，我提供环境和评估"
- World 抽象解耦：`world.step()` + `policy.get_action()` 两件事
- 所有 FoV 通过配置字典声明式控制，复现性由代码保证而非 README
- 类比：[[StableWM]] 之于 WM 研究 ≈ [[Gymnasium]] 之于 RL ≈ HuggingFace 之于 NLP

---

## 方法详解

### 系统架构

<!-- SWM 不是单一模型，而是一套库；这里描述其系统架构 -->

SWM 采用 **modular library** 架构，核心由五个组件构成：

- **输入侧**: 用户提供 `Policy` 类（实现 `get_action(world.infos)`）与 `PlanConfig` 配置
- **核心抽象**: [[World 抽象]] 统一管理多并行环境实例（默认 `num_envs=8`）
- **执行循环**: `world.reset()` → `world.step()` 内部把观测/状态/动作/奖励写到 `world.infos` 字典
- **输出侧**: 数据采集到 [[HDF5]] / image folder / mp4，评测产出标准化指标（success rate / return / distance-to-goal）
- **依赖栈**: 基于 [[Gymnasium]]、[[MuJoCo]]、[[OGBench]]、[[DMControl]]，可选 GPU rollout

### 核心模块

#### 模块 1: World Interface

**设计动机**: 把"环境步进 + 策略调用 + 多进程并行"这三件耦合的事拆开，让[[策略]]实现只关心动作，不关心环境如何并行。

**具体实现**:

- `swm.World(env_id, num_envs=N)` 创建 N 个并行 env
- 用 `world.set_policy(policy)` 注入策略对象
- `world.step()` 内部同步推进所有 env，结果原地写入 `world.infos` 字典
- 数据采集 (`world.record_dataset`) 与评测共用同一份 step 循环
- 关键代码：

```python
import stable_worldmodel as swm
world = swm.World('swm/PushT-v1', num_envs=8)
world.set_policy(YourExpertPolicy())
world.reset()
world.step()
world.record_dataset(num_episodes=1000)
```

#### 模块 2: Factors of Variation (FoV)

**设计动机**: WM 的[[Zero-shot Robustness|鲁棒性]]研究需要**可控的分布偏移**，传统做法是 hack 渲染器，难以复现。SWM 把 [[Factors of Variation|FoV]] 抽象成**声明式配置**。

**具体实现**:

- 每个环境暴露一组**命名 FoV**，例如 [[Push-T]] 暴露 16 个：
  - 视觉类：`agent.color`、`block.color`、`background.color`
  - 几何类：`agent.scale`、`block.shape`、`goal.position`、`goal.angle`
  - 物理类：`agent.velocity`、`block.mass`
- 用户通过 `options={"variation": ["agent", "block.color"]}` 在 `record_dataset` 调用时选择扰动哪些维度
- FoV 在 train/test 分布上独立采样，天然支持 OOD 评测，也支持 [[Continual Learning]] 设置

#### 模块 3: Planning 算法套件

**设计动机**: WM 当 planner 是经典用法，但 [[CEM]] / [[MPPI]] 这种采样规划器各家实现 hyperparam 不一致。

**具体实现**:

- 内置 [[CEM]]（默认，300 采样）、[[MPPI]]、梯度型（SGD/Adam）三类规划器
- 统一 `PlanConfig` 接口：`horizon`、`receding_horizon`、`warm_start`、`num_samples`、`elite_frac`
- 与 `World` 接口完全解耦，任何符合 `get_action` 协议的 planner 都能接入
- 自动支持 receding horizon 控制（[[模型预测控制|MPC]] 风格）

#### 模块 4: 评测协议

**设计动机**: goal-conditioned 评测有两种合理范式，社区没有统一。

**具体实现**:

- **Online 评测**: agent 自己采初始状态/目标，直接交互
- **Offline 评测**: 从预录数据集采样轨迹，保证可达性
- 指标：success rate、average return、distance-to-goal、planning latency

#### 模块 5: Baseline 库

**设计动机**: 论文宣称的"4 个 baseline"，给社区一个最小可比的 WM 起点。

**具体实现**:

- 复现 [[DINO-WM]]（[[DINOv2]] + latent predictor）作为 foundation-based 代表
- 复现 [[PLDM]] 作为端到端 [[JEPA]] 代表
- 简单 MLP latent dynamics 作为消融基线
- 第 4 个 baseline 未在论文正文具名（推测为 random / oracle policy）

---

## 关键公式

<!-- 公式标题使用 [[概念|名称]] 格式链接到概念库 -->
<!-- SWM 是基础设施论文，公式集中在 WM 形式化、评测和规划上 -->

### 公式 1: [[世界模型|World Model 形式化]]

$$
\hat{z}_{t+1} \;=\; f_\theta\!\left(z_t,\, a_t\right), \qquad z_t \;=\; \mathrm{enc}_\phi(o_t)
$$

**含义**: SWM 评测的目标对象 —— [[Latent World Model]] 的核心定义。把观测 $o_t$ 编码成 latent $z_t$，学一个 latent 动力学 $f_\theta$ 在该空间预测下一时刻。在 [[DINO-WM]] 中 $\mathrm{enc}_\phi$ 是冻结的 [[DINOv2]]，在 [[PLDM]] 中是端到端 [[JEPA]] 联合训练，在 [[DreamerV3]] 中 $f_\theta$ 还伴随 reward / value head。SWM 把这个抽象统一成 `world.infos['latent']` 这种字典字段，方便 swap encoder。

**符号说明**:

- $o_t \in \mathcal{O}$: 像素 / 状态观测
- $z_t \in \mathbb{R}^d$: latent 表征
- $a_t \in \mathcal{A}$: 动作（连续或离散）
- $\mathrm{enc}_\phi$: encoder（冻结的 [[DINOv2]] / [[V-JEPA]] 或联合学习）
- $f_\theta$: latent dynamics predictor

### 公式 2: [[JEPA|Joint-Embedding Predictive 损失]]

$$
\mathcal{L}_{\mathrm{JEPA}}(\theta, \phi) \;=\; \mathbb{E}_{(o_t, a_t, o_{t+1})}\Big[ \big\| f_\theta\!\left(\mathrm{enc}_\phi(o_t),\, a_t\right) - \mathrm{sg}\!\left(\mathrm{enc}_{\bar\phi}(o_{t+1})\right) \big\|^2 \Big]
$$

**含义**: SWM 实证关注的核心训练目标 —— [[JEPA]] 风格的潜空间预测损失。**关键**：右侧的 target encoder $\mathrm{enc}_{\bar\phi}$ 加 stop-gradient（`sg`）防止平凡解（[[表征坍塌]]）。[[DINO-WM]] 通过冻结 $\mathrm{enc}_\phi$ 来回避；[[V-JEPA]] / [[PLDM]] 通过 EMA target + 辅助正则；最新的 LeWM 通过 SIGReg 单一原则取代。SWM 让用户在同一 World API 下切换这些 encoder 策略。

**符号说明**:

- $\bar\phi$: target encoder 参数（EMA 或冻结）
- $\mathrm{sg}$: stop-gradient 算子
- $\| \cdot \|^2$: latent 空间 L2 距离

### 公式 3: [[CEM|Cross-Entropy Method 规划]]

$$
a_{t:t+H}^{*} \;=\; \arg\min_{a_{t:t+H}} \; \mathbb{E}_{\hat{z}_{t+H} \sim f_\theta(\cdot \mid z_t, a_{t:t+H})} \Big[ c(\hat{z}_{t+H},\, z_g) \Big]
$$

**含义**: 在已学到的世界模型 $f_\theta$ 上，搜索一段长度为 $H$ 的动作序列，使得 rollout 后的终态 $\hat{z}_{t+H}$ 离目标 latent $z_g$ 最近。SWM 的默认 planner 用 [[CEM]] 迭代式从高斯采样 → 选 top-k elite → 重新拟合高斯，逼近这个 argmin。[[TD-MPC]] 风格的 planner 会在公式右侧再加一个 $V_\psi(\hat{z}_{t+H})$ 价值项。

**符号说明**:

- $f_\theta$: 学到的潜在动力学，即 [[DINO-WM]] 或 [[PLDM]] 这种 WM
- $a_{t:t+H}$: 长度为 $H$ 的动作序列（规划 horizon）
- $\hat{z}_{t+H}$: WM 预测的终态（在 [[DINO-WM]] 里是 [[DINOv2]] latent）
- $z_g$: 目标 latent
- $c(\cdot, \cdot)$: cost 函数，常用 latent 空间欧氏距离

### 公式 4: [[模型预测控制|Receding Horizon Control]]

$$
a_t \;=\; \arg\min_{a_t} \; \mathbb{E}\!\Big[\, \sum_{k=0}^{H-1} \gamma^k\, c(\hat{z}_{t+k}, z_g) \,\Big], \qquad \hat{z}_{t+k+1} = f_\theta(\hat{z}_{t+k}, a_{t+k})
$$

**含义**: [[模型预测控制|MPC]] 风格的 receding horizon 控制。每一步只执行规划序列的第一个动作 $a_t$，下一步用新观测重新规划。这是 SWM `PlanConfig.receding_horizon=True` 时的默认行为，缓解 WM 长程预测的误差累积问题。

**符号说明**:

- $\hat{z}_{t+k}$: WM 在 latent 空间的 $k$ 步前推预测
- $\gamma \in (0, 1]$: 折扣因子
- $H$: planning horizon
- 实现细节：每个时间步重新求解，只执行首动作

### 公式 5: [[Factors of Variation|FoV 鲁棒性评测]]

$$
\mathrm{Robustness}(\pi, \mathcal{V}) \;=\; \mathbb{E}_{v \sim p(\mathcal{V})} \Big[\, \mathbb{E}_{\tau \sim \pi,\, \mathrm{env}(v)} \big[\, \mathbb{1}\{\mathrm{success}(\tau)\}\, \big] \,\Big]
$$

**含义**: 在 FoV 集合 $\mathcal{V}$ 上的边际化成功率 —— SWM 的核心评测指标。固定策略 $\pi$，对每一个 FoV 配置 $v$（如改变 agent 颜色）跑 episode，统计成功的轨迹比例。WM [[Zero-shot Robustness|鲁棒性]] = 不同 $v$ 下的平均 / 最差成功率。论文报告的 4-20% 数字就是这个指标在 [[DINO-WM]] 上的计算结果。

**符号说明**:

- $\pi$: 待评测的策略（如基于 [[DINO-WM]] 的 [[CEM]] planner）
- $\mathcal{V}$: FoV 集合（color、size、angle、position、shape、velocity 等）
- $v \sim p(\mathcal{V})$: 从 FoV 分布采样的扰动配置
- $\tau$: 在扰动环境 $\mathrm{env}(v)$ 下用策略 $\pi$ rollout 出的轨迹
- $\mathbb{1}\{\cdot\}$: 任务成功指示函数

---

## 关键图表

### Figure 1: SWM Environment Suite / 环境套件总览

![Figure 1](https://arxiv.org/html/2602.08968v2/x1.png)

**说明**: SWM 支持的代表性环境，分上下两行展示 default vs. perturbed（启用 FoV）两种渲染。从左到右是：(a) [[Push-T]] —— 蓝色 agent 推 T 形块到绿色锚点（2D 操作）；(b) [[TwoRoom]] —— 红色 agent 穿过门到达绿色目标（2D 导航）；(c) [[DMControl]] Humanoid —— 3D 人形 locomotion；(d) [[OGBench]] Scene —— 机械臂操作。每个环境同时附带的 perturbed 版本通过修改 agent 颜色 / 尺寸 / 物理参数生成，用同一套 [[Factors of Variation|FoV]] API 控制。

### Figure 2: All 16 Environments / 全部环境可视化（附录 D）

![Figure 2](https://arxiv.org/html/2602.08968v2/x9.png)

**说明**: 附录 D 中展示全部 16 个支持的环境的可视化，包括 2D 操作（Push-T 系列）、2D 导航（TwoRoom、迷宫）、3D locomotion（[[DMControl]] Humanoid / Walker / Cheetah / Hopper）、3D 操作（[[OGBench]] Scene / Cube）。每个环境都通过同一个 `swm.World(env_id)` 接口实例化，FoV 数量从 6 到 17 不等。

### Table 1: Latent World-Model 代码库对比

| 指标 | **SWM (ours)** | [[PLDM]] | [[DINO-WM]] |
|------|-----|------|---------|
| Backend | PyTorch | PyTorch | PyTorch |
| Documentation | ✓ | ✗ | ✗ |
| Test Coverage | **73%** | 0% | 0% |
| Type Checking | ✓ | ✓ | ✗ |
| # Baselines | **4** | 1 | 1 |
| # Environments | **16** | 2 | 4 |
| # FoV (per env) | **6 – 17** | 0 | 0 |
| Lines of Code | 3562 | 6796 | 4349 |

**说明**: SWM 用**更少的代码**（3562 LOC vs PLDM 的 6796）支持更多 baseline 和环境，并且是唯一有文档与高测试覆盖率的库。LOC 少 ≠ 功能弱，反映 SWM 在抽象层做的解耦工作。这张表也是论文的"市场定位"声明：填补 WM 研究基础设施真空。

### Table 2: DINO-WM 在 Push-T 上的零样本鲁棒性

| Factor | Property | Success Rate (%) |
|--------|----------|------------------|
| **Baseline (No variation)** | — | **94.0** |
| Random policy (sanity) | — | 12.0 |
| Color | Anchor | 20.0 |
| Color | Agent | 18.0 |
| Color | Block | 18.0 |
| Color | Background | 10.0 |
| Size | Anchor | 14.0 |
| Size | Agent | 4.0 |
| Size | Block | 16.0 |
| Angle | Anchor | 12.0 |
| Angle | Agent | 12.0 |
| Position | Anchor | 4.0 |
| Shape | Agent | 18.0 |
| Shape | Block | 8.0 |
| Velocity | Agent | 14.0 |

**说明**: [[DINO-WM]] 在原分布上有 **94%** 成功率，但**任何**单一 FoV 扰动都让成功率掉到 **20% 以下**，最差的 `Size.Agent` 和 `Position.Anchor` 跌到 **4%**，部分扰动接近 random policy 水平（12%）。这是论文最有冲击力的实验结果：揭示当前 latent WM 的[[Zero-shot Robustness|零样本鲁棒性]]严重不足，FoV 评测协议给出了量化证据，为后续工作（鲁棒 WM、[[Continual Learning|持续学习]] WM）画出 baseline 线。

### Table 3: 16 个环境与 FoV 汇总（附录 D）

| Environment ID | # FoV | 主要可变属性 |
|----------------|-------|-------------|
| swm/PushT-v1 | 16 | agent color/scale/position, block 属性, goal 设置 |
| swm/TwoRoom-v1 | 17 | agent speed/radius, 门配置, 墙属性 |
| swm/OGBCube-v0 | 11 | 机械臂颜色, cube 位置/尺寸, lighting |
| swm/OGBScene-v0 | 12 | 机械臂端点, 物体目标, 视觉属性 |
| swm/HumanoidDMControl-v0 | 7 | 关节密度, 地面摩擦, lighting |
| swm/CheetahDMControl-v0 | 7 | 同上 |
| swm/HopperDMControl-v0 | 7 | 同上 |
| swm/ReacherDMControl-v0 | 8 | 关节属性 + visual |
| swm/WalkerDMControl-v0 | 8 | 关节属性 + visual |
| swm/AcrobotDMControl-v0 | 8 | 同上 |
| swm/PendulumDMControl-v0 | 6 | 物理参数 |
| swm/CartpoleDMControl-v0 | 6 | 物理参数 |
| swm/BallInCupDMControl-v0 | 9 | 球 / 杯子属性 |
| swm/FingerDMControl-v0 | 10 | 末端执行器属性 |
| swm/ManipulatorDMControl-v0 | 8 | 关节 + 物体 |
| swm/QuadrupedDMControl-v0 | 7 | 关节 + 地形 |

**说明**: 每个环境暴露的 FoV 数量随复杂度而变（最少 6，最多 17），但都通过相同的 `options={"variation": [...]}` 接口控制。完整列表是 SWM 的"环境矩阵"，作为社区共享 benchmark 的底座。

---

## 实验结果

### 数据集 / 环境

| 环境 | 类别 | 用途 |
|------|------|------|
| [[Push-T]] | 2D manipulation | 主要鲁棒性实验对象 |
| [[TwoRoom]] | 2D navigation | 复现 PLDM/DINO-WM 实验 |
| [[DMControl]] Humanoid/Walker | 3D locomotion | 验证 SWM 支持 MuJoCo |
| [[OGBench]] Scene/Cube | 3D manipulation | 复用 OGBench benchmarks |
| 其他 12 个环境 | 杂 | 见附录 D Table 3 |

### 实现细节

- **WM 主对象**: [[DINO-WM]]（基于 [[DINOv2]] latent 的 latent dynamics model）
- **规划器**: [[CEM]]（默认），planning horizon $H$ 可配置，receding horizon 开启，300 个采样
- **并行环境数**: `num_envs=8`（默认配置）
- **数据格式**: [[HDF5]]（默认），可选 image folder / mp4
- **代码量**: 3562 LOC，测试覆盖 73%，type checking 全覆盖
- **后端**: PyTorch + [[MuJoCo]] + [[Gymnasium]]

### DINO-WM 零样本鲁棒性实验（论文主实验）

**核心发现**：

1. **In-distribution baseline**：用 expert demonstration 数据训练 [[DINO-WM]]，在 default [[Push-T]] 上跑 [[CEM]] planner，取得 **94.0% 成功率** —— 与原论文吻合，证明 SWM 的 DINO-WM 复现正确。

2. **Random policy 下界**：随机策略在同一任务上取得 **12% 成功率**，说明任务本身不平凡（不是 trivially solvable）。

3. **FoV 扰动后**：每一类单 FoV 扰动（color / size / angle / position / shape / velocity）下，成功率掉到 **4-20%**：
   - 最不鲁棒：`Size.Agent` 4%、`Position.Anchor` 4% —— 几何扰动比颜色更致命
   - 较鲁棒：`Color.Anchor` 20%、`Shape.Agent` 18%、`Color.Agent` 18% —— 但仍远低于 ID
   - **关键观察**：`Color.Background` 仅 10%，说明 [[DINO-WM]] 严重依赖背景特征，违反了"world model 应该理解物理"的初衷

4. **结论**：当前最强的 latent WM 之一对**任意视觉/几何扰动几乎完全失效**，[[Zero-shot Robustness|鲁棒性]]远未达到 deployment 标准。这给 [[V-JEPA]]、[[Dreamer 4]]、[[TD-MPC2]] 等后续 WM 工作画出了一条必须超越的 baseline 线。

### 对工业界与具身 AI 研究的意义

- **复现红利**: 想做 "WM 当 planner" 或 "WM 当 policy backbone" 的研究者，可以直接 `pip install` SWM 跑 baseline，省下 1-2 周搭环境时间。
- **诚实评估**: FoV 扰动让 WM 论文不能再只报 in-distribution 数字，必须报鲁棒性分布。这对 embodied AI 走向 sim-to-real 至关重要。
- **标准化基准**: 类似 [[VBench]] 之于视频生成的角色，SWM 想成为 WM-for-control 方向的标准评测。
- **JEPA 路线推广**: 论文 LeCun 署名，自带 [[JEPA]] 推广效应，把 [[V-JEPA]] / [[I-JEPA]] / [[PLDM]] 这条 [[Self-Supervised Learning|自监督]]路线变成"有标准评测"的方向。

---

## 批判性思考

### 优点

1. **基础设施论文做对了一件事**：解耦 World / Policy / Planner，让用户不需要懂 SWM 内部就能接入。
2. **FoV 抽象漂亮**：声明式配置 + train/test 独立采样，把鲁棒性研究从 hack 变成标准操作。
3. **LeCun + Mila + Brown 名头**：[[DINO-WM]] 同源团队背书，至少保证 DINO-WM baseline 实现是正确的；Yann LeCun 亲自署名意味着 [[JEPA]] 路线的官方推广通道。
4. **开源完整**：代码 + 文档 + 测试覆盖率 73%，不是放个 README 就走人。
5. **诚实揭短**：用自己的库测自家团队的 DINO-WM 揭示 94% → 4-20% 的脆弱性，这种"自我祛魅"在基础设施论文里少见。

### 局限性

1. **没有方法层创新**：纯工程论文，不能用来发 SOTA paper，价值取决于社区认不认。需要至少 3-5 个 follow-up 用 SWM 跑实验才能确立标准地位。
2. **只测了 [[DINO-WM]] 一个 WM**：标题说"评测套件"，但实证只跑了一个 baseline。如果不补 [[PLDM]]、[[Dreamer 4]]、[[DreamerV3]]、[[TD-MPC]]、[[V-JEPA]] 衍生 WM 的鲁棒性对比，结论的代表性有限。
3. **环境偏简单**：[[Push-T]]、[[TwoRoom]] 都是 2D 玩具任务；[[DMControl]] 是单 agent locomotion，离真机操作（[[LIBERO]]、[[CALVIN]] 级别）还差很远。
4. **没有真机接口**：所有环境都是 simulation。WM 想 deploy 到真机，至少要给一个 [[sim-to-real]] 的 placeholder。
5. **FoV 设计可能有隐藏偏置**：扰动是合成的，"agent color" 改变在真世界里可能不存在，鲁棒性结论对真实分布偏移的迁移性需要验证。
6. **缺乏与 [[DreamerV3]] 类 WM 的对比**：论文聚焦 [[Latent World Model]] 中的 [[JEPA]] 系列（DINO-WM、PLDM），忽略了同样重要的 RSSM 系（DreamerV3）和 model-based RL 系（TD-MPC）。

### 潜在改进方向

1. 接入更多 WM baseline（[[V-JEPA]]、[[Dreamer 4]]、[[DreamerV3]]、[[TD-MPC2]]）做横向对比
2. 加入真机 / [[sim-to-real]] 模式（如 [[ManiSkill]] / [[IsaacLab]] 桥接）
3. 提供 FoV 自动课程（用 SWM 训鲁棒 WM 而不只是评测，结合 [[Continual Learning]]）
4. 与 [[OGBench]]、[[LIBERO]]、[[CALVIN]] 等已有 benchmark 互通格式
5. 增加 multi-task / multi-environment 训练协议（当前每个 env 独立训）

### 可复现性评估

- [x] 代码开源（库即论文）
- [x] 测试覆盖率 73%
- [x] 训练细节完整
- [x] 数据集可获取
- [x] Type checking 完整
- [ ] 真机部署接口（暂无）
- [ ] 多种 WM baseline 完整复现（只复现了 DINO-WM）

---

## 关联笔记

### 基于

- [[Gymnasium]]: 环境接口标准基础
- [[DINO-WM]]: 实验对象 + 团队同源
- [[PLDM]]: 对比的代码库 baseline
- [[Push-T]]: 主要实验环境
- [[JEPA]]: 论文背后的方法论根基（LeCun 推广）

### 对比

- [[VBench]]: 视频质量评测，目标不同（pixel quality vs control）
- [[WBench]]: 世界模型综合评测，但目标偏向 video quality
- [[OGBench]]: 操作 benchmark，被 SWM 包装进 World 接口
- [[LIBERO]]: VLA 评测的事实标准，SWM 的 WM 版定位
- [[DreamerV3]]: RSSM 系世界模型代表（未在 SWM 实证中包含）
- [[TD-MPC]]: model-based RL 代表（未在 SWM 实证中包含）
- [[V-JEPA]]: 视频自监督代表，可作 SWM 的 encoder candidate

### 方法相关

- [[CEM]]: 默认规划器
- [[MPPI]]: 备选规划器
- [[Factors of Variation]]: 核心鲁棒性抽象
- [[World 抽象]]: SWM 自身的核心 API
- [[模型预测控制|MPC]]: receding horizon 控制范式
- [[Self-Supervised Learning]]: 训练 latent encoder 的范式
- [[Latent World Model]]: SWM 关注的 WM 类别

### 鲁棒性 / 评测相关

- [[Zero-shot Robustness]]: SWM 的核心评测目标
- [[Continual Learning]]: FoV 支持的下游研究方向
- [[sim-to-real]]: 仍未跨越的 gap

### 硬件/数据相关

- [[HDF5]]: 数据集存储格式
- [[MuJoCo]]: 后端物理引擎
- [[DMControl]]: 复用其控制 suite
- [[TwoRoom]]: 复用 PLDM/DINO-WM 的导航环境
- [[DINO]] / [[DINOv2]]: encoder backbone

---

## 速查卡片

> [!summary] stable-worldmodel-v1 (SWM)
> - **核心**: [[世界模型]]研究基础设施（统一 World API + FoV 可控环境 + 规划/评测套件）
> - **方法**: 解耦 World/Policy/Planner，声明式 FoV 配置，并行 env 同步执行
> - **结果**: [[DINO-WM]] 在 default [[Push-T]] 上 94% 成功率，单一 FoV 扰动后掉到 **4-20%**
> - **环境**: 16 个，FoV 6-17 个/env，覆盖 2D/3D、操作/导航/控制
> - **代码**: 3562 LOC, 73% 测试覆盖，开源（库即论文）
> - **价值**: 想做 WM-as-planner / WM robustness 的人省 1-2 周搭环境时间
> - **作者**: Lucas Maes (Mila) + Yann LeCun (NYU) + Randall Balestriero (Brown)

---

*笔记创建时间: 2026-06-02*
