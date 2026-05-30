---
title: "stable-worldmodel-v1: Reproducible World Modeling Research and Evaluation"
method_name: "SWM"
authors: [Lucas Maes, Quentin Le Lidec, Dan Haramati, Nassim Massaudi, Damien Scieur, Yann LeCun, Randall Balestriero]
year: 2026
venue: arXiv
tags: [world-model, evaluation, benchmark, infrastructure, reproducibility, jepa, robustness]
zotero_collection: 1-生成模型
image_source: online
arxiv_html: https://arxiv.org/html/2602.08968
created: 2026-05-30
---

# 论文笔记：stable-worldmodel-v1: Reproducible World Modeling Research and Evaluation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Mila & Université de Montréal, NYU, Brown University, Samsung SAIL |
| 日期 | February 2026 |
| 项目主页 | https://arxiv.org/abs/2602.08968 |
| 对比基线 | [[PLDM]]、[[DINO-WM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2602.08968) / [PDF](https://arxiv.org/pdf/2602.08968) |

---

## 一句话总结

> SWM 是 LeCun 团队打造的 [[世界模型]] 研究基础设施，统一 World API、可控 [[Factors of Variation|FoV]] 环境套件与规划/评测协议，目标是让 WM 研究像写 PyTorch 一样可复现。

---

## 核心贡献

1. **统一 World 接口**: 把 [[Gymnasium]] 风格的环境抽象成 `swm.World` 对象，控制逻辑与执行解耦，policy 通过 `get_action(world.infos)` 即可接入，数据采集和评测共用同一个执行循环。
2. **可控 FoV 环境套件**: 提供 16 个跨 2D/3D、操作/导航/控制的环境（含 [[Push-T]]、[[TwoRoom]]、[[DMControl]]、[[OGBench]]），每个环境都暴露 6-17 个可控的 [[Factors of Variation]]，覆盖视觉（颜色/纹理）、几何（尺寸/位置）、物理（摩擦/质量）。
3. **规划与评测套件**: 内置 [[CEM]]、[[MPPI]] 及梯度型规划器 baseline，以及在线/离线两种 goal-conditioned 评测协议。
4. **DINO-WM 鲁棒性实证**: 用 SWM 对 [[DINO-WM]] 做零样本 FoV 扰动测试，揭示其在分布外的脆弱性（专家 demo 94% → 扰动后 4-20%）。

---

## 问题背景

### 要解决的问题

世界模型（World Model, WM）研究存在严重的**实现碎片化**：每个团队都在自己造轮子，环境/数据/评测协议互不兼容，导致：
- 复现一篇 WM 论文要花两周搭环境
- 跨论文比较的数字不可信（环境实现都不一样）
- 鲁棒性研究无标准（每篇论文自己 hack 扰动）

作者举例：[[PLDM]] 和 [[DINO-WM]] 都重新实现了 Two-Room 环境，diff 高达 81 删、86 加、18 改——同一个环境的两份不可比实现。

### 现有方法的局限

- **[[Gymnasium]]** 解决了 RL 环境接口，但 WM 需要额外的**数据采集 + 规划 + goal-conditioned 评测**，Gym 不覆盖。
- **[[VBench]] / [[WBench]]** 这类视频质量评测和 control 任务无关，不能测"WM 当 planner"的下游能力。
- 各家 WM 代码库（PLDM、DINO-WM）耦合度高，policy/env/eval 全部交织在一起，几乎无法 fork 出去单独跑 baseline。

### 本文的动机

借鉴 PyTorch、HuggingFace 的成功路径：**不发明新算法，提供一套大家都能用的基础设施**。具体哲学：
- "你的 codebase 不要动，我提供环境和评估"
- World 抽象解耦：`world.step()` + `policy.get_action()` 两件事
- 所有 FoV 通过配置字典声明式控制，复现性由代码保证而非 README

---

## 方法详解

### 模型架构

<!-- SWM 不是一个模型，而是一套库；这里描述其系统架构 -->

SWM 采用 **modular library** 架构，核心由四个组件构成：

- **输入侧**: 用户提供 `Policy` 类（实现 `get_action`）与 `PlanConfig` 配置
- **核心模块**: [[World 抽象]] 统一管理多并行环境实例（默认 `num_envs=8`）
- **执行循环**: `world.reset()` → `world.step()` 内部把观测/状态/动作/奖励写到 `world.infos` 字典
- **输出侧**: 数据采集到 [[HDF5]] / image / mp4，评测产出标准化指标
- **依赖**: 基于 [[Gymnasium]]、[[MuJoCo]]、[[OGBench]]、[[DMControl]]

### 核心模块

#### 模块 1: World Interface

**设计动机**: 把"环境步进 + 策略调用 + 多进程并行"这三件耦合的事拆开，让 [[策略]] 实现只关心动作，不关心环境如何并行。

**具体实现**:
- `swm.World(env_id, num_envs=N)` 创建 N 个并行 env
- 用 `world.set_policy(policy)` 注入策略对象
- `world.step()` 内部同步推进所有 env，结果原地写入 `world.infos` 字典
- 数据采集 (`world.record_dataset`) 与评测共用同一份 step 循环

#### 模块 2: Factors of Variation (FoV)

**设计动机**: WM 的鲁棒性研究需要可控的分布偏移，传统做法是 hack 渲染器，难以复现。SWM 把 FoV 抽象成**声明式配置**。

**具体实现**:
- 每个环境暴露一组命名 FoV，例如 [[Push-T]] 暴露 16 个（agent color/scale/position, block 属性, goal 设置等）
- 用户通过 `options={"variation": ["agent", "block.color"]}` 在 `record_dataset` 调用时选择扰动哪些维度
- FoV 在 train/test 分布上独立采样，天然支持 OOD 评测

#### 模块 3: Planning 算法套件

**设计动机**: WM 当 planner 是经典用法，但 [[CEM]] / [[MPPI]] 这种采样规划器各家实现 hyperparam 不一致。

**具体实现**:
- 内置 [[CEM]]（默认）、[[MPPI]]、梯度型（SGD/Adam）三类规划器
- 统一 `PlanConfig` 接口：`horizon`、`receding_horizon`、`warm_start` 等参数
- 与 `World` 接口完全解耦，任何符合 `get_action` 协议的 planner 都能接入

#### 模块 4: 评测协议

**设计动机**: goal-conditioned 评测有两种合理范式，社区没有统一。

**具体实现**:
- **Online 评测**: agent 自己采初始状态/目标，直接交互
- **Offline 评测**: 从预录数据集采样轨迹，保证可达性
- 指标：success rate、average return、distance-to-goal 等

---

## 关键公式

<!-- SWM 是基础设施论文，公式集中在评测和规划上 -->

### 公式 1: [[CEM|Cross-Entropy Method 规划]]

$$
a_{t:t+H}^{*} = \arg\min_{a_{t:t+H}} \; \mathbb{E}_{s_{t+H} \sim f_\theta(\cdot \mid s_t, a_{t:t+H})} \big[ c(s_{t+H}, g) \big]
$$

**含义**: 在已学到的世界模型 $f_\theta$ 上，搜索一段长度为 $H$ 的动作序列，使得 rollout 后的终态 $s_{t+H}$ 离目标 $g$ 最近。SWM 的默认 planner 用 [[CEM]] 迭代式从高斯采样 → 选 top-k elite → 重新拟合高斯，逼近这个 argmin。

**符号说明**:
- $f_\theta$: 学到的潜在动力学，即 [[DINO-WM]] 或 [[PLDM]] 这种 WM
- $a_{t:t+H}$: 长度为 $H$ 的动作序列（规划 horizon）
- $s_{t+H}$: WM 预测的终态（在 [[DINO-WM]] 里是 [[DINOv2]] latent）
- $g$: 目标状态（goal-conditioned 任务）
- $c(\cdot, \cdot)$: cost 函数，常用 latent 空间欧氏距离

### 公式 2: [[Factors of Variation|FoV 鲁棒性评测]]

$$
\mathrm{Robustness}(\pi, \mathcal{V}) = \mathbb{E}_{v \sim p(\mathcal{V})} \Big[ \mathbb{E}_{\tau \sim \pi, \mathrm{env}(v)} \big[ \mathbb{1}\{\mathrm{success}(\tau)\} \big] \Big]
$$

**含义**: 在 FoV 集合 $\mathcal{V}$ 上的边际化成功率。SWM 的核心评测：固定策略 $\pi$，对每一个 FoV 配置 $v$（如改变 agent 颜色）跑 episode，统计成功的轨迹比例。WM 鲁棒性 = 不同 $v$ 下的平均 / 最差成功率。

**符号说明**:
- $\pi$: 待评测的策略（如基于 [[DINO-WM]] 的 [[CEM]] planner）
- $\mathcal{V}$: FoV 集合（color、size、angle、position、shape、velocity 等）
- $v \sim p(\mathcal{V})$: 从 FoV 分布采样的扰动配置
- $\tau$: 在扰动环境 $\mathrm{env}(v)$ 下用策略 $\pi$ rollout 出的轨迹
- $\mathbb{1}\{\cdot\}$: 任务成功指示函数

### 公式 3: receding horizon 控制

$$
a_t = \arg\min_{a_t} \; \mathbb{E}\Big[ \sum_{k=0}^{H-1} \gamma^k c(\hat{s}_{t+k}, g) \Big], \quad \hat{s}_{t+k+1} = f_\theta(\hat{s}_{t+k}, a_{t+k})
$$

**含义**: receding horizon 控制（MPC 风格）：每一步只执行规划序列的第一个动作 $a_t$，下一步重新规划。这是 SWM `PlanConfig.receding_horizon=True` 时的默认行为，可缓解 WM 长程预测误差累积。

**符号说明**:
- $\hat{s}_{t+k}$: WM 在 latent 空间的 $k$ 步预测
- $\gamma$: 折扣因子
- $H$: planning horizon
- 每个时间步重新求解，只执行首动作

---

## 关键图表

### Figure 1: SWM Environment Suite / 环境套件总览

![Figure 1](https://arxiv.org/html/2602.08968/x1.png)

**说明**: SWM 支持的代表性环境，分上下两行展示 default vs. perturbed（启用 FoV）两种渲染。从左到右是 (a) [[Push-T]]：蓝色 agent 推 T 形块到绿色锚点；(b) [[TwoRoom]]：红色 agent 穿过门到达绿色目标；(c) [[DMControl]] Humanoid：3D 人形 locomotion；(d) [[OGBench]] Scene：机械臂操作。每个环境同时附带的 perturbed 版本通过修改 agent 颜色 / 尺寸 / 物理参数生成，用同一套 FoV API 控制。

### Figure 2: All 16 Environments / 全部环境可视化（附录 D）

![Figure 2](https://arxiv.org/html/2602.08968/x9.png)

**说明**: 附录 D 中展示全部 16 个支持的环境的可视化，包括 2D 操作（Push-T 系列）、2D 导航（TwoRoom、迷宫）、3D locomotion（DMC Humanoid/Walker/Cheetah）、3D 操作（OGBench Scene/Cube）。每个环境都通过同一个 `swm.World(env_id)` 接口实例化，FoV 数量从 6 到 17 不等。

### Table 1: Latent World-Model 代码库对比

| 指标 | SWM | [[PLDM]] | [[DINO-WM]] |
|------|-----|------|---------|
| Documentation | ✓ | ✗ | ✗ |
| Test Coverage | 73% | 0% | 0% |
| Type Checking | ✓ | ✓ | ✗ |
| Baselines 数量 | 4 | 1 | 1 |
| 环境数量 | 16 | 少数 | 少数 |
| Lines of Code | 3562 | 6796 | 4349 |

**说明**: SWM 用更少的代码（3562 LOC vs PLDM 的 6796）支持更多 baseline 和环境，并且是唯一有文档与高测试覆盖率的库。LOC 少 ≠ 功能弱，反映 SWM 在抽象层上做的解耦工作。

### Table 2: DINO-WM 在 Push-T 上的零样本鲁棒性

| FoV 扰动 | Success Rate (%) |
|----------|------------------|
| Expert demo (in-distribution) | 94.0 |
| Random policy (sanity check) | 12.0 |
| Color 扰动 | 4.0 – 12.0 |
| Size 扰动 | 6.0 – 18.0 |
| Angle 扰动 | 8.0 – 20.0 |
| Position 扰动 | 4.0 – 16.0 |
| Shape 扰动 | 6.0 – 14.0 |
| Velocity 扰动 | 8.0 – 18.0 |

**说明**: [[DINO-WM]] 在原分布上有 94% 成功率，但**任何**单一 FoV 扰动都让成功率掉到 20% 以下，部分扰动接近 random policy 水平。揭示当前 latent WM 的鲁棒性严重不足，FoV 评测协议给出了量化证据。

### Table 3: 16 个环境与 FoV 汇总（附录 D 节选）

| Environment ID | # FoV | 主要可变属性 |
|----------------|-------|-------------|
| swm/PushT-v1 | 16 | agent color/scale/position, block 属性, goal 设置 |
| swm/TwoRoom-v1 | 17 | agent speed/radius, 门配置, 墙属性 |
| swm/OGBCube-v0 | 11 | 机械臂颜色, cube 位置/尺寸, lighting |
| swm/HumanoidDMControl-v0 | 7 | 关节密度, 地面摩擦, lighting |

**说明**: 每个环境暴露的 FoV 数量随复杂度而变，但都通过相同的 `options={"variation": [...]}` 接口控制。完整 16 环境列表见论文附录 D Table 3。

---

## 实验结果

### 数据集 / 环境

| 环境 | 类别 | 用途 |
|------|------|------|
| [[Push-T]] | 2D manipulation | 主要鲁棒性实验对象 |
| [[TwoRoom]] | 2D navigation | 复现 PLDM/DINO-WM 实验 |
| [[DMControl]] Humanoid/Walker | 3D locomotion | 验证 SWM 支持 MuJoCo |
| [[OGBench]] Scene/Cube | 3D manipulation | 复用 OGBench benchmarks |
| 其他 12 个环境 | 杂 | 见附录 D |

### 实现细节

- **WM 主对象**: [[DINO-WM]]（基于 [[DINOv2]] latent 的 latent dynamics model）
- **规划器**: [[CEM]]（默认），planning horizon $H$ 可配，receding horizon 开启
- **并行环境数**: `num_envs=8`（默认配置）
- **数据格式**: [[HDF5]]（默认），可选 image folder / mp4
- **代码量**: 3562 LOC，测试覆盖 73%

### DINO-WM 零样本鲁棒性实验

**核心发现**：

1. **In-distribution baseline**：用 expert demonstration 在 default Push-T 上跑，[[DINO-WM]] + [[CEM]] 取得 **94% 成功率**
2. **Random policy 下界**：随机策略 12% 成功率（任务本身不平凡）
3. **FoV 扰动后**：每一类单 FoV 扰动（color / size / angle / position / shape / velocity）下，成功率掉到 **4-20%**，其中 color 扰动最严重
4. **结论**：当前最强的 latent WM 之一对**任意视觉/几何扰动几乎完全失效**，鲁棒性远未达到 deployment 标准

### 对工业界的意义

- **复现红利**: 想做"WM 当 planner"或"WM 当 policy backbone"的研究者，可以直接 `pip install` SWM 跑 baseline，省下 2 周搭环境
- **诚实评估**: FoV 扰动让 WM 论文不能再只报 in-distribution 数字，必须报鲁棒性分布
- **标准化基准**: 类似 [[VBench]] 之于视频生成的角色，SWM 想成为 WM-for-control 方向的标准评测

---

## 批判性思考

### 优点

1. **基础设施论文做对了一件事**：解耦 World / Policy / Planner，让用户不需要懂 SWM 内部就能接入。
2. **FoV 抽象漂亮**：声明式配置 + train/test 独立采样，把鲁棒性研究从 hack 变成标准操作。
3. **LeCun + Mila 名头**：[[DINO-WM]] 同源团队背书，至少保证 DINO-WM baseline 实现是正确的。
4. **开源完整**：代码 + 文档 + 测试覆盖率 73%，不是放个 README 就走人。

### 局限性

1. **没有方法层创新**：纯工程论文，不能用来发 SOTA paper，价值取决于社区认不认。
2. **只测了 [[DINO-WM]] 一个 WM**：标题说"评测套件"，但实证只跑了一个 baseline。如果不补 [[PLDM]]、[[Dreamer 4]]、[[DreamerV3]] 等的鲁棒性对比，结论的代表性有限。
3. **环境偏简单**：Push-T、TwoRoom 都是 2D 玩具任务；[[DMControl]] 是单 agent locomotion，离真机操作还差很远。
4. **没有真机接口**：所有环境都是 simulation。WM 想 deploy 到真机，至少要给一个 sim2real 的 placeholder。
5. **FoV 设计可能有隐藏偏置**：扰动是合成的，"agent color" 改变在真世界里可能不存在，鲁棒性结论对真实分布偏移的迁移性需要验证。

### 潜在改进方向

1. 接入更多 WM baseline（[[V-JEPA]]、[[Dreamer 4]]、[[DreamerV3]]、[[TD-MPC2]]）做横向对比
2. 加入真机 / [[sim-to-real]] 模式（如 [[ManiSkill]] / [[IsaacLab]] 桥接）
3. 提供 FoV 自动课程（用 SWM 训鲁棒 WM，而不只是评测）
4. 与 [[OGBench]]、[[LIBERO]] 等已有 benchmark 互通格式

### 可复现性评估

- [x] 代码开源（库即论文）
- [x] 测试覆盖率 73%
- [x] 训练细节完整
- [x] 数据集可获取
- [ ] 真机部署接口（暂无）

---

## 关联笔记

### 基于

- [[Gymnasium]]: 环境接口标准基础
- [[DINO-WM]]: 实验对象 + 团队同源
- [[PLDM]]: 对比的代码库 baseline
- [[Push-T]]: 主要实验环境

### 对比

- [[VBench]]: 视频质量评测，目标不同（pixel quality vs control）
- [[WBench]]: 世界模型综合评测，但目标偏向 video quality
- [[OGBench]]: 操作 benchmark，被 SWM 包装进 World 接口
- [[LIBERO]]: VLA 评测的事实标准，SWM 的 WM 版定位

### 方法相关

- [[CEM]]: 默认规划器
- [[MPPI]]: 备选规划器
- [[Factors of Variation]]: 核心鲁棒性抽象
- [[World 抽象]]: SWM 自身的核心 API
- [[HDF5]]: 数据集存储格式

### 硬件/数据相关

- [[MuJoCo]]: 后端物理引擎
- [[DMControl]]: 复用其控制 suite
- [[TwoRoom]]: 复用 PLDM/DINO-WM 的导航环境

---

## 速查卡片

> [!summary] stable-worldmodel-v1 (SWM)
> - **核心**: WM 研究基础设施（统一 World API + FoV 可控环境 + 规划/评测套件）
> - **方法**: 解耦 World/Policy/Planner，声明式 FoV 配置，并行 env 同步执行
> - **结果**: [[DINO-WM]] 在 default Push-T 上 94% 成功率，单一 FoV 扰动后掉到 4-20%
> - **代码**: 3562 LOC, 73% 测试覆盖，开源（库即论文）
> - **价值**: 想做 WM-as-planner / WM robustness 的人省 2 周搭环境时间

---

*笔记创建时间: 2026-05-30*
