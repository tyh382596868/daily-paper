---
title: "World Model for Robot Learning: A Comprehensive Survey"
method_name: "RobotWM-Survey"
authors: [Bohan Hou, Gen Li, Jindou Jia, Tuo An, Xinying Guo, Sicong Leng, Haoran Geng, Yanjie Ze, Tatsuya Harada, Philip Torr, Oier Mees, Marc Pollefeys, Zhuang Liu, Jiajun Wu, Pieter Abbeel, Jitendra Malik, Yilun Du, Jianfei Yang]
year: 2026
venue: arXiv
tags: [survey, world-model, robot-learning, vla, video-generation, reinforcement-learning]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.00080v1
created: 2026-05-14
---

# 论文笔记：World Model for Robot Learning: A Comprehensive Survey

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | NTU MARS, Stanford, UC Berkeley, Oxford, MIT, ETH, Tokyo, etc. |
| 日期 | May 2026 |
| 项目主页 | https://ntumars.github.io/wm-robot-survey/ |
| 代码仓库 | https://github.com/NTUMARS/Awesome-World-Model-for-Robotics-Policy |
| 对比基线 | N/A（综述） |
| 链接 | [arXiv](https://arxiv.org/abs/2605.00080) / [HTML](https://arxiv.org/html/2605.00080v1) |

---

## 一句话总结

> 一篇以"策略-模拟器-视频生成"三视角组织的[[世界模型]]综述，用统一的联合概率分布框架，把 100+ 个机器人世界模型工作归类成 6 大策略架构 + 2 类模拟器用途 + 4 阶段视频生成演化。

---

## 核心贡献

1. **统一的概率框架**: 把[[策略]]、[[passive 世界模型|Passive World Model]]、[[Controllable 世界模型|Controllable World Model]]、[[Inverse Dynamics Model|逆动力学模型]] 视为同一联合分布 $p(o_{t+1:t+k}, a_{t+1:t+k} \mid o_t, l)$ 的不同边缘/条件，解释为何世界模型与策略天然耦合。
2. **三视角分类法**:
   - **Policy 视角**: 6 类架构（IDM 解耦 / 单 backbone 联合 / [[MoT|MoE-MoT]] 多专家 / 统一 [[VLA]] / 潜空间 / 符号），覆盖从 UniPi 到 [[DreamZero]]、[[Cosmos-Policy]]、[[FLARE]]、[[VLA-JEPA]] 等 30+ 方法
   - **Simulator 视角**: 世界模型作为 [[强化学习|RL]] 环境（[[UniSim]]、[[WMPO]]、[[WoVR]]）或策略评估器（[[GPC]]、[[IRASim]]、[[WorldEval]]）
   - **Video Generation 视角**: 想象 → 动作可控 → 结构感知 → 基础模型 四阶段演化
3. **基准与挑战梳理**: 把评估协议分为开环视觉质量、闭环任务效用、物理可执行性三类；指出 6 大开放挑战（因果条件、效率、多模态、经典控制、符号结构、评估指标）。

---

## 问题背景

### 要解决的问题

[[世界模型]] 在机器人学习中已经从早期的 [[DreamerV3]]/[[TD-MPC]] 风格的紧凑潜在动力学模型，发展为大规模视频生成模型（[[Cosmos-Policy]]、[[Vid2World]]、Genie-Envisioner）和与 [[VLA]] 深度耦合的统一架构。文献急速膨胀，分类标准混乱，缺乏一个能同时覆盖**策略架构、模拟器用途、视频生成范式**的统一视角。

### 现有综述的局限

- 通用视频生成综述忽略动作可控性与物理一致性
- [[强化学习]] 中的世界模型综述（DreamerV3 系）局限于紧凑潜空间，未涵盖近期视频驱动方法
- 缺少对"世界模型如何嵌入 [[VLA]] 架构"的系统性梳理

### 本文的动机

提出一个 **action-conditioned + downstream-useful** 的世界模型定义，并用一个统一的联合分布把策略、模拟器、视频生成三类用法串起来；既能容纳像素级生成方法，也能涵盖 [[JEPA]] 风格的潜空间方法。

---

## 方法详解

### 综述组织结构

本综述采用 **三视角 + 评估 + 挑战** 的结构：

- **Section 2 背景**: 形式化定义 [[世界模型]] / [[视频生成]] / [[Visuomotor Policy|视觉运动策略]] / [[VLA|VLA 策略]]
- **Section 3 Policy 视角**: 世界模型作为策略的内在组件（6 种架构范式）
- **Section 4 Simulator 视角**: 世界模型作为 [[强化学习|RL]] 环境与策略评估器
- **Section 5 Video Generation 视角**: 机器人视频生成从想象到 foundation model 的演化
- **Section 6 其他应用**: [[导航]] 与 [[自动驾驶]]
- **Section 7 基准/数据集/结果**
- **Section 8 挑战与未来方向**

### Section 2: 统一概率视角

综述把所有用法都归约到一个核心联合分布：

输入是当前观测 $o_t$、动作序列 $a_{t:t+H-1}$、高层目标 $l$（语言指令或目标图像）。
输出是某种"未来状态" $x_{t+1:t+H}$，可以是像素帧、潜变量、几何场或符号谓词。

四个相关的"投影"：
- **[[Visuomotor Policy|策略]]**: 边缘化掉观测，只保留动作分布
- **[[passive 世界模型|Passive WM]]**: 边缘化掉动作，得到无控制的环境演化
- **[[Controllable 世界模型|Controllable WM]]**: 以动作为条件，得到可控制的未来
- **[[Inverse Dynamics Model|IDM]]**: 给定观测序列反推动作

### Section 3: World Model for Policy（六种范式）

#### 3.2 IDM 风格解耦：predict-then-act

- **核心**: [[视频生成]] 模型 $\mathcal{W}$ 与 [[策略]] $\pi$ 架构上分离，先生成未来再回归动作
- **代表**: UniPi、VidMan、Vidar、Gen2Act、[[VPP]]、Video2Act、TC-IDM、LVP、Say-Dream-ACT
- **趋势**: 未来表示从原始像素 → 紧凑潜变量 → 几何轨迹 → 重定向计划

#### 3.3 单 backbone 联合建模

- **核心**: 单一 backbone $f_\theta$ 同时处理视觉与动作 token，$x = [z^v; z^a]$ 共享潜空间
- **代表**: [[UVA]]、UWA、VideoVLA、VideoPolicy、[[Cosmos-Policy]]、[[DreamZero]]、UD-VLA、GigaWorld-Policy
- **优势**: 利用视频预训练 backbone 的时空先验作为控制偏置

#### 3.4 [[MoT|MoE/MoT]] 风格多专家

- **核心**: 视频专家 + 动作专家通过 layer-wise 交叉注意力耦合，保留模态特化优化
- **代表**: GE-Act、Motus、LingBot-VA、BagelVLA、Fast-WAM、LDA-1B、FRAPPE、DiT4DiT
- **公式**: $(h^v_{\ell+1}, h^a_{\ell+1}) = F^{mix}_\ell(h^v_\ell, h^a_\ell; o_t, l)$

#### 3.5 统一 [[VLA]] 模型

把预测性结构内化进 VLA 架构，三个子类：

1. **显式未来预测**: [[GR-1]]、UP-VLA、[[WorldVLA]] — 联合预测动作与未来图像
2. **隐式/潜在未来建模**: [[DreamVLA]]、[[UniVLA]]、CoWVLA — 预测结构化世界知识或潜在视觉目标
3. **多专家统一**: F1、InternVLA-A1、HALO、TriVLA — 各子系统协同视觉前瞻

#### 3.6 潜空间世界建模

- **核心**: 世界建模完全在表征空间进行，不显式生成图像/视频
- **代表**: [[FLARE]]、[[VLA-JEPA]]、JEPA-VLA、WoG、DIAL
- **优势**: 规避生成解码的算力开销，但仍向动作生成注入预测结构

### Section 4: World Model as Simulator

#### 4.1 World Model for [[强化学习|RL]]

世界模型作为**学到的模拟器**替代真实交互：

- **第一层**: 把 WM 当固定环境 — [[UniSim]]、World-Env、VLA-RFT、DiWA、World4RL、World-Gymnast、PlayWorld、[[WMPO]]、RehearseVLA
- **第二层**: WM 与策略协同进化（co-evolution）— World-VLA-Loop、VLAW、[[WoVR]]，用策略 rollout 的失败案例反过来修正 WM

#### 4.2 World Model for Evaluation

四类用法：

| 用途 | 代表方法 |
|------|---------|
| Rollout-based 候选评估 | [[GPC]]、[[IRASim]]、World-in-World、DreamPlan |
| [[MPC|模型预测控制]] | [[TD-MPC2|TD-MPC2]]、[[LeWM|LeWorldModel]] |
| 策略评估 | Gemini Robotics 评估、[[WorldEval]]、WorldArena |
| 反馈增强模拟器 | World-Env、RISE（进度值模型）、[[CtrlWorld|Ctrl-World]] |

**关键洞察**: 评估器有效性取决于 action-fidelity；幻觉会污染评估信号。

### Section 5: World Model for Robotic Video Generation

四阶段演化：

1. **5.2 想象式**: UniPi、Video Language Planning、Dreamitate、ManipDreamer、DreMa、DreamGen — 任务条件，无显式动作输入
2. **5.3 动作可控**: [[IRASim]]、RoboMaster、[[CtrlWorld|Ctrl-World]]、EnerVerse-AC、Interactive World Simulator、EVA — 因果对齐动作与视觉
3. **5.4 结构感知**: Mask2IV、TesserAct、RoboVIP — 引入物体中心表征 + 几何先验
4. **5.5 基础世界模型**: [[Vid2World]]、Genie Envisioner、DreamDojo、Cosmos Predict 2.5、GigaWorld-0、WoW、ABot-PhysWorld — 基于 [[CogVideoX]] 等大规模 backbone

### Section 6: 其他应用

- **6.1 导航**: 通过预测自我视角的视觉未来支持具身导航
- **6.2 自动驾驶**: 多智能体轨迹预测与场景理解

### Section 7: 基准、数据集、结果

#### 7.1 评估基准三层划分

1. **开环预测质量**: 视觉保真度、时序一致性、action-consistency
2. **闭环任务效用**: 用 WM 训练或评估策略的实际任务成功率
3. **物理可执行性诊断**: 物理合理性、动作跟随精度、因果正确性

#### 7.2 训练数据集
开放机器人数据集 + 大规模视频预训练语料（综述中按需引用）。

#### 7.3 代表结果
基础模型规模的世界模型在跨任务泛化上明显优于任务特定模型。

### Section 8: 挑战与未来方向

| 维度 | 核心问题 |
|------|---------|
| 8.1 因果条件 | 预测未来未必反映动作的真实因果后果 |
| 8.2 效率瓶颈 | 生成式预测推理成本高，需要潜空间/截断 rollout |
| 8.3 多模态感知 | 多数 WM 只用 RGB，缺少本体感知、触觉、多视角融合 |
| 8.4 经典控制集成 | 学习式 WM 与 [[MPC]]/LQR 接口不顺 |
| 8.5 符号结构 | 神经预测与符号关系（物体、affordance）结合困难 |
| 8.6 评估指标 | 缺少统一的 action-faithfulness、物理合理性、下游任务效用度量 |

---

## 关键公式

### 公式1: [[世界模型|核心联合预测-控制分布]]

$$
p(x_{t+1:t+H} \mid x_t,\ a_{t:t+H-1},\ l)
$$

**含义**: 综述对"世界模型"的最广定义 — 给定当前状态、动作序列、语言指令，预测未来状态分布。这个状态 $x$ 可以实例化为像素帧、潜变量、几何场或符号谓词。

**符号说明**:
- $x_t$: 时刻 $t$ 的状态（实例化方式决定子类型）
- $a_{t:t+H-1}$: 长度 $H$ 的动作序列
- $l$: 高层目标（语言指令 / 目标图像）

### 公式2: [[视频生成|视频生成模型]]（WM 在像素空间的实例化）

$$
p(v_{t+1:t+H} \mid o_t,\ a_{t:t+H-1},\ l)
$$

**含义**: 视频生成模型是世界模型在视觉观测空间的实例化。

**符号说明**:
- $o_t$: 当前观测（通常为 RGB 图像）
- $v_{t+1:t+H}$: 未来视频帧序列

### 公式3: [[Visuomotor Policy|视觉运动策略]]

$$
p(a_{t+1:t+k} \mid o_t,\ l)
$$

**含义**: 现代视觉运动策略用生成模型（扩散、flow matching）建模完整动作分布。

### 公式4: 联合分布的四个边缘/条件

策略：
$$
p(a_{t+1:t+k} \mid o_t,l) = \int p(o_{t+1:t+k},\ a_{t+1:t+k} \mid o_t,l)\, do
$$

[[passive 世界模型|Passive WM]]：
$$
p(o_{t+1:t+k} \mid o_t,l) = \int p(o_{t+1:t+k},\ a_{t+1:t+k} \mid o_t,l)\, da
$$

[[Controllable 世界模型|Controllable WM]]：
$$
p(o_{t+1:t+k} \mid o_t,\ a_{t+1:t+k})
$$

[[Inverse Dynamics Model|IDM]]：
$$
p(a_{t+1:t+k} \mid o_{t:t+k})
$$

**含义**: 综述的核心理论贡献 — 这四个范式只是同一联合分布的不同投影，因此策略与世界模型可以自然耦合。

### 公式5: IDM 风格 predict-then-act

$$
\hat{o}_{t+1:t+H} = \mathcal{W}(o_t,\ l), \quad \pi(a \mid o_t,\ l,\ \Phi(\hat{o}_{t+1:t+H}))
$$

**含义**: 解耦架构 — 世界模型 $\mathcal{W}$ 先生成未来，策略 $\pi$ 通过特征提取器 $\Phi$ 利用未来。

**符号说明**:
- $\Phi$: 从生成未来中抽取动作相关特征（可以是 IDM、潜变量提取器、轨迹回归器）

### 公式6: 单 backbone 联合生成

$$
\hat{y} = f_\theta(\tilde{x}_\tau,\ o_t,\ l,\ \tau), \quad x = [z^v;\ z^a]
$$

**含义**: 单一 backbone $f_\theta$ 在共享潜空间 $[z^v; z^a]$ 内联合去噪视频 token 与动作 token。

**符号说明**:
- $\tilde{x}_\tau$: 噪声等级 $\tau$ 下的联合潜变量
- $z^v, z^a$: 视频与动作的潜表示

### 公式7: [[MoT|MoE/MoT 多专家融合]]

$$
(h^v_{\ell+1},\ h^a_{\ell+1}) = F^{mix}_\ell(h^v_\ell,\ h^a_\ell;\ o_t,\ l)
$$

**含义**: 视频专家与动作专家保持参数解耦，通过 layer-wise 交叉注意力 $F^{mix}_\ell$ 双向交换信息。

### 公式8: WM 作为 [[强化学习|RL]] 模拟器

转移采样：
$$
(\hat{o}_{t+1},\ \hat{r}_t,\ \hat{d}_t) \sim p_\phi(\cdot \mid o_{\le t},\ a_{\le t},\ l)
$$

策略目标：
$$
J(\theta) = \mathbb{E}_{\hat{\tau} \sim (\pi_\theta,\ p_\phi)} \left[ \sum_t \gamma^t \hat{r}_t \right]
$$

**含义**: 世界模型 $p_\phi$ 同时预测下一观测、奖励与终止信号；策略在想象 rollout 上做梯度优化。

**符号说明**:
- $\hat{r}_t$: WM 预测的奖励
- $\hat{d}_t$: WM 预测的终止信号
- $\gamma$: 折扣因子

### 公式9: World-Policy 协同进化

$$
\phi^{k+1} \leftarrow \text{UpdateWM}(\phi^k,\ \mathcal{D}_{\text{real}} \cup \mathcal{D}_{\text{policy}}(\pi_{\theta^k}))
$$

$$
\theta^{k+1} \leftarrow \text{UpdatePolicy}(\theta^k,\ \hat{\mathcal{D}}(\phi^{k+1}))
$$

**含义**: WoVR 等方法把 WM 当作可演化对象 — 用策略 rollout 的失败案例反过来修补 WM，再用更新后的 WM 改进策略，形成闭环。

---

## 关键图表

### Figure 1: 综述组织总览

![Figure 1](https://arxiv.org/html/2605.00080v1/x2.png)

**说明**: 综述的整体组织结构图。Section 3 从架构视角剖析世界模型与机器人[[策略]]的耦合方式，Section 4 把 WM 视作[[强化学习|RL]]模拟器或评估器，Section 5 聚焦机器人[[视频生成]]的能力演化。这张图把"Policy / Simulator / Video Generation"三视角串成一个统一的叙事。

### Figure 2: 代表工作的时间演化

![Figure 2](https://arxiv.org/html/2605.00080v1/x3.png)

**说明**: 机器人世界模型方法的时间轴。从早期解耦的 video+IDM 流水线（UniPi、VidMan 等）逐步演化到单 backbone 联合方法（[[Cosmos-Policy]]、[[DreamZero]]）和基于模拟器的方法（[[WMPO]]、[[WoVR]]）。展现了"分离 → 统一 → 协同进化"的清晰技术脉络。

### Figure 3: 策略 - 世界模型耦合的三种架构范式

![Figure 3](https://arxiv.org/html/2605.00080v1/x4.png)

**说明**: 三种代表性架构对比 —
- **(a) IDM 风格**: 视频生成与策略架构分离，predict-then-act
- **(b) 单 backbone 统一**: 视频与动作 token 在共享潜空间联合去噪
- **(c) MoT 风格**: 视频专家与动作专家通过 layer-wise 交叉注意力深度交互

### Figure 4: 基于 MLLM 的两条世界建模路线

![Figure 4](https://arxiv.org/html/2605.00080v1/x5.png)

**说明**: 把世界建模内化到 [[VLA]] 架构的两种路径 —
- **(a) 统一 VLA + 辅助输出**: 主分支生成动作，辅助分支输出未来图像/视频作为正则项（[[GR-1]]、UP-VLA、[[WorldVLA]]）
- **(b) MLLM backbone 内的潜空间世界建模**: 不解码像素，直接在 LLM 特征空间做未来预测（[[DreamVLA]]、CoWVLA、DIAL）

### Figure 5: 世界模型的两类模拟器用途

![Figure 5](https://arxiv.org/html/2605.00080v1/x6.png)

**说明**: 模拟器视角下 WM 的两类用法 —
- **(a) RL 学习模拟器**: 在想象转移上做策略优化（[[UniSim]]、[[WMPO]]、World-Env）
- **(b) 决策时评估器**: 对候选动作做 rollout 评估和验证（[[GPC]]、[[IRASim]]、World-in-World）

### Figure 6: 机器人视频世界模型的统一视图

![Figure 6](https://arxiv.org/html/2605.00080v1/x7.png)

**说明**: 视频世界模型从"想象式生成"到"动作可控"再到"结构感知"四阶段演化的统一示意。展示了每阶段的代表方法、关键技术贡献，以及不同阶段在 plan/data/sim/eval/sup 五种主要用途上的覆盖。

### Table 1: Policy 视角下的架构对比

| 范式 | 代表工作 | 推理时是否生成未来 | Backbone | 耦合方式 |
|------|---------|------|---------|---------|
| IDM 风格解耦 | UniPi, VidMan, [[VPP]], Gen2Act | 是 | 视频扩散 + IDM | 串联 |
| 单 backbone 统一 | [[UVA]], [[Cosmos-Policy]], [[DreamZero]] | 可选 | 视频扩散 backbone | 共享潜空间 |
| [[MoT|MoE/MoT]] 多专家 | GE-Act, Motus, BagelVLA | 是 | 双专家 | 交叉注意力 |
| 统一 [[VLA]] | [[GR-1]], [[DreamVLA]], [[WorldVLA]] | 辅助输出 | MLLM | 联合训练 |
| 潜空间 WM | [[FLARE]], [[VLA-JEPA]], WoG | 否（潜空间） | MLLM/JEPA | 表征对齐 |

**说明**: 综述对 Section 3 中 5 大类方法的横向对比。最重要的两个区分维度是：**推理时是否需要解码视频**（影响延迟）、**视频与动作通路如何耦合**（影响泛化与训练效率）。

### Table 2: Section 5 视频生成方法的能力阶段对比

| 能力维度 | Imagination-Based | Action-Controllable | Structure-Aware | Foundation WM |
|---------|------|------|------|------|
| Task-conditioned | ✓ | ✓ | ✓ | ✓ |
| Action-conditioned | ✗ | ✓ | ✓ | ✓ |
| Structure-aware | ✗ | ✗ | ✓ | 部分 |
| Foundation-scale | ✗ | ✗ | ✗ | ✓ |
| 主用途 | Plan/Data | Sim/Eval | Sup/Eval | All |
| 代表 | UniPi, DreamGen | [[IRASim]], [[CtrlWorld\|Ctrl-World]] | TesserAct, RoboVIP | [[Vid2World]], Cosmos Predict 2.5 |

**说明**: 视频世界模型从"只能想象任务结果"逐步加上动作因果性、几何结构、再扩展到 foundation 规模的演化。foundation WM 整合了前三阶段所有能力。

---

## 实验结果

> 综述本身不做实验，但 Section 7.3 汇总了代表方法在常见基准上的相对表现，关键观察：

- **跨任务泛化**: foundation-scale WM（[[Cosmos-Policy]]、[[Vid2World]]）显著优于任务特定 WM
- **动作可控性**: action-conditional 模型（[[IRASim]]、[[CtrlWorld|Ctrl-World]]）在闭环评估上明显优于纯 task-conditioned
- **协同进化**: World-VLA-Loop、[[WoVR]] 等 co-evolution 方法在长 horizon 任务上明显好于固定 WM

---

## 批判性思考

### 优点

1. **理论框架统一**: 用一个联合分布把 4 种 paradigm 统起来，比之前的综述更清晰
2. **覆盖广度**: 100+ 方法，涵盖 [[DreamerV3]] 紧凑潜空间一直到最新 [[Cosmos-Policy]]/[[DreamZero]] 大规模视频生成
3. **三视角划分独到**: Policy / Simulator / Video Generation 三视角避免了"每个方法只能放一个框里"的尴尬
4. **挑战章节务实**: 因果条件、效率、多模态、经典控制集成 — 都是当前的真实痛点

### 局限性

1. **数据集章节较薄**: Section 7.2 对训练数据集只是简要带过，缺少类似 Open X-Embodiment、AgiBot 等关键大规模数据的覆盖
2. **物理仿真器 vs 学习 WM 的边界讨论不足**: 没有充分讨论 MuJoCo/Isaac 这类基于物理的仿真器何时还有不可替代的价值
3. **缺乏统一的量化对比**: Table 形式的对比偏定性，没有给出统一基准下的硬性数字
4. **作者列表潜在偏向**: 作者集中在 NTU MARS + 几个顶尖 AI lab，[[强化学习]]社区视角（如 Dreamer 系）可能稍弱于 video diffusion 视角

### 潜在改进方向

1. **加入 latency / FLOPs 这种工程维度**: 推理时是否生成视频对部署影响巨大，应该有专门章节
2. **协同进化 vs Fine-tuning 的边界更精细划分**: WoVR、World-VLA-Loop 这些方法究竟和持续学习/在线 fine-tune 是什么关系？
3. **失败模式分类学**: 哪些任务类型上 WM 系统性地失败？（例如长链推理、多智能体、罕见物理事件）

### 可复现性评估

- [x] 代码索引开源（[Awesome 仓库](https://github.com/NTUMARS/Awesome-World-Model-for-Robotics-Policy)）
- [x] 项目主页
- [-] 综述本身不需要训练复现
- [x] 引用文献完整

---

## 关联笔记

### 基于

- [[世界模型]]: 综述核心主题
- [[VLA]]: 大量方法都是 VLA 的世界模型扩展

### 综述涵盖的代表工作（已有笔记）

#### Policy 视角
- [[Cosmos-Policy]]: 单 backbone 联合扩散
- [[WorldVLA]]: 统一 VLA + 动作-图像生成
- [[UniVLA]]: post-training 阶段做世界建模
- [[FLARE]]: 潜空间世界建模代表

#### Simulator 视角
- [[CtrlWorld|Ctrl-World]]: action-faithful rollout
- [[IRASim]]: 动作条件视频模拟器
- [[LeWM|LeWorldModel]]: 端到端 JEPA WM
- [[DINO-WM]]: foundation-based 潜空间 WM
- [[PLDM]]: LeWM 之前的 JEPA 基线
- [[RLA-WM]]: 残差潜在动作空间的 WM
- [[EA-WM]]: KVAF 几何对齐的视频 WM
- [[OA-WAM]]: object-addressable 注意力

#### 视频生成 backbone
- [[Vid2World]]: 视频 backbone 改造为 WM
- [[CogVideoX]]: 多个 WM 的预训练 backbone
- [[Wan2.2-TI2V]]: 视频扩散基线
- [[视频扩散模型]]: 通用底层范式

#### RL 视角
- [[DreamerV3]]: 紧凑潜空间 WM 代表
- [[TD-MPC]]: latent-space MPC

#### 概念基础
- [[扩散变换器]]: 多数现代 WM 的 backbone
- [[Flow Matching]]: 联合去噪常用工具
- [[JEPA]]: 潜空间预测的理论基础

### 对比阅读建议

如果你之前读过 [[EA-WM]]，这篇综述把 EA-WM 放在 **5.4 结构感知视频生成** 这一阶段；如果你关心 [[强化学习]] 中的 WM，重点看 4.1 节中 DreamerV3 → [[WMPO]] → [[WoVR]] 这条线。

---

## 速查卡片

> [!summary] World Model for Robot Learning: A Comprehensive Survey
> - **核心**: 用统一联合分布串起 Policy / Simulator / Video Generation 三视角的 100+ 个机器人世界模型工作
> - **方法分类**: 6 种 Policy 架构（IDM/单 backbone/MoT/统一 VLA/潜空间/符号）+ 2 类模拟器用途（RL/评估）+ 4 阶段视频生成演化
> - **关键贡献**: 把策略、passive WM、controllable WM、IDM 视为同一联合分布的不同投影
> - **代码**: [Awesome 仓库](https://github.com/NTUMARS/Awesome-World-Model-for-Robotics-Policy) / [项目主页](https://ntumars.github.io/wm-robot-survey/)

---

*笔记创建时间: 2026-05-14*
