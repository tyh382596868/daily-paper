---
title: "RoboMemArena: A Comprehensive and Challenging Robotic Memory Benchmark"
method_name: "RoboMemArena"
authors: [Huashuo Lei, Wenxuan Song, Huarui Zhang, Jieyuan Pei, Jiayi Chen, Haodong Yan, Han Zhao, Pengxiang Ding, Zhipeng Zhang, Lida Huang, Donglin Wang, Yan Wang, Haoang Li]
year: 2026
venue: arXiv
tags: [vla-memory, robot-manipulation, long-horizon, benchmark, dual-system, predictive-coding, keyframe]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.10921
created: 2026-05-14
---

# 论文笔记：RoboMemArena: A Comprehensive and Challenging Robotic Memory Benchmark

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | HKUST (GZ), Westlake University, ZJU 等多家联合 |
| 日期 | May 2026 |
| 项目主页 | https://robomemarena.github.io |
| 对比基线 | [[Pi05]] / [[MemER]] / [[MemoryVLA]] / [[HiF-VLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.10921) / [HTML](https://arxiv.org/html/2605.10921) |

---

## 一句话总结

> 提出 26 任务、平均 1076 步、68.9% 子任务依赖记忆的机器人长程操作 benchmark [[RoboMemArena]]，并配套一个双系统 [[VLA]] 方法 [[PrediMem]]，用 [[关键帧记忆库|Keyframe Memory Bank]] + [[预测编码头|Predictive Coding Head]] 把平均 TSR 从 27.3% 提升到 38.5%。

---

## 核心贡献

1. **RoboMemArena Benchmark**: 26 个长程操作任务 + 5 个真机任务，平均轨迹 >1000 步，68.9% 子任务需要记忆，覆盖 [[多目标转移]] / [[多目标遮挡]] / [[多目标计数]] / [[多目标顺序]] 四类记忆需求，是首个同时具备「长程 + 自动指令生成 + 原生 [[关键帧记忆库|关键帧]] 标注 + 真机配对」的记忆 benchmark。
2. **三阶段自动化数据生成流水线**: 用 [[VLM]] 驱动任务分解 → [[AnyGrasp]] 自主执行 → 多条件关键帧抽取（夹爪状态变化 ∪ 速度拐点），可规模化扩展到任意新任务。
3. **PrediMem 方法**: 把 [[Pi05|π₀.₅]] 升级为双系统结构 ([[双系统架构|S1/S2]])，S2 用 [[Qwen3-VL]] 做高层规划 + 关键帧选择 + 子任务分发，S1 做低频高频异步执行；引入 [[预测编码头|Predictive Coding Head]] 作为训练阶段辅助目标，推理时零开销。
4. **新 SOTA**: PrediMem 在 RoboMemArena 仿真上 TSR 38.5%（vs MemER 27.3%），真机上 52%（vs MemER 40%）。

---

## 问题背景

### 要解决的问题

现有机器人 benchmark 普遍是「反应式」的：轨迹短（多 <500 步）、子任务彼此独立、靠当前观测即可决策。但真实家庭/工业场景充斥「打开柜子放进去 → 关上柜子 → 等会儿再回来取」这类依赖**长期记忆**的长程任务。同时缺乏：
- 原生的 [[关键帧记忆库|keyframe]] 标注（多数是事后用启发式补的）
- 真机配对（多数纯仿真）
- 可规模化的自动生成流水线（多数靠人写脚本）

### 现有方法的局限

- [[Pi05|π₀.₅]] 等反应式 [[VLA]]：只看当前帧，遇到「记得刚才放哪了」就崩。
- [[MemoryVLA]]：用 token-level working memory，但容量有限、缺少显式关键帧抽取。
- [[MemER]]：双系统 + 关键帧检索是目前最强 baseline，但关键帧的选择依靠启发式，且缺少促进表征「对状态切换敏感」的训练信号。
- [[HiF-VLA]]：建模 hindsight / insight / foresight 运动，但不针对显式记忆任务。

### 本文的动机

记忆类任务的难点不在「存什么」，而在「**什么时候该记**」——关键帧检测应该和「物理状态切换」（夹爪开闭、速度归零）耦合。作者由此提出：
1. 用「物理交互锚点 + 运动学拐点」自动抽关键帧，比启发式更稳。
2. 加一个 [[预测编码头|Predictive Coding Head]] 作辅助任务，让表征对「下一帧会变什么」更敏感，从而帮助 S2 更准地判断是否要把当前帧入库。

---

## 方法详解

### Benchmark 设计：[[RoboMemArena]]

四类任务（共 26 个仿真 + 5 真机），按对记忆的需求方式划分：

- **[[多目标转移]] (Multi-Object Transferring, 4 个)**: 在视觉上无法区分的多个容器间搬运物体，必须记住「哪个容器里有什么」。
- **[[多目标遮挡]] (Multi-Object Occlusion, 11 个)**: 物体被放进抽屉/盒子等遮挡空间，要追踪隐藏状态。
- **[[多目标计数]] (Multi-Object Counting, 7 个)**: 「连续按按钮 N 次」「往锅里放 3 块肉」之类，需要计数。
- **[[多目标顺序]] (Multi-Object Sequence, 4 个)**: 顺序依赖、交叉引用，例如「先按红色再按按红色对面的蓝色」。

#### 关键统计

- 26 任务 × 100 成功演示/任务 = 2600 条长程轨迹
- 平均轨迹长度：**1076 步** / 任务
- 记忆依赖子任务比例：**68.9%** (104/151)
- 关键帧对齐短段：15,100 个，用于层次化监督
- 验证步骤：每任务 3–9 个 [[验证谓词|verification predicates]]

#### 数据生成流水线（三阶段，全自动）

1. **VLM 任务分解**: 用 [[VLM]] 把高层指令拆成原子子任务序列。
2. **[[AnyGrasp]] 自主执行**: 用抓取姿态估计 + 闭环重试自主完成。
3. **多条件关键帧抽取**: 物理 + 运动学双重条件（见公式 1–3）。

#### 评估指标

- **[[任务成功率|Task Success Rate (TSR)]]**: 所有验证谓词全过才算成功（公式 4）。
- **[[累积成功率|Cumulative Success Rate (CSR)]]**: 完成的验证阶段比例（公式 5），更细粒度。

### PrediMem 架构

[[PrediMem]] 采用 **[[双系统架构|双系统 (S1/S2)]]** 异步耦合架构：

- **输入**: 语言指令 $\ell$ + 当前观测 $o_t$ + 记忆库 $\mathcal{M}_t$
- **S2 (高层规划器)**: [[Qwen3-VL]]-8B-Instruct，1.06 Hz 慢速运行，负责
  - 预测当前应执行的**子任务** $c_t$
  - 决策当前帧是否入 [[关键帧记忆库|keyframe buffer]]
  - 维护**最近窗口** (recent buffer) 与**关键帧库** (keyframe buffer)
- **S1 (低层动作策略)**: [[Pi05|π₀.₅]] 风格的 [[VLA]]，3.40 Hz 高频运行，每条 S2 输出覆盖约 2.92 个 S1 [[Action Chunking|动作块]]
- **预测编码头**: 训练阶段附加，预测下一帧视觉表征，推理时移除

每个 S2 更新对应约 2.92 个 S1 动作块，保证 S1 永远在动、S2 不卡 S1。

### 核心模块

#### 模块 1: [[关键帧记忆库|Memory Bank]]

**设计动机**: 长程任务里大部分帧是冗余的，记下「物理状态发生切换」的少数帧足以重建语义历史。

**两部分组成**:
- **最近缓冲 $\mathcal{M}^{\text{rec}}$**: 5 帧滑动窗口，提供当前局部上下文。
- **关键帧缓冲 $\mathcal{M}^{\text{key}}$**: 无上限，存储由 S2 判定为「决策关键」的所有历史帧。

**写入策略**: 关键帧 = 物理交互锚点 ∪ 运动学拐点（公式 1–3），夹爪开闭与速度拐点都入库。

#### 模块 2: [[预测编码头|Predictive Coding Head]]

**设计动机**: 让视觉表征「对物理状态切换敏感」，从而帮助 S2 准确选择关键帧。借鉴 [[JEPA]] / [[V-JEPA]] 系列的非生成式预测思想。

**具体实现**:
- 用一个轻量 head 预测下一帧的视觉表征 $\hat{Z}_{t+1}$
- 目标是冻结的 [[ViT]] 特征 $Z_{t+1}$（用 `stop_gradient`）
- 损失同时使用 MSE 和余弦距离（公式 6）
- 权重 0.1（公式 6 上下文：$\mathcal{L}_{S2} = \mathcal{L}_{\text{text}} + 0.1\,\mathcal{L}_{\text{Pre}}$）
- **推理阶段移除**，不增加任何计算开销

#### 模块 3: 异步推理协议（Algorithm 1）

```
Input: 指令 ℓ, 初始观测 o₀
Init:  M₀ = ∅, recent_buffer = []
For t = 1..T:
  a_t ← π_S1(o_t, g)          # S1 高频出动作
  若 S2 idle: 异步触发 S2(o_t, M_t, ℓ)
  当 S2 返回 (g', k_τ):
    更新当前子任务 g ← g'
    若 k_τ = 1: M_t^key ← M_t^key ∪ {o_τ}
  更新 recent buffer
```

---

## 关键公式

### 公式 1: [[关键帧记忆库|关键帧集合定义]]

$$
\mathcal{K} = \mathcal{K}_{\text{phys}} \cup \mathcal{K}_{\text{kin}}
$$

**含义**: 关键帧由两类物理/运动学条件的并集决定，捕捉所有「状态切换」时刻。

**符号说明**:
- $\mathcal{K}_{\text{phys}}$: 物理交互锚点集合（公式 2）
- $\mathcal{K}_{\text{kin}}$: 运动学拐点集合（公式 3）

### 公式 2: 物理交互锚点

$$
\mathcal{K}_{\text{phys}} = \{ t \in [1, T] \mid g_t \neq g_{t-1} \}
$$

**含义**: 凡是夹爪状态发生变化的时刻都入库（抓 / 放都是关键时刻）。

**符号说明**:
- $g_t \in \{0, 1\}$: 时刻 $t$ 的夹爪状态（0 开 / 1 闭）
- $T$: 轨迹长度

### 公式 3: 运动学拐点

$$
\mathcal{K}_{\text{kin}} = \left\{ t \in [1, T] \;\middle|\; \|\mathbf{v}_t\| < \epsilon \;\lor\; \frac{\mathbf{v}_t \cdot \mathbf{v}_{t-1}}{\|\mathbf{v}_t\| \, \|\mathbf{v}_{t-1}\|} < \cos\theta \right\}
$$

**含义**: 速度接近零（停下来）或速度方向突变（拐弯）都视为关键帧。

**符号说明**:
- $\mathbf{v}_t$: 末端执行器速度
- $\epsilon$: 速度阈值（接近静止）
- $\theta$: 方向变化角度阈值

### 公式 4: [[任务成功率|Task Success Rate]]

$$
\text{TSR} = \frac{1}{N} \sum_{i=1}^{N} \prod_{k=1}^{K_i} \mathbf{1}\!\left[ \psi(s_i^{(k)}) \right]
$$

**含义**: 一整条轨迹必须**所有**验证谓词都通过才算成功，是「全有或全无」的严格指标。

**符号说明**:
- $N$: 评估轨迹数
- $K_i$: 第 $i$ 条轨迹的验证阶段数（3–9）
- $\psi(s_i^{(k)})$: 第 $k$ 阶段的验证谓词
- $\mathbf{1}[\cdot]$: 指示函数

### 公式 5: [[累积成功率|Cumulative Success Rate]]

$$
\text{CSR} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{K_i} \sum_{k=1}^{K_i} \mathbf{1}\!\left[ \psi(s_i^{(k)}) \right]
$$

**含义**: 衡量「平均完成了多少比例的验证阶段」，比 TSR 更细粒度，能体现「差一步成功」的进展。

**符号说明**: 同公式 4。

### 公式 6: [[预测编码头|预测编码损失]]

$$
\mathcal{L}_{\text{Pre}} = \mathrm{MSE}\!\left(\hat{Z}_{t+1},\, \mathrm{sg}(Z_{t+1})\right) + \left(1 - \cos\!\left(\hat{Z}_{t+1},\, \mathrm{sg}(Z_{t+1})\right)\right)
$$

**含义**: 训练阶段让模型预测下一帧的视觉表征，同时使用 MSE 和余弦距离两个项，迫使表征对物理状态切换敏感。

**符号说明**:
- $\hat{Z}_{t+1}$: 模型预测的下一帧表征
- $Z_{t+1}$: 真实的下一帧 [[ViT]] 特征（来自冻结的 vision tower）
- $\mathrm{sg}(\cdot)$: stop-gradient，防止预测目标反向传播

### 公式 7: S1 动作策略

$$
a_t = \pi_{S1}(o_t, c_t)
$$

**含义**: 低层动作策略以当前观测和最新子任务为输入，输出动作。

**符号说明**:
- $\pi_{S1}$: S1 [[VLA]] 策略
- $o_t$: 当前观测
- $c_t$: S2 最新下发的子任务

### 公式 8: S2 总训练损失

$$
\mathcal{L}_{S2} = \mathcal{L}_{\text{text}} + 0.1 \, \mathcal{L}_{\text{Pre}}
$$

**含义**: S2 的优化目标 = 指令调优文本损失 + 0.1 × 预测编码损失。权重 0.1 是经过 ablation 选出的最优值（见 Table 3）。

**符号说明**:
- $\mathcal{L}_{\text{text}}$: 自回归文本损失（标准 VLM 指令微调）
- $\mathcal{L}_{\text{Pre}}$: 预测编码损失（公式 6）

---

## 关键图表

### Figure 1: 四类任务的可视化

![Figure 1](https://arxiv.org/html/2605.10921v1/x5.png)

**说明**: 四类任务（[[多目标计数]] / [[多目标遮挡]] / [[多目标顺序]] / [[多目标转移]]）的样例。每行展示指令、子任务分解和执行 rollout。可以看到所有任务都涉及多个视觉相似物体、长程子任务串联与显式记忆需求。

### Figure 2: Benchmark 统计摘要

![Figure 2](https://arxiv.org/html/2605.10921v1/x6.png)

**说明**: 三个面板——(a) 平均轨迹长度 vs. 其他 benchmark，RoboMemArena 显著最长；(b) 任务组成饼图；(c) 记忆依赖子任务比例对比，RoboMemArena 高达 68.9%，远超 [[LIBERO]]、[[RoboCasa]] 等。

### Figure 3: PrediMem 整体流水线

![Figure 3](https://arxiv.org/html/2605.10921v1/x7.png)

**说明**: [[PrediMem]] 双系统流水线。S1（低层动作策略）执行当前子任务；S2（高层规划器）异步预测关键帧与下一个子任务；[[预测编码头|Predictive Coding Head]] 路径仅在训练阶段激活（虚线框）。最近缓冲与关键帧缓冲共同构成 [[关键帧记忆库|Memory Bank]]。

### Figure 4: 记忆行为分析

![Figure 4](https://arxiv.org/html/2605.10921v1/x8.png)

**说明**: (a) (b) CSR 对最近缓冲尺寸 / 关键帧库容量的敏感性曲线，最近缓冲 3–5 帧最优、关键帧库 uncapped 最优；(c) [[t-SNE]] 可视化显示加了 [[预测编码头|Predictive Coding]] 后关键帧表征聚成更紧凑、更可区分的簇。

### Figure S2: 真机任务演示

![Figure S2](https://arxiv.org/html/2605.10921v1/x9.png)

**说明**: 五个真机任务的代表性快照——Pour×2、Brush、Transfer、Shell、IHMB（双臂平台）。

### Figure S3: 每任务验证步骤分布

> 🖼️ **Figure S3** — 图片暂缺，arXiv 抓取失败（原图见 [arXiv HTML](https://arxiv.org/html/2605.10921)）

**说明**: (a) 每任务包含 3–9 个验证步骤，大多数任务超过 5 步的长程阈值；(b) 总体直方图。

### Table 1: Benchmark 对比（8 维特征 × 15 个 benchmark）

| Benchmark | Long Horizon | Auto Instr. | Atomic Subgoals | Scalable Gen. | Auto Grasp | State Oracle | Native Keyframes | Real-World |
|---|---|---|---|---|---|---|---|---|
| RLBench | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| RoboCerebra | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ |
| ARNOLD | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| ALFRED | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ |
| CALVIN | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| [[RoboCasa]] | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✓ |
| [[LIBERO]]-Long | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| VLABench | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ |
| RoboTwin | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ |
| RMBench | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| RoboMME | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| BEHAVIOR-1K | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ | ✓ |
| MIKASA | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ | ✓ |
| MemoryBench | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ | ✓ |
| **RoboMemArena** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**说明**: RoboMemArena 是表中**唯一**在所有 8 个维度都打勾的 benchmark，尤其是「自主抓取生成」(Autonomous Grasp) 这一列只有它有，说明其数据流水线最自动化。

### Table 2: 仿真主结果与消融（TSR / CSR %）

| Method | Transferring | Occlusion | Counting | Sequence | Avg TSR | Avg CSR |
|---|---|---|---|---|---|---|
| [[Pi05\|π₀.₅]] (reactive) | 20.0 / — | 12.7 / — | 14.3 / — | 60.0 / — | 21.5 | — |
| [[HiF-VLA]] | 17.5 | 12.7 | 8.6 | 42.5 | 16.9 | — |
| [[MemoryVLA]] | 15.0 | 7.3 | 14.3 | 37.5 | 15.0 | — |
| [[MemER]] | 20.0 | 16.4 | 27.1 | 65.0 | 27.3 | — |
| **PrediMem (Ours)** | **22.5** | **27.3** | **45.7** | **72.5** | **38.5** | — |
| — w/o Predictive Coding | 25.0 / 43.7 | 19.5 / 30.2 | 38.6 / 61.8 | 63.8 / 80.7 | 32.3 | 49.0 |
| — w/o Keyframe Bank | 17.5 / 33.3 | 6.4 / 22.8 | 20.0 / 61.7 | 45.0 / 66.3 | 17.7 | 41.6 |

**说明**: PrediMem 在所有四类任务上都领先，**记忆最敏感的 Counting 类提升最大**（45.7 vs 27.1）。消融显示**关键帧库的贡献远大于预测编码头**：去掉关键帧库直接掉 20.8 点，而去掉预测编码头只掉 6.2 点。

### Table 3: 预测编码损失权重消融

| $\lambda_{\text{Pre}}$ | 0.0 | **0.1** | 0.5 | 1.0 |
|---|---|---|---|---|
| TSR (%) | 32.3 | **38.5** | 31.0 | 29.8 |

**关键发现**: 权重 0.1 是「金发女郎」最佳值；过大反而损害文本损失，过小起不到作用。

### Table 4: 真机评估（10 rollouts/task）

| Method | Pour×2 | Brush | Transfer | Shell | IHMB | Avg |
|---|---|---|---|---|---|---|
| [[Pi05\|π₀.₅]] | 20% | 10% | 60% | 10% | 0% | 20% |
| [[MemER]] | 30% | 50% | 80% | 40% | 0% | 40% |
| **PrediMem** | **60%** | **60%** | **80%** | **50%** | **10%** | **52%** |

**说明**: 真机上相对优势比仿真还大（+12 点 vs +11 点仿真），说明记忆机制在真实噪声环境下更重要。注意 IHMB 任务所有方法都接近 0%，说明仍有显著挑战空间。

---

## 实验结果

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[RoboMemArena]] (sim) | 26 任务 × 100 轨迹 = 2600 条 | 平均 1076 步，68.9% 记忆依赖 | 训练 + 测试 |
| RoboMemArena 关键帧段 | 15,100 段 | 关键帧对齐短轨迹 | 层次化监督 |
| RoboMemArena (real) | 5 任务 × 10 rollouts | 双臂平台 | 真机测试 |

### 实现细节

- **Backbone**: [[Qwen3-VL]]-8B-Instruct（vision tower 冻结）
- **优化器**: 2 epochs fine-tune，learning rate $1 \times 10^{-5}$
- **硬件**: 4 × H100
- **损失权重**: $\mathcal{L}_{\text{Pre}}$ 权重 0.1（MSE + 余弦各 0.1）
- **记忆配置**: recent buffer 5 帧，keyframe buffer 无上限
- **S2/S1 频率**: S2 = 1.06 Hz，S1 = 3.40 Hz（约 2.92:1）
- **S1 基模型**: [[Pi05|π₀.₅]] 系列 [[VLA]]

### Scaling 研究

- **Recent buffer**: 最佳 3–5 帧，再大反而引入冗余信息。
- **Keyframe buffer**: 无上限最好，证明长程任务确实需要长记忆。
- **Backbone 规模**: [[Qwen3-VL]] 越大越好，但收益逐渐递减。

---

## 批判性思考

### 优点

1. **Benchmark 含金量高**: 8 维特征对比表里 RoboMemArena 是唯一全勾的，且 26 任务 + 真机配对的工作量真实可见。
2. **关键帧抽取有物理基础**: 公式 1–3 用「夹爪状态切换 + 速度拐点」的物理直觉，比纯学习式的关键帧选择更可靠且可解释。
3. **预测编码头零推理开销**: 训练时辅助、推理时移除，这是个非常实用的「附加正则」设计，对工程友好。
4. **异步双系统的频率比设计合理**: 2.92:1 的比例让 S1 永不卡顿，避免了双系统方案常见的「S2 太慢 S1 等」问题。

### 局限性

1. **预测编码头收益有限**: Table 2 显示去掉它只掉 6.2 点 TSR，相比关键帧库的 20.8 点贡献小很多。论文的「核心创新」其实是关键帧库选择策略，预测编码更像锦上添花。
2. **IHMB 任务普遍接近 0%**: 真机 Table 4 中 IHMB 所有方法都 ≤10%，说明记忆机制对真正复杂的真实任务还远不够。
3. **关键帧库 uncapped**: 对长程任务（>2000 步）的内存与推理延迟可能是问题，论文没充分讨论实际部署的内存上界。
4. **S2 频率过低**: 1.06 Hz 意味着子任务切换每秒只能更新一次，对快速变化的场景可能反应不过来。
5. **数据规模仍小**: 26 任务 × 100 演示 = 2600 条，相比 [[DROID]]、[[OXE]] 等大规模数据集仍偏小，且都在仿真。

### 潜在改进方向

1. **学习式关键帧选择**: 用 S2 的预测信号端到端学习关键帧入库判据，替代手工的 phys/kin 启发式。
2. **关键帧库压缩**: 加入「关键帧合并 / 遗忘机制」（参考 [[H-Net]] / [[SAM 3]]），限制无界增长。
3. **跨任务迁移**: 当前 PrediMem 每个任务独立训练，未充分利用 26 个任务的共享结构。
4. **与 [[World Model]] 融合**: 预测编码头可以扩展成完整的 future state prediction，做 model-based planning。
5. **更高 S2 频率**: 用更小的 S2 backbone 或蒸馏，让 S2 跑到 5 Hz 以上。

### 可复现性评估

- [x] 项目主页 (https://robomemarena.github.io)
- [ ] 代码开源（未明确）
- [ ] 预训练模型（未明确）
- [x] 训练细节完整（Qwen3-VL-8B / 2 epochs / lr 1e-5 / 4×H100）
- [x] 数据集可获取（仿真任务可生成，真机数据待发布）

---

## 关联笔记

### 基于

- [[Pi05|π₀.₅]]: S1 低层动作策略的基模型
- [[Qwen3-VL]]: S2 高层规划器的 VLM backbone
- [[JEPA]] / [[V-JEPA]]: 预测编码头的设计灵感来源
- [[AnyGrasp]]: 数据流水线的自主抓取模块

### 对比

- [[MemER]]: 最强 baseline，也是双系统 + 关键帧检索，但缺预测编码训练目标
- [[MemoryVLA]]: token-level working memory 路线，被 PrediMem 全面超越
- [[HiF-VLA]]: hindsight/insight/foresight 路线，不针对记忆任务
- [[RoboCasa]] / [[LIBERO]]: 主流操作 benchmark，但缺长程 + 记忆维度

### 方法相关

- [[关键帧记忆库|Memory Bank]]: PrediMem 的核心数据结构
- [[预测编码头|Predictive Coding Head]]: 训练辅助目标
- [[双系统架构]]: 整体推理范式
- [[VLA]]: 论文整体范畴

### 硬件/数据相关

- [[RoboMemArena]] (benchmark 同名概念): 26 仿真任务 + 5 真机任务

---

## 速查卡片

> [!summary] RoboMemArena & PrediMem
> - **核心**: 长程机器人记忆 benchmark + 双系统 VLA 解法
> - **方法**: S2 ([[Qwen3-VL]]) 选 [[关键帧记忆库|关键帧]] + 分发子任务，S1 ([[Pi05|π₀.₅]]) 高频执行动作；训练时加 [[预测编码头|Predictive Coding Head]] 辅助
> - **结果**: 仿真 TSR 38.5% (vs MemER 27.3%)，真机 52% (vs MemER 40%)
> - **项目页**: https://robomemarena.github.io

---

*笔记创建时间: 2026-05-14*
