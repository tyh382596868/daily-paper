---
title: "FASTER: Rethinking Real-Time Flow VLAs"
method_name: "FASTER"
authors: [Yuxiang Lu, Zhe Liu, Xianzhe Fan, Zhenya Yang, Jinghua Hou, Junyi Li, Kaixin Ding, Hengshuang Zhao]
year: 2026
venue: arXiv
tags: [vla, flow-matching, real-time-inference, action-chunking, asynchronous-inference, robot-manipulation]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2603.19199v3
created: 2026-05-19
---

# 论文笔记：FASTER: Rethinking Real-Time Flow VLAs

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | The University of Hong Kong, ACE Robotics |
| 日期 | March 2026 (v3: May 2026) |
| 项目主页 | https://innovator-zero.github.io/FASTER |
| 对比基线 | [[pi0.5]], [[X-VLA]], [[Real-Time Chunking]] |
| 链接 | [arXiv](https://arxiv.org/abs/2603.19199) / [Code](https://github.com/innovator-zero/FASTER) |

---

## 一句话总结

> 用 Horizon-Aware Schedule 让 [[Flow Matching]] VLA 的首动作单步去噪，把 [[Action Chunking|动作块]] 推理的反应延迟压缩 10×，并配合 streaming 客户端-服务端管线实现真实机器人的低延迟反应。

---

## 核心贡献

1. **反应时间的概率刻画**: 首次系统地把动作块策略的反应时间 $\Delta t_{react}$ 建模为关于 [[TTFA]] 和 [[执行视距]] 的均匀分布，并给出同步/异步推理的期望反应时间闭式表达。
2. **Pilot Study 揭示常数 schedule 的浪费**: 通过 [[去噪路径直线度]] 与 [[Clean Action Estimation|clean-action 提前估计]] 实验发现，常数 [[采样 schedule]] 在 [[Flow Matching]] VLA 中是延迟瓶颈 —— 近期动作其实可以用更少步数完成。
3. **Horizon-Aware Schedule (HAS)**: 提出指标依赖的时间分配机制，让首动作在 1 个去噪步内就绪，远期动作仍保持高质量；配合 mixed-schedule 训练，**无需额外训练成本**。
4. **流式 Client-Server 推理管线**: 服务端边算边吐出已就绪的动作，客户端边收边执行，把网络/计算延迟掩藏在动作执行间隔中。
5. **真实机器人验证**: 在动态乒乓球任务、Pick Beverage、Fold Towel 三个任务上验证，消费级 GPU (RTX 4060) 上 X-VLA 反应延迟从 599.5 ms 降到 229.2 ms（2.62×）。

---

## 问题背景

### 要解决的问题

[[VLA]] 通过 [[Action Chunking|动作块]] 一次预测 $H$ 步动作来提升时间一致性。但 [[Flow Matching]] 类 VLA（[[pi0.5]]、[[X-VLA]]、π₀）的多步去噪推理在消费级 GPU 上非常慢（X-VLA 在 RTX 4060 上单次推理 ≈ 400 ms），导致机器人对环境变化的**反应延迟过大**，尤其是动态任务（如接球、追物）。

### 现有方法的局限

- **同步推理**: 必须等整个动作块执行完才启动下一次推理，反应时间最坏 $\approx \Delta t_{infer} + \Delta t_{exec}$，无法处理快速变化的场景。
- **异步推理 / [[Real-Time Chunking|RTC]]**: 在动作块执行期间并行计算下一块，把推理延迟掩盖在执行时间里，但仍然要求 $\Delta t_{exec} \geq \Delta t_{infer}$，即**执行视距 $s$ 不能小于 $\lceil \Delta t_{infer}/\Delta t_{ctrl} \rceil$**。这导致最小反应时间被 $\Delta t_{infer}$ 卡住。
- **常数 schedule 的低效**: 现有 flow VLA 用同一组 $\tau$ 时间步去噪整块动作，但近期动作其实**早就稳定**，多步去噪是浪费。

### 本文的动机

如果能让**首个动作在第 1 步就足够准**，远期动作再继续 refine，那么：
1. **TTFA**（Time To First Action）从 $N\cdot\delta$ 降到 $\delta$；
2. 流式管线可以在第 1 个去噪步结束后立刻把首动作送给机器人，并行 refine 后续动作；
3. 反应延迟主要由网络 + 1 步去噪决定，不再受 $N$ 步去噪牵制。

---

## 方法详解

### 模型架构

FASTER 不引入新模型，而是改造已有 [[Flow Matching]] VLA（[[pi0.5]]、[[X-VLA]]）的**推理调度**和**服务接口**：

- **输入**: 语言指令 $l$ + 多视角观测 $o_t$（front + 双 wrist 相机）+ 机器人本体状态 $s_t$
- **Backbone**: 复用预训练的 $\pi_{0.5}$ / X-VLA，参数不变
- **核心改造**:
  - 训练侧：[[Mixed Schedule Training|混合 schedule 训练]]，以概率 $p$ 用 HAS、$1-p$ 用常数 schedule
  - 推理侧：[[Horizon-Aware Schedule|HAS]] 替代常数 $\tau$ schedule + [[Streaming Client-Server Pipeline|流式管线]]
- **输出**: [[Action Chunking|动作块]] $A_t = (a_t, a_{t+1}, \ldots, a_{t+H-1})$，但按去噪进度**渐进式 dispatch**

### 核心模块

#### 模块 1: Pilot Study —— 为什么常数 schedule 是浪费

**设计动机**: 在改 schedule 之前先证明它"该改"。作者用 [[去噪路径直线度|straightness metric]] 和 [[Clean Action Estimation]] 检查近期 vs 远期动作的去噪路径。

**关键发现**:
- 近期动作（chunk 前部）的去噪路径**接近直线**，意味着 1 步欧拉积分就能逼近最终动作；
- 远期动作（chunk 后部）路径更弯，需要多步积分；
- 即使近期动作的速度场更"直"，常数 schedule 仍然给它分配相同步数 → **可压缩**。

#### 模块 2: Horizon-Aware Schedule (HAS)

**设计动机**: 把"近直、远弯"的特性显式编码到 schedule 里 —— 让索引 $i$ 越小（越早执行）的动作越早完成去噪。

**具体实现**:
1. 定义每个动作 $a_{t+i}$ 的[[Hit Time|命中时间]] $u_i$，表示它"应该在全局采样进度 $\rho$ 降到 $u_i$ 时就已经去噪完成"；
2. 用幂律分配 $u_i$：$u_i = (1 - (i/(H-1))^\alpha)\cdot u_0$，$\alpha \in (0,1]$ 控制衰减形状；
3. 设 $u_0 = (N-1)/N$，使首动作在第 1 个全局步内就 $\rho \leq u_0$ → 完成去噪；
4. 每个动作的局部 $\tau_i^j$ 由 $(\rho^j - u_i)/(1-u_i)$ 给出，已完成的动作 $\tau_i^j = 0$（被冻结）。

#### 模块 3: Mixed Schedule Training

**动机**: 直接换 schedule 会让推理分布偏离训练分布。但**完全切换**到 HAS 又会丧失常数 schedule 的鲁棒性。

**做法**: 训练时，对每个 batch sample，以概率 $p$ 用 HAS 时间步 $\{\tau_i\}$，以 $1-p$ 用常数 $\tau$。这样模型同时见过两种 schedule，推理时自由切换，**无额外训练成本**。

#### 模块 4: Streaming Client-Server Pipeline

**动机**: 即便 HAS 让首动作秒出，传统"等整块算完才返回"的接口仍然浪费这点速度。

**具体实现**:
- **服务端**: 每完成一个动作的去噪（$u_i$ 被命中），立刻通过 TCP/UDP 把它发给客户端；同时并行 refine 后续动作；
- **客户端**: 持续监听 socket，把新到的动作追加到执行 buffer；
- **执行视距收缩**: 因为首动作可在 1 步内就绪，最小执行视距 $s_{min}$ 从 $\lceil \Delta t_{infer}/\Delta t_{ctrl}\rceil$ 降到 $\lceil \delta/\Delta t_{ctrl}\rceil$（$\delta$ 是单步去噪时间）；
- **early stopping**: 当所有要执行的 $s$ 个动作都完成 refine，提前结束当前 chunk 的去噪，立即触发下一次推理。

---

## 关键公式

### 公式 1: [[Flow Matching|流匹配训练损失]]

$$
\mathcal{L}(\theta) = \mathbb{E}_{\tau \sim \mathcal{U}(0,1)} \left\| v_\theta(o_t, A_t^\tau, \tau) - (\varepsilon - \hat{A}_t) \right\|^2
$$

**含义**: 训练 [[Flow Matching]] 速度场 $v_\theta$ 去回归从噪声 $\varepsilon$ 到真值动作 $\hat{A}_t$ 的位移向量，是 VLA 标准训练目标。

**符号说明**:
- $\tau \sim \mathcal{U}(0,1)$: 均匀采样的流时间
- $A_t^\tau = \tau\varepsilon + (1-\tau)\hat{A}_t$: 噪声与真值的线性插值（推理时方向反过来）
- $v_\theta$: 条件速度场（条件是观测 $o_t$）
- $\varepsilon \sim \mathcal{N}(0, I)$: 高斯噪声

### 公式 2: [[ODE 积分|去噪积分]]

$$
A_t^{\tau + \Delta\tau} = A_t^\tau + v_\theta(o_t, A_t^\tau, \tau)\Delta\tau
$$

**含义**: 推理时用一阶欧拉法把动作从 $\tau=1$（纯噪声）积分到 $\tau=0$（干净动作）。

**符号说明**:
- $\Delta\tau = -1/N$: 单步步长，$N$ 是总采样步数
- 共需 $N$ 步去噪才能拿到完整 clean 动作

### 公式 3: [[去噪路径直线度|Straightness Metric]]

$$
S(A) = \sum_{\tau=0}^{1} \mathbb{E}_t \left[ \left\| (A_t^1 - A_t^0) - v_\theta(o_t, A_t^\tau, \tau) \right\|^2 \right] \cdot (-\Delta\tau)
$$

**含义**: 测量去噪过程偏离"从 $A_t^1$ 到 $A_t^0$ 直线"的程度。$S=0$ 表示完全直线 → 1 步欧拉积分即可。论文用它证明近期动作的 $S$ 远小于远期动作。

**符号说明**:
- $A_t^1, A_t^0$: 起始噪声 / 终点 clean 动作
- $v_\theta(o_t, A_t^\tau, \tau)$: 当前点的速度场预测
- 求和 / 积分覆盖整个去噪过程

### 公式 4: [[Clean Action Estimation|Clean Action 提前估计]]

$$
\tilde{A}_t^{\tau \to 0} = A_t^\tau - v_\theta(o_t, A_t^\tau, \tau)\tau
$$

**含义**: 在去噪中途的任意 $\tau$ 上，用当前速度场**外推**出最终 clean 动作。如果 $\tilde{A}_t^{\tau\to 0}$ 在小 $\tau$ 时就趋于稳定，说明该动作已经"基本去噪完毕"，可以立即 dispatch。

**符号说明**:
- $\tilde{A}_t^{\tau\to 0}$: 在时间 $\tau$ 处对终点的估计
- 论文用它的稳定性验证 HAS 的可行性

### 公式 5: [[Hit Time|命中时间公式]]

$$
u_i = \left(1 - \left(\frac{i}{H-1}\right)^\alpha\right)\cdot u_0, \quad i \in [1, H-1]
$$

**含义**: HAS 的核心 —— 为索引 $i$ 的动作分配一个"应当完成去噪"的全局进度点 $u_i$。$\alpha < 1$ 让早期动作的 $u_i$ 接近 $u_0$（很早就完成），晚期动作的 $u_i$ 接近 0（最后才完成）。

**符号说明**:
- $i$: 动作在 chunk 中的索引（0 为最早）
- $H$: chunk 长度（论文默认 50）
- $\alpha \in (0,1]$: 形状参数，越小近期加速越激进
- $u_0 = (N-1)/N$: 首动作命中时间（保证 1 步内完成）

### 公式 6: [[局部时间步|Local Time Schedule]]

$$
\tau_i^j = \max\left(0, \frac{\rho^j - u_i}{1 - u_i}\right)
$$

**含义**: 把全局进度 $\rho^j$（$j$ 是去噪步数）映射成每个动作的**局部** $\tau$。一旦 $\rho^j \leq u_i$，则 $\tau_i^j = 0$，该动作被冻结（已完成）。

**符号说明**:
- $\rho^j$: 全局采样进度，从 1 降到 0
- $u_i$: 公式 5 给的命中时间
- $\max(0, \cdot)$: 已完成动作不再变化

### 公式 7: [[同步推理|同步反应时间期望]]

$$
\Delta t_{react}^{sync} \sim \mathcal{U}(\Delta t_{infer},\; 2\Delta t_{infer} + \Delta t_{exec}), \quad \mathbb{E}[\Delta t_{react}^{sync}] = 1.5\Delta t_{infer} + 0.5\Delta t_{exec}
$$

**含义**: 同步推理下，环境变化发生时机随机，最坏情况需要等当前推理完成 + 整块执行完 + 下一次推理。

### 公式 8: [[异步推理|异步反应时间期望]]

$$
\Delta t_{react}^{async} \sim \mathcal{U}(\Delta t_{infer},\; \Delta t_{infer} + \Delta t_{exec}), \quad \mathbb{E}[\Delta t_{react}^{async}] = \Delta t_{infer} + 0.5\Delta t_{exec}
$$

**含义**: 异步推理把"下一次推理"和当前执行并行，省掉一个 $\Delta t_{infer}$，但首动作仍要等整次推理完成。

### 公式 9: [[最小执行视距|离散化推理延迟]]

$$
d := \left\lceil \frac{\Delta t_{infer}}{\Delta t_{ctrl}} \right\rceil - 1 = \left\lfloor \frac{\Delta t_{infer}}{\Delta t_{ctrl}} \right\rfloor
$$

**含义**: 异步管线下，控制周期数表示的推理延迟。$s_{min} = d + 1$ 是不发生 pipeline stall 的最小执行视距。FASTER 通过单步去噪把 $\Delta t_{infer}$ 从 $N\delta$ 降到 $\delta$，从而把 $s_{min}$ 大幅压缩。

---

## 关键图表

### Figure 1: Overview / 反应速度对比

![Figure 1](https://arxiv.org/html/2603.19199v3/x1.png)

**说明**: FASTER 在 [[pi0.5]] 和 [[X-VLA]] 上分别取得 1.29× 和 2.54×（RTX 4090）/ 3.09×（RTX 4060）加速。核心源自 single-step 首动作采样把 [[TTFA]] 砍下来。

### Figure 2: Temporal Pipelines / 同步 vs 异步 vs FASTER

![Figure 2](https://arxiv.org/html/2603.19199v3/x2.png)

**说明**: 三种推理范式的时间线对比。同步推理在执行结束才启动下一次推理；[[异步推理]] 并行执行和下一次推理但要求 $s \geq \lceil\Delta t_{infer}/\Delta t_{ctrl}\rceil$；FASTER 用 [[Streaming Client-Server Pipeline|流式管线]] + [[Horizon-Aware Schedule|HAS]]，首动作 1 步就 dispatch，反应时间几乎只剩 1 个去噪步 + 网络延迟。

### Figure 3: Pilot Study —— Path Straightness

![Figure 3](https://arxiv.org/html/2603.19199v3/x3.png)
![Figure 3 (b)](https://arxiv.org/html/2603.19199v3/x4.png)
![Figure 3 (c)](https://arxiv.org/html/2603.19199v3/x5.png)

**说明**: 用 [[去噪路径直线度]] 和 [[Clean Action Estimation|clean-action 估计]] 可视化 chunk 中不同索引动作的去噪路径。**近期动作的路径明显更直，clean-action 估计也更早稳定**，证明它们用 1 步去噪就足够。

### Figure 4: Horizon-Aware Schedule 示意

![Figure 4](https://arxiv.org/html/2603.19199v3/x6.png)

**说明**: 对比 (左) 常数时间步 schedule 和 (右) [[Horizon-Aware Schedule|HAS]]。HAS 让索引小（早执行）的动作 $\tau$ 沿对角线快速降到 0，索引大的动作 $\tau$ 保持高位多步 refine。$u_i$ 的幂律分布是关键。

### Figure 5: 真实乒乓球任务

![Figure 5](https://arxiv.org/html/2603.19199v3/x7.png)

**说明**: 在 RTX 4090 上动态乒乓球任务的对比。FASTER 在球到达击球点时**球拍角度已经到位**，而 sync/async baseline 还在挥拍中途，球已飞过。最能体现"反应延迟"的实际意义。

### Figure 6: Pick Beverage & Fold Towel 表现

![Figure 6](https://arxiv.org/html/2603.19199v3/x8.png)

**说明**: 在两个准静态任务上，FASTER 成功率与 [[Real-Time Chunking|RTC]] 持平或更好，证明加速**不牺牲**任务精度。

### Figure 7: Async 推理细粒度时序

![Figure 7](https://arxiv.org/html/2603.19199v3/x9.png)

**说明**: 附录补充图，把异步推理的"推理触发 - 完成 - 执行追加"时序展开，解释为什么必须满足 $s \geq d+1$ 否则 stall。

### Figure 8: 附加 Pilot Study 可视化

![Figure 8 (a)](https://arxiv.org/html/2603.19199v3/x10.png)
![Figure 8 (b)](https://arxiv.org/html/2603.19199v3/x11.png)
![Figure 8 (c)](https://arxiv.org/html/2603.19199v3/x12.png)
![Figure 8 (d)](https://arxiv.org/html/2603.19199v3/x13.png)
![Figure 8 (e)](https://arxiv.org/html/2603.19199v3/x14.png)
![Figure 8 (f)](https://arxiv.org/html/2603.19199v3/x15.png)

**说明**: 在不同模型 ([[pi0.5]] / [[X-VLA]]) 和不同数据集 (LIBERO / CALVIN ABC) 上重复 Pilot Study，结论一致：近期动作去噪路径都更直。

### Figure 9: AgileX Cobot Magic 机器人平台

![Figure 9](https://arxiv.org/html/2603.19199v3/x16.png)

**说明**: 真实实验所用的 AgileX Cobot Magic 平台，4 个 6-DoF Piper 机械臂，1 个前视 D455 + 2 个手腕 D435。

### Figure 10: Pick Beverage / Fold Towel 任务可视化

![Figure 10](https://arxiv.org/html/2603.19199v3/x17.png)

**说明**: 两个任务的关键帧示意，展示双臂操作可形变物体的能力。

### Table 1: 同步 vs 异步反应时间分布

| Mode | $\Delta t_{react}$ 分布 | $\mathbb{E}[\Delta t_{react}]$ |
|------|------------------------|-------------------------------|
| Sync | $\mathcal{U}(\Delta t_{infer},\; 2\Delta t_{infer} + \Delta t_{exec})$ | $1.5\Delta t_{infer} + 0.5\Delta t_{exec}$ |
| Async | $\mathcal{U}(\Delta t_{infer},\; \Delta t_{infer} + \Delta t_{exec})$ | $\Delta t_{infer} + 0.5\Delta t_{exec}$ |

**说明**: 反应时间的概率形式 —— 这是论文的理论起点。

### Table 2: 反应能力对比

**RTX 4090**:

| Model | Method | TTFA | $s_{min}$ | $\mathbb{E}[\Delta t_{react}]$ | Speedup |
|-------|--------|------|-----------|--------------------------------|---------|
| [[pi0.5]] | Sync | 80.0±1.6 ms | 3 | 170.0 ms | - |
| [[pi0.5]] | Async | 80.0±1.6 ms | 3 | 130.0 ms | - |
| [[pi0.5]] | **FASTER** | **62.1±3.1 ms** | 3 | **112.1 ms** | **1.29×** |
| [[X-VLA]] | Sync | 113.7±0.8 ms | 4 | 237.2 ms | - |
| [[X-VLA]] | Async | 113.7±0.8 ms | 4 | 180.4 ms | - |
| [[X-VLA]] | **FASTER** | **44.8±0.3 ms** | **2** | **78.1 ms** | **2.54×** |

**RTX 4060**:

| Model | Method | TTFA | $s_{min}$ | $\mathbb{E}[\Delta t_{react}]$ | Speedup |
|-------|--------|------|-----------|--------------------------------|---------|
| [[pi0.5]] | Sync | 303.3±0.8 ms | 10 | 621.6 ms | - |
| [[pi0.5]] | Async | 303.3±0.8 ms | 10 | 470.0 ms | - |
| [[pi0.5]] | **FASTER** | **238.6±1.9 ms** | **8** | **371.9 ms** | **1.27×** |
| [[X-VLA]] | Sync | 399.5±8.5 ms | 12 | 799.2 ms | - |
| [[X-VLA]] | Async | 399.5±8.5 ms | 12 | 599.5 ms | - |
| [[X-VLA]] | **FASTER** | **129.2±2.4 ms** | **6** | **229.2 ms** | **3.09×** |

**说明**: FASTER 在所有配置下都降低 [[TTFA]] 和期望反应时间，**消费级 GPU 上加速更显著** —— X-VLA 在 RTX 4060 上提速 3.09×，几乎追上 RTX 4090 sync。同时最小执行视距 $s_{min}$ 也大幅缩短，意味着 chunk 可以拆得更细 → 更高控制频率。

### Table 3: 概率化反应速度分析

| Model | GPU | $P(\text{FASTER} < \text{Sync})$ | $P(\text{FASTER} < \text{Async})$ |
|-------|-----|----------------------------------|------------------------------------|
| [[pi0.5]] | RTX 4090 | 0.81 | 0.66 |
| [[X-VLA]] | RTX 4090 | **1.00** | **1.00** |
| [[X-VLA]] | RTX 4060 | **1.00** | **1.00** |

**说明**: X-VLA 上 FASTER 达到**确定性的全面优势**（1.00 概率反应更快）。pi0.5 也有 66–81% 的概率赢。

### Table 4: 推理延迟 vs 顺序执行（流式接口）

| Hardware | Time Required | Time Received |
|----------|---------------|---------------|
| RTX 4090 | 单 action 完成时间 | 客户端收到时间 |
| RTX 4060 | 同上 | 同上 |

**说明**: 论文用此表验证流式接口确实能让动作"边算边到"，而不是等整块完成。具体数值见原文 Appendix C。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| LIBERO | 标准 benchmark | 仿真操作任务 | Pilot Study 可视化 |
| CALVIN ABC | 标准 benchmark | 长程仿真任务 | Pilot Study 可视化 |
| Table Tennis | 335 episodes (~14 min) | 高度动态 | 真实反应速度 |
| Pick Beverage | 150 episodes | 物体定位 + 抓取 | 真实成功率 |
| Fold Towel | 150 episodes | 双臂可形变物体 | 真实成功率 |

### 实现细节

- **Backbone**: 复用预训练的 [[pi0.5]] 和 [[X-VLA]]，FASTER 仅微调
- **训练方式**: [[Mixed Schedule Training|Mixed Schedule]]，混合概率 $p$（论文未公开具体值）
- **HAS 超参**: $\alpha \in (0, 1]$，$u_0 = (N-1)/N$
- **Chunk 长度**: $H = 50$（默认），消融含 $H=30$
- **控制频率**: $f = 30$ Hz（$\Delta t_{ctrl} = 33.3$ ms）
- **机器人**: AgileX Cobot Magic，4× Piper 6-DoF 机械臂
- **相机**: 1× RealSense D455 (front) + 2× D435 (wrist)
- **推理硬件**: RTX 4090（高端）& RTX 4060（消费级）
- **训练硬件**: 论文未明确

### 可视化结果

- **乒乓球任务**: FASTER 在球到达瞬间球拍角度对位准确，sync/async 仍在中途挥拍 → 反应延迟直接决定任务成败；
- **Pick Beverage**: FASTER 抓取成功率与 [[Real-Time Chunking|RTC]] 持平，但 chunk 拆得更细，执行更流畅；
- **Fold Towel**: 双臂同步性保持，没因加速产生抖动。

---

## 批判性思考

### 优点

1. **理论刻画清晰**: 把反应时间分布 + 期望写成闭式表达，是首个把 VLA 实时性问题量化的工作，为后续研究立了 benchmark；
2. **零额外训练成本**: Mixed Schedule 只是改训练采样，不加新模块、不动模型权重；
3. **完全 plug-in**: 适用于任何 [[Flow Matching]] VLA，[[pi0.5]]、[[X-VLA]]、π₀ 都验证过；
4. **消费级 GPU 收益最大**: RTX 4060 上 3× 加速，让 VLA 真正能跑在低成本机器人上；
5. **Pilot Study 漂亮**: 用 [[去噪路径直线度]] 客观证明"近期动作可以加速"，不是拍脑袋的设计。

### 局限性

1. **只对 flow-based VLA 有效**: 对 [[Diffusion Policy]] 这类非线性 schedule 模型未必直接适用，因为去噪路径直线度的假设不一定成立；
2. **$\alpha$ 和 $p$ 的选择敏感**: 论文虽然给了机制，但具体值未公开，工程可复现性受影响；
3. **真实任务覆盖窄**: 只在 3 个真实任务上验证，乒乓球虽然亮眼但很特定；Pick Beverage / Fold Towel 是准静态，加速收益没那么关键；
4. **没和 Distillation 类方法对比**: 与 [[Consistency Distillation]]、[[DMD]] 这种从根本上降步数的方法没有直接对比，缺少"加速 vs 蒸馏"的取舍讨论；
5. **首动作激进单步可能引入误差**: 虽然 Mixed Schedule 训练缓解了分布偏移，但单步去噪在 OOD 场景下的鲁棒性未做压力测试。

### 潜在改进方向

1. **自适应 $\alpha$**: 根据当前观测的"紧迫程度"动态调整 $\alpha$，紧急情况下更激进；
2. **与蒸馏结合**: 在 HAS 基础上叠加 [[Consistency Distillation]]，把整块步数也降下来；
3. **扩展到 Diffusion Policy**: 重新定义直线度指标，把 HAS 思想搬到 DDPM/DDIM；
4. **服务端调度优化**: 流式管线可以考虑 priority queue，把"近 deadline"的动作 batch 起来一起算。

### 可复现性评估

- [x] 代码开源（https://github.com/innovator-zero/FASTER）
- [x] 预训练模型基于公开的 [[pi0.5]] / [[X-VLA]]
- [ ] 训练细节完整（$\alpha$、$p$、epochs、batch、lr 未公开）
- [x] 数据集可获取（LIBERO/CALVIN 公开，真实数据未必开放）

---

## 关联笔记

### 基于

- [[Flow Matching]]: 整个加速框架的去噪基础
- [[Action Chunking|动作块预测]]: VLA 的标准输出范式
- [[pi0.5]]: 主要 backbone 之一
- [[X-VLA]]: 另一 backbone（论文中加速比最高的）
- [[Real-Time Chunking]]: 最直接的对比 baseline，FASTER 的思想可视为对它的进一步极限化

### 对比

- [[Real-Time Chunking]]: 同样想降反应延迟，但仍要求 $s \geq \lceil\Delta t_{infer}/\Delta t_{ctrl}\rceil$；FASTER 通过单步首动作打破这一约束
- [[Consistency Distillation]]: 也是降步数，但需要 distillation 训练；FASTER 不蒸馏不重训
- [[DMD]]: 蒸馏到 1-4 步，路线不同，更彻底但更贵

### 方法相关

- [[Horizon-Aware Schedule]]: 本文核心机制
- [[Mixed Schedule Training]]: 关键训练 trick
- [[Streaming Client-Server Pipeline]]: 系统侧创新
- [[TTFA]]: Time to First Action，反应延迟的关键 metric
- [[去噪路径直线度]]: Pilot Study 的核心指标
- [[Clean Action Estimation]]: 提前估计 clean 动作的方法
- [[Hit Time]]: HAS 中每个动作的命中进度
- [[最小执行视距]]: 异步管线的硬约束

### 硬件/数据相关

- [[AgileX Cobot Magic]]: 真实实验平台
- [[LIBERO]]: 仿真 benchmark
- [[CALVIN]]: 长程仿真 benchmark

---

## 速查卡片

> [!summary] FASTER: Rethinking Real-Time Flow VLAs
> - **核心**: 用 [[Horizon-Aware Schedule|HAS]] 让 [[Flow Matching]] VLA 的首动作 1 步出，配合流式管线降反应延迟
> - **方法**: 幂律 hit-time $u_i = (1-(i/(H-1))^\alpha)u_0$ + Mixed Schedule 训练 + Streaming Client-Server
> - **结果**: RTX 4090 上 X-VLA 2.54× 加速，RTX 4060 上 3.09× 加速；乒乓球真机验证；零额外训练成本
> - **代码**: https://github.com/innovator-zero/FASTER

---

*笔记创建时间: 2026-05-19*
