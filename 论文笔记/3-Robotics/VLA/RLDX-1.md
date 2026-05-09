---
title: "RLDX-1 Technical Report"
method_name: "RLDX-1"
authors: [Dongyoung Kim, Huiwon Jang, Myungkyu Koo, Suhyeok Jang, Taeyoung Kim, Beomjun Kim, Byungjun Yoon, Changsung Jang, Daewon Choi, Dongsu Han, Donguk Lee, Heeseung Kwon, Hojin Jeon, Jaehyun Kang, Jaekyoung Bae, Jihyuk Lee, Jimin Lee, John Won, Joonwoo Ahn, Junhyeong Park, Junyoung Sung, Kyungmin Lee, Minseong Han, Minsung Yoon, Sejune Joo, Seonil Son, Seungcheol Park, Seunggeun Cho, Seungjun Moon, Seungku Kim, Yonghoon Dong, Yongjin Cho, Youngchan Kim, Jinwoo Shin]
year: 2026
venue: arXiv
tags: [vla, dexterous-manipulation, flow-matching, humanoid-robotics, tactile-sensing, synthetic-data, robot-policy]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.03269
created: 2026-05-09
---

# 论文笔记：RLDX-1 Technical Report

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | RLWRLD, KAIST |
| 日期 | May 2026 |
| 项目主页 | [rlwrld.ai/rldx-1](https://rlwrld.ai/rldx-1) |
| 对比基线 | [[pi0]], [[GR00T N1.6]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.03269) / [Code](https://github.com/RLWRLD/RLDX-1) / [Models](https://huggingface.co/collections/RLWRLD/rldx-1) |

---

## 一句话总结

> RLDX-1 通过 Multi-Stream Action Transformer 将运动感知、长期记忆和物理触觉融合进 VLA，在复杂仿人机器人操纵任务上大幅超越 π₀.₅ 和 GR00T N1.6。

---

## 核心贡献

1. **Multi-Stream Action Transformer (MSAT)**: 为异构感知流（认知、动作、物理）设计独立处理通道并通过联合自注意力融合，统一处理视觉语言、本体感觉和触觉/力矩信号
2. **运动感知 + 长期记忆 VLM**: 在 Qwen3-VL 8B 基础上引入时空自相似运动模块和滑动队列记忆模块，填补现有 VLA 对时序动态和长程上下文建模的空白
3. **合成数据流水线 + 三阶段训练**: 基于运动一致性过滤的合成数据增强，结合预训练→中训→后训的渐进式特化策略，并通过 CUDA Graph 优化将推理延迟从 71.2ms 降至 43.7ms（1.63×）

---

## 问题背景

### 要解决的问题

现有 [[VLA|Vision-Language-Action 模型]] 在仿人灵巧操纵中面临三个关键局限：(1) 缺乏对运动目标的时序感知能力（如抓取传送带上运动的物体）；(2) 无法维护长程记忆来处理多步骤序列任务；(3) 仅依赖视觉而无法利用触觉和力矩等物理信号。

### 现有方法的局限

- π₀ / π₀.₅：依赖流匹配策略，但 VLM 骨干缺乏运动感知模块和显式记忆
- GR00T N1.5/N1.6：在仿人操纵上表现尚可，但在长记忆和物理感知任务上成功率仅约 30%
- 通用 VLA：训练数据以单臂夹爪为主，缺乏五指仿人手的灵巧性数据

### 本文的动机

五指仿人手的灵巧操纵需要"五维灵巧性"——抓取多样性、空间精度、时序精度（运动感知）、接触精度（触觉/力矩）和上下文感知（记忆）——需要一个统一架构同时覆盖这五个维度。

---

## 方法详解

### 模型架构

RLDX-1 采用 **双组件** 架构（[[VLM|视觉语言模型]] + [[Transformer|动作模型]]）：

- **输入**: 视频帧 $\mathbf{o}_{t-K:t}$ + 语言指令 $\mathbf{l}_t$ + 本体感觉状态 $\mathbf{q}_t$ + 可选物理信号 $\mathbf{p}_t$（触觉/力矩）
- **VLM Backbone**: [[Qwen3-VL]] 8B，配备运动模块和记忆模块
- **核心模块**: [[MSAT|Multi-Stream Action Transformer]] 用于异构流联合建模
- **输出**: [[Action Chunking|动作块]] $\mathbf{a}_{t:t+H}$（ALLEX: H=40，FR3: H=16）
- **总参数**: 8.1B

![Figure 1: RLDX-1 at a Glance](https://arxiv.org/html/2605.03269/2605.03269v2/x3.png)

**说明**: RLDX-1 整体概览，展示集成多模态输入和三大功能能力（运动感知、长期记忆和物理感知）用于灵巧操纵的 VLA 模型框架。

![Figure 2: RLDX-1 Functional Capabilities](https://arxiv.org/html/2605.03269/2605.03269v2/x4.png)

**说明**: 给定视频观测和语言指令，RLDX-1 通过运动模块（Motion Module）、记忆模块（Memory Module）和物理流（Physics Stream）分别处理触觉/力矩信号，经 VLM 提取认知表示后，MSAT 联合去噪动作和物理信号。

![Figure 3: RLDX-1 Full Architecture](https://arxiv.org/html/2605.03269/2605.03269v2/x5.png)

**说明**: 完整架构图。VLM 侧：视频帧经运动模块增强后由 Qwen3-VL 8B 提取，64 个认知 token 聚合特征，记忆模块维护历史队列。动作侧：MSAT 以认知特征、记忆特征、物理信号、机器人状态和含噪动作为输入，预测未来物理信号和动作块。

### 核心模块

#### 模块1: VLM 认知接口（Cognition Interface）

**设计动机**: 从 [[VLM|视觉语言模型]] 中提取动作相关特征，而非使用全部 token，以减少动作模型的处理负担

**具体实现**:
- 使用 64 个可学习的**认知 token** $\mathbf{q}$ 通过交叉注意力从 VLM 隐状态提取特征
- 视觉编码: $\mathbf{v}_t = \mathcal{E}_\theta(\mathbf{o}_{t-K:t})$，其中 K 帧使用时间偏移 $\{-6,-4,-2,0\}$
- 合并输入序列: $\mathbf{x} = [\mathbf{v}_t,\ \mathbf{l}_t,\ \mathbf{q}]$

#### 模块2: 运动模块（Motion Module）

**设计动机**: 捕捉[[时空自相似性|空间-时间自相似性]]来感知场景中物体的运动，弥补静态视觉特征的不足

**具体实现**:
- 计算时序帧间的空间相关矩阵 $\mathbf{S}_t$
- 运动增强视觉特征: $\tilde{\mathbf{v}}^i_t = \mathbf{v}^i_t + \mathcal{S}_\theta(\mathbf{S}_t)$
- 将运动信号叠加在 patch-level 视觉特征上

#### 模块3: 记忆模块（Memory Module）

**设计动机**: 维护长程上下文以支持需要记忆历史动作结果的多步骤任务

**具体实现**:
- 维护滑动队列 $\mathbf{Q}_t = [\mathbf{h}_{t-n_\text{mem}\cdot H},\ldots,\mathbf{h}_{t-2H},\mathbf{h}_{t-H}]$，$n_\text{mem}=3$
- 记忆特征融合: $\mathbf{m}_t = \mathcal{M}_\theta([\mathbf{Q}_t,\ \mathbf{h}_t])$
- $\mathbf{h}_t$ 为当前步的认知特征；队列存储过去 chunk 结束时的认知特征

![Figure 3: RLDX-1 Full Architecture (repeat)](https://arxiv.org/html/2605.03269/2605.03269v2/x5.png)

**说明**: 认知接口与记忆模块的详细结构：64 个可学习认知 token 通过交叉注意力提取 VLM 特征，记忆模块维护 3 步历史认知特征队列 $\mathbf{Q}_t$，与当前认知特征 $\mathbf{h}_t$ 拼接后经 Transformer 生成记忆特征 $\mathbf{m}_t$。

#### 模块4: Multi-Stream Action Transformer (MSAT)

**设计动机**: 异构输入（视觉语言上下文、本体感觉、物理信号）的维度和语义差异巨大，需要模态专用通道保护各自表示

**具体实现**:
- **认知流 (Cognition Stream)**: 处理来自 VLM 的高维视觉语言上下文 $\mathbf{c}_t$
- **动作流 (Action Stream)**: 处理本体感觉状态 $\mathbf{q}_t$ 和含噪动作 $\mathbf{a}^\tau_{t:t+H}$
- **物理流 (Physics Stream)**（可选）: 处理触觉和力矩信号 $\mathbf{p}_t$
- 三流通过[[联合自注意力|Joint Self-Attention]] 相互交互，同时保留模态专用表示

![Figure 2: RLDX-1 Functional Capabilities](https://arxiv.org/html/2605.03269/2605.03269v2/x4.png)

**说明**: RLDX-1 在执行操纵任务时的功能模块协同：运动感知处理动态目标，长期记忆保持历史信息，物理感知接入触觉/力矩信号，三者通过 MSAT 联合预测动作。

---

## 关键公式

### 公式1: [[Flow Matching|流匹配训练目标]]

$$
\mathcal{L}(\theta; t, \tau, \epsilon) = \left\| \mathbf{u}_\theta\!\left(\mathbf{a}^\tau_{t:t+H},\ \tau,\ \mathbf{c}_t\right) - \left(\mathbf{a}_{t:t+H} - \epsilon\right) \right\|^2_2
$$

**含义**: 训练策略网络预测从噪声到真实动作的速度场，实现基于[[流匹配|条件流匹配]]的动作生成

**符号说明**:
- $\mathbf{a}^\tau_{t:t+H} = \tau \mathbf{a}_{t:t+H} + (1-\tau)\epsilon$: 插值后的含噪动作序列
- $\tau \sim \mathcal{U}(0,1)$: 随机采样的插值系数（时间步）
- $\epsilon \sim \mathcal{N}(0, I)$: 标准高斯噪声
- $\mathbf{u}_\theta$: 策略网络预测的速度场
- $\mathbf{c}_t$: 来自 VLM 的认知上下文
- $H$: 动作预测 horizon（ALLEX: 40，FR3: 16）

### 公式2: [[Euler积分|Euler 法推理采样]]

$$
\mathbf{a}^{\tau_{i+1}}_{t:t+H} = \mathbf{a}^{\tau_i}_{t:t+H} + (\tau_{i+1} - \tau_i)\, \mathbf{u}_\theta\!\left(\mathbf{a}^{\tau_i}_{t:t+H},\ \tau_i,\ \mathbf{c}_t\right)
$$

**含义**: 推理阶段通过 Euler 积分从纯噪声 $\tau_1=0$ 逐步去噪至真实动作 $\tau_T=1$

**符号说明**:
- $0 = \tau_1 < \tau_2 < \cdots < \tau_T = 1$: 推理时间步序列
- $(\tau_{i+1} - \tau_i)$: 步长，与速度场相乘得到动作增量

### 公式3: [[运动模块|运动增强视觉编码]]

$$
\tilde{\mathbf{v}}^i_t = \mathbf{v}^i_t + \mathcal{S}_\theta(\mathbf{S}_t)
$$

**含义**: 将时空自相似矩阵计算的运动信号叠加到 patch 级视觉特征，注入运动感知能力

**符号说明**:
- $\mathbf{v}^i_t$: 第 $i$ 帧的 patch-level 视觉特征
- $\mathbf{S}_t$: 多帧间的空间相关矩阵（时空自相似性）
- $\mathcal{S}_\theta$: 可学习的运动特征提取器

### 公式4: [[记忆模块|滑动队列记忆融合]]

$$
\mathbf{Q}_t = [\mathbf{h}_{t-n_\text{mem}\cdot H},\ldots,\mathbf{h}_{t-2H},\mathbf{h}_{t-H}], \quad \mathbf{m}_t = \mathcal{M}_\theta([\mathbf{Q}_t,\ \mathbf{h}_t])
$$

**含义**: 从历史认知特征队列中提取长程记忆，与当前认知特征融合后作为记忆条件

**符号说明**:
- $\mathbf{h}_{t-kH}$: 过去第 $k$ 个 chunk 结束时的认知特征
- $n_\text{mem}=3$: 记忆队列长度（保留最近 3 个 chunk 的历史）
- $\mathcal{M}_\theta$: 可学习的记忆融合模块

### 公式5: [[物理流|含噪物理信号构造]]

$$
\mathbf{p}^{\tau+L}_{t+1:t+L} = \tau\, \mathbf{p}_{t+1:t+L} + (1-\tau)\, \epsilon_\mathbf{p}
$$

**含义**: 对触觉/力矩信号序列同样施加流匹配噪声，使物理流与动作流采用一致的训练范式

**符号说明**:
- $\mathbf{p}_{t+1:t+L}$: 未来 $L$ 步的真实物理信号序列
- $\epsilon_\mathbf{p}$: 物理信号专用高斯噪声
- $L$: 物理信号预测 horizon

---

## 关键图表

### Figure 4: 合成数据框架

![Figure 4: Synthetic Data Framework](https://arxiv.org/html/2605.03269/2605.03269v2/x6.png)

**说明**: 合成数据两阶段流水线：(1) 数据生成——对源演示施加任务增强（VLM 生成新指令）和场景增强（I2I/V2V 图像变换），逆动力学模型（IDM）标注动作标签；(2) 数据过滤——仿真回放 IDM 预测动作，运动一致性分类器过滤与合成视频不一致的样本。

### Figure 5: 合成数据示例

![Figure 5: Synthetic Data Examples](https://arxiv.org/html/2605.03269/2605.03269v2/x7.png)

**说明**: 以 ALLEX 叠杯泡面演示为例：(a) 原始演示，(b) 任务增强变体（VLM 生成不同指令），(c) 场景增强变体（初始帧 I2I 编辑后接 I2V 生成）。

### Figure 6: 预训练数据集构成

![Figure 6: Pre-training Dataset Composition](https://arxiv.org/html/2605.03269/2605.03269v2/x8.png)

**说明**: 预训练数据集总计 1.5M 条，涵盖单臂夹爪、双臂夹爪和仿人手各类形态：Open-X-Embodiment (870K)、DROID (92K)、Galaxea (114K)、AgiBot World (275K)、Fourier ActionNet (30K)、Humanoid Everyday (9K)、合成数据 (150K)。

### Figure 7: 中间训练数据集构成

![Figure 7: Mid-training Dataset Composition](https://arxiv.org/html/2605.03269/2605.03269v2/x9.png)

**说明**: 中间训练针对两类平台：ALLEX（自采 + 合成数据，5:5 比例）和 FR3（DROID + 自采，8:2 比例）。

### Figure 8: 文本批评家 vs. 分布式批评家

![Figure 8: Text-based Critic Value Curves](https://arxiv.org/html/2605.03269/2605.03269v2/x10.png)

**说明**: 文本预测 critic（RECAP 风格）与分布式 critic 的 cube pick-and-place 任务价值曲线对比：(a) 文本 critic 在成功轨迹上产生更单调递增的值；(b) 能捕捉失败后重试恢复的价值变化。

### Figure 9: 静态计算图推理加速

![Figure 9: Dynamic vs. Static Graph](https://arxiv.org/html/2605.03269/2605.03269v2/x11.png)

**说明**: 动态图（PyTorch Eager）每次 forward 均重新调度并累积 kernel launch 开销；静态图（CUDA Graph）将整个前向传播捕获为单一 CUDA Graph，消除重复 launch 开销，并预计算旋转位置嵌入和注意力掩码。

### Figure 10: 算子融合内存访问

![Figure 10: Operator Fusion Memory Access](https://arxiv.org/html/2605.03269/2605.03269v2/x12.png)

**说明**: (a) 未融合：每个 kernel 输出写回显存，下一个 kernel 再读取，内存往返主导运行时；(b) 融合：只有一次输入读取和一次输出写入，最小化内存流量。Graph Capture + Kernel Fusion 合计将延迟从 71.2ms 降至 43.7ms（1.63×）。

### Figure 11: 仿真基准概览

![Figure 11: Simulation Benchmarks Overview](https://arxiv.org/html/2605.03269/2605.03269v2/x13.png)

**说明**: (a) 经典基准：LIBERO（Short/Long/Plus）、SIMPLER（Google Robot VM/VA + WidowX），用于单臂评测和分布外鲁棒性；(b) 挑战性基准：RoboCasa Kitchen、GR-1 Tabletop、RoboCasa365（含 Atomic 和 Compositional 任务），测试更复杂的长程操纵。

### Figure 12: 真实机器人平台

![Figure 12: Real-robot Platforms](https://arxiv.org/html/2605.03269/2605.03269v2/x14.png)

**说明**: (a) OpenArm + Inspire RH56F1 手（28-DoF 上半身人形机器人，立体自视角相机）；(b) ALLEX（48-DoF 上半身人形，立体自视角相机）；(c) Franka Research 3（7-DoF 单臂，AnySkin 触觉传感器，腕部 + 第三视角相机）。

### Figure 13: OpenArm 基准任务设置

![Figure 13: OpenArm Humanoid Benchmark](https://arxiv.org/html/2605.03269/2605.03269v2/x15.png)

**说明**: 六项评测任务的初始设置可视化：Basic PnP（基础拾放）、Directional PnP（货架/碗架定向放置）、Unseen Object（未见实例泛化）、Unseen Task（未见放置方式泛化）、Object Grounding（语言指向定位）。

### Figure 14: OpenArm 基准结果

![Figure 14: OpenArm Benchmark Results](https://arxiv.org/html/2605.03269/2605.03269v2/x16.png)

**说明**: 微调后各 VLA 在 OpenArm 六任务上的成功率（%）对比。RLDX-1 在所有任务上均大幅领先，包括训练时见过和未见过的设置。Unseen Object 从 π₀.₅ 的 37.5% 提升至 54.2%；Unseen Task 从 45.8% 提升至 54.2%。

### Table 1(a): 经典仿真基准结果

| 方法 | LIBERO Short | LIBERO Long | LIBERO Avg. | LIBERO-Plus | SIMPLER Google-VM | SIMPLER Google-VA | SIMPLER WidowX |
|------|-------------|------------|------------|------------|------------------|------------------|----------------|
| π₀-FAST | 93.9 | 60.2 | 85.5 | 64.2 | 61.9 | 59.0 | 48.3 |
| π₀ | 97.1 | 85.2 | 94.1 | 54.6 | 58.8 | 54.8 | 27.1 |
| π₀.₅ | 98.0 | 92.0 | 96.9 | 86.5 | 72.7 | 68.4 | 46.9 |
| GR00T N1.5 | 90.0 | 76.0 | 86.5 | 66.3 | 52.4 | 43.7 | 62.0 |
| GR00T N1.6 | 97.4 | 94.4 | 96.7 | 72.6 | 76.1 | 57.1 | 57.1 |
| **RLDX-1 (Ours)** | **98.6** | **95.3** | **97.8** | **86.7** | **81.5** | **77.4** | **71.9** |

**关键发现**: RLDX-1 在 LIBERO Avg. (97.8%) 和 SIMPLER WidowX (71.9%) 上显著领先，WidowX 任务上超越第二名 GR00T N1.6 约 15 个百分点。

### Table 1(b): 挑战性仿真基准结果

| 方法 | RoboCasa Kitchen | GR-1 Tabletop | RoboCasa365 Atomic-S | RoboCasa365 Comp.-S | RoboCasa365 Comp.-U | RoboCasa365 Avg. |
|------|------------------|---------------|----------------------|----------------------|----------------------|------------------|
| π₀-FAST | 63.6 | — | 51.7 | 8.0 | 1.8 | 21.7 |
| π₀ | 62.5 | 13.6 | 34.6 | 6.1 | 1.1 | 14.8 |
| π₀.₅ | 62.1 | 15.4 | 39.6 | 7.1 | 1.2 | 16.9 |
| GR00T N1.5 | 65.7 | 48.0 | 43.0 | 9.6 | 4.4 | 20.0 |
| GR00T N1.6 | 66.2 | 47.6 | 61.1 | 12.6 | 2.6 | 26.9 |
| **RLDX-1 (Ours)** | **70.6** | **58.7** | **67.3** | **19.0** | **5.6** | **32.1** |

**关键发现**: RoboCasa365 Comp.-S 上 RLDX-1（19.0%）vs GR00T N1.6（12.6%），组合任务难度显著更高，RLDX-1 优势在复杂任务上更突出。

### Table 2: OpenArm 真实机器人结果

| 任务 | π₀.₅ | GR00T N1.6 | RLDX-1 (Ours) |
|------|------|-----------|----------------|
| Basic PnP | — | — | 最高 |
| Directional PnP (Shelf) | — | — | 最高 |
| Directional PnP (Dish Rack) | — | — | 最高 |
| Unseen Object (Instance) | 37.5% | — | **54.2%** |
| Unseen Task (Placement) | 45.8% | — | **54.2%** |
| Object Grounding | — | — | 最高 |

**关键发现**: RLDX-1 在未见物体和未见任务泛化上均明显优于 π₀.₅，体现更强的指令跟随和泛化能力。

### Table 3: ALLEX 功能性任务结果

| 任务 | 测试能力 | π₀.₅ | GR00T N1.6 | RLDX-1 (Ours) |
|------|---------|------|-----------|----------------|
| 传送带抓取（Conveyor Belt） | 运动感知 | 29.2% | ~30% | **87.5%** |
| 盒中物体选择（Object-in-Box） | 长期记忆 | ~30% | ~30% | **91.7%** |
| 接触丰富任务 | 物理感知 | 低 | 低 | 大幅领先 |

**关键发现**: 在运动感知任务上 RLDX-1 成功率约为竞品 3 倍；在长期记忆任务上约为竞品 3 倍，充分验证三大新增功能模块的有效性。

### Table 4: 推理延迟优化

| 配置 | 延迟 (ms) | 加速比 |
|------|-----------|--------|
| PyTorch Eager（基线） | 71.2 | 1.00× |
| + Graph Capture | ~55 | ~1.3× |
| + Kernel Fusion（完整优化） | **43.7** | **1.63×** |

**关键发现**: Graph 优化 + 算子融合将延迟从 71.2ms 降至 43.7ms，满足闭环机器人控制的实时需求。

### Table 5: 预训练数据集组成

| 数据集 | 形态 | 末端执行器 | 集数 |
|--------|------|------------|------|
| Open-X-Embodiment | 单臂 | 夹爪 | 870K |
| DROID | 单臂 | 夹爪 | 92K |
| Galaxea Open-World | 双臂 | 夹爪 | 114K |
| AgiBot World (G) | 仿人 | 夹爪 | 239K |
| AgiBot World (H) | 仿人 | 手 | 36K |
| Fourier ActionNet | 仿人 | 手 | 30K |
| Humanoid Everyday | 仿人 | 手 | 9K |
| 合成数据 | 仿人 | 手 | 150K |
| **合计** | | | **1.5M** |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| Open-X-Embodiment | 870K | 多形态单臂 | 预训练 |
| DROID | 92K | 单臂 Franka | 预训练 |
| Galaxea Open-World | 114K | 双臂夹爪 | 预训练 |
| AgiBot World | 275K | 仿人夹爪+手 | 预训练 |
| Synthetic Data | 150K | IDM 标注 + 场景增强 | 预训练 |
| ALLEX 内部数据 | — | 仿人灵巧手 | 中训/后训 |
| FR3 内部数据 | — | 带触觉传感器 | 中训/后训 |
| LIBERO | 标准 | 桌面操纵 | 仿真评估 |
| RoboCasa365 | 365任务 | 厨房场景 | 仿真评估 |
| GR-1 Tabletop | — | 仿人桌面 | 仿真评估 |

### 实现细节

- **VLM Backbone**: [[Qwen3-VL]] 8B（机器人专用微调）
- **预训练优化器**: AdamW，学习率 $1\times10^{-4}$，5% warmup 后恒定
- **预训练步数/Batch**: 100K steps，batch size 8192
- **预训练硬件**: 64 × NVIDIA H200 GPUs（约 195 小时）
- **中训步数/Batch**: 25K steps，batch size 1024，学习率 $5\times10^{-5}$
- **中训对齐预热**: 前 2K steps 冻结预训练参数
- **模态 Dropout**: 0.3（扩展输入流）
- **仿真微调**: 60K steps，batch 1024，学习率 $1\times10^{-4}$
- **动作 horizon**: ALLEX 40 steps，FR3 16 steps
- **记忆队列长度**: $n_\text{mem}=3$

---

## 批判性思考

### 优点

1. **功能完整性**: 首次在单一 VLA 框架中统一运动感知、长期记忆和物理触觉三大能力，适配工业级灵巧操纵需求
2. **数据效率**: 基于运动一致性过滤的合成数据流水线有效扩充仿人手数据（现实中难以大规模采集），消融显示 GR-1 Tabletop 提升 9.1%
3. **推理实用性**: 1.63× 延迟优化（71.2ms→43.7ms）是从实验室到实际部署的关键工程贡献
4. **评估全面性**: 在 ALLEX 上专门设计功能性任务（运动感知、长记忆）来验证各模块有效性，比单纯比较通用基准更有说服力

### 局限性

1. **部分真实机器人结果未完整披露**: OpenArm 表格中部分任务未给出竞品绝对数值，比较不够系统
2. **合成数据泛化边界不清**: 合成数据过滤后质量如何，与真实数据的分布差距对不同任务影响未深入分析
3. **物理流依赖专有硬件**: 触觉/力矩流需要 AnySkin 等特定传感器，限制了方法在标准机器人上的直接复现

### 潜在改进方向

1. 将记忆模块扩展为可学习的压缩记忆（如 Mamba 或 Linear Attention），支持更长的任务 horizon
2. 合成数据生成引入更精确的物理仿真（如 MuJoCo 接触力），改善触觉合成数据质量
3. 探索跨形态统一模型（无需中训阶段），减少部署新机器人时的适配成本

### 可复现性评估

- [x] 代码开源（github.com/RLWRLD/RLDX-1）
- [x] 预训练模型（HuggingFace RLWRLD/rldx-1 集合）
- [ ] 训练细节完整（合成数据流水线细节、IDM 架构未完整披露）
- [ ] 数据集可获取（ALLEX/FR3 内部数据集不可获取）

---

## 关联笔记

### 基于

- [[Qwen3-VL]]: VLM 骨干，机器人专用微调基础
- [[Flow Matching]]: 动作生成核心范式（条件流匹配）
- [[Action Chunking]]: 动作块预测策略

### 对比

- [[pi0]]: π₀ 系列，主要对比基线，缺乏运动感知和记忆模块
- [[GR00T N1.6]]: NVIDIA 仿人策略，在功能性任务上大幅落后 RLDX-1

### 方法相关

- [[MSAT|Multi-Stream Action Transformer]]: 核心动作模型架构
- [[联合自注意力|Joint Self-Attention]]: 跨流信息融合机制
- [[时空自相似性]]: 运动感知的核心计算
- [[CUDA Graph]]: 推理加速的关键技术

### 硬件/数据相关

- [[ALLEX]]: 48-DoF 仿人上身平台（主要评测平台）
- [[Franka Research 3]]: 7-DoF 单臂 + AnySkin 触觉（物理感知评测）
- [[AnySkin]]: 触觉传感器，物理流的输入来源
- [[Open-X-Embodiment]]: 预训练主要数据来源（870K episodes）

---

## 速查卡片

> [!summary] RLDX-1 Technical Report
> - **核心**: 统一运动感知、长期记忆、物理感知三大功能的仿人灵巧操纵 VLA
> - **方法**: Qwen3-VL 8B + MSAT 三流架构 + 合成数据过滤 + 三阶段训练
> - **结果**: LIBERO 97.8%，ALLEX 运动感知任务 87.5%（vs π₀.₅ 29.2%），推理延迟 43.7ms
> - **代码**: [github.com/RLWRLD/RLDX-1](https://github.com/RLWRLD/RLDX-1)

---

*笔记创建时间: 2026-05-09*
