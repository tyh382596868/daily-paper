---
title: "KEMO: Event-Driven Keyframe Memory for Long-Horizon Robot Manipulation with VLA Policies"
method_name: "KEMO"
authors: [Yihan Zeng, Minghao Ye, Yiyuan Chen, Yide Shentu, Philipp Wu, Zike Yan, Zhongyu Li]
year: 2026
venue: arXiv
tags: [vla, memory-augmented-policy, long-horizon-manipulation, keyframe-detection, bimanual-robot]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.23589
created: 2026-06-23
---

# 论文笔记：KEMO: Event-Driven Keyframe Memory for Long-Horizon Robot Manipulation with VLA Policies

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | xdof.ai、香港具身AI实验室（HKUST）、中国科学技术大学、上海交通大学、香港中文大学 |
| 日期 | June 2026 |
| 项目主页 | - |
| 对比基线 | [[VLA]]（π₀.₅）、[[MemoryVLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.23589) |

---

## 一句话总结

> KEMO 是一个轻量级即插即用记忆框架，通过机器人运动学+视觉去重自动检测任务关键帧并将其编码为记忆 token 注入 VLA，显著提升长时域双臂操作的成功率。

---

## 核心贡献

1. **事件驱动关键帧检测**: 融合关节空间减速（运动学）与 [[DINOv2]] 视觉相似度去重，自动识别抓取/放置等任务关键时刻，无需人工标注
2. **带门控残差的跨注意力融合**: 用 [[Masked Cross-Attention]] + [[Gated Residual Fusion]] 将关键帧记忆 token 注入 VLA 当前视觉特征，初始化偏置为负值防止训练早期记忆干扰
3. **关键帧一致性训练加权**: 在关键帧附近时间步提高损失权重（λ=8），强化模型对任务状态转变的感知能力

---

## 问题背景

### 要解决的问题

长时域机器人操作存在"**阶段歧义**"问题：相同的视觉观测（如空桌面）可能出现在不同的任务阶段，而正确动作取决于已完成的历史操作。例如 Cover Blocks 任务需 50 秒、1480 步和 6 个子任务，仅凭当前帧无法判断当前处于哪个阶段。

### 现有方法的局限

- **密集历史记录**（如 ContextVLA、HAMLET）：保留所有历史帧需要大量压缩，计算代价高，且大量冗余帧稀释关键信息
- **近期帧窗口**（如 RoboFlamingo）：只保留最近若干帧，容易遗漏几十秒前发生的关键事件
- **VLM 触发检测**（如 MemER）：依赖语言模型判断关键帧，推理延迟高，不适合实时控制
- **均匀采样**：无区分地从历史中等间隔取帧，无法保证采到关键状态转变时刻

### 本文的动机

关键状态转变（抓取、放置、释放）往往伴随机器人的**减速**——通过检测关节速度峰值可以精准定位这些时刻，再用视觉相似度去重避免保存冗余帧，从而以极少量（4-8 个）高质量关键帧代替完整历史。

---

## 方法详解

### 模型架构

KEMO 以 **π₀.₅**（[[VLA|π₀.₅]]）为骨干，在推理路径上插入一个轻量记忆模块：

- **输入**: 语言指令 $l$ + 当前视觉观测 $I_t$ + 机器人关节状态 $q_t$
- **关键帧检测器**: 运动学事件评分 + [[DINOv2]] 视觉去重
- **记忆库**: 固定容量 $K$ 个关键帧，每帧由 [[SigLIP]] 编码 + 4×4 空间池化压缩
- **融合模块**: [[Masked Cross-Attention]] + [[Gated Residual Fusion]]
- **Backbone**: [[VLA|π₀.₅]] 主干不变（冻结或微调）
- **输出**: [[Action Chunking|动作块]] $a_{t:t+H}$（动作视野 $H=50$）

### 核心模块

#### 模块 1：事件驱动关键帧检测（Event-Driven Keyframe Detection）

**设计动机**: 利用 [[Robot Kinematics|机器人运动学]] 中减速对应操作事件这一先验知识，无监督检测任务关键时刻。

**Step 1 — 事件显著性评分**:

在滑动窗口 $w$ 内计算各关节空间位移均值 $\bar{\delta}_t$，转为显著性分数 $S_t$（机器人静止/减速时分数高）：

$$
S_t = \frac{1}{1 + \bar{\delta}_t}
$$

**Step 2 — 候选峰值检测**:

在 $S_t$ 序列上做局部峰值检测，设置不应期 $r$（refractory period）避免在同一事件附近重复触发。

**Step 3 — 视觉去重过滤**:

用冻结的 [[DINOv2]] 编码器计算候选帧 $I_t^c$ 与上一保存关键帧 $I_{k_{i-1}}$ 之间的视觉相异度 $v_i$：

$$
v_i = 1 - \frac{\phi(I_t^c)^\top \phi(I_{k_{i-1}})}{\|\phi(I_t^c)\|_2 \|\phi(I_{k_{i-1}})\|_2}
$$

只有当 $v_i > \varepsilon$（$\varepsilon=0.05$）时才将候选帧加入记忆库，排除视觉相似冗余帧。

#### 模块 2：关键帧记忆融合（Keyframe Memory Integration）

**编码与压缩**:

- [[SigLIP]] 视觉编码器将关键帧 $I_{k_i}$ 编码为 patch token $Z_{k_i} \in \mathbb{R}^{N \times D}$
- 4×4 空间池化将其压缩为 $F_{k_i} \in \mathbb{R}^{16 \times D}$，降低计算量

**时序嵌入**:

为每个关键帧添加可学习的时序位置嵌入（Temporal Positional Embedding），区分记忆库中不同时间的关键帧。

**门控跨注意力融合**:

以当前帧 token $X_t$ 为 Query，关键帧记忆 token 为 Key/Value，做 [[Masked Cross-Attention]]，得到记忆增强特征 $X_t'$；随后通过 [[Gated Residual Fusion]] 合并：

$$
\hat{X}_t = X_t + \sigma(g([X_t;\, X_t'])) \cdot X_t'
$$

门控网络 $g(\cdot)$ 初始化为负偏置，使训练初期 $\sigma(\cdot) \approx 0$，避免未充分训练的记忆干扰骨干网络。

#### 模块 3：关键帧一致性训练加权（Keyframe-Consistent Training）

**设计动机**: 关键帧附近的时间步对任务成功至关重要，应赋予更高学习信号。

对每个训练时间步 $t$，计算其到最近关键帧的距离，若在窗口 $\delta$ 内则加权：

$$
w_t = \begin{cases} \lambda & \text{if } \min_i |t - k_i| \leq \delta \\ 1 & \text{otherwise} \end{cases}
$$

其中 $\lambda = 8.0$，$\delta = 3$（时间步）。加权损失为：

$$
\mathcal{L} = \sum_t w_t \cdot \mathcal{L}_t^{\text{VLA}}
$$

---

## 关键公式

### 公式 1：[[Event Saliency Score|事件显著性评分]]

$$
S_t = \frac{1}{1 + \bar{\delta}_t}
$$

**含义**: 将关节位移均值转化为 $(0,1]$ 的显著性分数，机器人减速/静止时分数高，表明可能正在发生操作事件。

**符号说明**:
- $S_t$: 时刻 $t$ 的事件显著性分数
- $\bar{\delta}_t$: 滑动窗口 $w$ 内各关节空间位移的均值
- 分数越高 → 关节越慢 → 越可能是关键操作时刻

### 公式 2：[[Cosine Similarity|视觉相异度（余弦距离）]]

$$
v_i = 1 - \frac{\phi(I_t^c)^\top \phi(I_{k_{i-1}})}{\|\phi(I_t^c)\|_2 \|\phi(I_{k_{i-1}})\|_2}
$$

**含义**: 用 [[DINOv2]] 特征的余弦距离衡量候选帧与上一关键帧的视觉差异，排除冗余的相似帧。

**符号说明**:
- $v_i$: 视觉相异度，取值 $[0, 2]$，越大越不同
- $\phi(\cdot)$: 冻结 DINOv2 编码器
- $I_t^c$: 候选关键帧
- $I_{k_{i-1}}$: 记忆库中上一个关键帧
- $\varepsilon = 0.05$: 相异度阈值，$v_i > \varepsilon$ 才接受该候选帧

### 公式 3：[[Gated Residual Fusion|门控残差融合]]

$$
\hat{X}_t = X_t + \sigma(g([X_t;\, X_t'])) \cdot X_t'
$$

**含义**: 通过可学习门控动态调节记忆信息的注入量，初始化为关闭（负偏置），随训练逐渐开放。

**符号说明**:
- $X_t$: 当前帧视觉 token
- $X_t'$: 经跨注意力得到的记忆增强特征
- $g(\cdot)$: 门控网络（MLP），输出标量或向量
- $\sigma(\cdot)$: Sigmoid 激活
- $\hat{X}_t$: 融合后的特征，送入 VLA 骨干

### 公式 4：[[Keyframe-Consistent Training|关键帧一致性损失加权]]

$$
w_t = \begin{cases} \lambda & \text{if } \min_i |t - k_i| \leq \delta \\ 1 & \text{otherwise} \end{cases}
$$

$$
\mathcal{L} = \sum_t w_t \cdot \mathcal{L}_t^{\text{VLA}}
$$

**含义**: 对关键帧邻域时间步赋予更高损失权重，强迫模型在状态转变时刻学得更精准。

**符号说明**:
- $k_i$: 第 $i$ 个关键帧的时间步索引
- $\delta = 3$: 关键帧邻域半径（时间步数）
- $\lambda = 8.0$: 关键帧附近的损失放大系数
- $\mathcal{L}_t^{\text{VLA}}$: VLA 骨干在时刻 $t$ 的原始损失

---

## 关键图表

### Figure 1：整体介绍图

![Figure 1](https://arxiv.org/html/2606.23589v1/assets/introduce.png)

**说明**: 展示长时域双臂操作面临的阶段歧义问题，以及 KEMO 框架的核心思路（关键帧检测 → 记忆融合）。右侧柱状图展示 KEMO 相比 π₀.₅ 基线在 TSR 和 SCR 上的提升。

### Figure 2：系统架构总览

![Figure 2](https://arxiv.org/html/2606.23589v1/assets/overview.png)

**说明**: KEMO 完整系统流程图。左侧为关键帧检测器（运动学评分 + 视觉去重），右侧为记忆融合模块（SigLIP 编码 → 池化 → 跨注意力 → 门控融合 → VLA 骨干）。

### Figure 3a：关键帧检测流水线

![Figure 3a](https://arxiv.org/html/2606.23589v1/assets/keyframe_detection.png)

**说明**: 详细展示三步检测流程：事件显著性评分（关节速度 → $S_t$ 曲线）→ 局部峰值检测 → DINOv2 视觉去重过滤，最终选出进入记忆库的关键帧。

### Figure 3b：关键帧记忆融合模块

![Figure 3b](https://arxiv.org/html/2606.23589v1/assets/keyframe_memory.png)

**说明**: 展示记忆融合模块内部结构：SigLIP 编码、4×4 空间池化、时序位置嵌入、Masked Cross-Attention，以及门控残差融合如何将记忆 token 注入当前帧特征流。

### Figure 4：任务一——Swap Foods

![Figure 4](https://arxiv.org/html/2606.23589v1/assets/task_swap_foods.png)

**说明**: 双臂交换两个盘子里的食物。时长 28 秒，830 步，2 个子任务（取-放 × 2）。

### Figure 5：任务二——Find Block

![Figure 5](https://arxiv.org/html/2606.23589v1/assets/task_find_block.png)

**说明**: 用三个杯子盖住方块，再找出目标方块（需记住哪个杯子盖了目标）。时长 40 秒，1186 步，4 个子任务。典型阶段歧义场景。

### Figure 6：任务三——Cover Blocks

![Figure 6](https://arxiv.org/html/2606.23589v1/assets/task_cover_block.png)

**说明**: 从左到右依次盖住方块，再从右到左依次揭开。时长 50 秒，1480 步，6 个子任务。消融实验主要在此任务上进行。

### Figure 7：任务四——Box Refill

![Figure 7](https://arxiv.org/html/2606.23589v1/assets/task_box_refill.png)

**说明**: 从盒子里取出物品，再用匹配的新物品填充。时长 50 秒，1480 步，4 个子任务。

### Figure 8：任务五——Make Sandwich

![Figure 8](https://arxiv.org/html/2606.23589v1/assets/task_make_sandwich.png)

**说明**: 交替涂抹酱料和放置面包片制作三明治。时长 54 秒，1616 步，4 个子任务。

### Figure 9：任务六——Drawer Items Replacement

![Figure 9](https://arxiv.org/html/2606.23589v1/assets/task_drawer_items_replacement.png)

**说明**: 从抽屉中取出物品并用替换物品放回。时长 95 秒，2846 步，4 个子任务。最长最难任务。

### Figure 10：实验装置

![Figure 10](https://arxiv.org/html/2606.23589v1/assets/station.png)

**说明**: 实验采用双臂机器人平台，主从控制（Leader-Follower）采集数据，配备三个摄像头（左腕、右腕、头部）提供多视角观测。

### Table 1：六任务性能对比

| 方法 | Swap Foods TSR/SCR | Find Block TSR/SCR | Cover Blocks TSR/SCR | Box Refill TSR/SCR | Make Sandwich TSR/SCR | Drawer Items TSR/SCR |
|------|-------------------|-------------------|---------------------|-------------------|----------------------|---------------------|
| π₀.₅ (baseline) | 6/12, 1.500/2 | 0/12, 0.353/4 | 0/12, 1.333/6 | 4/12, 2.580/4 | 10/12, 3.330/4 | 0/12, 0.000/4 |
| [[MemoryVLA]] | 1/12, 0.750/2 | 0/12, 1.000/4 | 0/12, 0.333/6 | 0/12, 0.000/4 | 0/12, 1.000/4 | 0/12, 0.000/4 |
| **KEMO (ours)** | **8/12, 1.580/2** | **2/12, 3.167/4** | **9/12, 5.250/6** | **7/12, 3.330/4** | **11/12, 3.750/4** | **0/12, 1.417/4** |

**关键发现**: KEMO 在 5/6 个任务上大幅超越两个基线；Drawer Items（最长 95s）三种方法均难以完成，表明超长时域仍有挑战。

### Table 2：Cover Blocks 任务消融实验

| 配置 | TSR (out of 12) | SCR (out of 6) | 说明 |
|------|----------------|----------------|------|
| 均匀采样 | 0/12 | 1.833/6 | 关键帧质量不足 |
| 近期帧 | 1/12 | 0.700/6 | 遗漏早期关键事件 |
| 事件关键帧（无融合模块改进） | 3/12 | 2.500/6 | 仅检测，未加融合优化 |
| 无门控融合 | 0/12 | 0.000/6 | 记忆引入反而干扰 |
| 无损失加权 | 3/12 | 2.500/6 | 关键帧学习信号不足 |
| **完整 KEMO** | **9/12** | **5.250/6** | 所有组件协同发挥作用 |

**关键发现**: 门控融合最为关键——去掉后 TSR 从 9/12 降为 0/12，SCR 从 5.250 降为 0，说明无门控时记忆引入严重干扰 VLA 骨干。

---

## 实验

### 数据集

| 任务 | 时长 | 步数 | 子任务数 | 特点 |
|------|------|------|---------|------|
| Swap Foods | 28s | 830 | 2 | 最短，入门难度 |
| Find Block | 40s | 1186 | 4 | 需记住被遮挡物位置 |
| Cover Blocks | 50s | 1480 | 6 | 子任务最多，消融用 |
| Box Refill | 50s | 1480 | 4 | 物品配对替换 |
| Make Sandwich | 54s | 1616 | 4 | 交替操作序列 |
| Drawer Items | 95s | 2846 | 4 | 最长，最具挑战性 |

所有任务均为真实双臂机器人，采用主从遥操作采集演示数据。

### 实现细节

- **Backbone**: π₀.₅（[[VLA|Vision-Language-Action Model]]）
- **视觉编码器（记忆）**: [[SigLIP]]（patch token）
- **视觉相异度**: 冻结 [[DINOv2]]
- **优化器**: [[AdamW]]，学习率峰值 $2.5 \times 10^{-5}$，最小值 $2.5 \times 10^{-6}$
- **训练步数**: 30,000 步
- **Batch Size**: 32
- **动作视野**: $H = 50$ 步
- **记忆容量**: $K$ 根据任务子任务数手动设置（4-8 个）
- **硬件**: 2× A100 GPU
- **超参数**: $w$（滑动窗口大小）、$r$（不应期）、$\varepsilon=0.05$（视觉去重阈值）、$\lambda=8.0$、$\delta=3$

---

## 批判性思考

### 优点

1. **轻量即插即用**: 不修改 VLA 骨干，仅在推理路径插入记忆模块，可无缝对接不同 VLA 模型
2. **无监督关键帧检测**: 纯靠运动学信号（无需人工标注），实时可用，适合在线部署
3. **门控设计优雅**: 负偏置初始化保证训练稳定性，避免冷启动时记忆噪声

### 局限性

1. **超参数敏感**: 滑动窗口 $w$、不应期 $r$、记忆容量 $K$ 均需针对每类任务手动调整，泛化性存疑
2. **静态记忆容量**: $K$ 预先固定，无法自适应不同长度任务，限制跨任务迁移
3. **最长任务仍失败**: Drawer Items（95s）中 KEMO TSR 仍为 0/12，说明超长时域需要更强的时序推理能力
4. **仅在单一硬件平台验证**: 只有 xdof.ai 双臂机器人，泛化到其他平台未知

### 潜在改进方向

1. 自适应记忆容量（根据检测到的关键帧数量动态扩展）
2. 用强化学习或对比学习优化关键帧检测策略
3. 结合语言指令动态过滤与当前子任务无关的历史关键帧

### 可复现性评估

- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节完整
- [ ] 数据集可获取

---

## 关联笔记

### 基于

- [[VLA]]（π₀.₅）: 骨干 VLA 模型，KEMO 在其基础上增加记忆模块
- [[DINOv2]]: 用于视觉相异度计算的冻结特征提取器
- [[SigLIP]]: 用于关键帧视觉编码的 patch token 提取器

### 对比

- [[MemoryVLA]]: 对比基线，密集历史记忆方法，KEMO 在所有任务上全面超越
- [[Action-Gated Memory]]: 另一种门控记忆设计，与 KEMO 的门控残差融合思路相关

### 方法相关

- [[Masked Cross-Attention]]: 关键帧记忆融合的注意力机制
- [[Gated Residual Fusion]]: 核心融合设计，控制记忆注入量
- [[Action Chunking]]: VLA 输出的动作表示方式
- [[Robot Kinematics]]: 关键帧检测所依赖的运动学先验

### 硬件/数据相关

- [[Bimanual Manipulation]]: 任务类型，所有 6 个任务均为双臂操作

---

## 速查卡片

> [!summary] KEMO
> - **核心**: 运动学减速信号 + 视觉去重 → 自动选关键帧 → 记忆注入 VLA
> - **方法**: Event Saliency Score + DINOv2 去重 + Masked Cross-Attention + Gated Residual Fusion
> - **结果**: 相比 π₀.₅ TSR +23.6pp、SCR +34.1pp；相比 MemoryVLA TSR +50pp
> - **代码**: 未开源（arXiv 2606.23589）

---

*笔记创建时间: 2026-06-23*
