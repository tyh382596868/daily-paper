---
title: "SOLE-R1: Video-Language Reasoning as the Sole Reward for On-Robot Reinforcement Learning"
method_name: "SOLE-R1"
authors: [Philip Schroeder, Thomas Weng, Karl Schmeckpeper, Eric Rosen, Stephen Hart, Ondrej Biza]
year: 2026
venue: arXiv
tags: [video-language-reward, on-robot-rl, spatiotemporal-cot, dense-progress-reward, rlvr, grpo, reward-hacking, zero-shot-manipulation]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2603.28730v2
created: 2026-05-27
---

# 论文笔记：SOLE-R1: Video-Language Reasoning as the Sole Reward for On-Robot Reinforcement Learning

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | MIT + The AI Institute |
| 日期 | 2026 |
| 项目主页 | https://philip-mit.github.io/sole-r1/ |
| 对比基线 | [[GPT-5]], [[Gemini-3-Pro]], Gemini Robotics-ER 1.5, [[RoboReward]], [[ReWiND]], TOPReward, LRM, VLAC, LIV, Robometer |
| 链接 | [arXiv](https://arxiv.org/abs/2603.28730) / [Code](https://github.com/philipmit/sole-r1-model) / [Model](https://huggingface.co/pschro/SOLE-R1-8B) / [Data](https://huggingface.co/datasets/pschro/sole_training_data) |

---

## 一句话总结

> 把 [[VLM]] 训成"会做 [[时空思维链]]"的**密集进度估计器**，让机器人在**只有视频和语言目标**的情况下从零开始做 [[在线强化学习]]，单一信号即可驱动 24/40 个零样本操作任务成功。

---

## 核心贡献

1. **首个把 [[VLM]] 作为**[[在线强化学习]]**唯一奖励信号**的工作：不依赖任何 ground-truth reward 或人类演示，纯粹用 [[视频语言奖励模型]] 输出的进度估计驱动策略学习。
2. **[[时空思维链]] (Spatiotemporal CoT) 奖励范式**：每个时间步同时输出 `<think>` 推理过程和 `<answer>` 标量进度 $p_t \in [-100, 100]$，把"判断进度"显式分解为"先看变化，再打分"。
3. **大规模数据合成 pipeline**：1.2M [[时空 CoT 轨迹]] 来自 41K 视频，覆盖空间推理（[[SSR-CoT]]）+ 多帧时序（[[Spot-the-Difference]] / [[RoboVQA]]）+ 机器人进度（[[非专家轨迹合成]]）。
4. **抗 [[Reward Hacking|奖励攻击]] 的失败模式分析**：通过"感知成功 vs. 真实成功"散点图证明 SOLE-R1 的失败主要是 *signal-limited*（信号不足），而 baseline VLM 的失败主要是 *reward-hacked*（被策略欺骗）。
5. **混合训练框架**：[[SFT]] 注入推理能力 + [[RLVR]]（[[GRPO]]）通过可验证奖励 $r_{\text{format}} + r_{\text{acc}}$ 精修进度预测精度。

---

## 问题背景

### 要解决的问题

如何在**没有 ground-truth reward、没有人类演示**的情况下，让机器人通过 [[在线强化学习]] 在真实硬件上学会新的操作任务？

### 现有方法的局限

1. **稀疏奖励难以扩展**：手工设计 reward function 需要状态可观测的仿真，无法迁移到真实世界；
2. **演示驱动方法（[[模仿学习]] / RLHF）昂贵**：每个新任务都要大量遥操作；
3. **专用奖励模型 ([[RoboReward]], [[ReWiND]], LIV) 泛化弱**：训练数据局限于特定 embodiment 和任务族；
4. **通用 [[VLM]]（[[GPT-5]], [[Gemini-3-Pro]]）严重 [[Reward Hacking|奖励攻击]]**：策略学到"让画面看起来像成功"，但实际未完成任务（如手在目标位置摆造型而未抓取）；
5. **进度信号离散且滞后**：现有 VLM judge 只在 episode 结束给一个标签，缺乏时间稠密的引导信号。

### 本文的动机

把"任务进度"看作一个**视频条件的回归问题**：给定语言目标 $g$ 和视频帧序列 $o_{t-K+1:t}$，输出一个连续标量 $p_t$，并强制模型先做 [[时空思维链]] 推理再给分，让推理过程对抗 [[Reward Hacking|奖励攻击]]。这个标量本身就是**密集奖励**，从而把 RL 中"奖励工程"问题转化为"奖励模型推理质量"问题。

---

## 方法详解

### 模型架构

SOLE-R1 采用 **[[Qwen3-VL|Qwen3-VL-8B-Instruct]] backbone + 显式 CoT 输出头** 架构：

- **输入** $x_t$：自然语言目标 $g$ + 初始帧 $o_0$ + 滑动窗口最近 $K$ 帧 $o_{t-K+1:t}$ + 上一步进度预测 $p_{t-1}$
- **Backbone**：[[Qwen3-VL]]-8B，full fine-tuning
- **输出结构**：
  - `<think>` 通道：自由形式的[[时空思维链]]，描述视觉变化、任务推进、剩余子目标
  - `<answer>` 通道：标量进度 $p_t \in [-100, 100]$，可解析为奖励
- **窗口 $K$ 训练期随机化**：从 1 到若干帧变化，强制模型对**时间粒度灵活**
- **奖励转换**：把 $p_t$ 通过 [[进度奖励变换]] 映射为 RL 训练用的 dense reward

### 核心模块

#### 模块 1: [[时空思维链]] (Spatiotemporal CoT) 推理

**设计动机**：通用 VLM 直接打分容易被表象欺骗（[[Reward Hacking|奖励攻击]]）；强制先输出"我观察到什么变化、还差什么"的可读推理，能让模型把判断锚定在**物体状态变化**而非**画面相似度**上。

**具体实现**：
- 训练时所有进度标签都配有 GPT-4o 合成的 CoT rationale
- `<think>` 描述三类信息：空间关系（[[SSR-CoT]] 风格的 3D 相对位置）、跨帧变化（[[Spot-the-Difference]] 风格）、与目标的距离
- 推理时强制模型生成完整 `<think>` 再生成 `<answer>`，禁用直接出分

#### 模块 2: [[非专家轨迹合成]] (Non-Expert Trajectory Synthesis)

**设计动机**：进度回归需要见过"做对"、"做错"、"做一半又退回"这三种轨迹。专家数据只有单调上升的进度，模型容易把"画面好看"等同于"进度高"。

**具体实现**：
- **仿真侧**：在专家轨迹中**随机注入扰动动作**（推开物体、回退抓取），生成带回退的真实物理轨迹；进度由几何距离监督（如 end-effector 到目标的欧氏距离 / 物体到 goal 的距离）
- **真实世界侧**：无法访问仿真器，改用**时间反转**——把视频片段倒放，制造"明显退步"的样本；进度由**帧顺序**监督（正向 +Δ，反向 -Δ）

#### 模块 3: 两阶段 [[混合训练框架]]

**Stage 1 — [[SFT]] 注入推理基础**：
- 数据：10M 帧 / 4M CoT 轨迹，混合三类——
  1. 基础空间推理（[[SSR-CoT]]：image-depth-question-rationale-answer）
  2. 多帧时序推理（[[Spot-the-Difference]] + [[RoboVQA]] 等具身视频问答）
  3. 机器人进度数据（[[非专家轨迹合成]] 产物）
- 目标：最大似然，让模型学会"先 think 再 answer"的格式和基础视觉推理能力。

**Stage 2 — [[RLVR]] with [[GRPO]] 精修进度精度**：
- 用**可验证奖励** $r(o) = r_{\text{format}} + r_{\text{acc}}$ 替代人类偏好，避免标注成本
- $r_{\text{format}} \in [0, 0.5]$：检查 `<think>...</think><answer>...</answer>` 结构能否解析
- $r_{\text{acc}} \in [0, 1.5]$：用指数衰减度量预测进度与真值的距离
- [[GRPO]] 用 group baseline 估计 advantage，避免 critic 网络

#### 模块 4: 在线 RL 接入

**奖励变换**：原始 $p_t$ 直接当 reward 会有量纲和噪声问题，先 clip 再缩放：
- $r_t = \psi \cdot \text{clip}(p_t, -c, c)$
- 当推理频率低于控制频率时，用**线性插值**把稀疏的 $p_t$ 稠密化

**RL 算法**：策略侧用 [[DrQv2]] 或类似 off-policy actor-critic，观测为 wrist+external 双目 RGB + 本体感觉，动作为 end-effector delta + 夹爪开合。

---

## 关键公式

### 公式 1: [[时空进度输入]] 构造

$$
x_t = [g, o_0; \; o_{t-K+1:t}, \; p_{t-1}]
$$

**含义**：每个时间步 SOLE-R1 看到的上下文 = 语言目标 + 初始锚定帧 + 最近 $K$ 帧 + 上一步预测。$p_{t-1}$ 进入输入是为了让模型显式建模"我上一步说过什么、现在该如何更新"。

**符号说明**：
- $g$：自然语言任务目标
- $o_0$：episode 起始帧（充当 baseline 锚点，缓解时间漂移）
- $o_{t-K+1:t}$：当前窗口 $K$ 帧
- $p_{t-1}$：上一步标量进度（self-conditioning）
- $K$：训练时随机化的窗口大小

### 公式 2: [[进度奖励变换]]

$$
r_t = \psi \cdot \text{clip}(p_t, -c, c)
$$

**含义**：把模型输出的 $p_t \in [-100, 100]$ 截断到 $[-c, c]$ 后缩放为 RL 训练用奖励，限制极端值对 actor-critic 训练的扰动。

**符号说明**：
- $p_t$：SOLE-R1 在 $t$ 时刻 `<answer>` 通道输出的标量进度
- $c$：clip 阈值（超参）
- $\psi$：奖励缩放因子，使 reward 量纲与策略学习率匹配

### 公式 3: [[SFT|监督微调]] 损失

$$
\mathcal{L}_{\text{SFT}}(\phi) = -\mathbb{E}_{(i,q,r,a) \sim \mathcal{D}} \left[\sum_{t=1}^{|y|} \log p_\phi(y_t \mid i, q, y_{<t})\right]
$$

**含义**：标准 next-token 似然，但目标序列 $y$ 同时包含 `<think>` 的 rationale $r$ 和 `<answer>` 的进度 $a$，让模型一次性学会推理与打分的联合分布。

**符号说明**：
- $\phi$：[[Qwen3-VL]] 模型参数
- $(i, q, r, a)$：图像/视频帧 $i$、问题 $q$、CoT rationale $r$、答案 $a$（进度值）
- $y$：拼接后的目标序列 = rationale + answer

### 公式 4: [[GRPO]] 目标函数

$$
\mathcal{J}_{\text{GRPO}}(\phi) = \mathbb{E}_{q,\{o_i\}} \left[\frac{1}{G}\sum_{i=1}^{G} \min\!\big(\rho_i(\phi) A_i,\; \text{clip}(\rho_i(\phi), 1-\epsilon, 1+\epsilon)\, A_i\big)\right] - \beta\, D_{\text{KL}}(p_\phi \| p_{\text{ref}})
$$

**含义**：PPO 风格的 clipped surrogate，但 advantage $A_i$ 由组内归一化估计（不需要 critic 网络），并加 [[KL 散度]] 正则防止模型偏离 SFT reference 太远。

**符号说明**：
- $G$：每个 prompt $q$ 采样的 group 大小
- $\rho_i(\phi) = p_\phi(o_i \mid q) / p_{\phi_{\text{old}}}(o_i \mid q)$：重要性比
- $A_i$：组内归一化的 advantage = $(r_i - \text{mean}(r_{1:G})) / \text{std}(r_{1:G})$
- $\epsilon$：clip 半径
- $\beta$：KL 系数
- $p_{\text{ref}}$：SFT 后冻结的参考模型

### 公式 5: [[可验证奖励]] 组成

$$
r(o) = r_{\text{format}}(o) + r_{\text{acc}}(o), \qquad r_{\text{acc}}(o) = \alpha \exp\!\left(-\frac{|\hat{p}_t - p_t|}{\tau}\right)
$$

**含义**：[[RLVR]] 的核心——奖励完全可程序化验证，不需要人类偏好或学习奖励模型。$r_{\text{format}}$ 检查输出结构是否合法（能解析出 `<think>`/`<answer>`），$r_{\text{acc}}$ 用指数距离度量进度预测的精确度。

**符号说明**：
- $\hat{p}_t$：模型预测的进度
- $p_t$：真值进度（来自 [[非专家轨迹合成]]）
- $\alpha$：精度奖励上限（论文取 1.5）
- $\tau$：温度，控制对小误差的敏感度
- $r_{\text{format}} \in [0, 0.5]$, $r_{\text{acc}} \in [0, 1.5]$, 总 $r \in [0, 2]$

### 公式 6: [[进度归一化]] (训练数据生成)

$$
v_t = \frac{-y_t + \max(y)}{-\min(y) + \max(y)}
$$

**含义**：把任意标量信号 $y$（如 end-effector 到目标的距离）归一化为 $[0, 1]$ 的进度值。注意符号取反：距离越大进度越低。

**符号说明**：
- $y_t$：$t$ 时刻原始几何量（如距离、夹角等）
- $\max(y), \min(y)$：episode 内的极值
- $v_t$：归一化后的进度（再线性映射到 $[-100, 100]$ 即为 $p_t$）

---

## 关键图表

### Figure 1: 总览 — SOLE-R1 作为单一奖励信号

> 🖼️ **Figure 1** — 图片暂缺，arXiv 抓取失败（原图见 [arXiv HTML](https://arxiv.org/html/2603.28730)）

**说明**：SOLE-R1 接收视频帧 + 语言目标，输出 `<think>` 推理 + `<answer>` 标量进度，进度直接作为 RL 训练的密集奖励。在 40 个零样本任务上，**SOLE-R1 在 24 个任务上达到 ≥50% 成功率**，远超 GPT-5 (7) 和 Gemini-3-Pro (5)。覆盖 [[RoboSuite]] / [[ManiSkill]] / [[Meta-World|Metaworld]] / [[LIBERO]] 仿真环境和真实 [[Franka 研究臂|Franka]] 机器人。

### Figure 2: 训练数据合成构成

> 🖼️ **Figure 2** — 图片暂缺，arXiv 抓取失败（原图见 [arXiv HTML](https://arxiv.org/html/2603.28730)）

**说明**：训练数据由三层混合：
- **基础空间推理**：[[SSR-CoT]] 风格的 image-depth-QA-rationale 元组（1.2M）
- **多帧时序推理**：[[Spot-the-Difference]] / [[RoboVQA]] 等具身视频 QA
- **机器人进度数据**：1.2M [[时空 CoT 轨迹]] 来自 41K 视频，仿真侧用 [[非专家轨迹合成]]（动作扰动 + 状态插值），真实侧用[[时间反转]]生成回退样本

总规模约 10M 帧 / 4M CoT 轨迹。

### Figure 3: 零样本在线 RL 成功率对比

![Figure 3](https://arxiv.org/html/2603.28730v2/x1.png)

**说明**：在 40 个跨环境任务上，SOLE-R1 作为唯一奖励信号训练的策略，平均成功率显著高于以 [[GPT-5]] / [[Gemini-3-Pro]] / 专用奖励模型（[[RoboReward]] / [[ReWiND]] / VLAC / LIV）为奖励的对照组。仿真三个随机种子，真实机器人单种子。

### Figure 4: [[Reward Hacking|奖励攻击]] 分析（感知 vs. 真实成功）

![Figure 4](https://arxiv.org/html/2603.28730v2/x6.png)

**说明**：散点图横轴 = 奖励模型感知的成功率，纵轴 = 真实成功率。
- **SOLE-R1 的点集中在对角线附近**（感知 ≈ 真实），失败多是 *signal-limited*（信号太弱，policy 学不到东西）
- **baseline VLM 的点偏向右下**（高感知、低真实），即典型的 [[Reward Hacking|奖励攻击]]——策略学到了"骗 VLM"的捷径

这是论文最重要的一张图，证明了**强制 CoT + 进度回归**对抗 reward hacking 的有效性。

### Figure 5: 消融实验

![Figure 5](https://arxiv.org/html/2603.28730v2/x2.png)

**说明**：三个关键消融：
- **w/o CoT**：移除 `<think>` 通道，进度预测直接出 → reward 曲线变平、reward hacking 增加
- **w/o non-expert data**（仅专家数据）→ 模型把"画面好"等同于"高进度"，分布漂移时崩溃
- **w/o foundational data**（仅机器人数据）→ 对未见 embodiment / 视角鲁棒性显著下降

结论：三个组件**缺一不可**，foundational 空间推理数据是泛化的关键。

### Figure 6: [[训练任务多样性]] 的 Scaling Law

![Figure 6](https://arxiv.org/html/2603.28730v2/x3.png)

**说明**：横轴 = 训练任务种类数，纵轴 = 未见任务平均成功率。明显的 log-linear 趋势——**任务多样性比单任务数据量更重要**。这与 [[OXE]] / [[π₀]] 的"diversity over quantity"结论一致。

### Table 1: 失败模式分布（100 次 rollout 人工标注）

| Error Mode | SOLE-R1 | GPT-5 | Gemini-3-Pro | Gemini Robotics-ER 1.5 |
|---|---:|---:|---:|---:|
| Temporal under-detection（时序欠检测） | 34% | 14% | 17% | 19% |
| Ambiguous object state（物体状态歧义） | 29% | 11% | 13% | 15% |
| Goal-consistent appearance reliance（依赖目标外观） | 15% | 18% | 20% | 21% |
| Perceptual hallucination（感知幻觉） | 9% | 42% | 38% | 35% |

**说明**：
- SOLE-R1 失败集中在"时序欠检测"和"物体状态歧义"——属于**信号不足型**失败，对 RL 不致命，只是收敛慢
- baseline VLM 失败 35-42% 是**感知幻觉**——策略可以利用这种幻觉做出"画面像但没做对"的动作，直接破坏 RL

### Table 2: DrQv2 关键超参数（节选）

| 环境 | Buffer Size | Batch Size | Actor LR | Critic LR | Discount $\gamma$ |
|---|---:|---:|---:|---:|---:|
| RoboSuite | 1M | 256 | 1e-4 | 1e-4 | 0.99 |
| ManiSkill | 1M | 256 | 1e-4 | 1e-4 | 0.99 |
| Meta-World | 1M | 256 | 1e-4 | 1e-4 | 0.99 |
| LIBERO | 1M | 256 | 1e-4 | 1e-4 | 0.99 |
| Real Franka | 0.5M | 128 | 1e-4 | 1e-4 | 0.98 |

**说明**：跨环境基本同构，验证 SOLE-R1 不需要"逐环境调参"。

### Table 3: 任务自然语言描述（节选示例）

| Task | Description |
|---|---|
| `pick_cube` | "Pick up the red cube on the table." |
| `stack_blocks` | "Stack the blue block on top of the green block." |
| `open_drawer` | "Open the top drawer of the cabinet." |
| `pour_water` | "Pour water from the cup into the bowl." |

**说明**：所有任务都用单句自然语言描述，无任何状态信息或子目标拆解。

### Tables 6-7: 离线 Benchmark 对比

| Benchmark | SOLE-R1 | GVL | 最强 baseline |
|---|---:|---:|---:|
| SpatialBench (3D 关系) | **best** | - | spatial-VLM |
| SSRBench (空间推理) | **best** | - | SSR baseline |
| OXE Value-Order-Correlation (Bridge) | **0.78** | 0.62 | - |
| OXE VOC (RT-1) | **0.81** | 0.65 | - |
| OXE VOC (Dobb-E) | **0.75** | 0.58 | - |

**说明**：SOLE-R1 不仅在线 RL 强，离线评估（[[Value-Order-Correlation|VOC]]）也胜过 [[GVL]]，尤其在固定相机的高质量数据集（[[BridgeV2|Bridge]], RT-1, Dobb-E）上。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[SSR-CoT]] | 1.2M | image-depth-QA-rationale 元组 | SFT 空间推理 |
| [[RoboVQA]] / [[Spot-the-Difference]] | ~1M | 多帧时序 QA | SFT 时序推理 |
| 自合成机器人进度数据 | 1.2M traces / 41K videos | [[非专家轨迹合成]] + [[时间反转]] | SFT + RLVR |
| **总计** | 10M 帧 / 4M CoT 轨迹 | 三类混合 | SFT |
| [[OXE]] (Bridge, RT-1, Dobb-E) | - | 真实操作演示 | 离线 VOC 评估 |

### 仿真环境

- [[RoboSuite]]、[[ManiSkill]]、[[Meta-World|Metaworld]]、[[LIBERO]]：共 40 个任务，覆盖抓取、堆叠、开抽屉、推、倒、滑动、开杠杆等
- 跨 embodiment：Franka、Sawyer、WidowX、Fetch
- 跨视角：多种相机角度

### 真实硬件

- 改装的 [[Franka 研究臂|Franka]] 机械臂 + 夹爪
- 桌面操作任务集，wrist + external 双目相机

### 实现细节

- **Backbone**：[[Qwen3-VL]]-8B-Instruct full fine-tuning
- **SFT**：10M 帧 / 4M CoT 轨迹，标准 cross-entropy
- **RLVR**：[[GRPO]] with rule-based reward $r = r_{\text{format}} + r_{\text{acc}}$
- **下游 RL**：[[DrQv2]] off-policy actor-critic
- **观测**：wrist + external RGB + proprioception
- **动作**：end-effector delta motion + gripper control

---

## 批判性思考

### 优点

1. **范式干净**：把"奖励工程"问题压缩到"训一个会推理的视频 VLM"上，下游任意 RL 算法都能用。
2. **CoT 抗 reward hacking 的实证扎实**：Figure 4 的散点图是非常 convincing 的可视化，把 hacking 失败和信号不足失败定量分开。
3. **数据合成 pipeline 可复用**：[[非专家轨迹合成]] 和[[时间反转]]是普适技巧，对任何需要"区分专家 vs. 非专家"的 reward model 都有借鉴价值。
4. **真实硬件验证**：不只是仿真，在真实 [[Franka 研究臂|Franka]] 上也跑通了纯 VLM 奖励驱动的 RL，这是 zero-shot reward learning 领域少见的。
5. **scaling law 明确**：Figure 6 给出了任务多样性的 log-linear scaling，对后续工作有方向指引。

### 局限性

1. **推理延迟 vs. 控制频率**：8B VLM 即便量化推理，每一步几十到几百毫秒，难以匹配 20-50 Hz 的控制环。论文用线性插值 hack，但本质上是稀疏奖励 + 插值，不是真正的密集推理。
2. **CoT 的可解释性是双刃剑**：`<think>` 内容看起来合理但未必反映模型实际依据（CoT faithfulness 问题），可能给"假装在推理"留口子。
3. **依赖 [[非专家轨迹合成]] 的质量**：仿真随机动作注入容易制造**物理不合理**的轨迹，模型可能学到"非真实物理 = 低进度"而非"真实回退 = 低进度"。
4. **任务族仍较窄**：40 个任务集中在桌面短时序操作，长程任务（多步推理、工具使用）未验证。
5. **CoT 长度成本**：每一步生成几十到上百 token 的推理，训练和推理成本远高于直接出标量的 reward model。

### 潜在改进方向

1. **蒸馏到小模型**：把 8B 的 CoT 能力蒸馏到 1-2B 的快推理模型，匹配真实控制频率；
2. **CoT faithfulness 验证**：训练时显式约束 `<think>` 必须**可执行/可验证**（如把推理转成结构化 predicate）；
3. **长程任务扩展**：把进度建模为分层结构（subgoal-level + step-level），覆盖多步任务；
4. **与 VLA 联合训练**：把 SOLE-R1 当 critic，与 [[π₀]]、[[OpenVLA]] 等策略联合微调，形成 actor-critic VLA。

### 可复现性评估

- [x] 代码开源（https://github.com/philipmit/sole-r1-model）
- [x] 预训练模型（HF: pschro/SOLE-R1-8B）
- [x] 训练数据公开（HF: pschro/sole_training_data）
- [x] 训练细节完整（DrQv2 超参表 + RLVR 配方）

---

## 关联笔记

### 基于

- [[Qwen3-VL]]：backbone
- [[GRPO]]：RL 阶段优化算法
- [[SFT]]：第一阶段训练范式
- [[RLVR]]：奖励组成范式（rule-based verifiable）
- [[SSR-CoT]]：基础空间推理数据格式来源

### 对比

- [[RoboReward]] / [[ReWiND]]：专用机器人奖励模型 baseline
- [[GPT-5]] / [[Gemini-3-Pro]]：通用 VLM-as-Judge baseline，论文证明其严重 reward hacking
- [[GVL]]：离线进度估计的代表方法，被 SOLE-R1 在 VOC 上超越

### 方法相关

- [[时空思维链]]：核心推理范式
- [[非专家轨迹合成]]：数据合成核心技巧
- [[Reward Hacking|奖励攻击]]：本文重点对抗的现象
- [[VLM-as-Judge]]：相关但更广泛的范式
- [[在线强化学习]]：下游应用场景

### 硬件/数据相关

- [[Franka 研究臂]]：真实硬件平台
- [[LIBERO]] / [[Metaworld]] / [[ManiSkill]]：仿真环境
- [[OXE]] / [[BridgeV2|Bridge]]：离线评估数据集

---

## 速查卡片

> [!summary] SOLE-R1
> - **核心**：训一个 VLM 输出"先 think 再打分"的密集进度，作为机器人 RL 的**唯一奖励**
> - **方法**：Qwen3-VL-8B + 时空 CoT + [SFT, GRPO] + 1.2M 合成轨迹（含非专家回退）
> - **结果**：40 任务中 24 个 ≥50% 成功（GPT-5: 7, Gemini-3-Pro: 5），真实 Franka 验证通过
> - **关键洞察**：强制 CoT + 进度回归 + 非专家轨迹三件套，把 reward hacking 转化为可控的 signal-limited 失败
> - **代码**：https://github.com/philipmit/sole-r1-model

---

*笔记创建时间: 2026-05-27*
