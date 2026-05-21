---
title: "AR-VLA: True Autoregressive Action Expert for Vision-Language-Action Models"
method_name: "AR-VLA"
authors: [Yutong Hu, Jan-Nico Zaech, Nikolay Nikolov, Yuanqi Yao, Sombit Dey, Giuliano Albanese, Renaud Detry, Luc Van Gool, Danda Paudel]
year: 2026
venue: RSS 2026
tags: [vision-language-action, autoregressive-policy, kv-cache, temporal-memory, rope, asynchronous-inference, robot-manipulation]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2603.10126v2
created: 2026-05-19
---

# 论文笔记：AR-VLA: True Autoregressive Action Expert for Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | INSAIT, Sofia University "St. Kliment Ohridski" + ETH Zürich + KU Leuven |
| 日期 | May 2026 |
| 项目主页 | https://arvla.insait.ai |
| 对比基线 | [[OpenVLA]], [[Pi05\|π₀.₅]], [[CoT-VLA\|CogACT]], [[Diffusion Policy]], [[MemoryVLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2603.10126) / [Code](https://github.com/insait-institute/AR-VLA-lerobot) / [Models](https://hf.co/collections/INSAIT-Institute/ar-vla) |

---

## 一句话总结

> 把动作生成从"反应式快照"升级为"真正跨时间的[[自回归动作专家]]"，用[[Hybrid KV Cache]] 和 [[Dynamic Temporal Re-anchoring]] 解耦感知与控制的频率。

---

## 核心贡献

1. **真正的跨时间[[自回归动作专家]]**：与现有 [[VLA]] 仅在单步内自回归不同，AR-VLA 在时间维度持续自回归，动作 $a_t$ 显式依赖于完整历史 $a_{<t}, s_{<t}$，消除 [[Markovian Amnesia\|马尔可夫遗忘]]。
2. **[[Hybrid KV Cache\|Hybrid KV (HKV) Cache]]**：为快频动作 Token（滚动 FIFO）与慢频 [[VLM\|视觉-语言]] Token（可刷新单槽缓冲）设计两条异构 KV 流，让感知线程与控制线程解耦运行。
3. **[[Dynamic Temporal Re-anchoring]] (DTR)**：利用 [[RoPE]] 的相对位置不变性，让短序列（25 步）训练得到的模型可以泛化到长序列（500 步）推理。
4. **两阶段训练 + [[Stochastic History Masking\|历史随机掩码]]**：先在大规模运动学数据上预训练动作专家，再做 VL-动作对齐；0.6 mask 率最优，解决"[[Causal Confusion Paradox\|因果混淆悖论]]"。

---

## 问题背景

### 要解决的问题

如何让 [[VLA]] 模型在跨时间步上保持真正的[[历史条件化策略\|历史条件化]]，从而生成时序平滑且可恢复的动作轨迹？

### 现有方法的局限

1. **"伪自回归"**：[[OpenVLA]]、[[RT-2]] 等虽然名义上自回归，但只在单帧内对 token 自回归，跨时间步是独立的快照决策，丢失时序记忆；
2. **[[Action Chunking]] 是分段反应**：[[π₀]]、[[CogAct]] 用 [[Flow Matching]] / 扩散生成动作块，每次重新从观测开始决策，块与块边界存在抖动（[[Inter-Chunk Consistency\|块间不一致]]）；
3. **额外记忆模块成本高**：[[MemoryVLA]]、HAMLET 给 VLA 加外挂记忆模块，但动作专家本身仍是反应式的；
4. **训练-推理时序错配**：训练时用短序列、推理时长序列，绝对位置编码漂移导致模型无法外推。

### 本文的动机

把 VLA 拆成两类"觉知"：
- **情境觉知 (Situational Awareness)**：由冻结的 [[VLM]] 提供语义理解；
- **时序觉知 (Temporal Awareness)**：由自回归动作专家维护的运动学历史。

让两者通过**异步、解耦的 KV 缓存**桥接，控制频率不再被感知频率拖慢。

---

## 方法详解

### 模型架构

AR-VLA 采用 **冻结 [[VLM]] backbone + 独立[[自回归动作专家]]** 架构：

- **输入**：语言指令 $l$ + 多视角视觉 $v_t$ + 本体感觉 $s_t$
- **VLM Backbone**：[[PaliGemma|PaliGemma-3B]]（[[SigLIP]]-So400m 视觉编码器 + [[Gemma]]-2B 语言模型），全程冻结
- **Action Expert**：18 层 Transformer 解码器，隐藏维 1024，约 300M 参数
- **核心模块**：
  - [[Hybrid KV Cache]]：动作流滚动 FIFO + VL 流单槽刷新
  - [[Dynamic Temporal Re-anchoring]]：基于 [[RoPE]] 的相对位置对齐
- **输出**：单步连续动作 token $a_t \in \mathbb{R}^d$（线性投影到 $\mathbb{R}^D$）

### 核心模块

#### 模块 1：连续动作 Token 化

**设计动机**：避免 [[VQ-VAE]] 离散化损失，直接把动作向量当作 Transformer 的一个 token。

**具体实现**：
- 每个时间步动作 $a_t \in \mathbb{R}^d$ 通过线性层投影到模型维度 $D = 1024$；
- 一个 token 对应一个时间步（不是动作块），从而实现严格的逐步自回归。

#### 模块 2：[[Hybrid KV Cache]]（HKV）

**设计动机**：动作的更新频率（~30Hz）远高于 VLM 的推理频率（~14Hz），用统一缓存会被慢端拖累。

**具体实现**：
- **本体感觉/动作流 $\mathbf{KV}^X$**：滚动 [[FIFO]] 缓冲，长度 $H=20$；每步生成动作后 push KV，旧 KV pop；
- **视觉-语言流 $\mathbf{KV}^{VL}$**：单槽**可刷新**缓冲，只保留 VLM 最新一次推理的整段 KV，旧 VL 块被整体替换。

#### 模块 3：[[Dynamic Temporal Re-anchoring]]（DTR）

**设计动机**：训练时 batch 序列很短（25 步），但部署时要 rollout 几百步；如果 VL token 用绝对位置编码 $0$，动作 token 增长到 $500$，两者位置差 $\Delta t$ 远超训练分布。

**具体实现**：利用 [[RoPE]] 的**相对位置不变性**，让 VL prefix 的位置动态锚定到 "刚被刷新时的当前 timestep"，使得相对偏移 $\Delta t = m - n$ 保持训练-推理一致：

$$
\text{Score}(q_{m+T}, k_{n+T}^{VL}) = \text{Score}(q_m, k_n^{VL})
$$

训练时 $(m=25, n=20, \Delta t = 5)$ 与推理时 $(m=500, n=495, \Delta t = 5)$ 等价。

#### 模块 4：双线程异步推理

**Action Thread（高频）**：用 $\mathbf{KV}^X + \mathbf{KV}^{VL}$ 自回归生成动作，每 ~29ms 一步；
**Perception Thread（低频）**：以 VLM 原生频率（~70ms / 帧）处理图像，把新 VL KV 整段刷入 $\mathbf{KV}^{VL}$。

二者解耦后控制频率不再随感知延迟下降。

---

## 关键公式

### 公式 1：[[反应式策略|Reactive Actor]] 分布

$$
P_{\text{react}}(\tau) = \prod_{t=1}^{T} P\big(a_t \mid \Phi(v_t, l),\, s_t\big)
$$

**含义**：现有 VLA 默认建模——每步动作只依赖当前观测，与 $a_{<t}$ 独立，呈现 Markov 性质。

**符号说明**：
- $\tau$：完整动作轨迹
- $\Phi(\cdot)$：感知编码器（VLM）
- $v_t, l, s_t$：视觉、语言、本体感觉

### 公式 2：[[自回归动作专家|Autoregressive Actor]] 分布

$$
P_{\text{ar}}(\tau) = \prod_{t=1}^{T} P\bigg(a_t \,\bigg|\, \underbrace{\Phi(v_i, l)}_{\text{VL Prefix}},\, \underbrace{a_{<t}, s_{<t}}_{\text{Kinematic History}}\bigg)
$$

**含义**：AR-VLA 提出的建模方式，动作显式依赖**整段历史**和**最新 VL 前缀**（$i \le t$ 是最近处理的视觉帧）。

**符号说明**：
- $i \le t$：感知线程最近一次刷新的时间步索引
- $a_{<t}, s_{<t}$：截止当前的所有历史动作与状态
- 与公式 1 的本质差异是 $a_{<t}$ 出现在条件中

### 公式 3：带 [[RoPE]] 的注意力计算

$$
\text{Attn}(q_m, \mathbf{K}, \mathbf{V}) = \sum_n \text{softmax}\left(\frac{\text{Score}(q_m, k_n)}{\sqrt{d}}\right) v_n
$$

其中 RoPE 应用旋转矩阵：

$$
k_n = \mathbf{R}(n)\,\mathbf{k}, \quad v_n = \mathbf{v}
$$

**含义**：标准注意力公式，关键在于位置信息通过旋转矩阵 $\mathbf{R}(n)$ 注入到 key。

**符号说明**：
- $q_m$：查询 token，位置为 $m$
- $k_n, v_n$：位置 $n$ 的 key/value
- $\mathbf{R}(n)$：RoPE 的旋转矩阵

### 公式 4：DTR 的[[位移不变性|Shift-Invariance]] 性质

$$
\text{Score}(q_{m+T},\, k_{n+T}^{VL}) = \text{Score}(q_m,\, k_n^{VL})
$$

**含义**：只要相对偏移 $\Delta t = m - n$ 不变，注意力分数就不变；这是把短训练序列推广到长推理序列的理论基础。

**符号说明**：
- $T$：任意时间偏移
- $k_n^{VL}$：位置为 $n$ 的 VL token 的 key

### 公式 5：[[自回归动作专家|Phase 1]] 预训练损失

$$
\mathcal{L}_{\text{Phase1}} = \sum_{t=1}^{T} \mathcal{L}\big(x_t \mid x_{<t}\big)
$$

**含义**：纯动作自回归预训练，让模型学习**运动学句法**（kinematic syntax），与 VL 信息无关。

**符号说明**：
- $x_t$：第 $t$ 步动作 token
- $\mathcal{L}(\cdot)$：回归损失（translation/rotation 用 $\lambda=1.0$，gripper $\lambda=0.1$）

### 公式 6：Phase 2 VL-动作对齐（带 [[Stochastic History Masking|历史随机掩码]]）

$$
\mathcal{L}_{\text{Phase2}} = \sum_{k=0}^{M-1} \mathcal{L}\bigg(x_{H+k} \,\bigg|\, \mathcal{M}_k \odot \mathbf{x}_{\text{past}},\, \Phi(v_H, l_H),\, \mathbf{x}_{H:H+k-1}\bigg)
$$

**含义**：Priming（历史填入索引 $\{0, \dots, H-1\}$）+ Anchoring（VL 锚到索引 $H$）+ Stochastic Supervision（用随机掩码破坏历史，强制模型注意 VL prefix）。

**符号说明**：
- $H$：历史长度
- $M$：要预测的未来动作数量
- $\mathcal{M}_k \in \{0,1\}^H$：每 token 独立的二元 mask（最优 mask rate = 0.6）
- $\odot$：逐元素乘法

### 公式 7：推理时的条件下一 token 预测

$$
P(x_t \mid x_{<t},\, \Phi(v_i, l_i))
$$

**含义**：运行时把模型当作"在最新 VL 上下文条件下的下一 token 预测器"，每步只新增一个动作 token。

---

## 关键图表

### Figure 1: Performance Overview / 性能总览

![Figure 1](https://arxiv.org/html/2603.10126v2/x1.png)

**说明**：三联图——(a) 在 generalist (SIMPLER) 与 specialist (PushT/ALOHA) 基准上 AR-VLA 全面占优；(b) 轨迹平滑度对比，AR-VLA 的关节状态曲线明显比 [[OpenVLA]] 和 [[Flow Matching]] 基线更光滑；(c) 在 PushT2、Stack3 这类需要长程记忆的任务上，反应式基线失败而 AR-VLA 成功。

### Figure 2: AR-VLA Framework / 系统框架

![Figure 2](https://arxiv.org/html/2603.10126v2/x2.png)

**说明**：核心架构图。绿色 token 是 VL prefix（可刷新），橙色 token 是动作流（滚动 FIFO）。两条流通过 [[Dynamic Temporal Re-anchoring]] 在 [[RoPE]] 坐标系下对齐，VLM 和 Action Expert 异步运行。

### Figure 3: Heterogeneous FIFO Update Rules / 异构 FIFO 更新规则

![Figure 3](https://arxiv.org/html/2603.10126v2/x3.png)

**说明**：详细解释 [[Hybrid KV Cache]] 的双队列策略——VL 流是 block-wise 整体替换，动作流是 token-wise 单步滚动。

### Figure 4: Simulation Benchmarks Setups / 仿真基准设置

![Figure 4](https://arxiv.org/html/2603.10126v2/x4.png)

**说明**：覆盖 generalist（[[SimplerEnv]] BridgeV2 上 4 个 WidowX 任务）和 specialist（[[Push-T]]、ALOHA cube/insert）的两类基准。

### Figure 5: Real-world WidowX Zero-Shot Performance / 真机零样本结果

![Figure 5](https://arxiv.org/html/2603.10126v2/x5.png)

**说明**：AR-VLA 在 WidowX 真机零样本任务上达到 89% 平均成功率，"cup on plate" 和 "Lobster" 任务 100%。

### Figure 6: Smoothness Visualization / 平滑度可视化

![Figure 6](https://arxiv.org/html/2603.10126v2/x6.png)

**说明**：关节状态轨迹的时序对比。AR-VLA 的曲线明显比反应式基线更平滑，[[Jerk]] 指标最低（7.89 vs OpenVLA 10.13）。

### Figure 7: History-Awareness Evaluation / 历史感知评估

![Figure 7](https://arxiv.org/html/2603.10126v2/x7.png)

**说明**：[[Multi-Goal 任务族|PushT2]]（T 形块到达两个目标）和 Stack3（杯子盖电池后再叠杯）的可视化。反应式策略在中途目标不可观测后陷入"震荡"，AR-VLA 借助 KV 缓存的历史记忆保持正确决策。

### Figure 8: Three Different Action Experts / 三种动作专家变体

![Figure 8](https://arxiv.org/html/2603.10126v2/appendix_fig/vla3.png)

**说明**：对比三种共享 backbone 的动作专家——离散 token AR、扩散/Flow Matching、AR-VLA 的连续 AR，凸显 AR-VLA 的独特训练-推理范式。

### Figure 9: AR Actor Architecture / Specialist 架构

![Figure 9](https://arxiv.org/html/2603.10126v2/appendix_fig/art2.png)

**说明**：Specialist 版本（AR-Actor）是 4 层 Transformer 解码器，hidden=512，匹配 [[ACT (Action Chunking Transformer)|ACT]] 的尺寸，共享 ResNet-18 视觉编码器。

### Figure 10: Demonstration Collection / 历史感知任务数据采集

![Figure 10](https://arxiv.org/html/2603.10126v2/appendix_fig/history_collect.png)

**说明**：PushT2 与 Stack3 数据采集方法，故意构造"中途遮挡 / 目标不可观测"的初始布局。

### Figure 11: PushT2 Task Execution / PushT2 执行案例

![Figure 11](https://arxiv.org/html/2603.10126v2/appendix_fig/pusht2.png)

**说明**：两目标推动任务的典型 rollout。AR-VLA 完成第一个目标后能稳定记忆"接下来要去另一个目标"。

### Figure 12: Stack3 Real-world Execution / Stack3 真机执行

![Figure 12](https://arxiv.org/html/2603.10126v2/appendix_fig/stack_real.png)

**说明**：杯子盖住电池后第二只杯子叠上的任务流。基线方法在电池被遮挡后丢失记忆，AR-VLA 通过 KV 历史正确选择叠放方向。

### Figure 13: SIMPLER Simulator Execution / 仿真零样本截图

![Figure 13](https://arxiv.org/html/2603.10126v2/appendix_fig/vla_simpler.png)

**说明**：4 个 SIMPLER 任务（spoon, carrot, block, eggplant）的零样本执行截图。

### Figure 14: Real-world Execution / 真机零样本截图

![Figure 14](https://arxiv.org/html/2603.10126v2/appendix_fig/vla_real.png)

**说明**：5 个真机任务（eggplant, cup, chess, lobster, corn）的执行序列，包含失败后自动回退重试的鲁棒行为。

### Figure 15: Specialist Task Execution / Specialist 任务执行

![Figure 15](https://arxiv.org/html/2603.10126v2/appendix_fig/specialist.png)

**说明**：AR-Actor 在 [[Push-T]]、ALOHA cube transfer、ALOHA insertion 三个 specialist 任务上的轨迹可视化。

### Table I: BridgeV2 → SIMPLER 零样本成功率（%）

| Method | Spoon | Carrot | Block | Eggplant | Average |
|--------|-------|--------|-------|----------|---------|
| [[OpenVLA]] | 0 | 0 | 0 | 4.1 | 1.0 |
| Octo-Base | 12.5 | 8.3 | 0 | 43.1 | 16.0 |
| Octo-Small | 47.2 | 9.7 | 4.2 | 56.9 | 30.0 |
| SpatialVLA | 16.7 | 25.0 | 29.2 | 100.0 | 42.7 |
| CogACT | 58.3 | 37.5 | 20.8 | 91.7 | 52.1 |
| π₀-Fast* | 62.5 | 29.2 | 20.8 | 83.3 | 49.0 |
| [[Pi05\|π₀.₅]]* | 58.3 | 33.3 | 16.7 | 95.8 | 51.0 |
| **AR-VLA (Ours)** | **75.0** | **54.2** | **20.8** | **95.8** | **61.5** |

**说明**：AR-VLA 平均 61.5%，比第二名 CogACT 高 +9.4%。

### Table II: Specialist Policy Performance

| Method | pushT Max IoU | pushT Success | aloha-cube Script | aloha-cube Human | aloha-insert Script | aloha-insert Human |
|--------|---------------|---------------|-------------------|------------------|---------------------|--------------------|
| [[Diffusion Policy]] | 0.957 | 65.20 | 33.33 | 10.00 | 22.67 | 1.66 |
| [[ACT (Action Chunking Transformer)\|ACT]] | 0.800 | 52.00 | 86.00 | 50.00 | 32.00 | 20.00 |
| **AR (Ours)** | **0.920** | **60.40** | **97.33** | **67.33** | **54.67** | **6.67** |

**说明**：在 specialist 设置下，AR-Actor 在 ALOHA 双臂任务上大幅领先；pushT 上 IoU 仅次于 Diffusion Policy。

### Table III: Inference Smoothness & Latency

| Model | Jerk Avg (10² rad/s³) | Jerk Max | VLM Latency (ms/step) | Action Expert | Total/Act (ms) |
|-------|-----------------------|----------|------------------------|---------------|----------------|
| [[OpenVLA]] | 10.13 | 42.14 | 321.72/1 | - | 321.72 |
| Fast | 8.15 | 80.24 | 744.78/4 | - | 186.20 |
| FM | 9.39 | 45.33 | 69.77/4 | - | 84.26 |
| **AR (Ours)** | **7.89** | **39.83** | **69.56/4** | **28.86/1** | **46.25** |

**说明**：AR-VLA 在感知 70ms 的同时维持 29ms 控制频率，[[Jerk]] 最低（7.89）。

### Table IV: Ablation Study

| Variant | Val. Error | Sim. Success |
|---------|-----------|--------------|
| w/ Phase-1 (Full) | 4.2 | **61.5** |
| w/o Phase-1 (100% time) | 4.5 | 37.5 |
| w/o Phase-1 (200% time) | 4.0 | 54.2 |
| mask rate = 0.0 | 2.7 | **0.0** |
| mask rate = 0.2 | 3.1 | 27.1 |
| mask rate = 0.4 | 3.5 | 47.9 |
| **mask rate = 0.6 (Full)** | 4.2 | **61.5** |
| mask rate = 0.8 | 5.7 | 49.0 |
| mask rate = 1.0 | 2.9 | 28.1 |
| w/ Static Pos. Embedding | 3.0 | 3.1 |
| w/o Pos. Embedding | 2.9 | 29.2 |
| history length = 1 | 6.0 | 36.5 |
| history length = 5 | 5.2 | 50.0 |
| history length = 10 | 4.7 | 59.4 |
| **history length = 20 (Full)** | 4.2 | **61.5** |
| history length = 40 | 4.2 | 59.4 |

**关键发现**：
1. **Phase-1 预训练**带来 ~2× 收敛加速；
2. **0.6 mask rate** 是最优甜点，体现"[[Causal Confusion Paradox\|因果混淆悖论]]"——mask=0 时验证误差最低，但仿真成功率为 0%（模型只学到了 copy 历史）；
3. **[[RoPE]] re-anchoring 至关重要**：固定位置编码退化到 3.1% 成功率；
4. **历史长度 H=20** 是最佳点，继续增大无收益。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[BridgeV2]] | 大规模真机演示 | WidowX 抓取/操作 | Generalist 预训练 |
| [[SimplerEnv]] | 仿真 | WidowX 4 任务 | 仿真评估 |
| [[Push-T]] | 240 demos | 2D 推动 | Specialist 训练/评估 |
| ALOHA cube/insert | LeRobot | 双臂插入/转移 | Specialist 训练/评估 |
| PushT2 (本文) | 240 human demos | 多目标 + 历史依赖 | 历史感知评估 |
| Stack3 (本文) | 16 teleop episodes | 真机遮挡场景 | 历史感知评估 |

### 实现细节

**模型**：
- VLM Backbone：[[PaliGemma|PaliGemma-3B]]（[[SigLIP]]-So400m + [[Gemma]]-2B），全程冻结
- Action Expert：18 层 Transformer 解码器，hidden=1024，MLP=4096，~300M 参数；query=2048D，KV=256D（[[MQA\|多查询/分组查询]]）

**训练**：
- Phase-1（动作预训练）：AdamW，lr=1e-4，batch=1024，20k steps，~2h on A6000
- Phase-2（VL-动作对齐）：lr=5e-5，batch=512，30k steps，warmup=200
- Specialist：lr=1e-5，batch=8，200k steps，weight decay=1e-4
- History 长度：训练 16，测试 20；mask rate=0.6
- Loss 权重：translation=1.0，rotation=1.0，gripper=0.1

**硬件**：单卡 A6000（Phase-1）

### 可视化结果

1. **轨迹平滑**：Figure 6 显示 AR-VLA 关节曲线无可见块边界跳变；
2. **失败恢复**：真机 rollout 中，初次抓取失败后 AR-VLA 会主动抬升末端执行器并二次尝试（temporal awareness 的体现）；
3. **长程记忆**：在 PushT2 中，反应式策略到达第一个目标后会在两个目标间震荡，AR-VLA 直奔第二目标。

---

## 批判性思考

### 优点

1. **理论清晰**：把 VLA 的"自回归"概念从 token 维度升维到时间维度，定义干净（公式 1 vs 公式 2）；
2. **工程巧妙**：[[Hybrid KV Cache]] + [[Dynamic Temporal Re-anchoring]] 让感知与控制解耦运行，控制频率不再被 VLM 拖慢；
3. **泛化机制有理论支撑**：DTR 基于 RoPE 的位移不变性证明，可以从短训练序列外推到长推理序列；
4. **失败-恢复行为**：真机 rollout 中观察到自主重试，验证时序觉知的实用价值；
5. **消融充分**：Table IV 把 Phase-1、mask rate、位置编码、历史长度都做了完整 ablation，"因果混淆悖论"是有意思的发现。

### 局限性

1. **OOD 累积误差**：AR 范式天然怕分布外状态，错误会被编码进 KV 形成正反馈环。论文承认仅靠 [[Stochastic History Masking]] 做了部分缓解，未来需要 uncertainty-aware sampling；
2. **梯度耦合问题**：动作目标的梯度无法改善 VLM 的语义能力，所以只能冻结 VLM；这意味着 VL 表征与动作生成无法联合优化；
3. **模块化 vs 整合**：当前是动作专家与语言 token 分离的 modular 设计；如果做成 fully integrated（动作 token 与语言 token 在同一因果流），可能有更强的统一性；
4. **真机评估规模较小**：Stack3 只有 16 个 episode，统计意义有限。

### 潜在改进方向

1. **不确定性感知采样**：在 KV cache 上引入 uncertainty 估计，对低置信度的历史动作降权或重新生成；
2. **轻量 [[LoRA]] fine-tune VLM**：解决"动作梯度不能改善 VLM"的问题，又不破坏冻结的稳定性；
3. **与 [[Diffusion Policy]] / [[Flow Matching]] 融合**：在每个 AR step 内部用 flow head 采样动作分布，结合 AR 的时序连贯与 diffusion 的多模态；
4. **跨任务记忆迁移**：当前 KV cache 仅维护单次 rollout 内的历史，可以扩展为跨 episode 的长期记忆（类似 [[MemoryVLA]]）。

### 可复现性评估

- [x] 代码开源（GitHub: insait-institute/AR-VLA-lerobot）
- [x] 预训练模型（Hugging Face: INSAIT-Institute/ar-vla）
- [x] 训练细节完整（Appendix V-A/V-B 给出全部超参）
- [x] 数据集可获取（[[BridgeV2]]、[[SimplerEnv]] 公开，PushT2 同源公开）

---

## 关联笔记

### 基于

- [[OpenVLA]]：作为对比基线和"伪自回归"的反面教材
- [[Pi05|π₀.₅]] / [[π₀]]：[[Action Chunking]] + [[Flow Matching]] 路线的代表
- [[RoPE]]：DTR 的理论基础
- [[PaliGemma]]：VLM backbone

### 对比

- [[MemoryVLA]]：同样关注时序记忆，但是给反应式策略加挂记忆模块；AR-VLA 把"记忆"内嵌进自回归动作专家本身
- [[CoT-VLA|CogACT]]：用扩散动作头的同期 SOTA，被 AR-VLA 超越 +9.4%
- [[Diffusion Policy]]：specialist 任务的强基线
- [[ACT (Action Chunking Transformer)|ACT]]：与 AR-Actor 同尺寸的对比

### 方法相关

- [[自回归动作专家]]：本论文核心概念
- [[Hybrid KV Cache]]：本论文提出
- [[Dynamic Temporal Re-anchoring]]：本论文提出
- [[Stochastic History Masking]]：本论文提出
- [[Causal Confusion Paradox]]：本论文揭示
- [[Markovian Amnesia]]：本论文批评的现象
- [[Jerk]]：关键平滑度指标

### 硬件/数据相关

- [[BridgeV2]]：Generalist 预训练数据
- [[SimplerEnv]]：仿真评估
- [[Push-T]]：Specialist 评估
- WidowX：真机平台

---

## 速查卡片

> [!summary] AR-VLA: True Autoregressive Action Expert
> - **核心**：把 VLA 从"反应式快照"升级为"跨时间自回归"，用 Hybrid KV Cache + RoPE 位移不变性实现感知-控制异步解耦
> - **方法**：冻结 PaliGemma + 18 层连续动作 AR 专家；HKV（动作 FIFO + VL 单槽刷新）+ DTR（RoPE re-anchoring）+ 两阶段训练 + 0.6 mask rate
> - **结果**：SIMPLER 61.5%（+9.4% vs CogACT），WidowX 真机 89%，jerk 7.89（最低），控制延迟 29ms
> - **代码**：https://github.com/insait-institute/AR-VLA-lerobot

---

*笔记创建时间: 2026-05-19*
