---
title: "FineVLA: Fine-Grained Instruction Alignment for Steerable Vision-Language-Action Policies"
method_name: "FineVLA"
authors: [Xintong Hu, Xuhong Huang, Jinyu Zhang, Yutong Yao, Yuchong Sun, Qiuyue Wang, Mingsheng Li, Sicheng Xie, Yitao Liu, Junhao Chen, Yixuan Chen, Yingming Zheng, Shuai Bai, Tao Yu]
year: 2026
venue: arXiv
tags: [vla, fine-grained-instruction, steerable-policy, robot-manipulation, dataset, benchmark]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.27284
created: 2026-05-27
---

# 论文笔记：FineVLA: Fine-Grained Instruction Alignment for Steerable Vision-Language-Action Policies

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | XLang Lab (港大 / 联合实验室) |
| 日期 | May 2026 |
| 项目主页 | https://finevla.xlang.ai/ |
| 代码仓库 | https://github.com/xlang-ai/FineVLA |
| 数据集 | [FineVLA-Data](https://huggingface.co/datasets/xlangai/RoboFine-bench) / RoboFine-Bench |
| 模型 | [RoboFine-VLM-387B-A17B](https://huggingface.co/xlangai/RoboFine-VLM-387B-A17B) |
| 对比基线 | [[OpenVLA-OFT]] / [[GR00T N1.5]] / [[π0.7]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.27284) / [Project](https://finevla.xlang.ai/) |

---

## 一句话总结

> 把"高层目标指令"细化为"执行过程级指令"，并证明 1:1 的细粒度/粗粒度混合监督能让 [[VLA]] 策略既保住成功率又获得真正可被语言引导的细节可控性。

---

## 核心贡献

1. **FineVLA-Tool & FineVLA-Data**: 统一 10 个开源机器人数据集共 972,247 条轨迹，经 [[DTW]] 聚类挑选代表轨迹并人工核验，得到 47,159 条 [[Fine-Grained Instruction|细粒度指令]] 数据，平均指令长度由 9.3 词扩到 96.8 词（10.4× 信息密度）。
2. **RoboFine-Bench**: 500 个保留视频 + 10,816 条 atomic facts + 1,030 个 VQA 题，从 grounding/action/state 三轴评测"机器人视频细粒度理解"，与人评相关系数 0.97+。
3. **RoboFine-VLM**: 基于 Qwen3.5-397B-A17B 微调的机器人专用标注器，VQA 准确率 71.0% 超过 Gemini-3.1-Pro 8.9 pt，可作为后续可扩展的自动标注引擎。
4. **可操控策略 (Steerable Policy)**: 在 [[StarVLA]] 框架下训练 [[OpenVLA-OFT|StarVLA-OFT]] 与 [[GR00T N1.5|StarVLA-GR00T]] 两类策略，系统地研究 7 档 FG:Raw 混合比例，发现 1:2 ~ 1:1 是甜点比例，真机双臂任务从 49.9 → 62.7（满分 100）。

---

## 问题背景

### 要解决的问题

现有机器人轨迹数据集里，语言注释只描述"做什么"（goal-level / coarse instruction，例如 *"pick up the red block"*），却几乎不描述"怎么做"——执行过程中的关键细节如：
- **用哪只手** (active arm)
- **从哪个方向接近** (approach direction)
- **抓取部位** (contact region)
- **姿态/朝向** (pose & orientation)

这导致 [[VLA]] 策略学到的是"对指令大致响应"，而不是"按指令的细节执行"——也就缺乏 [[Steerable Policy|可操控性]]。

### 现有方法的局限

| 类别 | 代表 | 局限 |
|------|------|------|
| 粗粒度 VLA | [[OpenVLA]], [[RT-1]], [[OpenVLA-OFT]], [[π0]] | 指令短、过程信息缺失 |
| 已有的细粒度尝试 | Galaxea, RoboCOIN, RoboInter, STEER, PartInstruct | 数据规模或维度单一，没有统一的 schema 和评测 |
| 视频理解评测 | RoboVQA, RoboBench, HanDyVQA | 偏 high-level QA，没有覆盖"动作可执行细节"的 atomic-level 评测 |

### 本文的动机

作者认为，**指令的"信息密度"才是真正决定可操控性的瓶颈**：只要把过程级语义注入到训练数据里（且与原始 raw 指令按合适比例混合），不用换架构也能拿到显著且可控的提升。因此整个工作围绕三件事：**造高质量细粒度数据 → 建评测 → 训练并系统研究混合比例**。

---

## 方法详解

### 整体框架

FineVLA 构建了一个**闭环**：数据建设 ↔ 视频理解评测 ↔ 可扩展标注 ↔ 策略训练。

- **输入**: 语言指令 $l$（含 raw $l^R$ 和 fine-grained $l^F$）、观测帧序列 $\{o_t\}$、机器人状态 $s_t$
- **Backbone**: Qwen3.5-4B 视觉-语言主干
- **核心模块**: [[Fine-Grained Instruction|细粒度指令]] + [[Action Chunking|动作块]] 预测头
- **输出**: 多步动作块 $a_{t:t+k}$

### 模块 1：FineVLA-Tool 四阶段流水线

#### Stage 1 — 数据汇集与格式转换
- 聚合 10 个开源数据集（[[BridgeV2]]、[[BC-Z]]、[[RT-1]]、[[Galaxea]]、[[RoboMIND]] V1/V2、[[RoboCOIN]]、[[RH20T]]、[[RDT 数据集|RDT]]、[[DROID 数据集|DROID]]）
- 统一转为 [[LeRobot 2.1]] 格式，过滤无效/退化记录
- 共得 **972,247 条轨迹**

#### Stage 2 — 动作状态规范化
- 把所有轨迹的位姿转到**绝对坐标 + 归一化四元数**
- 对 action 与 state 的 [[DTW]] 距离设阈值，剔除越界、损坏的日志

#### Stage 3 — 轨迹聚类与代表采样
- 大量近重复演示（仅速度/位置微差）→ 浪费标注预算
- 用 [[DTW]] 算距离 + 层次聚类（Hierarchical Clustering）选代表
- 972,247 → **47,159 代表轨迹**

#### Stage 4 — 十维细粒度标注
- 先用 [[VLM]] 给出草稿，再由人工核验
- 十维 schema 详见下表

| 维度 | 含义 |
|------|------|
| Active Actor | 哪只手 / 哪个执行器在动 |
| Target Object | 操作对象 |
| Initial Configuration | 初始姿态/位置 |
| Action Sequence | 动作分解顺序 |
| Contact & Approach | 接触点 + 接近方向 |
| Trajectory & Orientation | 路径形状、末端朝向 |
| Body Motion | 躯干/底盘运动 |
| Object Interaction | 物体间相互作用 |
| Final Configuration | 完成态 |
| Failure & Recovery | 失败检测与重试 |

### 模块 2：RoboFine-Bench

- 500 个保留视频，提取出 10,816 条 **atomic facts**
- 生成 1,030 个 VQA 题，归到三轴：
  - **Entity & Scene Grounding**（谁/什么/在哪）
  - **Action & Motion Understanding**（怎么动）
  - **Interaction & State Reasoning**（产生什么状态变化）
- **VQA 轨**：确定性匹配
- **Caption 轨**：由 LLM 对生成 caption 与 ground-truth atomic facts 做逐条比对，输出五种标签：`match / partial / contradiction / omission / hallucination`，再聚合为：
  - **Consistency**（事实一致）
  - **Coverage**（覆盖率）
  - **Anti-Hallucination**（幻觉率倒数）

### 模块 3：RoboFine-VLM 可扩展标注器

- 基模 [[Qwen3.5-MoE|Qwen3.5-397B-A17B]]（397B 总参，约 17B 激活）
- 在 FineVLA-Data 上微调，学会按十维 schema 输出**时序对齐的步骤级 caption**
- 用途：未来在更大轨迹库上**自动**生成细粒度指令，绕开人工瓶颈

### 模块 4：FineVLA-Policy（StarVLA 双架构）

> 作者刻意**不**提出新架构，只切换语言监督，把"语言密度"与"架构选择"解耦。

- **StarVLA-OFT**: 跟 [[OpenVLA-OFT]] 同构——在 VL backbone 上挂一个轻量 MLP 头，读 action token 的隐藏状态，**并行**地用 L1 回归连续动作块
- **StarVLA-GR00T**: 跟 [[GR00T N1.5]] 同构——双系统设计，VL backbone 作 System 2 慢推理，[[DiT|DiT-based]] [[Flow Matching|流匹配]] 模块作 System 1 输出连续动作

### 训练数据混合（关键变量）

- 同一批源轨迹同时生成两个版本：
  - **FG 集**：代表轨迹 × 细粒度指令（RDT 1,287 / AlohaMix 5,872）
  - **Raw 集**：所有轨迹 × 原始 goal 指令（RDT 6,061 / AlohaMix 84,067）
- 测试 7 档 FG:Raw 比例：**Raw-only, 1:4, 1:2, 1:1, 2:1, 4:1, FG-only**

---

## 关键公式

> 论文正文不显式列公式，主要数学过程是 [[DTW]] 距离与混合比 $r$ 的扫频。下两式按附录 A.1.4 + 文本叙述整理。

### 公式 1: [[DTW|动态时间规整距离]]

$$
\mathrm{DTW}(X, Y) = \min_{\pi \in \mathcal{A}(X,Y)} \sum_{(i,j) \in \pi} \| x_i - y_j \|_2
$$

**含义**: 对两条不同长度/速度的轨迹 $X=(x_1,\dots,x_n)$ 与 $Y=(y_1,\dots,y_m)$，找一条单调对齐路径 $\pi$ 使逐点欧式距离之和最小。用于 (i) 过滤 action-state 不一致样本；(ii) 在聚类阶段衡量轨迹相似度。

**符号说明**:
- $\mathcal{A}(X,Y)$: 所有合法对齐路径集合（每步只能 ↑、→、↗）
- $x_i, y_j$: 规范化后的位姿向量（绝对坐标 + 单位四元数拼接）
- $\| \cdot \|_2$: 欧氏范数

### 公式 2: [[OpenVLA-OFT|StarVLA-OFT]] 训练目标

$$
\mathcal{L}_{\text{OFT}} = \mathbb{E}_{(o,l,a)\sim \mathcal{D}_{\text{mix}}} \big\| f_\theta(o, l)_{1:k} - a_{t:t+k} \big\|_1
$$

**含义**: 给定观测 $o$ 与（混合后的）指令 $l$，并行预测 $k$ 步连续动作块，用 [[L1 损失]] 监督。

**符号说明**:
- $\mathcal{D}_{\text{mix}}$: FG 集与 Raw 集按 $r$:$(1-r)$ 抽样得到的混合数据流
- $f_\theta$: 共享 Qwen3.5-4B backbone + MLP action head
- $k$: 动作块长度（chunk size）

### 公式 3: [[Flow Matching|流匹配]] 训练目标（StarVLA-GR00T 用）

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{\tau \sim \mathcal{U}(0,1),\, a \sim \mathcal{D}_{\text{mix}},\, a_\tau} \big\| v_\phi(a_\tau, \tau \mid h(o,l)) - (a - a_0) \big\|_2^2
$$

**含义**: System 2 的隐藏向量 $h(o,l)$ 作为条件，让 DiT 预测从噪声 $a_0$ 到真实动作 $a$ 的瞬时速度场 $v_\phi$。

**符号说明**:
- $\tau \sim \mathcal{U}(0,1)$: 流匹配时间步
- $a_\tau = (1-\tau) a_0 + \tau a$: 线性插值采样点
- $v_\phi$: DiT 速度网络
- $h(o,l)$: 来自 VL backbone 的条件特征

### 公式 4: FG:Raw 混合采样

$$
p(\text{batch sample} \in \mathcal{D}_{\text{FG}}) = \frac{r_{\text{FG}}}{r_{\text{FG}} + r_{\text{Raw}}}
$$

**含义**: 每个 batch 按设定比例 $r_{\text{FG}}:r_{\text{Raw}}$ 从两个池抽样；实验扫描 $r_{\text{FG}}:r_{\text{Raw}} \in \{0{:}1, 1{:}4, 1{:}2, 1{:}1, 2{:}1, 4{:}1, 1{:}0\}$。

---

## 关键图表

### Figure 1: Overview / 整体闭环

![Figure 1](https://arxiv.org/html/2605.27284v1/x2.png)

**说明**: FineVLA 的"动作-指令对齐"闭环：**数据建设**（Tool + Data）→ **视频理解评测**（RoboFine-Bench）→ **可扩展标注**（RoboFine-VLM）→ **可操控策略**（FineVLA-Policy）。四个组件互相喂养，形成自举的细粒度监督生态。

### Figure 2: FineVLA-Tool Pipeline / 四阶段数据流水线

![Figure 2](https://arxiv.org/html/2605.27284v1/x3.png)

**说明**: 异构机器人演示通过四阶段处理变成动作对齐的细粒度指令数据：(1) 格式统一到 [[LeRobot 2.1]]；(2) 绝对坐标 + 四元数规范化 + [[DTW]] 阈值过滤；(3) DTW + 层次聚类抽代表；(4) 十维 schema + VLM 草稿 + 人工核验。

### Figure 3: RoboFine-Bench Overview / 评测体系

![Figure 3](https://arxiv.org/html/2605.27284v1/x4.png)

**说明**: RoboFine-Bench 通过 **VQA 轨**与 **Caption 轨**互补评估细粒度视频理解：VQA 覆盖三轴（grounding / motion / state），Caption 用 LLM-as-Judge 在 atomic facts 上算 consistency / coverage / anti-hallucination。

### Figure 4 (Easy / Hard): Caption 评分与人评相关性

| Easy | Hard |
|------|------|
| ![Figure 4 Easy](https://arxiv.org/html/2605.27284v1/x5.png) | ![Figure 4 Hard](https://arxiv.org/html/2605.27284v1/x6.png) |

**说明**: 10 名人工评分员对 6 个模型在 500 视频上排序，与 benchmark 自动 caption 得分的 Pearson 相关系数在 easy/hard 设置下分别达到 ~0.980 / ~0.970，证明 RoboFine-Bench 与人类偏好高度一致。

### Figure 5: Paired Real-World Evaluation / 真机配对评测

![Figure 5](https://arxiv.org/html/2605.27284v1/x7.png)

**说明**: 每列固定同一视觉场景，仅改变指令中的一个**控制因子**（Arm / Color / Approach / Pose / Rotate），观察策略是否真的"听话"。这是论文核心实证之一：FG:Raw=1:1 在 Pose (+23)、Color (+18)、Approach (+18) 上对 Raw-only 的提升尤为巨大。

### Figure 6: RoboTwin 混合比例曲线 / 倒 U 形

![Figure 6](https://arxiv.org/html/2605.27284v1/x8.png)

**说明**: 7 档 FG:Raw 在 [[RoboTwin]] 上呈现一致的**倒 U 形**：纯 Raw 与纯 FG 都不是最优，1:2 ~ 1:1 是甜点。这也说明 raw 指令与 fine-grained 指令是**互补**的（5.2 节）。

### Table 1: FineVLA-Data 数据规模

| 源数据集 | 轨迹 | 步数 | Coarse 词数 | FG 词数 | 信息密度 |
|---|---:|---:|---:|---:|---:|
| [[BridgeV2]] | 4,958 | 21,554 | 10.1 | 61.7 | 6.1× |
| [[BC-Z]] | 1,513 | 5,313 | 5.2 | 51.2 | 9.8× |
| [[RT-1]] | 5,232 | 22,023 | 6.8 | 61.4 | 9.1× |
| [[Galaxea]] | 2,834 | 18,484 | 4.7 | 219.9 | 47.1× |
| [[RoboMIND]]-V1 | 4,605 | 20,341 | 8.6 | 72.8 | 8.5× |
| [[RoboMIND]]-V2 | 7,119 | 39,166 | 6.6 | 98.8 | 14.9× |
| [[RoboCOIN]] | 8,513 | 43,926 | 16.1 | 122.6 | 7.6× |
| [[RH20T]] | 1,387 | 5,560 | 7.9 | 92.1 | 11.7× |
| [[RDT 数据集|RDT]] | 1,275 | 8,437 | 16.9 | 114.0 | 6.7× |
| [[DROID 数据集|DROID]] | 9,723 | 35,802 | 8.0 | 90.9 | 11.3× |
| **合计** | **47,159** | **220,606** | **9.3** | **96.8** | **10.4×** |

**说明**: Galaxea 的 47× 是因为其原始指令极短（4.7 词），细粒度扩到 220 词。整体看，47K 条样本平均带有 ~97 词的过程级描述，是后续训练的"重燃料"。

### Table 2: RoboFine-Bench VQA 结果

| 模型 | Overall | AA | TO | IC | AS | C&A | T&O | BM | OI | FC | F&R |
|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| GPT-4o | ~56 | … | … | … | … | … | … | … | … | … | … |
| Gemini-3.1-Pro | 62.1 | … | … | … | … | … | … | … | … | … | … |
| Qwen3.5-VL | ~58 | … | … | … | … | … | … | … | … | … | … |
| **RoboFine-VLM** | **71.0** | best | best | best | best | best | best | best | best | best | best |

**说明**: RoboFine-VLM 比最强通用模型 Gemini-3.1-Pro 高 **+8.9 pt**，在十个细粒度维度上几乎全面领先（论文 Table 2 给出每维详值）。

### Table 3: RoboFine-Bench Caption 结果（%）

| 模型 | Easy Overall | Easy Cons. | Easy Cov. | Easy Anti-Hall. | Hard Overall |
|------|:-:|:-:|:-:|:-:|:-:|
| Gemini-3.1-Pro | ~80 | … | … | … | ~78 |
| **RoboFine-VLM** | **85.2** | best | best | best | **83.6** |

**说明**: caption 轨上同样大幅领先；hard 设置下幻觉率（Anti-Hallucination 反指）尤其拉开差距。

### Table 4: RoboTwin 仿真成功率（%）

| 架构 | 设置 | Raw-only | 1:4 | 1:2 | 1:1 | 2:1 | 4:1 | FG-only |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| RDT-OFT | Easy / Hard | base / base | … | … | … | … | … | +1.4 / +2.0 |
| RDT-GR00T | Easy / Hard | base / base | … | peak | peak | … | … | +7.0 / +8.1 |
| **AlohaMix-OFT** | **Easy / Hard** | 71.8 / 71.4 | … | … | **86.8 / 82.5** | … | … | +6.5 / +4.7 |

**说明**: 最佳点 (AlohaMix-OFT, FG:Raw=1:1) 比 Raw-only 提升 **+15.0 / +11.1 pt**。所有架构都呈倒 U 形，证明甜点比例是 1:2 ~ 1:1。

### Table 5: 真机评分（100 分制）

| 混合 | 7 个 In-domain 平均 | OOD Probe |
|---|:-:|:-:|
| Raw-only | 49.9 | 0 |
| FG:Raw = 1:2 | … | … |
| **FG:Raw = 1:1** | **62.7** | 10 |
| FG-only | … | … |

**说明**: 1:1 混合在真实双臂场景下把平均分从 49.9 推到 62.7；OOD（actor-target 绑定外推）依然只有 10/100，是当前最大短板（见 Limitations）。

### 因子级提升一览（5.4 节）

| 控制因子 | FG:Raw=1:1 提升 (pt) |
|---|:-:|
| Pose | **+23** |
| Color | **+18** |
| Approach | **+18** |
| Rotate | +10 |
| Arm | +4 |

**说明**: 提升最大的恰好是那些"在轨迹序列上**不直接可见**、必须靠语言指明"的因子——这正是细粒度监督的最大价值所在。

---

## 实验

### 数据集与环境

| 名称 | 类型 | 用途 |
|------|------|------|
| FineVLA-Data | 47,159 条 FG 标注轨迹 | 训练 RoboFine-VLM + 策略 |
| RoboFine-Bench | 500 视频 + 10,816 facts + 1,030 QA | 视频理解评测 |
| [[RoboTwin]] | 仿真 | 双臂操作策略评测（Easy/Hard） |
| 真机双臂平台 | 7 in-domain 任务 + OOD probe | Steerability 评测 |

### 实现细节

- **VL backbone**: Qwen3.5-4B（策略）/ Qwen3.5-397B-A17B（标注器）
- **动作头**: OFT 用 MLP + L1；GR00T 用 DiT + 流匹配
- **指令混合**: 同 batch 内按 7 档比例采样
- **评测**: VQA 用确定性匹配；caption 用 LLM-as-Judge 在 atomic facts 上对齐

### 关键结论（5.1 ~ 5.4）

1. **5.1 不损失目标级成功率**：FG-only 相对 Raw-only 在所有 RoboTwin 设置上都 **≥ +1.4 pt**（最高 +8.1）。
2. **5.2 互补性**：呈倒 U 形，1:2 ~ 1:1 最佳，最高 +15.0 pt（AlohaMix-OFT Easy）。
3. **5.3 架构 & 数据规模**：FG 监督**缩小 OFT 与 GR00T 的架构差距**，并且在数据更大时收益更显著。
4. **5.4 因子级可控**：Pose +23、Color +18、Approach +18、Rotate +10、Arm +4。

---

## 批判性思考

### 优点

1. **问题切得准**：抓住"语言信息密度"这一个被忽视的轴，做出"换数据不换架构"的清洁对照实验。
2. **基础设施完整**：Tool（造数据）+ Bench（评测）+ VLM（自动标注）+ Policy（应用），形成自闭环，复现门槛低。
3. **结论可操作**：1:2 ~ 1:1 是简单可迁移的训练 recipe，下次任何人训 VLA 都可以直接抄。
4. **真机验证扎实**：不仅做仿真，还把"配对场景只换一个因子"的实验形式带回来，这才是衡量 steerability 的正确姿势。

### 局限性

1. **OOD actor-target 绑定**仍然弱（10/100），说明语言条件在分布外组合下并未真正泛化，仍走数据记忆。
2. **十维 schema 偏 manipulation**，对 [[足式运动|locomotion]]、双手协同、长程任务覆盖不足。
3. **人工核验 47K 条**是工程级成本，论文虽提供 RoboFine-VLM 作为后续自动标注器，但闭环上线后是否会引入分布偏移仍未验证。
4. **没有公布 raw 指令的"信息密度"消融**——是否同样长度的 raw 指令（凑词数）也能起部分作用？这个对照缺失。

### 潜在改进方向

1. 把十维 schema 拓展为**任务自适应** schema（manipulation / locomotion / mobile 分别有不同子维度）。
2. 在 OOD 上引入**组合泛化**正则（如 actor 与 target 的解耦表征）。
3. 把"细粒度"做成**多粒度可控生成**：推理时按 plan 长度切换 raw ↔ FG。
4. 与 [[Memory in VLA|VLA 记忆]] 结合：长程任务中，FG 指令可以作为"中间记忆 token"显式存取。

### 可复现性评估

- [x] 代码开源（[GitHub](https://github.com/xlang-ai/FineVLA)）
- [x] 预训练模型（[HF Model](https://huggingface.co/xlangai/RoboFine-VLM-387B-A17B)）
- [x] 训练细节完整（附录提供超参与 DTW 阈值）
- [x] 数据集可获取（[HF Dataset](https://huggingface.co/datasets/xlangai/RoboFine-bench)）

---

## 关联笔记

### 基于
- [[StarVLA]]: 实验框架同源
- [[OpenVLA-OFT]]: StarVLA-OFT 动作头同构来源
- [[GR00T N1.5]]: StarVLA-GR00T 双系统同构来源
- [[Qwen3.5-MoE]]: VLM 标注器基模

### 对比 / 相关数据
- [[VLA]]: 上位概念
- [[Fine-Grained Instruction]]: 核心概念
- [[Steerable Policy]]: 核心目标
- [[OpenVLA]] / [[RT-1]] / [[π0]]: 粗指令 VLA 代表

### 方法相关
- [[DTW]]: 轨迹相似度与过滤
- [[Action Chunking]]: 输出形式
- [[Flow Matching]]: GR00T 系统 1 采样
- [[VLM]]: 用于自动标注

### 数据与硬件
- [[RoboTwin]]: 仿真评测
- [[BridgeV2]] / [[BC-Z]] / [[RT-1]] / [[RoboMIND]] / [[DROID 数据集|DROID]]: 数据来源
- [[LeRobot 2.1]]: 统一格式

---

## 速查卡片

> [!summary] FineVLA: Fine-Grained Instruction Alignment for Steerable VLA Policies
> - **核心**: 用十维细粒度指令把语言信息密度做大 10×，1:1 混合监督让策略既不掉成功率又能被语言精细驱动
> - **方法**: FineVLA-Tool（DTW 聚类 + 人工核验）→ RoboFine-Bench（500 视频 / 10K facts）→ RoboFine-VLM（397B MoE 自动标注）→ FineVLA-Policy（OFT + GR00T，7 档 FG:Raw 扫频）
> - **结果**: RoboTwin AlohaMix-OFT 86.8/82.5（+15.0/+11.1），真机双臂 62.7 vs 49.9，Pose +23 / Color +18 / Approach +18
> - **代码**: https://github.com/xlang-ai/FineVLA

---

*笔记创建时间: 2026-05-27*
