---
title: "WBench: A Comprehensive Multi-turn Benchmark for Interactive Video World Model Evaluation"
method_name: "WBench"
authors: [Kaining Ying, Hengrui Hu, Siyu Ren, Jiamu Li, Fengjiao Chen, Ziwen Wang, Xuezhi Cao, Xunliang Cai, Henghui Ding]
year: 2026
venue: arXiv
tags: [benchmark, interactive-world-model, video-generation, multi-turn-evaluation, vlm-as-judge, navigation]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.25874v1
created: 2026-05-26
---

# 论文笔记：WBench: A Comprehensive Multi-turn Benchmark for Interactive Video World Model Evaluation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Meituan LongCat Team, Fudan University |
| 日期 | May 2026 |
| 项目主页 | https://meituan-longcat.github.io/WBench |
| 代码仓库 | https://github.com/meituan-longcat/WBench |
| 数据集 | https://huggingface.co/datasets/meituan-longcat/WBench |
| 对比基线 | [[VBench]], [[WorldArena]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.25874) / [HTML](https://arxiv.org/html/2605.25874v1) / [Code](https://github.com/meituan-longcat/WBench) |

---

## 一句话总结

> WBench 是首个为 [[交互式世界模型]] 设计的多轮评测基准，用 5 维 22 指标 + 289 用例 / 1058 轮交互系统性诊断 20 个 SOTA 视频世界模型的能力短板。

---

## 核心贡献

1. **统一多轮基准**: 跨 [[文本驱动视频生成|文本驱动]]、[[Camera-Controlled 视频生成|相机控制]]、[[Action-Conditioned World Model|动作条件]] 三类范式提供统一的导航接口（文本 / 6-DoF pose / 离散动作互转），首次实现公平的跨范式对比。
2. **五维 22 指标评测体系**: 覆盖 Video Quality、Setting Adherence、Interaction Adherence、Consistency、Physical Compliance 五大维度，结合传统视觉模型与 [[VLM-as-Judge|VLM 作为评测器]]，Spearman ρ≥0.94 与人类偏好一致。
3. **多轮退化诊断**: 揭示 20 个 SOTA 模型在多轮交互下的系统性退化，Navigation 从第 1 轮到第 4 轮跌 33 分，远高于语义交互的退化幅度，首次量化"长视界世界模型短板"。

---

## 问题背景

### 要解决的问题

当前 [[世界模型]] / [[视频生成]] 评测存在三大缺口：

1. **单轮静态评测**: [[VBench]] 类基准只测文本→视频的单次生成，无法评测**多轮交互能力**（连续导航 / 动作 / 事件编辑 / 视角切换）。
2. **范式割裂**: 文本驱动、相机控制、动作条件模型使用不同输入格式，无法公平横向对比。
3. **维度缺失**: 缺乏对 **Physical Compliance**、**多轮一致性**、**视角切换保真度** 等关键交互能力的系统评测。

### 现有方法的局限

| 基准 | 短板 |
|------|------|
| [[VBench]] | 只测视频质量，无交互、无多轮 |
| [[WorldArena]] | 限于 [[RoboTwin]] 机器人仿真域，场景单一 |
| MIND | 只测记忆一致性 |
| Omni-WorldBench | 只覆盖第一视角，无第三视角 |
| Genie 系列内部评测 | 闭源、不可复现 |

### 本文的动机

[[交互式世界模型]] 的本质是给定历史观测和动作序列后预测未来帧。一个真正可用的世界模型需要在**多轮连续交互**下同时满足：高画质 + 遵守世界设定 + 准确响应交互 + 跨轮一致性 + 物理合理性。WBench 旨在用统一协议系统性量化这 5 个维度。

---

## 方法详解

### 任务形式化

[[交互式世界模型]] 的预测过程被建模为：给定历史观测 $o_{\leq t}=(o_0,\ldots,o_t)$ 与历史动作 $a_{\leq t}=(a_0,\ldots,a_t)$，预测下一帧观测 $o_{t+1}$。WBench 用一个**统一动作接口**适配三类范式：

- **输入**: 初始观测 $o_0$（图像）+ 世界设定文本 + 多轮交互序列 $\{a_t\}_{t=1}^T$
- **统一动作格式**: 文本指令 $\leftrightarrow$ 6-DoF 相机位姿 $\leftrightarrow$ 离散键盘动作（WASD + 鼠标），三者可双向转换
- **交互类型**: 导航（Navigation）/ 主体动作（Subject Action）/ 事件编辑（Event Editing）/ 视角切换（Perspective Switching）
- **输出**: 多轮视频序列 $\{V_t\}_{t=1}^T$
- **评测维度**: 5 大类 22 子指标

### 评测体系架构

WBench 包含四个核心模块：

#### 模块1: 数据构造（289 cases × 1058 turns）

**设计动机**: 覆盖足够丰富的场景 / 主体 / 风格 / 视角，使评测具有诊断性而非整体平均。

**具体实现**:
- **视角分布**: 第一视角 62% / 第三视角 38%，确保覆盖游戏类 FPV 与影视类 TPV
- **交互分布**: Navigation 57% / Subject Action 20% / Event Editing 17% / Perspective Switching 6%
- **场景分布**: 自然 31% / 城市 21% / 室内 17% / 影视作品 13% / 奇幻 10% / 体育 8%
- **风格分布**: 写实 52%，其余为动漫 / CG / 卡通等非写实风格
- **平均轮数**: 3.7 轮/case（最长 9 轮），支持显式多轮退化分析

#### 模块2: 统一动作控制接口

**设计动机**: 解决"文本驱动模型怎么测导航"和"相机模型怎么测语义事件"的范式鸿沟。

**具体实现**:
- 对每个 case 同时提供 **自然语言指令**、**6-DoF 相机轨迹**、**离散动作序列** 三种表示
- 文本驱动模型接收语言指令；相机控制模型接收 6-DoF；动作条件模型接收离散动作
- 用 [[MegaSaM]] 从生成视频中反推相机轨迹，跨范式比较"实际执行轨迹 vs 目标轨迹"

#### 模块3: 22 指标自动评测管线

**设计动机**: 评测必须**完全自动化**才可扩展到 20+ 模型，同时与人类判断对齐。

**具体实现**:
- **传统视觉指标**: 用 [[DINOv2]]、[[CLIP]]、[[SAM2]]、Depth Anything 3、TransNetV2、DreamSim 等专业模型计算
- **VLM-as-Judge**: 用 [[Qwen3-VL]]-30B 做语义层评测（事件触发、动作完成、视角切换语义），5 点李克特量表
- **物理合理性**: 微调 Qwen3-VL-30B 在伪影数据（几何畸变、穿模、不自然形变）上做视觉合理性回归
- 与 HPSv3 整合：对人类偏好打分归一化（percentile normalization）

#### 模块4: 人类对齐验证

**设计动机**: 自动指标必须与人类判断一致才有说服力。

**具体实现**:
- 标注员对 20 个模型在 10 个评测方面做配对偏好打分（pairwise win rate）
- 计算 per-model 自动得分 vs 人类胜率的 Spearman ρ
- 结果：所有 10 项 ρ≥0.94，4 项达到 ρ=1.00（事件编辑、主体动作、视角切换、空间一致性）

---

## 关键公式

### 公式1: [[交互式世界模型|交互式世界模型预测]]

$$
o_{t+1} \sim f_\theta(o_{t+1} \mid o_{\leq t}, a_{\leq t})
$$

**含义**: 交互式世界模型定义为：根据历史观测序列和历史动作序列采样下一帧观测的条件生成模型。这是 WBench 评测对象的统一形式化。

**符号说明**:
- $o_{\leq t}=(o_0,\ldots,o_t)$: 历史观测帧序列
- $a_{\leq t}=(a_0,\ldots,a_t)$: 历史动作序列（可以是文本、6-DoF pose 或离散动作）
- $f_\theta$: 参数为 $\theta$ 的世界模型
- $o_{t+1}$: 待预测的下一帧观测

### 公式2: [[Navigation Score|导航得分]]

$$
\mathrm{Nav} = 1 - \frac{1}{N}\sum_{i=1}^N \min\!\left(\frac{\mathrm{ATE}_i}{\tau},\, 1\right) + \lambda \cdot \mathrm{ConsistencyTerm}
$$

**含义**: 用 [[MegaSaM]] 从生成视频中估出实际相机轨迹，与目标轨迹比较 [[Absolute Trajectory Error|ATE]]，做归一化并加跨轮一致性项。

**符号说明**:
- $\mathrm{ATE}_i$: 第 $i$ 轮的绝对轨迹误差（米）
- $\tau$: 归一化阈值
- $N$: 总轮数
- $\lambda$: 跨轮一致性权重
- $\mathrm{ConsistencyTerm}$: 相邻轮起止点对接一致性

### 公式3: [[Spatial Consistency|空间一致性]]（Roundtrip）

$$
\mathrm{SC} = \mathrm{DreamSim}(o_0,\, o_T) \quad \text{when } a_{\leq T}\ \text{形成闭环}
$$

**含义**: 对回到起点的轨迹评估首末帧相似度，衡量世界模型是否能保持空间一致（同一地点走一圈回来应该看到同样的场景）。

**符号说明**:
- $o_0, o_T$: 起始帧与末帧
- $\mathrm{DreamSim}(\cdot,\cdot)$: 感知相似度
- 闭环轨迹：$a_{\leq T}$ 起点位置与终点位置重合

### 公式4: [[Geometric Consistency|几何一致性]]

$$
\mathrm{GC} = 1 - \frac{1}{|\mathcal{P}|}\sum_{(i,j)\in\mathcal{P}} \frac{\|\pi(D_i, K, T_{i\to j}) - p_j\|_2}{d_{\max}}
$$

**含义**: 用 Depth Anything 3 估深度，把帧 $i$ 的像素重投影到帧 $j$，计算重投影位移；位移越小，几何越一致。

**符号说明**:
- $D_i$: 帧 $i$ 的深度图
- $K$: 相机内参
- $T_{i\to j}$: 帧 $i$ 到 $j$ 的相机变换
- $\pi(\cdot)$: 透视投影
- $p_j$: 帧 $j$ 的对应像素
- $d_{\max}$: 归一化的最大允许位移

### 公式5: [[Photometric Consistency|光度一致性]]

$$
\mathrm{PC} = \mathrm{PSNR}\!\left(I_j,\, \mathrm{Warp}(I_i,\, D_i, T_{i\to j})\right)
$$

**含义**: 用深度和位姿把帧 $i$ warp 到帧 $j$ 的视角后，与真实帧 $j$ 比较 PSNR；衡量纹理 / 颜色一致性。

**符号说明**:
- $I_i, I_j$: 第 $i, j$ 帧
- $\mathrm{Warp}(\cdot)$: 基于深度和位姿的可微 warping
- $\mathrm{PSNR}$: 峰值信噪比

### 公式6: [[人类对齐 Spearman 相关|人机对齐 Spearman 相关]]

$$
\rho = 1 - \frac{6\sum_{m=1}^M d_m^2}{M(M^2-1)}
$$

**含义**: 对 $M$ 个模型计算自动得分排名与人类胜率排名的 Spearman 秩相关；WBench 在 10 个评测方面均 $\rho \geq 0.94$。

**符号说明**:
- $M$: 模型数量（20）
- $d_m$: 第 $m$ 个模型的自动排名与人类排名之差
- $\rho \in [-1, 1]$: 越接近 1 表示自动指标越能反映人类判断

---

## 关键图表

### Figure 1: Overview / 系统概览

![Figure 1](https://arxiv.org/html/2605.25874v1/x1.png)

**说明**: WBench 整体设计。**上半部分**展示一个 multi-turn case：用户依次发出 navigation、subject action、event editing、perspective switching 四种交互；世界模型连续生成多段视频。**下半部分**展示基准构成：世界设定（场景+主体+风格）、交互分类法、统一导航控制接口、以及 5 大评测维度（[[Video Quality|视频质量]]、[[Setting Adherence|设定遵循]]、[[Interaction Adherence|交互遵循]]、[[Consistency|一致性]]、[[Physical Compliance|物理合规]]）。

### Figure 2: Dataset Composition / 数据集构成

![Figure 2](https://arxiv.org/html/2605.25874v1/x2.png)

**说明**: WBench 沿 **8 条轴** 展示数据集分布：(1) 视角（FPV 62% / TPV 38%），(2) 交互类型（Nav 57% / SubAct 20% / EvtEdit 17% / PerSwitch 6%），(3) 场景类别（自然 / 城市 / 室内 / 影视 / 奇幻 / 体育），(4) 主体类型（人 64% / 动物 9% / 机器人 9% / 车辆 7% / 其他），(5) 动作子类型，(6) 事件编辑子类，(7) 视角切换子类，(8) 多轮深度分布（平均 3.7，最长 9 轮）。

### Figure 3: Cross-dimension Correlation Analysis / 维度间相关性分析

![Figure 3](https://arxiv.org/html/2605.25874v1/x3.png)

**说明**: 5 维评测的诊断价值。**(a) Pearson 相关热力图**: Navigation 与其他四维近零相关（r≈-0.12 vs Video Quality），说明"会导航不等于会渲染"；Physics Compliance 与 Video Quality 强相关（r=0.84），与 Navigation 反相关（r=-0.15）。**(c) Per-setting Z-score 偏差**: 不同视角 / 场景 / 主体下模型表现的相对难度热图，揭示哪类设定对哪类模型更困难。

### Figure 4: Per-turn Performance Degradation / 多轮退化曲线

![Figure 4](https://arxiv.org/html/2605.25874v1/x4.png)

**说明**: 模型在多轮交互下的退化幅度。**Navigation 从第 1 轮到第 4+ 轮跌 ~33 分**，是最大短板；**Event Editing 退化 ~13 分**；**Subject Action 退化 ~9 分**。证明现有世界模型在**长视界**下显著失效，与多轮 [[Consistency|一致性]] 密切相关。

### Figure 5: Human-Auto Alignment / 人机一致性

![Figure 5](https://arxiv.org/html/2605.25874v1/x5.png)

**说明**: 自动指标与人类配对偏好的 Spearman ρ 对比，覆盖 10 个评测方面。**所有方面 ρ≥0.94**，其中事件编辑、主体动作、视角切换、空间一致性 **达到 ρ=1.00**，验证 WBench 自动评测在模型排名粒度上完全反映人类偏好。

### Table 1: 22 指标定义

| 维度 | 指标编号 | 指标名 | 实现 |
|------|---------|--------|------|
| Video Quality | V.1 | Aesthetic Quality | LAION 美学预测器 |
| Video Quality | V.2 | Imaging Quality | MUSIQ |
| Video Quality | V.3 | Temporal Flickering | 帧间像素抖动 |
| Video Quality | V.4 | Dynamic Degree | RAFT 光流量级 |
| Video Quality | V.5 | Motion Smoothness | 插值帧 LPIPS |
| Video Quality | V.6 | HPSv3-Norm | 百分位归一化 HPSv3 |
| Setting Adherence | S.1 | Scene Adherence | [[Qwen3-VL]] 评分（场景+离屏元素） |
| Setting Adherence | S.2 | Subject Adherence | [[Qwen3-VL]] 评分（外观+动作先验） |
| Interaction | I.1 | Navigation Score | [[MegaSaM]] + ATE + 一致性 |
| Interaction | I.2 | Event Editing | [[VLM]] 5 点协议 |
| Interaction | I.3 | Subject Action | [[VLM]] 5 点协议 |
| Interaction | I.4 | Perspective Switching | 分类协议 |
| Consistency | C.1 | Spatial Consistency | DreamSim 闭环 |
| Consistency | C.2 | Gated Spatial | C.1 + 中间帧最小相似度 |
| Consistency | C.3 | Segment Continuity | TransNetV2 hard-cut |
| Consistency | C.4 | Perspective Consistency | [[SAM2]] 主体跟踪重心稳定性 |
| Consistency | C.5 | Geometric Consistency | Depth Anything 3 重投影 |
| Consistency | C.6 | Photometric Consistency | warp+PSNR |
| Consistency | C.7 | Subject Consistency | [[DINOv2]] + [[CLIP]] |
| Consistency | C.8 | Background Consistency | [[CLIP]] 相邻帧余弦 |
| Physical | P.1 | Causal Fidelity | [[VLM]] 两阶段（全局+7 物理子项） |
| Physical | P.2 | Visual Plausibility | 微调 Qwen3-VL-30B |

**说明**: 22 指标覆盖**像素级**（V.3、V.5、C.6）、**特征级**（C.7、C.8、C.1）、**几何级**（C.5、I.1）、**语义级**（S.1-2、I.2-4、P.1-2）四个抽象层级。

### Table 2: 20 个模型主结果（每范式最强）

| 维度 | 文本驱动最佳 | 相机控制最佳 | 动作条件最佳 |
|------|------------|-------------|------------|
| Video Quality | Seedance 1.5: **82.1** | LingBot-World: 78.9 | Happy Oyster: 77.3 |
| Setting Adherence | Wan 2.7: **91.4** | LingBot-World: 72.6 | Happy Oyster: 74.2 |
| Navigation | YUME 1.5: 72.0 | HY-World 1.5: **87.5** | Matrix-Game 3.0: 83.5 |
| Event Editing | Wan 2.7: **84.0** | N/A | N/A |
| Subject Action | Kling 3.0: **85.6** | N/A | N/A |
| Perspective Switch | Wan 2.7: **55.0** | N/A | N/A |
| Consistency | HY-Video 1.5: 87.4 | LingBot-World: **89.9** | Happy Oyster: 84.3 |
| Physical | Wan 2.7: **71.8** | LingBot-World: 71.2 | Astra: 65.2 |

**关键发现**:
- **没有任何模型在 5 维上全部领先**，每个范式都有自己的最强项与短板
- 文本驱动模型在 **Setting Adherence** 领先相机/动作模型 **17-20 分**（语义理解优势）
- 相机/动作世界模型在 **Native Navigation** 领先文本模型（专项训练优势）
- 视角切换（Perspective Switching）整体得分最低（最高仅 55），是当前世界模型最难的能力

### Table 3: 维度间相关性（节选）

| 维度对 | Pearson r |
|--------|-----------|
| Physics ↔ Video Quality | **0.84** |
| Physics ↔ Consistency | 0.72 |
| Physics ↔ Navigation | -0.15 |
| Navigation ↔ Video Quality | -0.12 |
| Navigation ↔ Setting Adherence | 0.08 |
| Camera Control ↔ Perspective Consistency | ~0.00 |

**关键发现**: Navigation 是一项**几乎正交**的能力，与其他维度近零相关；Physics 主要由渲染质量决定而非动作控制精度；**相机控制能力 ≠ 视角一致性维持能力**，这是关键工程教训。

---

## 实验结果

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| WBench | 289 cases / 1058 turns | 多轮 + 多范式 + 多视角 | 主基准 |
| 内部人类偏好集 | 20 模型 × 10 方面配对 | 标注员配对胜率 | 验证自动指标 |

### 评测设置

- **评测模型数**: 20 个 SOTA 系统
  - **文本驱动 (9)**: Seedance 1.5, Wan 2.7, Kling 3.0, YUME 1.5, HY-Video 1.5, LTX 2.3, [[LongCat-Video]], Kairos 3.0, Cosmos 2.5
  - **相机控制 (5)**: LingBot-World, HY-World 1.5, Fantasy-World, InSpatio-World, Astra
  - **动作条件 (6)**: Happy Oyster, Matrix-Game 3.0, Genie 3, Matrix-Game 2.0, HY-GameCraft, Infinite-World
- **评测预算**: 每个模型 × 289 cases × 平均 3.7 turns = ~1058 视频/模型
- **依赖外部模型**: [[MegaSaM]], Depth Anything 3, TransNetV2, DreamSim, [[SAM2]], [[DINOv2]], [[CLIP]], [[Qwen3-VL]]-30B
- **人类标注**: 多名标注员对配对结果做胜率打分

### 可视化结果

定性观察的核心信息：
1. **导航 + 渲染分离**: 高画质模型经常出现"看着真但走错路"
2. **长视界崩坏**: 多数模型在第 3-4 轮开始出现纹理漂移、几何畸变
3. **视角切换是公认难题**: 即便最强模型也低于 55 分
4. **范式专精化**: 文本模型擅长高层语义，世界模型擅长几何控制，但两者目前互斥

---

## 批判性思考

### 优点

1. **范式无关的统一接口**: 文本/位姿/动作互转的设计巧妙，解决了世界模型评测最大的工程痛点。
2. **自动 + 人类对齐**: 22 指标全部自动，且 Spearman ρ≥0.94 与人类一致，可扩展性强。
3. **多轮诊断维度**: 显式量化"长视界退化"，给后续研究指明短板方向（特别是 Navigation 的 33 分退化）。
4. **覆盖范围广**: 8 轴数据分布覆盖第一/第三视角、6 类场景、4 类交互，避免单一域偏置。

### 局限性

1. **依赖外部 VLM**: 大量指标依赖 [[Qwen3-VL]]-30B 等外部模型，外部模型更新可能导致分数不可复现。
2. **289 cases 规模偏小**: 相对 [[VBench]] 的几千 prompt 显得有限，长尾场景可能采样不足。
3. **多轮深度有限**: 平均 3.7 轮、最长 9 轮，对真正"分钟级"世界模型还不够长。
4. **物理评测仍是 VLM 判读**: 因果保真度由 VLM 评估而非物理仿真器对照，存在评估者偏置风险。
5. **闭源模型评测难复现**: 部分商业模型（Kling、Seedance、Genie 3）API 行为可能随时间漂移。

### 潜在改进方向

1. **延长多轮深度**: 扩展到 30+ 轮，对接 [[VLA Memory|世界模型记忆]] 评测。
2. **物理仿真对照**: 用 [[RoboTwin]] / Isaac Sim 提供 ground truth 物理轨迹，替代 VLM 评判。
3. **动态加入新指标**: 让自动管线可插拔接入新评测模型，避免被单一 VLM 锁定。
4. **跨平台标准化**: 与 [[VBench]]、[[WorldArena]] 建立分数转换桥，避免基准碎片化。
5. **失败案例库**: 系统收集所有模型的失败案例，构建对抗性评测集。

### 可复现性评估

- [x] 代码开源（GitHub）
- [x] 数据集开源（HuggingFace）
- [x] 评测协议详细
- [ ] 闭源模型评测受 API 漂移影响
- [x] 人类标注规则公开

---

## 关联笔记

### 基于
- [[VBench]]: 视频质量评测的前身，WBench 沿用其 6 个画质指标
- [[WorldArena]]: 机器人域交互评测的前身，WBench 推广到通用场景
- [[MegaSaM]]: 用于从生成视频反推相机轨迹的关键依赖
- [[DINOv2]] / [[CLIP]] / [[SAM2]]: 视觉特征 / 跟踪 backbone
- [[Qwen3-VL]]: VLM-as-Judge 核心评测器

### 对比
- VBench: 单轮、无交互
- WorldArena: 仅机器人域，仅 6 维
- MIND: 仅记忆评测
- Omni-WorldBench: 仅第一视角

### 方法相关
- [[VLM-as-Judge]]: 用 VLM 做评测的范式
- [[交互式世界模型]]: 评测对象的形式化定义
- [[Navigation Score]]: 导航专项指标
- [[Spatial Consistency]] / [[Geometric Consistency]] / [[Photometric Consistency]]: 多层级一致性指标

### 被评测模型相关
- [[LongCat-Video]]: 同团队文本视频模型
- Genie 3 / Matrix-Game: 动作条件世界模型
- Cosmos / Wan / Seedance: 文本驱动视频模型

---

## 速查卡片

> [!summary] WBench (Meituan LongCat + Fudan, 2026)
> - **核心**: 首个多轮、跨范式（文本/相机/动作）的交互式世界模型评测基准
> - **方法**: 5 维 22 指标 + 统一导航接口 + VLM-as-Judge + 人类对齐验证
> - **规模**: 289 cases / 1058 turns / 20 SOTA 模型
> - **结果**: 没有模型 5 维全胜；Navigation 多轮退化 33 分；Spearman ρ≥0.94 与人类一致
> - **代码**: https://github.com/meituan-longcat/WBench

---

*笔记创建时间: 2026-05-26*
