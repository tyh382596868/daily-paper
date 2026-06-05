---
title: "Dream.exe: Can Video Generation Models Dream Executable Robot Manipulation?"
method_name: "Dream.exe"
authors: [Rui Zhao, Kaiming Yang, Jifeng Zhu, Siyang Chen, Ziqi Wang, Weijia Wu, Kevin Qinghong Lin, Heng Wang, Mike Zheng Shou]
year: 2026
venue: arXiv
tags: [video-generation, robot-manipulation, world-model, benchmark, video-to-action, executable-video, physical-plausibility, robocasa]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.04811v2
created: 2026-06-05
---

# 论文笔记：Dream.exe: Can Video Generation Models Dream Executable Robot Manipulation?

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | NUS Show Lab、University of Oxford、Tencent |
| 日期 | June 2026 |
| 项目主页 | 待开源（论文中标注 code release） |
| 对比基线 | [[Cosmos-Policy]]、[[Wan2.2]]、[[LTX-Video]]、Veo 3.1、Kling 3.0、Hailuo 2.3、SeedDance 2.0 |
| 链接 | [arXiv](https://arxiv.org/abs/2606.04811) / [HTML](https://arxiv.org/html/2606.04811v2) |

---

## 一句话总结

> [[Dream.exe Benchmark|Dream.exe]] 把 [[Video Generation|视频生成模型]] 的产出转成可在 [[MuJoCo]] 里执行的机器人轨迹，用任务成功率而非画面好看程度来评判"世界模型"是否真的懂物理。

---

## 核心贡献

1. **可执行性评测范式**：首次把 [[Executable Video Generation|可执行视频生成]] 作为评测信号——用任务成功率取代单纯的 [[VBench]] 风格视觉打分，让 [[World Model|世界模型]] 这一概念有了可证伪的接口。
2. **Video-to-Trajectory 五步流水线**：[[Grounding DINO]] + [[SAM2]] 取掩膜 → [[CoTracker3]] 2D 跟踪 → 带 LoRA 微调的 DVD [[Depth Estimation|深度估计]] → 3D 提升 → [[Kabsch Alignment|Kabsch]] 旋转 + 夹爪先验，将"会做梦"的视频翻译成 7 自由度机器人指令。
3. **101 任务三难度套件**：在 [[RoboCasa]] 上手工挑选 Level 1（原子动作）/ Level 2（多物体耦合）/ Level 3（多阶段长程）三档任务，配套 8 个前沿/开源/机器人专用视频模型的统一榜单。
4. **关键反直觉发现**：[[Physical Plausibility|物理合理性]] 与任务成功率 [[Spearman Correlation|相关系数]] 仅 $r=-0.03$；LTX 2.3 视觉合理性第一，[[SR-B|二值成功率]] 倒数；[[Cosmos-Policy]] 机器人专训提升轨迹几何但损害任务级表现。

---

## 问题背景

### 要解决的问题

[[Sora]]/[[Kling]]/[[Veo]]/[[Wan]] 等大规模视频生成模型常被称作 [[World Model|世界模型]]，但它们到底"懂"多少物理？传统 benchmark（如 [[VBench]]、[[Physics IQ]]）只看像素层面的真实感，无法回答：**生成的机械臂运动如果真照样执行，能否完成任务？**

### 现有方法的局限

- **纯视觉指标失灵**：VBench、[[FVD]]、[[CLIPSIM]] 高分不代表物理合理；同一条视频里物体可能"凭空漂浮"却像素逼真。
- **缺少闭环执行**：[[Genie]]、[[UniSim]] 类工作可以"播放"想象世界，但无法在 [[MuJoCo]]/[[robosuite]] 里追踪真实奖励信号。
- **机器人专训模型偏窄**：[[Cosmos-Policy]] 等专门在机器人数据上 fine-tune 的生成器擅长几何复刻，但泛化任务覆盖少。

### 本文的动机

如果一个视频模型真的内化了物理定律，那么它"想象"出来的机械臂轨迹应当在仿真器里也能成功——所以**把任务成功率当成 grounding signal**，让评测从"看起来对"变成"做出来对"。

---

## 方法详解

### 模型架构

[[Dream.exe Benchmark|Dream.exe]] 不是新模型，而是一个 **评测框架**，采用四阶段流水线：

- **输入**：初始场景图像 $I_0$ + 任务文本指令 $l$
- **Stage 1 — 视频生成**：被测模型 $G_\theta$ 产出操控视频 $V = G_\theta(I_0, l)$（Level 1/2 限 5-8 秒，Level 3 更长）
- **Stage 2 — 视觉评测**：由 [[Gemini 3 Pro]] + [[Qwen3-VL]] 作为 [[VLM-as-Judge|VLM 评判者]]，三维打分（稳定性、物理合理性、任务依从性）
- **Stage 3 — 视频转轨迹**：[[Video-to-Trajectory|V2T]] 五步流水线，输出 [[End-Effector Trajectory|末端轨迹]] $\tau = \{(p_t, R_t, g_t)\}_{t=1}^T$，其中 $p_t \in \mathbb{R}^3$ 位置、$R_t \in \mathrm{SO}(3)$ 姿态、$g_t \in \{0,1\}$ 夹爪状态
- **Stage 4 — 仿真闭环执行**：在 [[robosuite]] / [[MuJoCo]] 上用 checkpoint 校正控制器闭环回放，统计任务成功率

### 核心模块

#### 模块 1：Video-to-Trajectory 提取

**设计动机**：把像素空间的"想象"翻译成可执行的 [[Action Chunking|动作序列]]，而无需重新训练任何 [[VLA]]。

**具体实现**（五步）：

1. **区域掩膜初始化**：仿真自带分割 *或* [[Grounding DINO]] + [[SAM2]] 自动定位末端执行器与目标物体
2. **2D 点跟踪**：[[CoTracker3]] 输出像素坐标 $\{u_t, v_t\}$ 与可见性掩码
3. **深度估计 + 3D 提升**：[[DVD Depth|DVD]] 视频深度模型 + [[LoRA]] 在机器人 rollout 上微调，在归一化视差空间训练，再用相机内参 $K$ 反投影到世界坐标
4. **末端姿态估计**：取被跟踪点的视觉中心作为位置；通过 [[Kabsch Alignment|Kabsch]] 与首帧锚点对齐估计 [[Rotation Matrix|旋转矩阵]]
5. **夹爪动作合成**：由末端–物体相对运动和任务先验（[[Gripper Action Prior|夹爪先验表]]）推断 open/close 时刻

#### 模块 2：闭环 Checkpoint 校正

**设计动机**：开环回放误差会指数级累积；论文设置位置 5 mm、姿态 0.03 rad 的 checkpoint 容差，超出则触发控制器修正，再继续推进 [[Action Chunking|动作块]]。

#### 模块 3：双轨评测协议

**Visual Track**：VLM 三维 + 人评四维（稳定/合理/依从/可执行预期）打分。
**Physical Track**：轨迹相似度（HSD/DYN/NDTW）+ 可执行性（E-SR/Pos95/Rot95/Smth）+ 任务级（SR-B/SR-P/子目标）。

---

## 关键公式

### 公式 1：[[Symmetric Hausdorff Distance|对称豪斯多夫距离]] HSD（轨迹形状偏差）

$$
d_H(A, B) = \max\!\left\{ \sup_{a \in A} \inf_{b \in B} \|a - b\|_2,\; \sup_{b \in B} \inf_{a \in A} \|a - b\|_2 \right\}
$$

**含义**：度量生成轨迹 $A$ 与参考轨迹 $B$ 的最坏点对偏差，对外形扭曲最敏感；Dream.exe 把它归一化到 $[0,1]$ 并取 $1 - \tilde{d}_H$ 当做相似度（越高越好）。

**符号说明**：
- $A, B \subset \mathbb{R}^3$：生成 / 参考 [[End-Effector Trajectory|末端轨迹]]点集
- $\inf_{b \in B} \|a-b\|_2$：点 $a$ 到集合 $B$ 的最近欧氏距离
- $\sup$：取最坏情况

### 公式 2：[[Wasserstein-1 Distance|Wasserstein-1 距离]] DYN（动力学差异）

$$
W_1(\mu, \nu) = \inf_{\gamma \in \Gamma(\mu, \nu)} \int_{\mathbb{R} \times \mathbb{R}} |x - y|\, \mathrm{d}\gamma(x, y)
$$

**含义**：在 *逐帧速度分布* 上度量两条轨迹的动力学差异。生成视频常出现"快进"或"漂移"，这一项能捕捉到。

**符号说明**：
- $\mu, \nu$：生成与参考轨迹的速度边缘分布 $\{\|p_{t+1}-p_t\|_2 / \Delta t\}$
- $\Gamma(\mu, \nu)$：所有以 $\mu, \nu$ 为边缘的联合分布集合
- $\gamma$：最优运输耦合

### 公式 3：[[Normalized DTW|归一化 DTW]] 对齐代价 NDTW

$$
\mathrm{NDTW}(A, B) = \frac{1}{|\pi^\star|} \sum_{(i, j) \in \pi^\star} \|a_i - b_j\|_2,\quad \pi^\star = \arg\min_{\pi \in \Pi(A,B)} \sum_{(i,j) \in \pi} \|a_i - b_j\|_2
$$

**含义**：时间错位下的最小累计偏差除以路径长度，反映"步调对齐"质量；论文还把它再次映射到 $[0,1]$ 相似度。

**符号说明**：
- $\pi^\star$：动态规划得到的最优 [[Dynamic Time Warping|DTW]] 对齐路径
- $\Pi(A,B)$：所有单调、连续、首末端点固定的对齐集合

### 公式 4：[[Kabsch Alignment|Kabsch 旋转]] 估计

$$
R^\star = \arg\min_{R \in \mathrm{SO}(3)} \sum_{i=1}^{N} \|R q_i - p_i\|_2^2 = V\, \mathrm{diag}(1, 1, \det(VU^\top))\, U^\top,\quad H = \sum_i p_i q_i^\top,\; H = U \Sigma V^\top
$$

**含义**：用 [[SVD]] 解析地求出把首帧锚点 $\{q_i\}$ 旋到当前帧 $\{p_i\}$ 的最优旋转矩阵，是 Dream.exe 给末端"补充姿态维度"的关键。

**符号说明**：
- $p_i, q_i$：当前帧与首帧的 3D 锚点（已去质心）
- $H$：交叉协方差矩阵
- $\mathrm{diag}(1,1,\det(VU^\top))$：消除反射，保证 $R \in \mathrm{SO}(3)$

### 公式 5：[[SR-P|进度型成功率]] 与 [[SR-B|二值成功率]]

$$
\mathrm{SR\text{-}B} = \frac{1}{N}\sum_{i=1}^N \mathbb{1}[s_i = \text{success}],\qquad \mathrm{SR\text{-}P} = \frac{1}{N}\sum_{i=1}^N \frac{1}{|\mathcal{K}_i|}\sum_{k \in \mathcal{K}_i} \mathbb{1}[\text{checkpoint } k \text{ reached}]
$$

**含义**：SR-B 只看最终是否成功，SR-P 把任务拆成子检查点 $\mathcal{K}_i$ 给出连续进度。Dream.exe 同时报告两者，避免长程任务"全 0"无法区分。

**符号说明**：
- $N$：任务总数（101）
- $\mathcal{K}_i$：任务 $i$ 的关键 checkpoint 集合
- $\mathbb{1}[\cdot]$：指示函数

### 公式 6：归一化视差空间（[[Depth Estimation|深度模型]] LoRA 训练目标）

$$
\mathcal{L}_{\text{depth}} = \mathbb{E}_{t}\big[\| \hat{d}_t - d^\star_t \|_1\big] + \lambda_s\, \mathcal{L}_{\text{grad-space}} + \lambda_t\, \mathcal{L}_{\text{grad-time}},\quad d_t = \mathrm{norm}\!\left(\frac{1}{z_t + \epsilon}\right)
$$

**含义**：DVD 深度在视差空间归一化训练，加上空间梯度与时间梯度匹配项，保证多帧深度一致；这是 V2T 能把 2D 点稳定提升到 3D 的前提。

**符号说明**：
- $\hat{d}_t, d^\star_t$：预测 / 真值归一化视差
- $z_t$：度量深度
- $\lambda_s, \lambda_t$：空间 / 时间梯度权重

---

## 关键图表

### Figure 1: 任务套件总览

![Figure 1](https://arxiv.org/html/2606.04811v2/x1.png)

**说明**：左侧是各难度等级的代表性场景与任务描述；右上是 101 个任务在 Level 1 / 2 / 3 上的分布；右下展示了刻意多样化的相机视角，覆盖前视、侧视、桌面斜视等，提升对视觉模型的泛化检验。

### Figure 2: Dream.exe 评测主流水线

![Figure 2](https://arxiv.org/html/2606.04811v2/x2.png)

**说明**：四阶段框架——给定初始图像和任务指令 → 视频生成模型产出操控视频 → 视觉质量与物理合理性评估 → [[Video-to-Trajectory|V2T]] 解码为轨迹 → 在 [[MuJoCo]] / [[robosuite]] 闭环执行，**任务成功率是最终裁判**。

### Figure 3: 成功与失败模式分类

![Figure 3](https://arxiv.org/html/2606.04811v2/x3.png)

**说明**：失败被归纳为三类：① **物体悬浮（Levitation）**——物体脱离接触却"飘"起来；② **幻影抓取（Phantom Grasp）**——夹爪未真正闭合却带走物体；③ **运动学崩溃（Kinematic Breakdown）**——关节超限或自穿透。论文统计显示"幻影抓取"与"运动学崩溃"是当前模型主要瓶颈。

### Figure 4: 详细的 Video-to-Execution 子流水线

![Figure 4](https://arxiv.org/html/2606.04811v2/x4.png)

**说明**：放大 Figure 2 中的 V2T + 执行模块，依次显示：2D tracklet 抽取 → 深度图标定 → 3D 点轨迹提升 → 末端旋转估计 → 夹爪动作合成 → 可执行轨迹。直观呈现"画面 → 控制指令"的完整翻译路径。

---

## 实验结果

### Table 1：视觉质量评估（VLM 打分 1-5）

| 模型 | Stability | Plausibility | Task Adherence | Overall |
|------|-----------|--------------|----------------|---------|
| Veo 3.1 | 4.10 | 2.18 | **3.03** | 3.10 |
| Kling 3.0 | 4.05 | 2.25 | 2.95 | 3.08 |
| Hailuo 2.3 | 3.85 | 2.30 | 2.70 | 2.95 |
| SeedDance 2.0 | 4.20 | 2.15 | 2.85 | 3.07 |
| Wan 2.7 | 4.15 | 2.28 | 2.78 | 3.07 |
| Wan 2.2 (open) | 3.55 | 2.10 | 2.40 | 2.68 |
| LTX 2.3 | 2.80 | **2.39** | 1.79 | 2.33 |
| CosmosPolicy-BenchCam | **7.53*** | 2.20 | 2.60 | — |

> *CosmosPolicy 数值反映其针对基准视角微调后的稳定性偏高。

**说明**：闭源前沿模型在 *任务依从* 与 *稳定性* 上占优；LTX 反而在 *物理合理* 维度最高——为后面的"视觉好≠能执行"埋下伏笔。

### Table 2：轨迹相似度（值越高越好，归一化到 [0,1]）

| 模型 | HSD-EEFvis | DYN-EEFvis | NDTW-EEFvis | HSD-OBJ | NDTW-OBJ |
|------|------------|------------|-------------|---------|----------|
| Wan 2.7 | **0.753** | **0.778** | **0.838** | 0.602 | 0.770 |
| Kling 3.0 | 0.728 | 0.751 | 0.812 | 0.591 | 0.755 |
| SeedDance 2.0 | 0.715 | 0.742 | 0.805 | 0.578 | 0.748 |
| Veo 3.1 | 0.692 | 0.738 | 0.793 | 0.560 | 0.732 |
| CosmosPolicy-BenchCam | 0.701 | 0.720 | 0.789 | **0.629** | **0.798** |
| LTX 2.3 | 0.512 | 0.498 | 0.580 | 0.430 | 0.585 |

**说明**：通用大模型（Wan 2.7）在末端轨迹上反超机器人专训的 Cosmos-Policy；但后者在 *物体轨迹* 上更准，说明它学会了"看物体"。

### Table 3：可执行性（E-SR ↑，nDTW / Pos95 / Rot95 / Smth ↓）

| 模型 | E-SR (Lvl 1/2/3) | nDTW | Pos95 (cm) | Rot95 (deg) | Smth |
|------|------------------|------|------------|-------------|------|
| CosmosPolicy-DefaultCam | **0.81 / 0.74 / 0.55** | 4.21 | 3.8 | 5.7 | 0.18 |
| Wan 2.7 | 0.72 / 0.62 / 0.40 | 4.08 | 4.5 | 6.9 | 0.22 |
| Kling 3.0 | 0.69 / 0.58 / 0.38 | **3.665** | 4.7 | 6.5 | 0.20 |
| SeedDance 2.0 | 0.70 / 0.61 / 0.41 | 3.92 | 4.6 | 7.0 | 0.21 |
| Veo 3.1 | 0.65 / 0.52 / 0.30 | 4.32 | 5.2 | 7.8 | 0.24 |
| LTX 2.3 | 0.40 / 0.25 / 0.10 | 6.10 | 9.5 | 12.5 | 0.41 |

**说明**：CosmosPolicy 在 checkpoint 到达率最高——它的轨迹几何最干净；但 Kling 在 nDTW（时间对齐）反而最好，说明前沿模型的*节奏感*已经不输专门训练。

### Table 4：任务级成功率（SR-B 二值 / SR-P 进度）

| 模型 | L1 SR-B | L2 SR-B | L3 SR-B | L1 SR-P | L2 SR-P | L3 SR-P |
|------|---------|---------|---------|---------|---------|---------|
| CosmosPolicy-BenchCam | **20.8%** | 2.4% | 0.0% | 0.45 | 0.18 | 0.05 |
| SeedDance 2.0 | 15.1% | **21.4%** | 0.0% | 0.40 | 0.42 | 0.08 |
| Wan 2.7 | 14.2% | **21.4%** | 0.0% | 0.38 | 0.41 | 0.10 |
| Kling 3.0 | 12.5% | 18.7% | **6.2%** | 0.36 | 0.39 | 0.18 |
| Veo 3.1 | 3.3% | 10.5% | 0.0% | 0.22 | 0.31 | 0.06 |
| LTX 2.3 | 0.0% | 0.0% | 0.0% | 0.05 | 0.02 | 0.00 |
| **参考：GT 深度回放** | 100% | 98% | 80% | 1.0 | 0.98 | 0.85 |

**说明**：① Level 3 几乎全员崩溃，**只有 Kling 3.0 拿到 6.2%**；② 通用大模型在 Level 2 反超机器人专训模型——印证"机器人微调把几何练好了，却没学会做事"的发现。

### Table 5：人评（4 位标注员 1-5 分）

| 模型 | Stability | Plausibility | Adherence | Execution Expect |
|------|-----------|--------------|-----------|------------------|
| CosmosPolicy-DefaultCam | **4.42** | **4.25** | 3.85 | 2.62 |
| Kling 3.0 | 4.30 | 4.10 | **4.58** | **2.77** |
| SeedDance 2.0 | 4.20 | 3.95 | 4.40 | 2.65 |
| Wan 2.7 | 4.18 | 3.90 | 4.32 | 2.55 |
| LTX 2.3 | 1.95 | 1.79 | 2.05 | 1.80 |

**说明**：人评与 VLM 评相互交叉验证；Kling 在"预期可执行性"最高，与它唯一拿到 Level 3 成功率的事实一致。

### 关键发现汇总

- **视觉合理性 ≠ 可执行性**：Plausibility 与 SR-B 的 [[Spearman Correlation|相关系数]] $r=-0.03$；Veo 3.1 任务依从最高但 Level 1 仅 3.3% 成功。
- **生成先验有效但短**：前沿模型在 Level 1/2 拿到 15–21% 非平凡成功率，但 Level 3 长程几乎全军覆没。
- **机器人微调 ≠ 任务成功**：[[Cosmos-Policy]] 几何 checkpoint 最强，Level 2 SR-B 只剩 2.4%——它对相机和场域过拟合。
- **LoRA 微调只改外观**：Wan 2.2-LoRA 变体在 HSD/DYN 提升但 SR-B 几乎不变，"学了画风没学物理"。
- **失败模式**：物体悬浮、幻影抓取、运动学崩溃三类，其中后两者占绝对多数。

---

## 数据集与实现细节

| 项目 | 内容 |
|------|------|
| 任务来源 | [[RoboCasa]] 365 中精选 101 episodes |
| 难度分层 | Level 1（原子动作）/ Level 2（多物体耦合）/ Level 3（多阶段长程） |
| 仿真器 | [[MuJoCo]] + [[robosuite]] |
| 视觉评测 | [[Gemini 3 Pro]] + [[Qwen3-VL]] 双 VLM 评判 |
| 深度估计 | DVD-DiT + [[LoRA]] 微调，归一化视差空间 |
| 跟踪 | [[CoTracker3]] |
| 分割 | [[Grounding DINO]] + [[SAM2]] |
| 执行控制 | OSC（Operational Space Controller）+ checkpoint 5 mm / 0.03 rad 容差 |
| 视频长度 | Level 1/2 限 5-8 秒，Level 3 更长 |

---

## 批判性思考

### 优点
1. **范式贡献**：把"评测视频生成"从美学打分转成可证伪的任务成功率，是世界模型研究急需的"硬指标"。
2. **流水线开源价值高**：V2T 模块自身就是一个有用的工具，可被任何 video → action 的工作复用。
3. **结论反直觉但站得住**：Plausibility 与成功率脱钩、机器人微调反伤任务，这些发现对当前"video model = world model"的乐观叙事是一剂冷水。

### 局限性
1. **依赖深度估计 + 跟踪的级联误差**：DVD 深度即使加了 LoRA，多帧一致性仍受限；末端"视觉中心"和真实 TCP 之间的偏移要靠初始帧标定，对新场景泛化未知。
2. **仿真域单一**：仅 [[RoboCasa]] / [[MuJoCo]]，未覆盖 [[Isaac Lab]]、[[ManiSkill]] 等不同物理引擎，可能高估机器人微调模型（[[Cosmos-Policy]]）的"分布外"表现。
3. **闭源模型评测不可复现**：Hailuo / Kling / Veo / SeedDance 的版本和参数无法锁定，未来再跑一次结果会漂移。
4. **VLM 评判一致性**：Gemini 3 Pro / Qwen3-VL 自身存在偏好，与人评相关性论文未深挖。
5. **任务规模偏小**：101 任务对"统计显著性"略弱，Level 3 仅占 24 个，单个任务偶然性大。

### 潜在改进方向
1. 引入 [[Isaac Lab]] / [[ManiSkill]] 等多仿真器交叉验证；
2. 用 [[Sora 2]] / [[Veo 3]] / [[Hailuo 03]] 等下一代模型重新跑 Level 3，看长程能力曲线；
3. 把 V2T 流水线本身变成 **可微分** 的 reward 信号，反向用于 [[RLHF for Video|视频 RL 微调]]；
4. 增加 [[Bimanual 任务族|双臂任务]] 和接触丰富 task，对当前以单臂 pick-and-place 为主的套件做扩展。

### 可复现性评估
- [x] 论文承诺代码 + 任务套件开源（待 release）
- [ ] 预训练模型（评测对象多为闭源 API）
- [x] 训练细节（V2T LoRA 训练目标与超参数有描述）
- [x] 数据集（基于公开 [[RoboCasa]] 365）

---

## 关联笔记

### 基于
- [[CoTracker3]]：2D 点跟踪
- [[SAM2]]：物体分割
- [[Grounding DINO]]：开放词汇检测
- [[Cosmos-Policy]]：机器人专用视频生成对比基线
- [[Wan2.2]] / [[LTX-Video]]：开源对比基线
- [[RoboCasa]]：任务来源

### 对比
- [[VBench]]：传统视觉质量基准，被本文证伪"视觉好=世界模型好"
- [[Physics IQ]]：只看物理合理性，无法验证可执行性
- [[DreamVLA]] / [[Dreamitate]] / [[RoboDream]]：同类"梦境驱动"机器人工作

### 方法相关
- [[Video-to-Trajectory]]：核心翻译模块
- [[Kabsch Alignment]]：姿态恢复
- [[Symmetric Hausdorff Distance]] / [[Wasserstein-1 Distance]] / [[Dynamic Time Warping]]：轨迹度量
- [[VLM-as-Judge]]：视觉评测范式

### 硬件 / 仿真相关
- [[MuJoCo]]：物理引擎
- [[robosuite]]：机器人仿真前端
- [[Franka Panda]]：RoboCasa 默认机械臂

---

## 速查卡片

> [!summary] Dream.exe
> - **核心**：用任务成功率而非画面好看程度评测视频生成模型的"世界模型"能力
> - **方法**：4 阶段流水线（生成 → 视觉评估 → V2T 五步 → 仿真闭环），8 模型 × 101 任务 × 3 难度
> - **结果**：视觉合理性与成功率几乎零相关；Level 1/2 前沿模型 15–21%，Level 3 仅 Kling 6.2%
> - **代码**：待开源（arXiv 2606.04811）

---

*笔记创建时间：2026-06-05*
