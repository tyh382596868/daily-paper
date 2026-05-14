---
title: "The DAWN of World-Action Interactive Models"
method_name: "DAWN"
authors: [Hongbo Lu, Liang Yao, Chenghao He, Haoyu Wang, Xiang Gu, Xianfei Li, Wenlong Liao, Tao He, Pai Peng]
year: 2026
venue: arXiv
tags: [world-action-interactive-model, autonomous-driving, action-denoising, latent-world-model, recursive-refinement, end-to-end-driving]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.11550v1
created: 2026-05-14
---

# 论文笔记：The DAWN of World-Action Interactive Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | COWARobot Co. Ltd, Shanghai Jiao Tong University, Hohai University |
| 日期 | May 2026 |
| 项目主页 | https://cowarobot-ai.github.io/ |
| 对比基线 | [[Drive-JEPA]], [[World4Drive]], [[Epona]], [[LAW]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.11550) / Code: N/A |

---

## 一句话总结

> DAWN 把世界预测与动作生成视作相互依赖的耦合变量，在语义潜空间里通过 **[[World Predictor]] ↔ [[World-Conditioned Action Denoiser]]** 的递归交互完成短程"action-aware"未来推演，实现感知-free 自动驾驶规划 SOTA。

---

## 核心贡献

1. **提出 [[WAIM|世界-动作交互模型]]（World-Action Interactive Model）范式**: 首次明确指出现有 [[World Action Model|WAM]] 缺失的"action-contingent reciprocity"原则——好的场景演化依赖于所选机动，而好的机动又依赖于场景如何演化，二者应作为自洽对联合推断。
2. **设计 DAWN 架构**: 在语义[[潜空间世界模型|潜空间]]内耦合一个 [[World Predictor|世界预测器]] 和一个 [[World-Conditioned Action Denoiser|世界条件动作去噪器]]，通过**短显式潜空间 rollout** 在零 rollout 与全 horizon rollout 之间取得最佳折中。
3. **在 NAVSIM v1 / nuScenes 上取得 SOTA**: 仅靠相机输入，perception-free 设定下达 **PDMS 89.1**（NAVSIM v1）与 **平均 L2 0.33m / 碰撞率 0.11%**（nuScenes），并以最优 TTC（Time-to-Collision）证明 interactive safety。

---

## 问题背景

### 要解决的问题

端到端 [[自动驾驶]] 中如何让 **未来场景预测** 与 **自车动作生成** 互相约束、共同推断，避免"过早承诺动作 → 场景预测沦为被动背景"或"先冻结未来 → 动作只能被动回应"的解耦缺陷。

### 现有方法的局限

- **并行式 WAM**（如 [[Drive-JEPA]]）: 两条分支并行输出 world 与 action，仅在特征层有相关但缺乏迭代互塑。
- **顺序 predict-then-plan**: 动作被条件在一个**冻结的未来**上，未来一旦预测错误动作无法纠正它，且全 horizon 像素 rollout 代价巨大。
- **Zero-rollout WAM**（如 Fast-WAM）: 直接从当前 latent 出发产生动作，丧失了"展望对方反应"的能力，在交互密集的城市道路场景受限。
- 大多 driving world model（GAIA-1, Drive-WM 等）在**像素空间**模拟未来，对长时规划不实用。

### 本文的动机

作者主张：在复杂交互场景下显式 future evolution 对推理动态障碍物仍然有用，但**不必跨越整个任务 horizon，也不必在像素空间进行**。如果把 rollout 限制在紧凑的语义[[潜空间世界模型|潜空间]]里、并让动作与世界在推理时**递归互修**，就能兼得 foresight 与 efficiency——这就是 [[WAIM]] 的中心论点。

---

## 方法详解

### 模型架构

DAWN 采用 **"双编码器 + 自编码 Resampler + 因果世界预测器 + DiT 风格动作去噪器" 的递归交互架构**：

- **输入**: 当前观测 $o$（多相机图像，4 帧 @ 2 Hz，256×512）+ 条件 $l$（自车状态、路由）
- **学生编码器**: $E_{stu}$（[[V-JEPA|V-JEPA 2 Large]]，patch size 16, tubelet 2），输出密集视觉特征 $u = E_{stu}(o)$
- **教师编码器**: $E_{tea}$（仅训练用，[[EMA]] 更新），与 [[I-JEPA|JEPA]] 系列一脉相承
- **[[Resampler|自编码 Resampler]]**: $R_{stu}, R_{tea}$，将密集特征压缩为 **16 个紧凑潜世界 token** $z = R_{stu}(u)$
- **[[World Predictor|世界预测器]]** $P_\theta$: 因果 [[Transformer]]，输入 $(z, c, a_{1:H})$ 输出短程未来潜世界 $z_{future}$
- **[[World-Conditioned Action Denoiser|世界条件动作去噪器]]** $G_\phi$: [[DiT]] 风格，承担两种角色——proposal 生成与 interactive refinement
- **Action Head** $H_{act}$: 解码最终轨迹 $\hat{\tau}$
- **输出**: 长度 $H=4s$ 的轨迹动作序列 $a_{1:H}$

### 核心模块

#### 模块1: 语义潜空间双编码 + Resampler 瓶颈

**设计动机**: 像素级未来生成代价高且与规划不直接对齐；用 [[JEPA]] 自监督 + Resampler 把场景压缩到紧凑 latent，使得**预测目标本身就是规划相关的语义**。

**具体实现**:
- 学生分支用大规模驾驶语料（[[OpenScene]]、[[DrivingDojo]]、[[CoVLA]]）做 [[V-JEPA]] 风格预训练
- 教师分支由学生 [[EMA]] 缓慢追踪，提供回归目标
- Resampler 训练为 token 级瓶颈，把 dense feature 压缩成 16 个潜世界 token，并能反解；保证下游预测器与去噪器都在同一紧凑 latent 上工作
- 消融显示 64 token 仅带来 +0.4 PDMS 但 latency 翻 3 倍，因此选 16

#### 模块2: 因果世界预测器（[[World Predictor]]）

**设计动机**: 提供 **action-conditioned 的短程未来 rollout**——即"如果我执行 $a_{1:H}$，场景在未来 $T_v$ 秒内潜变量会如何演化"。

**具体实现**:
- 因果 [[Transformer]] 处理 $(z, c, a_{1:H})$，输出未来 $T_v$ 时长（默认 4s）的潜空间未来 token $z_{future}$
- 仅做 **short rollout**，不是全 horizon 像素生成
- Stage 3 单独训练，目标是回归 Resampler 重编码的未来 frame token
- 第 4 阶段联合训练时与去噪器同步更新

#### 模块3: 世界条件动作去噪器（[[World-Conditioned Action Denoiser]]）

**设计动机**: 让动作生成 **显式条件在预测未来上**，并能在多轮中"看到自己上一轮决策对应的世界演化"后做修正。

**具体实现（双角色）**:

- **Proposal 角色**（首轮）:
$$a_{1:H}^{(0)} = G_\phi(q_{prop}, c, z)$$
仅看当前 latent，给出粗动作 proposal。
- **Refinement 角色**（之后每轮）:
$$a_{1:H}^{(r+1)} = G_\phi(q_{ref}^{(r)}, c, z_{future}^{(r)}, a_{1:H}^{(r)})$$
看上一轮的动作 **+** 该动作触发的未来世界 latent，重新去噪。
- $q_{prop}, q_{ref}^{(r)}$ 是角色特定 query embedding，让 DiT 区分两种调用模式。

#### 模块4: 递归交互推理（核心创新）

**设计动机**: 实现 WAIM 的自洽不动点 $(\hat{v}_{1:T}, \hat{a}_{1:H})$——通过 $K$ 轮 $\{$ 预测未来 → 用未来修正动作 → 用新动作再预测未来 $\}$ 的固定点迭代收敛到一致解。

**具体实现**（伪算法）:
1. 编码: $z = R_{stu}(E_{stu}(o))$
2. 初始 proposal: $a^{(0)} = G_\phi(q_{init}, c, z)$
3. for $k = 0, 1, ..., K-1$:
   - $z_{future}^{(k+1)} = P_\theta(z, c, a^{(k)})$
   - $a^{(k+1)} = G_\phi(q_{ref}^{(k)}, c, z_{future}^{(k+1)}, a^{(k)})$
4. 解码轨迹: $\hat{\tau} = H_{act}(a^{(K)})$

实验给出 $K=4$ 最佳（PDMS 87.9 → 5 轮反而 87.2），表明迭代有显著上升但**收益快速饱和**。

#### 模块5: 四阶段训练流水线

| Stage | 训练对象 | 数据 | 目的 |
|-------|----------|------|------|
| 1 | Student/Teacher Vision Encoder | OpenScene + DrivingDojo + CoVLA | [[V-JEPA]] 风格自监督表征 |
| 2 | Auto-Encoder Resampler | 同上 | token 级瓶颈，可编可解 |
| 3 | World Predictor | nuScenes / NAVSIM | learn short latent rollout |
| 4 | World Predictor + Action Denoiser 联合 | 目标数据集 | 联合训练 interactive refinement |

---

## 关键公式

### 公式1: [[World Action Model|WAM 联合分布]]

$$
p(v_{1:T}, a_{1:H} \mid o, l)
$$

**含义**: 标准世界-动作模型把未来世界状态 $v_{1:T}$ 与动作 $a_{1:H}$ 在同一联合分布下建模，通过对 $v_{1:T}$ 边缘化恢复策略 $p(a_{1:H}\mid o, l)$。

**符号说明**:
- $v_{1:T}$: 未来 $T$ 步世界状态（像素或 latent）
- $a_{1:H}$: 未来 $H$ 步动作序列
- $o$: 当前观测
- $l$: 任务/路由条件

### 公式2: [[WAIM|世界-动作自洽对]]

$$
\hat{v}_{1:T} = F_\theta(o, l, \hat{a}_{1:H}), \qquad \hat{a}_{1:H} = G_\phi(o, l, \hat{v}_{1:T})
$$

**含义**: WAIM 不再把 $v, a$ 视为联合采样的两个变量，而是寻找一对**互为函数**的不动点：未来由动作决定，动作又由未来决定。

**符号说明**:
- $F_\theta$: 世界预测器
- $G_\phi$: 世界条件动作生成器
- $\hat{v}_{1:T}, \hat{a}_{1:H}$: 自洽点（论文用 hat 表示固定点）

### 公式3: [[递归交互推理|交互迭代算子]]

$$
(v_{1:T}^{(k+1)}, a_{1:H}^{(k+1)}) = \mathcal{I}_\Theta\!\left(v_{1:T}^{(k)}, a_{1:H}^{(k)};\, o, l\right)
$$

**含义**: 用一个迭代算子 $\mathcal{I}_\Theta$ 把当前 $(v^{(k)}, a^{(k)})$ 映射到下一对，构成离散动力系统；DAWN 在推理时执行 $K=4$ 步该迭代以逼近自洽对。

**符号说明**:
- $\mathcal{I}_\Theta$: 组合算子，内部封装 $P_\theta$ 与 $G_\phi$
- $\Theta = (\theta, \phi)$: 预测器与去噪器的合并参数
- $k$: 迭代轮数索引

### 公式4: [[Resampler|学生侧潜表示]]

$$
u = E_{stu}(o), \qquad z = R_{stu}(u)
$$

**含义**: 当前观测先经视觉编码器得到密集特征 $u$，再被 Resampler 压成 16 token 的紧凑潜世界 $z$，下游所有推理都在 $z$ 上进行。

**符号说明**:
- $u$: dense visual features
- $z \in \mathbb{R}^{16 \times d}$: 紧凑潜世界 tokens
- $E_{stu}, R_{stu}$: 学生编码器与 Resampler

### 公式5: [[EMA|教师目标 latent]]

$$
z_{target} = R_{tea}\!\left(E_{tea}(o^{+})\right)
$$

**含义**: 教师分支用 [[EMA]] 维护，把"未来观测" $o^+$ 编码为目标 latent，供学生侧 JEPA 风格回归监督，避免 representation collapse。

**符号说明**:
- $o^+$: 未来时间点观测
- $E_{tea}, R_{tea}$: 教师侧 encoder 与 resampler（参数 EMA-update）
- $z_{target}$: 自监督学习的目标

### 公式6: [[World-Conditioned Action Denoiser|动作初始 proposal]]

$$
a_{1:H}^{(0)} = G_\phi(q_{prop}, c, z)
$$

**含义**: 在还没有任何未来 rollout 时，[[DiT]] 风格去噪器从查询 $q_{prop}$ 与当前 latent $z$ 中直接给出粗略动作序列。

**符号说明**:
- $q_{prop}$: proposal 角色 query embedding
- $c$: 条件 token（ego-state, route）
- $a^{(0)}$: 第 0 轮动作 proposal

### 公式7: [[World Predictor|action-conditioned 世界 rollout]]

$$
z_{future}^{(r)} = P_\theta(z, c, a_{1:H}^{(r)})
$$

**含义**: 把第 $r$ 轮的动作"打"进世界预测器，得到该动作假设下的短程未来潜表示——这是 WAIM 中 **action → world** 的方向。

**符号说明**:
- $z_{future}^{(r)}$: 第 $r$ 轮动作所诱导的未来 latent
- $P_\theta$: 因果 Transformer 世界预测器

### 公式8: [[World-Conditioned Action Denoiser|动作迭代修正]]

$$
a_{1:H}^{(r+1)} = G_\phi\!\left(q_{ref}^{(r)},\, c,\, z_{future}^{(r)},\, a_{1:H}^{(r)}\right)
$$

**含义**: 拿到该动作触发的未来后，去噪器**重新生成**动作；这是 WAIM 中 **world → action** 的方向，并与上一轮的动作一起作为去噪起点（类似 self-conditioning）。

**符号说明**:
- $q_{ref}^{(r)}$: refinement 角色 query embedding
- $z_{future}^{(r)}$: 上一步预测出的未来 latent
- $a^{(r)}, a^{(r+1)}$: 相邻两轮动作

### 公式9: 最终轨迹解码

$$
\hat{\tau} = H_{act}\!\left(a_{1:H}^{(K)}\right)
$$

**含义**: 经过 $K$ 轮 world-action 交互后得到的动作序列由 Action Head 解码为可执行的车辆轨迹（位置/姿态序列）。

**符号说明**:
- $K$: 总交互轮数（实验最优 $K=4$）
- $H_{act}$: 轨迹解码头
- $\hat{\tau}$: 最终输出轨迹

---

## 关键图表

> 论文共 11 张 Figure，下面全部列出；其中 1–5 为正文核心图，6–11 为附录定性/可视化。

### Figure 1: From WAMs to WAIM

![Figure 1](https://arxiv.org/html/2605.11550v1/x1.png)

**说明**: 对比 4 种 world-action 模型范式——(a) 并行 WAM，(b) 顺序 predict-then-plan，(c) 零 rollout 直接策略，(d) 本文 WAIM——核心区别在 WAIM 保留**短** latent rollout，并在推理时 **递归耦合** $F_\theta$ 与 $G_\phi$。

### Figure 2: Overview of DAWN

![Figure 2](https://arxiv.org/html/2605.11550v1/x2.png)

**说明**: DAWN 完整流水线。训练时 Student/Teacher Vision-Encoder + AE Resampler 学紧凑潜世界 token，[[World Predictor]] 监督短 latent rollout，[[World-Conditioned Action Denoiser]] 训练 trajectory 生成；推理时去噪器先从 resampler latent 初始化动作，再用预测器 rollout 递归修正——**全程不渲染像素未来**。

### Figure 3: Effect of Interactive Rounds

![Figure 3](https://arxiv.org/html/2605.11550v1/x3.png)

**说明**: PDMS 随交互轮数变化曲线。1 → 4 轮 PDMS 从 85.2 单调上升到 87.9，第 5、6 轮分别回落到 87.2、86.9，说明 **交互显著有效但快速饱和**，并非越多越好（多余轮数会引入累计误差）。

### Figure 4: Qualitative Planning Results

![Figure 4](https://arxiv.org/html/2605.11550v1/x4.png)

**说明**: 5 个典型驾驶场景（复杂路口、窄路、曲线连接）的轨迹对比，上排前视图、下排 BEV。对比 Human / [[Drive-JEPA]] / DAWN：DAWN 轨迹更贴合道路几何，并在曲线交叉口处与人类驾驶行为最一致。

### Figure 5: Latent World Rollout Design Space

![Figure 5](https://arxiv.org/html/2605.11550v1/x5.png)

**说明**: 一个二维设计谱——横轴是 rollout 长度（0 到 full horizon），纵轴是 rollout 空间（latent vs pixel）。Fast-WAM 在最左端零 rollout，传统 predict-then-plan 在最右端 pixel full rollout，DAWN 处于"短 latent rollout"中间区域，是论文论证的"甜蜜点"。

### Figure 6: Additional Qualitative Planning Results

![Figure 6](https://arxiv.org/html/2605.11550v1/x6.png)

**说明**: 附录补充更多场景的定性轨迹对比，进一步展示 DAWN 在 narrow streets 与 sharp turns 上的稳定性。

### Figure 7: Qualitative Future Prediction Results

![Figure 7](https://arxiv.org/html/2605.11550v1/x7.png)

**说明**: 通过 Resampler 反解将预测的未来 latent 投影回像素空间的可视化（**仅用于 inspection，不用于规划**），证明潜世界 token 真的编码了语义上合理的未来场景。

### Figure 8: More Future Prediction Results

![Figure 8](https://arxiv.org/html/2605.11550v1/x8.png)

**说明**: 第二批未来预测可视化，覆盖更多 ego maneuver 类型（变道、减速、汇入）。

### Figure 9: Further Future Prediction Results

![Figure 9](https://arxiv.org/html/2605.11550v1/x9.png)

**说明**: 第三批可视化样本，展示对动态行人/前车反应的预测能力。

### Figure 10: Feature Map Visualization

![Figure 10](https://arxiv.org/html/2605.11550v1/x10.png)

**说明**: 对学生编码器输出的特征图可视化，验证 [[V-JEPA]] 预训练学到的语义对齐性。

### Figure 11: Additional Feature Map Visualization

![Figure 11](https://arxiv.org/html/2605.11550v1/x11.png)

**说明**: 更多特征图样本，覆盖城市/郊区/夜间场景。

### Table 1: NAVSIM v1 主结果（perception-free 区）

| Type | Method | Inputs | NC↑ | DAC↑ | EP↑ | C↑ | TTC↑ | PDMS↑ |
|------|--------|--------|-----|------|-----|-----|------|-------|
| Perception-based | Transfuser | C & L | 97.7 | 92.8 | 79.2 | 100 | 92.8 | 84.0 |
| | Hydra-MDP | C & L | 98.4 | 97.7 | 85.0 | 100 | 94.5 | 89.9 |
| | Hydra-MDP++ | C & L | 97.6 | 96.0 | 80.4 | 100 | 93.1 | 86.6 |
| | DiffusionDrive | C & L | 98.2 | 96.2 | 82.2 | 100 | 94.7 | 88.1 |
| | GoalFlow | C & L | 98.4 | 98.3 | 85.0 | 100 | 94.6 | 90.3 |
| | DriveDPO | C & L | 98.5 | 98.1 | 84.3 | 100 | 94.8 | 90.0 |
| | iPad | Camera | 99.2 | 97.4 | 87.8 | 99.7 | 96.3 | 91.7 |
| | DriveSuprim | Camera | 98.6 | 98.6 | 91.3 | 100 | 95.5 | 93.5 |
| Perception-free | LAW | C & L | 97.4 | 93.3 | 78.8 | 100 | 91.9 | 83.8 |
| | World4Drive | C & L | 97.4 | 94.3 | 79.9 | 100 | 92.8 | 85.1 |
| | Epona | Camera | 97.9 | 95.1 | 80.4 | 99.9 | 93.8 | 86.2 |
| | Drive-JEPA | Camera | 98.7 | 96.2 | 82.9 | 100 | 95.5 | 89.0 |
| | **DAWN*** | Camera | 98.2 | 95.8 | 84.2 | 100 | 95.8 | 87.9 |
| | **DAWN** | Camera | **98.7** | 95.9 | **84.3** | **100** | **96.0** | **89.1** |

\* DAWN* 为 256×256 分辨率消融版本。

**说明**: DAWN 拿下 perception-free 区 **PDMS 与 TTC 双第一**，且最佳 EP（Ego Progress）；与 Drive-JEPA 相比 PDMS +0.1 但 TTC +0.5，体现 interactive safety 优势。

### Table 2: nuScenes 主结果

| Method | L2 1s↓ | L2 2s↓ | L2 3s↓ | L2 Avg↓ | Coll. 1s↓ | Coll. 2s↓ | Coll. 3s↓ | Coll. Avg↓ |
|--------|--------|--------|--------|---------|-----------|-----------|-----------|------------|
| ST-P3 | 1.33 | 2.11 | 2.90 | 2.11 | 0.23 | 0.62 | 1.27 | 0.71 |
| OccNet | 1.29 | 2.13 | 2.99 | 2.13 | 0.21 | 0.59 | 1.37 | 0.72 |
| UniAD | 0.48 | 0.96 | 1.65 | 1.03 | 0.05 | 0.17 | 0.71 | 0.31 |
| VAD | 0.41 | 0.70 | 1.05 | 0.72 | 0.07 | 0.18 | 0.43 | 0.23 |
| PPAD | 0.31 | 0.56 | 0.87 | 0.58 | 0.08 | 0.12 | 0.38 | 0.19 |
| GenAD | 0.28 | 0.49 | 0.78 | 0.52 | 0.08 | 0.14 | 0.34 | 0.19 |
| BEV-Planner | 0.30 | 0.52 | 0.83 | 0.55 | 0.10 | 0.37 | 1.30 | 0.59 |
| LAW | 0.26 | 0.57 | 1.01 | 0.61 | 0.14 | 0.21 | 0.54 | 0.30 |
| World4Drive | 0.23 | 0.47 | 0.81 | 0.50 | 0.00 | 0.12 | 0.33 | 0.16 |
| WorldRFT | 0.21 | 0.44 | 0.76 | 0.47 | 0.10 | 0.11 | 0.23 | 0.15 |
| **DAWN (Ours)** | **0.17** | **0.31** | **0.52** | **0.33** | **0.00** | **0.10** | **0.23** | **0.11** |

**说明**: 全表 L2 与 Coll. 全部第一；平均 L2 从前 SOTA 0.47m 降到 **0.33m（-30%）**，平均碰撞率 0.11% 也是历史最低。

### Table 3: 组件消融

| Resampler | Predictor | Interactive | PDMS↑ |
|-----------|-----------|-------------|-------|
|           |           |             | 82.9 |
| ✓         |           |             | 82.8 |
| ✓         | ✓         |             | 85.2 |
| ✓         | ✓         | ✓           | **87.9** |

**关键发现**: 仅加 Resampler 几乎不涨（语义瓶颈本身不带规划增益），加 Predictor 涨 +2.4，再加 Interactive 又涨 +2.7——证明 **交互推理本身是独立增益源**，不是 Predictor 的副产物。

### Table 4: Resampler Token 数消融

| # Tokens | PDMS↑ | Latency (ms)↓ |
|----------|-------|----------------|
| 16 | 82.8 | 331.253 |
| 64 | 83.2 | 963.645 |

**说明**: 把 token 数 ×4 只换 +0.4 PDMS 却 ×2.9 倍延迟——**16 是 Pareto 最优点**。

### Table 5: World-Action 耦合方向消融

| Method | NC↑ | DAC↑ | EP↑ | C↑ | TTC↑ | PDMS↑ |
|--------|-----|------|-----|-----|------|-------|
| DAWN | 98.2 | 95.8 | 84.2 | 100 | 95.7 | **87.9** |
| w/o World → Action | 96.6 | 91.9 | 78.6 | 99.9 | 91.6 | 81.6 |
| w/o Action → World | 97.3 | 94.3 | 80.2 | 100 | 92.7 | 84.9 |

**关键发现**: 拆掉 **World → Action**（动作不看预测未来）损失最大（-6.3 PDMS），证明"动作必须以未来为条件"是 WAIM 的主要价值；拆掉 **Action → World**（预测器不看动作）也损失 -3.0 PDMS，说明反向耦合也是必须的。

### Table 6: World Rollout Horizon 消融

| $T_v$ | $H_a$ | PDMS↑ | w/o Interactive↑ | Latency (ms)↓ |
|-------|-------|-------|------------------|----------------|
| 0s | 4s | 82.8 | 82.8 | 331.253 |
| 1s | 4s | 84.7 | 83.9 | 503.261 |
| 2s | 4s | 87.3 | 84.3 | 690.540 |
| 3s | 4s | 87.5 | 84.6 | 849.512 |
| 4s | 4s | **87.9** | 85.2 | 1067.975 |

**说明**: 同等延迟下"有交互"始终碾压"无交互"；rollout 2–3s 已抓住 80% 收益，4s 取得最佳 PDMS——证明 **短 rollout 已够**，但仍单调有益。

### Table 7: NAVSIM v2 主结果

| Method | NC↑ | DAC↑ | DDC↑ | TL↑ | EP↑ | TTC↑ | LK↑ | HC↑ | EC↑ | EPDMS↑ |
|--------|-----|------|------|-----|-----|------|-----|-----|-----|--------|
| Transfuser | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 |
| Hydra-MDP++ | 98.5 | 98.5 | 99.5 | 99.7 | 87.4 | 97.9 | 95.8 | 98.2 | 75.7 | 85.6 |
| iPad | 98.7 | 98.0 | 98.9 | 99.8 | 86.6 | 98.3 | 97.2 | 98.3 | 74.6 | 85.8 |
| DriveSuprim | 98.4 | 98.6 | 99.6 | 99.8 | 90.5 | 97.8 | 97.0 | 98.3 | 78.6 | 87.1 |
| Drive-JEPA | 98.4 | 98.6 | 99.1 | 99.8 | 88.4 | 97.8 | 97.6 | 97.9 | 84.8 | 87.8 |
| **DAWN** | 97.3 | 92.0 | 99.1 | 99.7 | 87.4 | 96.6 | 96.0 | 98.3 | **85.5** | 83.2 |

**说明**: NAVSIM v2 上 DAWN 仅夺得 **EC（Extended Comfort）第一**，整体 EPDMS 83.2 落后 Drive-JEPA 4.6——v2 的更严格规则与 DAWN 的 perception-free 设定不完全契合，作者承认这是当前主要短板。

### Table 8: NAVSIM v1 详细组件消融

| AE Resampler | Predictor | Interactive | NC↑ | DAC↑ | EP↑ | TTC↑ | C↑ | PDMS↑ |
|--------------|-----------|-------------|-----|------|-----|------|-----|-------|
|              |           |             | 97.1 | 92.2 | 78.8 | 91.5 | 100 | 82.9 |
| ✓            |           |             | 97.2 | 92.2 | 78.7 | 91.7 | 100 | 82.8 |
| ✓            | ✓         |             | 97.4 | 94.3 | 80.4 | 91.5 | 100 | 85.2 |
| ✓            | ✓         | ✓           | **98.2** | **95.8** | **84.2** | **95.8** | 100 | **87.9** |

**说明**: 比 Table 3 多出多指标列；最显著的是 **TTC 从 91.5 跳到 95.8（+4.3）**——交互推理改善了 interactive safety 而不仅是 trajectory 准确度。

### Table 9: 交互轮数细化

| # Rounds | NC↑ | DAC↑ | EP↑ | C↑ | TTC↑ | PDMS↑ |
|----------|-----|------|-----|-----|------|-------|
| 1 | 97.4 | 94.3 | 80.4 | 100 | 91.5 | 85.2 |
| 2 | 97.8 | 95.1 | 81.6 | 100 | 94.1 | 86.4 |
| 3 | 98.1 | 95.6 | 82.8 | 100 | 95.6 | 86.9 |
| 4 | **98.2** | **95.8** | **84.2** | 100 | **95.8** | **87.9** |
| 5 | 98.1 | 95.4 | 83.9 | 100 | 95.7 | 87.2 |
| 6 | 98.0 | 95.6 | 82.8 | 100 | 95.6 | 86.9 |

**说明**: 与 Figure 3 互补——逐指标看，**4 轮在 NC/DAC/EP/TTC/PDMS 五项全部最优**。

### Table 10: Resampler Token 数（多指标版）

| # Tokens | NC↑ | DAC↑ | EP↑ | C↑ | TTC↑ | PDMS↑ | Latency (ms)↓ |
|----------|-----|------|-----|-----|------|-------|----------------|
| 16 | 97.2 | 92.2 | 78.7 | 100 | 91.7 | 82.8 | 331.253 |
| 64 | 97.2 | 92.4 | 78.8 | 100 | 91.7 | 83.2 | 963.645 |

**说明**: Table 4 的多指标版本，验证 16 token 在每项指标上都几乎与 64 持平——容量瓶颈不是主要矛盾。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| OpenScene | 大规模驾驶语料 | 用于视觉编码器预训练 | Stage 1/2 |
| DrivingDojo | 多源驾驶视频 | 增强预训练多样性 | Stage 1/2 |
| CoVLA | Vision-Language 驾驶数据 | 增强预训练多样性 | Stage 1/2 |
| nuScenes | 6 相机 + LiDAR 多模态驾驶数据集 | 端到端规划评估 | Stage 3/4 + 测试 |
| NAVSIM v1 / v2 | 仿真闭环规则评估 | 规则导向 PDMS / EPDMS 评估 | Stage 3/4 + 测试 |

### 评估指标

- **NAVSIM v1**: NC（No-at-fault Collision）、DAC（Drivable Area Compliance）、EP（Ego Progress）、C（Comfort）、TTC（Time-to-Collision）→ 聚合为 **PDMS**
- **NAVSIM v2**: 进一步加入 DDC（Driving Direction Compliance）、TL（Traffic Light）、LK（Lane Keeping）、HC（History Comfort）、EC（Extended Comfort），聚合为 **EPDMS**
- **nuScenes**: 1s/2s/3s 与平均的 **L2 轨迹误差** 与 **碰撞率**

### 实现细节

- **视觉骨干**: [[V-JEPA|V-JEPA 2 Large]]，patch size 16，tubelet 2
- **采样**: 2 Hz，观测 4 帧，预测 12 帧
- **分辨率**: 主结果 256×512，消融 256×256
- **优化器**: AdamW，peak lr $1\times10^{-4}$，initial $5\times10^{-5}$，weight decay 0.04，8 warmup epochs
- **训练**: bfloat16 混合精度，150 epochs
- **硬件**: 80 × NVIDIA A100 GPU
- **推理 latency**: 最优配置 $T_v=4$s 下约 1.07 s/sample（含 4 轮交互）

### 可视化结果

定性观察（Figure 4/6/7-9）:
- 复杂路口与曲线连接处 DAWN 轨迹**最贴合道路几何**，且与人类驾驶最接近
- Resampler 反解的未来 frame 在前车减速、行人横穿等场景下能合理预测，但分辨率较低（语义级而非像素级）
- 特征图可视化（Figure 10/11）显示 V-JEPA 学到的特征在车道线/动态目标上有强响应

---

## 批判性思考

### 优点

1. **范式贡献清晰**: WAIM 的"自洽对 + 迭代算子"提法比此前并行/顺序 WAM 都更高层，可统一描述 Fast-WAM、Drive-JEPA 等多种现有方法为其退化情形（见 Figure 5）。
2. **latent + short rollout 双重瓶颈**: 既避开像素未来生成的算力坑，又把 rollout 长度限制在 4s——在效率/性能间找到非常清晰的 sweet spot（Table 6 / Figure 5）。
3. **耦合方向消融严谨**: Table 5 同时拆 W→A 和 A→W，定量证明**双向耦合都重要**，不只是"动作看未来"的单向条件。
4. **TTC 提升说明 interactive safety**: NAVSIM v1 TTC 96.0 全榜第一，呼应"交互推理对动态障碍物更友好"的论文主旨。

### 局限性

1. **NAVSIM v2 落后**: EPDMS 83.2 vs Drive-JEPA 87.8，作者也承认；尤其 DAC、TTC 在 v2 更严标准下退步——perception-free 设定下场景理解的细粒度仍不足。
2. **训练流水线重**: 4 stage + 80 A100 + 150 epochs，**复现门槛极高**；论文未开源代码。
3. **无收敛/安全保证**: 递归迭代算子 $\mathcal{I}_\Theta$ 没有不动点存在性证明，也没有 worst-case latency 上界（轮数自适应没尝试）。
4. **依赖 V-JEPA 预训练质量**: 学生编码器在 OpenScene/DrivingDojo/CoVLA 上预训练，但论文未给迁移到其他城市/天气的鲁棒性结果。
5. **latent 可解释性差**: 与显式 BEV occupancy / 物体级 world model 比，Resampler 16 token 的语义不直观，难以调试。
6. **rollout 长度短**: 4s 适合典型 driving horizon，但长程绕行/路口排队等场景可能需要更长 horizon——而延展会显著增加 latency。

### 潜在改进方向

1. **自适应交互轮数**: 在易场景早停（1 轮即收敛），在复杂场景多迭代——可用 token-level consensus 度量做停止准则。
2. **像素+latent 混合 rollout**: 难场景调用低分辨率 pixel rollout 做 inspect，常规场景保持纯 latent，兼顾 efficiency 与 interpretability。
3. **形式化 WAIM 自洽性**: 给迭代算子加 Lipschitz 约束或 contraction map 设计，提供不动点收敛理论保证。
4. **扩展到通用 [[Embodied AI|具身]] 规划**: WAIM 的"动作-未来自洽对"原则其实和机器人 manipulation 的 receding horizon control 高度同构，是否能迁移到 [[VLA]] 任务值得探索（与 [[Pi05]] / [[GR00T N1.7]] 等结合）。

### 可复现性评估

- [ ] 代码开源（暂未发布）
- [ ] 预训练模型（未提供）
- [x] 训练细节完整（实现细节、数据集、超参完整）
- [x] 数据集可获取（OpenScene/DrivingDojo/CoVLA/nuScenes/NAVSIM 全公开）

---

## 关联笔记

### 基于

- [[V-JEPA]]: 学生/教师视觉编码器骨架，提供 latent 表示空间
- [[JEPA]]: 自监督学习范式
- [[EMA]]: 教师网络更新机制
- [[DiT]]: 动作去噪器结构
- [[Transformer]]: 世界预测器主干
- [[World Action Model]]: 一般化范式，DAWN 是其交互式扩展

### 对比

- [[Drive-JEPA]]: 同样 perception-free + JEPA latent，但 world/action 并行，缺乏交互（NAVSIM v1 PDMS 89.0 vs DAWN 89.1）
- [[World4Drive]]: 顺序 predict-then-plan 代表（nuScenes L2 0.50 vs DAWN 0.33）
- [[Epona]]: perception-free 单 camera baseline
- [[LAW]]: latent action world model 早期工作
- [[UniAD]]: end-to-end 多任务 baseline（带感知）

### 方法相关

- [[WAIM]]: 本文提出的新范式
- [[World Predictor]]: DAWN 中的 $F_\theta$ 模块
- [[World-Conditioned Action Denoiser]]: DAWN 中的 $G_\phi$ 模块
- [[递归交互推理]]: WAIM 的推理时迭代算子
- [[Resampler]]: token 压缩瓶颈
- [[潜空间世界模型]]: 整体范式归类
- [[自动驾驶]]: 应用领域

### 硬件/数据相关

- [[nuScenes]]: 端到端规划评估
- [[NAVSIM]]: PDMS / EPDMS 评估
- [[OpenScene]], [[DrivingDojo]], [[CoVLA]]: 预训练数据集

---

## 速查卡片

> [!summary] DAWN: World-Action Interactive Models
> - **核心**: 把 world prediction 与 action generation 视为相互依赖的自洽对，推理时递归互修（WAIM 范式）
> - **方法**: V-JEPA 视觉骨干 + Resampler 16-token 潜世界 + 因果 Transformer World Predictor + DiT 风格 World-Conditioned Action Denoiser；4 轮 short latent rollout 交互
> - **结果**: NAVSIM v1 perception-free PDMS **89.1** SOTA（TTC 96.0 第一）；nuScenes 平均 L2 **0.33m**、碰撞率 **0.11%** 双 SOTA
> - **代码**: 未开源；项目页 https://cowarobot-ai.github.io/

---

*笔记创建时间: 2026-05-14*
