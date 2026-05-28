---
title: "EgoExo-WM: Unlocking Exo Video for Ego World Models"
method_name: "EgoExo-WM"
authors: [Danny Tran, Roberto Martín-Martín, Kristen Grauman]
year: 2026
venue: arXiv
tags: [world-model, egocentric-video, exo-to-ego, human-motion, video-diffusion, mpc-planning, embodied-ai]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.15477v2
created: 2026-05-27
---

# 论文笔记：EgoExo-WM: Unlocking Exo Video for Ego World Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | The University of Texas at Austin |
| 日期 | May 2026 |
| 项目主页 | https://vision.cs.utexas.edu/projects/EgoExo-WM/ |
| 对比基线 | [[PEVA]] · [[EgoControl]] · [[UniEgoMotion]] · Ego-WM |
| 链接 | [arXiv](https://arxiv.org/abs/2605.15477) / [Project](https://vision.cs.utexas.edu/projects/EgoExo-WM/) / Code: N/A |

---

## 一句话总结

> EgoExo-WM 通过 EgoX-Body 把第三人称视频转换成"第一人称视频 + 3D 人体动作"对，扩充[[Egocentric World Model|第一人称世界模型]]训练数据并支撑[[MPC|MPC]] 规划。

---

## 核心贡献

1. **EgoX-Body 视角转换**: 在 [[EgoX]] 的基础上引入显式人体骨架（[[SAM 3D Body]]）与手部运动学（[[HaMeR]]）双重先验，把任意第三人称视频可靠转换为视角对齐的第一人称视频。
2. **以 3D 人体动作为统一动作空间**: 用 69 维（3 维根平移 + 22 关节欧拉角）的人体动作 $a_t$ 同时驱动真实 ego 数据与转换后的 exo 数据，构成统一[[Action-Conditioned World Model|动作条件世界模型]]。
3. **从 in-the-wild 第三人称视频获益**: 仅用约 10 小时转换后的 exo 数据（HowTo100M / CrossTask / 100 Days of Hands）就让模型在 HOMAGE/LEMMA 上的 L2 latent 误差较 [[EgoControl]] 下降约 42%，并在视觉目标条件的 MPC 全身规划中取得最低 MPJPE。

---

## 问题背景

### 要解决的问题

[[Egocentric World Model|第一人称世界模型]] 需要"动作-观测"配对数据来学习人体行为对视野的影响，但带运动捕捉的 ego 数据极少（如 Nymeria 仅约 200 小时），远远不足以支撑大规模训练；而互联网上的[[exocentric video|第三人称视频]]存量巨大却没法直接使用。

### 现有方法的局限

- [[PEVA]] 把动作压缩成低维向量直接条件，但缺乏与视觉对齐的几何结构，且训练数据局限于带配对动作的 ego 数据。
- [[EgoControl]] 用相机姿态作为动作条件，无法描述"全身行为"对视野的影响（如蹲下、转身、双手姿态）。
- 直接做 exo→ego 视频翻译（[[EgoX]]）会因缺少结构约束导致生成不可靠：人体姿态、手物交互错位，从而无法当作可靠的"伪 ego 数据"。
- 没有显式的人体运动学先验，模型难以同时跨域并保持物理可行的人体行为。

### 本文的动机

人体动作是 ego 与 exo 视频天然的**桥梁**：同一动作在不同视角下产生不同的视觉观测，但 3D 人体姿态本身是视角无关的。因此，作者把 3D 人体动作作为统一动作空间，先用 EgoX-Body 把 exo 视频转换为对应的 ego 视频，再用 (转换 ego, 3D 动作) 对增强训练，从而把"互联网视频"变成"可用的 ego 训练数据"。

---

## 方法详解

### 模型架构

EgoExo-WM 采用**潜空间[[扩散变换器]] + 人体动作条件**的整体框架，包含三个分离训练的子系统：

- **输入**: ego 观测历史 $x_{t-H+1:t}$、当前 3D 人体动作 $a_t \in \mathbb{R}^{69}$
- **视觉编码**: 冻结的 [[DINOv3]] ViT-L/16，把 $224 \times 224$ 图像编码为 $14 \times 14 = 196$ 个 1024 维 patch token
- **世界模型主干**: [[CDiT|CDiT-L/2]]，24 层、1024 隐维、16 注意力头，预测下一帧的 [[潜空间世界模型|视觉潜变量]] $\hat{z}_{t+1}$
- **辅助监督**: 冻结的 6 层 [[Transformer]] 头从 $\hat{z}_{t+1}$ 预测 $16 \times 16$ [[手腕热图|手腕热图]]，提供细粒度交互监督
- **Exo→Ego 数据生成**: 基于扩散的 [[EgoX-Body]] 视频翻译模型（独立训练）
- **规划**: [[Latent MPC|潜空间 MPC]]，由 [[UniEgoMotion]] 采样候选动作

### 核心模块

#### 模块1: World Model 主干

**设计动机**: 用人体动作条件而非低维相机/末端动作，捕捉"全身行为如何影响第一人称视野"。

**具体实现**:
- 动作 $a_t$ 编码 3D 人体动作：3 维根平移 + 22 个关节的欧拉角共 66 维，总计 69 维。
- 视觉表示采用 [[DINOv3]] 潜空间而非像素空间，避免像素级生成开销。
- 前向：$\hat{z}_{t+1} = f_\theta(z_{t-H+1:t}, a_t)$，[[自回归]] rollout 时把 $\hat{z}$ 反馈回历史窗口。
- 训练数据混合 Nymeria 真 ego 数据与 EgoX-Body 生成的伪 ego 数据，二者共享同一 69 维动作空间。

#### 模块2: 手腕一致性损失（Wrist Consistency）

**设计动机**: 仅用全局 latent 误差不足以约束"手在哪里"，而手位置是 ego 视野中最重要的交互信号。

**具体实现**:
- 用 [[ViTPose]] 抽取每帧的手腕 2D 关键点，渲染成高斯热图作为伪标签 $V_{t+1}$。
- 6 层 Transformer 解码头 $h_\phi$ 从预测潜变量解码出 $\hat{V}_{t+1} = h_\phi(\hat{z}_{t+1})$。
- 与主 latent 损失一起监督，强迫世界模型在潜空间保留手腕几何信息。

#### 模块3: EgoX-Body 视角转换

**设计动机**: [[EgoX]] 缺少显式的人体先验，转换后的 ego 视频在姿态与手物交互上常常错位；EgoX-Body 在 exo/ego 两端都加上结构化骨架引导。

**具体实现（推理 pipeline）**:
- **Exo 侧**: 用 [[SAM 3D Body]] 抽出 [[SMPL-X]] 骨架并合成"骨架叠加"在 exo 帧上，作为视角无关的行为描述；用 [[VIPE|ViPE]] 抽取 4D 场景几何提供空间上下文。
- **Ego 侧**: 训练时用 [[HaMeR]] 拟合的手部姿态投影到 ego 视角作为骨架先验；推理时把估计的 3D 手位置从 SMPL-X 投影到 ego 相机，作为"手部条件"渲染。
- **去噪**: 两路 latent（干净 exo + 噪声 / 骨架叠加 exo + ego 手骨架）按通道拼接 + masking，输入到[[视频扩散模型]]生成 49 帧（约 2 秒）的伪 ego 视频。
- **运行时**: $384 \times 384$ 分辨率、49 帧，在 4× GH200 上每段约 3.25 分钟。

#### 模块4: 规划框架（MPC）

**设计动机**: 把世界模型从被动预测器变为主动规划器，让"目标图像 + 动作候选"在潜空间内被排序。

**具体实现**:
- 候选动作由 [[UniEgoMotion]] 采样，得到 $N=4$ 条 8 帧的物理可行人体动作 $\{a^{(i)}_{t:t+H}\}$。
- 每条候选在世界模型中自回归 rollout：$z^{(i)}_{t+1:t+H} = f_\theta(z_t, a^{(i)}_{t:t+H})$。
- 用 [[DINOv3]] 编码视觉目标图像得到 $z_g$，按 L2 latent 距离排序，选最小者执行。

---

## 关键公式

### 公式1: [[Action-Conditioned World Model|世界模型前向预测]]

$$
\hat{z}_{t+1} = f_\theta(z_{t-H+1:t}, a_t)
$$

**含义**: 给定历史 $H$ 帧的 ego 视觉潜变量与当前 3D 人体动作，预测下一帧潜变量。

**符号说明**:
- $z_t = E(x_t) \in \mathbb{R}^{196 \times 1024}$: 冻结 [[DINOv3]] 编码的 patch token
- $a_t \in \mathbb{R}^{69}$: 3 维根平移 + 22 关节欧拉角
- $f_\theta$: [[CDiT|CDiT-L/2]] 主干，24 层、1024 隐维

### 公式2: [[潜空间世界模型|Latent 预测损失]]

$$
\mathcal{L}_{\text{latent}} = \| \hat{z}_{t+1} - z_{t+1} \|_2^2
$$

**含义**: MSE 监督预测潜变量逼近真实下一帧编码。

**符号说明**:
- $z_{t+1} = E(x_{t+1})$: ground-truth 下一帧的 DINOv3 编码

### 公式3: [[手腕热图|手腕一致性损失]]

$$
\mathcal{L}_{\text{wrist}} = \| \hat{V}_{t+1} - V_{t+1} \|_2^2
$$

**含义**: 在预测潜空间中保持手腕位置可解码，避免世界模型抹掉细粒度交互信息。

**符号说明**:
- $\hat{V}_{t+1} = h_\phi(\hat{z}_{t+1})$: 6 层 Transformer 解码出的 $16 \times 16$ 热图
- $V_{t+1}$: 由 [[ViTPose]] 关键点渲染的高斯热图伪标签

### 公式4: 总训练目标

$$
\mathcal{L} = \mathcal{L}_{\text{latent}} + \lambda \, \mathcal{L}_{\text{wrist}}, \quad \lambda = 1
$$

**含义**: 主 latent 误差 + 手腕一致性的加权和；论文取 $\lambda = 1$。

**符号说明**:
- $\lambda$: 手腕损失权重，固定为 1

### 公式5: [[Latent MPC|MPC 候选评分]]

$$
C^{(i)} = \| z^{(i)}_{t+H} - z_g \|_2^2
$$

**含义**: 把候选动作序列 rollout 到 horizon 末端，与目标潜变量的 L2 距离作为代价。

**符号说明**:
- $z^{(i)}_{t+H}$: 候选 $i$ 在世界模型中 rollout 后的末端潜变量
- $z_g = E(x_g)$: 目标图像 $x_g$ 经 DINOv3 编码

### 公式6: 动作选择

$$
\hat{i} = \arg\min_{i \in \{1,\dots,N\}} C^{(i)}, \qquad a^*_{t:t+H} = a^{(\hat{i})}_{t:t+H}
$$

**含义**: 在 $N=4$ 条 UniEgoMotion 提议中选代价最小的一条执行。

**符号说明**:
- $N = 4$: 每步采样的候选动作数
- $H$: 规划地平线（实验中 8 帧 / 约 2 秒）

---

## 关键图表

### Figure 1: Overview / 系统概览

![Figure 1](https://arxiv.org/html/2605.15477v2/x1.png)

**说明**: 上半部分对比 ego 与 exo 视频"可见性鸿沟"——同一动作在两种视角下观测完全不同；下半部分展示 EgoExo-WM 的工作流：用 3D 人体动作作为桥梁，把 exo 视频转换为 ego，再用于训练[[Action-Conditioned World Model|动作条件世界模型]]并进行目标条件 [[MPC|规划]]。

### Figure 2: World Model Training Pipeline / 训练流程

![Figure 2](https://arxiv.org/html/2605.15477v2/x2.png)

**说明**: 训练数据双路汇入：(1) Nymeria 真 ego 视频 + 配对的 SMPL-X 动作；(2) HowTo100M/CrossTask/100 Days of Hands 等 exo 视频经 EgoX-Body 转换为伪 ego 视频，并用 [[SAM 3D Body]] 抽出 3D 动作。两路共享 69 维动作空间送入 [[CDiT|CDiT-L/2]] 主干训练。

### Figure 3: EgoX vs. EgoX-Body Qualitative / 视角转换对比

![Figure 3](https://arxiv.org/html/2605.15477v2/x3.png)

**说明**: 同一 exo 输入下，原始 [[EgoX]] 生成的 ego 视频常出现手部错位、姿态飘移；EgoX-Body 引入骨架与手部先验后，生成视频与真实 ego 的关键点对齐显著改善，尤其在"切菜/抓物"等手物交互上。

### Figure 4: EgoX-Body Inference Overview / 推理 pipeline

![Figure 4](https://arxiv.org/html/2605.15477v2/x4.png)

**说明**: 单 exo 视频输入 → SAM 3D Body 抽 SMPL-X 骨架 + ViPE 抽 4D 场景几何 → 骨架叠加到 exo 帧；3D 手位投影到 ego 视角作为 hand prior；两路 latent 通道拼接后输入到[[视频扩散模型]]去噪生成 49 帧伪 ego 视频。

### Figure 5: Qualitative Planning Results / 规划定性结果

![Figure 5](https://arxiv.org/html/2605.15477v2/x5.png)

**说明**: 给定起始 ego 帧与视觉目标图，对比 [[UniEgoMotion]] 单独采样、Ego-WM 排序、EgoExo-WM 排序三者；EgoExo-WM 选择的全身动作更接近 ground-truth，导航与操作场景下都更准。

### Table 1: Open-Loop World Model Evaluation

报告两段：**2-second（末帧）** / **avg（rollout 平均）** 两列，左 L2 越低越好，右 PCK@20 越高越好。

| 方法 | HOMAGE L2 (2s/avg) | HOMAGE PCK@20 (2s/avg) | LEMMA L2 | LEMMA PCK@20 | Ego-Exo4D-Bike L2 | Ego-Exo4D-Bike PCK@20 | Ego-Exo4D-Cooking L2 | Ego-Exo4D-Cooking PCK@20 |
|------|------|------|------|------|------|------|------|------|
| [[PEVA]]-L | 0.115/0.108 | 0.326/0.402 | 0.115/0.107 | 0.439/0.448 | 0.108/0.099 | 0.340/0.427 | 0.105/0.097 | 0.210/0.298 |
| PEVA-XL | 0.112/0.107 | 0.308/0.357 | 0.110/0.106 | 0.324/0.265 | 0.106/0.096 | 0.319/0.397 | 0.105/0.097 | 0.210/0.301 |
| PEVA-XXL | 0.109/0.103 | 0.321/0.365 | 0.109/0.101 | 0.363/0.420 | 0.103/0.093 | 0.255/0.420 | 0.102/0.095 | 0.197/0.303 |
| [[EgoControl]]* | 0.099/0.087 | 0.352/0.458 | 0.091/0.077 | 0.343/0.470 | 0.085/0.073 | 0.414/0.537 | 0.090/0.078 | 0.223/0.378 |
| Ego-WM | 0.069/0.058 | 0.313/0.396 | 0.068/0.055 | 0.433/0.527 | 0.050/0.042 | 0.468/0.561 | 0.063/0.053 | 0.460/0.459 |
| Naive EgoExo-WM | 0.065/0.053 | 0.347/0.447 | 0.064/0.049 | 0.439/0.561 | 0.048/0.040 | 0.382/0.525 | 0.062/0.052 | 0.368/0.443 |
| **EgoExo-WM** | **0.057/0.047** | **0.404/0.531** | **0.058/0.045** | **0.515/0.618** | **0.049/0.040** | **0.489/0.603** | **0.062/0.052** | **0.460/0.515** |

**说明**:
- 相比最强基线 [[EgoControl]]，HOMAGE/LEMMA 上 L2 误差降低约 42–50%。
- 朴素 EgoExo-WM（直接把未优化的 EgoX 数据加入训练）已经超过 Ego-WM；EgoX-Body 视角转换进一步带来 PCK@20 的明显提升（HOMAGE 0.347 → 0.404 → 0.531），证明"转换质量"比"数据量"更关键。

### Table 2: Model-Predictive Control 全身规划 (MPJPE 越低越好)

| 方法 | HOMAGE body/wrist | LEMMA body/wrist | Ego-Exo4D-Bike body/wrist | Ego-Exo4D-Cooking body/wrist |
|------|------|------|------|------|
| [[UniEgoMotion]] (无 WM 排序) | 0.404±0.035 / 0.471±0.038 | 0.444±0.016 / 0.493±0.016 | 0.292±0.044 / 0.367±0.039 | 0.533±0.011 / 0.580±0.011 |
| UniEgoMotion + Ego-WM | 0.383±0.010 / 0.447±0.014 | 0.414±0.012 / 0.455±0.010 | 0.267±0.007 / 0.341±0.005 | 0.519±0.015 / 0.568±0.023 |
| **UniEgoMotion + EgoExo-WM** | **0.362±0.012 / 0.421±0.012** | **0.396±0.008 / 0.438±0.006** | **0.245±0.016 / 0.320±0.013** | **0.498±0.018 / 0.549±0.016** |

**关键发现**: 转换 exo 数据带来的覆盖度提升让 MPC 选到更优全身动作；Bike 子集上 body MPJPE 从 0.292 降到 0.245（−16%）。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[Nymeria]] | ~200 h | 配对 SMPL-X 动作的真 ego 视频 | 训练（真 ego） |
| HowTo100M | ~5 h（采样） | 教学类 exo 视频 | EgoX-Body 转换后训练 |
| CrossTask | ~1 h | 跨任务 exo | EgoX-Body 转换后训练 |
| 100 Days of Hands | ~4 h | 手物交互 exo | EgoX-Body 转换后训练 |
| HOMAGE | — | 家居动作 ego | 测试 |
| LEMMA | — | 日常活动 ego | 测试 |
| [[Ego-Exo4D]] (Bike / Cooking) | — | 技能型同步 ego/exo | 测试 |

### 实现细节

- **视觉 backbone**: 冻结 [[DINOv3]] ViT-L/16，输入 $224 \times 224$ 输出 $14 \times 14$ patch
- **世界模型**: [[CDiT|CDiT-L/2]]，24 层 / 1024 隐维 / 16 头
- **手腕解码**: 6 层 Transformer，256 隐维，$16 \times 16$ 热图
- **优化器**: [[AdamW]]，lr $8 \times 10^{-5}$
- **训练**: batch size 512，100k iter，8× A40，bfloat16
- **EgoX-Body 推理**: $384 \times 384$ × 49 帧，4× GH200，约 3.25 min/段
- **规划**: $N=4$ 候选、$H=8$ 帧（~2 秒）

### 可视化结果

- Figure 3 显示 EgoX-Body 在手部姿态与对象接触点上比原始 EgoX 显著更准确，几乎能与 GT ego 视频在关键时刻对齐。
- Figure 5 显示规划任务中（家居走动、骑车、烹饪），EgoExo-WM 选出的动作轨迹更贴近真实人类完成目标的路径。

---

## 批判性思考

### 优点

1. **思路简洁有效**: 把"转换 exo 视频"作为数据增强而非新建模目标，3D 人体动作天然桥接两个视角。
2. **结构化先验**: 在 exo 与 ego 两端都注入显式骨架/手部先验，避免无约束的扩散漂移；消融显示 Naive EgoExo-WM → EgoExo-WM 的 PCK 提升来自 EgoX-Body 而非数据量。
3. **统一动作空间**: 69 维 SMPL-X 动作向量适用于真 ego 与伪 ego 数据，无需重新设计动作条件。
4. **应用扩展性**: 同一世界模型既能开环预测，又能闭环 MPC 规划，且与现成动作采样器 [[UniEgoMotion]] 无缝结合。

### 局限性

1. **时间尺度短**: EgoX-Body 一次只生成 49 帧（~2 秒），规划 horizon 仅 8 帧；长 horizon 下复合误差未解决。
2. **复杂交互失败**: 遮挡、精确接触、小物体操作容易让生成退化（论文明确提到出现"全黑/全白帧"的失败模式）。
3. **依赖多个估计器**: SAM 3D Body / HaMeR / ViTPose / ViPE 的误差会级联进伪 ego 数据，但论文没有系统量化级联误差。
4. **规模有限**: 只用了约 10 小时转换 exo 数据，远小于 200 h Nymeria，"互联网规模"愿景尚未真正释放。
5. **泛化到机器人本体**: 框架以人体为中心，迁移到非人形机器人（如桌面臂、四足）需要重设动作空间，论文未涉及。

### 潜在改进方向

1. **更长 horizon 转换**: 用滑窗或层次化生成把 EgoX-Body 扩展到 5–10 秒以上。
2. **联合训练 EgoX-Body 与世界模型**: 用世界模型损失反传到视角转换器，让生成的伪 ego 直接对世界模型有用。
3. **不确定度加权**: 为伪 ego 数据估计置信度，让世界模型在低置信样本上降低权重。
4. **大规模扩展**: 把全量 HowTo100M / Ego4D-Exo 跑通转换，验证 scaling law。

### 可复现性评估

- [ ] 代码开源（暂未发布）
- [ ] 预训练模型（暂未发布）
- [x] 训练细节完整（附录给了主要超参与硬件）
- [x] 数据集可获取（依赖的 Nymeria/HowTo100M/HOMAGE/LEMMA/Ego-Exo4D 均公开）

---

## 关联笔记

### 基于
- [[EgoX]]: EgoX-Body 的直接前身，本文加上骨架与手部先验。
- [[DINOv3]]: 提供冻结视觉潜空间。
- [[SAM 3D Body]] · [[HaMeR]] · [[VIPE]]: 人体/手部/场景几何估计器。
- [[UniEgoMotion]]: 提供物理可行的人体动作候选采样器。

### 对比
- [[PEVA]]: 用低维动作向量条件，无几何结构。
- [[EgoControl]]: 用相机姿态条件，无法描述全身行为。
- [[DINO-WM]] · [[JEPA-WM]]: 同样在 DINO 潜空间内做世界模型，但聚焦于桌面/物理仿真而非 ego。

### 方法相关
- [[Action-Conditioned World Model]]: 核心类别。
- [[Egocentric World Model]]: 直接所属的子方向。
- [[Latent MPC]]: 规划框架。
- [[exo-to-ego 视频生成]]: 数据生成关键技术。
- [[SMPL-X]]: 动作空间表征。

### 硬件/数据相关
- [[Nymeria]]: 主真 ego 训练源。
- [[Ego-Exo4D]]: 测试集之一。
- [[HOMAGE]] · [[LEMMA]]: ego 评测数据集。

---

## 速查卡片

> [!summary] EgoExo-WM
> - **核心**: 用 3D 人体动作把 exo 视频转换成伪 ego 数据训练[[Egocentric World Model|世界模型]]。
> - **方法**: EgoX-Body（SAM 3D Body 骨架 + HaMeR 手部先验）+ DINOv3 latent + CDiT-L/2 + 手腕一致性损失。
> - **结果**: HOMAGE/LEMMA L2 较 EgoControl 降 42–50%；MPC 全身规划 MPJPE 在 4 个数据集上全部最低。
> - **代码**: 暂未公开（项目页 https://vision.cs.utexas.edu/projects/EgoExo-WM/）。

---

*笔记创建时间: 2026-05-27*
