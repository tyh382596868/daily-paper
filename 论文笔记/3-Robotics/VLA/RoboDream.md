---
title: "RoboDream: Compositional World Models for Scalable Robot Data Synthesis"
method_name: "RoboDream"
authors: [Junjie Ye, Rong Xue, Basile Van Hoorick, Runhao Li, Harshitha Rajaprakash, Pavel Tokmakov, Muhammad Zubair Irshad, Vitor Guizilini, Yue Wang]
year: 2026
venue: arXiv
tags: [world-model, video-diffusion, robot-data-synthesis, compositional, droid, embodiment-anchoring]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.02577v1
created: 2026-06-02
---

# 论文笔记：RoboDream: Compositional World Models for Scalable Robot Data Synthesis

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | USC Physical Superintelligence (PSI) Lab; Toyota Research Institute |
| 日期 | June 2026 |
| 项目主页 | https://junjieye.com/RoboDream/ |
| 对比基线 | [[AdaWorld]], [[DreamZero]], [[RoboEnvision]], [[Cosmos-Policy]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.02577) / [Code (Coming Soon)](https://github.com/Jay-Ye/RoboDream) |

---

## 一句话总结

> RoboDream 把机器人运动学渲染 + 物体先验 + 场景先验解耦地灌进 [[视频扩散模型]]，零样本合成新物体、新场景、新视角的机器人演示，让一份 [[DROID]] 轨迹被「重生」为成百上千份训练数据。

---

## 核心贡献

1. **组合式世界模型 (Compositional World Model)**: 用三路独立条件 — 机器人渲染视频 $v_{rob}$、场景先验 $I_s$、物体先验 $I_o$ — 替代传统单图/语言条件，实现 [[Controllable 世界模型|可控合成]]，并避免对单一任务过拟合。
2. **Embodiment-Anchored 生成**: 在隐空间直接拼接已渲染的机器人轨迹视频，把 [[embodiment hallucination|具身幻觉]] 问题压到零 —— 生成的关节姿态严格遵守输入运动学，避免 [[DreamZero]] 一类方法机器人形态/动力学漂移。
3. **Retrieval & Rebirth**: 从 [[DROID]] 中按语义相似度检索旧轨迹，在 [[IsaacLab]] 重渲染为新视角下的机器人 only 视频，再叠加新物体/场景先验「重生」为新演示。
4. **Prop-Free Teleoperation**: 操作员不需要桌上有真实物体即可远程操作机器人，配合后续物体先验合成出有物体交互的演示，将单 episode 采集时间从 2 小时压到 55 分钟（2.2× 提速）。

---

## 问题背景

### 要解决的问题

[[VLA]] / [[Diffusion Policy]] 类机器人策略对训练数据规模高度饥饿，但真实机器人遥操作数据采集慢、贵、且场景/物体多样性差。如何**在不增加真实采集量的前提下，把分布扩展到新物体、新场景、新视角**是 scalable robot learning 的核心瓶颈。

### 现有方法的局限

- **In-place 视觉增广** ([[ROSIE]], [[RoboEnvision]]): 只能把已有帧里的物体/背景换皮，无法生成 *新的物理配置*。
- **DreamGen / [[DreamZero]] 式视频世界模型**: 端到端文本驱动视频生成，常出现 [[embodiment hallucination|具身幻觉]] — 机器人变形、关节漂移、抓取相位错乱，直接挂掉策略。
- **MimicGen / Real2Render2Real**: 依赖 3D 数字孪生 / 显式 [[sim-to-real]] 资产，难以扩展到任意视觉环境。
- **[[AnchorDream]]** (论文将其作为最接近的同期工作): 也用渲染机器人 anchoring，但需要任务特定微调，每换一个场景就要重新训练。

### 本文的动机

如果把**「机器人怎么动」**（kinematics）和**「环境长什么样」**（visual context）这两件事在条件层面彻底解耦，那么：

- 运动学 → 由 [[IsaacLab]] 物理仿真给出（精确、可控）
- 场景 → 由真实数据 inpaint 出干净背景
- 物体 → 由 [[Grounded-SAM]] 抠出，随机重组

视频扩散模型只负责把三者**渲染**成一致的照片级视频，这样既保住了 embodiment 真实性，又获得了组合泛化能力。

---

## 方法详解

### 模型架构

RoboDream 是一个**多模态 [[扩散变换器|视频 DiT]]**，从 [[Cosmos-Policy|Cosmos-Predict2]] 2B 基座微调而来。

- **输入**:
  - 语言指令 $\ell$
  - 机器人运动学渲染视频 $v_{rob} \in \mathbb{R}^{T \times H \times W \times 3}$（[[IsaacLab]] 渲染的纯机器人 + 黑背景）
  - 场景先验 $I_s \in \mathbb{R}^{H \times W \times 3}$（清场后的静态背景）
  - 物体先验 $I_o \in \mathbb{R}^{H \times W \times 3}$（任务相关物体拼贴图）
- **Backbone**: 多模态 [[Video Diffusion Model|视频扩散变换器]]，使用 [[Flow Matching|流匹配]] 训练目标（继承自 Cosmos-Predict2）
- **核心模块**: [[Embodiment Anchoring|具身锚定]] + [[Compositional Conditioning|组合式条件注入]]
- **输出**: 照片级演示视频 $o_{1:T}$
- **总参数**: 2B

#### 三路条件如何注入

- **机器人 + 场景**: 通过 [[VAE|3D Causal VAE]] $\mathcal{E}(\cdot)$ 编码后，在 **通道维拼接** 到带噪 latent $z_t$ 上 → 提供逐像素对齐的运动学/背景锚点；
- **物体先验**: tokenize 后通过 **扩展自注意力 (extended self-attention)** 注入 → 让模型在每个 spatial position 都能"看到"物体外观，但不强制其出现在固定位置（防止过拟合）；
- **语言指令**: 走原 Cosmos 的 cross-attention 通道。

### 核心模块

#### 模块1: 先验提取流水线 (Prior Extraction Pipeline)

**设计动机**: 把真实数据自动转成"清场场景 + 物体贴图"两份解耦先验，作为后续合成的素材库。

**具体实现**:

1. **物体识别**: 用 GPT-5-nano 看首帧，输出与任务相关物体名称列表；
2. **物体先验构造**: [[Grounded-SAM]] 分割 → 裁剪 → **随机旋转 + 缩放** → 拼到空白画布上（破除位置偏置）；
3. **场景先验构造**: [[OmniPaint|OmniPaint 扩散式 inpainting]] 抹除所有物体 → 得到干净的桌面/背景 $I_s$。

#### 模块2: 运动学锚定生成 (Embodiment-Anchored Generation)

**设计动机**: 让视频扩散模型只学"渲染"，不需要学"机器人怎么动"。

**具体实现**:

- 用 [[IsaacLab]] 仿真器在新视角 / 新位姿下渲染机器人骨架视频 $v_{rob}$（无物体、无背景）；
- $v_{rob}$ 编码后逐 patch 与 $z_t$ 拼接 — 像素级监督模型"机器人应该在哪个位置"；
- 因此抓取相位、夹爪开合、关节角度全部由 $v_{rob}$ 主导，模型只需把环境画上去。

这正是 RoboDream 相对 [[DreamZero]] / 通用 [[Video Diffusion Model|视频扩散模型]] 的关键差异：embodiment 不是从语言或文本"幻想"出来的，而是被显式渲染、显式锚定。

#### 模块3: Retrieval & Rebirth（轨迹重生）

**设计动机**: 把存量 [[DROID]] 数据当作"轨迹银行"，用一次采集生成多次数据。

**具体实现**:

1. **检索**: 用 T5 编码任务描述，在 [[DROID]] ~40k episodes 中语义相似度 top-k；
2. **重渲染**: 把检索到的关节序列在 [[IsaacLab]] 里以新视角重新渲染为 $v_{rob}$；
3. **重生**: 配上目标场景的 $I_s, I_o$，由 RoboDream 合成新演示；
4. 每条原轨迹可衍生 N 条新轨迹，组合扩展。

#### 模块4: Prop-Free Teleoperation（无道具遥操作）

操作员对着空桌面遥操作 [[Franka 研究臂|Franka Panda]] 完成"虚拟抓取"动作 → 得到 $v_{rob}$ → 离线送进 RoboDream + 物体/场景先验 → 合成"似乎在抓东西"的演示，省去布置道具与 reset 时间。

---

## 关键公式

### 公式1: [[轨迹|Trajectory]] 与观测序列定义

$$
\tau = \{(s_t, a_t)\}_{t=1}^{T}, \quad o_{1:T} = \{o_t\}_{t=1}^{T}
$$

**含义**: 一段机器人演示由轨迹 $\tau$（状态-动作序列）和观测视频 $o_{1:T}$ 组成。RoboDream 的目标是**给定 $\tau$ 合成新的 $o_{1:T}$**，从而把一份 $\tau$ 重复利用。

**符号说明**:
- $s_t$: 第 $t$ 步机器人本体状态（关节角度、末端位姿）
- $a_t$: 第 $t$ 步动作指令
- $o_t$: 第 $t$ 步第三视角 RGB 观测
- $T$: 时序长度（DROID 中典型为数百帧）

### 公式2: [[Compositional Conditioning|组合式条件分布]]

$$
p_\theta(o_{1:T} \mid v_{rob},\ I_s,\ I_o,\ \ell,\ \tau)
$$

**含义**: 这是 RoboDream 的核心建模目标 — 不再是 $p(o \mid \ell)$ 的纯文本驱动生成，而是把环境 (scene)、物体 (object)、运动学 (robot kinematics) 三路条件**显式分离**，使每一路都可独立替换以实现组合泛化。

**符号说明**:
- $v_{rob}$: [[IsaacLab]] 渲染的机器人 only 视频，提供 embodiment 锚点
- $I_s$: 经 [[OmniPaint]] 清场的背景图
- $I_o$: [[Grounded-SAM]] 抠出并随机拼贴的物体图
- $\ell$: 任务语言描述
- $\tau$: 原始机器人轨迹（用于驱动 $v_{rob}$ 渲染）

### 公式3: [[Embodiment Anchoring|具身锚定]] 的 latent 拼接

$$
x_{in} = \mathrm{Concat}\Big(z_t,\ \mathcal{E}(v_{rob}),\ \mathcal{E}(I_s^{\otimes T})\Big)
$$

**含义**: 在每个去噪步 $t$，把带噪 latent $z_t$ 与机器人渲染、复制 $T$ 帧的场景先验**沿通道维拼接**，作为 DiT 的输入。这一步使模型逐 patch、逐帧地"看到"机器人和背景应该在哪里，是 embodiment 不漂移的关键。

**符号说明**:
- $z_t$: 第 $t$ 步噪声 latent
- $\mathcal{E}(\cdot)$: 共享的 3D Causal VAE 编码器（来自 [[Cosmos-Policy|Cosmos]]）
- $I_s^{\otimes T}$: 场景先验沿时间维复制 $T$ 次以匹配视频长度

### 公式4: [[扩展自注意力|Extended Self-Attention]] 注入物体先验

$$
\mathrm{Attention}\big(Q_{vid},\ [K_{vid};\ K_{obj}],\ [V_{vid};\ V_{obj}]\big)
$$

**含义**: 视频 token 的 query 同时与视频自身和物体先验 token 做 attention — 物体外观信息通过 cross-attention 灌入视频生成路径，但**不绑定**任何空间位置，从而让模型自己决定物体出现在哪里、被怎么抓。

**符号说明**:
- $Q_{vid}, K_{vid}, V_{vid}$: 视频 latent 投影出的 Q/K/V
- $K_{obj}, V_{obj}$: 物体先验图 tokenize 后投影出的 K/V
- $[\,;\,]$: 沿 token 维度拼接

### 公式5: [[Flow Matching|流匹配训练目标]]（继承自 Cosmos-Predict2）

$$
\mathcal{L}_{FM} = \mathbb{E}_{t,\ z_0,\ z_1}\left[\,\big\|\ v_\theta(z_t,\ t,\ c) - (z_1 - z_0)\ \big\|_2^2\,\right]
$$

**含义**: 论文未显式列出训练损失，但既然从 [[Cosmos-Policy|Cosmos-Predict2]] 微调而来，沿用其 [[Flow Matching|流匹配]] 目标 — 学习从噪声 $z_0$ 到真实 latent $z_1$ 的速度场 $v_\theta$。条件向量 $c = (v_{rob}, I_s, I_o, \ell)$ 即上述三路加语言。

**符号说明**:
- $z_0 \sim \mathcal{N}(0, I)$: 初始噪声
- $z_1 = \mathcal{E}(o_{1:T})$: 真实视频的 latent
- $z_t = (1-t) z_0 + t z_1$: 线性插值
- $v_\theta$: DiT 预测的速度场
- $c$: 组合条件

---

## 关键图表

### Figure 1: Teaser — 组合式数据合成示意

![Figure 1](https://arxiv.org/html/2606.02577v1/x1.png)

**说明**: RoboDream 以一个组合式世界模型生成 *photorealistic demonstrations with novel objects, in novel scenes, and from novel viewpoints*。Teaser 中展示了同一条 [[DROID]] 轨迹被重新合成到不同物体（marker / cube）、不同桌面（实验室 / 木桌 / 厨房）、不同视角（正面 / 斜上 / 侧面）下的演示帧。

### Figure 2: RoboDream 架构

![Figure 2](https://arxiv.org/html/2606.02577v1/x2.png)

**说明**: 主体是一个 [[Video Diffusion Model|视频扩散变换器]]。机器人渲染视频 $v_{rob}$ 与场景先验 $I_s$ **在隐空间通道维拼接**到带噪 latent；物体先验 $I_o$ 通过 **extended self-attention** 注入。三路条件的分离正是组合泛化的关键。

### Figure 3: 先验提取流水线

![Figure 3](https://arxiv.org/html/2606.02577v1/x3.png)

**说明**: 从首帧开始，[[Grounded-SAM]] 分割任务相关物体 → 裁剪+随机旋转后拼成物体先验图；同一帧用 [[OmniPaint]] 把物体抹除得到清场背景。这套流水线把任意真实数据自动转成"可组合素材"。

### Figure 4: 实验设置 (DROID Franka + 四任务)

![Figure 4](https://arxiv.org/html/2606.02577v1/x4.png)

**说明**: 真实评估平台为 [[DROID]] 标准 [[Franka 研究臂|Franka Panda]]，四个评估任务：*Put Marker into Bowl*、*Remove Marker from Bowl*、*Put Cube into Cup*、*Wipe Table with Towel*。每个策略跑 20 次 rollout 统计成功率。

### Figure 5: Zero-Shot 演示重生 + 组合泛化

![Figure 5](https://arxiv.org/html/2606.02577v1/x5.png)

**说明**: 左为新物体/场景先验，右为从 [[DROID]] 检索到的源轨迹，中间为 RoboDream 合成的"重生"演示。同一张图还覆盖四个组合泛化维度：*Novel Instances*（不同 marker）/ *Novel Scenes*（新背景）/ *Novel Tasks*（marker→bowl vs cube→cup）/ *Novel Viewpoints*（不同相机位姿）。即便目标物体/桌面在源数据里从未出现，机器人动作仍因 $v_{rob}$ 锚定而严格守序，展示了一份骨架轨迹经组合扩展可覆盖巨大的视觉分布。

### Table 1: 策略学习成功率 (%) — 真实 vs 合成 vs 混合

| Task | Real-50 | Orig-100 | Orig-Mix | Gen-100 | **Gen-Mix** |
|------|--------:|---------:|---------:|--------:|------------:|
| Put Cube into Cup | 35 | 0 | 55 | 20 | **65** |
| Put Marker into Bowl | 30 | 0 | 35 | 15 | **55** |
| Remove Marker from Bowl | 20 | 0 | 20 | 5 | **35** |
| Wipe Table with Towel | 60 | 0 | 70 | 20 | **95** |
| **Average** | 36.3 | 0 | 45.0 | 15.0 | **62.5** |

**说明**: Real-50 仅用 50 条真实演示，Gen-Mix 在其上混入 RoboDream 合成数据，平均成功率从 36.3 → **62.5**（+26 个绝对百分点）。注意 Gen-100（纯合成 100 条）单独训练成功率 15.0 < Real-50 36.3 — 说明合成数据仍需与真实数据混合才能发挥最大价值，单独无法替代。

### Table 2: Prop-Free 无道具采集对比

| Task | Real-50 | Real w/ Gen Obs | Prop-Free |
|------|--------:|----------------:|----------:|
| Put Cube into Cup | 35 | 25 | 30 |
| Put Marker into Bowl | 30 | 20 | 20 |
| Remove Marker from Bowl | 20 | 15 | 20 |
| Wipe Table with Towel | 60 | 60 | 60 |
| **Average** | 36.3 | 30.0 | 32.5 |

**说明**: Prop-Free 模式只让操作员空手挥动机械臂，再事后合成物体外观；成功率 32.5% ≈ Real-50 36.3%（差距 < 4%），但采集速度提升 **2.2×**（55 min vs 2 h / 50 episodes），是 [[scalable robot data collection|可扩展数据采集]] 的强证据。

### Table 3: 生成数据规模 Scaling

| Task | Real-50 | Mix-100 | Mix-200 | Mix-300 | Mix-400 |
|------|--------:|--------:|--------:|--------:|--------:|
| Put Cube into Cup | 35 | 65 | 75 | 80 | 75 |
| Put Marker into Bowl | 30 | 55 | 70 | 70 | 70 |
| Remove Marker from Bowl | 20 | 35 | 45 | 50 | 50 |
| Wipe Table with Towel | 60 | 95 | 100 | 95 | 100 |
| **Average** | 36.3 | 62.5 | 72.5 | 73.75 | 73.75 |

**关键发现**: 在 Mix-200 (50 真实 + 200 合成) 达到 72.5% 后开始饱和 — 合成数据不是"越多越好"，存在一个最优混合比；这也间接说明真实-合成 distribution gap 仍然存在。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[DROID]] | ~40k episodes（带相机标定的子集） | 跨实验室真实操作、Franka 平台 | 训练 RoboDream + Retrieval 池 |
| 自采 Real-50 (4 任务 × 50 episode) | 200 episodes | 受控真实评估数据 | 下游 [[Diffusion Policy]] 训练 |

### 实现细节

- **Backbone**: 从 [[Cosmos-Policy|Cosmos-Predict2]] 2B 视频 DiT 微调
- **训练目标**: [[Flow Matching|流匹配]]（继承 Cosmos）
- **硬件**: 2 nodes × 8 NVIDIA A100 GPU，训练 1 周
- **下游策略**: [[Diffusion Policy]]（标准框架）
- **机器人**: [[Franka 研究臂|Franka Panda]] + DROID 平台（第三视角静态相机 + 腕部相机）
- **评估**: 每策略 20 次 rollout

### 可视化结果

- 合成视频中机器人姿态与 $v_{rob}$ 像素级对齐，无 [[embodiment hallucination|关节漂移]]；
- 物体可在不出现于 $v_{rob}$ 的位置自然出现（如夹爪经过时被"抓住"），证明 extended self-attention 灵活学到了物体-末端器交互；
- 视角切换零样本（俯视、侧视、新基座位姿）均能生成自洽帧。

---

## 实验结果

四个真实 [[Franka 研究臂|Franka Panda]] 任务、每策略 20 次 rollout 的关键数字（详见 Table 1 / 2 / 3）：

- **混合合成数据极大提升下游策略**: Real-50 baseline 36.3% → +RoboDream Mix-100 **62.5%**（+26 abs%，Table 1）→ Mix-200 **72.5%**（+36 abs%，Table 3）。
- **合成数据不能独立替代真实数据**: Gen-100 单独训练仅 15.0%，远低于 Real-50 36.3% — 真实-合成仍存在 distribution gap。
- **Prop-Free 模式有效**: 无道具采集（32.5%）≈ 真实采集（36.3%），差距 < 4%，但**采集速度 2.2× 提升**（55 min vs 2 h / 50 episodes）。
- **Scaling 饱和点**: Mix-200 后性能饱和（72.5 → 73.75 → 73.75），说明合成数据存在最优混合比，无脑堆量收益递减。
- **任务依赖性强**: 最简单任务 *Wipe Table* 在 Mix-200 即达 100% 成功；最难任务 *Remove Marker from Bowl* 才从 20% → 50% — 长尾任务仍是挑战。

> 与 [[AdaWorld]] / [[DreamZero]] 等同期工作相比，RoboDream 没有给出直接对照实验，仅在 related work 中口头比较 — 这是论文一个可改进点。

---

## 批判性思考

### 优点

1. **架构干净**: 三路条件解耦的思路非常自然，每一路都对应一个明确的物理量（运动学/背景/物体外观）；和 [[Compositional World Models|组合式世界模型]] 思想完美契合。
2. **Embodiment 锚定是杀手锏**: 通过显式渲染机器人，绕开了 [[DreamZero]] / DreamGen 一整类工作的痛点（机器人形态/动力学失真）。这是真正能落到物理硬件的合成数据。
3. **Prop-Free 给出了实际部署价值**: 2.2× 提速 + 仅 < 4% 成功率损失，是 [[scalable robot data collection|可扩展采集]] 的少有的实战级方案。
4. **充分对照 Real-50**: 用 36.3% 真实基线对比，让合成数据的增益（+26 abs%）可信。

### 局限性

1. **合成数据仍不能独立训练**: Table 1 中 Gen-100 单独训练只有 15% < Real-50 36%，说明 RoboDream 输出与真实分布有 sim2real / 视觉-动力学 gap。这削弱了"零真实数据"的可能性。
2. **依赖 [[IsaacLab]] 物理仿真和 DROID 相机标定**: Retrieval & Rebirth 必须能在 IsaacLab 渲染源轨迹 — 限制了对非标定 / 非 Franka 数据的扩展。
3. **任务面较窄**: 仅 4 个桌面操作 + 短时序，未验证长时序、双手、灵巧手或接触密集型任务（如插孔、折叠）。
4. **对 GPT-5-nano 的依赖**: 物体识别走闭源 VLM，复现性受限。
5. **物体先验是 2D 拼贴**: 没有处理物体 3D 形变、半遮挡、反光 — 视频中可能有微妙不一致。

### 潜在改进方向

1. **3D 物体先验**: 用 [[3D Causal VAE]] / NeRF / 3DGS 表示物体，让 attention 注入带几何感的特征；
2. **闭环执行下的世界模型**: 把 RoboDream 改造成 [[Action-Conditioned World Model|动作条件世界模型]]，做 [[WMPO|世界模型策略优化]]；
3. **跨 embodiment**: 用同样思路渲染 [[ALOHA|ALOHA]]、[[UR5]]、[[Galaxea]] 等不同形态 — 把 $v_{rob}$ 当成 [[AdaWorld|latent embodiment]] 的通用接口；
4. **与 [[AnchorDream]] 的端到端区分性实验**: 论文只口头比较，缺数值对照。

### 可复现性评估

- [ ] 代码开源（仓库标注 "Coming Soon"）
- [ ] 预训练模型
- [ ] 训练细节完整（数据量、硬件、训练步数 ✅；超参、学习率、batch size 未公开）
- [x] 数据集可获取（[[DROID]] 公开）

---

## 关联笔记

### 基于

- [[Cosmos-Policy]]: RoboDream 直接微调自 Cosmos-Predict2 2B 视频 DiT
- [[Flow Matching]]: 继承的训练目标
- [[Video Diffusion Model]]: 模型范式
- [[DROID]]: 训练 + retrieval 数据源

### 对比

- [[AdaWorld]]: 同样追求 *组合泛化*，但 AdaWorld 走 latent action 路线、把动作抽象成跨 embodiment 接口；RoboDream 反其道而行 — **显式渲染**机器人外观以保物理一致。
- [[DreamZero]]: 端到端文本驱动机器人视频生成，没有 embodiment 锚定 → 容易出现 [[embodiment hallucination|具身幻觉]]
- [[AnchorDream]]: 最接近的同期工作，也用渲染锚定，但需任务特定微调，RoboDream 强调 *零样本*
- [[RoboEnvision]]: in-place 视觉增广，只换皮不换物理配置
- [[ROSIE]]: 同 in-place 增广路线

### 方法相关

- [[embodiment hallucination|具身幻觉]]: 本文核心要解决的现象
- [[Compositional World Models|组合式世界模型]]: 本文范式归类
- [[Grounded-SAM]]: 物体分割
- [[OmniPaint]]: 背景 inpainting
- [[Diffusion Policy]]: 下游策略
- [[IsaacLab]]: 运动学渲染
- [[扩展自注意力|Extended Self-Attention]]: 物体先验注入机制

### 硬件/数据相关

- [[Franka 研究臂]]: 评估硬件
- [[DROID]]: 主数据源

---

## 速查卡片

> [!summary] RoboDream
> - **核心**: 把"机器人怎么动"和"环境长什么样"在条件维度解耦，由视频扩散模型组合渲染成新演示
> - **方法**: 三路条件 (rendered robot video + scene prior + object prior) 喂进 Cosmos-Predict2 2B，flow matching 微调
> - **结果**: Real-50 + RoboDream-Mix → 36.3% → 62.5% / 72.5%（4 真实任务平均）；Prop-Free 采集 2.2× 加速、< 4% 性能损失
> - **数据**: [[DROID]] 40k episodes
> - **代码**: https://github.com/Jay-Ye/RoboDream (Coming Soon)
> - **项目页**: https://junjieye.com/RoboDream/

---

*笔记创建时间: 2026-06-02*
