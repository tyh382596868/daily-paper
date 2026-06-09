---
title: "GRAIL: Generating Humanoid Loco-Manipulation from 3D Assets and Video Priors"
method_name: "GRAIL"
authors: [Tianyi Xie, Haotian Zhang, Jinhyung Park, Zi Wang, Bowen Wen, Jiefeng Li, Xueting Li, Qingwei Ben, Haoyang Weng, Yufei Ye, David Minor, Tingwu Wang, Chenfanfu Jiang, Sanja Fidler, Jan Kautz, Linxi Fan, Yuke Zhu, Zhengyi Luo, Umar Iqbal, Ye Yuan]
year: 2026
venue: arXiv
tags: [humanoid-robot, loco-manipulation, data-generation, human-object-interaction, sim-to-real, video-foundation-model, motion-retargeting]
zotero_collection: 3-Robotics
image_source: online
arxiv_html: https://arxiv.org/html/2606.05160v1
created: 2026-06-09
---

# 论文笔记：GRAIL: Generating Humanoid Loco-Manipulation from 3D Assets and Video Priors

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | NVIDIA, UCLA |
| 日期 | June 2026 |
| 项目主页 | [research.nvidia.com/labs/dair/grail](https://research.nvidia.com/labs/dair/grail/) |
| 代码 | [github.com/NVlabs/GRAIL](https://github.com/NVlabs/GRAIL) |
| 数据集 | [HuggingFace: PhysicalAI-Robotics-Locomanipulation-GRAIL](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL) |
| 对比基线 | [[SONIC]]、[[DeepMimic]]、[[GMR]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.05160) / [Code](https://github.com/NVlabs/GRAIL) |

---

## 一句话总结

> GRAIL 是一个全数字化的人形机器人 Loco-Manipulation 数据生成流水线，通过组合 3D 资产和[[视频基础模型]]先验，无需物理遥操作即可合成超 22,000 条物理可行的 4D 人-物交互轨迹，并在真实 Unitree G1 上取得 84% 物体抓取和 90% 爬楼梯成功率。

---

## 核心贡献

1. **全数字化数据生成框架**: 利用[[视频基础模型]]（Kling AI）作为交互先验，在完全虚拟环境中生成超 20,000 条人形机器人 [[Loco-Manipulation]] 序列，涵盖抓取、全身操作、坐下、地形穿越四类任务，无需物理重建或遥操作。
2. **交互感知 4D HOI 重建栈**: 利用已知的 3D 几何、度量尺度、相机参数、环境深度及机器人比例角色，从合成视频中重建出 [[SMPL-X]] 人体姿态 + 物体 6-DoF 的连贯度量 4D [[HOI]] 轨迹。
3. **任务通用追踪策略 + Sim-to-Real 部署**: 在 [[IsaacLab]] 中训练两种互补的 [[SONIC]] 追踪策略（物体感知的 Latent Adaptor + 场景感知的地形追踪器），并蒸馏为以头部相机 RGB 为输入的自中心视觉策略，成功部署到真实 Unitree G1。

---

## 问题背景

### 要解决的问题

扩展人形机器人 [[Loco-Manipulation]] 需要覆盖多样物体、全身运动和场景几何的机器人兼容演示数据。然而，遥操作和动作捕捉难以扩展，因为每次数据采集都依赖物理搭建、仪器化演员和机器人操作。

### 现有方法的局限

- **遥操作**：需要物理环境重建和人工操控，采集代价高昂，难以覆盖长尾场景。
- **动作捕捉**：需要仪器化演员，且捕捉到的人体动作不一定对机器人运动学可行。
- **纯仿真数据**：缺乏自然人类与物体交互的先验，生成的动作往往不自然。

### 本文的动机

通过先在完全指定的 3D 场景中放置与目标机器人比例一致的 SMPL-X 角色，利用[[视频基础模型]]生成自然的人-物交互视频，再从视频中重建物理可行的 4D HOI 轨迹。这样：
- 视频生成前已知 3D 几何、度量尺度、相机参数，大幅降低重建歧义；
- 全程在数字空间完成，直到部署前无需接触真实机器人；
- 可以无限扩展到任意物体和场景。

---

## 方法详解

### 总体 Pipeline 概览

GRAIL 的端到端流水线包含以下五个阶段：

- **阶段 1 — 3D 场景组装与视频生成**: 获取 3D 资产，在 Blender 中将预适配目标机器人比例的 [[SMPL-X]] 角色与物体置于场景，通过 OpenAI VLM 生成交互提示，用 **Kling 2.5 Turbo Pro** 生成含角色-物体交互的静态相机合成视频。
- **阶段 2 — 优化预处理**: 用 **[[MoGe-2]]** 做度量深度估计，用 **[[SAM2]]** 分割人体和物体，生成每帧点云作为优化目标（约 2 分钟/序列）。
- **阶段 3 — 4D HOI 重建**: 从合成视频重建 [[SMPL-X]] 人体姿态和物体 6-DoF 轨迹（约 8 分钟/序列）。
- **阶段 4 — 动作重定向**: 将 [[SMPL-X]] 人体姿态重定向到 Unitree G1 骨骼。
- **阶段 5 — 任务通用追踪与 Sim-to-Real**: 在 [[IsaacLab]] 中用 [[SONIC]] 追踪策略训练并蒸馏为自中心视觉策略部署到真机。

整条 pipeline 处理一条 5 秒 24fps 视频（121 帧）约需 **14 分钟**（单 NVIDIA A100）。

---

### 核心模块

#### 模块 1: 4D HOI 重建栈（Interaction-Aware HOI Reconstruction）

**设计动机**: 利用生成时已知的 3D 场景上下文（几何、度量尺度、相机参数），大幅减少从视频重建 4D 运动的歧义。

**具体实现**:

- **人体姿态估计（GEM-SMPL）**: 集成 HMR2 + ViTPose + VIMO + HMR4D，从视频帧中恢复每帧 [[SMPL-X]] 体态参数（身体关节 + 手部 DOF）；手部 DOF 额外用 **WiLoR** 估计 MANO 参数。
- **物体 6-DoF 估计（[[FoundationPose]]）**: 以 [[SAM2]] mask 和 [[MoGe-2]] 深度先验为条件，估计物体在每帧的旋转和平移。
- **多阶段 HOI 联合优化器**: 用关键点对齐损失、深度对齐损失、接触对齐损失和正则化，以 MoGe-2 深度和 SAM2 mask 的一致性为锚，联合优化人体和物体轨迹跨所有帧，产出度量一致的 4D HOI 序列。

**输出数据包含**: 源合成视频、4D HOI 重建（SMPL-X + 物体姿态）、重定向后 G1 机器人轨迹、RL 后物体轨迹、物体 USD 资产。

#### 模块 2: 动作重定向（Motion Retargeting via GMR）

**设计动机**: 人类 [[SMPL-X]] 骨骼与 Unitree G1 机器人骨骼形状、自由度不同，需要将人体动作转换为对机器人关节运动学可行的参考运动。

**具体实现**:
- 使用 **GMR（General Motion Retargeting，ICRA 2026）** 的 IK + 时序平滑引擎，将 [[SMPL-X]] 姿态序列重定向到 G1 骨骼。
- 同时处理手部 DOF 和每条 motion 的 USD 资产组装。

#### 模块 3: 任务通用追踪策略（SONIC-Based Task-General Trackers）

**设计动机**: 重定向后的参考运动需要在物理仿真中实现，训练物理可行的追踪策略是 Sim-to-Real 的关键。

**两种互补的追踪策略**:

- **物体感知 Latent Adaptor（Object-Aware Latent Adaptor）**:
  - 针对物体操作轨迹（pick-up、whole-body manipulation）训练。
  - 冻结预训练 [[SONIC]] 全身控制器，通过调制其 latent tokens 并输出手部动作来扩展操作能力。
  - 输入：本体感知 + 物体状态。
- **场景感知追踪器（Scene-Aware Tracker）**:
  - 针对地形穿越（楼梯、坡道）和坐下轨迹训练。
  - 联合微调 [[SONIC]] 控制器 + 高度图编码器，实现地形条件化全身控制。
  - 输入：本体感知 + 局部高度图。

#### 模块 4: 自中心视觉策略（Egocentric Visual Policy）

**设计动机**: 将追踪策略蒸馏为能在真机部署的 RGB 输入策略，实现 [[Sim-to-Real]] 迁移。

**具体实现**:
- 将物体感知 Latent Adaptor 和场景感知追踪器分别蒸馏为独立的自中心视觉策略（物体抓取 / 爬楼梯）。
- 输入：头部相机 RGB。
- 输出：[[SONIC]] 控制器的 latent tokens。
- 训练时使用**视觉域随机化**（Visual Domain Randomization）+ 相机对齐（Camera Alignment）以促进 Sim-to-Real 迁移。
- 真机推理频率：**10 Hz**，部署硬件：NVIDIA RTX 5090。
- 头部相机：**Luxonis OAK-D W**。

---

## 关键公式

### 公式 1: [[HOI|HOI 联合优化总损失]]

$$
\mathcal{L}_{\text{HOI}} = \lambda_{\text{kp}} \mathcal{L}_{\text{keypoint}} + \lambda_{\text{proj}} \mathcal{L}_{\text{projection}} + \lambda_{\text{depth}} \mathcal{L}_{\text{depth}} + \lambda_{\text{contact}} \mathcal{L}_{\text{contact}} + \lambda_{\text{reg}} \mathcal{L}_{\text{reg}}
$$

**含义**: 多阶段 HOI 优化器联合优化人体和物体轨迹，锚定到已知 3D 场景上下文，产出物理一致的度量 4D 序列。

**符号说明**:
- $\mathcal{L}_{\text{keypoint}}$: 关键点对齐损失，约束人体关节投影与检测结果对齐
- $\mathcal{L}_{\text{projection}}$: 投影一致性损失，确保 3D 轨迹与 2D 图像观测一致
- $\mathcal{L}_{\text{depth}}$: 深度对齐损失，以 MoGe-2 度量深度为锚点约束绝对尺度
- $\mathcal{L}_{\text{contact}}$: 接触对齐损失，用 VLM 预测接触时刻和位置，拉近手部与物体顶点的 3D 距离
- $\mathcal{L}_{\text{reg}}$: 正则化损失，防止过拟合和骨骼变形
- $\lambda_*$: 各项权重系数

### 公式 2: [[SMPL-X|SMPL-X 人体参数化]]

$$
\mathbf{M}(\boldsymbol{\theta}, \boldsymbol{\beta}, \boldsymbol{\psi}) = W(T(\boldsymbol{\beta}, \boldsymbol{\theta}, \boldsymbol{\psi}), J(\boldsymbol{\beta}), \boldsymbol{\theta}, \mathcal{W})
$$

**含义**: SMPL-X 将身体形状、姿态、面部/手部表情参数化为一个可微的网格模型，用于 4D HOI 重建中的人体表示。

**符号说明**:
- $\boldsymbol{\theta}$: 姿态参数（关节旋转）
- $\boldsymbol{\beta}$: 形状参数（体型）
- $\boldsymbol{\psi}$: 面部和手部表情参数
- $T(\cdot)$: T-pose 模板变形函数
- $J(\boldsymbol{\beta})$: 关节位置函数
- $W(\cdot)$: Linear Blend Skinning 函数
- $\mathcal{W}$: 蒙皮权重矩阵

### 公式 3: [[Sim-to-Real|域随机化视觉策略损失]]

$$
\mathcal{L}_{\text{vis}} = \mathbb{E}_{o \sim \mathcal{D}_{\text{sim}}} \left\| \pi_{\text{vis}}(o_{\text{rgb}}) - \pi_{\text{track}}(o_{\text{prop}}) \right\|^2
$$

**含义**: 视觉策略通过行为克隆蒸馏追踪策略，输入为域随机化后的渲染 RGB 图像，输出为 SONIC latent tokens，以 L2 损失监督。

**符号说明**:
- $\pi_{\text{vis}}$: 自中心视觉策略（待训练）
- $\pi_{\text{track}}$: 教师追踪策略（SONIC Latent Adaptor 或 Scene-Aware Tracker）
- $o_{\text{rgb}}$: 域随机化后的头部相机 RGB 输入
- $o_{\text{prop}}$: 本体感知状态输入
- $\mathcal{D}_{\text{sim}}$: 仿真中的观测分布

---

## 关键图表

### Figure 1: GRAIL 系统概览（Pipeline Overview）

> 图片来源：[arxiv.org/abs/2606.05160](https://arxiv.org/abs/2606.05160)（arXiv 服务器暂时限制外部访问，可直接查看论文原文）

**说明**: GRAIL 的端到端流水线。从左至右：3D 资产组装 + Blender 场景 → Kling AI 视频生成 → GEM-SMPL/[[FoundationPose]] 4D HOI 重建 → [[GMR]] 重定向到 G1 → [[SONIC]] 追踪策略训练 → 自中心视觉策略 [[Sim-to-Real]] 部署于真实 Unitree G1。整个过程在部署前保持全数字化。

### Figure 2: 4D HOI 重建栈细节

**说明**: 展示从合成视频重建 4D [[HOI]] 的细节。

- **输入**: Blender 渲染帧 + Kling 生成的合成视频
- **人体估计**: GEM-SMPL（HMR2 + ViTPose + VIMO + HMR4D）恢复 [[SMPL-X]] 体态；WiLoR 估计 MANO 手部参数
- **物体估计**: [[FoundationPose]] + [[SAM2]] mask + [[MoGe-2]] 深度先验，估计物体 6-DoF
- **联合优化**: 多阶段 HOI 优化器，以关键点/深度/接触/正则化损失联合优化全序列
- **输出**: 度量一致的 4D HOI 轨迹（SMPL-X + 物体 6-DoF）

### Figure 3: 任务通用追踪策略架构

**说明**: 两种互补追踪策略的架构对比。

- **物体感知 Latent Adaptor（操作任务）**: 冻结预训练 [[SONIC]] 全身控制器 → 添加 latent 调制模块 → 输出手部动作；输入：本体感知 + 物体状态
- **场景感知追踪器（地形任务）**: 联合微调 [[SONIC]] + 高度图编码器 → 地形条件化全身控制；输入：本体感知 + 局部高度图

### Figure 4: 真机部署结果（Real-World Deployment）

**说明**: Unitree G1 在真实环境中的部署结果。

- **多样物体抓取**: 84% 成功率（训练集物体），80% 成功率（新颖未见物体）
- **爬楼梯**: 90% 成功率
- 部署配置：Luxonis OAK-D W 头部相机，NVIDIA RTX 5090，10 Hz 推理

### Figure 5: 数据集多样性可视化

**说明**: GRAIL 生成的 22,000+ 序列的多样性展示，覆盖四类主要任务：

- **Pick-Up**: 桌面/地面物体抓取
- **Whole-Body Manipulation**: 全身操作
- **Sitting**: 坐下
- **Terrain Traversal**: 台阶/坡道/楼梯穿越

### Table 1: 真机实验成功率

| 任务 | 方法 | 成功率 |
|------|------|--------|
| 物体抓取（训练集物体） | GRAIL (Ours) | **84%** |
| 物体抓取（未见新颖物体） | GRAIL (Ours) | **80%** |
| 爬楼梯 | GRAIL (Ours) | **90%** |

**说明**: 所有策略仅用 GRAIL 生成的合成数据训练，无需任何真实机器人演示数据，即可在真机实现高成功率。

### Table 2: 数据集统计

| 类别 | 任务 | 序列数 |
|------|------|--------|
| Pick-Up | 桌面/地面物体抓取 | ~8,000+ |
| Whole-Body Manipulation | 全身操作 | ~4,000+ |
| Sitting | 坐下 | ~4,000+ |
| Terrain Traversal | 台阶/坡道/楼梯 | ~4,000+ |
| **总计** | | **>20,000** |

**说明**: GRAIL 生成的序列在四大任务类别上均匀覆盖，每条序列包含 5 秒 24fps 视频（121 帧）。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| GRAIL 合成数据集 | >20,000 序列，每条 5s/24fps | 全数字化生成，覆盖 4 大任务类 | 追踪策略 + 视觉策略训练 |
| 真实部署测试集 | 多样物体 + 新颖物体 | 真实 Unitree G1 环境 | Sim-to-Real 评估 |

### 实现细节

- **视频生成**: Kling 2.5 Turbo Pro（静态相机模式）
- **人体估计**: GEM-SMPL（HMR2 + ViTPose + VIMO + HMR4D）
- **手部估计**: WiLoR（MANO 参数）
- **物体 6-DoF**: [[FoundationPose]] + [[SAM2]] mask + [[MoGe-2]] 深度
- **深度估计**: MoGe-2
- **动作重定向**: GMR（General Motion Retargeting，ICRA 2026）+ 时序平滑
- **仿真平台**: [[IsaacLab]]，Unitree G1 机器人模型
- **基础控制器**: [[SONIC]] 全身控制器（预训练）
- **训练硬件**: NVIDIA A100（生成），NVIDIA RTX 5090（真机推理）
- **推理频率**: 10 Hz
- **头部相机**: Luxonis OAK-D W
- **每序列生成时间**: ~14 分钟（A100）

### 可视化结果

- 生成的合成视频展示出 SMPL-X 角色自然地与物体交互，视觉质量高。
- 重建的 4D HOI 轨迹在度量尺度和物理一致性上明显优于不使用 3D 场景上下文的基线。
- 真机部署视频显示 G1 能够流畅地抓取多种形状和大小的物体，以及在真实楼梯上稳定行走。

---

## 批判性思考

### 优点

1. **全数字化扩展**: 彻底摆脱物理数据采集的瓶颈，理论上可以无限扩展到任意物体和场景，极大降低了 Loco-Manipulation 数据的获取成本。
2. **3D 先验加速重建**: 生成时已知几何/尺度/相机参数，从根本上减少了 4D HOI 重建的歧义，产出的轨迹物理一致性好。
3. **模块化可扩展 Pipeline**: 五个阶段各自独立，单模块改进（如更好的视频生成模型）可直接提升整体质量，工程上有良好的扩展性。
4. **强的 Sim-to-Real 效果**: 仅用合成数据训练，真机 84%/90% 成功率已超过部分需要真实演示的方法。

### 局限性

1. **依赖商业视频生成模型**: 当前使用 Kling 2.5 Turbo Pro，其生成质量、授权条款、价格等可能影响大规模使用；开源替代模型效果未知。
2. **重建速度仍较慢**: 每条序列约 14 分钟，大规模生成（如百万级序列）仍需大量 GPU 资源。
3. **仅在 Unitree G1 上验证**: 重定向和策略训练针对 G1 定制化，迁移到其他机器人形态（如 H1、GR-1）的通用性未得到充分验证。
4. **评估任务较简单**: 真机评估以单次抓取和爬楼梯为主，更复杂的序列操作（如搬运 + 放置）尚未充分验证。

### 潜在改进方向

1. **更强的视频生成模型**: 使用更真实、可控的视频生成模型（如开源 Wan、CogVideoX）替代 Kling。
2. **在线 Sim-to-Real 自适应**: 结合少量真实数据做在线域自适应，进一步缩小 Sim-to-Real gap。
3. **多机器人形态泛化**: 将 Pipeline 中的重定向模块（GMR）扩展至多种人形机器人平台。

### 可复现性评估

- [x] 代码开源（[github.com/NVlabs/GRAIL](https://github.com/NVlabs/GRAIL)，含 Docker 镜像）
- [x] 数据集开放（HuggingFace）
- [ ] 预训练策略模型（待发布）
- [x] 训练细节完整（GitHub 文档 + 论文）

---

## 关联笔记

### 基于

- [[SMPL-X]]: 人体参数化模型，用于 4D HOI 重建中的人体表示
- [[SONIC]]: 预训练全身追踪控制器，GRAIL 在其基础上训练 Latent Adaptor
- [[FoundationPose]]: 物体 6-DoF 估计，用于 4D HOI 重建栈
- [[SAM2]]: 视频分割，用于人体和物体分割 mask
- [[MoGe-2]]: 度量深度估计，用于 HOI 优化器的深度锚点
- [[GMR]]: 动作重定向，将 SMPL-X 姿态重定向到 G1 骨骼
- [[IsaacLab]]: 物理仿真平台，用于追踪策略训练
- [[HOI]]: 人-物交互，GRAIL 的核心数据类型

### 对比

- [[DeepMimic]]: 早期基于动作捕捉参考的追踪策略训练方法
- [[CHOIS]]: 基于人体动作生成的 HOI 方法

### 方法相关

- [[视频基础模型]]: GRAIL 利用 Kling AI 作为交互先验
- [[Loco-Manipulation]]: GRAIL 的核心任务类型
- [[Sim-to-Real]]: GRAIL 的最终部署目标
- [[behavior cloning]]: 视觉策略训练采用 BC 蒸馏

### 硬件/数据相关

- [[Unitree G1]]: 目标机器人平台
- [[GENMO]]: 人体运动估计方法

---

## 速查卡片

> [!summary] GRAIL: Generating Humanoid Loco-Manipulation from 3D Assets and Video Priors
> - **核心**: 全数字化生成 >20,000 条人形机器人 Loco-Manipulation 轨迹，无需物理遥操作
> - **方法**: 3D 场景 + Kling 视频生成 → GEM-SMPL + FoundationPose 4D HOI 重建 → GMR 重定向 → SONIC 追踪策略 → 自中心视觉策略
> - **结果**: Unitree G1 真机：84% 物体抓取、80% 新颖物体抓取、90% 爬楼梯
> - **代码**: [github.com/NVlabs/GRAIL](https://github.com/NVlabs/GRAIL)

---

*笔记创建时间: 2026-06-09*
