---
title: "GaussianDream: A Feed-Forward 3D Gaussian World Model for Robotic Manipulation"
method_name: "GaussianDream"
authors: [Zijian Zhang, Yuqing Jiang, Qian Cheng, Si Liu, Ding Zhao, Ping Luo, Weitao Zhou, Haibao Yu]
year: 2026
venue: arXiv
tags: [world-model, 3d-gaussian-splatting, vla, robot-manipulation, feed-forward, embodied-ai]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.20752v1
created: 2026-05-23
---

# 论文笔记：GaussianDream: A Feed-Forward 3D Gaussian World Model for Robotic Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Tuojing Intelligence, UCAS, CASIA, HKU, BUAA, CMU |
| 日期 | May 2026 |
| 项目主页 | https://github.com/TuojingAI/GaussianDream |
| 对比基线 | [[Pi05]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.20752) / [Code](https://github.com/TuojingAI/GaussianDream) |

---

## 一句话总结

> GaussianDream 是一个前馈式 [[3D Gaussian Splatting|3D 高斯]] 世界模型插件，通过当前重建 + 未来预测的不对称训练给 [[VLA]] 注入显式 3D 时空监督，推理时只保留轻量 prefix。

---

## 核心贡献

1. **统一的 3D 高斯世界模型框架**: 首次将语言条件 [[VLA]] 策略与结构化 [[3D Gaussian Splatting|3D 高斯表示]] 统一，解决了 VLA 在显式 3D 空间 grounding、像素级密集监督、短期预测性环境模拟三方面的瓶颈。
2. **不对称训练-推理设计 (Asymmetric Training-Inference)**: 训练阶段进行完整 3D 重建 + 未来预测以获得密集监督；推理时丢弃所有解码头，仅保留紧凑的 GaussianDream prefix，与 baseline 同等推理成本。
3. **时空预测模块 [[TGE|Temporal Gaussian Evolver]]**: 基于 [[VGGT]] 几何 backbone + 12 层 attention 的时序模块，将机器人轨迹转化为像素级 RGB / 深度 / 3D 场景流的密集监督。
4. **强实证结果**: LIBERO 平均 98.4%，RoboCasa Human-50 平均 52.6%，真实双臂机器人 4 场景平均 50.0%（相对 π₀.₅ 提升 +15.6 个百分点）。

---

## 问题背景

### 要解决的问题

当前 [[VLA]] 策略在机器人操作任务上展示了强大的语义理解能力，但仍受限于三大瓶颈：

1. **缺乏显式 3D 空间 grounding**：基于 2D RGB 的 backbone 难以建立精确的度量空间关系
2. **训练时缺乏像素级密集监督**：仅靠动作模仿损失浪费了机器人轨迹中蕴含的视觉信息
3. **缺乏短期预测性环境模拟**：没有"未来观测"的概念，难以学到环境动力学

### 现有方法的局限

- **3D 增强策略**（如 3D-VLA、ManiGaussian）：使用显式 3D 表示但通常需要重建网络做推理时解码，开销大
- **视频 / latent [[World Model]]**（如 [[Dreamer 4]]、[[DINO-WM]]）：自回归 rollout 推理慢，且 2D 像素或 latent 缺乏度量几何
- **2D VLA**（[[Pi05|π₀.₅]]、OpenVLA）：高效但欠缺 3D 几何与未来感知

### 本文的动机

如果能在**训练阶段**用结构化 3D 高斯做密集监督，在**推理阶段**只保留一段紧凑的 prefix tokens 喂给动作 head，就能"既要又要"——既享受世界模型的几何 / 时序学习信号，又保留 VLA 前馈推理的效率。这就是 GaussianDream 的核心 motivation。

---

## 方法详解

### 模型架构

GaussianDream 采用 **前馈 prefix + 不对称双解码** 架构：

- **输入**: 语言指令 $l$ + 多视图观测序列 $o_{t-K:t}$（3 帧：$\{t-10, t-5, t\}$）+ 机器人状态 $s_t$
- **几何 Backbone**: 冻结的 [[VGGT]] 提取多尺度 3D-aware patch features，pooling 到 $32\times32$ grid
- **时序融合模块**: [[TGE|Temporal Gaussian Evolver]]，12 个 attention block × 8 head，512↔2048 维投影
- **GaussianDream Prefix**: $Z_t^{GD} \in \mathbb{R}^{1024 \times 2048}$，作为下游解码与动作生成的统一表征
- **输出**:
  - 训练时：当前帧高斯 $G_t$ + 未来帧高斯 $\hat{G}_{t+\Delta}$ + 动作块 $a_t$
  - 推理时：仅动作块 $a_t$，丢弃所有渲染/深度/速度 head
- **动作 head**: 基于 [[Pi05|π₀.₅]] 的 [[Flow Matching]] 策略

### 核心模块

#### 模块 1: 当前帧高斯重建 (Current Gaussian Reconstruction)

**设计动机**: 用 [[3D Gaussian Splatting]] 的可微分渲染做像素级监督，让 prefix tokens 显式编码 3D 几何。

**具体实现**:

- **Feature Decoding Backbone $B_G$**: 从 $32\times32$ token grid 上采样到 $256\times256\times128$ 的特征图
  - Block 1: Transposed conv (kernel 4, stride 2) → $64\times64$, 512 ch
  - Block 2: Transposed conv → $128\times128$, 256 ch
  - Block 3: Transposed conv → $256\times256$, 128 ch
  - Refinement: [[DPT]] 风格的多尺度残差融合
- **三个并行 Head**:
  - **Geometry head**: 8 通道（quaternion 4 + scale 3 + opacity 1）
  - **Appearance head**: 9 通道（degree-1 [[球面谐波系数|SH coefficients]]）
  - **Depth head**: 1 通道，用于反投影确定高斯中心位置 $\mu_i$
- **高斯总数**: 每帧 $N_t = 256 \times 256 = 65536$ 个高斯

#### 模块 2: 未来帧高斯预测 (Future Gaussian Prediction)

**设计动机**: 通过预测短期未来状态学到环境动力学，但只预测**位置位移**而复用其他属性，降低预测难度。

**具体实现**:

- **Horizon Embedding**: 可学习的时间偏移嵌入 $e_\Delta$，对每个预测 horizon $\Delta \in \mathcal{H}$ 编码
- **Velocity Head $H_{vel}$**: 输出 3 通道速度场（x, y, z 位移），通过 Tanh + horizon scaling $\alpha_\Delta$ 限制位移幅度
- **位置更新**: $\hat{\mu}_i^{t+\Delta} = \mu_i^t + \Delta x_i^{(\Delta)}$
- **属性复用**: 外观 / 几何属性 $\theta_i^t$ 不变，仅位置移动
- **监督信号**: 用伪 ground-truth 3D scene flow（[[光流|RAFT 光流]] + [[Depth Anything V2|深度]] 反投影得到）

#### 模块 3: [[TGE|Temporal Gaussian Evolver]]

**设计动机**: 让 GaussianDream prefix 同时编码"我在哪"（空间）和"我刚从哪儿来"（时序）。

**具体实现**:

- **输入**: 3 帧 [[VGGT]] patch features $P_{t-K:t} \in \mathbb{R}^{3 \times 32 \times 32 \times 512}$ + 可学习 query $Q^{GD}$
- **块结构（共 12 个）**:
  1. Within-frame self-attention（query + patch token 联合）
  2. Cross-frame temporal attention（同一 token 槽跨 3 帧）
  3. MLP（4× channel expansion）
- **维度变换**: $Q^{GD}$ 2048 → 512 进入 TGE → 输出投影回 2048
- **输出**: 当前帧的 token 子集作为 $Z_t^{GD}$

#### 模块 4: 不对称训练-推理设计

**训练**:
$$\text{Encoder} \to Z_t^{GD} \to \{\text{Reconstruction Head}, \text{Prediction Head}, \text{Action Head}\}$$
联合优化所有 loss。

**推理**:
$$\text{Encoder} \to Z_t^{GD} \to \text{Action Head only}$$
所有 rendering / depth / velocity head 丢弃，与 π₀.₅ 同等推理速度。

---

## 关键公式

### 公式 1: [[GaussianDream Prefix|GaussianDream Prefix 提取]]

$$
Z_t^{GD} = F_\omega(o_{t-K:t}, Q^{GD})
$$

**含义**: 编码器 $F_\omega$ 将历史观测 $o_{t-K:t}$ 与可学习 query $Q^{GD}$ 融合，得到统一的 GaussianDream prefix。

**符号说明**:
- $o_{t-K:t}$: 历史观测序列（3 帧）
- $Q^{GD}$: 可学习的 query token
- $Z_t^{GD} \in \mathbb{R}^{1024 \times 2048}$: 2048 维 prefix，$32\times32$ grid

### 公式 2: 当前重建与未来预测双解码

$$
G_t = R_\phi(Z_t^{GD}, o_t), \quad \hat{G}_{t+\Delta} = D_\psi(G_t, Z_t^{GD}, \Delta)
$$

**含义**: 当前重建头 $R_\phi$ 产生当前帧 3D 高斯；未来预测头 $D_\psi$ 基于当前高斯与 horizon $\Delta$ 预测未来高斯。

**符号说明**:
- $G_t$: 当前帧高斯集合
- $\hat{G}_{t+\Delta}$: 未来 $\Delta$ 步高斯
- $\Delta \in \mathcal{H}$: 预测 horizon 集合

### 公式 3: 动作生成

$$
a_t = \pi_\theta(o_t, l, s_t; Z_t^{GD})
$$

**含义**: 策略 $\pi_\theta$ 以观测、语言、机器人状态为输入，条件于 GaussianDream prefix，输出动作块。

**符号说明**:
- $l$: 语言指令
- $s_t$: 机器人状态（关节角、夹爪等）
- $Z_t^{GD}$: prefix（推理时唯一来自世界模型的信号）

### 公式 4: Feature Decoding

$$
F_t^G = B_G(\text{Grid}(Z_t^{GD})) \in \mathbb{R}^{256 \times 256 \times 128}
$$

**含义**: prefix 重排为空间 grid 后通过 $B_G$ 上采样为高分辨率特征图，供三个 head 共享。

**符号说明**:
- $\text{Grid}(\cdot)$: 将 1024 个 token 重排为 $32\times32$ 空间 grid
- $B_G$: 上采样 backbone（DPT 风格）

### 公式 5: 高斯属性预测

$$
\Theta_t = \{(\mathbf{q}_i^t, \mathbf{s}_i^t, o_i^t, \mathbf{c}_i^t)\}_{i=1}^{N_t}, \quad D_t = H_{depth}(F_t^G)
$$

**含义**: Geometry head 输出旋转/缩放/不透明度；Appearance head 输出 SH 颜色；Depth head 输出深度图。

**符号说明**:
- $\mathbf{q}_i \in \mathbb{R}^4$: 单位四元数旋转
- $\mathbf{s}_i \in \mathbb{R}^3$: 各向异性缩放
- $o_i \in \mathbb{R}$: 不透明度
- $\mathbf{c}_i \in \mathbb{R}^9$: degree-1 [[球面谐波系数|SH]] 系数

### 公式 6: 高斯状态组装

$$
G_t = \mathcal{A}(D_t, \Theta_t) = \{(\mu_i^t, \theta_i^t)\}_{i=1}^{N_t}
$$

**含义**: 通过深度反投影确定高斯中心 $\mu_i^t = \Pi^{-1}(u, v, D_t(u,v); K)$，与属性 $\theta_i^t$ 组成完整 3D 高斯。

**符号说明**:
- $\Pi^{-1}$: [[相机投影|反相机投影]]
- $K$: 相机内参矩阵
- $N_t = 256 \times 256 = 65536$ 个高斯每帧

### 公式 7: VGGT 时序特征提取

$$
P_{t-K:t}^{(m)} = W_m \cdot \mathcal{P}_{32\times32}(E_{VGGT}^{(m)}(o_{t-K:t}))
$$

**含义**: 用冻结 [[VGGT]] 提取多尺度特征，pooling 到 $32\times32$ grid 后用线性层 $W_m$ 投影到 512 维。

**符号说明**:
- $E_{VGGT}^{(m)}$: VGGT 第 $m$ 层特征
- $\mathcal{P}_{32\times32}$: 空间池化到 $32\times32$
- $W_m$: 投影矩阵

### 公式 8: TGE 块内更新

$$
\begin{aligned}
X &\leftarrow X + \text{SelfAttn}_{\text{within}}(X) \\
X &\leftarrow X + \text{SelfAttn}_{\text{temporal}}(X) \\
X &\leftarrow X + \text{MLP}(X)
\end{aligned}
$$

**含义**: 每个 TGE block 串联帧内自注意力、跨帧时序注意力和 MLP，共 12 层。

### 公式 9: Horizon-Conditioned Displacement

$$
\nu_t^{(\Delta)} = H_{vel}(B_{pred}(Z_t^{GD}), e_\Delta), \quad \Delta X_t^{(\Delta)} = \alpha_\Delta \nu_t^{(\Delta)}
$$

$$
\hat{\mu}_i^{t+\Delta} = \mu_i^t + \Delta x_i^{(\Delta)}, \quad \hat{G}_{t+\Delta} = \{(\hat{\mu}_i^{t+\Delta}, \theta_i^t)\}_{i=1}^{N_t}
$$

**含义**: 给定 horizon embedding $e_\Delta$，预测每个高斯的速度场 $\nu$，乘以 horizon scaling 得到位移；属性 $\theta_i^t$ 复用当前帧。

**符号说明**:
- $e_\Delta$: 可学习 horizon embedding
- $\alpha_\Delta$: 时间尺度因子（用 Tanh 限幅）
- $\Delta x_i^{(\Delta)}$: 第 $i$ 个高斯在 horizon $\Delta$ 下的位移

### 公式 10: Stage I 预训练损失

$$
\begin{aligned}
\mathcal{L}_{GD} = \;& \lambda_{cur}^{depth} \mathcal{L}_{cur}^{depth} + \lambda_{cur}^{render} \mathcal{L}_{cur}^{render} \\
&+ \sum_{\Delta \in \mathcal{H}} w_\Delta \left( \lambda_{depth} \mathcal{L}_{depth}^{(\Delta)} + \lambda_{render} \mathcal{L}_{render}^{(\Delta)} + \lambda_{flow} \mathcal{L}_{flow}^{(\Delta)} \right)
\end{aligned}
$$

**含义**: 当前重建损失（深度 + 渲染）+ 未来预测损失（深度 + 渲染 + 3D 流），每个 horizon $\Delta$ 有权重 $w_\Delta$。

**符号说明**:
- $\mathcal{L}_{cur}^{render}$: 当前帧 RGB 渲染 L1 + LPIPS 损失
- $\mathcal{L}_{cur}^{depth}$: 当前帧深度 L1 损失
- $\mathcal{L}_{flow}^{(\Delta)}$: 未来 horizon 的 3D 场景流损失

### 公式 11: Stage II 动作 [[Flow Matching]] 损失

$$
\mathcal{L}_{act} = \mathbb{E}_{\tau, \epsilon, a_t^*}\left[ \| v_\theta(\tau \epsilon + (1-\tau)a_t^*, c_t, \tau) - (\epsilon - a_t^*) \|_2^2 \right]
$$

**含义**: 基于 π₀.₅ 的 [[Flow Matching]] 目标，预测从噪声 $\epsilon$ 到真值 $a_t^*$ 的速度场。

**符号说明**:
- $\tau \sim \mathcal{U}(0, 1)$: 时间步
- $\epsilon \sim \mathcal{N}(0, I)$: 高斯噪声
- $c_t$: 条件信息（含 $Z_t^{GD}$）

### 公式 12-14: 3D 场景流构造

$$
(u', v') = (u, v) + f_{t \to t+1}(u, v)
$$

$$
\mathbf{x}_t = \Pi^{-1}(u, v, D_t(u, v); K), \quad \mathbf{x}_{t+1} = \Pi^{-1}(u', v', D_{t+1}(u', v'); K)
$$

$$
F^{3D}_{t \to t+1}(u, v) = \mathbf{x}_{t+1} - \mathbf{x}_t
$$

**含义**: 用 2D [[光流]] 在像素空间扭曲 → 用深度反投影到 3D → 得到 3D 场景流 ground truth。

**符号说明**:
- $f_{t \to t+1}$: 2D 光流（[[光流|RAFT]] 或 Farneback 备份）
- $D_t$: [[Depth Anything V2]] 深度图
- $\Pi^{-1}$: 相机反投影

### 公式 15-16: 渲染与深度损失

$$
\mathcal{L}_{render}^{(\Delta)} = \| I_{t+\Delta} - \mathcal{R}(\hat{G}_{t+\Delta}) \|_1 + \lambda_{lpips} \mathcal{L}_{lpips}
$$

$$
\mathcal{L}_{depth}^{(\Delta)} = \| D_{t+\Delta} - \mathcal{D}(\hat{G}_{t+\Delta}) \|_1
$$

**含义**: 对未来 horizon 预测的高斯做可微分渲染，与真实未来帧的 RGB / 深度对齐。

**符号说明**:
- $\mathcal{R}$: 可微分高斯渲染器
- $\mathcal{D}$: 深度渲染器

### 公式 17: 3D 流损失

$$
\mathcal{L}_{flow}^{(\Delta)} = \frac{\sum_i M_i^{(\Delta)} \| \Delta x_i^{(\Delta)} - F_i^{3D,(\Delta)} \|_1}{\sum_i M_i^{(\Delta)} + \varepsilon}
$$

**含义**: 仅对有效像素（mask $M_i^{(\Delta)}$ 过滤越界 / 无效深度）计算预测位移与伪 GT 3D 流的 L1 误差。

**符号说明**:
- $M_i^{(\Delta)}$: 有效性 mask
- $\varepsilon$: 数值稳定项

### 公式 18: 联合训练目标

$$
\mathcal{L} = \mathcal{L}_{act} + \lambda_{GD} \mathcal{L}_{GD}
$$

**含义**: Stage II 同时优化动作策略与世界模型监督，$\lambda_{GD}$ 平衡两者。

---

## 关键图表

### Figure 1: Manipulation Policy Paradigms / 操作策略范式对比

![Figure 1](https://arxiv.org/html/2605.20752v1/assets/comparation_3.png)

**说明**: 对比 2D 策略、3D 增强策略、视频/latent 世界模型与 GaussianDream。GaussianDream 用**当前 + 未来高斯**学到结构化 3D 监督，同时保留高效的 prefix-based 动作生成（无自回归 rollout）。

### Figure 2: GaussianDream Framework / 系统总图

![Figure 2](https://arxiv.org/html/2605.20752v1/assets/framework_final_v.drawio.png)

**说明**: 整体架构图。冻结的 [[VGGT]] backbone → [[TGE]] 时序融合 → GaussianDream Prefix $Z_t^{GD}$ → 三个并行分支：当前高斯重建（geometry / appearance / depth heads）、未来高斯预测（velocity head + horizon embedding）、动作策略（[[Flow Matching]] head）。推理时只保留 prefix → action head 这条路径。

### Figure 3: Real-Robot Evaluation Scenarios / 真机评测场景

![Figure 3](https://arxiv.org/html/2605.20752v1/assets/piper_setup_2.drawio.png)

**说明**: 真机测试覆盖 4 类场景：属性 grounding（颜色/形状）、空间关系（左右上下）、堆叠/拆解、长时程任务。每场景多个变体，全面评估策略的 3D 空间理解。

### Figure 4: Depth Rendering Visualization / 深度渲染可视化

![Figure 4](https://arxiv.org/html/2605.20752v1/assets/q_vis.drawio_final_2.png)

**说明**: 第 1、3 行为 ground-truth 深度序列；第 2、4 行为 GaussianDream 重建（当前帧）+ 预测（5 个未来 horizon）。展示时序一致性与物体布局/空间关系的保留。

### Figure 5: Depth & Scene-Flow Preparation / 伪 GT 准备可视化

![Figure 5](https://arxiv.org/html/2605.20752v1/assets/LIBERO_Robocasa_flow_depth.drawio_compressed.png)

**说明**: LIBERO 与 RoboCasa demo 上的伪深度（[[Depth Anything V2]]）与场景流（[[光流|RAFT]] 反投影）可视化，用于训练时的密集监督。

### Figure 6: Real-Robot Hardware Setup / 真机硬件设置

![Figure 6](https://arxiv.org/html/2605.20752v1/assets/hardware_compressed.png)

**说明**: 双臂 leader-follower 平台。Follower 臂安装在移动底座上，配 agent-view（全局工作空间）+ wrist-mounted（局部反馈）双视角相机。

### Figure 7: Real-Robot Visual Observations / 真机视觉观测

![Figure 7](https://arxiv.org/html/2605.20752v1/assets/piper_vis.drawio.png)

**说明**: agent-view（全局视角）与 wrist-mounted（手腕视角）的实拍样本，分别提供工作空间整体视图与细粒度交互反馈。

### Figure 8: Inference Smoothness / 推理轨迹平滑度

![Figure 8](https://arxiv.org/html/2605.20752v1/assets/appendix_smooth.png)

**说明**: GaussianDream 推理轨迹比 baseline 更平滑，归功于 prefix 中编码的 3D 空间一致性约束。

### Figure 9: Additional Qualitative Results / 更多定性结果

![Figure 9](https://arxiv.org/html/2605.20752v1/assets/LIBERO_appendix_vis.drawio.png)

**说明**: 附录中更多重建 + 预测案例，对比观测帧 / 重建几何 / 预测未来状态，验证模型在不同任务上的泛化。

### Table 1: LIBERO 主要结果

| Method | Spatial | Object | Goal | Long | Average |
|--------|---------|--------|------|------|---------|
| π₀ | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 |
| [[Pi05\|π₀.₅]] | 97.8 | 98.8 | 97.6 | 92.4 | 96.7 |
| **GaussianDream** | **99.0** | 99.6 | **99.0** | 96.0 | **98.4** |
| LingBot-VA | 98.5 | 99.6 | 97.2 | 98.5 | 98.5 |

**说明**: 在前馈策略类别下 GaussianDream 取得 SOTA（Spatial / Goal 最佳，整体 98.4%）。LingBot-VA 整体略高但**推理时需要自回归视频-动作 rollout**，不是同类比较。

### Table 2: RoboCasa Human-50 结果

| Method | Pick&Place | Doors/Drawers | Others | Average |
|--------|-----------|----------------|--------|---------|
| π₀ | 13.95 | 53.1 | 58.5 | 42.4 |
| [[Pi05\|π₀.₅]] | 36.0 | 46.5 | 39.5 | 40.1 |
| GeoPredict | 22.7 | 75.1 | 62.4 | 52.4 |
| **GaussianDream** | **43.8** | 65.2 | 52.0 | **52.6** |

**说明**: GaussianDream 在 pick-and-place 这类对**精细定位**最敏感的任务上提升最显著（+7.8 vs π₀.₅），整体 +12.5 个百分点。

### Table 3: 真实机器人 4 场景结果

| Method | Scene-A | Scene-B | Scene-C | Scene-D | Average |
|--------|---------|---------|---------|---------|---------|
| [[Pi05\|π₀.₅]] | 42.5 | 50.0 | 25.0 | 20.0 | 34.4 |
| **GaussianDream** | 55.0 | 70.0 | 35.0 | 40.0 | **50.0** |

**说明**: 真机 4 场景平均 +15.6 个百分点，验证 3D 监督学到的能力可迁移到真实场景。Scene-D（长时程）相对提升最大（+20）。

### Table 4: 消融实验

| Config | Recon | Pred | Render | Depth | LIBERO Avg |
|--------|---|---|---|---|---|
| 1 | ✓ | ✗ | ✗ | ✗ | 97.0 |
| 2 | ✓ | ✗ | ✓ | ✓ | 97.3 |
| 3 | ✓ | ✓ | ✗ | ✓ | 97.5 |
| 4 | ✓ | ✓ | ✓ | ✗ | 97.2 |
| **5** | ✓ | ✓ | ✓ | ✓ | **98.4** |

**关键发现**:
- 仅当前重建（无未来预测）已带来 +3.0 baseline 提升
- 加入未来预测 +0.5
- RGB 渲染监督 +0.3
- 深度监督是 metric geometry 的关键（无 depth 时降到 97.2）
- **四种监督叠加效果最强**（98.4%）

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO]] | 4 套件 × 10 任务 × 50 demo | 长时程、对象、空间、目标 | 仿真训练 / 测试 |
| [[RoboCasa]] Human-50 | 24 厨房任务 × 50 trial × 5 场景 | 大规模家庭 mujoco | 仿真训练 / 测试 |
| 真机 dual-arm | leader-follower + 移动底座 | agent + wrist cam | 真实评测 |

### 实现细节

- **Backbone**: [[VGGT]] (~1B 参数，冻结)
- **基础策略**: [[Pi05|π₀.₅]] + [[Flow Matching]] 头
- **Prefix 维度**: 1024 tokens × 2048 维（$32\times32$ grid）
- **TGE**: 12 blocks × 8 heads，512 维内 + 2048 维外
- **每帧高斯数**: 65536（$256\times256$）
- **历史帧**: 3 帧 $\{t-10, t-5, t\}$
- **预测 horizon**: $\mathcal{H}$（论文用 5 个未来步）
- **训练**: 两阶段，Stage I 仅 $\mathcal{L}_{GD}$，Stage II 联合 $\mathcal{L}_{act} + \lambda_{GD}\mathcal{L}_{GD}$
- **伪 GT**: [[Depth Anything V2]] 深度 + [[光流|RAFT]] 光流（Farneback fallback）
- **框架**: [[PyTorch]]

### 可视化结果

- **时序一致性**: Figure 4 显示 5 个未来 horizon 上预测深度仍保持物体几何与空间关系
- **真机平滑度**: Figure 8 显示 GaussianDream 轨迹明显更平滑（baseline 抖动频繁）
- **泛化性**: 在没见过的厨房布局和真机场景仍稳定

---

## 批判性思考

### 优点

1. **架构优雅**：不对称训练-推理设计是真正的"既要又要"，零额外推理成本但获得显著性能提升
2. **监督信号丰富**：RGB / 深度 / 3D 流三层监督叠加，消融实验验证四者均必要
3. **预测难度合理**：只预测位置位移，属性复用，让未来预测任务"足够难但又可学"
4. **真机泛化强**：+15.6 个百分点的真机提升说明 3D 监督学到的不是仿真 overfit
5. **可作为通用插件**：宣称是 plug-in 形式，理论上可挂到任意 VLA backbone

### 局限性

1. **依赖伪 GT 质量**：[[Depth Anything V2]] 深度和 [[光流|RAFT]] 光流在反光/透明物体上失效，会污染监督信号
2. **单视角 dense Gaussian 是有损表示**：65536 个 surface-attached 高斯无法重建被遮挡区域，未来预测在大姿态变化下可能崩溃
3. **horizon 必须是固定集合 $\mathcal{H}$**：不能任意泛化到训练未见过的预测步数
4. **训练开销大**：渲染 + 深度 + 流三套 head 显著增加显存与训练时间，尽管推理时丢弃
5. **TGE 仅 3 帧历史**：对长时程依赖的任务可能不足

### 潜在改进方向

1. **改进伪 GT**：用 [[MoGe-2]] 等更精确的几何先验替代 [[Depth Anything V2]]
2. **可变 horizon**：用连续时间编码（如 sinusoidal）支持任意未来步预测
3. **多视角融合**：当前只用单 agent-view 监督；融合 wrist cam 可改善遮挡
4. **更强 backbone**：换成 [[Pi3X]] 这类同时输出几何 + 语义的统一表示
5. **与 latent world model 互补**：把 latent rollout（如 [[DINO-WM]]）和 explicit Gaussian 监督结合，覆盖长 / 短期

### 可复现性评估

- [x] 代码开源（github.com/TuojingAI/GaussianDream）
- [ ] 预训练模型（未公开提及）
- [x] 训练细节较完整（公式 / 维度 / 损失权重）
- [x] 数据集可获取（LIBERO / RoboCasa 开源）

---

## 关联笔记

### 基于

- [[Pi05|π₀.₅]]: 动作 head 与 [[Flow Matching]] 训练范式直接继承
- [[VGGT]]: 几何 backbone，冻结使用
- [[3D Gaussian Splatting]]: 显式 3D 表示与可微分渲染
- [[Depth Anything V2]]: 伪 GT 深度
- [[光流|RAFT]]: 2D 光流估计

### 对比

- [[Pi05|π₀.₅]]: 同推理成本下 +15.6 (真机) / +1.7 (LIBERO) / +12.5 (RoboCasa) 个百分点
- GR-MG / 3D-VLA: 同样追求 3D 增强但通常需要推理时解码
- [[Dreamer 4]] / [[DINO-WM]]: latent world model，自回归 rollout 推理慢
- [[EA-WM]]: 视频扩散世界模型，但本文用显式 3D 高斯而非 2D 视频
- LingBot-VA: 整体 LIBERO 略高但需要 video-action rollout 推理

### 方法相关

- [[TGE|Temporal Gaussian Evolver]]: 核心时序融合模块
- [[GeoPredict]]: 同期相关工作（RoboCasa 上的对比基线，纯几何预测，无 RGB 监督）
- [[Flow Matching]]: 动作生成范式
- [[球面谐波系数|SH coefficients]]: 高斯外观参数化
- [[相机投影]]: 深度反投影到 3D 位置
- [[Action Chunking|动作块]]: 输出形式
- [[DPT]]: 上采样 backbone 设计

### 硬件/数据相关

- [[LIBERO]]: 主仿真 benchmark
- [[RoboCasa]]: 厨房任务仿真
- [[PyTorch]]: 实现框架

---

## 速查卡片

> [!summary] GaussianDream
> - **核心**: 前馈 3D 高斯世界模型插件，训练时给 VLA 加密集 3D 监督，推理时只保留 prefix
> - **方法**: VGGT + TGE → GaussianDream Prefix → 当前重建 + 未来预测 + 动作；推理只保留 prefix → action
> - **结果**: LIBERO 98.4% / RoboCasa 52.6% / 真机 50.0%（vs π₀.₅ 34.4%，+15.6）
> - **代码**: https://github.com/TuojingAI/GaussianDream

---

*笔记创建时间: 2026-05-23*
