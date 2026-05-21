---
title: "OrbiSim: World Models as Differentiable Physics Engines for Embodied Intelligence"
method_name: "OrbiSim"
authors: [Jiajian Li, Jingyuan Huang, Junru Gong, Qi Wang, Xiaokang Yang, Yunbo Wang]
year: 2026
venue: arXiv
tags: [world-model, differentiable-physics, robot-simulation, latent-diffusion, model-based-rl, real-to-sim]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.16395v1
created: 2026-05-19
---

# 论文笔记：OrbiSim: World Models as Differentiable Physics Engines for Embodied Intelligence

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 上海交通大学 MoE 人工智能重点实验室 / AI Institute / 计算机科学学院 |
| 日期 | May 2026 |
| 项目主页 | https://jjleejj85.github.io/projects/orbisim |
| 对比基线 | [[Vid2World]] / [[AdaWorld]] / [[DreamerV3]] / [[PPO]] / [[SAC]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.16395) / [Code (coming soon)](https://jjleejj85.github.io/projects/orbisim) |

---

## 一句话总结

> OrbiSim 把 [[世界模型]] 重塑为**端到端可微的物理引擎**，通过状态-像素解耦的双模块（[[OrbiSim-Dynamics]] + [[OrbiSim-Vision]]）让策略梯度可以直接穿过仿真环境，在机械臂操作任务上把 RL 成功率从 25% 提升到 42.7%。

---

## 核心贡献

1. **可微物理引擎范式**: 提出"世界模型即可微物理引擎"的统一视角，把场景资产 (asset) 条件化与神经动力学耦合在一起，构成端到端可微的仿真闭环，弥合了经典物理引擎（不可微）与生成式世界模型（缺物理状态）之间的鸿沟。
2. **状态-像素解耦双模块**: [[OrbiSim-Dynamics]] 负责显式物理状态预测，[[OrbiSim-Vision]] 负责状态条件下的图像渲染——梯度只在 Dynamics 中传播，规避了 monolithic [[视频扩散模型]] 反传时的梯度退化与训练不稳定。
3. **解析策略梯度求解稀疏奖励**: 利用全程可微性，直接对累计奖励反向传播得到 [[解析策略梯度|Analytic Policy Gradient]]，在 Robosuite Push 等稀疏奖励任务上显著超过 [[DreamerV3]] 与无模型 RL。
4. **Real-to-Sim 接口**: [[State Inference]] (StaInf) 从单帧 RGB 推断可见物理状态，[[Physics Inference]] (PhyInf) 从 64 帧视频窗口推断隐藏物理参数（密度、摩擦），把真实视频映射回可执行的仿真配置。

---

## 问题背景

### 要解决的问题

机器人具身智能需要一个**既能高保真模拟物理交互、又能为策略学习提供梯度信号**的环境。经典物理引擎（[[MuJoCo]]、[[IsaacLab]]）虽精确但**不可微**，状态转移黑盒化阻碍了梯度流；现有生成式 [[世界模型]]（[[DreamerV3]]、[[Vid2World]]、[[AdaWorld]]）虽端到端可学习，却**缺乏显式物理状态、不支持结构化资产、长 horizon 失真**。

### 现有方法的局限

- **经典模拟器**: 不可微，只能用无梯度 RL（PPO/SAC），样本效率低；难以表达隐藏物理参数推断任务。
- **像素级世界模型**: [[Vid2World]]、[[AdaWorld]] 等把动力学和视觉耦合在一个 [[视频扩散模型]] 中，梯度路径经过 VAE 解码器和长扩散链，**梯度爆炸/消失严重**，且无法访问刚体姿态、速度等结构化物理量。
- **隐空间世界模型**: [[DreamerV3]] 等在抽象隐空间想象 (latent imagination)，**物理可解释性差**，对资产几何/质量/摩擦不敏感。

### 本文的动机

把世界模型重新设计成"**可微物理引擎**"形式：用物理状态作为一等公民（first-class）、用 Transformer recurrent dynamics 建模多体相互作用、用 latent diffusion 解耦视觉合成。这样既保留了物理仿真器的可解释性，又获得了神经网络的可微性与可学习性。

---

## 方法详解

### 模型架构

OrbiSim 采用 **状态-像素双流可微仿真** 架构：

- **输入**:
  - 当前物理状态 $x_t$ = (机器人状态: 位置/线速度/旋转矩阵) ⊕ (物体状态: 位置/线速度/四元数)
  - 动作 $a_{t-1}$
  - 静态场景描述符 $\bar{x}$ = (几何、质量、惯量、摩擦、shape encoding、重力、接触参数)
  - （可选）RGB 观测 $o_t$
- **双模块**:
  - [[OrbiSim-Dynamics]]: 物体中心 (object-centric) 的 [[循环状态空间模型|RSSM]]，预测下一时刻物理状态 $\hat{x}_t$
  - [[OrbiSim-Vision]]: 以预测状态为条件的 [[LDM|Latent Diffusion]] 模型，渲染 RGB 帧 $\hat{o}_t$
- **输出**: $\hat{x}_t, \hat{o}_t$
- **可微性**: 解析梯度沿物理状态链 $\partial x_{t+1}/\partial x_t$ 与 $\partial x_{t+1}/\partial a_t$ 传播

### 核心模块

#### 模块1: OrbiSim-Dynamics — 物体中心循环动力学

**设计动机**: 直接预测显式物理状态而非像素，让梯度短路径地穿过仿真，避开 [[视频扩散模型]] 的长去噪链路。

**具体实现**:
- **Encoder** $f_\phi^{enc}$: 把物理状态 $x_t$ 与 hidden state $h_t$ 编码成潜在 token $z_t$
- **Coupling Module** $f_\phi^{cp}$: 基于 [[Transformer]] 的 self-attention，把上一时刻 token $z_{t-1}$、动作 $a_{t-1}$、静态描述符 $\bar{x}$ 融合成事件嵌入 $e_t$，捕获**多实体相互作用**（机器人↔物体、物体↔物体）
- **Recurrent Memory** $f_\phi^{rec}$: 更新隐藏状态 $h_t = f_\phi^{rec}(h_{t-1}, e_t)$
- **Transition Module** $f_\phi^{tra}$: 从 $h_t$ 预测下一时刻潜在状态 $\hat{z}_t$
- **Decoder** $f_\phi^{dec}$: 把 $\hat{z}_t$ 解码回物理状态预测 $\hat{x}_t$
- **AdaLN 调制**: 用 [[AdaLN|Adaptive Layer Norm]] 让动作和物理属性以 scale/shift 形式调制每层激活，避免 token 拼接造成的维度膨胀

#### 模块2: OrbiSim-Vision — 状态引导的潜空间扩散渲染

**设计动机**: 仿真闭环中视觉只在**需要 RGB 观测**时调用，因此可以和动力学解耦；用 [[LDM|Latent Diffusion]] 在 VAE 隐空间渲染图像。

**具体实现**:
- **条件 1 — 空间条件图 (Spatial Condition Map)**: 把预测物体姿态投影到图像平面，生成 heatmap / depth-like prior，concat 到 [[UNet]] 输入
- **条件 2 — 物体级 token**: 物体状态编码后通过 [[交叉注意力]] 注入去噪网络
- **条件 3 — 视觉锚定 (Visual Anchoring)**: 把最近 $K \in \{0,1,\ldots,6\}$ 帧的扰动 latent 拼到去噪输入中，保证时序一致性
- **条件 4 — 物理 grounding**: 显式编码 $\bar{x}$ 与 $x_t$，让扩散模型"知道"自己在渲染什么物理场景
- **训练策略**: 随机采样上下文长度 $K$，使推理时支持冷启动到 long-horizon 各种场景

#### 模块3: Real-to-Sim 推理 — 把真实视频映射回仿真

**StaInf (State Inference)**:
- 输入: 单帧 RGB
- 共享 [[OrbiSim-Vision]] 的冻结 encoder 提取 latent
- 输出: 机器人末端位姿、物体位置/旋转/尺寸、视觉属性

**PhyInf (Physics Inference)**:
- 输入: 64 帧视频窗口的 latent 序列
- 通过冻结的 [[OrbiSim-Dynamics]] 做 rollout，用预测状态与真实状态的一致性 loss 反向监督
- 输出: 隐藏物理参数（密度、摩擦系数等）

---

## 关键公式

### 公式1: [[世界模型|Dynamics 预测目标]]

$$
\hat{x}_t = f_\phi^{dyn}(x_{0:t-1}, a_{t-1}, \bar{x})
$$

**含义**: OrbiSim-Dynamics 是一个以历史状态、动作、静态资产描述符为输入的物理状态预测器，等价于一个神经物理引擎的状态转移函数。

**符号说明**:
- $x_t$: 时刻 $t$ 的完整物理状态（机器人+物体）
- $a_{t-1}$: 上一时刻动作
- $\bar{x}$: 静态场景描述符（几何、质量、摩擦等）
- $f_\phi^{dyn}$: 总动力学函数，参数 $\phi$

### 公式2: [[循环状态空间模型|RSSM 分解]]

$$
\begin{aligned}
e_t &= f_\phi^{cp}(z_{t-1}, a_{t-1}, \bar{x}) \\
h_t &= f_\phi^{rec}(h_{t-1}, e_t) \\
\hat{z}_t &= f_\phi^{tra}(h_t) \\
z_t &= f_\phi^{enc}(h_t, x_t) \\
\hat{x}_t &= f_\phi^{dec}(\hat{z}_t)
\end{aligned}
$$

**含义**: 把动力学拆成 Coupling → Recurrent → Transition → Encoder/Decoder 五个组件，分别处理多体耦合、时序记忆、潜在转移、状态编解码。

**符号说明**:
- $z_t$: 时刻 $t$ 的潜在状态 token
- $h_t$: 循环隐状态
- $e_t$: 融合后的事件嵌入
- $\hat{z}_t, \hat{x}_t$: 预测的潜在态与物理态

### 公式3: [[AdaLN|自适应层归一化]]

$$
\mathrm{AdaLN}(u, c) = \gamma(c) \odot \mathrm{LN}(u) + \beta(c)
$$

**含义**: 用条件向量 $c$（动作 / 物理属性）生成 scale 与 shift，调制 LayerNorm 后的特征，从而把控制信号注入每一层。

**符号说明**:
- $u$: 输入特征
- $c$: 条件（action + 物理属性）
- $\gamma(c), \beta(c)$: 由小 MLP 从 $c$ 预测的 scale / shift
- $\odot$: 逐元素乘

### 公式4: [[OrbiSim-Dynamics]] 训练损失

$$
\begin{aligned}
\mathcal{L}_{tra} &= \| \hat{z}_t - \mathrm{sg}(z_t) \|^2 \\
\mathcal{L}_{enc} &= \| z_t - \mathrm{sg}(\hat{z}_t) \|^2 \\
\mathcal{L}_{dec} &= \mathcal{L}_{state}(x_t, \hat{x}_t)
\end{aligned}
$$

**含义**: 三项分别约束 transition 预测对齐真实 encoder 输出、encoder 输出对齐 transition 预测、decoder 重建物理状态。使用 stop-gradient $\mathrm{sg}(\cdot)$ 避免相互坍塌。

**符号说明**:
- $\mathrm{sg}(\cdot)$: stop-gradient 操作
- $\mathcal{L}_{state}$: 物理状态的 L2 或带权重重建损失

### 公式5: [[OrbiSim-Vision]] 条件生成

$$
\hat{o}_t \sim p_\phi^{vis}(o_t \mid o_{t-K:t-1}, \hat{x}_t, \bar{x})
$$

**含义**: 视觉模块在给定预测物理状态 $\hat{x}_t$、静态描述符 $\bar{x}$ 与最近 $K$ 帧观测的条件下采样下一帧 RGB。

**符号说明**:
- $o_{t-K:t-1}$: 上下文帧（可为空）
- $p_\phi^{vis}$: 潜空间扩散过程

### 公式6: [[LDM|潜空间扩散]] 损失

$$
\mathcal{L}_{vis} = \mathbb{E}_{\sigma, \varepsilon}\left[ w(\sigma) \left\| \big( D_\phi(y_{t-K:t,(0)} + \sigma\varepsilon ; c_t) \big)_t - y_{t,(0)} \right\|_2^2 \right]
$$

**含义**: 标准 [[Flow Matching|EDM-style 扩散]] 训练目标，在 VAE latent $y$ 上去噪，由 $c_t = (\hat{x}_t, \bar{x})$ 作为条件。

**符号说明**:
- $\sigma$: 噪声尺度
- $\varepsilon \sim \mathcal{N}(0, I)$
- $w(\sigma)$: 噪声依赖的损失权重
- $D_\phi$: 去噪网络
- $y_{t,(0)}$: 干净 latent

### 公式7: [[解析策略梯度|Analytic Policy Gradient]]

$$
\nabla_\theta J(\theta) = \frac{\partial R}{\partial x_T} \sum_{t=0}^{T-1} \left( \prod_{k=t+1}^{T-1} \frac{\partial x_{k+1}}{\partial x_k} \right) \frac{\partial x_{t+1}}{\partial a_t} \frac{\partial \pi_\theta(x_t)}{\partial \theta}
$$

**含义**: 由于 [[OrbiSim-Dynamics]] 全程可微，奖励对策略参数 $\theta$ 的梯度可以**直接展开**为 Jacobian 链，每步穿过 dynamics 一次。这是 OrbiSim 相对 [[DreamerV3]] / [[PPO]] 的核心优势。

**符号说明**:
- $J(\theta)$: 期望累计奖励
- $R$: 最终奖励
- $T$: rollout 长度
- $\partial x_{k+1}/\partial x_k$: dynamics Jacobian（通过 OrbiSim-Dynamics 反传）
- $\pi_\theta$: 待优化策略

### 公式8: Robosuite Push 奖励函数

$$
\begin{aligned}
r_1 &= 1 - \min_{t=1}^{T} \frac{d_{r, c_1, t}}{d_{r, c_1, 0}} \\
r_2 &= 1 - \min_{t=1}^{T} \frac{d_{c_1, c_2, t}}{d_{c_1, c_2, 0}} \\
r_3 &= 1 - \frac{x_{\text{table\_border}} - x_{c_2, T}}{x_{\text{table\_border}} - x_{c_2, 0}}
\end{aligned}
$$

**含义**: 三阶段稀疏式塑形奖励——$r_1$ 鼓励机械臂 $r$ 接近 cube $c_1$，$r_2$ 鼓励 $c_1$ 推向 $c_2$，$r_3$ 鼓励把 $c_2$ 推下桌沿。这种分段奖励对 [[PPO]] / [[SAC]] 非常困难，但对解析梯度方法友好。

**符号说明**:
- $d_{a,b,t}$: $t$ 时刻 $a$ 与 $b$ 的距离
- $c_1, c_2$: 两个 cube
- $x_{\text{table\_border}}$: 桌沿坐标

---

## 关键图表

### Figure 1: Overview — OrbiSim 整体架构

![Figure 1](https://arxiv.org/html/2605.16395v1/x1.png)

**说明**: 与以往生成式 [[世界模型]] 不同，OrbiSim 被设计为**可微物理引擎**，与现代机器人仿真软件（[[IsaacLab]]、Robosuite）无缝集成，支持资产控制与状态-像素联合生成。左侧展示传统不可微仿真器 + 监督学习的脱节，右侧展示 OrbiSim 把资产配置、神经动力学、RL 三者用解析梯度贯通。

### Figure 2: OrbiSim-Dynamics 结构

![Figure 2](https://arxiv.org/html/2605.16395v1/x2.png)

**说明**: [[OrbiSim-Dynamics]] 是物体中心的循环动力学核心，预测下一步物理状态。基于 [[Transformer]] 的 Coupling Module 捕获多实体相互作用，动力学通过 [[AdaLN]] 接受控制动作与物理属性调制。图中可见 Encoder / Coupling / Recurrent / Transition / Decoder 五模块的数据流。

### Figure 3: 梯度通路

![Figure 3](https://arxiv.org/html/2605.16395v1/x3.png)

**说明**: 在 state-based 任务中，梯度**只**经过 [[OrbiSim-Dynamics]] 而不穿过 [[OrbiSim-Vision]]——这是规避 [[视频扩散模型]] 反传不稳的关键设计。

### Figure 4a: 高摩擦下的自回归仿真

![Figure 4a](https://arxiv.org/html/2605.16395v1/x4.png)

**说明**: 同样初始观测与动作序列下，OrbiSim 比 [[Vid2World]]、[[AdaWorld]] 更准确地仿真高摩擦场景下的物体停滞行为。

### Figure 4b: 低摩擦下的自回归仿真

![Figure 4b](https://arxiv.org/html/2605.16395v1/x5.png)

**说明**: 低摩擦场景下 OrbiSim 正确预测物体滑行更远的轨迹，体现对 $\bar{x}$ 中物理参数的敏感性。

### Figure 5: Isaac Lab Stack 长 horizon 仿真

![Figure 5](https://arxiv.org/html/2605.16395v1/x6.png)

**说明**: 在 [[IsaacLab]] Stack 任务上做 225 步长 horizon 自回归仿真，OrbiSim 准确捕捉机械臂复杂旋转与多物体碰撞，保持高物理保真度与稳定性。

### Figure 6a: AdaManip 铰接物体操作

![Figure 6a](https://arxiv.org/html/2605.16395v1/x7.png)

**说明**: [[AdaManip]] 数据集（9 类铰接物体、210 个实例）下的关节约束运动仿真，验证 OrbiSim 对 articulated dynamics 的处理能力。

### Figure 6b: Physion Drape 布料形变

![Figure 6b](https://arxiv.org/html/2605.16395v1/x8.png)

**说明**: [[Physion]] Drape 场景下的可形变物体仿真，几何条件化的布料形变与刚体接触在统一 asset-conditioned 框架下处理。

### Figure 7: Robosuite Push RL 学习曲线

![Figure 7-1](https://arxiv.org/html/2605.16395v1/x9.png)

![Figure 7-2](https://arxiv.org/html/2605.16395v1/x10.png)

**说明**: Robosuite Push 任务的 RL 奖励曲线，OrbiSim 与 [[SAC]]、[[PPO]]、PPO+RND、Behavior Cloning、[[DreamerV3]] 对比；OrbiSim 凭借 [[解析策略梯度|Analytic Policy Gradient]] 收敛更快、最终性能更高。

### Figure 8: OrbiSim-Vision 去噪过程

![Figure 8](https://arxiv.org/html/2605.16395v1/x11.png)

**说明**: [[OrbiSim-Vision]] 的扩散去噪以预测物理状态为条件，通过空间条件图与物体 token 注入，上下文帧提供补充视觉细节。展示从纯噪声到 RGB 帧的逐步去噪。

### Figure 9: Real-to-Sim 推理流水线

![Figure 9](https://arxiv.org/html/2605.16395v1/x12.png)

**说明**: [[State Inference]] 模块从单帧推断可见物理状态与物体属性，[[Physics Inference]] 模块从 64 帧视频估计隐藏物理参数。两者解耦输出可执行的仿真配置。

### Figure 10: 物理状态轨迹（摩擦敏感性）

![Figure 10](https://arxiv.org/html/2605.16395v1/x13.png)

**说明**: [[OrbiSim-Dynamics]] 的自回归 rollout 展示对物理参数的高敏感性——准确区分低/高摩擦区间，同时保持长 horizon 的平滑。

### Figure 11: 显存占用 vs 物体数

![Figure 11](https://arxiv.org/html/2605.16395v1/fig/orbisim_vram.png)

**说明**: 物体数 scaling 下的显存开销，验证 OrbiSim 的实用性。

### Figure 12: Newton 仿真器轨迹对比

![Figure 12](https://arxiv.org/html/2605.16395v1/x14.png)

**说明**: 在 NVIDIA Newton 可微仿真器中 rollout，比较用 Newton-warp 与 OrbiSim-Dynamics 优化得到的初始速度产生的轨迹，验证 OrbiSim 梯度的物理合理性。

### Figure 13: Newton 仿真器视觉 rollout

![Figure 13](https://arxiv.org/html/2605.16395v1/x15.png)

**说明**: 两行分别为 Newton-warp 与 OrbiSim-Dynamics 优化初始速度的视觉 rollout 对比。

### Figure 14: Robosuite Push 训练曲线（四子图）

![Figure 14a](https://arxiv.org/html/2605.16395v1/x16.png)

![Figure 14b](https://arxiv.org/html/2605.16395v1/x17.png)

![Figure 14c](https://arxiv.org/html/2605.16395v1/x18.png)

![Figure 14d](https://arxiv.org/html/2605.16395v1/x19.png)

**说明**: Robosuite Push 任务训练曲线，横轴为训练 episode，纵轴为归一化 episode 奖励，覆盖不同种子/配置下的收敛对比。

### Table 1: 视频级性能 (Robosuite Push)

| Method | PSNR₁₀ ↑ | PSNR₁₀₀ ↑ | LPIPS₁₀ ↓ | LPIPS₁₀₀ ↓ | FVD ↓ | TrajErr ↓ |
|--------|---------|----------|----------|-----------|------|----------|
| [[Vid2World]] | 22.20 | 17.89 | 0.1312 | 0.2551 | 1750.1 | 0.6754 |
| [[AdaWorld]] | 26.66 | 12.83 | 0.1183 | 0.3482 | 1305.8 | 1.8597 |
| **OrbiSim (Final)** | **26.71** | **19.98** | **0.1078** | **0.1428** | **533.9** | **0.4468** |

**说明**: OrbiSim 在所有视频质量指标上全面超越 [[Vid2World]] 与 [[AdaWorld]]，长 horizon (100 步) 优势更明显——LPIPS₁₀₀ 0.1428 vs 0.2551 / 0.3482，FVD 从 1300+ 降到 533.9。轨迹误差也降到约 0.45。

### Table 2: 分布外鲁棒性 (OOD)

| Setting | Method | PSNR₁₀ ↑ | PSNR₁₀₀ ↑ | LPIPS₁₀₀ ↓ | FVD ↓ |
|---------|--------|---------|----------|-----------|------|
| In-Distribution | OrbiSim | 26.71 | 19.98 | 0.1428 | 533.9 |
| Out-of-Distribution | OrbiSim | 27.19 | 20.11 | 0.1847 | 597.3 |
| OOD | [[AdaWorld]] | 26.35 | 13.22 | 0.3836 | 1288.5 |

**说明**: OrbiSim 在 OOD 设定下指标几乎不退化（PSNR 甚至略升），而 [[AdaWorld]] 在 OOD 下大幅崩坏；体现物理状态显式建模带来的泛化优势。

### Table 3: Real-to-Sim 推理精度

| 模块 | 物体位置 ↓ | 旋转 ↓ | 尺寸 ↓ | 机器人位置 ↓ | 速度 ↓ |
|------|-----------|-------|-------|--------------|-------|
| StaInf | 27.09 mm | 7.37° | 5.2% | 22.44 mm | — |
| PhyInf | 22.40 mm | 4.28° | — | 12.70 mm | 28.63 mm/s |

**说明**: [[State Inference]] 仅用单帧 RGB 即可达到 27 mm 物体位置误差；[[Physics Inference]] 用 64 帧视频窗口进一步把误差压到 22 mm 并恢复速度信息。

### Table 4: RL 成功率 (Robosuite Push, %)

| Method | OrbiSim | [[DreamerV3]] | BC | PPO+RND | [[PPO]] | [[SAC]] |
|--------|---------|---------------|-----|---------|---------|---------|
| Success | **42.71** | 25.00 | 19.79 | 2.08 | 1.04 | 0.00 |

**说明**: OrbiSim 凭借解析策略梯度达到 42.71% 成功率，远超 [[DreamerV3]] 的 25.00%；纯无模型 RL（PPO/SAC）在稀疏奖励下基本失败。

### Table 5: 多物体/多任务 scaling

| Scenario | Method | PSNR₁₀ ↑ | PSNR₁₀₀ ↑ | LPIPS₁₀₀ ↓ | FVD ↓ |
|----------|--------|---------|----------|-----------|------|
| Single (≤2 objects) | OrbiSim | 26.71 | 19.98 | 0.1428 | 533.9 |
| Multi (≤4 objects) | OrbiSim | 24.87 | 18.45 | 0.1959 | 863.1 |

**说明**: 物体数从 ≤2 增到 ≤4 时性能温和下降但仍优于 [[Vid2World]] / [[AdaWorld]] 在单物体上的表现。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| Robosuite Push | 2,600 episodes / ~50 步 | 2 物体，随机化摩擦/密度/尺寸 | 主基准 |
| [[IsaacLab]] Stack | ~2,200 episodes / ~230 步 | Franka 三块码垛，多阶段长 horizon | 长 horizon 仿真 |
| [[AdaManip]] | 4,200 episodes / 301 步 | 9 类铰接物体，210 实例 | 铰接物体 |
| [[Physion]] Drape | 数据集自带 | 布料 + 刚体形变 | 可形变物体 |

### 实现细节

- **Backbone**: Transformer-based RSSM + Latent Diffusion (UNet)
- **状态表示**: 机器人 (位置 + 线速度 + 旋转矩阵) + 物体 (位置 + 线速度 + 四元数) + 静态资产 (几何/质量/惯量/摩擦/shape encoding)
- **视觉条件**: 空间条件图（heatmap/depth-like prior）+ 物体 token (cross-attn) + 视觉锚定（$K \in \{0,\ldots,6\}$ 帧扰动 latent）+ 物理 grounding
- **训练**: 三项 dynamics loss（transition / encoding / decoding）+ EDM-style 扩散 loss
- **RL 策略优化**: 解析策略梯度，反传穿过冻结/在线的 OrbiSim-Dynamics

### 可视化结果

- 高/低摩擦下的物体滑行轨迹差异被 OrbiSim 正确捕捉（Fig.4 / Fig.10）
- 长 horizon (225 步) Stack 任务中机械臂旋转与碰撞保持稳定（Fig.5）
- 铰接物体与布料用统一框架处理（Fig.6）

---

## 批判性思考

### 优点

1. **范式创新**: 把世界模型重新定义为"可微物理引擎"，给具身智能社区提供了不同于 [[DreamerV3]] latent imagination 路线的全新框架。
2. **梯度通路设计精巧**: 状态-像素解耦让梯度只穿 dynamics、不穿扩散链，规避 monolithic 视频生成模型的训练不稳定。
3. **稀疏奖励 RL 显著提升**: 42.71% vs DreamerV3 25.00%，证明 [[解析策略梯度|Analytic Policy Gradient]] 在物理可微环境下的样本效率优势。
4. **Real-to-Sim 接口实用**: StaInf/PhyInf 为真实视频 → 仿真配置提供了结构化通道，对 sim-to-real loop 闭环有价值。
5. **OOD 泛化优势**: 显式物理状态让 OOD 下指标不退化，反观纯像素方法 [[AdaWorld]] 在 OOD 下 LPIPS 从 0.34 飙到 0.38、FVD 翻倍。

### 局限性

1. **覆盖形态受限**: 作者自述"距离通用工业仿真器仍很远"，目前主要在 Robosuite / IsaacLab / AdaManip / Physion 等少数仿真器家族内验证，未涉及软体、流体、复杂接触组合。
2. **依赖资产描述符 $\bar{x}$**: 必须事先知道几何/质量/摩擦等，对真实未知场景需要 PhyInf 推断且误差累积。
3. **视觉模块仍重**: OrbiSim-Vision 是 latent diffusion，推理延迟较高；虽然策略梯度不经过它，但 closed-loop 视觉控制（基于 RGB 观测）仍需要它跑前向。
4. **多物体 scaling 退化**: 从 ≤2 物体到 ≤4 物体 FVD 从 534 → 863，未来扩展到密集场景（10+ 物体）会面临 Transformer self-attention 的复杂度问题。
5. **Real-to-Sim 精度**: 物体位置 27 mm、机器人位置 22 mm 的误差对精细操作（如插孔）仍偏大。

### 潜在改进方向

1. 引入 [[3DGS|3D Gaussian Splatting]] 作为视觉端，把 Vision 模块改成可微渲染，避免 latent diffusion 的高延迟。
2. 用 Hierarchical / Sparse Transformer 处理 10+ 物体密集场景，缓解 self-attention 二次复杂度。
3. 把 PhyInf 扩展到在线 system identification，配合 MPC 做 adaptive control。
4. 与 LLM/VLM 结合，用语言指令条件化 $\bar{x}$ 的生成，实现 text-to-sim。

### 可复现性评估

- [ ] 代码开源（项目页注明 "Code coming soon"）
- [ ] 预训练模型
- [ ] 训练细节完整（论文中给出数据规模和模块结构，超参数详见附录）
- [x] 数据集可获取（Robosuite / IsaacLab / AdaManip / Physion 均公开）

---

## 关联笔记

### 基于

- [[DreamerV3]]: 提供 RSSM 与 model-based RL 的基础范式
- [[Vid2World]]: 前置的视频世界模型，是主要对比基线
- [[AdaWorld]]: 自适应世界模型，是主要对比基线
- [[LDM]]: OrbiSim-Vision 的渲染骨干
- [[Transformer]]: Coupling Module 的核心架构

### 对比

- [[DreamerV3]]: latent imagination 路线，没有显式物理状态，OrbiSim 用解析梯度超越其 RL 性能
- [[Vid2World]] / [[AdaWorld]]: 纯像素生成世界模型，OrbiSim 在 OOD 与长 horizon 下显著更鲁棒
- [[MuJoCo]] / [[IsaacLab]]: 经典不可微物理引擎，OrbiSim 提供可微替代

### 方法相关

- [[世界模型]]: 核心范式
- [[OrbiSim-Dynamics]]: 状态预测主模块
- [[OrbiSim-Vision]]: 视觉渲染主模块
- [[解析策略梯度|Analytic Policy Gradient]]: 关键 RL 技术
- [[AdaLN]]: 条件调制机制
- [[State Inference]] / [[Physics Inference]]: Real-to-Sim 模块
- [[循环状态空间模型|RSSM]]: 时序记忆架构

### 硬件/数据相关

- [[IsaacLab]]: Stack 任务平台
- [[AdaManip]]: 铰接物体数据集
- [[Physion]]: 可形变物体数据集

---

## 速查卡片

> [!summary] OrbiSim: World Models as Differentiable Physics Engines
> - **核心**: 状态-像素双流可微仿真，让策略梯度直接穿过物理引擎
> - **方法**: OrbiSim-Dynamics (Transformer RSSM + AdaLN) + OrbiSim-Vision (状态条件 Latent Diffusion) + Real-to-Sim 推理 (StaInf + PhyInf)
> - **结果**: Robosuite Push RL 成功率 42.71% (vs DreamerV3 25%)；长 horizon FVD 534 (vs Vid2World 1750)；OOD 几乎不退化
> - **代码**: Coming soon @ https://jjleejj85.github.io/projects/orbisim

---

*笔记创建时间: 2026-05-19*
