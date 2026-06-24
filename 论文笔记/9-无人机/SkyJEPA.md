---
title: "SkyJEPA: Learning Long-Horizon World Models for Zero-Shot Sim-to-Real Control of Quadrotors"
method_name: "SkyJEPA"
authors: [Pratyaksh Rao, Wancong Zhang, Randall Balestriero, Yann LeCun, Giuseppe Loianno]
year: 2026
venue: arXiv
tags: [world-model, quadrotor, jepa, sim-to-real, mppi, latent-dynamics, zero-shot]
zotero_collection: 9-无人机
image_source: online
arxiv_html: https://arxiv.org/html/2606.23444
created: 2026-06-24
---

# 论文笔记：SkyJEPA: Learning Long-Horizon World Models for Zero-Shot Sim-to-Real Control of Quadrotors

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | NYU Tandon School of Engineering; Meta FAIR |
| 日期 | June 2026 |
| 项目主页 | N/A |
| 对比基线 | [[MPPI]] (Predictive baseline) |
| 链接 | [arXiv](https://arxiv.org/abs/2606.23444) |

---

## 一句话总结

> SkyJEPA 将 [[JEPA]] 风格的潜在动力学模型与物理启发探针结合，实现四旋翼飞行器的长时域预测与零样本 Sim-to-Real 控制，避免自回归误差累积。

---

## 核心贡献

1. **JEPA 风格潜在动力学模型**: 在潜在空间而非状态空间建模动力学，配合 [[SIGReg]] 反坍缩正则化，大幅抑制长时域递归预测的误差累积
2. **物理启发探针（Physics-Inspired Prober）**: 冻结潜在动力学后，训练解码器将抽象嵌入映射到可解释物理状态，利用 [[SO(3)]] 指数映射保证姿态几何一致性
3. **零样本 Sim-to-Real 迁移流水线**: 通过域随机化（[[Domain Randomization]]）自动生成仿真数据集并结合 [[MPPI]] 采样控制，无需真实数据微调即可在真实飞行中部署

---

## 问题背景

### 要解决的问题

四旋翼飞行器的高频（100 Hz）实时控制要求世界模型能够准确进行**长时域预测**（数十步递归展开），同时需零样本泛化到多种任务和平台变体。

### 现有方法的局限

传统自回归状态空间预测（Predictive baseline）将单步误差 $\varepsilon$ 在递归展开中反复叠加：

$$
\tilde{\mathbf{x}}_{t+T} = h_\theta(\ldots h_\theta(h_\theta(\mathbf{x}_t, \mathbf{a}_t), \mathbf{a}_{t+1})\ldots, \mathbf{a}_{t+T})
$$

随着展开步数 $T$ 增大，误差以超线性速率放大（Compounding Ratio 在 $k=60$ 步时达到 ~2.4×），导致控制性能急剧下降。

### 本文的动机

[[JEPA]]（Joint Embedding Predictive Architecture）在视觉领域证明了潜在空间预测能学到更直线化（temporally straight）的轨迹表示，使递归展开更稳定。作者将此思路移植到四旋翼动力学建模，并引入物理约束使潜在预测可解码为真实物理状态，同时保持嵌入向量几何结构的合理性。

---

## 方法详解

### 模型架构

SkyJEPA 采用**潜在动力学 + 物理探针**两阶段架构：

- **输入**: 状态历史 $\mathbf{X}_t = [\mathbf{x}_{t-H}^\top \ldots \mathbf{x}_t^\top]^\top$，动作历史 $\mathbf{A}_t = [\mathbf{a}_{t-H}^\top \ldots \mathbf{a}_t^\top]^\top$
- **状态编码器** $\mathrm{Enc}_\theta$: [[TCN]]（时序卷积网络），通道 $[8, 8, 16]$，将状态历史压缩为潜在向量 $\mathbf{s}_t$
- **动作编码器** $\mathrm{Enc}_\phi$: [[TCN]]，通道 $[4, 4, 8]$，将动作历史编码为 $\mathbf{z}_t$
- **预测器** $\mathrm{Pred}_\phi$: 单层 [[GRU]]，隐藏维度 24，递归展开 $T=20$ 步（1.0 s at 20 Hz）
- **物理探针** $\psi$: 冻结以上模块后训练，输出残差修正量映射到可解释物理量
- **总参数**: ~99K，通过 NVIDIA TensorRT 部署于 Jetson Orin NX，推理时延 <10 ms

### 核心模块

#### 模块 1: JEPA 风格潜在动力学（Stage 1 训练）

**设计动机**: 在潜在空间建模动力学可让表示空间中的轨迹更线性（[[Temporal Straightening]]），减少误差在递归展开时的非线性放大。

**具体实现**:
- 使用 $\mathrm{Enc}_\theta$ 将状态历史编码为 $\mathbf{s}_t \in \mathbb{R}^{D_s}$
- 使用 $\mathrm{Enc}_\phi$ 将动作历史编码为 $\mathbf{z}_t \in \mathbb{R}^{D_z}$
- 通过 [[GRU]] 预测器递归展开：$\tilde{\mathbf{s}}_{t+k} = \mathrm{Pred}_\phi(\tilde{\mathbf{s}}_{t+k-1}, \mathbf{z}_{t+k-1})$
- 以目标编码器输出 $\mathbf{s}_{t+k}$ 为监督信号，计算多步潜在预测损失 $\mathcal{L}_\mathrm{pred}$
- 同时施加 [[SIGReg]]（Statistical Isotropy Gaussian Regularizer）防止表示坍缩

#### 模块 2: 物理启发探针（Stage 2 训练，冻结 Stage 1）

**设计动机**: 纯潜在预测缺乏可解释性且难以直接用于控制；通过物理约束将潜在向量解码为状态，同时利用 [[SO(3)]] 指数映射保持旋转矩阵的流形结构。

**具体实现**:
- 探针 $\psi$ 输出残差修正量：$\{\Delta\dot{\mathbf{v}}_{t+k}, \mathbf{K}_{t+k}\} = \psi(\tilde{\mathbf{s}}_{t+k})$
- 利用牛顿力学计算平动加速度：$\dot{\mathbf{v}}_t = \left(\sum f_{i,t}/m\right)\mathbf{R}_t\mathbf{e}_3 - \mathbf{g} + \Delta\dot{\mathbf{v}}_t$
- 力矩残差：$\Delta\boldsymbol{\tau}_t = \mathbf{K}_t \mathbf{a}_t$
- 通过物理积分传播状态，用 $\mathrm{exp}([\boldsymbol{\omega}_t]_\times \Delta t)$ 更新旋转矩阵

---

## 关键公式

### 公式 1: [[JEPA|多步潜在预测损失]]

$$
\mathcal{L}_\mathrm{pred} = \frac{1}{T} \sum_{k=1}^{T} \|\tilde{\mathbf{s}}_{t+k} - \mathbf{s}_{t+k}\|_2^2
$$

**含义**: 监督预测器在潜在空间的多步递归展开精度，以目标编码器的真实潜在向量为标签。

**符号说明**:
- $\tilde{\mathbf{s}}_{t+k}$: 预测器递归展开到 $k$ 步的潜在向量
- $\mathbf{s}_{t+k}$: 目标编码器对真实状态的编码
- $T$: 展开步数（本文 $T=20$）

---

### 公式 2: [[SIGReg|统计各向同性高斯正则化]]

**步骤**：在超球面 $\mathcal{S}^{D-1}$ 上采样 $M$ 个随机单位向量 $\{\boldsymbol{\xi}_m\}$，沿每个方向投影潜在批次：

$$
\mathbf{h}^{(m)} = \mathbf{S}\boldsymbol{\xi}_m \in \mathbb{R}^{T \times B}
$$

计算经验特征函数（与标准高斯特征函数比较）并积分得到 Epps-Pulley 统计量 $T^{(m)}$：

$$
\mathcal{L}_\mathrm{SIGReg} = \frac{1}{M} \sum_{m=1}^{M} T^{(m)}
$$

**含义**: 鼓励潜在嵌入在各方向上近似高斯分布，防止维度坍缩（collapse）。

**符号说明**:
- $M$: 随机投影方向数（本文 17）
- $B$: batch 内样本数
- $T^{(m)}$: 第 $m$ 方向投影与高斯分布的偏差统计量

---

### 公式 3: [[总训练损失]]

$$
\mathcal{L}_\mathrm{total} = \mathcal{L}_\mathrm{pred} + \lambda_\mathrm{sig} \cdot \mathcal{L}_\mathrm{SIGReg}
$$

**含义**: 第一阶段联合优化预测精度与表示各向同性。

**符号说明**:
- $\lambda_\mathrm{sig} = 0.02$: SIGReg 权重

---

### 公式 4: [[SO(3)|SO(3) 指数映射状态传播]]

$$
\mathbf{p}_{t+1} = \mathbf{p}_t + \mathbf{v}_t \Delta t
$$

$$
\mathbf{v}_{t+1} = \mathbf{v}_t + \dot{\mathbf{v}}_t \Delta t
$$

$$
\mathbf{R}_{t+1} = \mathbf{R}_t \exp([\boldsymbol{\omega}_t]_\times \Delta t)
$$

$$
\boldsymbol{\omega}_{t+1} = \boldsymbol{\omega}_t + \Delta\boldsymbol{\tau}_t \Delta t
$$

**含义**: 利用 SO(3) 流形上的指数映射传播旋转矩阵，确保姿态估计始终满足正交约束，避免欧拉角万向锁问题。

**符号说明**:
- $\mathbf{p}_t$: 位置
- $\mathbf{v}_t$: 速度
- $\mathbf{R}_t$: 旋转矩阵（$\in$ SO(3)）
- $\boldsymbol{\omega}_t$: 角速度
- $[\cdot]_\times$: 反对称矩阵（向量叉积算子）
- $\Delta\boldsymbol{\tau}_t$: 力矩残差修正（由探针 $\psi$ 提供）

---

### 公式 5: [[物理探针损失]]

$$
\mathcal{L}_\mathrm{prober} = \frac{1}{T} \sum_{k=1}^{T} \|\tilde{\mathbf{x}}_{t+k} - \mathbf{x}_{t+k}\|_2^2
$$

**含义**: 监督探针将冻结潜在预测解码为物理状态的精度，以真实状态轨迹为标签。

**符号说明**:
- $\tilde{\mathbf{x}}_{t+k}$: 探针积分后的预测物理状态
- $\mathbf{x}_{t+k}$: 真实物理状态

---

### 公式 6: [[MPPI|MPPI 轨迹代价函数]]

$$
J^{(s)} = \frac{1}{T} \sum_{k=1}^{T} \ell(\tilde{\mathbf{x}}^{(s)}_{t+k}, \mathbf{x}^\mathrm{ref}_{t+k}, \mathbf{a}^{(s)}_{t+k-1})
$$

$$
\ell(\cdot) = \|\tilde{\mathbf{x}}^{(s)} - \mathbf{x}^\mathrm{ref}\|^2_{Q_x} + \|\mathbf{a}^{(s)} - \mathbf{a}^\mathrm{ref}\|^2_{Q_a}
$$

**含义**: 对每条采样轨迹计算跟踪误差与控制代价的加权和，用于 MPPI 软最优控制。

**符号说明**:
- $Q_x = \mathrm{diag}(400, 40, 20, 20)$: 状态跟踪权重
- $Q_a = \mathrm{diag}(0.01, 0.05, 0.05, 0.10)$: 控制代价权重
- $\mathbf{x}^\mathrm{ref}$: 参考轨迹状态

---

### 公式 7: [[MPPI|MPPI 软最优权重]]

$$
w^{(s)} = \frac{\exp\!\left(-\frac{1}{\lambda}(J^{(s)} - J_\min)\right)}{\sum_r \exp\!\left(-\frac{1}{\lambda}(J^{(r)} - J_\min)\right)}
$$

**含义**: 以指数 softmax 赋予低代价轨迹更高权重，温度参数 $\lambda$ 控制权重集中程度。

**符号说明**:
- $\lambda = 10^{-4}$: 温度参数（越小越接近 argmin）
- $J_\min$: 当前采样批次中最小代价（数值稳定性）

---

### 公式 8: [[Temporal Straightening|时序直线化指标]]

$$
S_\mathrm{straight}^{(i)} = \frac{1}{T-2} \sum_{t=1}^{T-2} \frac{\langle \dot{\tilde{\mathbf{s}}}^{(i)}_t, \dot{\tilde{\mathbf{s}}}^{(i)}_{t+1} \rangle}{\|\dot{\tilde{\mathbf{s}}}^{(i)}_t\| \|\dot{\tilde{\mathbf{s}}}^{(i)}_{t+1}\|}
$$

**含义**: 衡量潜在轨迹的"直线度"——相邻步潜在速度方向的余弦相似性均值，越接近 1 表示轨迹越平滑线性。

**符号说明**:
- $\dot{\tilde{\mathbf{s}}}^{(i)}_t = \tilde{\mathbf{s}}^{(i)}_{t+1} - \tilde{\mathbf{s}}^{(i)}_t$: 第 $i$ 条轨迹在时刻 $t$ 的潜在速度

---

## 关键图表

### Figure 1: 四旋翼世界模型的四大期望属性

![Figure 1](https://arxiv.org/html/2606.23444v2/x1.png)

**说明**: 论文提出好的四旋翼 [[World Model]] 应同时满足四个属性：(1) 准确的长时域预测，(2) 物理可解释性，(3) 嵌入式实时推理能力，(4) 零样本任务泛化。SkyJEPA 是首个同时满足这四个属性的方法。

---

### Figure 2: SkyJEPA 整体框架概览

![Figure 2](https://arxiv.org/html/2606.23444v2/x2.png)

**说明**: 整体框架由 [[JEPA]] 风格潜在动力学模型与物理启发探针 $\psi$ 两部分组成。状态历史经 $\mathrm{Enc}_\theta$ 压缩为潜在向量，预测器递归展开后由探针解码为可解释的物理状态，最终接入 [[MPPI]] 采样控制器实现闭环飞行。

---

### Figure 3: 两阶段训练流水线

![Figure 3](https://arxiv.org/html/2606.23444v2/x3.png)

**说明**: **Stage 1** 训练编码器与预测器，以多步潜在预测损失 $\mathcal{L}_\mathrm{pred}$ 和 [[SIGReg]] 正则化为目标；**Stage 2** 冻结 Stage 1 参数，仅训练探针 $\psi$，监督信号为真实物理状态（含 [[SO(3)]] 积分约束）。

---

### Figure 4: 真实世界闭环评估三种场景

![Figure 4](https://arxiv.org/html/2606.23444v2/x4.png)

**说明**: 实验在 60×70 m² 户外场地进行，三种测试场景：(a) 标准轨迹跟踪（nominal），(b) 挂载 300 g 负载，(c) 螺旋桨切换（propeller switching）。后两种场景在不重新训练的情况下测试泛化能力。

---

### Figure 5: NVIDIA Orin NX 上的推理速度

![Figure 5](https://arxiv.org/html/2606.23444v2/x5.png)

**说明**: 推理时间随展开步数 $T$ 和 MPPI 采样数 $S$ 增加而增长。SkyJEPA（$T=15$, $S=512$）推理时延 <10 ms，满足 100 Hz 控制的实时要求。

---

### Figure 6: 递归展开误差分析（Compounding Ratio）

![Figure 6](https://arxiv.org/html/2606.23444v2/x6.png)

**说明**: 对比递归展开与 teacher-forced 预测的误差比（Compounding Ratio CR$_k$）。Predictive baseline 在 $k\approx12$ 步时 CR$_k$ 超过 1 并持续攀升至 ~2.4（$k=60$），而 SkyJEPA 保持在 ~1.4，误差增长率（Error Rate ER$_k$）在 $k=60$ 时为 0.11 对比 baseline 的 0.23。

---

### Figure 7: 潜在轨迹时序直线化分析

![Figure 7](https://arxiv.org/html/2606.23444v2/x7.png)

**说明**: [[Temporal Straightening]] 指标分布对比。Predictive baseline 均值约 -0.4（振荡），Reconstruction 模型约 0.95（最直），SkyJEPA（JEPA）约 0.75——表示潜在空间轨迹更平滑，有利于递归展开稳定性。

---

### Figure 8: 开环展开各模型对比（位置/速度/姿态 RMSE）

![Figure 8](https://arxiv.org/html/2606.23444v2/x8.png)

**说明**: 六种方法在开环状态预测上的完整对比，展示位置、速度和姿态随展开步数的 RMSE 演变。SkyJEPA + PI Prober 在所有指标上均优于其余五种 baseline。

---

### Figure 9: 观测噪声鲁棒性

![Figure 9](https://arxiv.org/html/2606.23444v2/x9.png)

**说明**: 在输入加入不同强度高斯噪声时的位姿 RMSE。零噪声时 SkyJEPA 较 baseline 降低约 55% RMSE；中等噪声时保持 25–30% 优势；高噪声时仍有约 10% 改善，说明潜在空间建模具有天然去噪效果。

---

### Figure 10: 真实世界零样本轨迹跟踪（误差色彩可视化）

![Figure 10](https://arxiv.org/html/2606.23444v2/x10.png)

**说明**: 五种参考轨迹（圆形、椭圆、8字形、鱼形、双纽线）在户外场地的实际飞行结果，轨迹用颜色编码跟踪误差。SkyJEPA 在最高速度 7.2 m/s、最大加速度 12.5 m/s² 的双纽线轨迹上位置 RMSE 仅 0.45 m，对比 baseline 的 0.61 m。

---

### Figure 11: 平台变化下的鲁棒性（无需重新训练）

![Figure 11](https://arxiv.org/html/2606.23444v2/x11.png)

**说明**: 挂载 300 g 负载和螺旋桨切换两种非标准场景下的跟踪性能。SkyJEPA 相比 Predictive baseline 在两种扰动下均保持 ~1.3–1.4× 的位置精度优势，验证了 [[Domain Randomization]] 的泛化覆盖能力。

---

### Table I: 域随机化参数范围

| 参数 | 随机化范围 |
|------|----------|
| 质量 $m$ | ±50% 标称值 |
| 转动惯量 $J$ | ±30% 标称值 |
| 电机时间常数 $\alpha$ | [0.01, 0.1] s |
| 阻力系数 $D$ | [0.1, 0.5] |
| 推力系数 $k_f$ | ±50% 标称值 |
| 力矩系数 $k_m$ | ±50% 标称值 |
| 随机化域数量 | 500 |
| 总轨迹数 | 20,000 |

**说明**: 500 个随机化域覆盖宽范围物理参数，确保模型学到对平台参数鲁棒的动力学特征，支撑零样本 Sim-to-Real 迁移。

---

### Table II: MPPI 控制器参数

| 参数 | 数值 |
|------|------|
| 预测时间步 $\Delta t$ | 0.05 s |
| 预测时域 $T$ | 15 |
| 采样数 $S$ | 512 |
| 温度 $\lambda$ | $10^{-4}$ |
| 动作噪声 $\Sigma$ | diag(0.60, 0.15, 0.15, 0.05) |
| 状态代价 $Q_x$ | diag(400, 40, 20, 20) |
| 控制代价 $Q_a$ | diag(0.01, 0.05, 0.05, 0.10) |

---

### Table III: 开环预测性能对比（6种方法）

| 方法 | 位置 RMSE [m] | 姿态误差 [°] |
|------|--------------|------------|
| Predictive | 8.80±2.3 | 53.4±14.9 |
| Predictive + Physics Reg | 7.12±2.1 | 49.1±13.2 |
| Reconstruction + Prober | 6.82±1.6 | 45.2±9.7 |
| Reconstruction + PI Prober | 1.53±0.13 | 5.28±0.70 |
| Ours + Prober | 5.56±1.31 | 40.20±9.30 |
| **Ours + PI Prober** | **1.43±0.10** | **4.71±0.50** |

**关键发现**: 物理启发探针（PI Prober）是决定性组件——对 JEPA 模型带来 3.9× 位置精度和 8.5× 姿态精度提升；JEPA 潜在动力学进一步优于重建（Reconstruction）风格的潜在模型。

---

### Table IV: 零样本 Sim-to-Real 闭环轨迹跟踪（位置 RMSE）

| 轨迹 | 最大速度 | 最大加速度 | SkyJEPA 位置 RMSE | Predictive baseline |
|------|---------|-----------|------------------|-------------------|
| 圆形 | 2.45 m/s | 1.40 m/s² | 0.24 m | 0.39 m |
| 椭圆 | 4.50 m/s | 3.78 m/s² | 0.33 m | 0.48 m |
| 8字形 | 5.20 m/s | 5.44 m/s² | 0.35 m | 0.51 m |
| 鱼形 | 5.70 m/s | 7.68 m/s² | 0.40 m | 0.59 m |
| 双纽线 | 7.20 m/s | 12.5 m/s² | 0.45 m | 0.61 m |

**关键发现**: SkyJEPA 在所有轨迹上均取得约 26–38% 的位置精度提升，在更高动态轨迹（鱼形、双纽线）上优势尤为明显。

---

### Table V: 非标准平台场景下的跟踪性能

| 场景 | 轨迹 | SkyJEPA 位置 | Predictive | 改善倍数 |
|------|------|-------------|-----------|--------|
| 螺旋桨切换 | 圆形 | 0.33 m | 0.45 m | 1.3× |
| 螺旋桨切换 | 8字形 | 0.35 m | 0.47 m | 1.3× |
| 螺旋桨切换 | 鱼形 | 0.39 m | 0.53 m | 1.4× |
| 300g 负载 | 圆形 | 0.46 m | 0.62 m | 1.3× |
| 300g 负载 | 8字形 | 0.49 m | 0.66 m | 1.3× |
| 300g 负载 | 鱼形 | 0.53 m | 0.72 m | 1.4× |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 仿真生成轨迹 | 20,000 条 × 10 s | 500 域随机化，[[Gaussian Process]] 生成参考轨迹，NMPC/MPPI 闭环采集 | 训练（80%）/验证（10%）/测试（10%） |

### 实现细节

- **编码器**: TCN（状态 $[8,8,16]$，动作 $[4,4,8]$），历史长度 $H=10$ 步（0.5 s）
- **预测器**: 单层 [[GRU]]，隐藏维度 24，展开步数 $T=20$
- **优化器**: Adam，权重衰减 $10^{-5}$，梯度裁剪 0.5
- **学习率调度**: 线性预热 $0 \to 5\times10^{-3}$（4,000 步），余弦衰减至 $10^{-4}$（20,000 步）
- **训练轮数**: 50 epochs，batch size 2048
- **硬件平台**: 1.3 kg 四旋翼，推重比 4:1，NVIDIA Orin NX（机载），PX4 飞控

### 可视化结果

Figure 10 展示五种参考轨迹的实际飞行结果，颜色编码跟踪误差从小（蓝）到大（红）。SkyJEPA 的误差分布更均匀，在轨迹高曲率区域（转弯处）的红色区域明显小于 baseline，说明潜在动力学在快速姿态变化时预测更准确。

---

## 批判性思考

### 优点

1. **系统性解决误差累积**: SIGReg 正则化 + 潜在空间预测 + 物理探针三者协同，从表示学习、训练目标、解码三个层面共同抑制长时域预测误差
2. **物理可解释 + 几何一致**: SO(3) 指数映射确保旋转矩阵始终满足流形约束，避免欧拉角万向锁，且探针输出有明确物理含义（加速度残差、力矩矩阵）
3. **计算效率优异**: ~99K 参数，TensorRT 部署后单步推理 <10 ms，满足 100 Hz 嵌入式实时控制需求

### 局限性

1. **数据来源单一**: 全部训练数据来自仿真，依赖域随机化的覆盖范围；对仿真中未建模的气动效应（如地效、侧风扰动）鲁棒性未充分验证
2. **架构规模受限**: GRU 隐藏维度仅 24、~99K 参数，表达能力有限，对高度非线性或欠驱动系统（如倒置飞行）能否泛化存疑
3. **两阶段训练耦合**: 探针质量上限受限于 Stage 1 的潜在表示质量，若 Stage 1 未充分收敛则 Stage 2 无法补救

### 潜在改进方向

1. 引入真实飞行少量数据微调（few-shot adaptation）以弥合仿真中未建模的气动残差
2. 将 JEPA 框架扩展至多智能体/多机编队控制场景

### 可复现性评估

- [ ] 代码开源（论文未提供）
- [ ] 预训练模型（未提供）
- [x] 训练细节完整（超参、架构、数据生成流程均详细描述）
- [ ] 数据集可获取（仿真生成，需自行复现）

---

## 关联笔记

### 基于

- [[JEPA]]: Joint Embedding Predictive Architecture，本文潜在动力学建模的核心范式
- [[MPPI]]: Model Predictive Path Integral，采样控制框架
- [[Domain Randomization]]: 零样本 Sim-to-Real 的数据增强策略

### 对比

- Predictive baseline（自回归状态空间预测）: 直接在状态空间递归预测，误差累积显著
- Reconstruction + PI Prober: 用重建目标替代 JEPA 目标的消融对比

### 方法相关

- [[SIGReg]]: 统计各向同性高斯正则化，防止表示坍缩
- [[SO(3)]]: 旋转群，姿态状态传播的数学基础
- [[Temporal Straightening]]: 衡量潜在轨迹线性度的核心指标
- [[TCN]]: 时序卷积网络，编码器骨干
- [[GRU]]: 门控循环单元，预测器骨干
- [[Gaussian Process]]: 用于生成多样化参考轨迹
- [[World Model]]: 本文的核心研究对象

### 硬件/数据相关

- [[PX4]]: 开源飞控系统
- [[NVIDIA Jetson Orin NX]]: 机载推理平台

---

## 速查卡片

> [!summary] SkyJEPA (arXiv 2026)
> - **核心**: JEPA 风格潜在动力学 + 物理启发探针，解决四旋翼长时域预测误差累积
> - **方法**: Stage 1 训练潜在预测（SIGReg 防坍缩）；Stage 2 冻结后训练 SO(3) 物理探针；MPPI 采样控制
> - **结果**: 开环位置 RMSE 1.43 m（对比 baseline 8.80 m）；零样本实飞位置 RMSE 0.24–0.45 m，改善 26–38%
> - **代码**: 未开源

---

*笔记创建时间: 2026-06-24*
