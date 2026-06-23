---
title: "SkyJEPA: Learning Long-Horizon World Models for Zero-Shot Sim-to-Real Control of Quadrotors"
method_name: "SkyJEPA"
authors: [Pratyaksh Rao, Wancong Zhang, Randall Balestriero, Yann LeCun, Giuseppe Loianno]
year: 2026
venue: arXiv
tags: [world-model, sim-to-real, quadrotor, jepa, mppi, latent-dynamics]
zotero_collection: 9-无人机
image_source: pending
arxiv_html: https://arxiv.org/html/2606.23444
created: 2026-06-23
---

# 论文笔记：SkyJEPA: Learning Long-Horizon World Models for Zero-Shot Sim-to-Real Control of Quadrotors

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | NYU (Pratyaksh Rao, Giuseppe Loianno), Meta AI / NYU (Randall Balestriero, Yann LeCun) |
| 日期 | June 2026 |
| 项目主页 | 暂无 |
| 对比基线 | [[MPPI]] (Predictive baseline), NMPC |
| 链接 | [arXiv](https://arxiv.org/abs/2606.23444) / Code 暂未公开 |

---

## 一句话总结

> SkyJEPA 将 [[JEPA]] 潜空间预测范式引入四旋翼无人机控制，通过物理启发式探针（Physics-Inspired Prober）将冻结潜变量映射到可解释的物理状态，结合 [[MPPI]] 实现零样本仿真到真实世界的迁移，相比自回归预测基线将位置 RMSE 降低 26–54%。

---

## 核心贡献

1. **JEPA 式潜动力学模型**: 在潜空间预测未来表征而非自回归重建状态，从根本上缓解了[[Compounding Errors|复合误差]]积累问题，实现长时域稳定滚出（60 步时的 Compounding Ratio 从 2.4 降至 1.4）。
2. **物理启发式探针 (Physics-Inspired Prober)**: 两阶段训练，第二阶段在冻结潜变量上训练轻量探针，通过可微分运动学积分器输出可解释的位置/速度/姿态状态，无需端到端重建。
3. **自动化数据合成管道**: 结合[[高斯过程]]轨迹生成 + 领域随机化仿真（500 个环境，2 万条轨迹），不依赖危险的真实世界数据采集。
4. **嵌入式实时控制**: 模型仅 99K 参数，在 NVIDIA Orin NX 上实现 100Hz 控制预算内推理，与 [[MPPI]] 采样优化器集成，零样本迁移到真实四旋翼。

---

## 问题背景

### 要解决的问题

无人机控制需要能进行**长时域预测**的动力学模型，以便 [[MPPI]] / MPC 类优化控制器在飞行时提前规划。现有神经网络动力学模型普遍采用自回归预测范式：预测下一时刻状态，再将预测值作为输入递归推演。

### 现有方法的局限

自回归范式存在根本缺陷——**[[Compounding Errors|复合误差]]（Compounding Error）**：单步预测误差 $\epsilon_t$ 在递归推演中不断累积，导致长时域轨迹严重偏离真实路径。物理约束正则化和在线自适应只能缓解局部误差，无法消除误差传播机制。此外，已有 MBRL 方法常将世界模型与特定奖励/策略耦合，泛化性差。

### 本文的动机

JEPA（Joint Embedding Predictive Architecture）在视觉表征领域证明了**预测潜在表征**比重建像素更稳定。作者将这一思想迁移至无人机动力学建模：在潜空间做多步预测，避免自回归误差传播；再用物理启发式探针将潜变量解码为控制器所需的物理状态量，兼顾可解释性与实时推理效率。

---

## 方法详解

### 模型架构

SkyJEPA 采用 **[[JEPA]] 式编码器-预测器** 架构，分两个阶段训练：

- **输入**: 状态历史 $\mathbf{X}_t = [\mathbf{x}_{t-H}^\top \cdots \mathbf{x}_t^\top]^\top$，动作历史 $\mathbf{A}_t = [\mathbf{a}_{t-H}^\top \cdots \mathbf{a}_t^\top]^\top$（窗口 $H=10$）
- **状态编码器** $\mathrm{Enc}_\theta$（[[TCN|时域卷积网络 TCN]]）: 将状态历史编码为上下文潜变量 $\mathbf{s}_t$
- **动作编码器** $\mathrm{Enc}_\phi$（TCN）: 将动作历史编码为动作潜变量 $\mathbf{z}_t$
- **潜动力学预测器** $\mathrm{Pred}_\varphi$（单层 [[GRU]]，隐藏维度 24）: 在潜空间递归滚出未来表征
- **物理探针** $\psi$（Physics-Inspired Prober）: 在冻结潜变量上训练，输出残差修正量，通过可微分运动学积分器得到物理状态
- **输出**: 物理状态预测 $\tilde{\mathbf{x}}_{t+1:t+T}$（位置、速度、姿态、角速度）
- **总参数**: 99K

### 核心模块

#### 模块 1：JEPA 潜动力学学习（Stage 1）

**设计动机**: 避免[[Compounding Errors|自回归复合误差]]，在[[表征坍塌|潜空间]]中做预测而非重建，同时用 [[SIGReg]] 防止表征坍塌。

**具体实现**:
- 状态/动作历史分别经过 [[TCN]] 编码器得到 $\mathbf{s}_t$、$\mathbf{z}_t$
- [[GRU]] 预测器以 $(\mathbf{s}_t, \mathbf{z}_t)$ 为输入，自回归滚出 $T=20$ 步潜变量 $\tilde{\mathbf{s}}_{t+1:t+T}$
- 目标潜变量 $\mathbf{s}_{t+1:t+T}$ 由同一编码器对未来真实状态编码得到（stop-gradient）
- 多步预测一致性损失 + [[SIGReg]] 正则化防止表征坍塌

#### 模块 2：物理启发式探针（Stage 2）

**设计动机**: 控制器需要 $\mathbf{x}_t$（位置、速度、姿态、角速度）才能计算跟踪代价，但潜变量 $\tilde{\mathbf{s}}$ 是抽象表征；冻结潜变量后单独训练探针，避免有监督损失影响已学到的动力学表征。

**具体实现**:
- 探针 $\psi(\tilde{\mathbf{s}}_{t+k}) \to \{\Delta\dot{\mathbf{v}}_{t+k}, \mathbf{K}_{t+k}\}$ 输出：残差平动加速度修正量 $\Delta\dot{\mathbf{v}}$ 和残差角加速度参数化矩阵 $\mathbf{K}$
- 通过可微分运动学积分器（位置/速度欧拉积分 + [[SO(3) 指数映射]]姿态更新）得到完整状态预测
- 仅探针参数可学习，编码器和预测器全程冻结

---

## 关键公式

### 公式 1：[[Compounding Errors|自回归误差传播]]

$$
\begin{aligned}
\tilde{\mathbf{x}}_{t+1} &= h_\theta(\mathbf{x}_t, \mathbf{a}_t) \\
\tilde{\mathbf{x}}_{t+2} &= h_\theta(\mathbf{x}_{t+1} - \epsilon_{t+1}, \mathbf{a}_{t+1}) \\
&\vdots \\
\tilde{\mathbf{x}}_{t+T} &= h_\theta(\mathbf{x}_{t+T-1} - \epsilon_{t+T-1}, \mathbf{a}_{t+T-1})
\end{aligned}
$$

**含义**: 自回归预测模型在递归滚出时，每步误差 $\epsilon_t$ 以输入形式传入下一步，导致误差持续累积。

**符号说明**:
- $h_\theta$: 神经网络动力学模型
- $\epsilon_{t+k}$: 第 $k$ 步预测误差
- $\mathbf{x}_t$: 系统真实状态，$\tilde{\mathbf{x}}_t$: 预测状态

---

### 公式 2：[[JEPA|JEPA 潜空间滚出]]

$$
\tilde{\mathbf{s}}_{t+T} = \mathrm{Pred}_\varphi(\cdots\mathrm{Pred}_\varphi(\mathrm{Pred}_\varphi(\mathbf{s}_t, \mathbf{z}_t), \mathbf{z}_{t+1})\cdots, \mathbf{z}_{t+T-1})
$$

**含义**: 潜动力学预测器在潜空间递归滚出，预测在抽象表征中进行，不直接操作物理状态，从根本上隔离了复合误差传播路径。

**符号说明**:
- $\mathbf{s}_t = \mathrm{Enc}_\theta(\mathbf{X}_t)$: 状态历史编码的上下文潜变量
- $\mathbf{z}_t = \mathrm{Enc}_\phi(\mathbf{A}_t)$: 动作历史编码的动作潜变量
- $\tilde{\mathbf{s}}_{t+k}$: 第 $k$ 步预测潜变量

---

### 公式 3：[[SIGReg|多步潜预测损失]]

$$
\mathcal{L}_{\mathrm{pred}} = \frac{1}{T}\sum_{k=1}^T \|\tilde{\mathbf{s}}_{t+k} - \mathbf{s}_{t+k}\|_2^2
$$

**含义**: 在整个预测时域 $T$ 上均匀施加监督，鼓励编码器和预测器学习在多步滚出下保持一致的表征。

**符号说明**:
- $\tilde{\mathbf{s}}_{t+k}$: 预测器滚出的潜变量
- $\mathbf{s}_{t+k}$: 真实未来状态经编码器得到的目标潜变量（stop-gradient）

---

### 公式 4：[[SIGReg]] 正则化

$$
\varphi_N(t;\, \mathbf{h}^{(m)}) = \frac{1}{B}\sum_{b=1}^B e^{it h_b^{(m)}}
$$

$$
T^{(m)} = \int_{-\infty}^{\infty} w(t)\left|\varphi_B(t;\, \mathbf{h}^{(m)}) - \varphi_0(t)\right|^2 dt
$$

$$
\mathcal{L}_{\mathrm{SIGReg}} = \frac{1}{M}\sum_{m=1}^M T^{(m)}
$$

**含义**: 通过比较潜变量在随机单位向量 $\mathbf{h}^{(m)}$ 方向上的投影分布与标准正态分布的特征函数差异，鼓励潜空间分布呈各向同性高斯（由[[Cramér-Wold 定理]]保证等价于联合分布匹配），防止[[表征坍塌]]。

**符号说明**:
- $\varphi_N$: 经验[[特征函数]]，是投影分布的傅里叶变换
- $\varphi_0(t) = e^{-t^2/2}$: 标准正态分布的特征函数
- $M$: 随机方向数，$B$: batch size
- $w(t)$: 权重函数

---

### 公式 5：总训练目标

$$
\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{pred}} + \lambda_{\mathrm{sig}}\mathcal{L}_{\mathrm{SIGReg}}
$$

**含义**: 多步预测一致性损失与 [[SIGReg]] 正则化的线性组合，两者协同工作：预测损失保证动力学建模精度，正则化防止表征退化。

**符号说明**:
- $\lambda_{\mathrm{sig}} = 0.02$: 正则化权重

---

### 公式 6：[[物理启发式探针|Physics-Inspired Prober 输出]]

$$
\{\Delta\dot{\mathbf{v}}_{t+k},\; \mathbf{K}_{t+k}\} = \psi(\tilde{\mathbf{s}}_{t+k})
$$

**含义**: 探针网络 $\psi$ 从冻结潜变量中解码出两个残差修正量：平动加速度残差 $\Delta\dot{\mathbf{v}}$ 和角加速度参数化矩阵 $\mathbf{K}$，用于修正名义运动学模型。

---

### 公式 7：残差修正运动学积分

$$
\dot{\mathbf{v}}_t = \left(\sum_{i=0}^3 f_{i,t}/m\right)\mathbf{R}_t\mathbf{e}_3 - \mathbf{g} + \Delta\dot{\mathbf{v}}_t, \quad \Delta\boldsymbol{\tau}_t = \mathbf{K}_t \mathbf{a}_t
$$

$$
\begin{aligned}
\mathbf{p}_{t+1} &= \mathbf{p}_t + \mathbf{v}_t \Delta t \\
\mathbf{v}_{t+1} &= \mathbf{v}_t + \dot{\mathbf{v}}_t \Delta t \\
\mathbf{R}_{t+1} &= \mathbf{R}_t \exp([\boldsymbol{\omega}_t]_\times \Delta t) \\
\boldsymbol{\omega}_{t+1} &= \boldsymbol{\omega}_t + \Delta\boldsymbol{\tau}_t \Delta t
\end{aligned}
$$

**含义**: 名义运动学模型（推力加速度 + 重力）叠加潜空间学到的残差修正，通过[[SO(3) 指数映射]]保持旋转矩阵的几何结构，最终得到完整物理状态预测。

**符号说明**:
- $f_{i,t}$: 第 $i$ 个电机推力，$m$: 质量
- $\mathbf{R}_t \in SO(3)$: 旋转矩阵，$\mathbf{e}_3$: 机体坐标系 z 轴
- $[\boldsymbol{\omega}_t]_\times$: 角速度 $\boldsymbol{\omega}_t$ 的反对称矩阵

---

### 公式 8：[[MPPI]] 重要性采样权重

$$
w^{(s)} = \frac{\exp\!\left(-\left(\mathcal{J}^{(s)} - \mathcal{J}_{\min}\right)/\lambda\right)}{\sum_{r=1}^S \exp\!\left(-\left(\mathcal{J}^{(r)} - \mathcal{J}_{\min}\right)/\lambda\right)}
$$

$$
\mathbf{a}_k^{\mathrm{nom}} \leftarrow \mathbf{a}_k^{\mathrm{nom}} + \sum_{s=1}^S w^{(s)}\delta\mathbf{a}_k^{(s)}
$$

**含义**: MPPI 以 softmax 权重对 $S=512$ 条采样轨迹加权平均，更新名义控制序列；温度 $\lambda=10^{-4}$ 控制权重的集中程度。

**符号说明**:
- $\mathcal{J}^{(s)}$: 第 $s$ 条采样轨迹的总代价
- $\mathcal{J}_{\min}$: 最小代价（数值稳定）
- $\delta\mathbf{a}_k^{(s)} = \mathbf{a}_k^{(s)} - \mathbf{a}_k^{\mathrm{nom}}$: 扰动量

---

### 公式 9：[[Compounding Ratio|Compounding Ratio（复合比）]]

$$
\mathrm{CR}_k = e^{\mathbf{x}}_{k,\mathrm{rollout}} \;/\; e^{\mathbf{x}}_{k,\mathrm{TF}}
$$

**含义**: 开环滚出误差与教师强制（Teacher Forcing）误差之比，值越接近 1 表示自回归递归带来的额外误差越小；SkyJEPA 在 60 步时 CR=1.4，预测基线为 2.4。

**符号说明**:
- $e^{\mathbf{x}}_{k,\mathrm{rollout}}$: 第 $k$ 步自由滚出的归一化 RMSE
- $e^{\mathbf{x}}_{k,\mathrm{TF}}$: 同一模型在[[Teacher Forcing]]模式下的 RMSE

---

### 公式 10：高斯过程参考轨迹生成

$$
p_j(t) \sim \mathcal{GP}(0,\; k_j(t,t')), \quad j \in \{x,y,z\}
$$

**含义**: 用[[高斯过程]]在三个空间轴独立采样平滑随机轨迹，核函数为多个周期核之和，生成兼具全局慢变和局部快变特性的多样化参考轨迹，替代手工设计轨迹。

---

## 关键图表

### Figure 1: 四旋翼世界模型的四个期望特性

> 🖼️ **Figure 1: 四旋翼世界模型的期望特性** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.23444)）

**说明**: 作者认为一个好的四旋翼[[World Model|世界模型]]应同时满足：(1) 精确的长时域预测，(2) 可解释的物理状态输出，(3) 实时推理能力，(4) 零样本任务泛化。SkyJEPA 通过 [[JEPA]] 潜预测 + 物理探针 + [[MPPI]] 集成实现了所有四个目标。

---

### Figure 2: SkyJEPA 整体框架概览

> 🖼️ **Figure 2: 整体框架概览** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.23444)）

**说明**: 上方为状态/动作历史编码路径，[[TCN]] 编码器将历史输入编码为潜变量；中间为[[GRU]]预测器递归滚出未来潜变量；下方为物理探针将潜变量通过可微分运动学积分器解码为物理状态，最终输入 [[MPPI]] 控制器。

---

### Figure 3: 两阶段训练流水线

> 🖼️ **Figure 3: 两阶段训练流水线** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.23444)）

**说明**: **Stage 1**（左）：编码器 + [[GRU]] 预测器联合训练，最小化 $\mathcal{L}_{\mathrm{pred}} + \lambda\mathcal{L}_{\mathrm{SIGReg}}$；**Stage 2**（右）：编码器和预测器权重冻结，仅训练物理探针 $\psi$，最小化 $\mathcal{L}_{\mathrm{prober}}$。两阶段的解耦是防止有监督状态恢复损失破坏已学潜动力学的关键设计。

---

### Figure 4: 真实世界评估场景

> 🖼️ **Figure 4: 真实世界评估场景** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.23444)）

**说明**: 三种评估场景：(a) 标准轨迹跟踪，(b) 载荷运输（+300g，改变质量和惯性），(c) 螺旋桨切换（改变执行特性）。所有场景均为**零样本迁移**，无需真实环境微调。

---

### Figure 5: NVIDIA Orin NX 推理速度分析

> 🖼️ **Figure 5: 推理速度分析** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.23444)）

**说明**: 推理时间随滚出长度和 MPPI 采样数 $S$ 增加。选定参数（$T=15$，$S=512$）在 10ms 实时控制预算内。99K 参数的轻量模型设计是嵌入式部署的关键。

---

### Figure 6: 递归滚出误差分析

> 🖼️ **Figure 6: 递归滚出误差分析** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.23444)）

**说明**: 左图：[[Compounding Ratio|Compounding Ratio]]（复合比）随步数变化，SkyJEPA（CR=1.4）显著低于预测基线（CR=2.4）；右图：误差增长率（Error Growth Rate），SkyJEPA 的误差增量随步数接近平稳，而自回归基线持续增大。

---

### Figure 7: 潜轨迹时序平滑性分析

> 🖼️ **Figure 7: 潜轨迹时序平滑性分析** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.23444)）

**说明**: (a) 笛卡尔坐标轨迹示例；(b) 对应潜轨迹的 PCA 投影，SkyJEPA 的潜轨迹在表征空间中呈平滑直线演化；(c) Temporal Straightening Score，SkyJEPA 得分 0.75 远高于预测基线的 -0.4，表明潜空间中时间演化更平滑一致，有利于稳定控制。

---

### Figure 8: 开环预测精度对比

> 🖼️ **Figure 8: 开环预测精度对比** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.23444)）

**说明**: 不同动力学模型在位置 RMSE、速度 RMSE、姿态误差上随滚出长度的变化。SkyJEPA + PI Prober 在所有指标上均最优，且随步数增加误差增速最慢，体现了 [[JEPA]] 潜预测在长时域的核心优势。

---

### Figure 9: 观测噪声鲁棒性

> 🖼️ **Figure 9: 观测噪声鲁棒性** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.23444)）

**说明**: 在增加 i.i.d. 高斯观测噪声下的位置 RMSE。SkyJEPA 在噪声逐渐增大时表现出明显更好的鲁棒性，得益于潜空间建模对逐步噪声扰动的自然抵抗能力。

---

### Figure 10: 真实世界零样本轨迹跟踪

> 🖼️ **Figure 10: 真实世界零样本轨迹跟踪** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.23444)）

**说明**: 真实飞行轨迹叠加在参考轨迹上，颜色表示跟踪误差（厘米）。五种轨迹（圆形、椭圆、8字形、鱼形、双纽线）均实现零样本迁移，最高速度达 7.2 m/s，最大加速度 12.5 m/s²。

---

### Figure 11: 平台变化下的鲁棒性

> 🖼️ **Figure 11: 平台变化下的鲁棒性** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.23444)）

**说明**: (a) 载荷运输：附加 300g 载荷修改质量和惯性参数；(b) 螺旋桨切换：更换螺旋桨改变执行特性。SkyJEPA 无需重训，在所有扰动场景下位置误差比预测基线低约 1.3–1.35 倍。

---

### Table I: 领域随机化参数

| 参数 | 随机化范围 |
|------|----------|
| 质量 $m$ (kg) | ±50% 标称值 |
| 惯性 $\mathbf{J}$ (kg·m²) | ±30% 标称值 |
| 电机时间常数 $\alpha$ (s) | [0.01, 0.1] |
| 阻力系数 $\mathbf{D}$ | [0.1, 0.5] |
| 推力系数 $k_f$ | ±50% 标称值 |
| 扭矩系数 $k_m$ | ±50% 标称值 |
| 仿真域数量 | 500 |
| 总轨迹数 | 20,000 |
| 单条轨迹时长 (s) | 10 |

**关键发现**: 大范围领域随机化覆盖质量、阻力、惯性和执行器延迟的主要仿真-真实差距来源，是零样本迁移成功的数据基础。

---

### Table II: MPPI 控制参数

| 参数 | 值 |
|------|---|
| 预测时间步 $\Delta t$ | 0.05 s |
| 预测时域 $T$ | 15 |
| 采样数 $S$ | 512 |
| 温度 $\lambda$ | $10^{-4}$ |
| 动作噪声 $\Sigma$ | diag(0.60, 0.15, 0.15, 0.05) |
| 状态代价 $\mathbf{Q}$ | diag(400, 40, 20, 20) |
| 控制代价 $\mathbf{R}$ | diag(0.01, 0.05, 0.05, 0.10) |

---

### Table III: 开环预测分析（位置 RMSE / 姿态误差）

| 方法 | 位置 RMSE [m] 均值 | 位置 RMSE [m] 方差 | 姿态误差 [°] 均值 | 姿态误差 [°] 方差 |
|------|-----------------|-----------------|----------------|----------------|
| Predictive | 8.80 | 2.3 | 53.4 | 14.9 |
| Predictive + Physics Reg. | 7.12 | 2.1 | 49.1 | 13.2 |
| Reconstruction + Prober | 6.82 | 1.6 | 45.2 | 9.7 |
| Reconstruction + PI Prober | 1.53 | 0.13 | 5.28 | 0.70 |
| Ours (JEPA) + Prober | 5.56 | 1.31 | 40.20 | 9.30 |
| **Ours (JEPA) + PI Prober** | **1.43** | **0.10** | **4.71** | **0.50** |

**关键发现**: 物理启发式探针（PI Prober）的引入是最关键改进（从 5.56→1.43m RMSE），而 JEPA 潜建模相比自回归在有 PI Prober 时同样提供了一定增益，且方差更小（0.10 vs 0.13），表明预测更稳定。

---

### Table IV: 真实世界轨迹跟踪（均值/方差）

| 轨迹 | $\|\mathbf{v}\|_{\max}$ [m/s] | $\|\dot{\mathbf{v}}\|_{\max}$ [m/s²] | 位置 RMSE [m]（Ours/Pred+Phy/Pred） | 姿态误差 [°]（Ours/Pred+Phy/Pred） |
|-----|------|------|------|------|
| 圆形 | 2.45 | 1.40 | 0.24/0.36/0.39 | 7.87/10.99/11.95 |
| 椭圆 | 4.50 | 3.78 | 0.33/0.44/0.48 | 9.11/15.20/16.53 |
| 8 字形 | 5.20 | 5.44 | 0.35/0.47/0.51 | 9.25/17.75/20.20 |
| 鱼形 | 5.70 | 7.68 | 0.40/0.54/0.59 | 10.78/20.95/22.78 |
| 双纽线 | 7.20 | 12.5 | 0.45/0.56/0.61 | 19.43/26.83/29.16 |

**关键发现**: SkyJEPA 在所有轨迹上均取得最低误差，位置 RMSE 改善 26–54%；姿态误差在高动态轨迹（双纽线）上改善尤为显著（33–54%），体现了长时域预测稳定性在高速飞行中的优势。

---

### Table V: 平台变化下的鲁棒性（均值/方差）

| 场景 | 轨迹 | 位置 RMSE（Ours/Pred+Phy/Pred） | 姿态误差（Ours/Pred+Phy/Pred） |
|------|------|------|------|
| **螺旋桨切换** | 圆形 | 0.33/0.43/0.45 [m] | 9.54/12.40/12.88 [°] |
| | 8 字形 | 0.35/0.46/0.47 [m] | 10.21/13.27/13.78 [°] |
| | 鱼形 | 0.39/0.51/0.53 [m] | 10.89/14.16/14.70 [°] |
| **载荷运输** | 圆形 | 0.46/0.60/0.62 [m] | 10.11/13.14/13.65 [°] |
| | 8 字形 | 0.49/0.64/0.66 [m] | 9.44/12.27/12.74 [°] |
| | 鱼形 | 0.53/0.69/0.72 [m] | 11.87/15.43/16.02 [°] |

**关键发现**: SkyJEPA 在螺旋桨切换和载荷运输两种平台变化下均保持约 1.3× 的位置误差优势，体现了学到的潜表征对硬件参数变化的内在鲁棒性。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 领域随机化仿真数据 | 20,000 条轨迹 | 500 个随机化仿真环境，[[高斯过程]]参考轨迹，NMPC+MPPI 闭环采集 | 训练 |
| 真实四旋翼飞行数据 | 5 种轨迹 × 多次试验 | 无人机室内飞行，无需额外采集用于训练 | 测试（零样本） |

### 实现细节

- **状态定义**: $\mathbf{x}_t = [\mathbf{p}_t^\top\; \mathbf{v}_t^\top\; \mathbf{r}_{x,t}^\top\; \mathbf{r}_{y,t}^\top\; \mathbf{r}_{z,t}^\top\; \boldsymbol{\omega}_t^\top]^\top \in \mathbb{R}^{18}$（位置 3 + 速度 3 + 旋转矩阵列向量 9 + 角速度 3）
- **编码器**: [[TCN]] (Temporal Convolutional Network)，历史窗口 $H=10$（0.5s）
- **预测器**: 单层 [[GRU]]，隐藏维度 24
- **训练时域**: $T=20$ 步（1.0s）
- **控制时域**: $T=15$ 步（0.75s），MPPI 采样 $S=512$
- **控制频率**: 2Hz（每 0.5s 控制一次）
- **硬件**: 1.3kg 四旋翼，推重比 4:1，NVIDIA Orin NX 嵌入式推理
- **总参数**: 99K
- **SIGReg 权重**: $\lambda_{\mathrm{sig}}=0.02$

### 可视化结果

- 潜轨迹的 PCA 投影（Figure 7）直观展示了 SkyJEPA 在表征空间中的时序平滑性（Temporal Straightening Score=0.75）
- 真实飞行轨迹覆盖误差热力图（Figure 10）显示在双纽线轨迹（7.2 m/s，12.5 m/s²）上也能维持厘米级精度

---

## 批判性思考

### 优点

1. **理论动机清晰**: 从[[Compounding Errors|复合误差]]问题出发，系统性地论证了 [[JEPA]] 潜预测相对于自回归预测的优越性，实验结果（CR=1.4 vs 2.4）有力支撑。
2. **物理结构利用充分**: Physics-Inspired Prober 在潜空间学习之上叠加物理知识，两阶段解耦训练使动力学表征学习不受状态重建任务干扰，设计精巧。
3. **工程完整度高**: 从数据合成、模型训练到嵌入式部署完整覆盖，99K 参数 + Orin NX 实时推理展示了实用价值。
4. **评估全面**: 开环预测 + 闭环跟踪 + 噪声鲁棒性 + 平台变化多维度评估，并且所有对比在零样本迁移条件下进行。

### 局限性

1. **全状态观测假设**: 当前方法依赖完整状态观测（位置、速度、姿态、角速度），未探讨仅有视觉输入时的扩展。
2. **控制频率偏低**: 2Hz 的控制频率在极端高动态飞行场景下可能不足，100Hz 边界基于嵌入式硬件约束。
3. **无在线自适应**: 零样本迁移在平台参数变化不太大时效果好，但无在线自适应机制，面对更大扰动（如电机故障）时可能性能下降。
4. **代码未开源**: 目前无法复现，限制了社区验证和应用。

### 潜在改进方向

1. 引入视觉输入（相机/深度传感器），从像素级观测学习潜动力学（类似 V-JEPA）
2. 结合在线域自适应，在飞行中实时更新探针参数
3. 探索更激进的域随机化（如风扰动、电机失效）以提升极端场景鲁棒性

### 可复现性评估

- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节完整（超参数、架构均已公开）
- [x] 数据集可获取（基于仿真自动生成，可复现）

---

## 关联笔记

### 基于

- [[JEPA]]: 联合嵌入预测架构，本文核心方法论来源
- [[MPPI]]: 采样式模型预测路径积分控制器，SkyJEPA 集成的控制后端
- [[SIGReg]]: Sketched Isotropic Gaussian Regularization，防止潜空间表征坍塌
- [[Teacher Forcing]]: 自回归训练范式，本文对比的主要误差来源
- [[GRU]]: 门控循环单元，用作轻量级潜动力学预测器
- [[World Model]]: 世界模型范式，本文实现了面向控制的高效世界模型

### 对比

- [[JEPA-WM]]: JEPA 式世界模型的通用框架，SkyJEPA 将其具体化到四旋翼控制
- [[MAD]]: 另一篇面向无人机的学习动力学方法（同在 9-无人机 目录）
- [[Latent MPC]]: 潜空间 MPC 控制，与本文的 JEPA+MPPI 路线相近

### 方法相关

- [[Compounding Errors]]: 自回归预测的核心缺陷，本文主要解决的问题
- [[TCN]]: 时域卷积网络，用作状态/动作历史编码器
- [[Latent Dynamics Rollout]]: 潜动力学滚出的通用概念
- [[SO(3) 指数映射]]: 用于姿态积分，保持旋转矩阵的 SO(3) 结构

### 硬件/数据相关

- [[Drone Racing]]: 四旋翼高动态飞行的相关场景
- [[高斯过程]]: 用于自动生成多样化参考轨迹

---

## 速查卡片

> [!summary] SkyJEPA
> - **核心**: 将 JEPA 潜预测引入四旋翼控制，解决自回归模型的复合误差问题
> - **方法**: TCN 编码器 + GRU 潜预测器 + Physics-Inspired Prober + MPPI 控制
> - **结果**: 零样本仿真到真实迁移，位置 RMSE 降低 26–54%，Compounding Ratio 1.4 vs 2.4
> - **代码**: 暂未公开

---

*笔记创建时间: 2026-06-23*
