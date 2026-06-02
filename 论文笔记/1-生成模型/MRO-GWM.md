---
title: "Learning Action-Conditional and Object-Centric Gaussian Splatting World Models for Rigid Objects"
method_name: "MRO-GWM"
authors: [Jens U. Kreber, Lukas Mack, Joerg Stueckler]
year: 2026
venue: arXiv
tags: [world-model, 3d-gaussian-splatting, object-centric, action-conditional, rigid-body-dynamics, embodied-ai, robot-manipulation]
zotero_collection: 1-生成模型
image_source: online
arxiv_html: https://arxiv.org/html/2606.01950v1
created: 2026-06-02
---

# 论文笔记：Learning Action-Conditional and Object-Centric Gaussian Splatting World Models for Rigid Objects

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Embodied Vision Group, University of Augsburg |
| 日期 | June 2026 (arXiv:2606.01950) |
| 项目主页 | https://embodiedvision.github.io/mro-gwm/ |
| 代码库 | — (尚未公开) |
| 对比基线 | [[NeRF-Dynamics\|CompNeRFDyn]]（最近可比工作，无开源代码）|
| 链接 | [arXiv](https://arxiv.org/abs/2606.01950) / [HTML](https://arxiv.org/html/2606.01950v1) / [Project](https://embodiedvision.github.io/mro-gwm/) |

---

## 一句话总结

> 把多刚体场景表示为**物体中心规范帧中的一组 [[3D Gaussian Splatting|3D Gaussian]]**，让一个稀疏的 [[Spatio-Temporal Attention|时空注意力]] [[Transformer]] 从历史观测 + 未来末端执行器轨迹直接预测每个物体的未来 SE(3) 位姿，得到可直接渲染、可用于 [[MPC]] 规划的 3D [[Action-Conditioned World Model|动作条件世界模型]]。

---

## 核心贡献

1. **MRO-GWM 架构**: 首个把 [[Object-Centric Representation|物体中心]] [[3D Gaussian Splatting]] 表征与 [[Action-Conditioned World Model|动作条件世界模型]]结合的方法，原生支持多物体、任意形状、可渲染。
2. **稀疏时空 Transformer**: 提出基于 [[Point Transformer|PTv2]] 向量注意力的多尺度网格 + 时空 k-近邻注意力层，能同时捕捉物体内部 / 物体间 / 跨时间的运动耦合。
3. **规范帧 + 显式 SE(3) 预测**: 不直接预测 Gaussian 位置，而是为每个物体输出未来 SE(3) 增量位姿，作用到规范帧锚点，天然保持刚体性、避免 Gaussian 偏离物体几何。
4. **多视图合成训练数据**: 用 [[ManiSkill]] + [[YCB]] 物体生成 100+ 互动场景，每场景 32 视角 RGB-D 重建出物体的 3DGS 锚点，作为世界模型的输入态。
5. **MPC 规划验证**: 把世界模型当作仿真器嵌入 MPC，在"推物到目标点"和"清理桌面中心"两类多物体任务中达成 66%–82% 的子物体成功率。

---

## 问题背景

### 要解决的问题

如何为 [[Embodied AI|具身智能体]] 学一个**多刚体、动作条件、3D 可渲染、可用于规划**的[[世界模型]]，让它能在多物体接触/推挤场景中提前数秒地预测未来场景？

### 现有方法的局限

| 范式 | 代表 | 关键弱点 |
|------|------|---------|
| 2D 视频 [[扩散模型]] | [[CogVideoX]]、Genie | 仅像素级；动作弱条件；无法显式拆分物体几何 |
| 全局 latent 世界模型 | Dreamer、[[JEPA-WM]] | 多物体扩展差；潜变量与物理量弱绑定 |
| [[NeRF-Dynamics\|动态 NeRF]] | D-NeRF、CompNeRFDyn | 渲染慢；多物体扩展复杂；CompNeRFDyn 无开源 |
| 单 Scene-Level [[3D Gaussian Splatting\|3DGS]] | 4D-GS | 形变以 per-Gaussian 自由度建模，违背刚体先验，难支撑长时多物体动力学 |

### 本文的动机

作者认为，要同时拿到 **(a) 显式 3D + 可渲染、(b) 物理上保持刚体性、(c) 多物体可扩展、(d) 动作可条件** 这四点，最自然的选择是：

1. 用 **[[3D Gaussian Splatting]]** 解决 (a) 渲染速度与显式 3D；
2. 用 **[[Object-Centric Representation|物体中心规范帧]]** 解决 (b) 刚体性、(c) 多物体；
3. 用 **[[Transformer]] + 动作条件** 解决 (d) 动作可控。

→ 因此提出 MRO-GWM：把 Gaussian 锚点在物体规范帧中固定，预测每个物体的 SE(3) 演化。

---

## 方法详解

### 整体架构

MRO-GWM 接收一段**多视图 RGB-D + 物体掩码 + 物体位姿历史 + 末端执行器未来位姿**，输出每个物体在未来 $p$ 步的 SE(3) 位姿，再用 [[3D Gaussian Splatting|3DGS]] 直接光栅化得到未来 RGB / 深度。

**输入 / 输出符号**

- 多视图观测：$I \in \mathbb{R}^{n_{img} \times w \times h \times 4}$（RGB-D），$O \in \mathbb{N}^{n_{img} \times w \times h \times N_{obj}}$（物体掩码）
- 历史物体位姿：$\tau^{obj}_{1:h} = (\xi^{obj}_1, \dots, \xi^{obj}_h)$，$\xi^{obj}_i \in \mathrm{SE}(3)$
- 未来末端执行器轨迹：$\tau^{ee}_{h+1:h+p}$
- 输出：$\hat{\tau}^{obj}_{h+1:h+p}$

### 模块 1: 物体中心 Gaussian 表征

对每个物体 $k$：

- 在多视图重建阶段，运行 ~5000 步 [[3D Gaussian Splatting|3DGS]] 优化，得到该物体在**规范帧**中的锚点集 $\\{a_j^k\\}$（1 cm 网格分辨率，每个锚点附 5 个 Gaussian）。
- 锚点的位置、协方差、不透明度、SH 色都被固定到物体坐标系中。
- 世界坐标下的实际 Gaussian 位置由刚体变换给出：

$$
\mu_j^{world}(t) = \xi^{obj}_k(t) \cdot a_j^k
$$

这种"规范帧 + SE(3)"分解是 MRO-GWM 区别于普通 4D-GS 的关键 —— 它强制刚体约束，并把所有动力学压缩到 $\mathrm{SE}(3)$ 序列上。

### 模块 2: 多尺度稀疏时空 Transformer

每个时间步，物体锚点构成一个稀疏 3D 点集 $\\{(p_i, f_i)\\}$。Transformer 按多尺度处理：

- **空间网格池化** $\mathrm{pool}_i$: 网格大小依次 $[0.019, 0.031, 0.049, 10]$，逐级聚合
- **正弦位置编码** $\mathrm{pe}$（时间索引）
- **每尺度由 $N_{blocks}=2$ 个残差块组成**，每块包含四类层（按顺序）：
  1. [[Point Transformer|PTv2 空间向量注意力]]（k=16 最近邻，物体内+物体间）
  2. **[[Spatio-Temporal Attention|时空注意力]]**（同样 k-近邻，但邻居跨过去时间步）
  3. 时间注意力（同一空间位置跨时间）
  4. 逐点 MLP

整体特征流可形式化为：

$$
s_i = b_{i,N_{blocks}} \circ \cdots \circ b_{i,1} \circ \mathrm{pe} \circ \mathrm{pool}_i
$$

每尺度特征维度 $[48, 88, 160, 296]$，深度 2。末端通过反池化得到每物体的 SE(3) 增量预测。

### 模块 3: SE(3) 输出与规范化

- **平移**：直接回归三维向量，按训练集运动幅度的均值范数归一化 → 损失为 L2。
- **旋转**：先回归一个 $3\times3$ 矩阵 $\mathbf{R}$，按训练集元素级标准差归一化，再做 [[奇异值分解|SVD]] 正交化得到合法 SO(3) 矩阵。

### 模块 4: MPC 规划

把 MRO-GWM 当作可微 (实际是黑箱) 仿真器：

- 候选末端执行器轨迹 → 模型 rollout 多步 → 估计每条轨迹的代价（如目标距离）
- 在 5000+ 候选中迭代采样选择 best → 执行第一步 → 滚动更新（标准 [[MPC]] 流程）
- 单次规划 45–90 分钟（非实时，验证用）

---

## 关键公式

### 公式1: [[3D Gaussian Splatting|物体中心 Gaussian 世界位置]]

$$
\mu_j^{world}(t) = \xi^{obj}_k(t) \cdot a_j^k, \quad \xi^{obj}_k(t) \in \mathrm{SE}(3)
$$

**含义**: 每个 Gaussian 锚点在物体规范帧中固定 $a_j^k$，世界位置由该物体的 SE(3) 位姿驱动 —— 把所有动力学压缩到每物体 6 个自由度上。

**符号说明**:
- $a_j^k$: 物体 $k$ 在规范帧中的第 $j$ 个锚点（位置、协方差、不透明度、SH 色）
- $\xi^{obj}_k(t)$: 物体 $k$ 在时间 $t$ 的 6D 位姿
- $\mu_j^{world}(t)$: 锚点在世界坐标下的位置（用于渲染）

### 公式2: [[Object-Centric Representation|场景输入张量]]

$$
\mathcal{S}_t = \big\\{\, I_t,\; O_t,\; \\{\xi^{cam}_i\\},\; \\{\xi^{obj}_k\\}_{k=1}^{N_{obj}},\; \xi^{robot}_t \,\big\\}
$$

**含义**: 单时刻的完整场景状态包含多视图 RGB-D、物体掩码、相机外参、所有物体位姿和机器人位姿，作为 Transformer 输入。

**符号说明**:
- $I_t \in \mathbb{R}^{n_{img}\times w\times h\times 4}$: 多视图 RGB-D
- $O_t \in \mathbb{N}^{n_{img}\times w\times h\times N_{obj}}$: 物体实例掩码
- $\xi^{cam}_i, \xi^{obj}_k, \xi^{robot}_t \in \mathrm{SE}(3)$: 相应实体位姿

### 公式3: [[Spatio-Temporal Attention|稀疏时空 Transformer]] 复合形式

$$
s_i = b_{i,N_{blocks}} \circ \cdots \circ b_{i,1} \circ \mathrm{pe} \circ \mathrm{pool}_i
$$

**含义**: 第 $i$ 个尺度的特征处理由"网格池化 + 时间正弦位置编码 + $N_{blocks}$ 个残差块"组成；每个残差块包含空间注意力、时空注意力、时间注意力、MLP 四层。

**符号说明**:
- $\mathrm{pool}_i$: 第 $i$ 尺度的空间网格池化
- $\mathrm{pe}$: 正弦时间位置编码
- $b_{i,n}$: 第 $n$ 个残差块（空间 → 时空 → 时间 → MLP）

### 公式4: 旋转损失（归一化矩阵差）

$$
\tilde{\mathbf{R}} = (\mathbf{R} - \mathbf{I}) \oslash \mathbf{\Sigma}_{rot}, \quad \mathcal{L}_{rot} = \|\tilde{\mathbf{R}}_{pred} - \tilde{\mathbf{R}}_{gt}\|_2^2
$$

**含义**: 直接对 SO(3) 矩阵 L2 损失会因量级失衡而难学；本文用训练集每个分量的标准差 $\mathbf{\Sigma}_{rot}$ 做归一化，再算 L2。

**符号说明**:
- $\mathbf{R} \in \mathrm{SO}(3)$: 预测/真值旋转矩阵
- $\mathbf{I}$: 单位阵
- $\mathbf{\Sigma}_{rot}$: 训练集元素级标准差（element-wise）
- $\oslash$: Hadamard 除法

### 公式5: 联合训练目标

$$
\mathcal{L}_{total} = \lambda_{pos}\,\mathcal{L}_{pos} + \lambda_{rot}\,\mathcal{L}_{rot}, \quad \lambda_{pos}=1.0,\; \lambda_{rot}=0.5
$$

**含义**: 平移 L2 + 旋转 L2 的加权和；权重经验值 1.0/0.5。

**符号说明**:
- $\mathcal{L}_{pos}$: 归一化平移 L2 损失（按训练集运动幅度归一化）
- $\mathcal{L}_{rot}$: 公式4 中的旋转损失

---

## 关键图表

### Figure 1: 方法总览

![Figure 1: Overview](https://arxiv.org/html/2606.01950v1/x1.png)

**说明**: 左侧展示 [[Object-Centric Representation|物体中心]] [[3D Gaussian Splatting|3D Gaussian]] 表征 —— 每个物体的 Gaussian 锚点固定在规范帧中，整体被该物体的 SE(3) 位姿统一变换。右侧是稀疑时空 Transformer：多尺度网格池化 → [[Point Transformer|PTv2 空间注意力]] → 新颖的 [[Spatio-Temporal Attention|时空注意力]] → 时间注意力 → MLP；从历史物体锚点 + 未来末端执行器位姿预测每个物体未来 SE(3)。

### Figure 2: 预测误差随时间地平线

![Figure 2: Error vs Horizon](https://arxiv.org/html/2606.01950v1/x2.png)

**说明**: 比较不同时间分辨率（5Hz / 10Hz）与预测步数（4 / 8 / 12 步）下的位置 / 旋转误差。中位数 + 25%–75% 分位带；并区分"全部物体"与"实际发生运动的物体子集"。趋势：误差随地平线增大近似线性增长；10Hz 与 5Hz 在等总时长下表现接近。

### Figure 3: 项目主页架构图（SVG 版）

![Figure 3: Architecture (project page)](https://embodiedvision.github.io/mro-gwm/static/images/arch_plain.svg)

**说明**: 项目主页提供的简化架构示意图，更清晰展示规范帧锚点 + 多尺度 ST-Transformer 的数据流。

### Table 1: 模型变体消融

| 变体 | 中位位置误差 (cm) | 平均位置误差 (cm) | 中位旋转误差 (°) | 平均旋转误差 (°) |
|------|-------------------|-------------------|------------------|------------------|
| **YCB-100-100 (主模型)** | **0.45 (0.01)** | **0.73 (0.01)** | **5.54 (0.27)** | **7.08 (0.05)** |
| YCB-50-200 | 0.46 | 0.75 | 6.25 | 7.14 |
| YCB-200-50 | 0.47 | 0.75 | 5.86 | 7.14 |
| YCBV-100-100 (限定 16 物体) | 0.52 | 0.83 | 5.75 | 7.84 |
| Anchor size 0.005 | 0.47 | 0.75 | 5.56 | 7.15 |
| Anchor size 0.02 | 0.48 | 0.76 | 5.87 | 7.20 |
| Explicit Gaussians (无锚点) | 0.48 | 0.76 | 5.80 | 7.11 |
| Limited view (180°) | 0.47 | 0.75 | 5.98 | 7.27 |
| **w/o 时空注意力** | 0.49 | 0.77 | 5.94 | 7.27 |
| **w/o 多尺度下采样** | 0.57 | 0.86 | 6.95 | 8.03 |

**说明**: 括号内为 5 个种子的标准差；预测步数 4，10Hz。两个关键消融：去掉[[Spatio-Temporal Attention|时空注意力]]使位置误差 +9%、旋转 +7%；去掉多尺度池化使位置误差 +27%、旋转 +25% —— 多尺度对长程交互最关键。

### Table 2: MPC 规划结果

| 任务 | 完成时间 (s) | 成功率 | SR@2cm | SR AUC | 最终目标距离 (cm) | 初始目标距离 (cm) |
|------|--------------|--------|--------|--------|------------------|------------------|
| 任务1：推物到目标 | 20.9 (0.6) | **0.77 (0.02)** | 0.80 | 0.83 | 2.87 | 14.17 |
| 任务2：清理中心 (全场景成功) | 26.7 (1.1) | 0.66 (0.05) | 0.72 | 0.68 | 3.12 | 30.4 |
| 任务2：清理中心 (按物体计) | — | **0.82 (0.03)** | 0.88 | 0.81 | — | — |

**说明**: 用 MRO-GWM 作仿真器嵌入 [[MPC]]，2 种 5-物体任务上均有效。按"每个物体是否被清出中心"计，成功率达 82%；按"整场所有物体都成功"计降至 66%，反映多物体联合任务的累积难度。

---

## 实验结果

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| **[[YCB]]-100-100** | 100 原始场景 × 100 布局增强 | 65 物体子集 | 主训练集 |
| YCB-50-200 / YCB-200-50 | 同总量不同切分 | 探究数据多样性 | 消融 |
| YCBV-100-100 | 16 物体（更受限） | YCB-Video 子集 | 跨子集泛化 |
| 测试集 | 100 场景 × 3 轨迹 | 1–5 物体均衡 | 预测评测 |
| 规划集 | 40 场景 | 5 物体 | MPC 评测 |

**仿真**: [[ManiSkill]] + [[Franka 研究臂]] + 棒状末端执行器，100Hz 物理，20Hz 控制。**重建**: 32 个相机，仰角 30°/60°，半径 0.4m，全 360° 方位角。

### 实现细节

- **优化器**: [[AdamW]] (默认参数)
- **学习率**: $4.4 \times 10^{-4}$，cosine 调度，3k 步预热，60k 总步
- **Batch Size**: 30
- **训练时间**: ~22 小时 (单 Nvidia A40)
- **推理**: 30-batch 0.65 秒
- **规划时间**: 每场景 45–90 分钟（非实时）

### 关键发现

1. **物体中心 + 规范帧 > 显式 Gaussian 直接预测**：Explicit Gaussians 变体在位置上稍差，且失去刚体保证。
2. **多尺度池化最重要**：去掉后误差 +25% 以上，远超去掉时空注意力的影响。
3. **数据增广有效**：100×100 优于 200×50（更多布局增强 > 更多原始场景）。
4. **跨数据子集稍降但鲁棒**：YCBV 限定 16 物体仅使旋转误差从 5.54° 升到 5.75°。
5. **2.4 秒地平线内可用于规划**：Figure 2 显示误差线性增长，10Hz × 24 步合理可用。
6. **MPC 验证可控性**：在 5 物体场景下达成 66–82% 子任务成功率，证明世界模型质量足以驱动规划。

---

## 批判性思考

### 优点

1. **物理保证天然**: 通过规范帧 + SE(3) 参数化，从架构层面保证刚体不变形，无需软约束。
2. **3D 可渲染**: 区别于 latent 世界模型，可直接渲染任意视角，对调试和闭环验证非常友好。
3. **多物体可扩展**: 物体数变化只改变 token 数，不改变架构 —— 优于全局 latent 方法。
4. **稀疏架构高效**: 多尺度 + k-NN 注意力使训练 22 小时即可收敛，适合学术规模复现。
5. **消融充分**: 数据切分、anchor 大小、视图范围、注意力类型都给出了对照实验。

### 局限性

1. **依赖 GT 物体掩码与位姿**: 训练和推理都需已知物体分割与初始位姿，限制了在真实图像（无标注）上的直接部署。
2. **仅刚体**: 不能处理布料、流体、铰链关节，相比 4D-GS 适用面变窄。
3. **未实时**: 规划 45–90 分钟，距离真机闭环还有数量级差距。
4. **没有强基线对比**: 作者承认 CompNeRFDyn 无开源；缺少与 latent 世界模型（如 Dreamer 改造版）的横向对比。
5. **接触建模未显式**: 多物体接触/碰撞被注意力隐式学习，外推到接触主导场景时风险未知。

### 潜在改进方向

1. 用 [[SAM2]] / FoundationPose 替代 GT 掩码与位姿，迈向真实图像输入。
2. 把动作条件扩展到铰链 / 软体物体（与 [[NeRF-Dynamics|动态 NeRF]] 方法做混合）。
3. 用 [[DiT]] / 扩散先验做未来位姿的概率建模，处理多模态运动（如不同接触结果）。
4. 与 [[Diffusion Policy]] 闭环耦合，做端到端 "世界模型 + 策略" 联合训练。
5. 实时化：知识蒸馏 / KV-cache 加速推理至 100ms 级。

### 可复现性评估

- [ ] 代码开源（截至论文发布未开源）
- [ ] 预训练模型（未提及）
- [x] 训练细节完整（学习率/batch/硬件齐全）
- [x] 数据集可获取（YCB / ManiSkill 公开）
- [x] 项目主页提供视频与图示

---

## 关联笔记

### 基于

- [[3D Gaussian Splatting]]: 场景表征基础
- [[Action-Conditioned World Model]]: 上位范式
- [[Point Transformer|PTv2]]: 向量注意力骨架

### 对比

- [[NeRF-Dynamics|CompNeRFDyn]]: 最相关多物体动力学工作，但用 NeRF + 闭源
- [[World Model|Dreamer 系列]]: 全局 latent 路线，缺乏物体显式分解
- [[StableWM]] / [[DINO-WM]]: latent 世界模型的视频/特征路线

### 方法相关

- [[Object-Centric Representation]]: 多物体表征
- [[Spatio-Temporal Attention]]: 关键架构组件
- [[MPC]]: 下游规划框架

### 硬件 / 数据 / 仿真

- [[YCB]]: 物体集
- [[ManiSkill]]: 仿真器
- [[Franka 研究臂]]: 机械臂硬件

---

## 速查卡片

> [!summary] MRO-GWM
> - **核心**: 物体规范帧 + 3D Gaussian + SE(3) 预测 + 稀疏时空 Transformer
> - **方法**: 多尺度 PTv2 + 时空注意力，从历史物体位姿和末端执行器位姿预测未来 SE(3)
> - **结果**: YCB 5 物体场景 2.4 秒地平线位置中位误差 0.45 cm；MPC 推物任务 77% 成功率
> - **代码**: 暂无开源，项目主页 https://embodiedvision.github.io/mro-gwm/

---

*笔记创建时间: 2026-06-02*
