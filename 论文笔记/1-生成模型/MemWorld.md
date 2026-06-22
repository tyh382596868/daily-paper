---
title: "Mem-World: Memory-Augmented Action-Conditioned World Models for Persistent Robot Manipulation"
method_name: "Mem-World"
authors: [Zirui Zheng, Jiaqian Yu, Xiongfeng Peng, Jun Shi, Mingyi Li, Chao Zhang, Weiming Li, Dong Wang, Huchuan Lu, Xu Jia]
year: 2026
venue: arXiv
tags: [world-model, robot-manipulation, memory-augmented, surfel, video-generation, action-conditioned, multi-view]
zotero_collection: 1-生成模型
image_source: online
arxiv_html: https://arxiv.org/html/2606.18960v2
created: 2026-06-22
---

# 论文笔记：Mem-World: Memory-Augmented Action-Conditioned World Models for Persistent Robot Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 大连理工大学 / Samsung R&D Institute China-Beijing (SRCB) |
| 日期 | June 2026 |
| 项目主页 | 暂无 |
| 对比基线 | [[CtrlWorld]] / [[Cosmos Predict 2.5]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.18960) / Code 暂未公开 |

---

## 一句话总结

> 提出 Mem-World，通过 W-VMem（4D 腕部视角 surfel 索引记忆模块）解决机器人操作世界模型在长时序 rollout 中因遮挡和视角运动导致的场景遗忘问题，将策略评估与真实世界的 Pearson 相关性从 0.85 提升至 0.97。

---

## 核心贡献

1. **W-VMem 记忆模块**: 提出 4D 腕部视角中心的 [[Surfel|surfel 索引记忆]]，将历史观测锚定到时间演化的表面元素上，实现几何感知的历史帧检索
2. **多视角动作条件世界模型**: 将 [[Action-Conditioned World Model|动作条件视频生成]] 与显式 3D 几何记忆结合，支持持久化多视角长时序 rollout
3. **完整机器人学习管线验证**: 世界模型作为策略评估器（r=0.97）和策略训练数据源（合成数据将成功率从 58% 提升到 72%）

---

## 问题背景

### 要解决的问题

[[机器人操作]]中的动作条件世界模型需要在长时序预测（long-horizon rollout）中维持一致的场景信息。腕部相机随末端执行器移动，遭受快速的自中心运动（egocentric motion），并被末端执行器或被操作物体频繁遮挡。

### 现有方法的局限

- [[CtrlWorld]]、Interactive World Simulator 等方法仅用固定步长采样（stride sampling）或关节位姿相似度（joint-pose similarity）来选取历史帧
- 这些策略无法捕捉末端执行器遮挡引入的可见性约束
- [[VMem]] 虽用 surfel 索引视频记忆，但假设相机位姿直接可用，且不显式区分时间变化的场景元素

### 本文的动机

通过将历史观测与时间演化的表面元素绑定，并根据几何可见性、任务相关性、时间衰减三维指标综合检索，可以在遮挡和快速运动下仍然找到最相关的历史帧，从而解决场景遗忘问题。

---

## 方法详解

### 模型架构

Mem-World 以 [[CtrlWorld]] 为骨干，在其多视角 [[视频扩散模型]] 基础上增加 W-VMem 记忆模块：

- **输入**: 初始多视角观测 $o_{t=0}$（腕部视角 + 2 个第三视角）+ 动作序列 $a_{t+1:t+H}$
- **记忆模块**: [[W-VMem]]——时间感知的 surfel 索引记忆，按几何相关性检索 $K$ 帧历史
- **骨干网络**: Ctrl-World（基于 [[WAN 视频模型]] 的动作条件视频扩散）
- **输出**: 多视角未来观测 $\hat{o}_{t+1:t+H}$（腕部 + 第三视角）

整体推理循环：

$$
\hat{o}_{t+1:t+H} \sim W(\cdot \mid \hat{o}_t,\, \mathcal{H}_t,\, a_{t+1:t+H})
$$

其中 $\mathcal{H}_t = \{o_{g_1}, \ldots, o_{g_K}\}$ 为 W-VMem 检索到的 $K$ 个历史时间步观测。

### 核心模块

#### 模块1: W-VMem（腕部视角 4D Surfel 记忆）

**设计动机**: 利用 [[Surfel|surfel（表面元素）]] 作为几何锚点，将历史观测按 3D 位置索引，并引入时间和任务相关性维度，解决遮挡导致的检索失效。

**Surfel 定义**:

$$
s_k = (p_k,\; n_k,\; r_k,\; t_k,\; m_k)
$$

- $p_k$: 3D 位置
- $n_k$: 表面法向量
- $r_k$: 半径
- $t_k$: 创建/更新时间步
- $m_k \in \{0, 1\}$: 被操作物体标志（manipulated-object flag）

**初始化**: 从第一帧三个相机视角重建初始 surfel 点云。

**更新策略**: 仅使用腕部视角帧更新 surfel，不使用第三视角帧——这样可防止大量 surfel 被反复刷新时间戳，保留时间关联性。

#### 模块2: 几何感知检索评分

对于候选历史时间步 $t$，计算其所有可见 surfel 的聚合分数，选取 top-K（配合 [[Non-Maximum Suppression]] 去冗余）。

#### 模块3: 腕部位姿预测（Forward Kinematics）

未来腕部位姿由策略输出的关节空间动作通过前向运动学（[[Forward Kinematics]]）推导，利用腕部相机与末端执行器之间的固定几何变换，无需额外位姿估计器。

---

## 关键公式

### 公式1: [[Mem-World 推理|世界模型 Rollout 核心方程]]

$$
\hat{o}_{t+1:t+H} \sim W\!\left(\cdot \mid \hat{o}_t,\, \mathcal{H}_t,\, a_{t+1:t+H}\right)
$$

**含义**: 世界模型 $W$ 以当前观测、检索历史帧集合、未来动作序列为条件，自回归生成下一时刻起 $H$ 步的多视角观测。

**符号说明**:
- $\hat{o}_t$: 当前（预测的）多视角观测
- $\mathcal{H}_t = \{o_{g_1}, \ldots, o_{g_K}\}$: W-VMem 检索出的 $K$ 个历史帧
- $a_{t+1:t+H}$: 未来 $H$ 步动作块（[[Action Chunking]]）
- $H$: 动作/预测时域长度

### 公式2: [[Surfel|Surfel 定义]]

$$
s_k = (p_k,\; n_k,\; r_k,\; t_k,\; m_k)
$$

**含义**: W-VMem 中每个 surfel 的完整属性，扩展了 VMem 的静态定义，增加了时间戳 $t_k$ 和操作物标志 $m_k$。

**符号说明**:
- $p_k \in \mathbb{R}^3$: surfel 的 3D 世界坐标
- $n_k \in \mathbb{R}^3$: 表面法向量（单位向量）
- $r_k > 0$: surfel 半径（影响覆盖面积）
- $t_k$: surfel 的创建/上次更新时间步
- $m_k \in \{0, 1\}$: 是否属于被操作物体

### 公式3: [[W-VMem 检索评分|几何感知时间衰减评分函数]]

$$
\mathrm{score}(s, t) = \frac{\langle \mathbf{n}_s,\, \bar{\mathbf{v}}_w \rangle}{1 + d_s} \cdot \ln(e + m_s) \cdot \left[\lambda_{\min} + (1 - \lambda_{\min})\, 2^{-\frac{T - t}{H}}\right]
$$

**含义**: 对历史时间步 $t$ 的每个可见 surfel $s$，综合三项因子得到检索分数，分数越高表示该历史帧越值得作为上下文提供给世界模型。

**符号说明**:
- $\mathbf{n}_s$: surfel 的表面法向量
- $\bar{\mathbf{v}}_w$: 未来腕部视角方向的平均向量（由前向运动学推导）
- $d_s$: surfel 到未来相机的深度距离
- $m_s \in \{0, 1\}$: 被操作物标志（$\ln(e + 1) \approx 1.31$ vs $\ln(e) = 1$，对操作物加权）
- $T$: 当前最新时间步
- $t$: 候选 surfel 的更新时间步
- $H$: 时间半衰期（设置为 $0.3 \times T_{\max}$）
- $\lambda_{\min} = 0.1$: 最小衰减因子（防止历史帧分数完全归零）

**三项因子解读**:
1. **几何可见性因子** $\frac{\langle \mathbf{n}_s, \bar{\mathbf{v}}_w \rangle}{1+d_s}$：法向量与视角对齐 + 深度越近越大
2. **任务相关性因子** $\ln(e + m_s)$：被操作物 surfel 获得更高分数
3. **时间衰减因子**：指数衰减使近期帧优先，但 $\lambda_{\min}$ 保留远古帧最低权重

---

## 关键图表

### Figure 1: Mem-World Pipeline 总览

![Figure 1: Pipeline](https://arxiv.org/html/2606.18960v2/figs/pipeline.png)

**说明**: Mem-World 的完整 rollout 管线。腕部视角 4D surfel 记忆（W-VMem）维持过去观测的几何感知表示，根据未来腕部位姿检索相关历史帧，连同未来动作序列一起喂入多视角动作条件世界模型，生成的新观测再反馈回记忆模块，实现迭代 rollout。

### Figure 2: 长时序 Rollout 定性结果

![Figure 2: Qualitative comparison](https://arxiv.org/html/2606.18960v2/figs/comparision.png)

**说明**: 长时序 rollout 定性对比。Mem-World 展现出持久一致的世界建模：盖子在被夹爪短暂遮挡后仍能正确恢复，黑色零食在相机来回运动后仍能一致地重现；而对比方法出现物体变形或消失。

### Figure 3: 记忆检索策略消融结果

![Figure 3: Ablation on memory retrieval](https://arxiv.org/html/2606.18960v2/figs/ablation.png)

**说明**: W-VMem 通过检索几何相关历史帧，帮助世界模型在未来观测中保持一致的物体外观。Short-term Mem（仅用近期帧）和 Stride Mem（固定步长采样）提供的上下文信息量不足，导致勺子等物体在 rollout 中变形或消失。

### Figure 4: 策略评估相关性分析

![Figure 4: Quantitative correlations](https://arxiv.org/html/2606.18960v2/figs/correlation.png)

**说明**: 在 5 个任务上，Mem-World（r=0.97，p<0.001）与真实世界表现的 [[Pearson 相关系数]] 显著高于 Ctrl-World（r=0.85，p<0.01），说明 Mem-World 可以作为更可靠的虚拟策略评估器。

### Figure 5: 策略训练效果（合成数据提升）

![Figure 5: Policy improvement](https://arxiv.org/html/2606.18960v2/figs/policy_improvement.png)

**说明**: 在合成数据上后训练的策略（π₀.₅）在长时序任务平均成功率从 58% 提升到 72%，验证 Mem-World 生成的合成轨迹可有效补充真实数据。

### Figure 6: π₀.₅ Rollout 对比（真实 vs. Ctrl-World vs. Mem-World）

![Figure 6: Policy evaluation comparison](https://arxiv.org/html/2606.18960v2/figs/policy_eval.png)

**说明**: π₀.₅ 在真实世界、Ctrl-World 和 Mem-World 中的 rollout 帧对比，直观展示 Mem-World 在场景一致性上的优势。

### Table 1: 世界模型一致性定量评估

| 相机 | 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Obj. Con. ↑ |
|------|------|--------|--------|---------|------------|
| 第三视角 | Cosmos Predict 2.5 | 22.80 | 0.819 | 0.089 | 0.579 |
| 第三视角 | Ctrl-World | 23.17 | 0.828 | 0.076 | 0.573 |
| 第三视角 | **Mem-World (Ours)** | **25.30** | **0.878** | **0.054** | **0.619** |
| 腕部视角 | Ctrl-World | 17.34 | 0.623 | 0.281 | 0.476 |
| 腕部视角 | **Mem-World (Ours)** | **19.21** | **0.691** | **0.236** | **0.524** |

**关键发现**: Mem-World 在第三视角 PSNR 上比 Ctrl-World 高 2.13 dB，LPIPS 改善约 29%；腕部视角也有显著提升，说明几何记忆辅助对两类视角均有效。

### Table 2: 记忆设计消融实验

| 相机 | 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Obj. Con. ↑ |
|------|------|--------|--------|---------|------------|
| 第三视角 | Short-term Mem | 21.25 | 0.802 | 0.082 | 0.526 |
| 第三视角 | Stride Mem | 22.58 | 0.814 | 0.079 | 0.544 |
| 第三视角 | **W-VMem (Ours)** | **24.78** | **0.869** | **0.062** | **0.597** |
| 腕部视角 | Short-term Mem | 15.04 | 0.526 | 0.353 | 0.401 |
| 腕部视角 | Stride Mem | 17.06 | 0.614 | 0.295 | 0.463 |
| 腕部视角 | **W-VMem (Ours)** | **18.97** | **0.680** | **0.248** | **0.502** |

**关键发现**: Short-term Mem 性能最差，说明机器人操作场景中"仅看最近帧"远不够；W-VMem 在腕部视角 PSNR 上比 Stride Mem 高约 1.9 dB，体现几何感知检索的价值。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| DROID | 95,599 条轨迹 | 多任务机器人操作，含遮挡和大运动 | 世界模型训练 / 一致性评估 |
| 真机采集 | 50 集/任务 | Franka Panda，含 5 类任务 | 策略 + 世界模型后训练 |

### 实现细节

- **机器人**: Franka Emika Panda，2-DoF 夹爪，1 个腕部相机 + 2 个外部第三视角相机
- **骨干**: Ctrl-World（基于 WAN 视频扩散）
- **训练数据**: 从 DROID 采样 11K 条轨迹，几何相关上下文替换时序相邻上下文
- **超参数**: $\lambda_{\min} = 0.1$，时间半衰期 $H = 0.3 \times T_{\max}$
- **世界模型微调**: 8 × H100，batch size 32，约 2 天，上调腕部视角预测损失权重
- **策略后训练**: 4 × H100，20K steps；世界模型后训练：4 × H100，5K steps
- **评估集**: 34 条精心筛选的包含遮挡或大幅往返运动的轨迹

### 策略评估任务

| 任务 | 类型 |
|------|------|
| Pick cube to plate | 短时序 |
| Wipe table | 短时序 |
| Pick 1 object to 2 targets | 长时序 |
| Pick 2 objects to 1 target | 长时序 |
| Pick 2 objects to 2 targets | 长时序 |

### 可视化结果

Mem-World 在长时序 rollout 中显著优于 Ctrl-World：遮挡后物体外观恢复正确，来回运动后黑色零食等细小物体保持一致，避免了 Ctrl-World 中常见的 hallucination（如物体突然变形、颜色漂移）。

---

## 批判性思考

### 优点

1. **几何先验与时间先验融合设计优雅**: 评分函数将几何可见性、任务相关性、时间衰减三者乘积组合，各因子物理意义明确且可独立调节
2. **训练成本合理**: 相对于从头训练，基于 Ctrl-World 后训练仅需 2 天 8 卡，具备实用性
3. **端到端验证完整**: 覆盖世界模型质量（PSNR/SSIM）、策略评估器（Pearson 相关）和策略训练（成功率提升）三个使用场景

### 局限性

1. **多视角一致性利用不足**: 初始腕部帧被遮挡时，未充分利用第三视角的一致性约束来辅助 surfel 初始化
2. **缺乏显式物理约束**: 无接触力建模、无反向抓取（antipodal grasping）验证，长期操作仿真的物理合理性有限
3. **仅腕部视角更新 surfel 的策略**: 虽然有效防止时间戳刷新，但当腕部视角长时间不可用时，surfel 信息将停止更新

### 潜在改进方向

1. 引入第三视角辅助 surfel 更新，在腕部遮挡时提供补充几何信息
2. 结合神经渲染（如 [[3D Gaussian Splatting]]）提升 surfel 表示的精细度
3. 扩展到移动操作（mobile manipulation）场景，surfel 记忆可能需要动态地图管理

### 可复现性评估

- [ ] 代码开源（暂未公布）
- [ ] 预训练模型（暂未公布）
- [x] 训练细节完整（超参数、硬件、步数均已说明）
- [x] 数据集可获取（DROID 公开）

---

## 关联笔记

### 基于

- [[CtrlWorld]]: 骨干多视角动作条件世界模型
- [[VMem]]: surfel 索引视频记忆原型，W-VMem 在其基础上增加时间与任务感知

### 对比

- [[CtrlWorld]]: 主要对比基线，策略评估相关性 r=0.85 vs. 本文 r=0.97
- [[Cosmos Predict 2.5]]: 通用视频预测模型，作为无动作条件基线

### 方法相关

- [[Surfel]]: 核心几何表示，W-VMem 的索引单元
- [[Forward Kinematics]]: 由关节空间动作推导腕部位姿的方法
- [[Non-Maximum Suppression]]: 检索后去冗余策略
- [[Action Chunking]]: 动作块预测，策略输出形式
- [[视频扩散模型]]: 骨干生成模型类型
- [[Pearson 相关系数]]: 策略评估器质量度量指标

### 硬件/数据相关

- [[Franka Panda]]: 实验用机器人
- [[DROID]]: 训练数据集（95,599 条轨迹）

---

## 速查卡片

> [!summary] Mem-World (arXiv 2606.18960)
> - **核心**: 4D surfel 记忆 (W-VMem) + 多视角动作条件视频扩散，解决长时序操作 rollout 中的场景遗忘
> - **方法**: 几何可见性 × 任务相关性 × 时间衰减评分函数，检索最相关历史帧作为世界模型上下文
> - **结果**: 策略评估相关性 r=0.97（+14.5% vs Ctrl-World）；策略训练成功率 58%→72%
> - **代码**: 暂未公开

---

*笔记创建时间: 2026-06-22*
