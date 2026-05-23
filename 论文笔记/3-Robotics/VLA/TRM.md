---
title: "Beyond Euclidean Proximity: Repairing Latent World Models with Horizon-Matched Trajectory Reachability Metrics"
method_name: "TRM"
aliases: [TRM, Trajectory Reachability Metric, Trajectory Reachability Metrics]
authors: [Liangyu Li, Shengzhi Wang, Qingwen Liu]
year: 2026
venue: arXiv
tags: [world-model, latent-planning, model-predictive-control, terminal-cost, reachability, pairwise-ranking]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.22164v1
created: 2026-05-23
---

# 论文笔记：Beyond Euclidean Proximity: Repairing Latent World Models with Horizon-Matched Trajectory Reachability Metrics

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Tongji University |
| 日期 | May 2026 |
| 项目主页 | — |
| 对比基线 | [[LeWM]]、[[PLDM]]（同 manifest 替换终端代价） |
| 链接 | [arXiv](https://arxiv.org/abs/2605.22164) / [HTML](https://arxiv.org/html/2605.22164v1) |

---

## 一句话总结

> TRM 用一个在「时间间隔」上监督的小型成对排序头替换 latent MPC 的欧氏终端代价，**不动 encoder/dynamics/sampler**，hard TwoRoom 成功率从 7.0% 提到 97.0%。

---

## 核心贡献

1. **诊断: Euclidean 终端代价是 latent MPC 的核心痛点**: 即使 latent 里有控制相关变量（XY 位置 $R^2=0.998$ 线性可解码），原始 latent MSE 仍会把"被墙挡住的直线方案"排在"绕过门洞的可行路线"前面，因为 XY 子空间只占 latent MSE 不到 1%。
2. **方法: 后验式 [[Trajectory Reachability Metric|TRM]] 排序头**: 训练一个小 MLP $m_\phi(z_i, z_j)$ 预测 latent 对之间的时间间隔 $|t_i - t_j|$，作为可达性距离；encoder、dynamics、CEM sampler、evaluation manifest 全部冻结，只换终端代价。
3. **设计核心: [[Horizon-Matched Supervision|horizon-matched 监督]]**: 训练对要在长 horizon 上均衡覆盖；只用短 max-$\Delta$=50 的对即便给同样 100k 训练对预算也只到 35.0%，而 balanced full-episode 可到 97.5%。
4. **机制证据 (TwoRoom 案例)**: 提出 [[SCSA|Same-Candidate Selection Audit]]：在**同一批** CEM 候选上比较不同排序，TRM 把 oracle-best 候选从第 32 百分位拉到第 4 百分位（与 [[Spearman 相关系数|Spearman]] 几何距离 0.018 → 0.729）。
5. **边界条件 (PushT)**: 在富接触操作 PushT 上，TRM-style 排序在 SCSA 上仍显著改善（go50 Spearman 0.957），但闭环成功率提升较小——因为瓶颈在接触/rollout 保真度而非排序；此时应做 **hybrid** 而不是替换。
6. **三重消融控制**: shuffled 时间标签（同架构同数据但打乱标签）在所有 6 个 LeWM/PLDM 种子上 0.0%，证明并非"任何学过的代价都能起作用"。

---

## 问题背景

### 要解决的问题

[[Latent MPC]] 的常见做法：将候选动作序列展开到 $H$ 步，用预测的终端 latent $\hat z_{t+H}$ 与目标 latent $z_g$ 的**欧氏距离**作为代价排序候选。但这种 [[Terminal Proximity Cost|终端邻近]]目标的隐含假设——「原始 latent 距离对可达性相关变量加权正确」——并不成立。

### 现有方法的局限

- **生成式 WM**（[[DreamerV3]]、IRIS）需要奖励信号或像素重建，不属于 fixed-latent MPC 这一干净设置。
- **JEPA / latent WM**（[[LeWM]]、[[PLDM]]、[[DINO-WM]]）训得很好的潜空间，仍可能在终端代价上彻底失败：[[TwoRoom]] 这种"绕墙"任务上，LeWM 原始 latent MSE 只有 7.0%，PLDM 32.7%，远低于 oracle task-state cost 的 100.0%。
- **加大搜索 / 标准化 latent**: 把 CEM 提到 1000 样本/20 迭代仍只到 2.5%；逐维标准化后 2.5%。说明问题不在搜索强度，而在**度量**本身。
- **直接训新 encoder/dynamics**: 代价高、改变接口，无法迁移到已部署的 fixed WM 上。

### 本文的动机

如果 latent 状态里**有**任务相关信号（XY 线性可解码 R²=0.998），但**埋在小子空间里被欧氏距离淹没**，那么修复点应该在**「规划器看到的代价」这一层**——只重新加权终端排序，不动 encoder/dynamics/sampler。具体地：用「轨迹中两个 latent 的时间间隔」作为**可达性**的代理监督，训练一个成对头来取代 / 增强欧氏代价。

---

## 方法详解

### 模型架构

<!-- 用结构化列表描述, 不用 ASCII 流程图 -->

[[TRM]] 是一个**接口层**修复，所有改动集中在 [[CEM]] 候选评分这一步：

- **冻结模块**: encoder $f_\theta$、latent dynamics $F_\theta$、候选采样器、优化器、evaluation manifest 全部不变
- **唯一新增**: 成对排序头 $m_\phi: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}_{\ge 0}$
- **输入特征**: $\psi = [z_i, z_j, z_i - z_j, |z_i - z_j|]$（拼接 latent + 差 + 绝对差）
- **架构**: 两层 256-unit MLP，隐藏层 [[SiLU]] 激活，输出 [[Softplus]] 强制非负
- **训练监督**: 来自 logged trajectory 的成对时间间隔 $y_{ij} = |t_i - t_j|$
- **部署**: 替换 ($c_\text{TRM}$) 或混合 ($c_\text{hyb}$) 模式，替代原始 latent MSE 用作终端代价

### 核心模块

#### 模块 1: [[Horizon-Matched Supervision|Horizon-Matched 成对采样]]

**设计动机**: 终端候选排序是「跨长 horizon」的问题——如果训练对集中在小 $\Delta$（如 0–50 步），头学到的是局部度量，无法外推到 200-step horizon 的 terminal 排序。所以监督的**时间间隔分布**必须匹配规划器实际看到的间隔分布。

**具体实现**:
- 对每个 episode (长度 $L_e$)：先**均匀**采 $\Delta \sim \mathcal{U}[1, L_e-1]$
- 再均匀采起点 $t \sim \mathcal{U}[0, L_e - \Delta - 1]$
- 形成对 $(z_t, z_{t+\Delta})$，标签 $y = \Delta$
- "Balanced full-episode" 模式: 在所有 $\Delta$ 范围上额外**均衡化**覆盖

**消融关键发现** (Table 3): 用同样的 100k 训练对预算，max-$\Delta$=50 → 35.0%；full-episode → 97.5%。说明**时间跨度比对数量更重要**。

#### 模块 2: [[Pairwise Ranking Head|成对排序头 $m_\phi$]]

**设计动机**: 用一个轻量、可微的标量头近似"两个 latent 之间的可达时间"。设计要点：

- **非负输出**: Softplus 保证 $m_\phi \ge 0$，符合"距离"语义
- **特征对称性**: 同时输入 $z_i - z_j$ 与 $|z_i - z_j|$，方便头识别方向无关的可达性
- **小参数量**: 两层 256 MLP，使训练只需 100k 对、几分钟内完成
- **冻结上游**: 不反传到 encoder，保持 fixed-WM 假设

**损失**: [[Smooth-L1 Loss|Smooth-L1 (Huber)]] on scaled distances，距离尺度 $s = 224$

#### 模块 3: 终端代价的 Replacement 与 Hybrid 模式

**Replacement 模式 (TwoRoom)**: 完全替换原始 latent MSE。

**Hybrid 模式 (PushT)**: 当原始 latent MSE 仍携带部分有用信号（如接触场景下的 rollout 保真度信号），用**逐 batch 标准化**的加权和：

$$
c_{\text{hyb}}^{(i)} = \frac{c_{\text{lat}}^{(i)} - \mu_{\text{lat}}}{\sigma_{\text{lat}} + \epsilon} + \lambda \cdot \frac{c_{\text{TRM}}^{(i)} - \mu_{\text{TRM}}}{\sigma_{\text{TRM}} + \epsilon}
$$

逐 batch 标准化是关键设计：避免两个代价的绝对尺度差异主导排序。

#### 模块 4: [[SCSA|Same-Candidate Selection Audit]]（诊断工具）

**设计动机**: 闭环成功率混合了"采样到好候选"和"排序到好候选"两个独立变量。SCSA 把它们**解耦**：用 CEM 真实采到的同一批候选，比较不同终端代价**选哪一个**作为最终方案。

**输出**:
1. Spearman $\rho$ vs 欧氏距离 / 测地距离
2. Best-rank percentile: oracle-best 候选被该代价排在第几百分位

SCSA 让"排序质量"独立于"采样质量"，是判断 TRM 是否真的"修了排序"的核心证据。

---

## 关键公式

<!-- 公式标题用 [[概念|名称]] 链接到概念库 -->

### 公式 1: [[Latent MPC|Latent 状态编码]]

$$
\mathbf{z}_t = f_\theta(o_t)
$$

**含义**: 用冻结 encoder 把当前观测 $o_t$ 映射到 latent。
**符号说明**:
- $o_t$: 当前观测（像素 / 状态）
- $f_\theta$: 冻结的 encoder，参数 $\theta$ 不更新
- $\mathbf{z}_t \in \mathbb{R}^d$: latent 表示

### 公式 2: [[Latent Dynamics Rollout|Latent 动力学展开]]

$$
\hat{\mathbf{z}}_{t+H} = F_\theta(\mathbf{z}_t, \mathbf{a}_{0:H-1})
$$

**含义**: 在 latent 空间展开 $H$ 步，得到终端预测 latent。
**符号说明**:
- $F_\theta$: 冻结的 latent dynamics
- $\mathbf{a}_{0:H-1}$: 候选动作序列
- $\hat{\mathbf{z}}_{t+H}$: $H$ 步后的预测终端 latent

### 公式 3: [[Terminal Proximity Cost|原始 latent MSE 终端代价]]

$$
c_{\text{lat}}(\mathbf{a}_{0:H-1}) = \|\hat{\mathbf{z}}_{t+H} - \mathbf{z}_g\|_2^2
$$

**含义**: 当前 latent MPC 的标准做法——预测终端 latent 与目标 latent 的欧氏平方距离。**本文要替换的对象**。
**符号说明**:
- $\mathbf{z}_g$: 目标 latent
- $\|\cdot\|_2^2$: 欧氏平方距离

### 公式 4: [[Pairwise Ranking Head|成对特征图]]

$$
\psi(z_i, z_j) = [z_i, z_j, z_i - z_j, |z_i - z_j|]
$$

**含义**: 把一对 latent 拼成包含原始向量、差向量、绝对差向量的特征，让 MLP 容易学到方向无关的距离。

### 公式 5: [[Horizon-Matched Supervision|时间间隔监督标签]]

$$
y_{ij} = |t_i - t_j|
$$

**含义**: 同一轨迹中两点的时间间隔，作为"可达性距离"代理监督。注意是**绝对值**——只关心相隔多少步，不关心方向。

### 公式 6: [[Pairwise Ranking Head|成对头前向]]

$$
m_\phi(\mathbf{z}_i, \mathbf{z}_j) = \mathrm{Softplus}\big(W_3 \, \sigma(W_2 \, \sigma(W_1 \psi + b_1) + b_2) + b_3\big)
$$

**含义**: 两层 256-unit MLP，SiLU 激活 ($\sigma = \mathrm{SiLU}$)，Softplus 输出保证非负。
**符号说明**:
- $W_1, W_2, W_3, b_1, b_2, b_3$: MLP 参数
- $\sigma$: [[SiLU]] 激活
- $\mathrm{Softplus}(x) = \log(1 + e^x)$: 平滑非负输出

### 公式 7: [[Smooth-L1 Loss|TRM 训练目标]]

$$
\min_\phi \; \mathbb{E}_{(i,j)} \, \ell_{\text{Huber}}\!\left(m_\phi(\mathbf{z}_i, \mathbf{z}_j), \, y_{ij} / s \right)
$$

**含义**: Huber 损失对 outliers 鲁棒；除以距离尺度 $s=224$ 把目标缩到 $\sim 1$ 量级方便优化。
**符号说明**:
- $\ell_{\text{Huber}}$: [[Smooth-L1 Loss]]
- $s = 224$: 距离尺度（与训练 episode 最大长度同量级）

### 公式 8: [[TRM|Replacement TRM 代价]]

$$
c_{\text{TRM}}(\mathbf{a}_{0:H-1}^{(i)}) = m_\phi\!\left(\hat{\mathbf{z}}^{(i)}_{t+H}, \, \mathbf{z}_g\right)
$$

**含义**: 完全替换 $c_\text{lat}$，用 TRM 头评分候选。

### 公式 9: [[Hybrid Terminal Cost|Hybrid 终端代价]]

$$
c_{\text{hyb}}^{(i)} = \frac{c_{\text{lat}}^{(i)} - \mu_{\text{lat}}}{\sigma_{\text{lat}} + \epsilon} + \lambda \cdot \frac{c_{\text{TRM}}^{(i)} - \mu_{\text{TRM}}}{\sigma_{\text{TRM}} + \epsilon}
$$

**含义**: 把两种代价分别按 batch 内均值方差标准化，再用 $\lambda$ 加权。
**符号说明**:
- $\mu, \sigma$: batch 内统计量
- $\lambda$: 混合权重，论文实验中取 $\lambda \approx 1$
- $\epsilon$: 防零除小常数

### 公式 10: [[Subspace Surgery|XY 子空间投影 (机制实验)]]

$$
P_W = W^\top (W W^\top)^\dagger W
$$

**含义**: 用线性 XY probe 的行空间矩阵 $W$ 构造投影算子，把 latent 投到 XY-relevant 子空间或其残差。
**符号说明**:
- $W \in \mathbb{R}^{2 \times d}$: 线性 XY probe 的权重矩阵
- $(W W^\top)^\dagger$: Moore-Penrose 伪逆
- $P_W$: 投影到 XY-rowspace 的算子；$I - P_W$ 是残差投影

这是机制分析的核心工具，用于证明「XY 子空间承担几乎全部控制信号」。

---

## 关键图表

<!-- 三张外链图 + 四个由文字描述的示意图 -->

### Figure 1: Why terminal proximity is not enough (示意图)

**说明**: 三块拼图揭示「latent 邻近陷阱」：
- **(a) Latent-proximity trap**: 在 latent 空间中两个终端状态**看起来很近**，原始欧氏代价无法区分
- **(b) World topology**: 现实世界里被墙阻断，红色直线方案不可行，需绕道蓝色门洞路径
- **(c) Planner-interface repair**: 同一批 CEM 候选下，原始 latent 代价选了被墙挡住的方案，TRM 把可行方案排前

### Figure 2: TRM method and evidence roadmap (示意图)

**说明**: 论文方法 + 证据组织图：
- **Deployed path**: 固定 encoder + dynamics + sampler + optimizer + manifest，只换 CEM 候选评分
- **Diagnostic path** (非部署):
  - [[SCSA]] 同候选排序审计
  - [[Subspace Surgery]] 子空间投影测试任务态是否"在场但被低估"
  - Shuffled labels / weak-search 控制实验排除"任何学过的代价都行"和"加搜索就行"
  - [[PushT]] 标记排序改善但不足以解决接触控制的边界

### Figure 3: TRM 修复控制与候选排序

![Figure 3](https://arxiv.org/html/2605.22164v1/x1.png)

**说明**: 双面板核心结果。
- **(A) Hard TwoRoom 成功率（三种子均值）**: 与 Table 1 对应。LeWM raw → 7.0%；LeWM TRM → 97.0%；shuffled → 0.0%。PLDM 同样 32.7% → 84.0% → 0.0%。
- **(B) SCSA 排序审计 (LeWM seed 3072)**: TRM 真实时间标签头**改善了 CEM 看到的排序**；shuffled 标签头停留在原始 latent MSE 附近。

### Figure 4: 同候选排序解剖 (Hard n100 SCSA audit, 示意图)

**说明**: x 轴是每个 selector 给 oracle-best 候选的百分位排名 (越低越好)：
- 原始 latent MSE 把最佳候选埋在第 **31.7** 百分位
- TRM true labels 把它推到前 **3.9** 百分位
- Shuffled temporal labels 与原始 latent MSE 接近，没用

### Figure 5: TRM 为什么奏效

![Figure 5](https://arxiv.org/html/2605.22164v1/x2.png)

**说明**: 双面板机制 + 消融。
- **(A) 机制证据**: XY-probe rowspace 只占终端 latent MSE **不到 1%**，但承载几乎全部控制有用信号；残差子空间占据 MSE 主体却几乎无控制用
- **(B) 方法消融**: 时间监督**必须跨规划 horizon**。短 horizon temporal head（max $\Delta$=50）失败；full-episode + balanced full-horizon 才能恢复

### Figure 6: PushT 边界条件

![Figure 6](https://arxiv.org/html/2605.22164v1/x3.png)

**说明**: PushT 三协议结果对比。
- **go25** (近饱和)：raw 已经 88%，pair-head 反而 78%、hybrid 86% → 不需要 TRM
- **go50** (中等)：raw 40% → true hybrid 52.7%，shuffled hybrid 42.7% → TRM 有用但提升不大
- **go75** (最难)：raw 16% → true hybrid 22%，shuffled hybrid 17.3%
- **要点**: PushT 闭环受限于接触/rollout/recovery，所以排序改善（见 Appendix B.1 的 SCSA 数据）不能完全转化为闭环成功率提升。**应作为 hybrid 辅助代价而非替换**。

### Figure 7: 任务执行示意图

**说明**: 两个 benchmark 的视觉对比：
- **TwoRoom**: 红色直线被墙阻断，必须走蓝色门洞路径
- **PushT**: T 形物体要靠富接触推到目标姿态，「排序对了不等于能推过去」

### Table 1: Hard n100 TwoRoom 主要替换实验 (三种子均值)

| Cost | Success | Same | Cross | Wrong | Stuck |
|------|---------|------|-------|-------|-------|
| LeWM raw latent MSE | 7.0 | 10.0 | 4.0 | 45.3 | 10.3 |
| LeWM temporal head (TRM) | **97.0** | 95.3 | 98.7 | 0.0 | 0.3 |
| LeWM shuffled head | 0.0 | 0.0 | 0.0 | 46.0 | 8.7 |
| PLDM raw latent MSE | 32.7 | 4.0 | 61.3 | 12.7 | 14.0 |
| PLDM temporal head (TRM) | **84.0** | 79.3 | 88.7 | 3.0 | 2.0 |
| PLDM shuffled head | 0.0 | 0.0 | 0.0 | 50.0 | 2.7 |

**说明**: TRM 在两种 fixed WM 上都把跨墙 (Cross) 成功率推到 88-99%，shuffled 控制完全失败说明「学过的代价 ≠ 任何代价都行」。

### Table 2: Hard n100 SCSA candidate-ordering audit (LeWM seed 3072)

| Candidate cost | ρ Euclid. | ρ geodesic | Best-rank pct. |
|---|---|---|---|
| Raw latent MSE | 0.021 | 0.018 | 31.71 |
| Decoded XY | 0.876 | 0.860 | 0.59 |
| TRM true labels | 0.720 | 0.729 | **3.86** |
| TRM shuffled labels | 0.105 | 0.119 | 34.00 |

**说明**: TRM 的 Spearman 接近 oracle Decoded XY，把 best-rank 从 31.7 推到 3.86 百分位。

### Table 3: TRM 采样 + horizon 消融 (LeWM seed 3072, matched b50)

| Sampling | Sample rows | Train pairs | Success |
|---|---|---|---|
| Random full episode | 20k | 100k | 42.5 |
| Random full episode | 100k | 20k | 62.5 |
| Random full episode | 100k | 100k | 90.0 |
| Balanced full episode | – | 100k | **97.5** |
| Balanced max Δ=50 | – | 100k | 35.0 |

**说明**: 关键证据——同样 100k 对预算，短 horizon 跌到 35.0%，全 episode + balanced 到 97.5%。**Horizon 跨度比对数量更关键**。

### Table 4: TwoRoom 三种子成功率（其它 protocol）

| Model | Balanced b50 | Balanced b150 | Matched b50 | Matched b150 |
|---|---|---|---|---|
| LeWM np512 | 55.0 | 59.2 | 1.7 | 10.0 |
| PLDM full e10 | 66.7 | 76.7 | 19.2 | 29.2 |

**说明**: 在更宽松的 protocol 上原始 latent 仍能拿到中等成绩，hard n100 才是放大问题的关键 manifest。

### Table 5: 三种子 matched b50 planner trace

| Model | Success | Cost vs final distance | Cost vs geodesic progress |
|---|---|---|---|
| LeWM np512 | 1.7 | -0.054 | 0.046 |
| PLDM full e10 | 19.2 | 0.504 | -0.438 |

**说明**: 原始 latent 代价与最终距离、测地进度之间几乎**无相关**——直接的「代价相关性诊断」。

### Table 6: 排除替代解释的控制实验

| Alternative explanation | Test | Outcome |
|---|---|---|
| 任务/预算不可行 | Oracle task-state cost, 同 manifest + CEM 预算 | **100.0%** |
| 搜索太弱 | seed-3072 raw-latent 加强搜索 (1000 samples, 20 iters, top-k 100) | 2.5% |
| Spatial 信息缺席 | Linear XY probe on LeWM latents | RMSE ≈ 1.8 px, **R² = 0.998** |
| 简单 latent scale 病理 | 每维标准化后 MSE | 2.5% |
| 任何学过的头都有效 | Shuffled temporal labels, 6 runs | **0.0% 全失败** |

**说明**: 五个控制全部指向：问题是**度量**，不是任务、搜索、信息存在性或代价规模。

### Table 7: Decoded-XY 代价在 matched b50 下救场

| Cost | Success | Same-room | Cross-wall | Notes |
|---|---|---|---|---|
| Latent MSE | 1.7 | 1.7 | 1.7 | LeWM 三种子均值 |
| Decoded XY Euclidean | **94.2** | 88.3 | 100.0 | Linear probe readout |
| Decoded XY geodesic | 93.3 | 88.3 | 98.3 | 相对 Euclidean 无增益 |

**说明**: 用线性探针在 latent 上**线性解码** XY 再做欧氏距离，立刻达到 94.2%，进一步证明信号在 latent 里，只是被加权错了。

### Table 8: Latent 度量子空间手术 (matched b50)

| Cost | Mean success | Cross-wall success | Spearman vs oracle |
|---|---|---|---|
| Raw latent MSE | 1.7 | 1.7 | 0.028 |
| XY-rowspace latent MSE | **90.8** | 95.0 | 0.779 |
| Residual-only latent MSE | 1.7 | 1.7 | 0.018 |

**说明**: 只用 XY-rowspace 子空间做 MSE → 90.8%；只用残差 → 1.7%。直接确认**信号集中在 <1% 的子空间里**。

### Table 9: PushT 边界总结

| Regime | Closed-loop result | SCSA ordering | SCSA selected distance |
|---|---|---|---|
| go25 | Raw 88.0; pair 78.0; hybrid 86.0 | – | – |
| go50 | Raw 40.0; true hybrid 52.7; shuffled 42.7 | True ρ=0.957; shuffled 0.071 | Raw 127.3; true hybrid 75.3 |
| go75 | Raw 16.0; true hybrid 22.0; shuffled 17.3 | True ρ=0.941; shuffled 0.118 | Raw 191.6; true hybrid 114.5 |

**说明**: SCSA 排序在 go50/go75 上**极强**（true ρ > 0.94），但闭环只小幅提升——瓶颈在接触/rollout 而非排序。

### Table 10: TRM 头实现细节

| Component | Setting |
|---|---|
| Input features | $[z_i, z_j, z_i - z_j, |z_i - z_j|]$ |
| Architecture | MLP, 两层 256-unit 隐藏 |
| Nonlinearity | 隐藏层 [[SiLU]]，输出 [[Softplus]] |
| Optimizer | AdamW |
| Learning rate | $10^{-3}$ |
| Weight decay | $10^{-4}$ |
| Batch size | 1024 pairs |
| Loss | Smooth-L1 on scaled distances |
| Distance scale | 224 |
| Headline train pairs | 100,000 |

**说明**: 整个 TRM 头是**轻量、极简、可复现**的工程——和 encoder/dynamics 的训练成本相比可以忽略不计。

---

## 实验

### 数据集 / 环境

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[TwoRoom]] (hard n100) | 50 same-room + 50 cross-wall | 47/50 cross-wall 必须过门洞 | 主战场 |
| [[TwoRoom]] (matched / balanced) | 40 goals | 控制欧氏距离 | 子实验、SCSA |
| [[PushT]] go25/go50/go75 | 50 episodes/protocol | T 形物体推到目标 | 边界条件 |

### 实现细节

- **Base WM**: [[LeWM]] (np512 配置) + [[PLDM]] (full e10 配置)，皆冻结，使用各自三种子 (3072/3073/3074)
- **TRM head**: 两层 256-unit MLP，SiLU + Softplus，AdamW lr=1e-3, wd=1e-4, batch=1024
- **训练对**: 100k pairs，balanced full-episode 采样（除 ablation）
- **CEM 设置**: 与 LeWM/PLDM 原文一致（np512 / full e10），保证公平对比
- **Manifest**: hard n100 (主)、matched b50/b150、balanced b50/b150 (sub)

### 关键消融

- **Sampling 策略** (Table 3): balanced full-episode > random full-episode > random short-horizon
- **Pair 预算**: 100k pairs 远远够用；20k 也能拿到 62.5%
- **Replacement vs Hybrid**: TwoRoom 上 replacement 即可；PushT 上 hybrid 更安全
- **Shuffled labels 控制**: 6 个种子 / 2 个 WM 全部 0.0% → 排除"任何学过的代价都行"

---

## 批判性思考

### 优点

1. **极致简洁的 fixed-WM 接口修复**: 不动 encoder/dynamics/sampler，只加一个小 MLP，工程门槛极低，可以直接挂到任何 fixed latent WM 上。
2. **诊断 + 修复双线证据**: 不仅给出 method，还提出 [[SCSA]] / [[Subspace Surgery]] / shuffled labels / weak-search 多重控制，**把"为什么 latent MSE 会失败"讲透**，是机制层面真正的贡献。
3. **指出 Horizon-Matched 监督是关键**: 同 100k 对预算下 35.0% vs 97.5% 的 ablation 极有说服力——表明问题不在表达能力或数据量，而在**监督信号的时间分布**与规划器实际看到的分布是否匹配。
4. **诚实标注边界**: PushT 上 SCSA 极强但闭环增益小，作者明确说「应做 hybrid 而非替换」「瓶颈在接触/rollout 不在排序」，没有过度推销 TwoRoom 上的 97% 戏剧性数字。
5. **直接服务于 [[LeWM]]/[[PLDM]] 系列**: 论文等于给整个 latent-space MPC 流派打了一个**通用补丁**，影响远大于单一论文。

### 局限性

1. **依赖 logged 轨迹**: 训练对需要从 episode 里采，对于纯随机数据 / 探索期数据可能监督质量差。
2. **TRM 头是"伪可达性"**: 用时间间隔代理可达性，假设 episode 是"近最优 / 至少可行"的——如果数据集里很多失败轨迹，时间标签和真实可达成本相差很远。
3. **缺乏对动作维度的显式建模**: $m_\phi(z_i, z_j)$ 不知道 $a$，因此在动作维度强烈影响可达性的场景（如复杂操作）可能不够。
4. **PushT 表明接触类任务的瓶颈不在终端排序**: 富接触任务里，"排序对了" 不等于 "推得过去"——这说明 TRM 不是 latent MPC 的银弹，只是**修了一个特定的瓶颈**。
5. **没有与 [[TD-MPC]] 价值学习方案直接对比**: TD-MPC 一类方法用 critic 做终端代价，与 TRM 等价问题但训练方式完全不同；两者孰优本文没给。
6. **PLDM 上提升弱于 LeWM**: 32.7% → 84.0% vs 7.0% → 97.0%。猜测 PLDM 的 latent 几何更乱、XY 子空间不那么"线性独立"，导致单头 MLP 拟合更难。论文没深入分析。

### 潜在改进方向

1. **动作条件 TRM**: $m_\phi(z_i, z_j, a_{i \to j})$，让头看到中间动作，能区分"几步内可达"与"几步内有解但需要特定动作"。
2. **联合训练 encoder + TRM**: 端到端把"可达性"作为辅助监督传回 encoder——可能改善 [[PLDM]] 上提升较弱的问题。
3. **TRM + [[TD-MPC]] 价值学习对比/融合**: critic-based 终端代价 vs 时间间隔代理的系统比较。
4. **拓展到接触类任务**: 用接触-aware 监督（如 contact phase 标注）训练 TRM 头，可能突破 PushT 的瓶颈。
5. **理论分析**: 形式化「为什么 balanced full-horizon 是必要条件」——是否可用 [[Rademacher Complexity]] / 分布偏移理论给出证明。

### 可复现性评估

- [x] 实现细节完整（Table 10 全部超参）
- [ ] 代码 / 模型开源：论文未提供 URL（截至本笔记日期）
- [x] 数据可用：基于公开的 [[LeWM]]、[[PLDM]]、[[TwoRoom]]、[[PushT]]
- [x] 训练资源极低：100k pairs + 小 MLP，几分钟一台 GPU 即可

---

## 关联笔记

### 基于

- [[LeWM]]: 主要 fixed WM 之一，TRM 直接挂在 LeWM 的 CEM 上
- [[PLDM]]: 第二个 fixed WM，证明方法跨 WM 通用
- [[Latent MPC]]: 整个问题设定的背景
- [[Terminal Proximity Cost]]: 本文要替换的基线方案

### 对比

- [[TD-MPC]]: 另一种用 critic 替换终端代价的思路，本文未直接对比但属于同一问题空间
- [[DreamerV3]]: 生成式 WM + 价值学习，不在 fixed-latent MPC 设定内
- [[DINO-WM]]: 高 token 数 foundation-based WM，本文未替换其 CEM

### 方法相关

- [[TRM|Trajectory Reachability Metric]]: 本文核心方法
- [[Horizon-Matched Supervision]]: 关键设计选择
- [[Pairwise Ranking Head]]: 头的形式
- [[Hybrid Terminal Cost]]: PushT 上推荐的部署模式
- [[SCSA|Same-Candidate Selection Audit]]: 诊断工具
- [[Subspace Surgery]]: 机制实验工具
- [[Smooth-L1 Loss]]: 训练损失
- [[SiLU]] / [[Softplus]]: 头的激活函数
- [[CEM]] / [[MPC]]: 规划求解器
- [[Spearman 相关系数]]: SCSA 评估指标

### 数据/任务

- [[TwoRoom]]: 主战场 navigation benchmark
- [[PushT]]: 接触类操作 benchmark
- [[World Model]]: 总体范式

---

## 速查卡片

> [!summary] TRM: Trajectory Reachability Metrics
> - **核心**: 用「同轨迹两点时间间隔」监督一个小 MLP 头 $m_\phi(z_i, z_j)$，**替换 latent MPC 的欧氏终端代价**——encoder/dynamics/sampler 全不动
> - **方法**: $[z_i, z_j, z_i-z_j, |z_i-z_j|]$ → 2-layer 256 MLP + SiLU/Softplus，Smooth-L1 训练 100k pairs，几分钟搞定
> - **关键设计**: [[Horizon-Matched Supervision]]——训练对必须均衡覆盖整个 episode horizon，max-$\Delta$=50 → 35.0%, full-episode → 97.5%
> - **结果**: hard TwoRoom 上 [[LeWM]] 7.0% → 97.0%, [[PLDM]] 32.7% → 84.0%；[[PushT]] 排序极佳但闭环受限于接触，应做 hybrid
> - **机制**: XY 子空间占 latent MSE <1% 却携带几乎全部控制信号；shuffled-label 控制 0.0% 排除"任何学过的代价都行"
> - **代码**: 论文未公开

---

*笔记创建时间: 2026-05-23*
