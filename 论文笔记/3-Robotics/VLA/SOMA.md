---
title: "Spatial Memory for Out-of-Vision Manipulation in Vision-Language-Action"
method_name: "SOMA"
authors: [Pengteng Li, Weiyu Guo, He Zhang, Tiefu Cai, Xiao He, Yandong Guo, Hui Xiong]
year: 2026
venue: ICML 2026
tags: [vla, spatial-memory, out-of-vision, manipulation, vggt, multi-view, embodied-ai]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.22283v1
created: 2026-05-23
---

# 论文笔记：Spatial Memory for Out-of-Vision Manipulation in Vision-Language-Action

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | HKUST (Guangzhou)、AI<sup>2</sup>Robotics 等 |
| 日期 | May 2026 (ICML 2026 Accept) |
| 项目主页 | Code will be released soon |
| 对比基线 | [[GR00T N1.5]], [[OpenVLA-OFT]], [[Pi05\|π₀.₅]], [[MemoryVLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.22283) / [HTML](https://arxiv.org/html/2605.22283v1) / [PDF](https://arxiv.org/pdf/2605.22283) |

---

## 一句话总结

> SOMA 用可动头部相机扫描场景构建持久 [[空间记忆专家|空间记忆]]，让 [[VLA]] 摆脱"目标必须在视野内"的隐含假设，可对**视野外（OOV）目标**进行操作。

---

## 核心贡献

1. **首次正面对 [[VLA]] 的"视野内偏置"建模**: 现有 VLA 隐式假设任务目标始终可见，遇到视野外目标只能"反应式"乱搜；SOMA 用一个**持久空间记忆**替代瞬时观察。
2. **统一的空间-语义记忆框架（[[SOMA 三阶段框架]]）**: 由 *Spatial Memory Construction*、*Dynamic Memory Refinement*、*Contextual Memory Retrieval* 三阶段组成，把 [[VGGT]] 的几何先验 + [[DINOv3]] 的语义特征 + [[YOLO-World]] 的开放词汇检测对齐到一个全局 3D 坐标系。
3. **新的真实世界 [[OOV (Out-of-Vision)|OOV]] PnP benchmark**: 五个递增难度的真机任务（不可见→不可见、可见→不可见、不可见→可见、序列双物体、双臂协调），并定义五个**行为指标**（首次注视时间、头部搜索路径长度、视角修正次数、抓取尝试次数、抓取耗时）。
4. **大幅 SOTA**: 在 [[SimplerEnv]] Visual Matching 上 63.2% > RoboVLM 60.6%，在 RoboCasa GR1（300 demo）上 53.3% 显著超过 [[GR00T N1.5]] 的 35.7%，真机 OOV 任务行为指标普遍降低 40–60%。

---

## 问题背景

### 要解决的问题

[[VLA]] 模型在长程操作里有一个被忽视的根本缺陷：**目标可能根本不在当前帧里**。现有策略 $a_t = \pi_\theta(o_t, l)$ 把 $o_t$ 当成"完整的世界"，一旦目标物体在杯子背后、抽屉里、桌子的另一侧，策略就只能盲目摇头或反复试抓。

### 现有方法的局限

- **反应式感知**: [[OpenVLA]]、[[Pi05|π₀.₅]] 等仅消费当前帧，没有跨视角累积。
- **token-level memory 不够**: [[MemoryVLA]] 用 token 记忆，但容量受 context window 限制，对显式 3D 几何不友好。
- **关键帧库不空间化**: [[关键帧记忆库]] 储存的是 2D 图像，无法回答"杯子在头部相机的左后 30°"这种空间查询。
- **NeRF / 3DGS 重建太慢**: 几何一致的方法如 [[3D Gaussian Splatting]] 需要离线优化，不能实时支持机器人。

### 本文的动机

借助预训练好的多视图 3D 模型 [[VGGT]]，可以**一次前向**得到一致的相机位姿和粗糙几何，再把 [[DINOv3]] 语义特征和 [[YOLO-World]] 检测框 lift 到 3D，就能拿到一个"全局可查询的物体记忆"。这种记忆既空间又语义，正好填补 OOV 操作的空白。

---

## 方法详解

### 模型架构

SOMA 在 [[GR00T N1.5]] 主干上插入**空间记忆 - 跨注意力**通路：

- **输入**:
  - 扫描序列 $\mathcal{V} = \{f_i\}_{i=1}^{N_v}$（头部相机围绕工作空间转一圈采集的 $N_v$ 张 RGB）
  - 当前操作时刻的观测 $o_t$、状态 $s_t$、语言指令 $l$
- **三个感知专家**:
  - [[VGGT]]: 输出每帧相机位姿 + 粗糙深度，得到全局 3D 坐标
  - [[YOLO-World]]: 开放词汇 2D 检测，得到逐帧物体框
  - [[DINOv3]]: 高维视觉外观特征
- **记忆构建**:
  - 把每帧 2D 检测框 lift 到 3D 得到 $b_j^{(i)} \in \mathbb{R}^{8\times 3}$
  - 用 cosine + 3D IoU 做跨视角实例关联
  - 得到全局物体级 token 集合 $M_0 = \{m_k^0\}_{k=1}^{N_I}$
- **动态记忆**: 操作过程中观测继续更新 $M_0 \to \tilde{M}^t$
- **检索**: VLM token 作为 query，记忆 token 作为 key/value，做 [[Cross-Attention]]
- **输出**: [[DiT]] block 输出 [[Action Chunking|动作块]] $a_{t:t+k}$（7-DoF 双臂 + 2-DoF 头部）

### 核心模块

#### 模块1: Spatial Memory Construction（空间记忆构建）

**设计动机**: 一次扫描 + 一次 [[VGGT]] 前向，把"杯子在哪儿、桌子有多高、抽屉怎么摆"全部刻进 3D token。

**具体实现**:

1. 头部相机执行**预定义扫描轨迹**，采集 $N_v$ 帧 $\mathcal{V}$。
2. 三个专家并行：
   - [[VGGT]] 估计相机位姿 $\{T_i\}$ 和点图（global 坐标）。
   - [[YOLO-World]] 在每帧输出框 + 类别 $\{(c_j^{(i)}, \text{bbox}_j^{(i)})\}$。
   - [[DINOv3]] 输出 patch feature，并用 ROI Align 得到每个框的外观特征 $f_j^{(i)}$。
3. 每个框 2D → 3D：把 4 个角根据 [[VGGT]] 深度反投影，得到 3D 包围盒 $b_j^{(i)} \in \mathbb{R}^{8\times 3}$。
4. 类别内做实例关联，相似度 = $\lambda_{\text{app}} \cdot \cos(f_j, f_k) + \lambda_{\text{geo}} \cdot \text{IoU}_{3D}(b_j, b_k)$。
5. 每个全局实例表示为三元组 $(f_k, c_k, b_k)$，并投影成记忆 token $m_k^0$。

#### 模块2: Dynamic Memory Refinement（动态记忆刷新）

**设计动机**: 桌面会变（人手伸进去、机器人推动物体），记忆必须**温和**更新——既不能忘掉初始扫描，又要纳入新观察。

**具体实现**:

- 操作中每隔若干帧，对当前观测重新跑感知专家得到 $M^t$。
- 与 $M_0$ 做同类匹配，对每对 $(m_k^{t-1}, m_j^t)$ 计算相似度 $s_{kj}^t$ 和融合权重 $g_{kj}^t$。
- 用[[EMA|时序指数滑动平均]]做软更新，权重 $\alpha = g \cdot s$，相似度高才大幅更新。
- 这一步把短期波动（手挡住一瞬间）滤掉，同时支持物体被搬动的长期变化。

#### 模块3: Contextual Memory Retrieval（情境化记忆检索）

**设计动机**: 任务"把红杯子放到抽屉里"只关心两个物体，不能让所有记忆 token 都干扰策略。

**具体实现**:

- VLM 编码 $(o_t, l)$ 得到 token 序列 $X_{vl}$。
- 用 $X_{vl}$ 作 query，$\tilde{M}^t$ 作 key / value 跑一层 [[Cross-Attention]]。
- 输出 boost 信号注入 [[DiT]] 主干，等价于一种"语言条件的空间检索"。
- 这相当于在动作生成前先做一次"我应该看哪里"的隐式推理。

---

## 关键公式

### 公式1: [[空间记忆专家|空间-语义记忆构建]]

$$
m_k^0 = \Phi_{\text{mem}}(f_k) + p_k
$$

**含义**: 把每个全局实例的 [[DINOv3]] 外观特征 $f_k$ 投影到记忆空间，再加上 3D 位置编码 $p_k$，得到该实例的初始记忆 token。

**符号说明**:
- $f_k$: 第 $k$ 个全局实例的聚合 [[DINOv3]] 外观特征
- $\Phi_{\text{mem}}$: 投影 MLP，把外观特征映射到记忆维度
- $p_k$: 3D 位置嵌入（基于全局坐标系下的物体中心）
- $m_k^0$: 初始记忆 token（spatial-semantic 双信息）

### 公式2: [[相似度-融合双权重]]（动态更新打分）

$$
\begin{aligned}
s_{kj}^t &= \sigma\!\left(\Phi_{\text{sim}}\!\left([m_k^{t-1} - m_j^t]\right)\right) \\
g_{kj}^t &= \sigma\!\left(\Phi_{\text{fuse}}\!\left([m_k^{t-1}, m_j^t]\right)\right)
\end{aligned}
$$

**含义**: 同时学两套门控——一个判断"这两个 token 是不是同一个物体"，一个判断"现在该不该融合"，避免错误关联和过快遗忘。

**符号说明**:
- $m_k^{t-1}$: 历史全局记忆 token
- $m_j^t$: 当前观测得到的候选 token
- $\sigma$: sigmoid 激活，把 logits 映射到 $(0,1)$
- $\Phi_{\text{sim}}, \Phi_{\text{fuse}}$: 两个独立小 MLP
- $s_{kj}^t$: 相似度分数
- $g_{kj}^t$: 融合门控分数

### 公式3: [[EMA|时序指数滑动平均]]更新

$$
m_k^t = \alpha_{kj}^t \cdot m_j^t + (1 - \alpha_{kj}^t) \cdot m_k^{t-1}, \quad \alpha_{kj}^t = g_{kj}^t \cdot s_{kj}^t
$$

**含义**: 记忆按"两个门的乘积"做软更新，乘积大才大幅替换，乘积小则维持历史，实现"稳定但可塑"的长期记忆。

**符号说明**:
- $\alpha_{kj}^t \in (0,1)$: 综合更新权重
- $m_k^t$: 时刻 $t$ 的全局记忆 token

### 公式4: [[Cross-Attention|情境化记忆检索]]

$$
X_{\text{boost}} = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{C}}\right) V
$$

**含义**: 用 VLM token 当 query 去"问"全局空间记忆，得到与指令最相关的空间证据，作为 boost 信号送入 [[DiT]]。

**符号说明**:
- $Q = X_{vl}$: 来自 [[VLM]] 的 token 序列
- $K, V = \tilde{M}^t$: 经过动态刷新的记忆 token 序列
- $C$: 注意力头维度
- $X_{\text{boost}}$: 上下文化的空间检索结果，注入主干 [[DiT]]

### 公式5: [[VLA]] 动作生成（含记忆 boost）

$$
a_{t:t+k} = \pi_\theta\!\left(o_t, s_t, l,\ X_{\text{boost}}(\tilde{M}^t)\right)
$$

**含义**: 在标准 VLA 动作头的基础上，把空间记忆检索结果显式地作为额外条件输入，使策略能引用视野外信息。

**符号说明**:
- $o_t, s_t, l$: 当前视觉观测、本体状态、语言指令
- $\tilde{M}^t$: 经动态刷新的全局记忆
- $a_{t:t+k}$: 长度 $k$ 的 [[Action Chunking|动作块]]

---

## 关键图表

### Figure 1: OOV 问题示意

![Figure 1](https://arxiv.org/html/2605.22283v1/x1.png)

**说明**: 现有 [[VLA]] 的"反应式感知"在目标移出视野时崩溃——头部相机帧里看不到杯子，策略就无法规划。本图通过对比凸显 [[OOV (Out-of-Vision)|OOV]] 设置和标准 in-view 设置的差异，并展示 SOMA 利用空间记忆"记住目标位置"的关键能力。

### Figure 2: SOMA 框架总览

![Figure 2](https://arxiv.org/html/2605.22283v1/x2.png)

**说明**: 三阶段流水线 —— (A) **Spatial Memory Construction**：扫描序列经 [[VGGT]] + [[YOLO-World]] + [[DINOv3]] 构建全局物体级记忆 $M_0$；(B) **Dynamic Memory Refinement**：操作时用相似度-融合双门控更新；(C) **Contextual Memory Retrieval**：[[VLM]] token 作 query 跨注意力检索，输出 boost 注入 [[DiT]] 主干。

### Figure 3: 真机 OOV 任务集

![Figure 3](https://arxiv.org/html/2605.22283v1/x3.png)

**说明**: 五个真实世界 PnP 任务的实拍：(1) Invisible→Invisible PnP、(2) Visible→Invisible PnP、(3) Invisible→Visible PnP、(4) Sequential Dual-Object PnP、(5) Dual-Arm Coordination PnP。任务难度递增，目标物体可见性逐步降低。

### Figure 4: 真机性能对比

![Figure 4](https://arxiv.org/html/2605.22283v1/x4.png)

**说明**: SOMA 与 [[StarVLA]]、[[SpatialVLA]]、[[GR00T N1.5]] 在五个 OOV 任务上的成功率柱状图。SOMA 在所有任务上稳定领先，最难的双臂任务上优势最明显（GR00T 几乎为 0，SOMA 仍能完成）。

### Figure 5: 真机执行序列

![Figure 5](https://arxiv.org/html/2605.22283v1/x5.png)

**说明**: 五个任务的关键帧轨迹。可以看到 SOMA 的头部主动扫描 → 定位目标 → 一次抓取的清晰路径，相比 baseline 的"反复摇头"行为差异显著。

### Figure 6: 自研双臂头主动机器人

![Figure 6](https://arxiv.org/html/2605.22283v1/x6.png)

**说明**: 硬件平台：双 7-DoF Realman ZM73 + 自适应夹爪 + 2-DoF 主动头（pan-tilt） + Intel RealSense D435（头 + 腕） + NVIDIA Jetson AGX Orin 64GB。主动头是 SOMA 收集多视角扫描序列的关键。

### Figure 7: VR 遥操作系统

![Figure 7](https://arxiv.org/html/2605.22283v1/img/VR.png)

**说明**: 用 Meta Oculus Quest 3 做数据采集的遥操作平台，每个任务收集 400 条专家演示，作为微调数据。

### Table 1: 行为指标对比（GR00T N1.5 vs SOMA，五个真机任务）

| 指标 | 任务 1 | 任务 2 | 任务 3 | 任务 4 | 任务 5 |
|------|--------|--------|--------|--------|--------|
| First-Fixation Time (s) ↓ | 7.6 → **4.2** (-45%) | 21.0 → **12.7** (-40%) | 14.8 → **8.2** (-45%) | 10.9 → **4.9** (-55%) | 11.5 → **4.7** (-59%) |
| Head Search Path (deg) ↓ | 50.5 → **27.8** (-45%) | 51.0 → **28.1** (-45%) | 83.8 → **50.3** (-40%) | 109.2 → **54.6** (-50%) | 164.0 → **70.4** (-57%) |
| Viewpoint Correction ↓ | 1.6 → **0.9** (-44%) | 1.9 → **1.1** (-42%) | 1.4 → **0.8** (-43%) | 3.4 → **1.7** (-50%) | 5.3 → **2.3** (-57%) |
| Grasp Attempt Count ↓ | 1.8 → **1.0** (-44%) | 2.0 → **1.2** (-40%) | 1.7 → **1.0** (-41%) | 2.4 → **1.2** (-50%) | 3.7 → **1.6** (-57%) |
| Time-to-Grasp (s) ↓ | 58.0 → **32.3** (-44%) | 30.0 → **16.8** (-44%) | 50.0 → **29.7** (-41%) | 65.5 → **30.4** (-54%) | 36.5 → **14.6** (-60%) |

**说明**: 所有五个行为指标在所有五个任务上一致下降 40–60%，最显著的是难度最高的双臂任务（Task 5），头部搜索路径缩短 57%，证明空间记忆**真的减少了盲目搜索**。

### Table 2: 扫描 vs 记忆的双因子消融

| OOV Task | Scan + GR00T | No-Scan SOMA | Scan-only SOMA | **Full SOMA** |
|----------|-------------|-------------|---------------|---------------|
| Task 1 | 19.0 | 20.0 | 25.0 | **30.0** |
| Task 2 | 22.0 | 24.0 | 29.0 | **35.0** |
| Task 3 | 16.0 | 17.5 | 22.5 | **27.5** |
| Task 4 | 25.0 | 26.0 | 30.0 | **32.5** |
| Task 5 | 10.5 | 11.7 | 14.2 | **16.7** |
| **平均 SR (%)** | 18.5 | 19.8 | 24.1 | **28.3** |

**说明**: "只给 GR00T 扫描视频"几乎没用（18.5%），"只用空间记忆但不扫描"也只有 19.8%，**扫描 + 记忆的乘积效应**才把 SR 提到 28.3%。这说明 SOMA 的关键不是哪个单独模块，而是"主动扫描收集 → 空间-语义对齐 → 任务条件检索"的完整闭环。

### Table 3: RoboCasa Tabletop GR1（Container Interaction 类别）

| Method | 30 Demos | 100 Demos | 300 Demos | Full |
|--------|----------|-----------|-----------|------|
| [[Diffusion Policy]] | 35.1 | 49.5 | 54.2 | — |
| StarVLA | 1.4 | 1.1 | 1.8 | 3.0 |
| [[GR00T N1.5]] | 35.3 | 33.7 | 35.7 | 40.3 |
| **SOMA** | **52.3** | **52.0** | **53.3** | **55.6** |

**说明**: 在低数据 regime (30 demos) 上 SOMA 大幅领先 [[GR00T N1.5]] 17 个点，说明空间记忆**提供的 inductive bias 显著降低样本复杂度**。Cooking / Tabletop Serving / Dish Transfer / Tray Organization 类别表现相似（详见原文）。

### Table 4a: [[SimplerEnv]] - Visual Matching 设置

| Model | Pick Coke Can | Move Near | Open/Close Drawer | Average |
|-------|---------------|-----------|-------------------|---------|
| Moto | 74.0 | 60.4 | 43.1 | 59.2 |
| RoboVLM | 77.3 | 61.7 | 43.5 | 60.6 |
| TraceVLA | 28.0 | 53.7 | 57.0 | 42.0 |
| [[Pi05\|π₀.₅]] | 72.7 | 65.3 | 38.3 | 58.8 |
| [[Pi05\|π₀.₅]] + FAST | 75.3 | 67.5 | 42.9 | 61.9 |
| [[OpenVLA-OFT]] | 72.3 | 69.6 | 47.2 | 63.0 |
| [[GR00T N1.5]] | 47.0 | 70.0 | 18.1 | 45.0 |
| **SOMA (Ours)** | **85.0** | **73.0** | 31.5 | **63.2** |

**说明**: SOMA 在 Visual Matching 平均 63.2% 超过所有 SOTA（包括 [[OpenVLA-OFT]] 63.0%），抽屉任务弱于 [[OpenVLA-OFT]] 是因为 SOMA 的几何先验对"接触受限"任务收益较小。

### Table 4b: [[SimplerEnv]] - Variant Aggregation 设置

| Model | Pick Coke Can | Move Near | Open/Close Drawer | Average |
|-------|---------------|-----------|-------------------|---------|
| [[OpenVLA]] | 54.5 | 47.7 | 17.7 | 39.8 |
| RoboVLM | 75.6 | 60.0 | 10.6 | 51.3 |
| TraceVLA | 60.0 | 56.4 | 31.0 | 45.0 |
| [[OpenVLA-OFT]] | 65.3 | 59.0 | 12.2 | 45.5 |
| [[GR00T N1.5]] | 46.7 | 62.9 | 17.5 | 42.4 |
| **SOMA (Ours)** | 55.5 | **76.6** | **25.4** | **52.5** |

**说明**: Variant Aggregation 引入分布偏移，对鲁棒性要求更高。SOMA 平均 52.5%，比次优 RoboVLM 高 1.2 pp，说明**空间记忆对视角扰动天然鲁棒**——记忆是 3D 的，不绑定特定相机视角。

### Table 5: 记忆组件消融（GR1 Benchmark）

| Ablation | w/o Geo. cues | w/o Object Semantics | w/o Dynamic Update | **Full** |
|----------|--------------|---------------------|---------------------|----------|
| Overall Success (%) | 45.1 | 43.7 | 41.5 | **49.3** |

**关键发现**: 三个组件**都不可缺**，但**动态更新最关键**（-7.8 pp），其次是语义（-5.6 pp），再是几何（-3.7 pp）。说明记忆不能是"扫一次定终身"，必须在操作中持续刷新；同时语义信息比纯几何更重要，因为物体类别决定了 VLM 该 retrieve 哪些 token。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| OXE | 大规模 | Open X-Embodiment 综合机器人数据 | 预训练 |
| Fractal | Google Robot | SimplerEnv 对应的真机迁移源 | 微调 ([[SimplerEnv]]) |
| RoboCasa Tabletop GR1 | 5 类任务 × 30/100/300/Full | 仿真桌面操作 benchmark | 训练 + 评测 |
| 自采真机 PnP | 5 任务 × 400 demo | 自研双臂 + 主动头 + VR 遥操作 | 真机 OOV 评测 |

### 实现细节

- **Backbone**: [[GR00T N1.5]]（VLM 主干 + [[DiT]] 动作头）
- **感知专家**: [[VGGT]]（冻结）+ [[YOLO-World]]（冻结）+ [[DINOv3]]（冻结）
- **优化器**: 详见原文（标准 AdamW 风格）
- **Batch Size**: 60
- **训练步数**: 30,000
- **硬件**: 32 × NVIDIA H200 训练；推理 1 × RTX 4090
- **推理延迟**: 详见 Appendix D.6（可达实时控制频率）

### 真机硬件

- 双 7-DoF Realman ZM73 + 自适应夹爪
- 2-DoF 主动头（pan-tilt 电机）
- Intel RealSense D435 RGB-D × 3（头 + 双腕）
- 控制板：NVIDIA Jetson AGX Orin 64GB

### 可视化结果

定性来看（Figure 5），SOMA 表现出**目的性的头部运动**：在拿到指令后先做一段定向扫描（约 1–2 秒），随后头部稳定指向目标，与 baseline "持续扫视 + 反复试抓"的混乱行为形成鲜明对比。这与 Table 1 的行为指标完全一致。

---

## 批判性思考

### 优点

1. **问题非常精准**: 把"VLA 默认目标可见"这一被广泛忽视的隐含假设抽出来正面攻击，是过去一年记忆型 VLA 研究的重要补全（[[MemoryVLA]] / [[PrediMem]] 解决时间记忆，SOMA 解决空间记忆）。
2. **构件几乎都是冻结的预训练模型**: [[VGGT]] / [[YOLO-World]] / [[DINOv3]] 全冻结，训练成本可控，且预训练知识不会被小规模机器人数据"训坏"。
3. **行为指标新颖且可复现**: 五个行为指标（注视时间、搜索路径、修正次数、抓取次数、抓取耗时）远比单一成功率更能反映"是不是真的减少了盲目搜索"。
4. **多 benchmark 验证**: 真机 + RoboCasa + SimplerEnv 三套互补评测，结论稳健。

### 局限性

1. **依赖主动头**: 没有可动相机的固定臂平台无法直接受益，这限制了部署面（如桌面单臂工业机器人）。
2. **扫描成本**: 每次任务开始前需要一段预扫描，对短指令长尾任务不经济；论文也没讨论扫描-操作延迟权衡。
3. **[[VGGT]] 的几何精度上限**: VGGT 在弱纹理 / 反光 / 透明物体上的位姿估计本就不稳，会直接限制空间记忆质量（与 [[空间记忆专家]] 在缺纹理环境下的局限同源）。
4. **没有处理动态物体（人手介入）**: Dynamic Memory Refinement 用 EMA 软更新，但物体被人猛地搬走时，EMA 滞后会导致策略仍指向旧位置。
5. **代码未释出**: 论文写作时间是 2026 年 5 月，"Code will be released soon" 短期内无法复现细节。

### 潜在改进方向

1. **结合 [[关键帧记忆库]]**: 当前 SOMA 只有空间记忆没有时间关键帧；与 [[PrediMem]] 风格的关键帧组合可能在长程任务上更强。
2. **可微分扫描策略**: 扫描轨迹目前是预定义的，可改成可学习的"主动感知策略"，端到端联合优化。
3. **支持 transparent / reflective 物体**: 用 [[3D Gaussian Splatting]] 或基于扩散的几何补全替换 VGGT，处理 VGGT 弱项。
4. **轻量化 student**: 三个感知专家加在一起算力不小，蒸馏到单一 student backbone 有望部署到边缘平台。

### 可复现性评估

- [ ] 代码开源（声明 soon，未释出）
- [ ] 预训练模型
- [x] 训练细节（基本完整：32×H200, BS 60, 30k steps）
- [x] 数据集可获取（OXE / Fractal / RoboCasa 公开；真机 PnP 自采，需自己构建）

---

## 关联笔记

### 基于

- [[VGGT]]: 提供全局一致的相机位姿和场景几何
- [[DINOv3]]: 高维语义外观特征
- [[YOLO-World]]: 开放词汇 2D 检测
- [[GR00T N1.5]]: VLA 主干

### 对比

- [[MemoryVLA]]: token-level 时间记忆，对比反映 SOMA 的"显式 3D"优势
- [[PrediMem]]: 用 [[关键帧记忆库]] + 预测编码做时间记忆，与 SOMA 的空间记忆互补
- [[OpenVLA-OFT]]: 当前 [[SimplerEnv]] 强基线
- [[Pi05|π₀.₅]]: 经典 VLA 基线
- [[Diffusion Policy]]: 在 RoboCasa 上的非 VLA 强基线

### 方法相关

- [[空间记忆专家]]: 同样把空间信号注入策略，但走的是 SLAM/位姿先验路线；SOMA 走的是物体级 3D token 路线
- [[关键帧记忆库]]: 显式时间记忆，与 SOMA 的空间记忆形成互补轴
- [[Cross-Attention]]: 实现 contextual memory retrieval
- [[Action Chunking]]: SOMA 输出动作块
- [[DiT]]: 动作生成主干
- [[EMA]]: 用于动态记忆刷新

### 硬件 / 数据相关

- [[RoboCasa]]: 主要仿真 benchmark
- [[SimplerEnv]]: 跨机器人迁移 benchmark
- 自研双臂 + 主动头平台（Realman ZM73 × 2 + 2-DoF head + RealSense D435 × 3）

---

## 速查卡片

> [!summary] SOMA: Spatial Memory for Out-of-Vision Manipulation
> - **核心**: 用 [[VGGT]] + [[DINOv3]] + [[YOLO-World]] 构 3D 物体级空间记忆，让 VLA 处理视野外目标
> - **方法**: 三阶段 Construction → Refinement → Retrieval，[[Cross-Attention]] 注入 [[DiT]]
> - **结果**: 真机 OOV 行为指标降 40-60%，SimplerEnv-VM 63.2% 第一，RoboCasa Container 300-demo 53.3% 远超 GR00T 35.7%
> - **代码**: Code will be released soon

---

*笔记创建时间: 2026-05-23*
