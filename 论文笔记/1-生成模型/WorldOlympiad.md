---
title: "WorldOlympiad: Can Your World Model Survive a Triathlon?"
method_name: "WorldOlympiad"
authors: [Yuke Zhao, Wangbo Zhao, Weijie Wang, Zeyu Zhang, Dakai An, Akide Liu, Yinghao Yu, Jiasheng Tang, Fan Wang, Wei Wang, Bohan Zhuang]
year: 2026
venue: arXiv
tags: [world-model-benchmark, video-generation-evaluation, physical-faithfulness, geometric-consistency, interaction-fidelity]
zotero_collection: 1-生成模型
image_source: online
arxiv_html: https://arxiv.org/html/2606.11129v1
created: 2026-06-12
---

# 论文笔记：WorldOlympiad: Can Your World Model Survive a Triathlon?

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Zhejiang University, DAMO Academy (Alibaba), HKUST, Monash University |
| 日期 | June 2026 |
| 项目主页 | [WorldOlympiad](https://alibaba-damo-academy.github.io/WorldOlympiad) |
| 对比基线 | [[VBench]] / [[EWMBench]] / [[WorldEval]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.11129) / [Code](https://github.com/alibaba-damo-academy/WorldOlympiad) |

---

## 一句话总结

> WorldOlympiad 是首个从物理合规性、三维一致性、交互保真度三个维度对视频世界模型进行综合评测的"三项全能"基准，揭示当前世界模型在视觉质量掩盖下的深层结构性缺陷。

---

## 核心贡献

1. **三轨评测体系**: 构建物理赛道、几何赛道、交互赛道三条独立评测轨道，覆盖 [[世界模型]] 生成的本质能力维度。
2. **1000 视频高质量数据集**: 涵盖游戏、机器人、现实世界三个场景，1000 条长视频配备精细的动作-字幕时序标注。
3. **强人类对齐验证**: Spearman 相关系数 ρ=0.95，证明自动评分与人类偏好高度一致，实现可扩展的无人工评测。

---

## 问题背景

### 要解决的问题

现有 [[视频生成]] 评测指标（如 [[FID]]、[[FVD]]、PSNR）仅关注视觉外观质量，无法诊断世界模型是否真正理解物理规律、三维结构和长时序交互逻辑。

### 现有方法的局限

- [[VBench]]、[[VBench++]]、[[VBench 2.0]] 专注于单轮视频质量，不评测动作条件生成。
- [[EWMBench]]、[[WorldEval]]、[[WorldArena]] 聚焦于特定场景（游戏或机器人），缺乏跨域泛化评测。
- [[MIND]] 等方法不覆盖 3D 几何一致性这一关键维度。
- 所有现有 benchmark 均不评测长视频滚动生成中的状态一致性（transition smoothness、global coherence）。

### 本文的动机

视频生成模型可以产生主观上"看起来不错"的输出，但仍可能违背重力、忽略材料属性、无法在相邻 chunk 间保持对象状态——这些缺陷在视觉质量指标下完全隐形。WorldOlympiad 采用 [[高斯泼溅|Gaussian Splatting]] 3D 重建和 [[MLLM]] 物理合规判断，将"看起来对"与"物理上对""结构上对""交互上对"解耦。

---

## 方法详解

### 数据集构成

WorldOlympiad 包含 **1000 条高质量长视频**，来自三个场景：

| 场景 | 数量 | 来源 | 筛选方式 |
|------|------|------|----------|
| Gaming | 400 | GameGen-X | 随机采样，长视频分段 |
| Robotics | 400 | RoboCOIN bimanual manipulation | 人工筛选 |
| Real-world | 200 | LVD-2M | 时长 >60s，运动分数 >50 |

### 三轨评测框架

#### 赛道一：物理赛道（Physical Track）

**设计动机**：利用 [[MLLM]] 作为评判员，评估生成视频是否遵守物理定律，替代人工标注实现可扩展评测。

**评测维度**：
- **力学（Mechanics）**：重力、浮力、压缩、碰撞
- **热力学（Thermodynamics）**：熔化、升华、汽化、凝结、沉积、冷冻
- **材料属性（Material Properties）**：混色、溶解度、硬度、燃烧性

**评分流程**：
1. **相关性分数（Relevance）**：判断生成视频与参考视频在查询物理属性上是否可比较
2. **合规分数（Compliance）**：评估生成视频是否符合对应物理规律

**关键发现**：力学合规性最强（顶级模型 0.8-1.0），热力学相变（熔化、汽化）是最薄弱环节（多数模型 0.0-0.6）。

#### 赛道二：几何赛道（Geometry Track）

**设计动机**：通过 [[高斯泼溅|3D Gaussian Splatting]] 重建，检测世界模型生成的视频是否具有自洽的三维结构——视觉上合理但 3D 不一致的生成无法成功重建。

**评分体系**：

- $S_{\mathrm{recon}}$：重建质量分（Gaussian splatting 保真度）
- $S_{\mathrm{meta}}$：元视角分（渲染诊断视角的图像质量）
- $S_{\mathrm{traj}}$：轨迹分（相机运动一致性）

最终 3D 综合分为三项等权平均（见公式部分）。

**关键发现**：最优模型 Hunyuan-WorldPlay 仅得 0.424，揭示视觉质量高不等于 3D 结构一致——"geometry-simulation gap"。

#### 赛道三：交互赛道（Interaction Track）

**设计动机**：评估模型在分 chunk 滚动生成长视频时，是否能持续响应动作指令并保持对象状态。

**三层评分**：
- **Chunk 级（$S_{\mathrm{chunk}}$）**：每段视频内字幕语义对齐
- **过渡平滑度（$S_{\mathrm{trans}}$）**：相邻 chunk 之间的连贯性
- **全局一致性（$S_{\mathrm{global}}$）**：全视频长时序连贯

[[MLLM]] 评分与 [[CLIP]] 语义对齐加权融合：
- $\lambda = 0.1$（CLIP 权重）
- 每个 chunk 包含 $m_i$ 帧，对应动作字幕 $p_i$

**关键发现**：状态重置（对象在 chunk 边界消失/复位）是最常见失败模式，transition 分数显著低于 chunk 分数。

### 数据标注流水线

三阶段时序分块与字幕生成：

- **Stage I**：识别视频中的连续执行区间（动作不间断的时间段）
- **Stage II**：为每个 chunk 生成动作标签和详细字幕
- **Stage III**：结合全视频上下文精化字幕，确保时序连贯性

---

## 关键公式

### 公式一：[[高斯泼溅|3D Gaussian Splatting 重建流程]]

$$
\mathcal{F}_{\mathrm{DA3}}(\bar{V}) \rightarrow (\mathcal{G}, \{E_i, K_i\}_{i=1}^{N})
$$

$$
\hat{V}_{\mathrm{GS}} = \mathcal{R}(\mathcal{G}, \{E_i, K_i\}_{i=1}^{N})
$$

$$
\hat{I}_{\mathrm{meta}} = \mathcal{R}(\mathcal{G}, E_{i^*}, K_{i^*})
$$

**含义**：将生成视频 $\bar{V}$ 输入 DA3 重建器，得到 3D Gaussian 场 $\mathcal{G}$ 和各帧相机外参/内参 $\{E_i, K_i\}$；再从任意视角渲染重建视频 $\hat{V}_{\mathrm{GS}}$ 和元视角图像 $\hat{I}_{\mathrm{meta}}$。

**符号说明**：
- $\mathcal{F}_{\mathrm{DA3}}$：DA3 3D Gaussian Splatting 重建函数
- $\bar{V}$：生成的视频
- $\mathcal{G}$：3D Gaussian 场
- $E_i, K_i$：第 $i$ 帧的相机外参与内参
- $\hat{V}_{\mathrm{GS}}$：重建后再渲染的视频
- $\hat{I}_{\mathrm{meta}}$：诊断元视角图像
- $E_{i^*}, K_{i^*}$：选定元视角的相机参数

### 公式二：[[3D一致性评分|几何赛道子分计算]]

$$
S_{\mathrm{recon}} = \mathrm{clamp}(J_{\mathrm{vid}}(\hat{V}_{\mathrm{GS}}, p), 0, 1)
$$

$$
S_{\mathrm{meta}} = \mathrm{clamp}(J_{\mathrm{img}}(\hat{I}_{\mathrm{meta}}, p), 0, 1)
$$

**含义**：用 MLLM 视频/图像评判函数对重建结果打分，clamp 到 $[0,1]$。

**符号说明**：
- $J_{\mathrm{vid}}$：MLLM 视频质量评判函数
- $J_{\mathrm{img}}$：MLLM 图像质量评判函数
- $p$：评测提示词（物理/几何属性描述）
- $S_{\mathrm{recon}}$：重建质量分
- $S_{\mathrm{meta}}$：元视角质量分

### 公式三：[[相机轨迹对齐|轨迹评分归一化]]

$$
\tilde{T}_i = T_1^{-1}T_i, \quad \tilde{\hat{T}}_i = \hat{T}_1^{-1}\hat{T}_i
$$

$$
S_{\mathrm{traj}} = A_{\mathrm{motion}}(S_t, S_r; \{\tilde{T}_i\}_{i=1}^{L})
$$

**含义**：将所有相机位姿归一化到第一帧坐标系，再计算参考轨迹与生成轨迹的平移/旋转对齐分。

**符号说明**：
- $T_i, \hat{T}_i$：第 $i$ 帧的参考/生成相机位姿矩阵
- $\tilde{T}_i, \tilde{\hat{T}}_i$：归一化后的位姿
- $A_{\mathrm{motion}}$：运动对齐评分函数
- $S_t, S_r$：平移与旋转一致性分

### 公式四：[[3D综合评分|3D 赛道总分]]

$$
S_{3D} = \frac{1}{3}(S_{\mathrm{recon}} + S_{\mathrm{meta}} + S_{\mathrm{traj}})
$$

**含义**：重建质量、元视角质量、轨迹一致性三项等权平均，得到最终 3D 综合分。

**符号说明**：
- $S_{\mathrm{recon}}$：重建保真度分
- $S_{\mathrm{meta}}$：诊断视角渲染分
- $S_{\mathrm{traj}}$：相机轨迹对齐分

### 公式五：[[CLIP语义对齐|交互赛道 CLIP 评分]]

$$
s_i^{\mathrm{clip}} = \frac{1}{m_i}\sum_{j=1}^{m_i}\mathrm{sim}(\mathrm{CLIP}_v(f_{i,j}), \mathrm{CLIP}_t(p_i))
$$

$$
S_{\mathrm{clip}} = \frac{\sum_{i=1}^{T}\sum_{j=1}^{m_i}\mathrm{sim}(\mathrm{CLIP}_v(f_{i,j}), \mathrm{CLIP}_t(p_i))}{\sum_{i=1}^{T}m_i}
$$

$$
\widetilde{S}_{\mathrm{clip}} = \mathrm{clip}\!\left(\frac{S_{\mathrm{clip}} - \tau_{\min}}{\tau_{\max} - \tau_{\min}},\ 0,\ 1\right)
$$

**含义**：计算每帧视觉 [[CLIP]] 嵌入与对应动作字幕文本嵌入的相似度，求全局加权平均后线性归一化到 $[0,1]$。

**符号说明**：
- $f_{i,j}$：第 $i$ 个 chunk 第 $j$ 帧
- $m_i$：第 $i$ 个 chunk 的帧数
- $p_i$：第 $i$ 个 chunk 的动作字幕
- $\mathrm{CLIP}_v, \mathrm{CLIP}_t$：CLIP 视觉/文本编码器
- $\tau_{\min}, \tau_{\max}$：CLIP 相似度的归一化边界值
- $T$：总 chunk 数

### 公式六：[[MLLM评分|交互赛道 MLLM 三层评分]]

$$
S_{\mathrm{chunk}} = \frac{1}{5T}\sum_{i=1}^{T}a_i, \quad S_{\mathrm{trans}} = \frac{1}{5(T-1)}\sum_{i=1}^{T-1}b_i, \quad S_{\mathrm{global}} = \frac{g}{5}
$$

$$
S_{\mathrm{mllm}} = \frac{1}{3}(S_{\mathrm{chunk}} + S_{\mathrm{trans}} + S_{\mathrm{global}})
$$

**含义**：MLLM 分别从 chunk 级字幕对齐、过渡平滑度、全局一致性三层打分（满分 5 分），取等权平均。

**符号说明**：
- $a_i \in [0,5]$：第 $i$ 个 chunk 的 MLLM 评分
- $b_i \in [0,5]$：第 $i$ 个 transition 的平滑度评分
- $g \in [0,5]$：全视频全局一致性评分
- $T$：chunk 总数

### 公式七：[[交互保真度评分|交互赛道总分]]

$$
S_{\mathrm{interact}} = (1-\lambda)S_{\mathrm{mllm}} + \lambda\widetilde{S}_{\mathrm{clip}}
$$

**含义**：MLLM 语义评分（高权重）与 CLIP 对齐分（低权重）加权融合，$\lambda=0.1$。

**符号说明**：
- $\lambda = 0.1$：CLIP 分权重
- $S_{\mathrm{mllm}}$：MLLM 三层评分均值
- $\widetilde{S}_{\mathrm{clip}}$：归一化 CLIP 对齐分

### 公式八：[[世界模型综合评分|WorldOlympiad 总分]]

$$
S_{\mathrm{all}} = \frac{1}{3}(S_{\mathrm{phys}} + S_{3D} + S_{\mathrm{interact}})
$$

**含义**：物理、几何、交互三个赛道等权平均，得到最终综合评分。

**符号说明**：
- $S_{\mathrm{phys}}$：物理赛道分
- $S_{3D}$：几何赛道分
- $S_{\mathrm{interact}}$：交互赛道分

---

## 关键图表

### Figure 1: WorldOlympiad 整体流程

![Figure 1](https://arxiv.org/html/2606.11129v1/x1.png)

**说明**：WorldOlympiad 的完整 pipeline 概览。左侧为数据采集（游戏/机器人/现实世界三个场景），中间为长视频生成（[[MLLM]] 辅助的分块动作字幕标注），右侧为三轨评测（物理、几何、交互赛道）。

### Figure 2: 数据采集概览

![Figure 2](https://arxiv.org/html/2606.11129v1/x2.png)

**说明**：三类视频来源的数据采集流程。展示 GameGen-X（游戏）、RoboCOIN（机器人双臂操作）、LVD-2M（现实世界长视频）的数据特征与筛选标准。

### Figure 3: 数据标准化流水线

![Figure 3](https://arxiv.org/html/2606.11129v1/x3.png)

**说明**：从原始视频到精细动作-字幕标注的三阶段处理流程：时序分块识别 → chunk 字幕生成 → 全局上下文精化。确保每个 chunk 的字幕在全视频时序中自洽。

### Figure 4: 数据流水线统计

![Figure 4](https://arxiv.org/html/2606.11129v1/x4.png)

**说明**：数据处理的数量统计，涵盖处理阶段覆盖率、标注完成率、最终评测可用样本数量。

### Figure 5: WorldOlympiad 主要评测结果

![Figure 5](https://arxiv.org/html/2606.11129v1/x5.png)

**说明**：8 个被评测世界模型在三条赛道上的得分统计。雷达图或柱状对比图，揭示各模型的能力分布差异：顶级模型物理分高（~0.94）但 3D 分仍不及 0.45，交互分中等（~0.73）。

### Figure 6: 典型案例分析

![Figure 6](https://arxiv.org/html/2606.11129/2606.11129v1/x6.png)

**说明**：WorldOlympiad 检测到的典型成功案例与失败案例对比。成功案例展示物理行为保持完好的高质量生成；失败案例包括违反重力约束、几何不一致（3D 重建崩溃）、chunk 边界状态重置等典型错误。

### Table 1: Benchmark 横向对比

| Benchmark | 长视频 | 物理评测 | 几何评测 | 交互评测 | Gaming | Robotics | Real-world |
|-----------|--------|----------|----------|----------|--------|----------|------------|
| VBench | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| VBench++ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| EWMBench | ✗ | 部分 | ✗ | 部分 | ✓ | ✓ | ✗ |
| WorldEval | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ |
| MIND | ✗ | 部分 | ✗ | ✓ | ✗ | ✓ | ✗ |
| **WorldOlympiad** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

**说明**：WorldOlympiad 是唯一覆盖长视频、三条评测轨道和三个应用场景的全维度基准。

### Table 2: 主要评测结果（8 模型综合排名）

| 模型 | 物理（$S_{\mathrm{phys}}$） | 3D（$S_{3D}$） | 交互（$S_{\mathrm{interact}}$） | 总分（$S_{\mathrm{all}}$） | 排名 |
|------|------|------|------|------|------|
| LingBot-World | **0.942** | 0.373 | **0.734** | **0.683** | 1 |
| Cosmos-Predict-2.5 | 0.906 | **0.399** | 0.707 | 0.671 | 2 |
| Rolling Forcing | 0.873 | 0.321 | 0.636 | 0.610 | 3 |
| Yume-1.5 | 0.863 | 0.301 | 0.649 | 0.604 | 4 |
| LongLive | 0.863 | 0.363 | 0.526 | 0.584 | 5 |
| Hunyuan-WorldPlay | 0.692 | **0.424** | 0.316 | 0.477 | 6 |
| WoW | 0.708 | 0.250 | 0.345 | 0.434 | 7 |
| Matrix-Game 2.0 | 0.325 | 0.255 | 0.113 | 0.231 | 8 |

**说明**：LingBot-World 凭借极高的物理合规性（0.942）和交互保真度（0.734）排名第一；所有模型的 3D 几何分普遍偏低（最高仅 0.424），揭示世界模型的最大短板。

### Table 3: 各场景细分结果

| 模型 | Gaming 综合 | Robotics 综合 | Real-world 综合 |
|------|------------|--------------|----------------|
| LingBot-World | 0.676 | 0.684 | 0.688 |
| Cosmos-Predict-2.5 | 0.636 | 0.712 | 0.664 |
| Hunyuan-WorldPlay | — | 0.600（3D） | 0.235 |
| Matrix-Game 2.0 | 0.211 | — | — |

**说明**：LingBot-World 跨三个场景最为均衡；Matrix-Game 2.0 在自身专长场景（Gaming）表现也差，说明单域特化训练不保证域内泛化；Hunyuan-WorldPlay 在机器人场景 3D 分意外强，但在真实世界场景急剧下降。

---

## 实验

### 数据集

| 数据集 | 规模 | 场景 | 特点 |
|--------|------|------|------|
| GameGen-X | 400 视频 | Gaming | 游戏视频，动作多样 |
| RoboCOIN | 400 视频 | Robotics | 双臂操作，精细物理交互 |
| LVD-2M | 200 视频 | Real-world | 时长 >60s，高运动分数 |

### 实现细节

- **物理评判模型**：[[MLLM]]（具体模型未披露）
- **3D 重建工具**：DA3 [[高斯泼溅|Gaussian Splatting]] 框架
- **语义对齐**：[[CLIP]] ViT 系列
- **标注工具**：GPT-4 系列用于三阶段字幕精化
- **人类评测**：独立人工排名，Spearman ρ=0.95 与自动评分对齐

### 可视化结果

- 顶级模型（LingBot-World）在力学上（重力、碰撞）表现出色，但在热力学相变场景（如冰块融化）中普遍失效
- 状态重置是交互赛道最频繁的失败模式：chunk 边界处对象消失或复位
- 3D 重建失败通常表现为场景几何"爆炸"——即 Gaussian 场无法收敛，揭示生成视频内部存在深层几何矛盾

---

## 批判性思考

### 优点

1. **三维正交设计**：物理/几何/交互三条赛道相互独立，每条赛道均有自动化评测方案，避免维度间混淆。
2. **跨域覆盖**：同时评测游戏、机器人、现实场景，是目前覆盖最广的世界模型 benchmark。
3. **强人类对齐**：ρ=0.95 的 Spearman 相关系数是评测方法可信度的坚实证据。

### 局限性

1. **3D 重建依赖**：几何赛道的评测依赖 [[高斯泼溅|Gaussian Splatting]] 重建质量，对相机运动轨迹的假设可能不适用于所有生成场景（如无参考轨迹的纯生成视频）。
2. **MLLM 判断偏差**：物理和交互赛道均依赖 MLLM 评判，MLLM 自身的物理知识盲区（尤其是热力学）可能影响评分公正性。
3. **数据规模限制**：1000 条视频仅是初步规模，特别是真实世界场景仅 200 条，统计显著性有限。
4. **内存机制缺失**：作者指出 KV-cache 复用、显式 3D 记忆、线性注意力等架构变体的贡献尚未被单独分析。

### 潜在改进方向

1. 针对记忆机制（KV-cache vs. 3D Scene Memory vs. [[线性注意力]]）进行受控消融实验。
2. 扩展热力学和材料属性测试用例数量，减少 MLLM 判断的统计噪声。
3. 引入动态难度分级，区分"基础物理"与"复杂多体交互"。

### 可复现性评估

- [x] 代码开源（GitHub 已公开）
- [x] 预训练模型（评测数据集公开）
- [x] 训练细节完整（评测流程详述）
- [x] 数据集可获取（三个来源数据均已公开）

---

## 关联笔记

### 基于

- [[VBench]]: 视频质量评测基线，WorldOlympiad 超越其评测范围
- [[高斯泼溅]]: 几何赛道的核心 3D 重建技术
- [[CLIP]]: 交互赛道的语义对齐工具

### 对比

- [[EWMBench]]: 同为世界模型评测，但不含 3D 维度且场景更窄
- [[WorldEval]]: 物理评测类似但不含几何和交互维度
- [[WorldArena]]: 竞技场式评测，缺乏结构化指标

### 方法相关

- [[世界模型]]: 核心评测对象
- [[MLLM]]: 物理/交互赛道的评判员
- [[视频生成]]: 被评测的生成模型类型

### 数据集相关

- [[GameGen-X]]: Gaming 场景数据来源
- [[RoboCOIN]]: Robotics 场景数据来源（双臂操作）
- [[LVD-2M]]: Real-world 场景数据来源

---

## 速查卡片

> [!summary] WorldOlympiad: Can Your World Model Survive a Triathlon?
> - **核心**: 首个三轨（物理/几何/交互）世界模型综合评测基准，揭示视觉质量掩盖下的结构性缺陷
> - **方法**: MLLM 物理判断 + 3D Gaussian Splatting 几何重建 + CLIP/MLLM 交互对齐
> - **结果**: 最优模型 LingBot-World 总分 0.683，3D 几何分全面低位（最高 0.424），物理分已达 0.9+ 但热力学普遍失效
> - **代码**: https://github.com/alibaba-damo-academy/WorldOlympiad

---

*笔记创建时间: 2026-06-12*
