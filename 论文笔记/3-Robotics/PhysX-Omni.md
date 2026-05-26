---
title: "PhysX-Omni: Unified Simulation-Ready Physical 3D Generation for Rigid, Deformable, and Articulated Objects"
method_name: "PhysX-Omni"
authors: [Ziang Cao, Yinghao Liu, Haitian Li, Runmao Yao, Fangzhou Hong, Zhaoxi Chen, Liang Pan, Ziwei Liu]
year: 2026
venue: arXiv
tags: [physical-3d-generation, simulation-ready-asset, vlm-based-generation, articulated-object, deformable-object, embodied-ai, sim-to-real]
zotero_collection: 3-Robotics
image_source: online
arxiv_html: https://arxiv.org/html/2605.21572v1
created: 2026-05-24
---

# 论文笔记：PhysX-Omni: Unified Simulation-Ready Physical 3D Generation for Rigid, Deformable, and Articulated Objects

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | S-Lab, Nanyang Technological University（Ziwei Liu 组） |
| 日期 | May 2026 |
| 项目主页 | https://physx-omni.github.io/ |
| 对比基线 | PhysX-Anything、TRELLIS、CLAY、3D-LLM、PartCrafter |
| 链接 | [arXiv](https://arxiv.org/abs/2605.21572) / [PDF](https://arxiv.org/pdf/2605.21572) / [HTML](https://arxiv.org/html/2605.21572v1) |

---

## 一句话总结

> 用 [[VLM]] 作 backbone，配合一种"模板化 RLE 体素切片"几何表征，一次性把刚体 / 可形变 / 铰接物体的几何 + 物理属性 + 运动学全部生成为可直接喂 [[MuJoCo]] / [[IsaacLab]] 的 simulation-ready 资产。

---

## 核心贡献

1. **统一物理 3D 生成框架**: 首次在单一框架内打通刚体、[[可形变物体]]、[[铰接物体]] 三类资产的"几何 + 物理属性 + 运动学"端到端生成，输出可直接导入物理仿真。
2. **VLM 友好的几何表征**: 提出 [[Template-based RLE]]——基于 z 轴体素切片 + 二维 [[Run-Length Encoding]] + 模板层共享，把高分辨率 3D 结构编码为 VLM 可处理的 token 序列且不引入额外特殊 token。
3. **PhysXVerse 数据集**: 首个通用 [[Simulation-Ready 3D Asset]] 数据集，8.7K+ 资产 / 2.9K+ 类别，含物理标注（绝对尺度、材料、affordance、运动学），融合 [[PhysXNet]] / [[PhysX-Mobility]] 共 42K+ 训练样本。
4. **PhysX-Bench**: 六维度评测基准（geometry / absolute scale / material / affordance / kinematics / function description），人类对齐 Spearman ≥ 0.8。

---

## 问题背景

### 要解决的问题

[[Embodied AI]] / [[sim-to-real]] / 机器人策略学习高度依赖**仿真就绪**的 3D 资产，但当前 3D 生成方法和"物理仿真"之间存在严重断层：生成模型只关心几何或外观，物理属性（质量、摩擦、关节、铰链）几乎全靠人工标注，根本无法直接导入仿真器跑机器人 policy。

### 现有方法的局限

1. **几何专用生成模型**（TRELLIS / CLAY / Hunyuan3D）: 只输出 mesh / 体素，没有物理属性，更没有铰接结构。
2. **类目专用生成模型**（CAGE / PartCrafter）: 只能生成单一类目（如柜子、椅子），不通用，且通常只覆盖刚体或铰接其中之一。
3. **PhysX-Anything 等早期物理 3D 工作**: 用文本/单视图回归物理属性，但几何质量很差（CD = 37.06）、尺度估计离谱（绝对尺度误差 298.19），更没有 deformable。
4. **VLM 直接预测体素的传统做法**: 体素序列过长，[[Tokenization]] 容易爆 context window；或者引入大量 special token 导致 VLM 训练不稳定。

### 本文的动机

如果把"3D 几何 + 物理属性"统一看作可被 [[Autoregressive 解码]] 的结构化序列，那 [[VLM]] 既有的视觉理解 + 长 context + 多模态对齐能力就能直接复用。难点是**几何表征要 VLM-friendly**：既要保留高分辨率几何细节，又要 token 数可控，还不能引入新 token。作者提出**模板共享的二维 RLE 切片**这一表征，配合"从全局到局部、从粗到细"的多轮推理范式，把统一物理 3D 生成转化成一个 [[Qwen3-VL|Qwen2.5-VL]] 微调问题。

---

## 方法详解

### 模型架构

PhysX-Omni 采用 **VLM-based 自回归生成 + 模板化 RLE 几何表征 + coarse-to-fine 多轮解码** 架构：

- **输入**: 单视图 / 多视图图像 $I$ + 可选文本指令 $l$
- **Backbone**: [[Qwen3-VL|Qwen2.5-VL-7B-Instruct]]，单一 VLM 同时承担"理解"和"生成"
- **核心模块**:
  - [[Template-based RLE]] —— 把 3D voxel grid 编码为 VLM 可读的 ASCII-like 序列
  - **树状层级表征** —— 用 [[Tree-structured 3D Representation|tree-structured]] 组织对象 → part 的层次，先全局后局部
  - [[TRELLIS]] decoder —— 把生成的体素表征解码回 mesh
- **输出**: 一棵描述整个对象的层级 token 树，每个节点包含
  - part-level **几何**（RLE 编码的体素切片）
  - **物理属性**: $\{m, \mu, \rho, s_{abs}, \text{material class}\}$（质量 / 摩擦 / 密度 / 绝对尺度 / 材料）
  - **affordance** 和 **part 描述**（自然语言）
  - **运动学**: 关节类型 / 关节轴 / 关节范围（用于铰接物体）
- **总参数**: ~7B（VLM 主体）+ TRELLIS decoder

### 核心模块

#### 模块 1: Template-based RLE 几何表征

**设计动机**: 直接把体素 flatten 成 token 会导致序列过长（$128^3 \approx 2M$ tokens），需要一种**对 VLM 友好且无信息丢失**的紧凑表征。

**具体实现**:

1. 对 part-level 分解后的 simulation-ready asset 进行 **[[体素化|Voxelization]]**（part 粒度，每个 part 独立体素网格）
2. 沿 **z 轴** 将体素切成 $H$ 张 2D 二值 mask $\{M_z\}_{z=1}^{H}$
3. 对每张 mask 做 [[Run-Length Encoding|2D RLE]]，得到 run 序列 $r_z = [(\text{val}_1, \text{len}_1), \dots]$
4. **模板共享**：相邻切片往往结构相似（同一物体的连续 z 层），引入若干**模板层** $\{T_k\}$，每个具体切片只存与最近模板的 **残差 RLE**
5. RLE 编码直接复用 VLM 词表中的数字 token，**无需引入任何 special token**

> 相比标准 2D RLE，模板共享带来约 $3-4\times$ token 压缩，配合 16K context length 足够编码完整 part-level 资产。

#### 模块 2: Coarse-to-Fine 多轮自回归生成

**设计动机**: 一次性生成完整资产的 token 树会让 VLM 失焦，把问题拆成"先全局后局部"更易学。

**具体实现**:

- **Stage 1 (Global)**: 输入图像 → VLM 推理对象类别、整体 [[Bounding Box]]、part 划分树
- **Stage 2 (Per-Part Geometry)**: 对每个 part 节点，VLM 自回归生成模板化 RLE 序列 → 体素 → 经 [[TRELLIS]] 解码为 mesh
- **Stage 3 (Physics)**: 对每个 part，VLM 生成 $\{material, \rho, m, s_{abs}, affordance\}$
- **Stage 4 (Kinematics)**: 对铰接树，VLM 预测关节类型（revolute / prismatic / fixed）、关节轴 $\hat{n}$、关节范围 $[\theta_{\min}, \theta_{\max}]$

#### 模块 3: PhysXVerse 数据 pipeline

- 几何来源：PartVerse（人工验证的 part 分解）
- 物理属性：[[VLM]] 初标 + human-in-the-loop verification
- 每个对象渲染 **25 个多视角图**用于训练（提高视角鲁棒性）
- 与 [[PhysXNet]]、[[PhysX-Mobility]] 合并得 **42K+** 训练样本

---

## 关键公式

### 公式 1: [[Template-based RLE|模板化 RLE 编码]]

$$
\text{RLE}(M_z) = T_{k^*} \oplus \Delta(M_z, T_{k^*}), \quad k^* = \arg\min_{k} \| M_z - T_k \|_1
$$

**含义**: 对第 $z$ 张体素切片 $M_z$，先找最近模板 $T_{k^*}$，然后只存其与模板的差异 $\Delta$，整体序列由模板索引和残差 RLE 拼接。

**符号说明**:
- $M_z \in \{0,1\}^{W \times W}$: 第 $z$ 层体素二值切片
- $T_k$: 第 $k$ 个共享模板层（在训练集上聚类得到）
- $\Delta(\cdot, \cdot)$: 残差的 2D RLE 编码
- $\oplus$: 序列拼接

### 公式 2: [[Voxel-Mesh 一致性|体素-Mesh 解码]]

$$
\hat{\mathcal{M}} = \text{TRELLIS}(\mathcal{V}), \quad \mathcal{V} = \bigcup_{z=1}^{H} \text{RLE}^{-1}(\text{token}_z)
$$

**含义**: 把每层 RLE token 反解码为体素切片后并集得到完整体素网格 $\mathcal{V}$，再用 [[TRELLIS]] 转成水密 mesh。

**符号说明**:
- $\mathcal{V}$: 完整 3D 体素网格
- $\hat{\mathcal{M}}$: 输出 mesh
- $\text{RLE}^{-1}$: RLE 反解码

### 公式 3: [[绝对尺度估计|Absolute Scale]] 损失

$$
\mathcal{L}_{\text{scale}} = \left| \log\frac{\hat{s}_{abs}}{s_{abs}^{*}} \right|
$$

**含义**: 用对数空间的绝对误差监督绝对尺度预测，避免大尺度物体（车 / 建筑）的损失把小物体（杯子）淹没。

**符号说明**:
- $\hat{s}_{abs}$: VLM 预测的最大边长（米）
- $s_{abs}^{*}$: 来自 PhysXVerse 标注 + VLM 校准

### 公式 4: 总训练目标（多任务自回归）

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{lm}}^{\text{geo}} + \lambda_1 \mathcal{L}_{\text{lm}}^{\text{phys}} + \lambda_2 \mathcal{L}_{\text{lm}}^{\text{kin}} + \lambda_3 \mathcal{L}_{\text{scale}}
$$

**含义**: 几何 / 物理 / 运动学三个分支都用 [[Next Token Prediction|next-token prediction]] 交叉熵监督，绝对尺度额外用 log-L1 监督。

**符号说明**:
- $\mathcal{L}_{\text{lm}}^{\text{geo/phys/kin}}$: 三个生成分支的语言建模损失
- $\lambda_{1,2,3}$: 任务权重
- 训练分阶段调度：先 geometry，再加 physics，最后加 kinematics

### 公式 5: PhysX-Bench 绝对尺度评分

$$
S_{\text{scale}} = 1 - \min\left(1,\ \frac{2 \cdot |\hat{s} - s^{*}|}{|\hat{s}| + |s^{*}|}\right)
$$

**含义**: 对生成尺度与参考尺度计算 **对称百分比误差** 并映射到 $[0,1]$，作为绝对尺度的 plausibility 分数。

**符号说明**:
- $\hat{s}, s^{*}$: 预测 / 参考绝对尺度（最大边长）
- $S_{\text{scale}}$: 越大越好

### 公式 6: PhysX-Bench 运动学评分

$$
S_{\text{kin}} = w_1 \cdot S_{\text{prior}} + w_2 \cdot S_{\text{reveal}} + w_3 \cdot S_{\text{global}}, \quad \sum_i w_i = 1
$$

**含义**: 运动学评分是三个子指标的加权：先验 part 运动一致性、新揭示 part 的合理性、整体铰接树的全局一致性。

**符号说明**:
- $S_{\text{prior}}$: prior-part motion consistency
- $S_{\text{reveal}}$: revealed-entity plausibility
- $S_{\text{global}}$: global articulation coherence

---

## 关键图表

### Figure 1: Teaser / 统一物理 3D 生成

![Figure 1](https://arxiv.org/html/2605.21572v1/x1.png)

**说明**: 在 PhysXVerse 高多样性数据驱动下，PhysX-Omni 单一模型生成 [[刚体]]、[[可形变物体]]、[[铰接物体]] 三类资产，每个资产带物理属性、关节、可直接进 [[MuJoCo]] / [[IsaacLab]]。

### Figure 2: Coarse-to-Fine 生成流水线

![Figure 2](https://arxiv.org/html/2605.21572v1/x2.png)

**说明**: VLM 先从图像推理出全局信息（类别、part 树、整体 bbox），再多轮自回归生成 part 级几何、物理属性和运动学。每一轮的输出回填到 context 作为下一轮的条件。

### Figure 3: Template-based RLE 几何表征

![Figure 3](https://arxiv.org/html/2605.21572v1/x3.png)

**说明**: (a) 对比不同几何表征（点云 token / 直接体素 / 隐 latent / 本文 RLE），本文方案在 token 效率与几何保真度之间取得平衡；(b) z 轴切片 → 2D RLE → 模板层共享的具体流程，相邻切片只存与模板的残差。

### Figure 4: PhysXVerse 数据统计

![Figure 4](https://arxiv.org/html/2605.21572v1/x4.png)

**说明**: 2.9K+ 类别覆盖 family（家具 / 车辆 / 机器人 / 空中系统 / 大型结构等）、part 数量从 1 到 65、语义多样性显著高于既有数据集。

### Figure 5: PhysX-Bench 六维度评测

![Figure 5](https://arxiv.org/html/2605.21572v1/x5.png)

**说明**: 六个评测维度——geometry / absolute scale / material / affordance / kinematics / description。每个维度都对齐人类评分（Spearman ≥ 0.8）。

### Figure 6: 定性对比（rigid / deformable / articulated）

![Figure 6](https://arxiv.org/html/2605.21572v1/x6.png)

**说明**: 与 PhysX-Anything、TRELLIS、PartCrafter 等基线相比，PhysX-Omni 在复杂几何（多 part 椅子、自行车）和可形变物体（布料、绳）上明显更接近 GT。

### Figure 7: 性能 + 人类对齐验证

![Figure 7](https://arxiv.org/html/2605.21572v1/x7.png)

**说明**: 左：跨方法的总体性能雷达；右：PhysX-Bench 各维度的 Spearman 相关系数 vs 人类打分，全部 ≥ 0.8。

### Figure 8: 复杂场景额外定性结果

![Figure 8](https://arxiv.org/html/2605.21572v1/x8.png)

**说明**: 在 65-part 复杂铰接物体（如多抽屉柜、自行车）上 PhysX-Omni 仍能正确推理 part 划分和关节连接。

### Figure 9: 可形变物体物理模拟

![Figure 9](https://arxiv.org/html/2605.21572v1/x9.png)

**说明**: 生成的 deformable assets 直接导入 simulator 做自由落体 / 受力变形，行为符合材料属性预测（橡胶弹性 vs 织物褶皱）。

### Figure 10: 几何表征消融

![Figure 10](https://arxiv.org/html/2605.21572v1/x10.png)

**说明**: 对比 token-only / 直接 voxel / 标准 RLE / 本文 Template-RLE，在相同 context budget 下本文方案 CD 最低，证明模板共享真正起作用。

### Figure 11: 机器人操作 demo

![Figure 11](https://arxiv.org/html/2605.21572v1/x11.png)

**说明**: 把生成资产直接放进 [[MuJoCo]] / [[IsaacLab]]，机器人对铰接抽屉、可形变物体执行 grasp / pull，物理行为合理（关节正确开合、布料正确变形），验证 simulation-ready 的实用性。

### Figure 12: 场景级生成

![Figure 12](https://arxiv.org/html/2605.21572v1/x12.png)

**说明**: 配合深度估计（[[Depth Anything V2]] / [[MoGe-2]]）从单张室内照片重建出"可仿真的整个房间"，每个物体都是 simulation-ready 的，可与 [[RoboCasa]]、[[OmniWorld]] 这类仿真场景数据集互补。

### Table 1: 常规几何 + 物理指标定量对比

| Method | PSNR ↑ | CD ↓ | F-score ↑ | Abs Scale Err ↓ | Material ↑ | Affordance ↑ | Kinematic ↑ | Desc ↑ |
|--------|--------|------|-----------|------------------|------------|--------------|-------------|--------|
| TRELLIS | 19.34 | 14.20 | 78.40 | — | — | — | — | — |
| PartCrafter | 17.85 | 22.10 | 70.55 | — | — | — | — | — |
| PhysX-Anything | 16.40 | 37.06 | 54.20 | 298.19 | 0.42 | 0.48 | 0.4191 | 0.58 |
| **PhysX-Omni (Ours)** | **21.52** | **2.95** | **91.28** | **2.79** | **0.78** | **0.81** | **0.9185** | **0.83** |

**说明**: 在 PhysXVerse 和 PhysX-Mobility 上几何质量 CD 直接从 37.06 降到 2.95（约 12×），绝对尺度误差从 298.19 降到 2.79（两个量级），运动学评分翻倍。

### Table 2: PhysX-Bench 综合评测

| Method | CLIP ↑ | 3D Cons. ↑ | Visual ↑ | Abs Scale ↑ | Material ↑ | Affordance ↑ | Kinematic ↑ | Desc ↑ |
|--------|--------|------------|----------|-------------|------------|--------------|-------------|--------|
| TRELLIS | 24.85 | 0.72 | 3.42 | — | — | — | — | — |
| 3D-LLM | 22.50 | 0.65 | 2.95 | 0.41 | 0.45 | 0.52 | 55.30 | 0.62 |
| PhysX-Anything | 23.20 | 0.68 | 3.15 | 0.48 | 0.49 | 0.55 | 65.99 | 0.65 |
| **PhysX-Omni** | **26.90** | **0.85** | **4.20** | **0.81** | **0.78** | **0.84** | **80.72** | **0.85** |

**说明**: 六维度全面领先；运动学维度（铰接物体最难评测的部分）从 65.99 拉到 80.72。

### Table 3: 几何表征消融（与 Figure 10 配套）

| 表征 | Tokens / Asset | CD ↓ | F-score ↑ |
|------|----------------|------|-----------|
| Raw Voxel Flatten | > 100K | OOM | — |
| Latent Token Only | 8K | 8.40 | 78.20 |
| Standard 2D RLE | 12K | 4.85 | 86.30 |
| **Template-based RLE** | **8K** | **2.95** | **91.28** |

**关键发现**: 模板共享在相同 token 预算下显著提升几何质量；没有模板共享的标准 RLE 用了 1.5× tokens 还差 1.9 个 CD。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| **PhysXVerse**（本文） | 8.7K+ 资产 / 2.9K+ 类别 | 首个通用 simulation-ready，含 deformable | 训练 + 测试 |
| [[PhysXNet]] | 既有 | 铰接物体为主 | 训练补充 |
| [[PhysX-Mobility]] | 既有 | 运动学标注丰富 | 训练 + 评测 |
| 合并训练集 | 42K+ 资产 | rigid / deformable / articulated 全覆盖 | 训练 |

### 实现细节

- **Backbone**: [[Qwen3-VL|Qwen2.5-VL-7B-Instruct]]
- **Decoder**: [[TRELLIS]]（voxel → mesh）
- **优化器**: AdamW，cosine schedule，warmup ratio 0.03
- **峰值学习率**: $2 \times 10^{-5}$
- **Batch Size**: 128 (effective)
- **Context Length**: 16,384 tokens
- **训练**: 5 epochs ≈ 14 天
- **硬件**: 64 × NVIDIA A100
- **每对象 view 数**: 25 multi-view 渲染图

### 可视化结果

把生成资产直接灌进 [[MuJoCo]] / [[IsaacLab]]：抽屉能正确开合（revolute / prismatic 自动判别），布料受力产生符合预测密度的形变，机器人抓取生成杯子时摩擦系数合理，单次生成到仿真验证全流程 <1 分钟。

---

## 批判性思考

### 优点

1. **真正打通了"3D 生成 → 物理仿真"的最后一公里**: 以往工作要么只生成 mesh，要么只生成关节，要么只生成材质；这是第一篇把四者打包并验证 sim-ready 的工作。
2. **几何表征本身是独立贡献**: Template-based RLE 是个干净的 idea，相比 dense latent 的好处是**显式、可解释、可裁剪**——这对 VLM-style 自回归生成是关键。
3. **数据 + 基准 + 模型三件套**: PhysXVerse / PhysX-Bench / PhysX-Omni 同时放出，对社区是结构性贡献，类似 [[ImageNet]] 之于视觉。
4. **跟具身的连接非常具体**: Figure 11/12 直接拿到机器人仿真和场景级重建去用，不是 isolated 几何工作。

### 局限性

1. **VLM token 上限是天花板**: 16K context 配 8K tokens/asset 看似宽裕，但碰到超复杂场景（如完整工厂）就堆不下，模板共享的压缩比也不是常数（高频细节多了模板就没那么有效）。
2. **可形变物体的"物理参数"准确性存疑**: paper 给出了 free-fall / water-drop 评测，但没和真实物理实验对比——你怎么知道生成的 "Young's modulus" 是真物理还是 VLM 编的？
3. **Kinematics 评测仍是软评测**: $S_{\text{kin}}$ 三项子指标都依赖 VLM 打分，不是真在仿真器里跑长 horizon 任务测，仍可能存在 reward hacking。
4. **没回答"sim-to-real gap"**: 资产在仿真器里跑得通不等于训出来的 policy 能上真机，论文只展示 sim 内 demo 没有 real-world roll-out。
5. **数据规模相对 [[Objaverse]] 仍小**: 8.7K + 42K 训练样本对 7B VLM 来说不大，预测在分布外类别（医疗器械、特殊工业零件）会很脆。

### 潜在改进方向

1. **接入 [[OmniWorld]] / [[RoboCasa]] 做端到端 sim-to-real**: 资产 + 场景 + policy 一起验证。
2. **替换 [[TRELLIS]] decoder 为更新的 3D foundation model**（如 [[VGGT]]、[[Pi3X]]）以支持 unbounded 分辨率。
3. **加 Real-to-Sim 闭环**: 用真实物体扫描验证生成的物理参数，把误差信号 backprop 回 VLM。
4. **把 PhysX-Omni 当 [[World Model]] 的 asset generator**: 上层 VLA / WM 需要"生成一个新场景做 rollout"时直接调用本文 → 闭环。
5. **支持流体 / 关节柔体**: 当前 deformable 还是 mass-spring 级，物理 fidelity 不够。

### 可复现性评估

- [x] 代码开源（按项目主页承诺，arXiv 提交时通常会同步）
- [x] 预训练模型（Qwen2.5-VL 微调权重）
- [x] 训练细节完整（64 A100 / 14 天 / batch 128 都披露）
- [x] 数据集可获取（PhysXVerse 公开）

---

## 关联笔记

### 基于

- [[VLM]]: 整个框架的 backbone 思路
- [[Qwen3-VL|Qwen2.5-VL]]: 直接微调对象
- [[TRELLIS]]: voxel → mesh 解码器
- [[PhysXNet]] / [[PhysX-Mobility]]: 前作 + 训练数据来源
- [[Run-Length Encoding]]: 几何表征的底层工具

### 对比

- **PhysX-Anything**: 同一组前作，PhysX-Omni 把它在所有维度上甩开 12× 几何精度
- [[TRELLIS]]: 纯几何生成 SOTA，但没有物理属性
- **PartCrafter / CLAY**: 类目专用 3D 生成，覆盖窄

### 方法相关

- [[Template-based RLE]]: 本文核心几何表征
- [[Tree-structured 3D Representation]]: 层级 part 表征
- [[Voxelization]]: 体素化预处理
- [[Simulation-Ready 3D Asset]]: 概念笔记

### 硬件/数据相关

- [[MuJoCo]] / [[IsaacLab]]: 下游仿真器
- [[RoboCasa]] / [[OmniWorld]]: 同方向的场景级数据集
- [[Depth Anything V2]] / [[MoGe-2]]: 场景级应用所需的深度先验
- [[sim-to-real]]: 最终目标

---

## 速查卡片

> [!summary] PhysX-Omni
> - **核心**: 用 VLM + 模板化 RLE 体素切片，一次性生成"几何 + 物理 + 运动学"齐备的 sim-ready 3D 资产
> - **方法**: Qwen2.5-VL-7B 微调；z 轴体素切片 → 2D RLE → 模板层共享；coarse-to-fine 多轮解码；TRELLIS decoder 出 mesh
> - **结果**: CD 2.95（PhysX-Anything 37.06）、绝对尺度误差 2.79（vs 298.19）、运动学 0.9185（vs 0.4191）；六维 PhysX-Bench 全面领先
> - **配套**: PhysXVerse 数据集（8.7K+ 资产，2.9K+ 类别）+ PhysX-Bench 评测基准
> - **项目**: https://physx-omni.github.io/

---

*笔记创建时间: 2026-05-24*
