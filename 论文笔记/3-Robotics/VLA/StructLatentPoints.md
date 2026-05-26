---
title: "Learning Structural Latent Points for Efficient Visual Representations in Robotic Manipulation"
method_name: "StructLatentPoints"
authors: [Yicheng Jiang, Jiaxu Wang, Junhao He, Zesen Gan, Junhao Li, Qiang Zhang, Jingkai Sun, Jiahang Cao, Mingyuan Sun, Xiangyu Yue, Qiming Shao]
year: 2026
venue: ICRA
tags: [3d-gaussian-splatting, visual-representation, robot-manipulation, pretraining, point-cloud, variational-autoencoder, vae]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.21258v1
created: 2026-05-24
---

# 论文笔记：Learning Structural Latent Points for Efficient Visual Representations in Robotic Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | HKUST, HKUST(Guangzhou), CUHK MMLab, HKU, Tsinghua |
| 日期 | May 2026 |
| 项目主页 | （未公开） |
| 对比基线 | [[PonderV2]]、[[GaussianDream]]、SPA、[[VC-1]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.21258) / [PDF](https://arxiv.org/pdf/2605.21258) |

---

## 一句话总结

> 在 [[Point Transformer V3]] 编码后的潜空间塞一个 **point-wise VAE**，同时正则化 point 的 feature 与坐标到 Gaussian 先验，用轻量 [[3D Gaussian Splatting]] rendering 做预训练监督，得到介于 [[NeRF]] 与 [[3DGS]] 之间的混合表征，下游 manipulation 性能超越 [[PonderV2]] 系全栈 baseline。

---

## 核心贡献

1. **结构化潜点表征 (Structural Latent Points)**: 提出在 point-cloud autoencoder 的瓶颈层引入 [[Point-wise Latent VAE|PL-VAE]]，对每个 latent point 的 *coordinate* 和 *feature* 都做 Gaussian 正则——产物是显式（[[3DGS]]）与隐式（[[NeRF]]）之间的 hybrid representation。
2. **几何推理模块 (Reasoned Geometric Properties)**: 通过 [[3DGS Rendering|3D Gaussian Splatting]] feature splatting 渲染 image / depth / semantic feature 三种视角监督，强迫 latent point 自适应预测 Gaussian 属性（中心、旋转、尺度、不透明度、颜色），避免 PonderV2 的 NeRF MLP 重型解码。
3. **训练高效性**: 在 ScanNetV2 上预训练仅 **~166 GPU-hours @ 1×A800**，相比 [[VC-1]]（~10k h）、SPA（~4k h @ 80×A100）、[[PonderV2]]（~768 h @ 8×A100）有数量级优势。
4. **多平台 SoTA**: 在 [[RLBench]] 9 任务、[[ManiSkill]]2 6 任务以及真机 AgileX Piper 6 任务上均取得 Mean Rank 第 1 的成绩，明确超越 [[Lift3D]]、[[PonderV2]]、SPA 等 3D pretraining 路线。

---

## 问题背景

### 要解决的问题

机器人 manipulation 任务的视觉表征预训练，长期面临"几何精度 vs. 渲染效率 vs. 下游 policy 友好度"的三难。当前主流路线：
- **2D ViT-based ([[VC-1]], [[MultiMAE]])**: 语义足够但 3D 空间 grounding 弱
- **NeRF-based ([[PonderV2]], SPA)**: 几何强但训练慢、解码网络重，downstream 不友好
- **GS-based ([[GaussianDream]], Lift3D)**: 渲染快但 representation 过度依赖精确点位

### 现有方法的局限

1. **隐式表征 ([[NeRF]])**：需要昂贵 volumetric MLP query，几何信息被深埋在 weights 里，难以提取 task-aligned feature
2. **显式表征 ([[3D Gaussian Splatting|3DGS]])**：每个 Gaussian 必须精确对应 3D 物理位置，对预训练阶段是过强的约束——manipulation 不需要像素级几何
3. **统一 implicit/explicit 中间态缺失**：缺少一种"保留结构性但不被精确几何束缚"的表征

### 本文的动机

作者认为：manipulation 真正需要的是 *coarse-grained but structured* 表征。如果能让 latent point **既有粗略的空间位置（用于 attention 和 3D grounding）**，又**不被强制对应到精确物理点（避免 3DGS 的优化僵硬性）**，就能拿到 implicit 与 explicit 的双重好处。VAE 正则化是天然的实现路径——把 latent point 的 coordinate 和 feature 同时拉向 Gaussian 先验，得到的就是"结构化但模糊"的潜点分布。

---

## 方法详解

### 模型架构

StructLatentPoints 采用 **encoder–PL-VAE–decoder + GS rendering head** 的双任务训练架构：

- **输入**: 单帧或多帧 RGB-D 重建得到的稠密点云 $P_{dense} \in \mathbb{R}^{N\times 6}$（坐标 + 颜色）
- **Backbone**: [[Point Transformer V3|PTv3]] encoder $E_\theta$ 下采样至稀疏 $P_{sparse} \in \mathbb{R}^{M\times C}$
- **核心模块**: [[Point-wise Latent VAE|PL-VAE]] 在每个 sparse point 上做 feature/coordinate 双重 VAE 编码，得到 $z_{vae}$
- **解码侧 1（重建）**: PTv3 decoder 上采样，回归原始坐标和颜色（[[重建损失|Reconstruction Loss]]）
- **解码侧 2（渲染）**: Geometric Reasoning Module $G_\theta$ 预测每个 latent point 的 Gaussian 属性 → [[3DGS Rendering|3D Gaussian Splatting]] 渲染 multi-view 监督
- **下游接入**: 训练完毕后丢弃 decoder，把 $z_{vae}$ 作为 [[ACT (Action Chunking Transformer)|ACT]] policy 的视觉输入

### 核心模块

#### 模块1: Point Transformer V3 编码器

**设计动机**: 利用 [[Point Transformer V3]] 的 serialization-based attention 实现高效稀疏点云特征提取，避免传统点云网络的 KNN 邻域开销。

**具体实现**:
- 输入稠密点云 $P_{dense}$ 经过 PTv3 多层下采样得到 $P_{sparse}$ 与对应稀疏特征
- 稀疏点数 $M \ll N$，每个 sparse point 携带 $C$ 维语义特征

#### 模块2: Point-wise Latent VAE (PL-VAE)

**设计动机**: 既要保留 sparse point 的 spatial 锚定能力，又要让 latent 具备生成性和 Gaussian 平滑性——VAE 的 Gaussian prior 天然适合做"模糊化的 anchor"。

**具体实现**:
- 对每个 sparse point，PointNet backbone $\Psi$ 提取局部特征
- 通过 attention pooling 得到全局特征 $f_{global}$，与每点特征 $f_i$ 拼接得到 hybrid context $h_i$
- 两组独立 MLP $\phi_f, \phi_p$ 分别预测 [[特征分布|feature 分布]] $(\mu^f_i, \sigma^f_i)$ 和 [[坐标分布|coordinate 分布]] $(\mu^p_i, \sigma^p_i)$
- 用 [[重参数化|Reparameterization]] 采样得到 $z_{vae}$（feature + coord 拼接）

#### 模块3: Geometric Reasoning Module (Reasoned Geo)

**设计动机**: 不让 PL-VAE 直接输出 Gaussian splat 属性（容易塌陷到精确点位），而是设一个独立 head $G_\theta$ 从 latent point 推理出 splat 属性，留出"latent 模糊 / splat 精确"的解耦空间。

**具体实现**:
- $G_\theta$ 接 latent point，输出每个 Gaussian 的 center、rotation、scale、opacity、color
- 配合 feature decoder $H_c$（color projector）和 $H_s$（semantic feature projector）
- 输出可被 [[3DGS Rendering|GS rasterizer]] $R_v$ 渲染成 RGB / depth / feature map

#### 模块4: 双任务训练目标

**重建任务**（保持 latent 的可解码性）+ **渲染任务**（注入 3D 几何监督）+ **VAE 正则**（拉向 Gaussian 先验）。三者通过 annealing weight $w(t)$ 动态平衡：训练初期偏向渲染（几何 grounding），后期偏向重建（latent 表达力）。

---

## 关键公式

### 公式1: [[Point-wise Latent VAE|PL-VAE 编码]]

$$
\begin{aligned}
f_{global}, f_i &= \mathrm{Agg}(\Psi(P_{sparse})), \quad \mathrm{MLP}(\Psi(P_{sparse})) \\
h_i &= \mathrm{cat}(f_{global}, f_i) \\
(\mu^f_i, \sigma^f_i) &= \phi_f(h_i), \quad (\mu^p_i, \sigma^p_i) = \phi_p(h_i)
\end{aligned}
$$

**含义**: 把每个 sparse point 的特征同时编码成 *feature 分布* 与 *coordinate 分布*，二者用独立 MLP 处理，得到双重正则化的 latent 表示。

**符号说明**:
- $\Psi$: PointNet-style backbone，提取局部特征
- $\mathrm{Agg}$: attention pooling，得到全局上下文
- $h_i$: 第 $i$ 个 sparse point 的全局+局部 hybrid context
- $\phi_f, \phi_p$: feature 与 coordinate 分布的独立预测 MLP
- $(\mu^f_i, \sigma^f_i)$: 第 $i$ 点特征维度的 Gaussian 参数
- $(\mu^p_i, \sigma^p_i)$: 第 $i$ 点坐标维度的 Gaussian 参数

### 公式2: [[KL 散度|KL Divergence Loss]]（双正则化）

$$
\mathcal{L}_{KL} = \mathrm{KL}\!\left(\mathcal{N}(\mu^f_i, \sigma^f_i) \,\|\, \mathcal{N}(0, I)\right) + \mathrm{KL}\!\left(\mathcal{N}(\mu^p_i, \sigma^p_i) \,\|\, \mathcal{N}(0, I)\right)
$$

**含义**: 同时把 feature 后验和 coordinate 后验都拉向标准 Gaussian 先验——这是本文最核心的设计，让 latent point 既不丢结构（feature 维度受控）又不死板（coordinate 维度有可变形性）。

**符号说明**:
- $\mathcal{N}(0, I)$: 标准 Gaussian 先验
- 两项 KL 各自独立，权重相同

### 公式3: [[3DGS Rendering|Rendering Loss]]

$$
\mathcal{L}_{render} = \frac{1}{V}\sum_{v=1}^{V}\left[\beta_1 \|I_v - \hat{I}_v\|_1 + \beta_2 \|D_v - \hat{D}_v\|_1 + \beta_3 \|F^s_v - \hat{F}^s_v\|_1\right]
$$

**含义**: 在 $V$ 个 novel view 上同时监督 RGB、深度、语义特征三种 modality——这是 3DGS feature splatting 的三通道损失。

**符号说明**:
- $V$: 监督视角数量
- $I_v, \hat{I}_v$: 视角 $v$ 的真实 / 渲染 RGB 图
- $D_v, \hat{D}_v$: 视角 $v$ 的真实 / 渲染 depth map
- $F^s_v, \hat{F}^s_v$: 视角 $v$ 的真实 / 渲染 semantic feature map（来自 [[DINOv2]] 等 backbone）
- $\beta_1 = 1$, $\beta_2 = 0.2$, $\beta_3 = 0.1$（论文默认）

### 公式4: [[重建损失|Reconstruction Loss]]

$$
\mathcal{L}_{recon} = \frac{1}{N}\sum_{i=1}^{N}\left[\|p_i - \hat{p}_i\|_1 + \|c_i - \hat{c}_i\|_1\right]
$$

**含义**: 经过 PL-VAE 瓶颈后，decoder 仍需还原原始稠密点云的坐标和颜色——保证 latent 不丢信息。

**符号说明**:
- $p_i, \hat{p}_i$: 第 $i$ 个 dense point 的真实 / 重建坐标
- $c_i, \hat{c}_i$: 真实 / 重建 RGB 颜色
- $N$: 稠密点数量

### 公式5: [[VAE 重建损失|VAE Reconstruction Loss]]

$$
\mathcal{L}_{vae} = \frac{1}{M}\sum_{i=1}^{M}\left[\omega_1 \|p_i - \hat{p}_i\| + \omega_2 \|f_i - \hat{f}_i\|\right]
$$

**含义**: 在 sparse point 层面单独约束 VAE 重建——保证 PL-VAE 的 sampling 不破坏 sparse point 自身的语义和坐标。

**符号说明**:
- $M$: sparse point 数量
- $p_i, f_i$: 第 $i$ sparse point 的坐标 / 特征
- $\omega_1 = 1, \omega_2 = 0.1$

### 公式6: 总损失（带 annealing）

$$
\mathcal{L}_{total} = (1 - w(t))\,\mathcal{L}_{render} + w(t)\,\mathcal{L}_{recon} + \mathcal{L}_{vae} + \mathcal{L}_{KL}
$$

**含义**: 训练初期 $w(t) \approx 0$，loss 主要来自 rendering（几何 grounding 阶段）；后期 $w(t) \to 1$，loss 转向 reconstruction（latent 表达力阶段）。两个 VAE 项始终全权参与。

**符号说明**:
- $w(t)$: 关于 epoch $t$ 的 annealing 权重，从 0 单调增长到 1
- 四个 loss 之间隐含权重均为 1（除 $w(t)$ 调度的两项）

### 公式7: Feature Splatting 输出

$$
(\hat{F}_v, \hat{D}_v) = R_v(\mathcal{G}, P_{dense})
$$

**含义**: 通过 splatting rasterizer $R_v$ 把 latent point 推理出的 Gaussian 集合 $\mathcal{G}$ 投影到视角 $v$，同时输出 feature map 和 depth map。

**符号说明**:
- $\mathcal{G} = \{(\text{center}, \text{rot}, \text{scale}, \text{opacity}, \text{color}, \text{feat})_k\}$: Gaussian 集合
- $R_v$: 视角 $v$ 的 [[3D Gaussian Splatting|3DGS rasterizer]]

### 公式8: 下游 Policy 输入

$$
\hat{a} = \pi(z_{vae}, O_{agent})
$$

**含义**: 预训练后，[[ACT (Action Chunking Transformer)|ACT]] policy $\pi$ 接 $z_{vae}$ 作为视觉 token，加上 agent proprioception $O_{agent}$ 输出动作。

**符号说明**:
- $z_{vae}$: PL-VAE 输出的 latent point 集合
- $O_{agent}$: 关节角、末端位姿等本体状态
- $\hat{a}$: 输出动作（[[Action Chunking|动作块]]）

---

## 关键图表

### Figure 1: Main Pipeline / 系统概览

![Figure 1](https://arxiv.org/html/2605.21258v1/x1.png)

**说明**: StructLatentPoints 的整体训练管线。RGB-D → 稠密点云 → [[Point Transformer V3|PTv3]] encoder → sparse point features → [[Point-wise Latent VAE|PL-VAE]] 双正则化 → latent point cloud $z_{vae}$。两条解码路径并行：(1) PTv3 decoder 重建原始稠密点云；(2) Geometric Reasoning Module 推理 Gaussian 属性后由 [[3DGS Rendering|GS rasterizer]] 渲染 RGB / depth / feature map。下游 policy 只取 $z_{vae}$ 接 [[ACT (Action Chunking Transformer)|ACT]]。

### Figure 2: Reasoned Geometry Ablation / 几何推理模块消融

![Figure 2](https://arxiv.org/html/2605.21258v1/x2.png)

**说明**: 对比有 / 无 Reasoned Geometric Properties 模块时的渲染 depth map 质量。无该模块时 splat 直接来自 latent point 坐标，depth 出现明显空洞和噪声；启用后 depth 平滑、物体边缘清晰——证明独立的 geometric head 是 latent 表达力的关键解耦点。

### Figure 3: PCA Feature Visualization / 特征可视化

![Figure 3](https://arxiv.org/html/2605.21258v1/x3.png)

**说明**: 用 PCA 对 latent feature 降维可视化。"Post feature decoding"指 feature-only rendering 后的可视化。结果显示：feature-only rendering 比 feature+color 联合 rendering 拥有**更强的语义可分性**——同一物体在不同视角下的 latent feature 聚类更紧，跨物体边界更清晰。这一观察支撑了"先 feature 后 color"的解耦渲染设计。

### Figure 4: Real-World Tasks / 真实机器人任务

![Figure 4](https://arxiv.org/html/2605.21258v1/x4.png)

**说明**: 在 AgileX Piper 机械臂 + Orbbec Femto Bolt ToF 相机平台上的 6 个真机任务：clean table with sponge、place toy pea in bowl、hang pliers、place coffee cup on plate、lift up bottle、pour water。每任务 40 个训练 demo + 10 个测试 demo，30 Hz 采集。

### Table I: RLBench 9 任务结果

| 方法 | Mean S.R. | Mean Rank |
|------|-----------|-----------|
| MultiViT | 0.21 | 8.56 |
| PTv3 (scratch) | 0.41 | 6.50 |
| SpUNet (scratch) | 0.32 | 7.39 |
| [[VC-1]] | 0.31 | 6.50 |
| [[MultiMAE]] | 0.31 | 6.44 |
| [[PonderV2]] | 0.42 | 4.83 |
| GSRL | 0.38 | 6.11 |
| Lift3D | 0.50 | 3.61 |
| No.VAE (ablation) | 0.49 | 3.56 |
| **Ours** | **0.56** | **1.50** |

**说明**: StructLatentPoints 在 9 个 RLBench 任务上平均 Mean S.R. 0.56，Mean Rank 1.50——明确超越 NeRF-based 的 [[PonderV2]]（+14 pts）、ViT-based [[VC-1]]（+25 pts）和最强 baseline Lift3D（+6 pts）。No.VAE 消融跌至 0.49，证明 PL-VAE 是关键贡献。

### Table II: ManiSkill2 6 任务结果

| 方法 | Mean S.R. | Mean Rank |
|------|-----------|-----------|
| MultiViT | 0.33 | 8.92 |
| PTv3 (scratch) | 0.39 | 8.17 |
| SpUNet (scratch) | 0.49 | 7.42 |
| [[VC-1]] | 0.49 | 6.67 |
| [[MultiMAE]] | 0.44 | 8.67 |
| SPA | 0.48 | 6.67 |
| [[PonderV2]] | 0.53 | 5.92 |
| GSRL | 0.50 | 6.50 |
| Lift3D | 0.63 | 1.83 |
| No.VAE (ablation) | 0.59 | 3.92 |
| **Ours** | **0.64** | **1.33** |

**说明**: 在 [[ManiSkill]]2 PickCube / StackCube / TurnFaucet / Hang / Fill / Pour 6 任务上，Mean Rank 1.33 与 Lift3D 拉近距离但仍居第一。

### Table III: Geometric Reasoning Module 消融

| 配置 | TurnTap | CloseJar | MeatOffGrill | StackCube | Hang |
|------|---------|----------|--------------|-----------|------|
| No Reasoned Geo | 0.00 | 0.24 | 0.72 | 0.28 | 0.92 |
| **Ours** | **0.08** | **0.48** | **0.76** | **0.36** | **0.96** |

**说明**: 去掉 Geometric Reasoning Module 后，所有任务都退化，TurnTap 直接到 0——证明把 Gaussian 属性预测从 latent 中显式解耦是必要的。

### Table IV: 训练效率对比

| 方法 | GPU 时数 | 硬件 |
|------|----------|------|
| [[VC-1]] | ~10,000 h | - |
| SPA | ~4,000 h | 80×A100 |
| [[PonderV2]] | ~768 h | 8×A100 |
| **Ours** | **~166 h** | **1×A800** |

**说明**: 训练成本相比同类 3D pretraining 路线低 1-2 个数量级。轻量化 3DGS rendering 比 NeRF volumetric query 高效得多是核心原因。

### Table V: 真实机器人结果（部分）

| 任务 | Pointnet | [[PonderV2]] | Ours (scratch) | **Ours** |
|------|----------|--------------|----------------|----------|
| Place mandarin | 56 | 72 | 76 | **84** |
| Lift up bottle | 64 | 48 | 40 | **68** |
| Place starbucks | 36 | 32 | 28 | **36** |

**说明**: 真机三任务（成功率 %）。预训练版本（Ours）显著优于 from-scratch（Ours (scratch)）和 [[PonderV2]]——尤其在 Lift up bottle 上 PonderV2 反而不如 scratch，说明 NeRF-based 预训练在高对比度纹理任务上可能 overfit 几何。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| ScanNetV2 | ~1.5k 室内 RGB-D 扫描 | 真实多视角点云 | 预训练 |
| [[RLBench]] | 9 任务 × 50 demo | 多视角 manipulation | 仿真下游评估 |
| [[ManiSkill]]2 | 6 任务（PickCube/StackCube/TurnFaucet/Hang/Fill/Pour） | 多样化物理 manipulation | 仿真下游评估 |
| AgileX Piper 真机数据 | 6 任务 × (40+10) demo | Femto Bolt ToF 相机 30 Hz | 真机评估 |

### 实现细节

- **Backbone**: [[Point Transformer V3|PTv3]] encoder/decoder
- **VAE 模块**: PointNet + attention pooling + 双 MLP（feature/coordinate 分布头）
- **下游 Policy**: [[ACT (Action Chunking Transformer)|ACT]]
- **优化器**: AdamW, lr=5e-5
- **训练 epoch**: 10,000
- **硬件**: 预训练 1×A800 (~166h)；下游 1×RTX 3090
- **超参**: $\beta_1=1, \beta_2=0.2, \beta_3=0.1$（rendering 三通道权重）；$\omega_1=1, \omega_2=0.1$（VAE 重建权重）

### 可视化结果

PCA visualization（Figure 3）证明 feature-only rendering 比 feature+color 联合 rendering 拥有更清晰的语义边界，这一发现支撑了 decoder 解耦设计——color 与 feature 应分别由不同 projector 处理。

---

## 批判性思考

### 优点

1. **设计干净**：抓住"manipulation 不需要像素级几何，但要比 ViT feature 多一点结构"这个 sweet spot，PL-VAE 是恰到好处的实现
2. **训练高效**：166 GPU-h 在同行里属于"绿色"水平，对 academic lab 友好
3. **三个平台齐刷 SoTA**：RLBench / ManiSkill2 / 真机一致超越 baseline，可信度高于单 benchmark 工作
4. **解耦设计经过消融验证**：Geometric Reasoning Module 与 PL-VAE 两个核心组件都有清晰的 ablation 支撑

### 局限性

1. **没跟最新 VLA 比较**：评测选 RLBench / ManiSkill2 而非 LIBERO / RoboCasa，无法直接对照 [[OpenVLA]] / [[π₀]] / [[Pi05]] 等最新策略
2. **预训练数据有限**：仅用 ScanNetV2，未利用更大规模的 Ego4D / OpenScene 等数据
3. **命名营销**："Structural Latent Points" 本质就是 VAE-regularized point feature，没必要起新名字
4. **Real-world 任务规模偏小**：6 任务 × 50 demo，距离 [[GR00T N1.5]] 那种规模化真机评估还有距离
5. **真机 Place starbucks 提升有限**（Ours 36 = Pointnet 36），说明在低成功率长尾任务上的优势不稳健

### 潜在改进方向

1. **替换更强 backbone**：[[Point Transformer V3|PTv3]] 已经不是 SoTA，可以试 PTv4 或更现代的 sparse point transformer
2. **替换更强 rendering supervision**：当前用 [[3D Gaussian Splatting|3DGS]] feature splatting，可以引入 [[GaussianDream]] 那种 temporal Gaussian evolver 做时序预测
3. **结合 VLA**：把 $z_{vae}$ 作为 [[OpenVLA]] 的视觉前缀，看能否替代 [[DINOv2]] / [[SigLIP]] 等 2D backbone
4. **扩到 mobile manipulation**：当前只在固定相机视角下评估，对相机视角变化的鲁棒性未知

### 可复现性评估

- [ ] 代码开源（论文未明确给出 URL）
- [ ] 预训练模型（未明确）
- [x] 训练细节较完整（公开了 loss 权重、lr、epoch）
- [x] 数据集可获取（ScanNetV2 / RLBench / ManiSkill2 都是公开）

---

## 关联笔记

### 基于

- [[Point Transformer V3]]: 点云编码 backbone
- [[3D Gaussian Splatting]]: rendering supervision 的核心机制
- [[VAE]]: PL-VAE 的理论基础

### 对比

- [[PonderV2]]: NeRF-based 预训练路线的前辈，本文证明 3DGS supervision 比 NeRF MLP query 更高效
- [[GaussianDream]]: 同样把 3DGS 作为 manipulation 预训练目标，但 GaussianDream 是"feed-forward 3DGS 作为世界模型 prefix"，本文是"3DGS 作为 representation pretraining loss"，思路互补
- [[VC-1]] / [[MultiMAE]]: 2D ViT-based 预训练对照组

### 方法相关

- [[Point-wise Latent VAE]]: 核心创新，本文独立提出的概念
- [[3DGS Rendering]]: rendering supervision 机制
- [[Geometric Reasoning Module]]: Gaussian 属性预测头
- [[KL 散度]]: VAE 正则化基础

### 硬件/数据相关

- [[ManiSkill]]: 主要仿真评测平台
- [[RLBench]]: 第二仿真评测平台
- [[ScanNetV2]]: 预训练数据集

---

## 速查卡片

> [!summary] Learning Structural Latent Points
> - **核心**: PL-VAE 在 latent 上双重正则化 feature 与 coordinate，拿到 implicit/explicit 之间的混合表征
> - **方法**: PTv3 encoder + PL-VAE + 双解码（重建 + 3DGS feature splatting rendering），下游 ACT policy
> - **结果**: RLBench Mean Rank 1.50 / ManiSkill2 Mean Rank 1.33，~166 GPU-h 训练（比 PonderV2 快 5×）
> - **代码**: 未公开

---

*笔记创建时间: 2026-05-24*
