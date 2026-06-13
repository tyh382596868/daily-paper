---
title: "MoVerse: Real-Time Video World Modeling with Panoramic Gaussian Scaffold"
method_name: "MoVerse"
authors: [Yang Zhou, Ziheng Wang, Yuqin Lu, Haofeng Liu, Jun Liang, Shengfeng He, Jing Li]
year: 2026
venue: arXiv
tags: [video-world-model, gaussian-splatting, panoramic-generation, novel-view-synthesis, real-time-rendering, knowledge-distillation, interactive-scene]
zotero_collection: 1-生成模型
image_source: pending
arxiv_html: https://arxiv.org/html/2606.13376
created: 2026-06-13
---

# 论文笔记：MoVerse: Real-Time Video World Modeling with Panoramic Gaussian Scaffold

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Orange Team, Youku Moku-Lab, HUJING Digital Media & Entertainment Group |
| 日期 | June 2026 |
| 项目主页 | https://orange-3dv-team.github.io/MoVerse/ |
| 对比基线 | [[ViewCrafter]] / [[新视角合成]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.13376) / [Code](https://github.com/Orange-3DV-Team/MoVerse)（代码审查中，预计一月内开源） |

---

## 一句话总结

> MoVerse 将单张窄视场图像扩展为可实时交互漫游的三维世界，通过全景生成 → 高斯脚手架 → 自回归视频渲染三级流水线，在单块 RTX 4090 上达到 8 FPS 实时输出。

---

## 核心贡献

1. **Topology-Aware 全景扩展**: 用拓扑感知扩散模型将单张 NFOV 图像补全为重力对齐、水平周期连续的 360° ERP 全景图，在 3D 推理之前填补缺失视野。
2. **全景几何感知 3D 高斯脚手架**: 在 angular–inverse-depth 空间中进行前馈残差预测，将全景图直接提升为密集可渲染的 [[3D Gaussian Splatting|3D 高斯脚手架]]，作为持久空间记忆。
3. **双向教师蒸馏 + 因果自回归学生**: 训练双向扩散教师模型提供高质量条件渲染，再蒸馏为因果自回归学生模型，实现有界延迟的流式输出，最终达到实时交互漫游。

---

## 问题背景

### 要解决的问题

从单张 NFOV（Narrow Field-of-View，窄视场）图像出发，构建一个可实时交互漫游的完整三维世界。该设定极具挑战性：输入仅覆盖环境的一小片区域，而交互漫游需要完整的周围世界、持久的几何结构、可控的相机运动，以及时序一致的高保真观测。

### 现有方法的局限

- **纯生成方法**（如视频扩散）：缺乏显式 3D 几何，长程一致性差，相机控制精度不足。
- **基于 NeRF/3DGS 重建方法**：需要多视角输入，单张图像场景不完整，存在遮挡区域的幻觉问题。
- **现有世界模型**：通常将世界构建与观测渲染耦合在一起，难以同时保证几何一致性和感知质量。
- **实时性瓶颈**：高质量视频生成模型（如双向扩散）延迟高，无法满足交互漫游需求。

### 本文的动机

将**世界构建（world construction）**与**观测渲染（observation rendering）**解耦：先用生成模型补全场景几何（全景图 → 3D 高斯脚手架），再用高效渲染模型沿用户轨迹生成视频。利用 [[3D Gaussian Splatting]] 的显式可渲染性作为控制信号，桥接几何一致性与生成质量，再通过 [[Knowledge Distillation|知识蒸馏]] 从双向扩散模型提炼出因果自回归模型，解决实时延迟问题。

---

## 方法详解

### 模型架构

**MoVerse** 采用**三阶段流水线**架构：

```
输入: 单张 NFOV 图像
  ↓ Stage I
全景生成 (Topology-Aware Diffusion)
  → 重力对齐、水平周期的 360° ERP 全景图
  ↓ Stage II
3D 高斯脚手架构建 (Feed-Forward Residual Prediction)
  → 密集可渲染的 Gaussian 持久空间记忆
  ↓ Stage III
自回归视频精化 (Gaussian-Conditioned Video Renderer)
  → 沿用户指定相机轨迹的逼真视频流
输出: 实时可交互漫游场景 (8 FPS, RTX 4090)
```

- **输入**: 单张 NFOV 图像
- **Backbone**: [[Diffusion Model|扩散模型]] (Stage I) + 前馈 3DGS 提升网络 (Stage II) + 因果 [[Video Diffusion Model|自回归视频模型]] (Stage III)
- **核心模块**: [[3D Gaussian Splatting]] 作为可渲染空间记忆 + [[Knowledge Distillation]] 实现实时化
- **输出**: 用户可控漫游视频帧
- **推理速度**: 8 FPS（单 NVIDIA RTX 4090）

### 核心模块

#### 模块 1: Topology-Aware 全景扩散（Stage I）

**设计动机**: 利用 [[Diffusion Model|扩散模型]] 的生成能力填补单张 NFOV 图像缺失的视野，同时保证重力对齐（天地方向正确）和水平方向拓扑连续性（360° 首尾相接）。

**具体实现**:
- 以单张 NFOV 图像为条件，扩散模型生成完整 360° ERP（Equirectangular Projection，等距柱状投影）全景图
- "Topology-aware"指生成过程中约束水平方向周期边界条件，避免全景图左右边缘不匹配
- 重力对齐确保垂直方向的物理合理性，为后续 3D 提升提供稳定基础

#### 模块 2: 全景几何感知残差预测（Stage II）

**设计动机**: 利用全景图的等距柱状特性，在 [[高斯泼溅|Gaussian]] 参数空间中进行前馈预测，直接生成可渲染的 3D 高斯脚手架，避免优化-per-scene 的慢速流程。

**具体实现**:
- 表示空间: 采用 **angular–inverse-depth** 坐标系（角度 + 逆深度），适应全景图的球面几何
- **"Geometry-aware"**: 利用全景深度估计（可能借助 [[深度估计]] 先验）作为初始几何
- **"Residual prediction"**: 前馈网络预测相对于基础几何的残差，提升预测精度和稳定性
- 输出: 一组密集 [[3D Gaussian Splatting|3D 高斯点]]，构成持久空间记忆（不随相机移动消失）

#### 模块 3: 高斯条件视频渲染器（Stage III）

**设计动机**: 以 3D 高斯脚手架的光栅化渲染结果为几何先验/控制信号，生成感知质量更高的视频，同时保证相机轨迹可控性。

**具体实现**:

**训练阶段（教师-学生蒸馏）**:
- **双向扩散教师 (Bidirectional Diffusion Teacher)**: 可访问全序列上下文（双向注意力），生成高质量条件渲染结果；适合离线高质量生成，但不适合实时流式输出
- **因果自回归学生 (Causal Autoregressive Student)**: 仅访问历史帧（因果/单向），通过 [[Knowledge Distillation|知识蒸馏]] 从教师处学习，实现有界延迟的流式推理

**推理阶段**:
- 沿用户指定相机轨迹渲染高斯脚手架 → 获得几何一致的草图帧
- 因果自回归学生以草图帧为条件，自回归生成每一帧精化视频
- 有界延迟确保实时交互可行

---

## 关键公式

> 由于论文 HTML 版本尚未生成且 PDF 超出访问限制，以下公式基于方法描述推断，待正式公式获取后更新。

### 公式 1: [[高斯泼溅|Gaussian 渲染]]

$$
I(\mathbf{p}) = \sum_{i \in \mathcal{N}} c_i \alpha_i \prod_{j < i}(1 - \alpha_j)
$$

**含义**: 沿射线对 3D 高斯点进行体积渲染，得到像素颜色 $I(\mathbf{p})$。

**符号说明**:
- $\mathbf{p}$: 像素坐标
- $c_i$: 第 $i$ 个高斯的颜色（球面谐波系数解码）
- $\alpha_i$: 第 $i$ 个高斯在该像素的不透明度贡献
- $\mathcal{N}$: 按深度排序的高斯点集合

### 公式 2: [[Knowledge Distillation|蒸馏训练目标]]

$$
\mathcal{L}_{distill} = \mathbb{E}\left[\left\| f_{\theta}^{student}(x_{<t}, c_t) - f_{\phi}^{teacher}(x_{1:T}, c_t) \right\|^2\right]
$$

**含义**: 学生模型（因果）预测在给定当前条件 $c_t$ 时匹配教师模型（双向）的输出，教师的双向上下文信息被蒸馏到因果模型中。

**符号说明**:
- $f_{\theta}^{student}$: 因果自回归学生网络，参数 $\theta$
- $f_{\phi}^{teacher}$: 双向扩散教师网络，参数 $\phi$
- $x_{<t}$: 历史帧序列（因果输入）
- $x_{1:T}$: 完整帧序列（双向输入）
- $c_t$: 当前时刻的高斯脚手架渲染条件

### 公式 3: [[高斯泼溅|Angular-Inverse-Depth 表示]]

$$
\mathbf{g}_i = (\theta_i, \phi_i, d_i^{-1}, \mathbf{c}_i, \sigma_i, \mathbf{R}_i, \mathbf{s}_i)
$$

**含义**: 每个 3D 高斯点在全景坐标系下的参数化表示，以方位角、仰角和逆深度替代笛卡尔坐标。

**符号说明**:
- $\theta_i, \phi_i$: 方位角和仰角（angular 坐标）
- $d_i^{-1}$: 逆深度（inverse-depth，近处分辨率高）
- $\mathbf{c}_i$: 颜色特征
- $\sigma_i$: 不透明度
- $\mathbf{R}_i, \mathbf{s}_i$: 旋转矩阵和尺度（定义椭球形状）

---

## 关键图表

> 图片暂缺（arXiv HTML 版本尚未生成，项目主页返回 403，原图见 [arXiv](https://arxiv.org/abs/2606.13376) / [GitHub](https://github.com/Orange-3DV-Team/MoVerse)）

### Figure 1: 系统概览 (Pipeline Overview)

> 🖼️ **Figure 1: MoVerse Pipeline Overview** — 图片暂缺（原图见 [arXiv HTML](https://arxiv.org/html/2606.13376)，GitHub asset: `pipeline_overview.png`）

**说明**: 展示 MoVerse 三阶段流水线：单张 NFOV 输入 → 拓扑感知全景扩散 → 全景几何感知 3D 高斯脚手架构建 → 高斯条件自回归视频渲染 → 实时漫游输出。

### Figure 2: 全景扩展示意

> 🖼️ **Figure 2: Panoramic Generation** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.13376)）

**说明**: 单张 NFOV 图像（覆盖约 1/6～1/4 视野）经 Topology-Aware Diffusion 扩展为 360° ERP 全景图，展示重力对齐和水平周期连续性。

### Figure 3: 3D 高斯脚手架

> 🖼️ **Figure 3: 3D Gaussian Scaffold** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.13376)）

**说明**: 全景图提升为 3D 高斯脚手架后的点云可视化，展示密集、几何一致的空间记忆结构及其可渲染性。

### Figure 4: 教师-学生蒸馏框架

> 🖼️ **Figure 4: Teacher-Student Distillation** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.13376)）

**说明**: 双向扩散教师（离线高质量）与因果自回归学生（实时流式）的对比，展示蒸馏训练过程与推理时序对比。

### Figure 5: 定性结果 (Qualitative Results)

> 🖼️ **Figure 5: Qualitative Results** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.13376)）

**说明**: 多种场景（古代遗迹、赛博朋克街道、动漫场景、真实室内）的漫游渲染结果，展示跨域泛化能力。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 待补充（论文未公开详情） | — | 室内/室外多样场景 | 训练 |
| 室内/室外评估集 | — | 古代遗迹、赛博朋克、动漫、真实室内 | 测试 |

> 注：数据集细节因 PDF 无法完整读取而待补充。

### 实现细节

- **推理速度**: 8 FPS
- **硬件**: 单块 NVIDIA RTX 4090（24GB）
- **输入**: 单张 NFOV 图像
- **输出**: 实时流式视频帧
- 代码和预训练模型正在企业合规审查中，预计约一月内开源

### 可视化结果

MoVerse 支持多种风格场景的自由漫游：古代遗迹、赛博朋克街道、动漫场景和真实室内环境，展示跨域泛化的实用性。

---

## 批判性思考

### 优点

1. **解耦架构清晰**: 将世界构建（几何）与观测渲染（感知）解耦，各阶段目标明确，易于优化和替换组件。
2. **实时交互实用**: 8 FPS on RTX 4090 是视频世界模型中罕见的实时性能，教师-学生蒸馏方案有工程价值。
3. **全景先验合理**: 先补全 360° 全景再做 3D 提升，比直接从 NFOV 单图做重建更具信息完整性。
4. **显式 3D 记忆**: 高斯脚手架是持久的显式表示，避免隐式神经网络在长程一致性上的退化问题。

### 局限性

1. **场景静态假设**: 3D 高斯脚手架为静态场景，动态物体（行人、车辆）难以在脚手架中持久表示。
2. **全景幻觉传播**: Stage I 的全景生成存在幻觉风险（尤其是看不见的区域），错误会传播到 Stage II/III。
3. **缺乏公开实验数据**: 论文提交时间较短，GitHub 仅有 README，定量结果（PSNR/SSIM/LPIPS 等）无法验证。
4. **企业背景限制**: 来自优酷/爱奇艺系公司，代码审查周期不确定，可复现性存疑。

### 潜在改进方向

1. **动态场景支持**: 引入 4D 高斯或时序高斯变形，处理动态元素。
2. **在线更新脚手架**: 随相机移动发现新区域时在线更新 3D 脚手架，支持更大范围探索。
3. **更高 FPS**: 通过更激进的蒸馏或量化进一步提升推理速度。

### 可复现性评估

- [ ] 代码开源（审查中）
- [ ] 预训练模型（审查中）
- [ ] 训练细节完整（论文发布后待评估）
- [ ] 数据集可获取（待确认）

---

## 关联笔记

### 基于

- [[3D Gaussian Splatting]]: 空间记忆的核心表示形式
- [[Diffusion Model]]: Stage I 全景生成的基础
- [[Knowledge Distillation]]: 双向→因果模型的实时化手段
- [[Video Diffusion Model]]: Stage III 视频渲染的生成框架

### 对比

- [[ViewCrafter]]: 单图视频扩展方法，缺乏显式 3D 脚手架
- [[新视角合成]]: 传统新视角合成方法，通常需要多视角输入

### 方法相关

- [[高斯泼溅]]: 核心 3D 表示技术
- [[深度估计]]: Stage II 几何初始化可能依赖
- [[Causal Forcing]]: 类似的因果自回归视频生成思路
- [[Camera-Controlled 视频生成]]: 相机可控视频生成的相关工作

### 硬件/数据相关

- NVIDIA RTX 4090: 实时推理目标硬件

---

## 速查卡片

> [!summary] MoVerse: Real-Time Video World Modeling with Panoramic Gaussian Scaffold
> - **核心**: 单张 NFOV 图像 → 实时可漫游 3D 世界（8 FPS, RTX 4090）
> - **方法**: 拓扑感知全景扩散 → 全景几何感知 3D 高斯脚手架 → 双向教师蒸馏因果自回归视频渲染
> - **结果**: 室内外多种场景实时漫游，首个单图输入实时视频世界模型之一
> - **代码**: https://github.com/Orange-3DV-Team/MoVerse（审查中，约一月开源）

---

*笔记创建时间: 2026-06-13*
