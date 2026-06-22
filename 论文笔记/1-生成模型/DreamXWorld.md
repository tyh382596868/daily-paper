---
title: "DreamX-World 1.0: A General-Purpose Interactive World Model"
method_name: "DreamX-World"
authors: [Yancheng Bai, Rui Chen, Xiangxiang Chu, Rujing Dang, Hao Dou, Bingjie Gao, Qiwen Gu, Siyu Hong, Jiachen Lei, Geng Li, Jifan Li, Ruimin Lin, Qingfeng Shi, Bingze Song, Lei Sun, Jing Tang, Ruitian Tian, Jun Wang, Jiahong Wu, Pengfei Zhang, Shen Zhang, Jiashu Zhu]
year: 2026
venue: arXiv
tags: [world-model, interactive-generation, camera-control, video-generation, diffusion-transformer, autoregressive, reinforcement-learning, long-horizon]
zotero_collection: 1-生成模型
image_source: pending
arxiv_html: https://arxiv.org/html/2606.16993
created: 2026-06-22
---

# 论文笔记：DreamX-World 1.0: A General-Purpose Interactive World Model

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | DreamX Team (AMAP-ML) |
| 日期 | June 2026 |
| 项目主页 | [dreamx-world.github.io](https://dreamx-world.github.io) |
| 对比基线 | [[HY-WorldPlay]]、[[LingBot-World]]、[[Matrix-Game]]、[[Yume]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.16993) / [Code](https://github.com/AMAP-ML/DreamX-World) |

---

## 一句话总结

> DreamX-World 1.0 是通用交互式世界模型，通过 E-PRoPE 相机控制、几何引导场景记忆、可组合事件控制和 DMD 蒸馏，实现跨域长时程视频生成，并达到 8 卡 RTX 5090 上 16 FPS 的实时推理。

---

## 核心贡献

1. **E-PRoPE 高效相机控制**: 对空间降采样的 token 应用投影注意力，在保持相机控制精度的同时将推理延迟降低约 30%
2. **记忆条件化场景持久性**: 基于相机几何的检索机制，使生成世界在长时程交互中能够一致地重访先前场景
3. **可组合事件指令微调**: 支持多实体、区域引导、跨对象交互的结构化事件控制，是对比系统中唯一完整支持的模型
4. **AR 蒸馏 + RL 后训练**: 通过 [[DMD]] 蒸馏和强化学习后训练，在保持视频质量的同时显著提升推理速度

---

## 问题背景

### 要解决的问题

构建一个**通用交互式世界模型**：能够以文本/图像为条件，生成可精确相机导航、支持场景重访、响应可提示事件的长时程视频，并覆盖真实感、游戏和风格化等多种视觉域。

### 现有方法的局限

- 现有世界模型在长时程生成时，离开上下文窗口后场景内容会发生漂移，无法一致地重访先前区域
- [[PRoPE]] 等相机控制方法计算开销大，18,480 个 token 导致推理延迟高
- 现有交互世界模型（HY-WorldPlay、LingBot-World）不支持多实体组合事件控制，或不支持区域引导与跨对象交互

### 本文的动机

将世界建模视为**全栈问题**——数据策划、训练、评估、推理加速须从全局视角统筹优化。通过在各环节的系统性设计，在 5B 参数模型上超越 8B 和 14B 基线。

---

## 方法详解

### 模型架构

DreamX-World 1.0 基于 [[Diffusion Transformer]]（DiT）[[自回归]] 框架，采用分阶段渐进训练流水线：

- **输入**: 文本/图像条件 + 相机轨迹 + 可选事件指令
- **Backbone**: [[Diffusion Transformer]] (DiT)，5B 参数
- **核心模块**: [[E-PRoPE]] 相机控制 + [[Memory-Conditioned Scene Persistence]] 场景记忆 + 事件指令微调
- **推理**: [[DMD]] 蒸馏学生模型 + [[自回归流式推理]]
- **总参数**: 5B

> 🖼️ **Figure 2: System Overview** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.16993)）

**说明**: DreamX-World 1.0 的系统全貌，整合了相机精确数据、高效相机控制、[[自回归]]蒸馏、长程记忆、交互对齐和优化服务六大模块。

---

### 核心模块

#### 模块一：E-PRoPE（高效投影位置编码）

**设计动机**: [[PRoPE]] 将相机几何编码为世界到图像的投影矩阵，但对所有 18,480 个 token 执行投影注意力计算开销过大。

**具体实现**:
- 将输入 token 从 18,480 空间下采样至 4,096（约 4.5× 压缩）
- 在 [[DiT]] 每个注意力层附加 E-PRoPE 模块
- 省略 [[RoPE]] 分量，仅保留投影子矩阵
- 冻结 DiT backbone，仅对 PRoPE 模块反向传播
- 推理时即插即用

> 🖼️ **Figure 5: E-PRoPE Architecture** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.16993)）

**说明**: E-PRoPE 模块附接到每个 DiT 注意力层，对空间降采样的 token 应用投影注意力，忽略 adaLN 调制以简化说明。

**性能**: 相机控制分数 73.75 vs 完整 PRoPE 的 73.89，推理延迟从 80s 降至 59s（1280×720 分辨率）。

---

#### 模块二：记忆条件化场景持久性（Memory-Conditioned Scene Persistence）

**设计动机**: 长时程生成中，离开上下文窗口的区域在相机返回时会发生外观漂移，需要显式记忆机制。

**具体实现**:
- **几何引导记忆检索**: 根据相机姿态和视野重叠度检索非局部记忆帧
- **自注意力流拼接**: 将记忆帧 $z_M$、近期历史 $z_H$ 和目标帧一起送入 [[self-attention]]
- **时序位置编码**: 使用 [[NTK-aware RoPE]] 缩放和 [[YaRN]] 方法处理长序列
- **曝光偏差缓解**: 从 [[Stable Video Infinity]] 借鉴错误注入方法，消除训练-测试分布差距

> 🖼️ **Figure 6: Memory-Conditioned Training** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.16993)）

**说明**: 记忆条件化场景持久性训练框架，几何引导检索选取非局部记忆帧，通过残差循环条件化路径送入网络。

---

#### 模块三：可组合事件指令微调（Event Instruction Tuning）

**设计动机**: 用户需要精细控制视频中多个实体的行为，现有系统不支持结构化多实体事件组合。

**数据结构**: 层级化标注，包含：
- 全局场景描述
- 实体级事件记录（实体引用、事件谓词、空间锚点、时间区间）

**训练**: 对完整 DiT 微调，使用保守更新 + 严格梯度裁剪，混合事件指令样本与非事件样本。

**能力矩阵**（对比竞品）：

| 能力 | LingBot-World | HY-WorldPlay 1.5 | Matrix-Game 3.0 | Yume-1.5 | **DreamX-World 1.0** |
|------|:---:|:---:|:---:|:---:|:---:|
| 可提示事件 | ✓ | ✓ | ✗ | ✓ | **✓** |
| 对象级事件 | ✓ | ✓ | ✗ | ✓ | **✓** |
| 区域引导事件 | △ | ✗ | ✗ | ✗ | **✓** |
| 多实体组合 | △ | △ | ✗ | ✗ | **✓** |
| 跨对象交互 | ✗ | ✗ | ✗ | ✗ | **✓** |

---

#### 模块四：自回归长视频生成与蒸馏

**训练阶段**（渐进式流水线）：
1. 对大规模高质量视频数据进行 [[Causal Forcing]] 训练
2. 使用 [[Infinity-RoPE]] 进行长序列适应
3. 集成相机控制 E-PRoPE 分支
4. [[DMD]] 强制匹配：学生 rollout 与双向教师对齐

> 🖼️ **Figure 7: DMD-Forcing Distillation** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.16993)）

**说明**: 相机控制长视频蒸馏的 DMD-forcing 流程，E-PRoPE AR 学生模型通过局部时间窗口上的 DMD 监督从双向 E-PRoPE 教师模型中蒸馏。

---

#### 模块五：强化学习后训练

**目标**: 在 DMD 蒸馏后进一步提升视频质量和相机可控性。

**奖励模型**:
- 水平平移和旋转精度奖励
- 视觉质量评估奖励

**训练策略**:
- 渐进式更新调度，防止模型崩溃
- 长时程 rollout 生成 + 短片段采样计算奖励
- [[DiffusionNFT]] 软更新 + KL 正则化
- 解耦优化窗口与 rollout 时域

> 🖼️ **Figure 8: RL Training Overview** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.16993)）

**说明**: RL 训练概览，模型先生成长时程自回归 rollout，再采样短片段用于奖励评估。

---

## 关键公式

### 公式1: [[PRoPE|投影旋转位置编码矩阵分解]]

$$
D_s^{\text{PRoPE}} = \begin{bmatrix} D_s^{\text{Proj}} & 0 \\ 0 & D_s^{\text{RoPE}} \end{bmatrix}
$$

**含义**: [[PRoPE]] 将注意力位置编码矩阵分解为投影子矩阵（编码相机几何）和 [[RoPE]] 子矩阵（旋转位置编码）两部分。

**符号说明**:
- $D_s^{\text{Proj}}$: 编码相机世界到图像投影几何的子矩阵
- $D_s^{\text{RoPE}}$: 复制旋转位置编码的子矩阵

---

### 公式2: [[相机控制误差|相机控制评估误差]]

$$
e_{\text{camera}} = \sqrt{e_\theta \cdot e_t}
$$

**含义**: 综合旋转误差和平移误差的几何均值，作为相机控制精度的统一评估指标。

**符号说明**:
- $e_\theta$: 尺度无关的旋转误差
- $e_t$: 尺度无关的平移误差

---

### 公式3: [[Memory-Conditioned Scene Persistence|记忆拼接打包]]

$$
z_{\text{pack}} = [z_M \mid z_H \mid z_C^\tau]
$$

**含义**: 将记忆帧、近期历史帧和加噪目标帧拼接后一同送入注意力流，监督信号只作用于目标帧。

**符号说明**:
- $z_M$: 几何检索得到的非局部记忆帧 latent
- $z_H$: 近期历史帧 latent
- $z_C^\tau$: 噪声水平为 $\tau$ 的目标帧 latent

---

### 公式4: [[场景重访|重访帧对筛选准则]]

$$
|\theta_i - \theta_j| \leq \tau_\theta, \quad \|t_i - t_j\|_2 \leq \tau_t
$$

**含义**: 在评估记忆一致性时，筛选视角相近（旋转差 ≤ 2°、平移差 ≤ 0.1）但时间上相距较远（≥ 20% 总帧数）的帧对作为重访评估对。

**符号说明**:
- $\theta_i, \theta_j$: 两帧的旋转角度
- $t_i, t_j$: 两帧的平移向量
- $\tau_\theta = 2°$: 旋转差阈值
- $\tau_t = 0.1$: 平移差阈值
- $|j - i| \geq \lfloor 0.2T \rfloor$: 最小时间间隔约束

---

### 公式5: [[场景记忆增益|记忆一致性增益评估]]

$$
\text{Gain} = \begin{cases} S_{\text{revisit}} - S_{\text{baseline}} & \text{（相似度指标：PSNR, SSIM, DINO-Sim 等）} \\ S_{\text{baseline}} - S_{\text{revisit}} & \text{（感知距离：LPIPS）} \end{cases}
$$

**含义**: 通过重访帧与首次访问帧的相似度增益，量化世界模型的长程场景记忆能力。

---

## 关键图表

### Figure 1: 系统 Teaser

> 🖼️ **Figure 1: DreamX-World Teaser** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.16993)）

**说明**: DreamX-World 1.0 在真实感、游戏风格和风格化视觉域中生成精确相机控制和事件控制的交互式视频。

---

### Figure 3/4: 数据流水线

> 🖼️ **Figure 3: UE Data Pipeline** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.16993)）

**说明**: UE 合成数据生成流水线，轨迹在线采集验证，离线渲染生成相机姿态、动作和元数据，支持分布式渲染和故障恢复。

> 🖼️ **Figure 4: Cleaning/Filtering Pipeline** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.16993)）

**说明**: 数据清洗、过滤和属性标注流水线概览，包含基础过滤、几何清洗和多维度标注。

---

### Figure 9: 自回归流式推理

> 🖼️ **Figure 9: Autoregressive Streaming Inference** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.16993)）

**说明**: 蒸馏采样器逐块从噪声中生成帧序列，更新滚动 KV 缓存，使用块相对相机控制实现低延迟流式输出。

---

### Figure 10: 定性结果

> 🖼️ **Figure 10: Qualitative Results** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.16993)）

**说明**: DreamX-World-1.0-5B 定性结果，每行展示一段生成视频序列均匀采样的五个关键帧，覆盖多种场景类型。

---

### Figure 11/12: 评估轨迹与人类偏好

> 🖼️ **Figure 11: Revisit Trajectory Templates** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.16993)）

**说明**: 三种评估轨迹模板的鸟瞰图（往返、平移旋转、闭环），颜色从蓝（起点）到红（终点）编码时序进展。

> 🖼️ **Figure 12: Human Preference Study** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.16993)）

**说明**: 与 HY-WorldPlay 1.5 和 LingBot-World 的人类偏好研究，展示胜/平/负百分比。

---

### Table 1: E-PRoPE 消融（PRoPE vs E-PRoPE）

| 模型 | 相机控制 ↑ | 图像质量 ↑ | 动态度 ↑ | 场景切换检测 ↑ | 时序闪烁 ↑ | 运动平滑 ↑ | 延迟 (s) ↓ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| PRoPE | 73.89 | 66.15 | 87.5 | 96.67 | 96.02 | 98.65 | 80 |
| **E-PRoPE** | **73.75** | **66.75** | 85.83 | **98.33** | **96.17** | **98.79** | **59** |

**关键发现**: E-PRoPE 相机控制仅损失 0.14 分，推理延迟降低 ~26%（80s→59s）。

---

### Table 2: 基础评估（5 秒片段）

| 模型 | 参数 | 相机 ↑ | 质量 ↑ | 场景切换 ↑ | 闪烁 ↑ | 平滑 ↑ | 动态 ↑ | 伪影 ↑ | 总分 ↑ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| HY-WorldPlay 1.5 | 8B | 65.12 | 68.23 | 98.33 | 96.45 | 99.05 | 66.67 | 71.66 | 80.79 |
| LingBot-World | 14B | 71.73 | 67.76 | 85.00 | 94.94 | 97.06 | 88.33 | 58.33 | 80.45 |
| **DreamX-World-1.0-5B** | **5B** | **73.75** | 66.75 | **98.33** | 96.17 | 98.79 | **85.83** | **73.75** | **84.76** |

**关键发现**: 5B 参数以最小模型规模取得最高总分，相机控制分数领先 14B 的 LingBot-World（73.75 vs 71.73）。

---

### Table 3: 长时程评估（30 秒 rollout）

| 模型 | 参数 | 相机 ↑ | 质量 ↑ | 场景切换 ↑ | 闪烁 ↑ | 平滑 ↑ | 动态 ↑ | 伪影 ↑ | 总分 ↑ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| HY-WorldPlay 1.5 | 8B | 65.86 | 63.02 | 91.00 | 97.00 | 99.11 | 52.00 | 14.00 | 68.85 |
| LingBot-World | 14B | 63.76 | 60.81 | 54.00 | 96.59 | 97.86 | 87.00 | 12.00 | 67.43 |
| **DreamX-World-1.0-5B** | **5B** | 62.03 | **64.11** | 80.00 | 96.35 | 98.41 | 75.00 | **17.00** | **70.41** |

**关键发现**: 长时程下 DreamX-World 视觉保真度最高（质量 64.11），LingBot-World 场景切换检测急剧下降（54.00），说明其长时程场景一致性差。

---

### Table 4: 记忆一致性评估（重访帧对）

| 模型 | ΔPSNR ↑ | ΔSSIM ↑ | ΔLPIPS ↑ | ΔDINO-Sim ↑ | ΔVPR-Sim ↑ | ΔSP-Match ↑ | CLIP-V ↑ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| LingBot-World | 0.61 | 0.019 | 0.039 | 0.090 | 0.100 | 0.088 | 0.987 |
| HY-WorldPlay 1.5 | 3.19 | 0.079 | 0.202 | 0.200 | 0.110 | 0.251 | 0.992 |
| **DreamX-World-1.0-5B** | **3.92** | **0.098** | **0.232** | **0.246** | **0.142** | 0.216 | 0.991 |

**指标说明**:
- ΔPSNR/ΔSSIM: 像素级保真度增益
- ΔLPIPS: 感知一致性增益（对应 LPIPS 下降，越大越好）
- ΔDINO-Sim: 语义身份保持
- ΔVPR-Sim: 跨视角地点识别
- ΔSP-Match: 几何结构对应
- CLIP-V: 时序平滑度（绝对值）

**关键发现**: DreamX-World 在像素级、感知级和语义级一致性上全面领先，验证了几何引导记忆检索的有效性。

---

## 推理加速技术

### DiT 优化

- **INT8 SageAttention**: 注意力层 INT8 量化
- **FP8 AngelSlim**: FFN 层 FP8 量化
- **序列并行**: 跨 GPU 分片时空 token
- **Fused Triton Kernels**: 高频算子融合
- **[[TeaCache]] 残差复用**: 在稳定时间步区域复用残差

### VAE 解码优化

- **75% 剪枝率**: 基于 Matrix-Game 3.0 VAE 的结构剪枝
- **[[ParaVAE]] 并行解码**: 跨 GPU 并行解码
- **高度维度切分** + patch 聚合
- **torch.compile** 用于后续迭代

### 服务架构

- **异步流水线并行**: DiT 生成与 VAE 解码重叠执行
- **连续块发射**: 解码完成即刻发出，不等待全部完成

**结果**: 8 卡 RTX 5090 达到 **16 FPS**。

---

## 数据工程

### 数据来源

| 类型 | 数据集 | 相机姿态来源 |
|------|--------|-------------|
| UE 合成数据 | 自建 UE 引擎 | 地面真值（Ground Truth） |
| 真实世界 | SpatialVID, RealEstate10K, Sekai, DL3DV | 位姿恢复 |
| 游戏数据 | Sekai-Game, OmniWorld-Game | 引擎导出 |

### 过滤流水线

- **基础过滤**: 时长、帧率、视觉变化（CLIP embedding）
- **几何清洗**: 内参验证、轨迹平滑度
- **属性标注**: 字幕、美学质量、运动强度、场景类别、视觉风格、主体类型、运动类别（3D vs 4D）

---

## 实验

### 实现细节

- **Backbone**: [[Diffusion Transformer]] (DiT)，5B 参数
- **推理硬件**: 8 × RTX 5090 GPU
- **推理速度**: 16 FPS（8 卡）
- **分辨率**: 1280×720（标准评估）
- **长时程**: 最长可生成约 1 分钟视频

### 可视化结果

DreamX-World-1.0-5B 在多种场景类型下生成的长视频均呈现：
- 精确的相机轨迹跟随
- 跨域（真实感/游戏/风格化）视觉一致性
- 重访场景时保持外观一致

---

## 批判性思考

### 优点

1. **全栈系统设计**: 从数据、训练到推理的系统性优化，5B 模型超越 14B 基线，展示了工程质量的重要性
2. **E-PRoPE 高效性**: token 降采样思路简洁，30% 延迟节省实用价值高
3. **独特事件控制能力**: 跨对象交互是所有对比系统中唯一支持的，开创了更丰富的世界交互范式
4. **严格记忆评估协议**: 基于相机几何的重访帧对筛选方法，为场景记忆评估提供了可复现的标准

### 局限性

1. **长时程外观/布局漂移**: 扩展交互后物体外观或场景布局可能大幅偏移
2. **控制信号冲突**: 事件内容与字幕指定的世界设定不兼容时存在矛盾
3. **自动评估不完整**: 开放式交互仍需人类和任务导向评估作为补充

### 潜在改进方向

1. **以角色为中心的世界模型**: 保持多角色跨长序列的持久身份，支持多角色交互
2. **原生音视频生成**: 同步生成语音、环境音和动作相关音效

### 可复现性评估

- [x] 代码开源（GitHub: AMAP-ML/DreamX-World）
- [ ] 预训练模型（未确认是否公开权重）
- [x] 训练细节完整（论文描述较详细）
- [ ] 数据集可获取（部分依赖私有 UE 合成数据）

---

## 关联笔记

### 基于

- [[PRoPE]]: 投影位置编码，E-PRoPE 的基础
- [[DMD]]: Distribution Matching Distillation，蒸馏框架
- [[RoPE]]: 旋转位置编码，时序 token 位置建模
- [[Infinity-RoPE]]: 长序列 RoPE 外推方法
- [[Stable Video Infinity]]: 曝光偏差缓解方法来源

### 对比

- [[HY-WorldPlay]]: 8B 交互世界模型，相机控制分 65.12 vs 73.75
- [[LingBot-World]]: 14B 交互世界模型，长时程场景切换检测急剧下降
- [[Matrix-Game]]: 不支持可提示事件
- [[Yume]]: 不支持区域引导和跨对象交互

### 方法相关

- [[Diffusion Transformer]]: 核心 backbone 架构
- [[自回归生成]]: 长视频生成的推理范式
- [[TeaCache]]: 推理加速中的残差缓存技术
- [[DiffusionNFT]]: RL 训练中的软更新方法

### 硬件/数据相关

- [[RTX 5090]]: 推理硬件，8 卡达 16 FPS
- [[SpatialVID]], [[RealEstate10K]], [[DL3DV]]: 真实世界训练数据集

---

## 速查卡片

> [!summary] DreamX-World 1.0: A General-Purpose Interactive World Model
> - **核心**: 通用交互式世界模型，支持相机导航、场景重访和可组合事件控制
> - **方法**: E-PRoPE（高效相机控制）+ 几何引导场景记忆 + 事件指令微调 + DMD 蒸馏 + RL 后训练
> - **结果**: 总分 84.76，超越 8B/14B 基线；记忆一致性全维度最优；8 卡 RTX 5090 达 16 FPS
> - **代码**: [github.com/AMAP-ML/DreamX-World](https://github.com/AMAP-ML/DreamX-World)

---

*笔记创建时间: 2026-06-22*
