---
title: "ImageWAM: Do World Action Models Really Need Video Generation, or Just Image Editing?"
method_name: "ImageWAM"
authors: [Yuyang Zhang, Wenyao Zhang, Zekun Qi, He Zhang, Haitao Lin, Jingbo Zhang, Yao Mu, Xiaokang Yang, Wenjun Zeng, Xin Jin]
year: 2026
venue: arXiv
tags: [world-action-model, image-editing, vla, flow-matching, robot-manipulation]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.19531
created: 2026-06-22
---

# 论文笔记：ImageWAM

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 东方理工大学、香港大学、上海交通大学等 |
| 日期 | June 2026 |
| 项目主页 | [zhangwenyao1.github.io/ImageWAM](https://zhangwenyao1.github.io/ImageWAM/) |
| 对比基线 | [[FastWAM]]、[[π₀]]、[[π₀.₅]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.19531) |

---

## 一句话总结

> 用预训练图像编辑模型替代视频生成，以 1/6 FLOPs、1/4 延迟实现与视频 WAM 相当甚至更优的机器人操作性能。

---

## 核心贡献

1. **提出 ImageWAM 框架**: 将预训练[[图像编辑模型]]重新用于指令条件下的机器人动作预测，规避了视频生成的冗余时序建模
2. **编辑感知中间表示**: 从图像编辑[[扩散模型]]的去噪过程中提取 KV cache，作为[[Action Expert|动作专家]]的条件信号，无需解码完整图像
3. **多骨干泛化验证**: 在 OmniGen2、Ovis-U1、FLUX.2 三种编辑骨干上均可无缝替换，大模型带来额外鲁棒性

---

## 问题背景

### 要解决的问题

[[World Action Model|世界动作模型（WAM）]]通过预测未来视频序列作为中间推理步骤来生成机器人动作。这种方法的计算代价极高，且需要对大量与动作无关的视觉细节（背景纹理、光照变化等）建模。

### 现有方法的局限

- **视频生成 WAM**（如 FastWAM-IDM）：需要生成完整的未来帧序列（密集 token），FLOPs 高达 63.65 TFLOPs，延迟 1081ms
- **视频伪影问题**：视频生成 WAM 在任务相关对象周围产生几何扭曲、空间布局不一致等伪影，这些伪影会误导动作预测
- **时序冗余**：视频生成需对所有中间帧建模，而大量时序细节对动作决策无实质帮助

### 本文的动机

图像编辑天然对齐"当前状态 → 目标状态"的变换，与机器人操作任务的语义完全一致：操作指令描述的正是对场景的编辑（"把杯子放到架子上"）。预训练图像编辑模型已经学会了如何关注指令相关的视觉变化区域，可以直接复用这种能力而无需重新学习时序动力学。

---

## 方法详解

### 模型架构

ImageWAM 采用**图像编辑骨干 + [[Action Expert|动作专家]]旁路**架构：

- **输入**: 语言指令 $l$ + 当前观测 $o_t$
- **Backbone**: 预训练图像编辑模型（OmniGen2 / Ovis-U1 / FLUX.2）
- **核心模块**: [[KV Cache|KV 缓存]]提取 → [[Joint Attention|联合注意力]]条件化 → [[Flow Matching|流匹配]]动作解码
- **输出**: [[Action Chunking|动作块]] $\mathbf{a}_{t:t+H}$（未来 H 步动作序列）
- **推理特点**: 仅需单次前向传播（固定去噪时间步 $\tau^*$），无需完整去噪轨迹

对比传统 WAM 流水线：

```
视频 WAM: (o_t, l) → 未来视频帧序列 → 动作
ImageWAM: (o_t, l) → 单帧终态图像（内部缓存）→ 动作
```

### 核心模块

#### 模块1：图像编辑骨干（Editing Backbone）

**设计动机**: 利用预训练[[图像编辑模型]]已学到的"指令-视觉变化对齐"能力，将操作任务重新表述为指令引导的视觉变换。

**具体实现**:
- 输入当前观测 $o_t$ 和语言指令 $l$，模型合成目标终态帧 $\hat{o}_{\text{edit}}$
- 在去噪过程的某个固定时间步 $\tau^*$ 提取各层的[[KV Cache|Key-Value 缓存]] $\mathcal{C}_{\text{edit}}^{\tau^*}$
- 训练时用[[Flow Matching|流匹配]]损失监督图像重建，确保骨干保持对任务的语义理解
- 推理时**不需要**完整去噪到最终图像，只需一次前向传播提取缓存即可

#### 模块2：动作专家（Action Expert）

**设计动机**: 通过[[Joint Attention|联合注意力]]将图像编辑的 KV 缓存注入动作解码，使动作生成以"视觉变化信息"为条件。

**具体实现**:
- 接收噪声动作 $\mathbf{a}_s$ 和流时间步 $s$，以编辑缓存 $\mathcal{C}_{\text{edit}}^{\tau}$ 为条件
- 通过[[Flow Matching|流匹配]]目标训练：预测速度场 $v_\theta(\mathbf{a}_s, s \mid o_t, l, \mathcal{C}_{\text{edit}}^{\tau})$
- 两个损失（图像编辑损失 + 动作流匹配损失）**联合优化**
- VLM 和多模态编码器权重冻结，只训练扩散图像分支和动作专家

---

## 关键公式

### 公式1: [[Action Chunking|动作块]] 定义

$$
\mathbf{a}_{t:t+H} = (a_t, a_{t+1}, \ldots, a_{t+H})
$$

**含义**: 策略输出未来 $H$ 步的动作序列，实现动作分块执行。

**符号说明**:
- $\mathbf{a}_{t:t+H}$: 动作块，包含 $H$ 个连续时间步的动作
- $H$: 动作预测视野（论文中 $H=16$）

---

### 公式2: [[World Action Model|WAM]] 传统流水线

$$
(o_t, l) \rightarrow \hat{o}_{t+1:t+H+1} \rightarrow \mathbf{a}_{t:t+H}
$$

**含义**: 传统 WAM 先生成完整未来视频序列，再从中推断动作，计算代价高。

**符号说明**:
- $o_t$: 当前观测
- $l$: 语言任务指令
- $\hat{o}_{t+1:t+H+1}$: 预测的未来帧序列（密集 token）

---

### 公式3: ImageWAM 图像编辑流水线

$$
(o_t, l) \rightarrow \hat{o}_{\text{edit}} \equiv \hat{o}_{t+H+1} \rightarrow \mathbf{a}_{t:t+H}
$$

**含义**: 只预测单个终态帧，省去中间帧，大幅降低计算量。

**符号说明**:
- $\hat{o}_{\text{edit}}$: 图像编辑生成的目标状态帧（源条件终点帧）

---

### 公式4: [[KV Cache|编辑缓存]] 提取

$$
\mathcal{C}_{\text{edit}}^{\tau} = \{(K_\ell^{\tau}, V_\ell^{\tau})\}_{\ell=1}^{L} = f_{\text{edit}}^{\tau}(o_t, l)
$$

**含义**: 在去噪时间步 $\tau$ 处，从图像编辑骨干各层提取 Key-Value 缓存，作为任务条件化的视觉变换信息。

**符号说明**:
- $\tau$: 去噪时间步
- $L$: Transformer 层数
- $K_\ell^{\tau}, V_\ell^{\tau}$: 第 $\ell$ 层在时间步 $\tau$ 的 Key、Value 矩阵
- $f_{\text{edit}}^{\tau}$: 图像编辑骨干在时间步 $\tau$ 的前向函数

---

### 公式5: [[Flow Matching|图像编辑流匹配]]损失

$$
\mathcal{L}_{\text{img}} = \mathbb{E}_{z^*, \epsilon_z, r}\left[\left\|u_\phi(z_r, r \mid o_t, l) - (\epsilon_z - z^*_{t+H+1})\right\|_2^2\right]
$$

其中插值图像潜变量：

$$
z_r = (1-r)z^*_{t+H+1} + r\epsilon_z
$$

**含义**: 监督扩散分支预测从噪声到目标帧潜变量的速度场，保持编辑骨干的语义对齐能力。

**符号说明**:
- $z^*_{t+H+1}$: 目标帧的潜变量
- $\epsilon_z \sim \mathcal{N}(0, I)$: 图像噪声
- $r \in (0,1)$: 图像流时间步
- $u_\phi$: 扩散分支的速度预测网络

---

### 公式6: [[Flow Matching|动作流匹配]]损失

$$
\mathcal{L}_{\text{act}} = \mathbb{E}_{\mathbf{a}^*, \epsilon_a, s, \tau}\left[\left\|v_\theta(\mathbf{a}_s, s \mid o_t, l, \mathcal{C}^{\tau}_{\text{edit}}) - (\epsilon_a - \mathbf{a}^*_{t:t+H})\right\|_2^2\right]
$$

其中插值动作样本：

$$
\mathbf{a}_s = (1-s)\mathbf{a}^*_{t:t+H} + s\epsilon_a
$$

**含义**: 动作专家学习在编辑缓存条件下，从噪声动作恢复真实动作序列的速度场。

**符号说明**:
- $\mathbf{a}^*_{t:t+H}$: 真实动作块
- $\epsilon_a \sim \mathcal{N}(0, I)$: 动作噪声
- $s \in (0,1)$: 动作流时间步
- $v_\theta$: 动作专家速度预测网络

---

### 公式7: 推理时高效缓存提取

$$
\mathcal{C}_{\text{edit}}^{\tau^*} = f_{\text{edit}}^{\tau^*}(o_t, l)
$$

$$
\hat{\mathbf{a}}_{t:t+H} \sim p_\theta(\mathbf{a}_{t:t+H} \mid o_t, l, \mathcal{C}^{\tau^*}_{\text{edit}})
$$

**含义**: 推理时固定去噪时间步 $\tau^*$，单次前向传播提取缓存，完全跳过完整去噪轨迹和图像解码。

**符号说明**:
- $\tau^*$: 推理时固定的最优去噪时间步
- $p_\theta$: 动作专家学到的动作分布

---

## 关键图表

### Figure 1: 方法对比总览

![Figure 1](https://arxiv.org/html/2606.19531/2606.19531v1/x1.png)

**说明**: 左侧展示传统视频生成 WAM 的流水线——从 $(o_t, l)$ 生成大量未来视频 token，存在计算冗余和视觉伪影问题；右侧展示 ImageWAM——直接生成单帧终态图像并提取 [[KV Cache|KV 缓存]]，聚焦于动作相关的当前到目标视觉差异。

### Figure 2: ImageWAM 流水线架构

![Figure 2](https://arxiv.org/html/2606.19531/2606.19531v1/x2.png)

**说明**: 完整 ImageWAM 流水线图。给定语言指令和当前观测 $o_t$，图像编辑骨干合成未来帧；[[Action Expert|动作专家]]通过 [[Joint Attention|联合注意力]]集成去噪过程中的中间 KV 特征 $\mathcal{C}_{\text{edit}}^{\tau}$，经[[Flow Matching|流匹配]]解码动作块 $\mathbf{a}_{t:t+H}$。

### Figure 3: 实验环境

![Figure 3](https://arxiv.org/html/2606.19531/2606.19531v1/x3.png)

**说明**: 四个实验环境：RoboTwin 2.0（仿真，含 clean/randomized 设置）、LIBERO（仿真，四个任务套件）、LIBERO-Plus（分布偏移鲁棒性测试）、以及真实世界 Dobot XTrainer 机械臂上的四个任务（叠碗、折毛巾、开抽屉存标记、挂杯）。

### Figure 4: 注意力可视化

![Figure 4](https://arxiv.org/html/2606.19531/2606.19531v1/x4.png)

**说明**: ImageWAM 编辑缓存的注意力集中在任务相关的变化区域（被操作的物体、目标容器、接触区域），有效抑制背景；而 FastWAM 视频缓存的注意力则更为分散，包含大量与动作无关的背景信息。

### Figure 5: 视频生成伪影导致动作失败

![Figure 5](https://arxiv.org/html/2606.19531/2606.19531v1/x5.png)

**说明**: 视频生成 WAM 在任务相关对象周围产生可见的几何扭曲和空间布局不一致，这类伪影污染了动作条件化上下文，直接导致任务失败。ImageWAM 规避了这一问题。

---

### Table 1: RoboTwin 2.0 结果对比

| Method | P.T. | Clean | Rand. | Avg. |
|--------|:----:|------:|------:|-----:|
| π₀ | ✓ | 65.92 | 58.40 | 62.16 |
| π₀.₅ | ✓ | 82.74 | 76.76 | 79.75 |
| ABot-M0 | ✗ | 81.20 | 80.40 | 80.80 |
| Motus | ✓ | 88.66 | 87.02 | 87.80 |
| LingBot-VA | ✓ | 92.90 | 91.50 | 92.20 |
| FastWAM | ✗ | 91.88 | 91.78 | 91.83 |
| **ImageWAM** | **✗** | **93.20** | **93.56** | **93.38** |

**说明**: ImageWAM 在无需预训练的前提下，以 93.38% 平均成功率超越所有基线（包括有预训练的 π₀.₅、LingBot-VA 和 FastWAM），Clean/Randomized 两种设置下均取得最优。

---

### Table 2: LIBERO 结果对比

| Method | P.T. | Spatial | Object | Goal | Long | Avg. |
|--------|:----:|--------:|-------:|-----:|-----:|-----:|
| OpenVLA | ✓ | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| GR00T N1 | ✓ | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π₀ | ✓ | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 |
| π₀.₅ | ✓ | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| LingBot-VA | ✓ | 98.5 | 99.6 | 97.2 | 98.5 | 98.5 |
| Motus | ✓ | 96.8 | 99.8 | 96.6 | 97.6 | 97.7 |
| Fast-WAM | ✗ | 98.2 | 100.0 | 97.0 | 95.2 | 97.6 |
| **ImageWAM** | **✗** | **97.2** | **99.2** | **98.8** | **98.4** | **98.4** |

**说明**: ImageWAM 平均 98.4%，在 Goal 和 Long 套件上超越所有基线，且无需预训练，与 Fast-WAM 相比 Avg 高 0.8 个百分点。

---

### Table 3: LIBERO-Plus 分布偏移鲁棒性

| Method | P.T. | Camera | Robot | Language | Light | BG | Noise | Layout | Avg |
|--------|:----:|-------:|------:|---------:|------:|---:|------:|-------:|----:|
| UniVLA | ✓ | 1.8 | 46.2 | 69.6 | 69.0 | 81.0 | 21.2 | 31.9 | 42.9 |
| OpenVLA-OFT | ✓ | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| π₀ | ✓ | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| π₀-Fast | ✓ | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| WorldVLA | ✓ | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 10.9 | 38.0 | 25.0 |
| FastWAM | ✗ | 16.4 | 44.5 | 68.9 | 78.2 | 53.7 | 37.7 | 60.7 | 51.5 |
| ImageWAM (OmniGen2) | ✗ | 80.0 | 49.2 | 70.9 | 82.6 | 69.4 | 77.1 | 71.8 | 71.8 |
| ImageWAM (Ovis-U1) | ✗ | 63.3 | 58.4 | 75.4 | 86.3 | 66.7 | 75.2 | 74.6 | 71.2 |
| **ImageWAM (FLUX.2 4B)** | **✗** | **80.8** | **50.3** | **91.4** | **98.1** | **85.5** | **93.8** | **80.5** | **83.1** |

**说明**: ImageWAM 在 7 种分布偏移类型上全面领先，FLUX.2 4B 版本平均 83.1%，远超有预训练的 OpenVLA-OFT（69.6%）。Camera 和 Robot 变体对所有方法均最难，但 ImageWAM 依然取得最优。

---

### Table 4: 真实机器人评估成功率（%）

| Method | T1(叠碗) | T2(折毛巾) | T3(开抽屉存标记) | T4(挂杯) | Avg |
|--------|:--------:|:---------:|:--------------:|:--------:|----:|
| π₀ | 57 | 58 | 54 | 54 | 55.8 |
| π₀.₅ | 83 | 77 | 74 | 55 | 72.3 |
| FastWAM | 88 | 75 | 77 | 76 | 79.0 |
| **ImageWAM** | **94** | **84** | **78** | **82** | **84.5** |

**说明**: 真实机器人实验中 ImageWAM 全面超越基线，四个任务均取得最优，平均 84.5% vs FastWAM 79.0%。

---

### Table 5: 效率对比

| Method | 延迟 (ms) | TFLOPs | 中间表示 |
|--------|----------:|-------:|---------|
| FastWAM-IDM | 1081 | 63.65 | 视频帧 |
| FastWAM (1 步) | 302 | 13.21 | KV Cache |
| **ImageWAM** | **263** | **9.72** | **KV Cache** |

**说明**: ImageWAM 相比视频生成 WAM 减少 FLOPs 至 1/6（9.72 vs 63.65），延迟降至 1/4（263ms vs 1081ms）；相比 FastWAM 仍有 15% 延迟改善和 26% FLOPs 节省。

---

### Table 6: 统一理解-生成模型对比

| Method | P.T. | LIBERO | RoboTwin Clean | RoboTwin Clean2Hard |
|--------|:----:|-------:|:--------------:|:-------------------:|
| UniVLA | ✓ | 95.5 | – | – |
| BagelVLA (w/ K.F.) | ✓ | – | 75.3 | 20.9 |
| BagelVLA (w/o K.F.) | ✓ | – | 56.7 | 15.9 |
| **ImageWAM** | **✗** | **98.4** | **84.4** | **18.3** |

**说明**: 与统一理解-生成模型（UniVLA、BagelVLA）对比，ImageWAM 在 LIBERO 和 RoboTwin Clean 上均更优，支持了"解耦设计优于统一模型"的假设。

---

### Table 7: 更大编辑骨干效果（LIBERO-Plus）

| Method | P.T. | Camera | Robot | Language | Light | BG | Noise | Layout | Avg |
|--------|:----:|-------:|------:|---------:|------:|---:|------:|-------:|----:|
| ImageWAM (FLUX.2 4B) | ✗ | 80.8 | 50.3 | 91.4 | 98.1 | 85.5 | 93.8 | 80.5 | 83.1 |
| **ImageWAM (FLUX.2 9B)** | **✗** | **79.8** | **58.7** | **95.2** | **96.1** | **91.2** | **93.3** | **83.1** | **85.2** |

**说明**: 4B → 9B 骨干提升 2.1 个百分点（83.1 → 85.2），在 Robot、Language、Background 偏移上增益最显著，验证"更大编辑模型带来更强鲁棒性"。

---

### Table 8: 训练超参数

| 参数 | 值 |
|------|---|
| GPU | 8× NVIDIA H20 |
| 分布式策略 | DeepSpeed ZeRO-1（大模型用 ZeRO-2）|
| 精度 | bf16 |
| 优化器 | AdamW，betas=(0.9, 0.95) |
| 学习率 | $1\times10^{-4}$ |
| 权重衰减 | $1\times10^{-2}$ |
| LR 调度 | Warmup Cosine（warmup=5% 总步数，最小 LR=1%）|
| 梯度裁剪 | 1.0 |

---

### Table 9: 数据集配置

| 参数 | LIBERO | RoboTwin |
|------|--------|---------|
| 输入视角 | 2 视角 | 3 视角 |
| 视角布局 | 水平拼接 | 腕部+水平+竖直 |
| 输入分辨率 | 224×448 | 288×256 |
| 未来视野 | 16 帧 | 16 帧 |
| 动作块长度 | 16 | 16 |
| 训练轮数 | 10 | 5 |

---

### Table 10: 推理延迟详细对比（含编译优化）

| 变体 | 延迟 (ms) | 加速比 |
|------|----------:|------:|
| FastWAM (1× 视频去噪) | 302 | 1.00× |
| ImageWAM (1× 视频去噪) | 263 | 1.15× |
| FastWAM (Prefix Only) | 194 | 1.56× |
| + Compiled | 80 | 3.78× |
| ImageWAM (Prefix Only) | 198 | 1.53× |
| + Action Loop Compile | 85 | 3.55× |
| + Image Prefill Compile | 77 | 3.92× |
| + Action Static Graph | 69 | 4.38× |

**说明**: 在完整编译优化（Static Graph）下 ImageWAM 可达 69ms（4.38× 加速），具备实时部署潜力。

---

## 实验

### 数据集

| 数据集 | 类型 | 特点 | 用途 |
|--------|------|------|------|
| RoboTwin 2.0 | 仿真 | Clean/Randomized 双设置，多任务 | 主要仿真评估 |
| LIBERO | 仿真 | 4 个任务套件（Spatial/Object/Goal/Long） | 综合性能评估 |
| LIBERO-Plus | 仿真 | 7 种分布偏移（相机/机器人/语言/光照/背景/噪声/布局） | 鲁棒性评估 |
| Real-World | 真实 | Dobot XTrainer 机械臂，4 个任务 | 真实场景验证 |

### 实现细节

- **骨干**: OmniGen2 / Ovis-U1 / FLUX.2 4B / FLUX.2 9B（可互换）
- **优化器**: AdamW，lr=$1\times10^{-4}$
- **Batch Size**: 每 GPU 10–48 样本（视任务而定）
- **训练时间**: LIBERO 18 小时，RoboTwin 5 天（8× H20）
- **硬件**: 8 NVIDIA H20 GPU，DeepSpeed ZeRO-1/2

### 可视化结果

注意力可视化（Figure 4）表明，ImageWAM 编辑缓存的注意力热图高度集中于任务相关对象和接触区域，证明图像编辑预训练自然产生了动作感知的视觉表示。相比之下，视频 WAM 的注意力更加分散，包含大量背景信息。

---

## 批判性思考

### 优点

1. **计算效率显著**: FLOPs 1/6、延迟 1/4，工程意义突出，有真实部署价值
2. **无需预训练即超越有预训练基线**: 说明图像编辑预训练的知识迁移极为高效
3. **骨干无关设计**: 兼容多种图像编辑模型，随编辑模型社区进步自动受益
4. **真实机器人验证有力**: 四个多样化任务的真实机器人实验增强了可信度

### 局限性

1. **固定去噪时间步 $\tau^*$**: 论文中对最优 $\tau^*$ 的选择依据描述不够详细，不同任务是否需要调整 $\tau^*$？
2. **仅单帧终态建模**: 对于需要精确中间状态的复杂长程任务（如多步骨牌任务），单帧是否足够？
3. **图像编辑骨干的幻觉风险**: 图像编辑模型偶尔生成语义正确但几何不准确的帧，可能引入与视频伪影类似的问题

### 潜在改进方向

1. **自适应时间步选择**: 根据任务复杂度动态选择 $\tau^*$，而非固定单一值
2. **多帧稀疏采样**: 关键中间状态帧 + 终态帧，兼顾效率与复杂任务建模能力
3. **与视觉语言推理结合**: 在图像编辑前加入 CoT 推理步骤，进一步提升长程规划能力

### 可复现性评估

- [ ] 代码开源（项目主页存在但代码未见）
- [ ] 预训练模型（未提供）
- [x] 训练细节完整（超参数、硬件、时间均有详细记录）
- [x] 数据集可获取（RoboTwin 2.0、LIBERO 均为公开数据集）

---

## 关联笔记

### 基于

- [[FastWAM]]: 主要对比方法，视频 WAM 的代表，ImageWAM 继承其 KV Cache 条件化范式但以图像编辑替换视频生成
- [[Flow Matching]]: 动作预测和图像编辑均采用流匹配目标
- [[FLUX.2]]: 最强编辑骨干，4B 和 9B 版本均在 LIBERO-Plus 取得最优结果

### 对比

- [[π₀]]: 最常见基线 VLA，ImageWAM 在所有 benchmark 上全面超越
- [[π₀.₅]]: 更强的 VLA 基线，ImageWAM 在 RoboTwin 和 LIBERO 上均优于它
- [[UniVLA]]: 统一理解-生成模型代表，ImageWAM 的解耦设计优于它
- [[WorldVLA]]: 另一视频 WAM 基线，LIBERO-Plus 上 ImageWAM 遥遥领先

### 方法相关

- [[图像编辑模型]]: 核心骨干
- [[KV Cache]]: 编辑骨干的中间表示，用于条件化动作专家
- [[Action Expert]]: 动作解码模块
- [[Flow Matching]]: 训练和推理的核心算法
- [[Joint Attention]]: 将编辑缓存注入动作专家的机制
- [[World Action Model]]: ImageWAM 所属的方法类别

### 硬件/数据相关

- [[RoboTwin 2.0]]: 主要仿真评估环境
- [[LIBERO]]: 综合性能评估 benchmark
- [[Dobot XTrainer]]: 真实机器人平台

---

## 速查卡片

> [!summary] ImageWAM
> - **核心**: 用图像编辑替代视频生成作为 WAM 的世界模型骨干
> - **方法**: 提取图像编辑扩散模型的 KV Cache，经 Joint Attention 条件化流匹配动作专家
> - **结果**: RoboTwin 93.38%、LIBERO 98.4%、真实机器人 84.5%；FLOPs 降至视频 WAM 的 1/6
> - **代码**: 暂未开源（[项目主页](https://zhangwenyao1.github.io/ImageWAM/)）

---

*笔记创建时间: 2026-06-22*
