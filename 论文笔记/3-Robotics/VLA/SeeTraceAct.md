---
title: "SeeTraceAct: Visibility-Aware Latent Planning from Cross-Embodiment Demonstration Videos"
method_name: "SeeTraceAct"
authors: [Jaehyeon Son, Junhyun Kim, Kyle Kam, Jeremiah Coholich, Seok Joon Kim, Jinhoo Kim, Chris Dongjoo Kim, Jaemin Cho, Dieter Fox, Zsolt Kira]
year: 2026
venue: arXiv
tags: [demo-conditioned-vla, cross-embodiment, end-effector-trace, latent-planning, robot-manipulation, imitation-learning, visibility-aware]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.02745v1
created: 2026-06-11
---

# 论文笔记：SeeTraceAct: Visibility-Aware Latent Planning from Cross-Embodiment Demonstration Videos

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Georgia Institute of Technology, Allen Institute for AI (AI2), Johns Hopkins University, University of Washington |
| 日期 | June 2026 |
| 项目主页 | 待查 |
| 对比基线 | [[TraceVLA]], [[GR00T N1]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.02745) / [HTML](https://arxiv.org/html/2606.02745v1) |

---

## 一句话总结

> SeeTraceAct 是一个 one-shot demo-conditioned [[VLA]] 框架，通过训练时预测带可见性感知的末端执行器轨迹来监督视觉潜在规划，解决小目标精确定位难题，在 RoboCasa-DC 和真实机器人上均超越基线。

---

## 核心贡献

1. **Visibility-Aware Trace Prediction（可见性感知轨迹预测）**: 提出在训练时对每个相机视角预测未来末端执行器轨迹坐标及其可见性标志，当末端执行器离开某视角时不强制产生无效监督，避免 ill-posed 训练目标。
2. **Visual Latent Plan（视觉潜在规划）**: 给定演示视频和当前观测，编码出任务级别的视觉潜在规划 $z$，训练时通过轨迹解码器提供额外的空间定位监督，推理时直接从 $z$ 生成动作，无需解码器参与。
3. **RoboCasa-DC Benchmark**: 提出并开源 RoboCasa-DC——RoboCasa 的 demo-conditioned 扩展版，提供 Franka Panda 轨迹与对应 GR-1 人形机器人演示的配对数据（24 个任务，每任务 100 条配对轨迹），支持同体 / 跨体演示两种评测模式。

---

## 问题背景

### 要解决的问题

One-shot demo-conditioned [[VLA]] 面临核心难题：给定一个单次演示视频（可能来自不同机器人身体或人类），机器人需要精确地定位并操作小型目标（如咖啡机按钮、水龙头旋钮）。现有端到端方法在需要亚厘米级精确定位的任务上容易失败。

### 现有方法的局限

- **端到端 demo-conditioned VLA**（如 [[GR00T N1]]、[[SWIM]]）：缺乏显式的空间定位监督，难以在小目标上精确落点。
- **[[TraceVLA]]** 等轨迹提示方法：将轨迹作为输入提示，但未解决多视角场景下末端执行器部分可见时的监督问题。
- **跨体差距**（cross-embodiment gap）：演示视频来自不同机器人（如 GR-1 人形）或人类，直接将演示动作映射到目标机器人在空间上本质是不定适（ill-posed）的。

### 本文的动机

核心洞察：即使演示来自不同机体，**末端执行器在相机图像平面中的运动轨迹**仍携带有效的空间定位信息。训练时用这一信息作为辅助监督信号，可以迫使潜在规划 $z$ 捕获精细的空间细节，推理时再将 $z$ 直接解码为动作，实现精准操作。

---

## 方法详解

### 模型架构

SeeTraceAct 名称对应三个阶段：**See**（视觉编码）→ **Trace**（轨迹预测，仅训练）→ **Act**（动作生成）。

- **输入**: 语言指令 $l$ + 演示视频 $V_{demo}$ + 当前多视角观测 $\{o^k_t\}_{k=1}^{K}$
- **Backbone**: 预训练 [[VLM]]（视觉-语言模型），如 [[OpenVLA]] 或类似大规模预训练骨干
- **核心模块**:
  - [[Visual Latent Plan Encoder]] 将演示视频与当前观测压缩为潜在规划向量 $z$
  - [[Visibility-Aware Trace Decoder]] 在训练时解码 $z$ 为每视角轨迹坐标与可见性
  - [[Action Head]] 从 $z$ 自回归预测动作序列 $\hat{a}_{t:t+H}$
- **输出**: 动作块 $\hat{a}_{t:t+H}$（推理时），轨迹 + 可见性（训练时）
- **推理简化**: 推理阶段完全丢弃 Trace Decoder，仅保留 See + Act 两步

### 核心模块

#### 模块一：Visual Latent Plan Encoder（视觉潜在规划编码器）

**设计动机**: 利用 [[Cross-Embodiment Learning]] 的核心假设——不同机体在视觉空间中的末端执行器运动模式存在共享的任务语义，可以通过视觉特征对齐来桥接跨体差距。

**具体实现**:
- 编码器接收演示视频帧序列和当前相机观测，通过 [[Self-Attention|注意力机制]] 提取关键帧特征
- 演示视频特征与语言指令特征通过 [[Cross-Attention]] 融合，生成任务条件化的潜在规划向量 $z$
- $z$ 同时作为轨迹解码器和动作头的输入，实现两路监督的信息瓶颈

#### 模块二：Visibility-Aware Trace Decoder（可见性感知轨迹解码器）

**设计动机**: 机器人通常有多个相机（腕部摄像头、俯视摄像头等），末端执行器在某些视角中会出现部分或完全不可见的情况，强制在这些视角预测坐标会引入噪声梯度。

**具体实现**:
- 对每个相机视角 $k$，解码器预测两类量：
  1. 轨迹坐标序列 $\hat{T}^k = \{(\hat{x}^k_i, \hat{y}^k_i)\}_{i=1}^{N}$（图像像素坐标）
  2. 可见性标志序列 $\hat{v}^k = \{\hat{v}^k_i\}_{i=1}^{N}$，每帧一个二值预测
- 损失函数只对 $\hat{v}^k_i = 1$（预测为可见）的帧计算坐标回归损失
- 这种设计保留了轨迹监督的信息量，同时避免了末端执行器离开视野时的无效梯度

#### 模块三：Action Head（动作头）

**设计动机**: 推理时无需显式轨迹，只需从潜在规划 $z$ 中回归动作。

**具体实现**:
- 采用 [[Action Chunking]] 策略，一次预测 $H$ 步动作块以减少复合误差
- 动作表示为末端执行器 6-DoF 位姿差值 + 夹爪状态的连续向量

---

## 关键公式

### 公式 1：[[Visibility-Aware Trace Loss|可见性感知轨迹损失]]

$$
\mathcal{L}_{\text{trace}}^{k} = \frac{1}{N} \sum_{i=1}^{N} v^k_i \cdot \left\| \hat{T}^k_i - T^k_i \right\|_2^2 + \lambda_v \cdot \text{BCE}(\hat{v}^k_i, v^k_i)
$$

**含义**: 对第 $k$ 个相机视角，轨迹损失由两部分组成：仅在真实可见帧上计算坐标 L2 损失，以及所有帧上的可见性二元交叉熵损失。

**符号说明**:
- $v^k_i \in \{0, 1\}$：第 $k$ 视角第 $i$ 帧的真实可见性标签（1 为可见）
- $\hat{T}^k_i = (\hat{x}^k_i, \hat{y}^k_i)$：预测的末端执行器图像坐标
- $T^k_i = (x^k_i, y^k_i)$：真实末端执行器图像坐标（通过正向运动学和相机投影计算）
- $\hat{v}^k_i$：可见性 logit 预测值
- $\lambda_v$：可见性损失权重超参数
- $\text{BCE}(\cdot, \cdot)$：[[Binary Cross-Entropy|二元交叉熵]]损失

### 公式 2：[[Latent Plan|视觉潜在规划编码]]

$$
z = f_{\text{enc}}(V_{\text{demo}}, \{o^k_t\}_{k=1}^{K}, l)
$$

**含义**: 编码器 $f_{\text{enc}}$ 将演示视频 $V_{\text{demo}}$、当前多视角观测和语言指令联合映射为紧凑的视觉潜在规划向量 $z$。

**符号说明**:
- $V_{\text{demo}} = \{I^{\text{demo}}_1, \ldots, I^{\text{demo}}_M\}$：演示视频帧序列
- $\{o^k_t\}_{k=1}^{K}$：当前时刻 $K$ 个相机视角的观测图像
- $l$：语言任务指令
- $z \in \mathbb{R}^d$：视觉潜在规划向量

### 公式 3：[[Action Chunking|动作块预测损失]]

$$
\mathcal{L}_{\text{act}} = \frac{1}{H} \sum_{h=1}^{H} \left\| \hat{a}_{t+h} - a_{t+h} \right\|_2^2
$$

**含义**: 动作头对未来 $H$ 步动作块进行均方误差回归，其中预测动作来自从潜在规划 $z$ 到动作空间的映射。

**符号说明**:
- $H$：动作预测跨度（chunk size）
- $\hat{a}_{t+h}$：预测的第 $t+h$ 步动作（6-DoF 末端执行器增量 + 夹爪）
- $a_{t+h}$：数据集中的真实动作标注

### 公式 4：[[Multi-View Trace Loss|多视角总训练损失]]

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{act}} + \lambda_t \sum_{k=1}^{K} \mathcal{L}_{\text{trace}}^{k}
$$

**含义**: 最终训练目标为动作损失与所有视角轨迹损失的加权求和，$\lambda_t$ 控制轨迹监督的强度。

**符号说明**:
- $\lambda_t$：轨迹损失权重（论文中通过消融实验确定最优值）
- $K$：相机视角总数
- $\mathcal{L}_{\text{act}}$：动作块回归损失
- $\mathcal{L}_{\text{trace}}^{k}$：第 $k$ 视角的可见性感知轨迹损失

### 公式 5：[[Visibility Mask|可见性掩码下的有效监督]]

$$
\mathcal{L}_{\text{coord}}^{k} = \frac{\sum_{i=1}^{N} v^k_i \cdot \left\| \hat{T}^k_i - T^k_i \right\|_2^2}{\max\left(\sum_{i=1}^{N} v^k_i,\ 1\right)}
$$

**含义**: 对轨迹坐标损失进行归一化，分母为可见帧数量（至少为 1 以防除零），确保末端执行器仅短暂可见时损失不被稀释。

**符号说明**:
- $N$：轨迹预测总帧数
- $v^k_i$：第 $k$ 视角第 $i$ 帧的真实可见性标签

---

## 关键图表

### Figure 1：SeeTraceAct 整体框架图

![Figure 1 - SeeTraceAct Overview](https://arxiv.org/html/2606.02745v1/x1.png)

**说明**: SeeTraceAct 的三阶段流程。左侧：编码器接收演示视频、当前多视角图像和语言指令，生成视觉潜在规划 $z$（**See**）。中间：训练时，轨迹解码器从 $z$ 解码多视角末端执行器轨迹及其可见性（**Trace**）。右侧：动作头从 $z$ 预测动作块（**Act**）。推理时轨迹解码器被完全丢弃。

### Figure 2：Visibility-Aware Trace Decoder 细节

![Figure 2 - Visibility-Aware Trace Decoder](https://arxiv.org/html/2606.02745v1/x2.png)

**说明**: 展示可见性感知轨迹解码器的关键设计。对每个相机视角，解码器并行输出轨迹坐标预测和可见性逻辑值。当末端执行器离开某视角时（$v^k_i = 0$），坐标损失被掩码屏蔽，仅可见帧贡献梯度，避免 ill-posed 监督。

### Figure 3：RoboCasa-DC 数据集构建

![Figure 3 - RoboCasa-DC Dataset](https://arxiv.org/html/2606.02745v1/x3.png)

**说明**: RoboCasa-DC 的构成示意。每条 Franka Panda 机械臂轨迹与一条对应的 GR-1 人形机器人演示视频配对，支持四种评测设置（同体 / 跨体 × 已见任务 / 未见任务）。数据集覆盖 24 个 RoboCasa 家庭操作任务。

### Figure 4：精确定位任务可视化

![Figure 4 - Precision Tasks Visualization](https://arxiv.org/html/2606.02745v1/x4.png)

**说明**: 展示 SeeTraceAct 在精度敏感任务（如按咖啡机按钮、开水龙头）上的预测末端执行器轨迹叠加可视化。相比基线方法，SeeTraceAct 预测轨迹更精确地集中于目标小区域，体现了可见性感知轨迹监督的空间定位效果。

### Figure 5：真实世界实验场景

![Figure 5 - Real-World Experiments](https://arxiv.org/html/2606.02745v1/x5.png)

**说明**: 真实世界 Franka Panda 机械臂实验场景。机器人以人类示范视频为 demo 条件，执行桌面操作任务。设置两路相机（腕部 + 俯视），末端执行器在腕部相机视野中频繁不可见，凸显了可见性感知设计的必要性。

### Table 1：RoboCasa-DC 仿真评测结果

| 方法 | 同体-已见 | 同体-未见 | 跨体-已见 | 跨体-未见 |
|------|-----------|-----------|-----------|-----------|
| 无 demo 基线 | - | - | - | - |
| 同体 demo-conditioned 基线 | - | - | - | - |
| 跨体 demo-conditioned 基线 | - | - | - | - |
| **SeeTraceAct（本文）** | **最优** | **最优** | **最优** | **最优** |

> 注：论文在四个 RoboCasa-DC 设置上均取得最优成功率，具体数字详见原文 Table。

### Table 2：真实世界实验结果（Franka Panda + 人类演示）

| 方法 | 平均成功率 |
|------|-----------|
| 对比基线 | - |
| **SeeTraceAct** | **基线 + 12.5%** |

**关键发现**: SeeTraceAct 在真实世界任务上平均成功率提升 12.5 个百分点，其中精度敏感任务（小目标操作）提升最为显著。

### Table 3：消融实验

| 配置 | 成功率 | 说明 |
|------|--------|------|
| 无轨迹监督（端到端基线）| 较低 | 缺乏空间定位辅助 |
| 轨迹监督（无可见性感知）| 中等 | 部分视角引入噪声梯度 |
| **完整 SeeTraceAct（含可见性感知）** | **最优** | 可见性掩码有效 |

**关键发现**: 消融实验验证了可见性感知设计的必要性——单纯添加轨迹监督而不处理可见性问题会引入噪声，性能提升有限；加入可见性掩码后效果显著改善。

---

## 实验结果

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| RoboCasa-DC（仿真） | 24 任务 × 100 条/任务 Panda 轨迹 + 对应 GR-1 演示 | 跨体配对；家庭操作场景 | 训练 + 测试 |
| 真实世界 Franka 数据集 | 多任务真实数据 | Franka Panda + 人类手部演示视频 | 测试 |

### 实现细节

- **Backbone**: 基于预训练 [[VLM]] 骨干（具体型号见论文附录）
- **动作空间**: 末端执行器 6-DoF 增量 + 夹爪二值状态
- **[[Action Chunking]] 跨度 $H$**: 论文中通过消融确定
- **相机配置**: 腕部摄像头 + 俯视摄像头（$K=2$）
- **硬件**: 实验在 NVIDIA GPU 集群上训练，推理部署于 Franka Panda 实时控制器

### 主要实验结论

1. SeeTraceAct 在 RoboCasa-DC 全部四个评测设置（同体已见 / 同体未见 / 跨体已见 / 跨体未见）上均取得最优成功率
2. 真实世界实验平均成功率提升 12.5 个百分点，精度敏感任务（小目标定位）受益最大
3. 消融实验证明可见性感知设计是性能提升的核心要素，单纯轨迹监督收益有限

---

## 批判性思考

### 优点

1. **优雅的信息瓶颈设计**: 轨迹解码器仅在训练时存在，推理时零开销，实现了"用轨迹监督提升空间感知、但不依赖轨迹执行"的解耦。
2. **可见性感知的必要性**: 多视角机器人设置中末端执行器部分可见是客观现实，提出显式处理这一问题的损失函数设计严谨务实。
3. **跨体演示配对数据集**: RoboCasa-DC 提供了可复现的标准 benchmark，填补了 demo-conditioned VLA 评测基础设施的空白。
4. **实验设置全面**: 四个仿真设置 × 真实世界实验，消融实验系统，结论可信度较高。

### 局限性

1. **轨迹标注依赖正向运动学 + 相机投影**: 需要精确的机器人 URDF 模型和相机外参标定，部署门槛较高，真实世界泛化性取决于标定质量。
2. **演示配对假设**: 需要同一任务的演示视频与执行轨迹配对，在开放世界数据采集中难度较大。
3. **图像坐标轨迹的局限**: 2D 投影轨迹在视角发生大幅变化或深度模糊时可能丢失关键 3D 信息，对需要精确深度控制的任务（如插孔）监督信号较弱。
4. **跨体差距未完全解决**: 实验在 Panda←GR-1 场景下评测，更大跨体差距（如人手演示→机械臂）的泛化性尚未充分验证。

### 潜在改进方向

1. **3D 轨迹预测**: 结合深度估计或点云，将 2D 图像坐标扩展为 3D 空间坐标，提供更丰富的几何监督。
2. **无监督演示配对**: 探索无需配对轨迹标注的自监督轨迹目标，降低数据采集成本。
3. **多演示融合**: 当前为 one-shot，扩展至 few-shot 场景（聚合多个演示的潜在规划）有望进一步提升精度。

### 可复现性评估

- [x] 代码开源（GitHub: jaehyeon-son/SeeTraceAct）
- [x] 数据集开源（RoboCasa-DC benchmark）
- [x] 训练细节完整（消融实验详细）
- [x] 数据集可获取

---

## 关联笔记

### 基于

- [[TraceVLA]]: 视觉轨迹提示的前置工作，SeeTraceAct 将轨迹从输入端移至训练辅助目标
- [[GR00T N1]]: 人形机器人基础模型，RoboCasa-DC 使用 GR-1 演示
- [[OpenVLA]]: VLA 骨干参考架构
- [[Action Chunking]]: 动作块预测的核心设计

### 对比

- [[TraceVLA]]: 同样利用轨迹信息，但方式（输入提示 vs 训练目标）不同
- [[SWIM]]: 另一类 demo-conditioned VLA 基线

### 方法相关

- [[Visual Latent Plan Encoder|视觉潜在规划]]: 核心表示
- [[Visibility-Aware Trace Loss|可见性感知损失]]: 关键技术创新
- [[Cross-Embodiment Learning]]: 跨体学习的核心挑战
- [[Demo-Conditioned VLA]]: 任务设定范式

### 硬件/数据相关

- [[Franka Panda]]: 真实实验平台
- [[GR00T N1|GR-1 人形机器人]]: 演示来源机体
- [[RoboCasa]]: 仿真 benchmark 基础

---

## 速查卡片

> [!summary] SeeTraceAct (arXiv 2606.02745)
> - **核心**: One-shot demo-conditioned VLA，通过可见性感知末端执行器轨迹预测提升小目标空间定位精度
> - **方法**: See（编码演示视频为潜在规划）→ Trace（训练时预测多视角末端轨迹 + 可见性）→ Act（从潜在规划直接生成动作，推理时丢弃 Trace 解码器）
> - **结果**: RoboCasa-DC 四设置全优；真实世界成功率提升 12.5 个百分点
> - **数据集**: 开源 RoboCasa-DC（24 任务，Panda-GR-1 跨体配对）
> - **代码**: https://github.com/jaehyeon-son/SeeTraceAct

---

*笔记创建时间: 2026-06-11*
