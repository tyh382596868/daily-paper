---
title: "Hallucination in World Models is Predictable and Preventable"
method_name: "MMBench2"
authors: [Nicklas Hansen, Xiaolong Wang]
year: 2026
venue: arXiv
tags: [world-model, hallucination, data-coverage, benchmark, flow-matching, curiosity-driven, visual-world-model]
zotero_collection: 1-生成模型
image_source: mixed
arxiv_html: https://arxiv.org/html/2606.27326
created: 2026-06-26
---

# 论文笔记：Hallucination in World Models is Predictable and Preventable

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | UC San Diego |
| 日期 | June 2026 |
| 项目主页 | https://nicklashansen.com/mmbench2 |
| 对比基线 | [[Dreamer 4]]、SD-VAE、[[Cosmos3]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.27326) / [Code](https://github.com/nicklashansen/mmbench2) / [Dataset](https://huggingface.co/datasets/nicklashansen/mmbench2) |

---

## 一句话总结

> 生成式世界模型的幻觉本质上是数据覆盖问题，可以用无标注运行时信号预测，并通过覆盖感知训练与好奇心驱动采集来消除——仅需 50 条轨迹即可适配未见环境。

---

## 核心贡献

1. **MMBench2 基准数据集**: 427 小时、210 任务、10 个领域、65,600 条轨迹，提供真实动作/奖励和可运行仿真器，用于评估视觉[[世界模型]]质量
2. **三类幻觉模式的系统表征**: 将世界模型幻觉分解为感知幻觉、动作边缘化幻觉、场景发散幻觉，与三阶段 pipeline 一一对应
3. **无监督幻觉预测信号**: 三个运行时可计算的指标（$u_r$、$u_f$、$u_s$）与展开误差 Spearman 相关系数 ρ ≈ 0.80
4. **覆盖感知缓解策略**: 训练时任务均匀重采样 + 推理时好奇心驱动数据采集，50 条轨迹达到 oracle 的 ~90% 性能

---

## 问题背景

### 要解决的问题

生成式视觉[[世界模型]]能生成视觉连贯但动力学不准确的"幻觉展开"（hallucinated rollout）：画面逼真，却与真实物理动力学脱节，导致下游的[[Model-Based RL|基于模型强化学习]]和[[MPC|模型预测控制]]失效。

### 现有方法的局限

- 现有世界模型研究专注于视觉质量（FVD / PSNR），缺乏系统性幻觉诊断框架
- 没有大规模跨域 benchmark 能同时提供真实动作标注和在线仿真器
- 幻觉被当作模型架构缺陷处理，而非数据问题

### 本文的动机

幻觉集中在状态-动作空间的低覆盖区域。这意味着：
1. **预测**：可以用反映数据覆盖度的运行时信号（无需标注）检测幻觉发生位置
2. **预防**：用相同信号指导数据采集或重采样，填补覆盖缺口

---

## 方法详解

### 模型架构

本文训练了一个 350M 参数的视觉[[世界模型]]，采用两阶段 pipeline：

- **输入**: 224×224 RGB 帧序列 + 控制动作（最高 $d_a = 16$ 维）+ 语言任务指令（[[CLIP]] ViT/B 嵌入）
- **Stage 1**: [[视频分词器|视频 Tokenizer]]（100M 参数）—— 对称编码器-解码器 Transformer
- **Stage 2**: [[动态模型|Dynamics Model]]（250M 参数）—— 因果 Transformer + [[Shortcut Flow Matching]]
- **输出**: 未来帧潜码 $\hat{z}$，以及可选的奖励预测与行为克隆策略
- **总参数**: 350M

### 核心模块

#### 模块 1：视频 Tokenizer

**设计动机**: 将连续像素帧压缩为紧凑连续潜码 $z \in [-1, 1]^{64 \times 64}$，供下游动力学模型使用。

**具体实现**:
- 编码器：14-stride patchification → 256 个图像块 token + 64 个可学习查询 → tanh 激活至 $[-1,1]^{64 \times 64}$
- 解码器：从潜码重建原始帧
- 训练目标：MAE 风格的[[Masked Reconstruction|遮掩重建]]（每帧随机遮掩 $p \sim \mathcal{U}(0, 0.9)$ 比例的图像块），损失为 pixel MSE + [[LPIPS]]，按运行 RMS 归一化
- 预训练：300k steps，14 GPU-days（H100）

#### 模块 2：动力学模型（Dynamics Model）

**设计动机**: 在冻结 Tokenizer 的潜空间上，通过[[Shortcut Flow Matching|捷径流匹配]]预测下一帧潜码，实现可控的未来状态预测。

**每个时间步的输入 token**:
- 动作 token：2 层 MLP，将 $d_a = 16$ 维填充动作压缩
- 捷径条件 token：编码噪声水平 $\sigma$ 和步长 $d$
- 空间潜码 token：32 个打包后的空间潜 token
- 寄存器 token：4 个
- 可选 Agent token：奖励/BC 读出头

**训练**:
- 目标：[[Shortcut Flow Matching]] 单步回归 + 自一致性 bootstrap
- 推理：4 步 Euler 子步采样下一帧
- 额外预测头：L=8 多步离散回归（symlog two-hot 编码）预测奖励；确定性高斯策略 + MSE 行为克隆
- 预训练：180k steps，24 GPU-days

**上下文长度**: $T = 24$ 帧

---

## 幻觉表征：三种失效模式

### 4.1 感知幻觉（Perceptual Hallucination）

- **发生阶段**: 编码器-解码器阶段，$H = 0$ 时即存在
- **机制**: Tokenizer 将分布外场景投影到最近的训练分布样本
- **例子**: 未见过的迷宫布局被重建为不同的墙壁配置

### 4.2 动作边缘化幻觉（Action-Marginalized Hallucination）

- **发生阶段**: 动力学模型阶段
- **机制**: 预测潜码对输入动作不敏感，坍缩为动作边缘化分布
- **特征**: 随机打乱输入动作（action shuffling）对输出影响极小
- **例子**: 无论向左还是向右操控，机械臂展开结果相同

### 4.3 场景发散幻觉（Scene-Diverging Hallucination）

- **发生阶段**: 多步展开阶段（$H \geq 1$）
- **机制**: 错误在低覆盖区域累积，产生物理上不可能的事件
- **例子**: Pong 中球瞬间传送，物体违反惯性消失或出现

---

## 关键公式

### 公式 1：[[Round-Trip Residual|分词器往返残差]] $u_r$

$$
u_r = \|\hat{z} - \mathrm{Encode}(\mathrm{Decode}(\hat{z}))\|
$$

**含义**: 将动力学预测的潜码 $\hat{z}$ 解码再重编码，残差大说明 $\hat{z}$ 落在 Tokenizer 重建流形之外，对应感知幻觉。

**符号说明**:
- $\hat{z}$：动力学模型预测的下一帧潜码
- $\mathrm{Decode}(\cdot)$：Tokenizer 解码器
- $\mathrm{Encode}(\cdot)$：Tokenizer 编码器

---

### 公式 2：[[Flow Instability|流不稳定性]] $u_f$

$$
u_f = \frac{1}{S/2} \sum_{s=S/2}^{S-1} \| \hat{x}_1^{(s+1)} - \hat{x}_1^{(s)} \|
$$

**含义**: 测量 Euler 积分后半段各步中去噪器对清洁目标 $\hat{x}_1$ 预测的波动程度。动作条件良好的预测快速收敛（低 $u_f$）；动作边缘化时持续震荡（高 $u_f$）。

**符号说明**:
- $S$：Euler 积分总步数（本文为 4）
- $\hat{x}_1^{(s)}$：第 $s$ 步时去噪器预测的清洁目标

---

### 公式 3：[[Inter-Seed Variance|跨噪声种子方差]] $u_s$

$$
u_s = \mathrm{Var}_{k}\bigl[\hat{z}^{(k)}\bigr], \quad k = 1, \ldots, N
$$

**含义**: 固定（上下文, 动作）对，用 $N$ 个不同噪声种子独立去噪，预测结果分散说明该区域训练数据不足，对应场景发散幻觉的认知不确定性。

**符号说明**:
- $k$：噪声种子索引
- $\hat{z}^{(k)}$：第 $k$ 个种子下的下一帧潜码预测

---

### 公式 4：[[Motion-Normalized Predictor|运动归一化幻觉预测量]] $u^{\text{norm}}$

$$
u^{\text{norm}} = \frac{u}{m}
$$

**含义**: 用场景运动幅度 $m$ 对幻觉预测量归一化，消除高动态场景天然产生的高预测值干扰。

**符号说明**:
- $u$：$u_r$、$u_f$ 或 $u_s$ 之一
- $m$：逐步 RMS 潜表示变化量（可用数据集逐任务均值或在线估计）

---

## 关键图表

### Figure 1：三类幻觉模式的可视化

> 🖼️ **Figure 1a: 感知幻觉（编码重建失真）** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.27326)）

> 🖼️ **Figure 1b: 感知幻觉（解码重建失真）** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.27326)）

![Figure 1 - 动作边缘化（动作被忽略，向下）](https://arxiv.org/html/2606.27326v1/visualizations/hallucination/hallucination-maniskill-action-down2.png)

![Figure 1 - 动作边缘化（动作被忽略，向上）](https://arxiv.org/html/2606.27326v1/visualizations/hallucination/hallucination-maniskill-action-up2.png)

> 🖼️ **Figure 1e: 场景发散（Pong 球瞬移，t=34）** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.27326)）

> 🖼️ **Figure 1f: 场景发散（Pong 球瞬移，t=36）** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.27326)）

**说明**: 三类幻觉对应 pipeline 的三个不同阶段：感知幻觉在 $H=0$ 时的编码-解码阶段发生；动作边缘化在动力学模型阶段导致控制失效；场景发散在多步展开累积误差后产生物理不合理事件。

---

### Figure 2：MMBench2 任务样本（210 任务中的 36 个）

![Figure 2 - bird attack](https://arxiv.org/html/2606.27326v1/visualizations/tasks/pygame-bird-attack-2.png)
![Figure 2 - open drawer](https://arxiv.org/html/2606.27326v1/visualizations/tasks/rd-open-drawer-1.png)
![Figure 2 - whirlpool](https://arxiv.org/html/2606.27326v1/visualizations/tasks/pygame-whirlpool-1.png)
![Figure 2 - anymal reach](https://arxiv.org/html/2606.27326v1/visualizations/domains/maniskill/ms-anymal-reach.png)
![Figure 2 - point maze](https://arxiv.org/html/2606.27326v1/visualizations/domains/pygame/pygame-point-maze-var1.png)
![Figure 2 - bipedal walker](https://arxiv.org/html/2606.27326v1/visualizations/domains/box2d/bipedal-walker-rugged.png)

**说明**: MMBench2 覆盖 10 个领域：DMControl、DMControl-Extended、Meta-World、ManiSkill3、MuJoCo、MiniArcade（Pygame）、Box2D、RoboDesk、OGBench、Atari，包含连续控制、操作、游戏等多种形态。

---

### Figure 3：数据集组成（各任务帧数）

![Figure 3 - 数据集组成](https://arxiv.org/html/2606.27326v1/x1.png)

**说明**: MMBench2 总计 65,600 条轨迹（23M 帧，224×224），混合专家策略、人类游玩和随机策略数据，每个任务的帧数分布不均——这种不均匀性正是覆盖感知重采样要解决的问题。

---

### Figure 4：数据覆盖度与幻觉的相关性

![Figure 4 - Point Maze 状态密度](https://arxiv.org/html/2606.27326v1/visualizations/panels/og-point-maze_density.png)
![Figure 4 - Point Maze 往返残差热图](https://arxiv.org/html/2606.27326v1/visualizations/panels/og-point-maze_hallu_u_r.png)
![Figure 4 - Rocket Collect 状态密度](https://arxiv.org/html/2606.27326v1/visualizations/panels/pygame-rocket-collect_density.png)
![Figure 4 - Rocket Collect 往返残差热图](https://arxiv.org/html/2606.27326v1/visualizations/panels/pygame-rocket-collect_hallu_u_r.png)
![Figure 4 - Cup Catch 状态密度](https://arxiv.org/html/2606.27326v1/visualizations/panels/cup-catch_density.png)
![Figure 4 - Cup Catch 往返残差热图](https://arxiv.org/html/2606.27326v1/visualizations/panels/cup-catch_hallu_u_r.png)

**说明**: 跨三个任务的状态密度热图与 $u_r$ 残差热图对比。低密度区域（数据稀疏）精确对应高 $u_r$ 区域（高幻觉风险），验证了"幻觉 = 覆盖缺口"的核心假设。

---

### Figure 5：幻觉预测量追踪展开误差

![Figure 5 - 三个预测量 vs 展开误差](https://arxiv.org/html/2606.27326v1/x2.png)

**说明**: 在 99k 条保留序列上，$u_r^{\text{norm}}$、$u_f^{\text{norm}}$、$u_s^{\text{norm}}$ 三个指标与展开 PSNR 增益（$\Delta P$）的 Spearman 相关系数均约为 **ρ ≈ 0.80**，验证了无监督幻觉预测的可行性。

---

### Figure 6：不同数据采集策略的状态覆盖对比

![Figure 6 - Cup Catch 专家策略覆盖](https://arxiv.org/html/2606.27326v1/visualizations/data-collection-panels/cup-catch-var1_expert.png)
![Figure 6 - Cup Catch 好奇心驱动覆盖](https://arxiv.org/html/2606.27326v1/visualizations/data-collection-panels/cup-catch-var1_curiosity_u_r_norm.png)
![Figure 6 - Cup Catch 人类游玩覆盖](https://arxiv.org/html/2606.27326v1/visualizations/data-collection-panels/cup-catch-var1_human.png)
![Figure 6 - Reacher 专家策略覆盖](https://arxiv.org/html/2606.27326v1/visualizations/data-collection-panels/pygame-reacher-easy_expert.png)
![Figure 6 - Reacher 好奇心驱动覆盖](https://arxiv.org/html/2606.27326v1/visualizations/data-collection-panels/pygame-reacher-easy_curiosity_u_r_norm.png)
![Figure 6 - Reacher 人类游玩覆盖](https://arxiv.org/html/2606.27326v1/visualizations/data-collection-panels/pygame-reacher-easy_human.png)

**说明**: 专家策略数据覆盖度集中（偏向成功轨迹），好奇心驱动采集主动探索低密度区域，人类游玩介于两者之间。好奇心采集在仅 50 条轨迹时实现与 oracle（专家）相当的幻觉缓解效果。

---

### Figure 7：10 个任务域的代表样本

| 域 | 样本 |
|---|---|
| DMControl | ![](https://arxiv.org/html/2606.27326v1/visualizations/domains/dmcontrol/quadruped-run.png) |
| DMControl-Extended | ![](https://arxiv.org/html/2606.27326v1/visualizations/domains/dmcontrol-extended/spinner-spin.png) |
| Meta-World | ![](https://arxiv.org/html/2606.27326v1/visualizations/domains/metaworld/mw-stick-pull.png) |
| ManiSkill3 | ![](https://arxiv.org/html/2606.27326v1/visualizations/domains/maniskill/ms-poke-cube.png) |
| MuJoCo | ![](https://arxiv.org/html/2606.27326v1/visualizations/domains/mujoco/mujoco-walker.png) |
| MiniArcade | ![](https://arxiv.org/html/2606.27326v1/visualizations/domains/pygame/pygame-coconut-dodge.png) |
| Box2D | ![](https://arxiv.org/html/2606.27326v1/visualizations/domains/box2d/bipedal-walker-obstacles.png) |
| RoboDesk | ![](https://arxiv.org/html/2606.27326v1/visualizations/domains/robodesk/rd-open-slide.png) |
| OGBench | ![](https://arxiv.org/html/2606.27326v1/visualizations/domains/ogbench/og-antball.png) |
| Atari | ![](https://arxiv.org/html/2606.27326v1/visualizations/domains/pygame/pygame-pong.png) |

---

### Table 1：覆盖感知训练的效果（vs 基础模型，200 任务均值变化量）

| 指标 | 仅微调 Tok | 仅微调 Dyn | 两者同调 |
|------|-----------|-----------|---------|
| 重建 PSNR (dB) ↑ | +0.46 | -0.01 | **+0.44** |
| 动作打乱比例 ↑ | +0.02 | +0.27 | **+0.29** |
| 展开 ΔP (dB) ↑ | +0.42 | +0.68 | **+0.88** |
| $u_r^{\text{norm}}$ ↓ | -0.07 | -0.16 | **-0.20** |
| $u_f^{\text{norm}}$ ↓ | -0.03 | -0.06 | **-0.07** |
| $u_s^{\text{norm}}$ ↓ | -0.06 | -0.13 | **-0.14** |

**关键发现**: Tokenizer 和 Dynamics Model 均受益于覆盖感知重采样。联合微调（Both）在所有指标上最优，对应 30k Tokenizer + 30k Dynamics 额外训练步数。

---

### Table 2：未见任务的目标数据采集对比（每方法 50 条轨迹）

| 数据来源 | 重建 PSNR ↑ | 展开 ΔP ↑ | 动作打乱比 ↑ | $u_r^{\text{norm}}$ ↓ | MPC 任务性能 ↑ |
|---------|-----------|---------|-----------|-----------------|--------------|
| Base（无微调） | 17.37 | -12.44 | 1.12 | 3.860 | — |
| 覆盖感知重采样 | 17.21 | -12.52 | 1.29 | 3.769 | 0.276 |
| No-op 动作 | 17.21 | -11.66 | 1.41 | 4.175 | — |
| 随机策略 | 35.81 | +2.66 | 2.00 | 1.201 | 0.228 |
| 专家策略 | 35.86 | +2.84 | 2.04 | 1.131 | 0.362 |
| 人类游玩 | 37.11 | +3.89 | **2.42** | 1.002 | 0.362 |
| **好奇心 ($u_r^{\text{norm}}$)** | 36.05 | +3.00 | 2.00 | 1.144 | **0.325** |
| All（组合） | **37.91** | **+4.02** | 2.34 | **0.975** | **0.390** |

**关键发现**: 好奇心驱动采集无需任何特权行为信号，即达到专家策略 oracle 的约 **90%** 性能（0.325 vs 0.362）；组合所有来源效果最佳。

---

### Table 3：与现成 Tokenizer 的对比

| Tokenizer | 参数 | 已见 PSNR | 未见 PSNR | 差距 ΔS-U |
|-----------|------|---------|---------|---------|
| Ours（基础） | 102M | 38.29 | 17.34 | +20.95 |
| Ours（覆盖感知） | 102M | 38.93 | 17.12 | +21.81 |
| **Ours（微调后）** | **102M** | **39.66** | **38.04** | **+1.62** |
| SD-VAE-MSE | 84M | 33.32 | 32.39 | +0.93 |
| Cosmos-CV8x8x8 | 106M | 32.80 | 32.72 | +0.08 |
| Wan 2.1 VAE | 127M | 36.45 | 36.62 | -0.17 |
| DC-AE-f32c32 | 323M | 31.49 | 32.15 | -0.66 |

**关键发现**: 领域内训练的 Tokenizer 在训练任务上远超通用模型，但泛化极差（差距 +21 dB）；目标微调后泛化差距从 +21.81 dB 缩小至 +1.62 dB，超越所有现成方案。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| MMBench2 | 427h / 65,600 轨迹 / 23M 帧 | 10 域 210 任务，含真实动作、奖励和仿真器 | 训练 + 评估 |
| 200 保留任务 | 99k 序列 | 与训练任务重叠，评估检测信号 | 幻觉检测验证 |
| 10 未见任务 | — | 完全分布外 | 泛化与适配评估 |

### 实现细节

- **Tokenizer**: 100M，对称编码解码 Transformer，14-stride patchification
- **Dynamics Model**: 250M，块因果 Transformer + Shortcut Flow Matching
- **优化器**: AdamW
- **上下文**: $T = 24$ 帧
- **Tokenizer 预训练**: 300k steps，14 GPU-days
- **Dynamics 预训练**: 180k steps，24 GPU-days
- **总计**: 58 GPU-days（8× NVIDIA H100）
- **好奇心采集**: 展开步长 $H = 32$，每 $K = 16$ 步重新规划，用 $u_r^{\text{norm}}$ 评分选轨迹

---

## 批判性思考

### 优点

1. **问题重新定框**: 将幻觉从"架构缺陷"重新界定为"数据覆盖问题"，提供了可操作的解决路径
2. **零标注检测**: 三个运行时信号无需任何人工标注或仿真器访问，部署成本低
3. **数据效率**: 仅 50 条轨迹即可适配未见环境，对实际机器人部署极具实用价值

### 局限性

1. **仅限视觉连续控制**: 当前 benchmark 未覆盖语言指令驱动的机器人操作、导航等任务类型
2. **Tokenizer 泛化瓶颈**: 即使微调后，未见 PSNR 仍低于已见 1.62 dB，跨分布泛化仍是挑战
3. **好奇心信号局限**: $u_r^{\text{norm}}$ 在需要精确目标导向覆盖时不如人类游玩（0.325 vs 0.362），反映了"广覆盖"与"任务相关覆盖"的根本张力

### 潜在改进方向

1. 将检测信号与语言条件结合，支持更通用的 VLA 幻觉检测
2. 探索主动学习框架，在目标任务分布已知时定向填补覆盖缺口
3. 扩展到真实机器人硬件上的在线自适应

### 可复现性评估

- [x] 代码开源（https://github.com/nicklashansen/mmbench2）
- [x] 预训练模型（https://huggingface.co/nicklashansen/mmbench2-models）
- [x] 训练细节完整（论文附录 F）
- [x] 数据集可获取（https://huggingface.co/datasets/nicklashansen/mmbench2）

---

## 关联笔记

### 基于

- [[Dreamer 4]]: 本文的基础架构，两阶段 WM pipeline（Tokenizer + Dynamics）来自 Dreamer 4
- [[Shortcut Flow Matching]]: 动力学模型使用的生成目标
- [[TD-MPC2]]: 同一作者（Nicklas Hansen）的先前工作，本文在 MPC 评估中也使用了该框架

### 对比

- [[Cosmos3]]: 以视频生成质量为目标的通用世界模型；本文专注连续控制的幻觉诊断
- [[WorldEval]]: 现有世界模型评估 benchmark；MMBench2 补充了动作标注和在线仿真器

### 方法相关

- [[世界模型]]: 核心研究对象
- [[Flow Matching]]: 动力学模型的去噪目标基础
- [[Shortcut Flow Matching]]: 具体使用的单步捷径流匹配变体
- [[MPC]]: 下游评估方式（模型预测控制）
- [[Model-Based RL]]: 本文幻觉问题影响的主要应用场景
- [[LPIPS]]: Tokenizer 训练损失之一
- [[CLIP]]: 语言任务指令的嵌入方式

### 硬件/数据相关

- [[MMBench2 数据集]]: 本文贡献的核心 benchmark（210 任务，427h）

---

## 速查卡片

> [!summary] Hallucination in World Models is Predictable and Preventable
> - **核心**: 世界模型幻觉 = 数据覆盖缺口，可预测可预防
> - **方法**: 三类幻觉表征 + 三个无监督运行时检测信号（ρ≈0.80）+ 覆盖感知训练/好奇心采集
> - **结果**: 50 条轨迹适配未见任务达 oracle 的 ~90%；覆盖感知微调展开 PSNR +0.88 dB
> - **代码**: https://github.com/nicklashansen/mmbench2

---

*笔记创建时间: 2026-06-26*
