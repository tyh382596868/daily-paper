---
title: "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels"
method_name: "LeWM"
aliases: [LeWorldModel, LeWM]
authors: [Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, Randall Balestriero]
year: 2026
venue: arXiv
tags: [world-model, jepa, self-supervised-learning, latent-planning, anti-collapse, model-predictive-control]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2603.19312v2
created: 2026-05-11
---

# 论文笔记：LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Mila & Université de Montréal、New York University、Samsung SAIL、Brown University |
| 日期 | March 2026 |
| 项目主页 | https://le-wm.github.io |
| 对比基线 | [[PLDM]]、[[DINO-WM]]、[[DreamerV3]]、[[TD-MPC]] |
| 链接 | [arXiv](https://arxiv.org/abs/2603.19312) / [Code](https://github.com/lucas-maes/le-wm) |

---

## 一句话总结

> LeWM 用「预测损失 + SIGReg 高斯正则」两项目标，第一次让基于像素的端到端 [[JEPA]] 世界模型稳定训练，规划速度比 foundation model 方案快 48×。

---

## 核心贡献

1. **首个端到端稳定的像素级 JEPA**：无需 stop-gradient、[[EMA]]、预训练编码器或辅助监督即可避免[[表征坍塌]]，在原始像素上联合训练 encoder + predictor。
2. **SIGReg 反坍塌正则项**：基于 [[Cramér-Wold 定理]] 与 [[Epps-Pulley 检验]]，通过把 latent 投影到随机方向并强制其分布趋近高斯，提供了**有理论保障**的反坍塌机制。
3. **超紧凑、超高效**：仅 ~15M 参数、单 GPU 数小时训练，**latent 规划速度比 [[DINO-WM]] 快 48×**，在 2D/3D 控制任务上与 foundation-model-based 世界模型保持竞争力。
4. **可解释的潜空间**：[[线性探测]]验证 latent 编码了智能体位置、物块位置、角度等物理量；[[违反预期]]（VoE）评估表明对物理违反（瞬移）显著敏感（p<0.01），对纯视觉扰动（颜色变化）则不敏感。

---

## 问题背景

### 要解决的问题

如何从**原始像素**端到端学习一个可用于规划的[[世界模型]]，同时避免 JEPA 类方法常见的[[表征坍塌]]问题？目标设定是：完全离线、无奖励、无任务标签、无特权状态访问、无预训练 encoder。

### 现有方法的局限

- **生成式世界模型**（IRIS、DIAMOND、[[DreamerV3]]）在像素空间预测，需要奖励信号或特权状态。
- **JEPA 自监督变体**（[[I-JEPA]]、[[V-JEPA]]）依赖 stop-gradient + EMA 这种缺乏理论保障的启发式手段防坍塌。
- **基于预训练 backbone**（[[DINO-WM]]、OSVI-WM）冻结 [[DINOv2]] encoder 来回避坍塌，但放弃了端到端学习，且 token 数高（>200×）导致规划昂贵。
- **唯一已存在的端到端方案 [[PLDM]]** 使用 [[VICReg]]，含 7 项损失、6+ 超参，训练曲线非单调、常常炸掉，缺乏理论保障。

### 本文的动机

如果有一种**单一、原理性强**的反坍塌正则可以替代 EMA / stop-gradient / 多项 VICReg 损失，那 JEPA 训练流程会大幅简化。SIGReg 正是这样的设计：要求 latent 分布趋近各向同性高斯。由 [[Cramér-Wold 定理]]，多元分布等于高斯当且仅当其所有一维投影都为高斯，因此把 latent 投到随机方向上再做 1D 正态性检验即可。该机制天然防止任何"集中到子空间"的坍塌模式。

---

## 方法详解

### 模型架构

LeWM 采用 **encoder + 自回归 predictor + SIGReg 正则** 的极简 [[JEPA]] 架构：

- **输入**：观测序列 $\mathbf{o}_{1:T}$（原始 RGB 像素）+ 动作序列 $\mathbf{a}_{1:T}$
- **Encoder**: [[ViT-Tiny]]（patch=14、12 层、3 头、hidden=192，~5M 参数），取 [CLS] token 经 1 层 [[BatchNorm]] MLP 投影
- **Predictor**: 6 层 16 头的 [[Transformer]]（dropout=0.1，~10M 参数），动作通过 [[AdaLN]] 注入（AdaLN 参数零初始化），含 N 帧历史与时间因果掩码，做自回归预测，后接与 encoder 同结构的 projector
- **训练目标**: 仅两项 —— 预测损失 $\mathcal{L}_{\text{pred}}$ + 正则项 $\text{SIGReg}$
- **关键差异**: **无** stop-gradient、**无** EMA、**无** 预训练权重，全部端到端联合优化
- **总参数**: ~15M（encoder 5M + predictor 10M）

### 核心模块

#### 模块 1：Encoder（[[ViT-Tiny]] + 投影）

**设计动机**：用极轻量的 backbone 把每帧压缩到一个紧凑 latent token，使后续规划只在 1 个 token 上展开（对比 DINO-WM 每帧 ~200 tokens），从根本上提速 [[模型预测控制|MPC]]。

**具体实现**：
- 输入帧 $\mathbf{o}_t$ 切成 14×14 patches，过 12 层 ViT
- 取 [CLS] token，经过含 [[BatchNorm]] 的 1 层 MLP 得到 $\mathbf{z}_t \in \mathbb{R}^d$
- 与 predictor 共享 projector 结构

#### 模块 2：Predictor（动作条件自回归 [[Transformer]]）

**设计动机**：在 latent 空间建模 $p(\mathbf{z}_{t+1} \mid \mathbf{z}_{1:t}, \mathbf{a}_t)$，并通过 [[AdaLN]] 让动作以乘性方式调制每层归一化参数，零初始化保证训练初期保留 encoder 几何结构。

**具体实现**：
- 输入：N 帧历史 latent + 当前动作
- 6 层 [[Transformer]]（含 dropout 0.1）+ 时间因果掩码 → 自回归预测 $\hat{\mathbf{z}}_{t+1}$
- 输出再过 projector，与 encoder 输出处于同一空间

#### 模块 3：SIGReg —— 高斯潜空间反坍塌正则

**设计动机**：用 [[Cramér-Wold 定理]] 把"多元高斯"问题降维为"任意一维投影都高斯"问题，再用 [[Epps-Pulley 检验]] 这个解析、可微的正态性检验作为 loss 项。这样：

- 不需要 stop-gradient / EMA（因为正则本身就阻止坍塌）
- 不需要预训练 backbone
- 只引入 1 个可调超参 $\lambda$

**具体实现**：见下方公式 2、3。每步从单位球面 $S^{d-1}$ 采样 $M=1024$ 个随机方向，把 latent 投影后逐一做 Epps–Pulley 统计量并平均。

---

## 关键公式

### 公式 1：[[JEPA|预测损失]]（Next-Embedding Prediction Loss）

$$
\mathcal{L}_{\text{pred}} \triangleq \|\hat{\mathbf{z}}_{t+1} - \mathbf{z}_{t+1}\|^2_2,
\quad
\hat{\mathbf{z}}_{t+1} = \text{pred}_\phi(\mathbf{z}_t, \mathbf{a}_t)
$$

**含义**：要求预测器在 latent 空间一步预测下一帧的 encoder 嵌入；这是 JEPA 的核心目标。

**符号说明**：
- $\mathbf{z}_t = \text{enc}_\theta(\mathbf{o}_t)$：当前帧 latent
- $\hat{\mathbf{z}}_{t+1}$：predictor 给出的预测
- $\mathbf{z}_{t+1}$：encoder 实际给出的下一帧 latent（**注意没有 stop-gradient**，与 [[I-JEPA]] / [[V-JEPA]] 不同）

### 公式 2：[[SIGReg]] 正则项

$$
\text{SIGReg}(\mathbf{Z}) \triangleq \frac{1}{M}\sum_{m=1}^{M} T(\mathbf{h}^{(m)}),
\quad
\mathbf{h}^{(m)} \triangleq \mathbf{Z}\mathbf{u}^{(m)},
\quad
\mathbf{u}^{(m)} \in S^{d-1}
$$

**含义**：把 batch latent $\mathbf{Z}$ 沿 $M$ 个随机单位方向投影成 1D 序列 $\mathbf{h}^{(m)}$，对每个 1D 分布跑正态性检验 $T(\cdot)$，求平均得到正则值。由 [[Cramér-Wold 定理]]：当且仅当所有 1D 投影都是高斯时，多元分布是高斯。

**符号说明**：
- $\mathbf{Z} \in \mathbb{R}^{N \times B \times d}$：history 长 $N$、batch $B$、维度 $d$ 的 latent
- $M = 1024$：随机投影数（消融显示对 $M$ 不敏感）
- $\mathbf{u}^{(m)}$：从单位球面均匀采样的方向

### 公式 3：[[Epps-Pulley 检验]] 统计量

$$
T^{(m)} = \int_{-\infty}^{\infty} w(t)\, |\phi_N(t;\mathbf{h}^{(m)}) - \phi_0(t)|^2 \, dt,
\quad
\phi_N(t;\mathbf{h}) = \frac{1}{N}\sum_{n=1}^{N} e^{i t h_n}
$$

**含义**：基于经验[[特征函数]] $\phi_N$ 与标准正态特征函数 $\phi_0(t)=e^{-t^2/2}$ 的加权 $L^2$ 距离，是闭式可微的 1D 正态性检验。极限性质：$\text{SIGReg}(\mathbf{Z}) \to 0 \Leftrightarrow \mathbb{P}_\mathbf{Z} \to \mathcal{N}(0, \mathbf{I})$。

**符号说明**：
- $w(t)$：高斯权重函数，使积分有闭式
- $\phi_N(\cdot;\mathbf{h})$：基于 $N$ 个样本的经验特征函数
- $\phi_0$：标准正态分布的特征函数

### 公式 4：LeWM 总损失

$$
\mathcal{L}_{\text{LeWM}} \triangleq \mathcal{L}_{\text{pred}} + \lambda \cdot \text{SIGReg}(\mathbf{Z})
$$

**含义**：仅有的两项损失。$\lambda$ 是**唯一**实际起作用的超参（设为 0.1），相比 [[PLDM]] / [[VICReg]] 的 6 个权重大幅简化。

**符号说明**：
- $\mathcal{L}_{\text{pred}}$：见公式 1
- $\lambda = 0.1$：SIGReg 权重

### 公式 5：[[CEM|交叉熵方法]] 规划目标

$$
\mathbf{a}^*_{1:H} = \arg\min_{\mathbf{a}_{1:H}} \mathcal{C}(\hat{\mathbf{z}}_H),
\quad
\mathcal{C}(\hat{\mathbf{z}}_H) = \|\hat{\mathbf{z}}_H - \mathbf{z}_g\|^2_2
$$

**含义**：在 latent 空间做 [[模型预测控制|MPC]]，目标是经过 H 步自回归 rollout 后让最后一帧 latent 接近目标 latent $\mathbf{z}_g = \text{enc}_\theta(\mathbf{o}_g)$。每次只执行前 $K$ 步动作再重规划，缓解自回归误差累积。

**符号说明**：
- $H$：规划地平线
- $\mathcal{C}$：终端目标匹配代价
- $\mathbf{z}_g$：目标观测的 encoder 嵌入

### 训练算法（伪代码）

```python
def LeWorldModel(obs, actions, lambd=0.1):
    """
    obs: (B, T, C, H, W)  原始像素序列
    actions: (B, T, A)    动作序列
    """
    emb = encoder(obs)                      # (B, T, D)
    next_emb = predictor(emb, actions)      # (B, T, D)

    # 公式 1：next-embedding 预测损失
    pred_loss = F.mse_loss(emb[:, 1:], next_emb[:, :-1])

    # 公式 2：逐步 SIGReg（反坍塌）
    sigreg_loss = mean(SIGReg(emb.transpose(0, 1)))

    # 公式 4：总损失
    return pred_loss + lambd * sigreg_loss
```

---

## 关键图表

### Figure 1: Training Pipeline / 训练流水线

![Figure 1](https://arxiv.org/html/2603.19312v2/x1.png)

**说明**：LeWM 训练总览。Encoder 把帧序列 $\mathbf{o}_{1:t}$ 编码成低维 latent $\mathbf{z}_{1:t}$，predictor 自回归预测 $\hat{\mathbf{z}}_{t+1}$，与真实 $\mathbf{z}_{t+1}$ 做 MSE 损失（公式 1）。同时 latent 沿 $M$ 个随机方向投影后做正态性检验，得到 SIGReg 正则（公式 2、3）。**关键卖点**：无 stop-gradient、无 EMA、无预训练。

### Figure 2: 不同 Latent 世界模型的特征对比

![Figure 2](https://arxiv.org/html/2603.19312v2/x2.png)

**说明**：把 latent 世界模型按训练范式分组对比。**End-to-end 类**（[[PLDM]]）联合训 encoder + predictor，但需要大量超参且无形式化坍塌保证；**Foundation-based 类**（[[DINO-WM]]）冻结预训练 encoder 来回避坍塌但放弃了端到端；**Task-specific 类**（[[DreamerV3]]、[[TD-MPC]]）需要奖励或特权状态。LeWM 同时满足：端到端、任务无关、像素输入、无需重建/奖励、单超参、有反坍塌理论保证。

### Figure 3: Planning Time and Performance Under Fixed Compute

![Figure 3a (Planning Time)](https://arxiv.org/html/2603.19312v2/x3.png)

![Figure 3b (Push-T)](https://arxiv.org/html/2603.19312v2/x4.png)

![Figure 3c (OGBench-Cube)](https://arxiv.org/html/2603.19312v2/x5.png)

**说明**：左图为规划时间（50 次平均），LeWM 因为每帧只需 ~1 个 token（DINO-WM 需 ~200 个），与 PLDM 持平、比 DINO-WM 快 ~50×；中右图在固定 FLOPs 预算下，[[Push-T]] 与 [[OGBench]]-Cube 上 LeWM 显著优于 DINO-WM。

### Figure 4: LeWM Latent Planning / 潜空间规划

![Figure 4](https://arxiv.org/html/2603.19312v2/x6.png)

**说明**：给定起始观测 $\mathbf{o}_1$ 和目标 $\mathbf{o}_g$，分别 encode 得到 $\mathbf{z}_1$ 和 $\mathbf{z}_g$。Predictor 自回归 rollout 至 horizon $H$，比较终端 latent 与 $\mathbf{z}_g$ 的代价，由 [[CEM]] 等求解器迭代优化动作序列 $\mathbf{a}_{1:H}$（公式 5）。

### Figure 5: 评估环境

![Figure 5](https://arxiv.org/html/2603.19312v2/x7.png)

**说明**：四个连续控制环境。**[[Push-T]]**：2D 推方块到目标姿态（机器人 benchmark 经典任务）；**[[OGBench]]-Cube**：3D 视觉更复杂的机械臂操作；**Two-Room**：简单 2D 导航；**Reacher**：2 关节臂在平面中触达目标。

### Figure 6: 跨环境规划性能

![Figure 6a (Two-Room)](https://arxiv.org/html/2603.19312v2/x8.png)

![Figure 6b (Reacher)](https://arxiv.org/html/2603.19312v2/x9.png)

![Figure 6c (PushT)](https://arxiv.org/html/2603.19312v2/x10.png)

![Figure 6d (OGBench-Cube)](https://arxiv.org/html/2603.19312v2/x11.png)

**说明**：在 Push-T 与 Reacher 上 LeWM 一致超越 [[PLDM]] 与 [[DINO-WM]]；在 3D 复杂场景 OGBench-Cube 上略输 DINO-WM（作者归因于 3D 视觉复杂度让端到端 encoder 更难学）；在最简单的 Two-Room 上反而被两者超越，原因可能是高维 latent 空间的高斯先验与该环境很低的本征维度不匹配。

### Figure 7: Predictor Rollouts on PushT and OGBench-Cube

![Figure 7a (PushT)](https://arxiv.org/html/2603.19312v2/x12.png)

![Figure 7b (OGBench-Cube)](https://arxiv.org/html/2603.19312v2/x13.png)

**说明**：用 3 帧上下文 + 给定动作序列让 predictor 在 latent 中开环 rollout，再用**训练时未参与**的 decoder 把 latent 解回图像。生成的"想象帧"与真实帧整体结构一致，证明 latent 捕获了主要场景动力学。**局限**：对末端执行器姿态等细节捕捉不全。

### Figure 8: Decoder Visualization During Training

![Figure 8](https://arxiv.org/html/2603.19312v2/x14.png)

**说明**：尽管训练时**不使用任何重建损失**，随训练推进 decoder 能解码出越来越接近真实场景的图像，说明 latent 越来越完整地保留了视觉信息。早期 decode 出的是"慢特征"（[[Slow Features]]），与文献观察一致。

### Figure 9: Latent Space Visualization on Push-T

![Figure 9](https://arxiv.org/html/2603.19312v2/x15.png)

**说明**：在 Push-T 中按规则网格采样状态（左：在 x-y 平面移动 agent 与 block），将每个状态对应的 latent 用 [[t-SNE]] 投到 2D（右）。可见拓扑结构与状态空间一致，证明 latent 编码了几何信息。

### Figure 10: Violation-of-Expectation Evaluation

![Figure 10a (TwoRoom)](https://arxiv.org/html/2603.19312v2/x16.png)

![Figure 10b (PushT)](https://arxiv.org/html/2603.19312v2/x17.png)

![Figure 10c (OGBench-Cube)](https://arxiv.org/html/2603.19312v2/x18.png)

**说明**：在三个环境中沿三种轨迹画"惊讶度"（surprise，即 predictor 预测残差）：未扰动参考、视觉扰动（物体颜色突变）、物理扰动（物体瞬移）。**结果**：物理瞬移在所有环境中都引起显著惊讶峰值（paired t-test，p<0.01），而颜色变化基本无影响——说明 LeWM latent 更关注**物理一致性**而非视觉外观。这是 [[违反预期|VoE]] 范式在世界模型上的典型展示。

### Table 1: Push-T 上的物理量[[线性探测]]

| Property | Model | Linear MSE ↓ | Linear r ↑ | MLP MSE ↓ | MLP r ↑ |
|----------|-------|--------------|------------|------------|----------|
| Agent Location | DINO-WM | 1.888±0.500 | **0.977** | **0.003±0.022** | **0.999** |
| Agent Location | PLDM | 0.090±0.311 | 0.955 | 0.014±0.119 | 0.993 |
| Agent Location | LeWM | **0.052±0.149** | 0.974 | 0.004±0.056 | **0.998** |
| Block Location | DINO-WM | **0.006±0.007** | **0.997** | 0.002±0.003 | **0.999** |
| Block Location | PLDM | 0.122±0.341 | 0.938 | 0.011±0.066 | 0.994 |
| Block Location | LeWM | 0.029±0.073 | 0.986 | **0.001±0.006** | **0.999** |
| Block Angle | DINO-WM | **0.050±0.101** | **0.979** | **0.009±0.052** | **0.995** |
| Block Angle | PLDM | 0.446±0.625 | 0.745 | 0.056±0.184 | 0.972 |
| Block Angle | LeWM | 0.187±0.359 | 0.902 | 0.021±0.139 | 0.990 |

**说明**：在 frozen latent 上训练线性 / MLP 回归头预测物理量。LeWM 一致超过 PLDM，与 DINO-WM 互有胜负——**注意公平性**：DINOv2 在 ~124M 张图上预训练，比 LeWM 看到的数据多 2 个数量级，仍能被一个端到端 15M 模型追平/超越，说明 SIGReg 学到的 latent 几何质量很高。

### Table 2: PLDM 最优超参（网格搜索结果）

| Loss Coefficient | Initial Value |
|------------------|---------------|
| α | 18.0 |
| β | 12 |
| γ | 0.2 |
| ζ | 0.7 |
| ν | 0.0 |
| μ | 0.0 |

**说明**：作者用网格搜索复现了 PLDM 的最佳配置——**6 个**需要调的损失系数。对比 LeWM 只调一个 $\lambda$，凸显 SIGReg 的工程优势。

---

## 实验结果

### 数据集 / 环境

| 环境 | 类型 | 特点 | 用途 |
|------|------|------|------|
| Two-Room | 2D 导航 | 内在维度低 | 简单 sanity check |
| Reacher | 2D 关节控制 | 2-joint 触达 | 中等连续控制 |
| [[Push-T]] | 2D 操作 | 推 T 形物体到目标姿态 | 主流 benchmark |
| [[OGBench]]-Cube | 3D 操作 | 视觉复杂、3D 几何 | 高难度评估 |

### 实现细节

- **Encoder**: [[ViT-Tiny]]，patch=14，12 层，3 头，hidden=192（~5M）
- **Predictor**: 6 层 16 头 [[Transformer]]，dropout 0.1（~10M），动作通过 [[AdaLN]]（零初始化）注入
- **总参数**: ~15M
- **优化器**: 默认 AdamW（论文未给出全部细节，详见附录）
- **训练硬件**: 单 GPU，几小时完成
- **SIGReg 配置**: $M=1024$ 随机方向，$\lambda=0.1$
- **规划**: [[CEM]] + 短 horizon [[模型预测控制|MPC]]，每次执行前 $K$ 步后重规划

### 消融

- **投影数 $M$**：从小到大几乎不影响性能 → 实践中 $M=1024$ 是安全默认
- **Embedding 维度**：超过某阈值后性能饱和
- **Encoder 选择**：[[ViT-Tiny]] 与 ResNet-18 表现相当
- **$\lambda$**：唯一明显有效的超参

### 训练稳定性

- LeWM loss 曲线**单调平滑**收敛；预测损失稳定下降，SIGReg 早期快速降至平台
- PLDM 在同样 7 项损失下曲线噪声大、非单调，且常常需要早停或重启

---

## 批判性思考

### 优点

1. **理论完备**：基于 Cramér–Wold + Epps–Pulley 给出的反坍塌保证比 EMA / stop-gradient 这种工程 trick 强得多。
2. **极简超参**：从 6+ 缩到 1，工程上对小团队特别友好。
3. **极快规划**：每帧 1 个 token 的设计直接换来 48× 速度优势，对真实机器人 MPC 关键。
4. **物理可解释**：[[违反预期|VoE]] + 物理量探测两条互补证据线表明 latent 不是表面的视觉特征，而是带物理结构的状态。

### 局限性

1. **Two-Room 翻车暴露 prior 不匹配问题**：高维高斯先验与低本征维度环境冲突时性能下降，未给出自适应方案。
2. **3D 视觉复杂度仍有差距**：在 OGBench-Cube 上略输 DINO-WM，端到端学 3D encoder 比直接用 [[DINOv2]] 难。
3. **短 horizon 限制**：自回归误差累积让长 horizon 规划不可行（这是所有 latent WM 的通病）。
4. **依赖动作标注**：必须有 $(\mathbf{o}, \mathbf{a})$ 配对，无法直接吃 web 视频；可以引入逆动力学缓解但本文没做。
5. **Decoder 仅作可视化**：训练时 decoder 不参与，但若希望部署在需要可解释中间结果的场景，仍需额外训练 decoder。

### 潜在改进方向

1. **层级化 LeWM**：在低层 LeWM 上叠加更慢时间尺度的 LeWM，做长 horizon 规划
2. **从大规模自然视频预训练**：替代仅靠任务数据，对小数据/低多样性环境的高斯先验问题应有帮助
3. **逆动力学联合训练**：减少对动作标注的依赖
4. **自适应 latent 维度 / 自适应 prior**：根据环境本征维度动态调整 SIGReg 的目标分布
5. **结合 [[TD-MPC]] 的价值预测**：补足 LeWM 仅做终端代价的不足，让 horizon 内的中间状态也能引导规划

### 可复现性评估

- [x] 代码开源（https://github.com/lucas-maes/le-wm）
- [x] 项目主页（https://le-wm.github.io）
- [x] 训练细节给出（架构、参数量、超参、伪代码）
- [x] 数据集可获取（Push-T、OGBench、Reacher、Two-Room 均开源 benchmark）

---

## 关联笔记

### 基于

- [[JEPA]]: 直接继承的整体架构思想（latent 空间预测）
- [[I-JEPA]] / [[V-JEPA]]: JEPA 在图像/视频上的代表实现，LeWM 摆脱了它们对 EMA + stop-gradient 的依赖
- [[VICReg]]: PLDM 用的方差-不变-协方差正则，LeWM 用 SIGReg 替代

### 对比

- [[PLDM]]: 唯一已存在的端到端像素 JEPA 基线，LeWM 在稳定性、超参数量、性能上全面胜出
- [[DINO-WM]]: 冻结 [[DINOv2]] encoder 的 foundation-based 方案，LeWM 比它快 48× 且大多数任务更强
- [[DreamerV3]]: 像素重建 + 价值学习的代表，需要奖励
- [[TD-MPC]]: latent + 价值学习，需要奖励/特权状态

### 方法相关

- [[SIGReg]]: 本文提出的反坍塌正则
- [[Cramér-Wold 定理]]: 多元高斯检验的理论基础
- [[Epps-Pulley 检验]]: 1D 正态性检验
- [[CEM]]: 用于 latent 规划的求解器
- [[模型预测控制|MPC]]: 短 horizon 重规划策略
- [[ViT-Tiny]]: encoder 架构
- [[AdaLN]]: 动作注入方式
- [[违反预期|VoE]]: 物理理解评估范式
- [[线性探测]]: latent 质量评估方式
- [[Slow Features]]: 解释训练早期 decoder 输出现象
- [[表征坍塌]]: 本文核心要解决的问题

### 数据/任务

- [[Push-T]] / [[OGBench]]: 主要评估 benchmark
- [[世界模型]] / [[World Model]]: 总体范式

---

## 速查卡片

> [!summary] LeWorldModel (LeWM)
> - **核心**：用 SIGReg（高斯潜空间正则）替代 EMA / stop-gradient，让端到端像素 JEPA 第一次稳定训练
> - **方法**：ViT-Tiny encoder + Transformer predictor + 两项损失（MSE 预测 + Epps–Pulley 正态性检验），15M 参数，单 GPU 数小时训完
> - **结果**：[[CEM]] latent 规划比 [[DINO-WM]] 快 48×，Push-T / Reacher 一致领先；VoE 实验证明 latent 编码物理结构
> - **代码**：https://github.com/lucas-maes/le-wm

---

*笔记创建时间: 2026-05-11*
