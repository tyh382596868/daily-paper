---
title: "Gamma-World: Generative Multi-Agent World Modeling Beyond Two Players"
method_name: "Gamma-World"
authors: [Fangfu Liu, Kai He, Tianchang Shen, Tianshi Cao, Sanja Fidler, Yueqi Duan, Jun Gao, Igor Gilitschenski, Zian Wang, Xuanchi Ren]
year: 2026
venue: arXiv
tags: [world-model, multi-agent, video-diffusion, interactive-simulation, rope, permutation-symmetry, distillation]
zotero_collection: 3-Robotics/1-VLX/世界模型
image_source: mixed
arxiv_html: https://arxiv.org/abs/2605.28816
created: 2026-05-28
---

# 论文笔记：Gamma-World: Generative Multi-Agent World Modeling Beyond Two Players

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | NVIDIA、Tsinghua University、University of Toronto、Vector Institute |
| 日期 | May 2026（arXiv:2605.28816v1） |
| 项目主页 | https://research.nvidia.com/labs/sil/projects/gamma-world |
| 代码库 | https://github.com/nv-tlabs/Gamma-World |
| 对比基线 | [[Solaris]]（多人 Minecraft 世界模型）、Multiverse（Frame Concat） |
| 链接 | [arXiv](https://arxiv.org/abs/2605.28816) / [Project Page](https://research.nvidia.com/labs/sil/projects/gamma-world) / [Code](https://github.com/nv-tlabs/Gamma-World) |

---

## 一句话总结

> 用单纯形 RoPE 编码代替学习的"槽位 ID"，配合 hub-mediated 稀疏跨智能体注意力，让视频[[World Model|世界模型]]从单 agent 扩展到 4 个 agent 而无需重训，并以 24 FPS 实时流式 rollout。

---

## 核心贡献

1. **Simplex Rotary Agent Encoding（单纯形旋转智能体编码）**: 一种 parameter-free 的 3D [[RoPE]] 扩展，把 agent 当作正单纯形顶点，从而所有 agent 两两等距 + 置换等价，无需学习的 per-slot ID 嵌入。
2. **Sparse Hub Attention（稀疏 Hub 注意力）**: 引入少量可学习 hub token 作为跨 agent 通信中转，把跨 agent 注意力代价从 $\mathcal{O}(P^2)$ 降到 $\mathcal{O}(P)$。
3. **Conditional Self-Forcing Distillation**: 双向多智能体教师 → 块因果 [[Diffusion Forcing|多步学生]] → 少步条件蒸馏（[[DMD]]），实现 24 FPS 实时流式 rollout，并在仅 2-agent 训练数据上零样本泛化到 4 个 agent。

---

## 问题背景

### 要解决的问题

把交互式视频[[World Model|世界模型]]从单智能体扩展到**多智能体共享世界**：多个玩家、机器人或具身 agent 在同一时空里同步行动，未来观测必须**跨时间 + 跨视角**双重一致。

### 现有方法的局限

- **大多数视频世界模型**（DiamondGen、Matrix-Game、WorldMem 等）只建模单 agent，单条 action 流，单视角 rollout。
- **[[Solaris]]**（同期工作）：把所有 agent token 拼起来做 dense joint attention + 学习的 **per-player ID embedding**。两个结构性缺陷：
    1. Dense joint attention 对 agent 数量 $P$ 的复杂度为 $\mathcal{O}(P^2)$，超过 2 玩家就不实时。
    2. 学到的 per-slot ID 破坏了"等能力 agent 应当可交换"的对称性，且玩家阵容固定，扩到更多人需重训。
- **Frame Concat baseline**（Multiverse 风格）：把多视角直接拼成单流，丢失了每个 agent 独立的视角对齐。

### 本文的动机

把"agent 身份"做成几何上**等距而互异**的旋转相位，配合**hub 中介**的稀疏通信，得到一个：(i) 可独立控制、(ii) 置换对称、(iii) 跨 agent 计算线性增长、(iv) 训练于少 agent 即可扩展到更多 agent 的视频世界模型。

---

## 方法详解

### 模型架构

设多智能体潜在视频为 $\mathbf{Z}_0 \in \mathbb{R}^{P \times T \times H \times W \times C_z}$，相对单智能体增加了**显式 agent 轴** $p \in \{1,\dots,P\}$。Gamma-World 基于 [[Cosmos-Predict2.5]]-2B 的 [[DiT]]（hidden $D=2048$、28 个 block、16 head、head dim 128）并做如下改动：

- **输入**: 每个 agent 的初始观测 $\{o_1^p\}_{p=1}^P$ + 每个 agent 的同步 action 序列 $\{a_{1:T}^p\}_{p=1}^P$。
- **Patch 化**: 共享视觉 tokenizer，每个 agent 流独立 token 化。
- **Action 编码**: 共享 action encoder $f_a$（两路 MLP + 4× 步长 1D 时序卷积 + 投影到 $D=2048$），每个 transformer block 把 action 特征加为 **per-block bias**（见公式 4-5）。
- **位置编码**: [[Simplex Rotary Agent Encoding|4D RoPE]] —— 把头维划分为 $(d_t, d_p, d_h, d_w) = (64, 32, 16, 16)$；agent 轴的旋转由**正单纯形顶点**给出（§3.2、公式 6-10）。
- **跨 agent 通信**: [[Sparse Hub Attention]] —— 每帧 $K=8$ 个可学习 hub token，agent 只与自己流 + hub 交互；agent 之间走 agent → hub → agent 两跳路径（公式 11-13）。
- **因果生成**: 学生模型用 [[Block-Causal Attention|块因果注意力]] + 24 帧滑动窗口 + KV cache；teacher 用 dense bidirectional attention。
- **输出**: 每个 agent 下一帧 latent $\{o_{t+1}^p\}_{p=1}^P$，共享同一世界状态的不同视角。

### 核心模块

#### 模块 1: [[Simplex Rotary Agent Encoding]]

**设计动机**: 用 1D 标量 $\theta_p = p\omega$ 编码 agent 会让不同 agent 对之间的旋转距离不同（取决于 $|p-q|$），从而结构性地"特殊化"某些槽位；学习的 per-slot ID 又会绑定固定阵容。需要一个**等距 + 等价**的几何结构。

**具体实现**:
- 取定一个**单纯形池大小** $V \le d_p/2 + 1$（$d_p$ 是 agent band 维度），实现里 $V=4$。
- 在零均值子空间里构造 $V$ 个单位长度顶点（公式 7），让所有顶点两两等距（公式 8）。
- 训练时对 active agent 随机采样**单射** $\pi: \{1,\dots,P\} \to \{1,\dots,V\}$，agent $p$ 的 agent-band 角为 $\theta_p = \alpha s_{\pi(p)}$（公式 9）。
- 这等价于把 3D RoPE 扩展为 4D RoPE：$R_{\text{simp-4D}}(t, p, h, w) = \text{diag}(R_t(t), R_{\text{simp}}(\pi(p)), R_h(h), R_w(w))$（公式 10）。
- 推理时新增 agent 只需启用未用的顶点，无需新参数。
- 对于已有 video DiT 没有 agent band 的情况，按 [[ReRoPE]] 思路从**时间 band 低频端**借 $d_p$ 维。

#### 模块 2: [[Sparse Hub Attention]]

**设计动机**: Dense 跨 agent 注意力代价 $\mathcal{O}(P^2 n^2 L^2)$（$L=HW$ 每帧 token，$n$ block 帧数）随 agent 数二次增长。共享世界中 agent 之间的耦合往往可以**通过紧致的环境状态**间接表达。

**具体实现**:
- 加入 $K$ 个可学习 **hub token**（$\mathbf{H} \in \mathbb{R}^{K \times D}$），跨帧广播。
- Attention mask: agent token 只 attend 自己流 + hub；hub 同时 attend 所有 agent + 其它 hub（公式 11）。
- 跨 agent 信息走 **agent → hub → agent** 两跳路径。
- 结合 block-causal 因果约束：$\mathcal{M}(i,j) = \mathbb{1}[b(j) \le b(i)] \cdot \mathcal{M}_{\text{hub}}(i,j)$（公式 12）。
- Hub token 复用所在帧的时间 RoPE 相位，agent / 空间 band 使用单位旋转，时间对齐但 agent / 空间中立。
- 复杂度从 $\mathcal{O}(P^2 n^2 L^2)$ 降到公式 13 的线性 $\mathcal{O}(P)$ 形式。

#### 模块 3: 三阶段训练（Bidirectional → Causal → Distilled）

**Stage 1 — Bidirectional Teacher**：dense bidirectional attention + 单一共享噪声等级，在 2-agent 数据上用 [[Flow Matching]] 训练。

**Stage 2 — Causal Student**：[[Block-Causal Attention]] + [[Sparse Hub Attention]] + [[Diffusion Forcing]]（每个时间 block 独立采样噪声等级），作为**完整的多步扩散模型**训练而非短暂 warm-up。

**Stage 3 — Conditional Self-Forcing Distillation**：把多步学生蒸馏为少步条件生成器，复用同一 conditioning $\mathcal{C}$（首帧 + per-agent action），用 [[DMD]] + autoregressive self-rollout，把生成的 block 写入 KV cache 作为后续历史，匹配推理时分布。每个 block 用 4 步降噪 $\{1000, 750, 500, 250\}$，flow shift 5.0。

---

## 关键公式

### 公式 1: [[Flow Matching|线性插值]]（前向加噪）

$$
\mathbf{z}_\sigma = (1 - \sigma)\, \mathbf{z}_0 + \sigma\, \epsilon
$$

**含义**: 在 clean latent $\mathbf{z}_0$ 和噪声 $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ 之间做线性插值，得到噪声级 $\sigma$ 下的样本。

**符号说明**:
- $\mathbf{z}_0 \in \mathbb{R}^{T \times H \times W \times C_z}$: 干净视频 latent
- $\epsilon \sim \mathcal{N}(0, \mathbf{I})$: 高斯噪声
- $\sigma \in [0, 1]$: 噪声等级

### 公式 2: [[Flow Matching]] 损失

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{\mathbf{z}_0, \epsilon, \sigma} \big[\, \| v_\theta(\mathbf{z}_\sigma, \sigma, \mathcal{C}) - (\epsilon - \mathbf{z}_0) \|_2^2 \,\big]
$$

**含义**: 学习速度场 $v_\theta$ 拟合时间导数 $\epsilon - \mathbf{z}_0$，即 flow matching 训练目标。

**符号说明**:
- $v_\theta$: 速度场预测网络（即扩散 transformer）
- $\mathcal{C}$: 条件信号（首帧观测、per-agent action）

### 公式 3: 3D [[RoPE]] 算子（preliminaries）

$$
\mathbf{R}_{\text{3D}}(t, h, w) = \text{diag}(\mathbf{R}_t(t), \mathbf{R}_h(h), \mathbf{R}_w(w))
$$

**含义**: 沿时间、高度、宽度三轴分别旋转 query/key 特征。

**符号说明**:
- $\mathbf{R}_x(x)$: 沿 $x$ 轴的 2D 块对角旋转，角度按标准 RoPE 频率
- 头维分配 $d_t + d_h + d_w = d_{\text{rope}}$

### 公式 4-5: Action Bias 注入

$$
\beta_{\ell, t}^p = g_\ell(\mathbf{u}_t^p) \in \mathbb{R}^D
$$

$$
\mathbf{x}_{\ell, p, t, h, w} \leftarrow \mathbf{x}_{\ell, p, t, h, w} + \beta_{\ell, t}^p
$$

**含义**: 第 $\ell$ 层把每个 agent 的 action 特征 $\mathbf{u}_t^p$ 投影成 layer-specific 偏置 $\beta_{\ell,t}^p$，并加到对应 agent + 帧的所有空间 token 上。

**符号说明**:
- $g_\ell$: 第 $\ell$ 层的投影网络
- $\mathbf{u}_t^p = f_a(a_t^p)$: 共享 action encoder 输出

### 公式 6: 4D RoPE 算子（加入 agent 轴）

$$
\mathbf{R}_{\text{4D}}(t, p, h, w) = \text{diag}(\mathbf{R}_t(t), \mathbf{R}_p(p), \mathbf{R}_h(h), \mathbf{R}_w(w))
$$

**含义**: 把 agent 轴加入 RoPE，头维分为 $d_t + d_p + d_h + d_w = d_{\text{rope}}$（实现里为 $(64, 32, 16, 16)$）。

### 公式 7: [[Simplex Rotary Agent Encoding|单纯形顶点]] 构造

$$
\mathbf{s}_v = \sqrt{\frac{V}{V-1}}\, \mathbf{Q} \left( \mathbf{e}_v - \frac{1}{V}\mathbf{1} \right) \in \mathbb{R}^{d_p / 2}, \quad v = 1, \dots, V
$$

**含义**: 把 $V$ 个 one-hot 向量减去均值（投到零均值子空间）后等距嵌入到 $d_p/2$ 维 agent-angle 空间。

**符号说明**:
- $\mathbf{e}_v \in \mathbb{R}^V$: 第 $v$ 个 one-hot
- $\mathbf{Q}$: 把 $V$ 维零均值子空间映到 $\mathbb{R}^{d_p/2}$ 的线性等距
- $V$: 单纯形池大小，限制为 $V \le d_p/2 + 1$

### 公式 8: 单位长度 + 等距性质

$$
\|\mathbf{s}_v\|_2 = 1, \qquad \|\mathbf{s}_v - \mathbf{s}_{v'}\|_2^2 = \frac{2V}{V-1}, \quad \forall v \ne v'
$$

**含义**: 所有 agent 顶点单位长度，且两两平方距离都等于 $\frac{2V}{V-1}$（附录 B 给出推导：$\bar{\mathbf{s}}_p^\top \bar{\mathbf{s}}_q = -1/V$，$\|\mathbf{s}_p\|_2^2 = 1$，从而 $\|\mathbf{s}_p - \mathbf{s}_q\|_2^2 = 2 + 2/(V-1)$）。这是**置换对称性**的几何根源。

### 公式 9: Agent-band 旋转角

$$
\theta_p = \alpha\, \mathbf{s}_{\pi(p)}
$$

**含义**: 实际旋转相位是简单形顶点乘以 agent 分离强度 $\alpha$，$\pi$ 是训练时随机采样的单射。

**符号说明**:
- $\alpha > 0$: 控制 agent 间分离强度
- $\pi: \{1, \dots, P\} \to \{1, \dots, V\}$: 随机单射，等价于随机选 $P$ 个顶点

### 公式 10: Simplex-4D RoPE 算子

$$
\mathbf{R}_{\text{simp-4D}}(t, p, h, w) = \text{diag}\bigl(\mathbf{R}_t(t), \mathbf{R}_{\text{simp}}(\pi(p)), \mathbf{R}_h(h), \mathbf{R}_w(w)\bigr)
$$

**含义**: 把公式 6 的标准 $\mathbf{R}_p$ 替换成 simplex 旋转 $\mathbf{R}_{\text{simp}}$，是论文的核心位置编码算子。

### 公式 11: [[Sparse Hub Attention|Hub-and-Spoke Mask]]

$$
\mathcal{M}_{\text{hub}}(i, j) = \mathbb{1}\bigl[\rho(i) = \rho(j) \;\vee\; \rho(i) = \text{hub} \;\vee\; \rho(j) = \text{hub}\bigr]
$$

**含义**: 同一 agent 流之间相互可见；任一端是 hub 也可见；不同 agent 流之间被屏蔽。

**符号说明**:
- $\rho(i) \in \{1, \dots, P, \text{hub}\}$: token $i$ 的身份（哪个 agent 或 hub）

### 公式 12: 块因果 + Hub 复合 mask

$$
\mathcal{M}(i, j) = \mathbb{1}[b(j) \le b(i)] \cdot \mathcal{M}_{\text{hub}}(i, j)
$$

**含义**: 第一项是 [[Block-Causal Attention|块因果]] 约束（只看同 block 或更早），第二项是 hub 拓扑。两者乘积同时保证因果性与 hub-mediated 通信。

**符号说明**:
- $b(i)$: token $i$ 的时间 block 索引

### 公式 13: 单 block 复杂度

$$
\mathcal{O}\bigl(P n L (n L + n K)\bigr) + \mathcal{O}\bigl(n K (P n L + n K)\bigr)
$$

**含义**: SHA 的单 block self-attention 代价，对 agent 数 $P$ **线性**（在 $n, L, K$ 固定时），对比 dense 的 $\mathcal{O}(P^2 n^2 L^2)$。

**符号说明**:
- $n$: block 内帧数
- $L = HW$: 每帧空间 token 数
- $K$: hub token 数（实现里 $K=8$）

### 公式 14-27（附录 B）: 单纯形等距性证明（节选）

$$
\bar{\mathbf{s}}_p^\top \bar{\mathbf{s}}_q = -\frac{1}{V}, \quad p \ne q
$$

$$
\mathbf{s}_p^\top \mathbf{s}_q = -\frac{1}{V-1}, \quad \|\mathbf{s}_p - \mathbf{s}_q\|_2^2 = \frac{2V}{V-1}
$$

**含义**: 在零均值子空间里两两内积仅取决于 $V$，归一化后给出固定负相关 $-1/(V-1)$，从而距离全相等。

### 公式 28-37（附录 B）: 复 RoPE 空间的等距推广

$$
\|\Phi_p - \Phi_q\|_2^2 = \sum_{r=1}^{d_p / 2} 2\bigl(1 - \cos(\theta_p^r - \theta_q^r)\bigr) \approx \|\theta_p - \theta_q\|_2^2 = \alpha^2 \frac{2V}{V-1}
$$

**含义**: 旋转后用复表示 $\Phi_p = \exp(i\theta_p)$，对小 $\alpha$ 用 $1-\cos x \approx x^2/2$ 近似，得到**复 RoPE 空间中所有 agent 对也近似等距**。

---

## 关键图表

### Figure 1: 系统 Teaser

![[Gamma-World_fig1.png|600]]

**说明**: 展示 $\gamma$-World 从 2 agent → 4 agent → 真实世界双臂机器人的多场景能力。每个分块都是一个 agent 的视角，跨视角共享同一个底层世界状态。来自虚拟游戏（Minecraft 类）到真实场景的统一多 agent 框架。

### Figure 2: 方法总览

![[Gamma-World_fig2.png|600]]

**说明**: 整个 pipeline。左：同步多 agent 输入（每个 agent 自己的初始观测 $o_1^p$ 与 action 序列 $a_{1:T}^p$）。中：Causal Multi-Agent DiT 学生，包含 [[Sparse Hub Attention]]（自身 + hub 可见，其他 agent 被 mask）、Action Bias 注入、4D Simplex RoPE 旋转 $R_{\text{simp-4D}}$、KV Cache。右下：Simplex RoPE 几何示意——线性 RoPE（baseline）把 agent 放在一维直线上不对称，单纯形 RoPE 让 4 个 agent 两两等距、置换等价。

### Figure 3: Sparse Hub Attention 效率对比

![[Gamma-World_fig3.png|600]]

**说明**: 在 2 / 4 / 8 agent 下对比 Dense Attention 与 Sparse Hub Attention 的 DiT 延迟、self-attention 延迟、self-attention FLOPs。8 agent 时 dense 需要 611 ms / 17.6 ms / 7.6 T FLOPs，SHA 仅 246 ms / 4.5 ms / 981 G FLOPs，**自注意力延迟降低约 4 倍，FLOPs 降低近 8 倍**。

### Figure 4: 双 agent 交互定性结果

![[Gamma-World_fig4.png|600]]

**说明**: 两个 agent 的同步 rollout。每行一个任务（移动、挖矿、战斗、建造）。模型在 agent 短暂离开对方视野后依然保持物体与 agent grounding，说明它学到了**共享 latent 世界状态**而非独立单 agent 视频。

### Figure 5: 4 agent 零样本泛化

![[Gamma-World_fig5.png|600]]

**说明**: 仅在 2-agent 数据上训练的模型，**零样本**生成 4-agent 同步 rollout。每行第一帧是某一个 agent 的初始状态，后续展示四个 agent 在同一世界里的同步 rollout。这种泛化由（1）Simplex RoPE 不绑定固定 slot、（2）SHA 共享通信路径 共同支撑。

### Figure 6: 真实双臂机器人

> 🖼️ **Figure 6** — 图片暂缺，arXiv 抓取失败（原图未能获取）

**说明**: 把左右机械臂视为两个交互 agent，在 RealOmin-Open 数据集上生成未来帧。同一多 agent 世界模型框架可以从 Minecraft 玩家迁移到双臂操作场景，保持双臂协调运动和空间布局。

### Table 1: 与多智能体 baseline 的全方位对比

| Method | Memory FVD ↓ | Memory FID ↓ | Grounding FVD ↓ | Grounding FID ↓ | Movement FVD ↓ | Movement FID ↓ | Building FVD ↓ | Building FID ↓ | Consistency FVD ↓ | Consistency FID ↓ |
|--------|------------:|------------:|---------------:|----------------:|---------------:|---------------:|---------------:|---------------:|------------------:|------------------:|
| Frame Concat [9] | 450.6 | 69.8 | 528.3 | 63.2 | 556.9 | 65.0 | 551.8 | 87.3 | 576.0 | 123.2 |
| [[Solaris]] [47] | 333.8 | 51.7 | 301.9 | 36.1 | 311.1 | 36.3 | 448.6 | 71.0 | 443.1 | 94.8 |
| **$\gamma$-World (Ours)** | **184.1** | **24.8** | **199.3** | **24.0** | **191.5** | **21.2** | **264.5** | **32.1** | **280.0** | **46.9** |

**说明**: 5 个评估维度（记忆、Grounding、移动、建造、一致性）× FVD/FID 共 10 个指标，Gamma-World **全部领先**。相对 Solaris：FVD 平均下降约 40%，FID 平均下降约 50%。

### Table 2: 架构消融

| Setting | Composition | Agent Encoding | Interaction | FVD ↓ | FID ↓ | LPIPS ↓ | PSNR ↑ | SSIM ↑ |
|---------|-------------|----------------|-------------|------:|------:|--------:|-------:|-------:|
| Spatial Concat | Spatial concat | None | Full | 312.4 | 38.7 | 0.326 | 24.8 | 0.782 |
| Sequence Concat | Sequence concat | None | Full | 285.6 | 35.2 | 0.298 | 25.6 | 0.798 |
| View Embedding | Sequence concat | View emb. | Full | 256.3 | 32.4 | 0.281 | 26.4 | 0.815 |
| Simplex Encoding | Sequence concat | Simplex | Full | 228.5 | 29.6 | 0.265 | 27.5 | 0.830 |
| **$\gamma$-World (Full)** | Sequence concat | Simplex | **Sparse Hub** | **223.4** | **30.2** | **0.269** | **27.7** | **0.836** |

**关键发现**:
- Sequence Concat 优于 Spatial Concat：保持每 agent 空间分辨率而非堆成大画布。
- Simplex Encoding 显著优于 View Embedding（FVD 256→228）：等距对称几何 > 学习的 slot 嵌入。
- 加上 [[Sparse Hub Attention]] 后 FVD 微降到 223.4（FID 略升 30.2），主要价值在效率，质量基本不损失。

### Table 3: 游戏 action 字段（25 维 = 23 离散 + 2 连续）

| Index | Field | 描述 |
|------:|-------|------|
| 0 | inventory | 打开物品栏 |
| 1 | ESC | 退出/取消菜单 |
| 2–10 | hotbar.1–hotbar.9 | 选 hotbar 槽 1–9 |
| 11–14 | forward, back, left, right | 行走 |
| 15–17 | jump, sneak, sprint | 跳/潜行/冲刺 |
| 18 | swapHands | 双手交换持物 |
| 19–22 | attack, use, pickItem, drop | 攻击/使用/拾取/丢弃 |
| 23–24 | cameraX, cameraY | 水平 yaw、垂直 pitch |

**说明**: Minecraft 风格控制，每帧每 agent 25 维。

### Table 4: 机器人 action 字段（10 维连续）

| Index | Field | 描述 |
|------:|-------|------|
| 0–2 | pos_x, pos_y, pos_z | 末端位置 |
| 3–8 | rot_6d_0 – rot_6d_5 | 末端 6D 旋转表征 |
| 9 | gripper | 夹爪开度 |

**说明**: 左右机械臂用同一 10 维格式，分别一条时间对齐的 action 序列。

### Table 5: 三阶段训练变体对比

| Variant | FVD ↓ | FID ↓ | LPIPS ↓ | PSNR ↑ | SSIM ↑ |
|---------|------:|------:|--------:|-------:|-------:|
| Bidirectional (Teacher) | **227.3** | **31.0** | **0.272** | **27.7** | **0.828** |
| Causal (Student) | 266.4 | 34.4 | 0.277 | 26.2 | 0.805 |
| Distilled | 239.7 | 30.9 | 0.273 | 26.8 | 0.811 |

**说明**: Teacher 因可见全局帧表现最好；Causal 由于只看历史掉了 39 FVD；Distilled 通过蒸馏把 teacher 大部分质量"挤"回因果学生，FVD 回到 239.7，且支持流式推理。

### Table 6: Hub Token 数 $K$ 消融

| $K$ | FVD ↓ | FID ↓ | LPIPS ↓ | PSNR ↑ | SSIM ↑ |
|----:|------:|------:|--------:|-------:|-------:|
| 1 | 250.9 | 31.5 | 0.271 | 27.3 | 0.825 |
| 8 | 223.4 | 30.2 | 0.269 | 27.7 | 0.836 |
| 32 | 221.8 | 29.8 | 0.267 | 27.9 | 0.838 |
| 128 | **220.5** | **29.5** | **0.266** | **28.0** | **0.839** |

**说明**: hub token 是跨 agent 通信瓶颈，$K=1$ 容量不足质量显著降；$K=8$ 起接近饱和（FVD 收益边际），$K=128$ 也仅多提 3 个 FVD，因此 $K=8$ 是质量/成本 sweet spot。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 自建 Minecraft 2-agent | 大规模同步轨迹 | 受 [[SolarisEngine]] 启发的可控 episode 脚本 + 协调 bot + 对齐视觉-动作记录 | 训练 + 评估 |
| 自建 Minecraft 4-agent | 同 pipeline 扩展 | 用于评估 zero-shot 多 agent 泛化 | 评估 |
| RealOmin-Open Dataset [16] | 真实双臂机器人 | 左右臂视为两个 agent | 真实场景评估 |

### 评估指标

- **生成质量**: FVD ↓、FID ↓
- **感知/像素质量**: LPIPS ↓、PSNR ↑、SSIM ↑
- **多维场景**: Memory、Grounding、Movement、Building、Consistency（5 个）
- **效率**: DiT 延迟、self-attention 延迟、self-attention FLOPs（agent 数 2 / 4 / 8）

### 实现细节

- **Backbone**: [[Cosmos-Predict2.5]]-2B TI2V checkpoint（hidden $D=2048$、28 个 transformer block、16 head × dim 128、MLP ratio 4、[[AdaLN]]-LoRA rank 256）
- **4D RoPE 头维划分**: $(d_t, d_p, d_h, d_w) = (64, 32, 16, 16)$
- **Simplex 池**: $V = 4$，每步采样 2 顶点 + 随机置换槽位
- **Action encoder**: 两路 MLP（各 128 维）+ 4× stride 1D 时序卷积 + 投影到 2048 维
- **Hub tokens**: $K = 8$
- **Local window**: 24 latent frames per view
- **Resolution**: 每视角 $320 \times 480$
- **训练**:
    - Stage 1 teacher: 93-frame 段（24 latent frame）训练 10k 步 → 189-frame 段（48 latent frame）微调 6k 步
    - Stage 2 student: 93-frame 段训练 15k 步
    - 优化器: AdamW, lr $3 \times 10^{-5}$, weight decay $10^{-3}$, $(\beta_1, \beta_2) = (0.0, 0.999)$, 100-step warmup, grad clip 0.1
    - Stage 3 蒸馏: 189-frame 段, 400 iter, lr student $2 \times 10^{-6}$ / critic $4 \times 10^{-7}$, generator:critic = 1:4
- **硬件**: 32 张 NVIDIA GB200
- **推理**: 4 步降噪 $\{1000, 750, 500, 250\}$, flow shift 5.0, 24 FPS 流式 rollout，KV cache rolling 24 帧

### 可视化结果

- **2 agent**：双流同步、即使其中一个 agent 暂离对方视野，物体和 agent 位姿仍然 grounding 一致。
- **4 agent**：纯零样本（训练只有 2 agent），生成 4 视角同步 rollout 无需任何架构改动。
- **真实机器人**：双臂协调运动 + 空间布局 保持稳定。

---

## 批判性思考

### 优点

1. **几何解析的对称性**：Simplex RoPE 用单纯形顶点把"置换对称"做成了**封闭形式的几何性质**（附录 B 给出闭式等距证明），相比学习的 ID 嵌入更具可解释性和扩展性。
2. **agent-count 泛化**：训练时只见 2 agent，推理时直接换更多顶点就能跑 4 agent，是个非常实用的能力（Solaris 必须固定 roster）。
3. **复杂度从 $\mathcal{O}(P^2)$ 降到 $\mathcal{O}(P)$**：Hub 中介是简单但有效的稀疏化，且与块因果 / KV cache 自然兼容（hub 单独缓存即可）。
4. **完整的 pipeline**：从双向 teacher → 因果学生 → DMD 蒸馏，做到了 24 FPS 实时，是少数真正强调 *interactive simulator* 落地的工作。
5. **跨域验证**：Minecraft + 双臂机器人都跑通，说明 multi-agent video world model 的范式有迁移性。

### 局限性

1. **强 Cosmos-Predict 依赖**：基于 NVIDIA 自家 [[Cosmos-Predict2.5]]-2B，且训练用了 32 × GB200（教师 + 学生 + 蒸馏各一轮）。学术界很难复现。
2. **2 agent → 4 agent 的"零样本"是相对慷慨的描述**：simplex 池 $V=4$ 是训练时就预留好的，等价于"训练时随机激活其中 2 个"。如果要支持 $V=8$ agent 需要重新选 $d_p$ 并可能重训，论文没给出 $V > 4$ 的实验。
3. **缺乏物理一致性显式约束**：作者自己承认"不显式强制 3D 几何或物理约束，长 rollout 可能累积不一致"。在长视野下，hub 中介的稀疏交互能不能保持物体永久守恒还需要更长 horizon 验证。
4. **评估在 Minecraft + 双臂机器人**：场景同质性较高（要么室内积木风格游戏，要么固定相机的桌面操作），没有触及驾驶 / 多视角无人机 / 户外协作等更复杂多 agent 场景。
5. **hub token 是黑盒**：$K=8$ 的 hub 学到了什么？论文没分析 hub 是否聚类成了"环境 / 物体 / agent 角色"等可解释的概念，留下了机会。
6. **8 agent 实验只测了延迟**：Figure 3 给出 8 agent 的 FLOPs/延迟，但没有生成质量数字（Table 1 都是 2 agent，Figure 5 才有 4 agent）。**$P \to 8$ 的质量退化曲线** 缺席。

### 潜在改进方向

1. **层次化 agent 分组**：当 $V$ 很大时可以做 simplex 嵌套或两级 hub（agent → 局部 hub → 全局 hub），保持稀疏性的同时增加表达能力。
2. **3D 几何先验**：把 [[3D Gaussian Splatting]] 或 NeRF style scene token 作为 hub，可能解决物理一致性问题。
3. **跨 agent 异质化**：现在 agent 同质（共享 action encoder、相同 RoPE 顶点池）。如果 agent 能力不同（一个机械臂 + 一只无人机），需要支持**异质 simplex 子池**。
4. **更长 horizon 评估**：Self-Forcing 解决短期 exposure bias，但 24 秒的视频是否能持续到 1 分钟？需要长时一致性实验。
5. **hub token 可视化与压缩**：分析 hub 学到的 token 表达，可能进一步压缩到 $K < 8$。

### 可复现性评估

- [x] 代码开源（项目主页声称 GitHub repo，但需查看）
- [ ] 预训练模型（论文未明确）
- [x] 训练细节完整（附录 D 给出超参、阶段、迭代数）
- [ ] 数据集可获取（Minecraft 自建数据未公开，RealOmin-Open 是 Gen Robot 私有数据集）
- [ ] 训练硬件可及（32 × GB200 一般实验室没有）

---

## 关联笔记

### 基于

- [[Cosmos-Predict2.5]]: 2B 视频 DiT 底座，论文整套 DiT 架构从这里 fine-tune
- [[Diffusion Forcing]]: 块因果 + per-block 独立噪声级的训练范式
- [[Self-Forcing]] / [[CausVid]]: 双向 teacher → 因果学生的蒸馏路线
- [[Flow Matching]]: 训练目标
- [[DMD]] (Distribution Matching Distillation): 蒸馏少步生成器的损失
- [[RoPE]]: 旋转位置编码基础
- [[ReRoPE]]: 把额外坐标轴塞进现有 RoPE band 的思路

### 对比

- [[Solaris]]: 最主要对手 — dense joint attention + per-player ID embedding，γ-World 各项指标都领先 40%–50%
- Multiverse (Frame Concat baseline): 单流多视角拼接，丢失独立性
- [[WorldMem]] / [[Matrix-Game]] / [[Hy-World]]: 同期单 agent 交互视频世界模型，对比维度不同
- [[GAIA-1]] / [[UniSim]]: 早期单 agent 世界模型

### 方法相关

- [[Simplex Rotary Agent Encoding]]: 核心创新点 — 用正单纯形顶点编码 agent 身份
- [[Sparse Hub Attention]]: 核心创新点 — hub 中介的稀疏跨 agent 注意力
- [[Block-Causal Attention]]: 块因果注意力
- [[DiT]] / [[AdaLN]]: 底层架构
- [[KV Cache]]: 流式推理
- [[World Model]]: 总领概念

### 硬件 / 数据相关

- 32 × NVIDIA GB200: 训练硬件
- RealOmin-Open Dataset: 真实双臂数据来源
- Minecraft：游戏域 2-agent / 4-agent 数据

---

## 速查卡片

> [!summary] Gamma-World: Generative Multi-Agent World Modeling Beyond Two Players
> - **核心**: 用单纯形 RoPE + Hub 稀疏注意力把视频世界模型扩展到多 agent
> - **方法**: Simplex Rotary Agent Encoding（4D RoPE 加 agent 轴）+ Sparse Hub Attention（$\mathcal{O}(P)$ 跨 agent 通信）+ 双向 teacher → 块因果学生 → DMD 蒸馏
> - **结果**: 5 个评估维度 × FVD/FID 共 10 个指标全面超 Solaris（FVD ↓约 40%、FID ↓约 50%）；只在 2 agent 训练即可零样本扩到 4 agent；24 FPS 实时
> - **代码**: https://github.com/nv-tlabs/Gamma-World

---

*笔记创建时间: 2026-05-28*
