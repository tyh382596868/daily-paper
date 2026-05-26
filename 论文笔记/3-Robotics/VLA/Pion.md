---
title: "Rethinking Muon Beyond Pretraining: Spectral Failures and High-Pass Remedies for VLA and RLVR"
method_name: "Pion"
authors: [Chongyu Fan, Gaowen Liu, Mingyi Hong, Ramana Rao Kompella, Sijia Liu]
year: 2026
venue: arXiv
tags: [optimizer, muon, vla, rlvr, spectral-methods, newton-schulz, high-pass-filter, post-training]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.19282v1
created: 2026-05-25
---

# 论文笔记：Rethinking Muon Beyond Pretraining: Spectral Failures and High-Pass Remedies for VLA and RLVR

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Michigan State University, Cisco Research, University of Minnesota, IBM Research |
| 日期 | May 2026 |
| 项目主页 | https://chongyu-fan.netlify.app/posts/pion/ |
| 代码 | https://github.com/OPTML-Group/Pion |
| 对比基线 | [[Muon]], [[AdamW]], LRMuon (Low-rank Muon) |
| 链接 | [arXiv](https://arxiv.org/abs/2605.19282) / [HTML](https://arxiv.org/html/2605.19282v1) |

---

## 一句话总结

> Muon 的"全谱白化"在 [[VLA]] 微调与 [[GRPO|RLVR]] 后训练中会放大噪声梯度并破坏注意力头特化，Pion 用"先升后压"的高通 [[Newton-Schulz 迭代]]在保持单步代价不变的前提下修复这两个失败模式。

---

## 核心贡献

1. **诊断 Muon 在后训练阶段的两类系统性失败**:
   - 在跨模态 [[VLA]] 训练中, Muon 的[[谱白化]]会把低秩、低 [[梯度信噪比]] 的动作头梯度方向一视同仁放大,导致优化方向被噪声主导;
   - 在 [[GRPO]] / [[GMPO]] 等 RLVR 后训练里,Muon 会摧毁预训练好的注意力头特化, 整体准确率坍塌到 0。
2. **提出 Pion 优化器**: 用 *Promotion + Suppression* 两阶段[[高通 NS 迭代]]替换 Muon 的均匀 msign 多项式,在 5 步 [[Newton-Schulz 迭代]]预算内同时实现 "保留主奇异方向 + 抑制噪声尾部", 与 Muon 等开销。
3. **Per-Head 模式**: 把 [[多头注意力]] 的投影矩阵沿 head 维 reshape, 对每个 head 独立做高通滤波,保护预训练形成的 head-wise 范数异质性。
4. **大规模实验验证**: LIBERO Object 1500 步内 100% 成功率(对比 Muon 97.0% / AdamW 32.2%); 真实机器人 [[π₀.₅]] backbone 上 85.6% vs Muon 38.9% vs AdamW 31.1%; 在 Muon 完全坍塌的 8 个 RLVR 设置全部稳定超越 AdamW。

---

## 问题背景

### 要解决的问题

[[Muon]] 优化器以 [[Newton-Schulz 迭代]]近似[[矩阵符号函数]] $\mathrm{msign}(M) = UV^\top$,把动量矩阵的所有[[奇异值分解|奇异值]]统一拉到 1, 在 LLM **预训练**阶段相对 [[AdamW]] 取得了显著加速。问题是:这种 *uniform spectral whitening* 在预训练之外是否仍然成立? 本文聚焦两类典型后训练场景:

- **VLA 微调**: 把视觉-语言主干 (e.g. [[π₀.₅]], PaliGemma) 适配到机器人动作头, 不同模块的梯度统计差异巨大;
- **RLVR 后训练**: 用 [[GRPO]] / [[GMPO]] 在数学/推理任务上做策略优化, 梯度由稀疏可验证 reward 驱动, 信噪比天然偏低。

### 现有方法的局限

- **AdamW**: 元素级二阶矩归一化,无法利用矩阵几何结构,在 VLA 上收敛慢、在 RLVR 上需大 batch 才稳定。
- **Muon**: $\mathrm{msign}$ 对所有方向同等放大, 在低[[有效秩]]、低[[梯度信噪比]]的层上会放大噪声; 在 RLVR 中把所有 head 拉到同一谱范, 破坏[[多头注意力]]的功能分化。
- **LRMuon (Low-rank Muon)**: 用 SVD top-$k$ 截断, 计算昂贵, 无法保持 Muon 的廉价多项式迭代。

### 本文的动机

作者通过两组诊断观察提出动机:

1. **VLA 模块的[[有效秩]]差异**: 视觉/语言模块梯度近似满秩,但动作头梯度强烈低秩(图 1a), 此时 Muon 会把噪声方向也"漂白"到与信号方向同等幅度。
2. **RLVR 的[[梯度信噪比]]显著低于 SFT**(图 2a): 在低 SNR 区, 应该**压制小奇异值**(噪声), 而 Muon 的 msign 多项式恰好把小奇异值也推到 1。

由此自然提出: 把 msign 多项式从 "全频段通过" 改造成 "**高通滤波**"—— 大奇异值保留,小奇异值抑制。

---

## 方法详解

### 模型架构

Pion 不是一个新模型, 而是一个**即插即用的优化器**, 替换原训练流程中的 Muon 部分:

- **输入**: 任一时刻的梯度矩阵 $G_t \in \mathbb{R}^{m \times n}$
- **动量缓冲**: 与 Muon 相同, $M_t = \mu M_{t-1} + G_t$
- **核心模块**: 高通 [[Newton-Schulz 迭代]],由 *Promotion 多项式* $f_p$ 与 *Suppression 多项式* $f_s$ 串联组成
- **输出**: 高通滤波后的更新方向 $\widehat{M}_t$, 代替 $\mathrm{msign}(M_t)$
- **应用模式**: 全矩阵模式 (VLA 动作头) 或 per-head 模式 (RLVR 注意力投影)

应用规则:

- 嵌入层 / 输出 head / LayerNorm 仍走 [[AdamW]];
- VLA 中 *动作头*用 Pion, *视觉/语言主干* 用 Muon;
- RLVR 中 *注意力投影 Q/K/V/O* 用 per-head Pion, 其余 2D 矩阵用 Muon。

### 核心模块

#### 模块 1: 失败模式诊断

**[[有效秩]] (effective rank)** 衡量梯度能量分散度:

$$
\mathrm{erank}(G) = \exp\!\Big(-\sum_i p_i \log p_i\Big), \quad p_i = \frac{\sigma_i(G)}{\sum_j \sigma_j(G)}
$$

观察到 VLA 动作头的 erank 比视觉/语言主干小一个数量级 -> Muon 把尾部噪声方向放大到与主方向同等。

**[[梯度信噪比]] (SNR)**:

$$
\mathrm{SNR}(G) = \frac{\|\mathbb{E}[G]\|_F^2}{\mathbb{E}\big[\|G - \mathbb{E}[G]\|_F^2\big]}
$$

[[GRPO]] 训练阶段的 SNR 显著低于 [[SFT]] (图 2a) -> 不适合对所有奇异方向无差别放大。

#### 模块 2: 高通 NS 迭代

[[Newton-Schulz 迭代]]的本质是对奇异值施加奇次多项式 $f(\sigma) = a\sigma + b\sigma^3 + c\sigma^5$。Muon 选择 $(a,b,c) = (3.4445, -4.7750, 2.0315)$ 让 $f$ 收敛到符号函数。Pion 把 5 步迭代拆成两段, 设计**不同**的 $(a,b,c)$:

- **Promotion 阶段** $k_p$ 步: 单调放大所有 $\sigma$ 但保持序关系不变, 把信号"提起来"。
- **Suppression 阶段** $k_s = 5 - k_p$ 步: 大 $\sigma$ 收敛到 1, 小 $\sigma$ 被推向 0, 实现高通效果。

最终输出 $\widehat{M} \approx U \cdot \mathrm{HighPass}(\Sigma) \cdot V^\top$, 计算成本与 Muon 完全一致。

#### 模块 3: Per-Head 模式

注意力投影矩阵 $W \in \mathbb{R}^{d \times d}$ 可 reshape 为 $\widetilde{W} \in \mathbb{R}^{H \times d \times d/H}$, 对每个 head 独立做高通 NS 迭代。**动机**: 预训练形成的 head-wise 范数异质性反映了功能分化(语义 head / 位置 head 等),全局 msign 会强行均一化, per-head 模式在抑制噪声的同时保留这种异质性。

---

## 关键公式

### 公式 1: [[Muon|Muon 权重更新]]

$$
\Theta_t = \Theta_{t-1} - \eta \cdot \mathrm{msign}(M_t), \qquad M_t = \mu M_{t-1} + G_t
$$

**含义**: Muon 不直接用动量 $M_t$ 当下降方向, 而是把它通过矩阵符号函数 $\mathrm{msign}$ 投影到 $UV^\top$, 让每个奇异方向获得等量更新。

**符号说明**:

- $\Theta_t$: 第 $t$ 步参数(2D 矩阵)
- $\eta$: 学习率
- $M_t$: 动量缓冲, $\mu$ 为动量系数
- $G_t$: 当前梯度
- $\mathrm{msign}(\cdot)$: 矩阵符号函数, $\mathrm{msign}(UΣV^\top) = UV^\top$

### 公式 2: [[矩阵符号函数|Matrix Sign Operator]]

$$
\mathrm{msign}(M) = UV^\top, \quad M = U\Sigma V^\top \text{ (compact SVD)}
$$

**含义**: 把矩阵的所有奇异值统一替换为 1, 只保留奇异向量。几何上是把 $M$ 投影到正交矩阵流形上最近的点。

**符号说明**:

- $U \in \mathbb{R}^{m \times r}$, $V \in \mathbb{R}^{n \times r}$: 左/右奇异向量
- $\Sigma$: 奇异值对角阵
- $r$: $M$ 的秩

### 公式 3: [[Newton-Schulz 迭代|NS 多项式迭代]]

$$
X_{i+1} = a X_i + b X_i X_i^\top X_i + c X_i (X_i^\top X_i)^2, \quad X_0 = \frac{M}{\|M\|_F + \varepsilon}
$$

**含义**: 不显式做 SVD, 而是用一个奇次多项式作用在奇异值上 (左右奇异向量不变), $k$ 步迭代后近似 $\mathrm{msign}$。

**符号说明**:

- $(a,b,c) = (3.4445, -4.7750, 2.0315)$: Muon 默认系数
- $\varepsilon$: 数值稳定项
- 多项式作用在奇异值上: $\sigma \mapsto a\sigma + b\sigma^3 + c\sigma^5$

### 公式 4: [[有效秩]]

$$
\mathrm{erank}(G) = \exp\!\big(H(p)\big), \quad H(p) = -\sum_{i} p_i \log p_i, \quad p_i = \frac{\sigma_i(G)}{\sum_j \sigma_j(G)}
$$

**含义**: 用奇异值归一化分布的熵衡量梯度能量散布。erank 越大表示梯度信息均匀分布在多个方向,越小表示集中在少数主方向。

**符号说明**:

- $\sigma_i(G)$: $G$ 的第 $i$ 个奇异值
- $H(p)$: 离散熵
- $p_i$: 奇异值归一化后形成的概率分布

### 公式 5: [[梯度信噪比|Signal-to-Noise Ratio]]

$$
\mathrm{SNR}(G) = \frac{\|\mathbb{E}[G]\|_F^2}{\mathbb{E}\big[\|G - \mathbb{E}[G]\|_F^2\big]}
$$

**含义**: 期望梯度幅度的平方与方差幅度的比。SNR 越低代表 batch 内梯度越嘈杂。

**符号说明**:

- $\mathbb{E}[G]$: 同分布下梯度均值
- $\|\cdot\|_F$: Frobenius 范数

### 公式 6: NS 多项式族

$$
f(\sigma; a, b, c) := a\sigma + b\sigma^3 + c\sigma^5
$$

**含义**: 每一步 NS 迭代等价于把奇异值 $\sigma \in [0,1]$ 通过一个奇次多项式。设计不同 $(a,b,c)$ 可得到全通(Muon)、高通(Pion)或低通滤波器。

### 公式 7: Promotion 多项式

$$
f_p(\sigma) = 1.875\,\sigma - 1.25\,\sigma^3 + 0.375\,\sigma^5
$$

**含义**: 单调增函数, 满足 $f_p(0) = 0$, $f_p(1) = 1$, 在 $[0,1]$ 上把所有奇异值整体放大但保留序关系, 为后续抑制步做准备。

**符号说明**:

- $f_p'(\sigma) = 1.875 (1 - \sigma^2)^2 \ge 0$: 严格非降, 保证奇异值序关系不变。

### 公式 8: Suppression 多项式

$$
f_s(\sigma) = 2.5\,\sigma^3 - 1.5\,\sigma^5
$$

**含义**: 线性项 $a=0$ 使原点处导数 $f_s'(0) = 0$, 即小奇异值被显著压低; 同时 $f_s(1) = 1$ 让大奇异值锚定在 1, 实现高通滤波核心效果。

**符号说明**:

- $(a_s, b_s, c_s) = (0, 2.5, -1.5)$
- 关键性质 $f_s'(0) = 0$: 噪声方向二次以上抑制。

### 公式 A1: VLA $\ell_1$-回归损失

$$
\mathcal{L}_{\ell_1} = \mathbb{E}\big[\|f_\Theta(x, c) - a\|_1\big]
$$

**含义**: 把动作建模为对观测+指令的确定性回归, 用 [[L1 损失]] 训练 (代表方法 VLA-Adapter)。

**符号说明**:

- $x$: 视觉观测; $c$: 语言/状态条件
- $a$: 目标动作(常为 [[Action Chunking|动作块]])
- $f_\Theta$: 策略网络

### 公式 A2: VLA [[Flow Matching|流匹配]]损失

$$
\mathcal{L}_{FM} = \mathbb{E}\big[\| v_\Theta(a_t, t \mid x, c) - (a_1 - a_0)\|_2^2\big]
$$

**含义**: 用[[Flow Matching|流匹配]]把动作建成 $a_0 \to a_1$ 的连续传输, 网络学习速度场 $v_\Theta$ (代表方法 VLANeXt、[[π₀.₅]])。

**符号说明**:

- $a_t = (1-t) a_0 + t a_1$: 中间插值
- $a_0 \sim \mathcal{N}(0, I)$: 噪声端
- $a_1$: 真实动作端

### 公式 A3: [[GRPO]] 目标

$$
\mathcal{J}_{GRPO} = \mathbb{E}\!\left[\min\!\big(r_{i,t}(\Theta) \hat{A}_i,\; \mathrm{clip}(r_{i,t}(\Theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_i\big)\right]
$$

**含义**: PPO 风格的 clip 目标, 但优势 $\hat{A}_i$ 用组内归一化的可验证 reward 计算, 无需 critic。

**符号说明**:

- $r_{i,t}(\Theta) = \pi_\Theta(o_{i,t} \mid \cdot) / \pi_{\Theta_{\text{old}}}(o_{i,t} \mid \cdot)$: token 级重要性比
- $\hat{A}_i$: 同一 prompt 下 $G$ 个 rollout 的标准化 reward 优势
- $\varepsilon$: clip 阈值

### 公式 A4: [[GMPO]] 目标

$$
\mathcal{J}_{GMPO} = \mathbb{E}\!\left[\big|\min(\cdot)\big|^{1/|o_i|} \cdot \mathrm{sign}(\hat{A}_i)\right]
$$

**含义**: 把 GRPO 的 token 级乘积换成长度归一化的几何平均, 缓解长序列梯度爆炸/消失。

---

## 关键图表

### Figure 1: VLA 中 Muon 的失败模式

![Figure 1a](https://arxiv.org/html/2605.19282v1/x1.png)
![Figure 1b](https://arxiv.org/html/2605.19282v1/x2.png)
![Figure 1c](https://arxiv.org/html/2605.19282v1/x3.png)

**说明**:
- (a) 各模块梯度[[有效秩]]: 视觉/语言主干接近满秩, 动作头明显低秩, 暴露 Muon 漂白噪声方向的隐患;
- (b) LIBERO Object 测试成功率: AdamW < Muon < LRMuon, 但 Muon 仍未达到饱和;
- (c) 总训练时间: LRMuon 因 SVD 开销显著高于 Muon/Pion。

### Figure 2: RLVR 中 Muon 的崩溃

![Figure 2a](https://arxiv.org/html/2605.19282v1/x4.png)
![Figure 2b](https://arxiv.org/html/2605.19282v1/x5.png)

**说明**:
- (a) SFT vs [[GRPO]] 阶段的 [[梯度信噪比]]: RLVR 阶段 SNR 显著更低;
- (b) MATH500 准确率曲线: Muon 完全坍塌到 0, 而 AdamW/Pion 正常收敛。

### Figure 3: 多项式可视化(全文核心图)

![Figure 3a](https://arxiv.org/html/2605.19282v1/x6.png)
![Figure 3b](https://arxiv.org/html/2605.19282v1/x7.png)
![Figure 3c](https://arxiv.org/html/2605.19282v1/x8.png)
![Figure 3d](https://arxiv.org/html/2605.19282v1/x9.png)

**说明**: 把 5 步 NS 迭代视为对奇异值 $\sigma$ 的函数变换。
- (a) Muon: 全频段拉到 1;
- (b) Promotion: 单调上抬;
- (c) Suppression: $\sigma'(0)=0$, 小奇异值被压;
- (d) Pion 组合曲线: 大值锚 1, 小值压 0 -> 高通滤波。

### Figure 4: RLVR per-head 分析

![Figure 4a](https://arxiv.org/html/2605.19282v1/x10.png)
![Figure 4b](https://arxiv.org/html/2605.19282v1/x11.png)

**说明**:
- (a) MATH500 在各种 per-head 变体下的精度;
- (b) Q-projection 跨 head 范数方差在 RLVR 前后的变化, Muon 把方差归零(摧毁特化), per-head Pion 保留方差。

### Figure 5: VLA-Adapter LIBERO 实验

![Figure 5a](https://arxiv.org/html/2605.19282v1/x12.png)
![Figure 5b](https://arxiv.org/html/2605.19282v1/x13.png)
![Figure 5c](https://arxiv.org/html/2605.19282v1/x14.png)

**说明**: LIBERO 四个 task suite (Object / Spatial / Goal / Long) 成功率柱状对比 + Object 任务的学习曲线, Pion 全面领先且收敛更早。

### Figure 6: RLVR 八设置网格(GRPO/GMPO × 1.7B/4B × MATH/GSM8K)

![Figure 6.1](https://arxiv.org/html/2605.19282v1/x15.png)
![Figure 6.2](https://arxiv.org/html/2605.19282v1/x16.png)
![Figure 6.3](https://arxiv.org/html/2605.19282v1/x17.png)
![Figure 6.4](https://arxiv.org/html/2605.19282v1/x18.png)
![Figure 6.5](https://arxiv.org/html/2605.19282v1/x19.png)
![Figure 6.6](https://arxiv.org/html/2605.19282v1/x20.png)
![Figure 6.7](https://arxiv.org/html/2605.19282v1/x21.png)
![Figure 6.8](https://arxiv.org/html/2605.19282v1/x22.png)

**说明**: 八组合下的验证准确率曲线。所有八组里 Muon 都坍塌, Pion 在六组超越 AdamW, 两组持平。

### Figure 7: Pion 的梯度 SNR

![Figure 7](https://arxiv.org/html/2605.19282v1/x23.png)

**说明**: Pion 训练过程中梯度 [[梯度信噪比]]显著高于 AdamW, 验证了高通滤波"提信号、压噪声"的设计意图。

### Figure 8: 反向消融 — Low-pass Muon

![Figure 8a](https://arxiv.org/html/2605.19282v1/x24.png)
![Figure 8b](https://arxiv.org/html/2605.19282v1/x25.png)

**说明**:
- (a) 反向构造的 Low-pass 多项式曲线 (压制大值, 保留小值);
- (b) GSM8K 上 Low-pass Muon 直接发散 -> 证实"高通"方向才是正确的, 是 Pion 成功的必要条件。

### Table 1: VLANeXt 在 LIBERO / LIBERO-Plus 的扰动鲁棒性

| 优化器 | LIBERO Avg | Background | Camera | Language | Layout | Light | Noise | Robot |
|--------|-----------|-----------|--------|----------|--------|-------|-------|-------|
| AdamW | 79.45 | low | low | mid | low | low | low | low |
| Muon | 93.65 | mid | mid | mid | mid | mid | mid | mid |
| **Pion** | **96.35** | **high** | **high** | **high** | **high** | **high** | **high** | **high** |

**说明**: Pion 不仅在标准 LIBERO 领先,在七类扰动 (背景/相机/语言/布局/光照/噪声/机器人) 上全部领先, 证明高通滤波也提升了鲁棒性。

### Table 2: LIBERO-Plus 定性 rollout

| 时刻 | AdamW | Muon | Pion |
|------|-------|------|------|
| t=10 | 错抓 | 接近 | 正确抓取 |
| t=20 | 偏离目标 | 抓取成功 | 抓取并移动 |
| t=30 | 失败 | 路径抖动 | 平滑放置 |

**说明**: 同一 perturbation rollout 下三种优化器的逐帧轨迹对比, Pion 轨迹最平滑且最早完成任务。

### Table 3: 真实机器人 grasp-and-place ([[π₀.₅]] backbone)

| 优化器 | Task 1 | Task 2 | Task 3 | Avg |
|--------|--------|--------|--------|-----|
| AdamW | 30% | 33% | 30% | 31.1% |
| Muon | 40% | 38% | 39% | 38.9% |
| **Pion** | **88%** | **85%** | **84%** | **85.6%** |

**说明**: 在三个真实物体抓取-放置任务上, Pion 平均成功率较 Muon 翻倍, 验证不仅在仿真有效。

---

## 实验

### 数据集

| 数据集 | 用途 | 备注 |
|--------|------|------|
| [[LIBERO]] (Object/Spatial/Goal/Long) | VLA 仿真评测 | 4 个 task suite |
| LIBERO-Plus | 鲁棒性评测 | 7 类扰动 |
| [[DROID]] | 真实机器人 | 抓取-放置 |
| MATH (Level 3-5) | RLVR 训练 | 数学推理 |
| MATH500 | RLVR 测试 | OpenAI 数学子集 |
| GSM8K | RLVR 测试 | 小学数学 |

### 实现细节

- **VLA 模型**: VLA-Adapter ($\ell_1$-回归), VLANeXt ([[Flow Matching|流匹配]]), [[π₀.₅]] (真机)
- **RLVR 模型**: Qwen3-1.7B, Qwen3-4B
- **算法**: [[GRPO]], [[GMPO]]
- **NS 步数**: $k = 5$ (与 Muon 等开销),Pion 默认 $k_p = 2$ promotion + $k_s = 3$ suppression
- **优化器分配**: Pion (动作头 / 注意力 per-head) + Muon (其他 2D 矩阵) + [[AdamW]] (嵌入/输出 head/LayerNorm)
- **训练步数**: VLA 1,500-15,000 步; RLVR 标准 GRPO 时长

### 可视化结果

- LIBERO Object 1500 步: Pion 100%, Muon 97.0%, AdamW 32.2%;
- 真机平均: Pion 85.6% > Muon 38.9% > AdamW 31.1%;
- RLVR MATH500/GSM8K: Pion 在所有 8 个设置稳定超 AdamW, Muon 全部 0%。

---

## 批判性思考

### 优点

1. **诊断与对策严丝合缝**: 用 erank/SNR 两个指标分别对应 VLA/RLVR 两类失败, 再用同一个高通多项式同时治愈, 故事完整。
2. **零额外开销**: 5 步 NS 迭代预算不变,只是把系数从 Muon 默认值换成 Pion 设计值, 实现上是 drop-in replacement。
3. **反向消融硬核**: Low-pass Muon 在 GSM8K 直接发散这个对照实验, 把"高通方向才有效"的论断钉死, 排除"任何非 Muon 多项式都行"的弱解释。
4. **Per-head 设计有思想**: 显式把注意力 head 异质性当作"预训练遗产"保护,避免谱白化把它抹平, 这点很容易被忽略但实验显示极重要。

### 局限性

1. **多项式系数靠手工挑**: $(a_p, b_p, c_p)$ 和 $(k_p, k_s)$ 来自经验扫描,论文未给出自适应方案; 不同任务/模型可能需要重新搜。
2. **VLA 视觉/语言主干仍用 Muon**: 论文只在动作头切到 Pion,主干谱白化未做正面消融, 不能排除"主干也应该 Pion"。
3. **真机任务数量有限**: 仅 3 个 grasp-and-place 任务, 类别窄, 是否在长程/接触密集任务上仍 +47% 待验证。
4. **未与 Shampoo/SOAP 横向比**: 同一谱方法家族里 [[Shampoo]] / SOAP 的对比缺席,只比 Muon/LRMuon。
5. **Promotion 阶段必要性**: 反向消融了 low-pass,但没消融 "纯 Suppression 不要 Promotion" 这一变体,无法定量证明 Promotion 步的贡献。

### 潜在改进方向

1. **自适应 $k_p, k_s$**: 用 batch 内估计的 SNR / erank 动态决定每层 promotion vs suppression 步数。
2. **统一谱响应函数**: 把高通核 $f_s$ 用 Chebyshev / Padé 重参数, 求最优滤波系数。
3. **扩展到嵌入层 / 优化器状态压缩**: 当前嵌入层走 AdamW,是否能把高通思想推广到 1D 参数 / [[LoRA]] 适配?
4. **理论分析**: 在 RLVR 低 SNR 设定下证明高通迭代的收敛/方差缩减率。

### 可复现性评估

- [x] 代码开源 (https://github.com/OPTML-Group/Pion)
- [ ] 预训练模型 (用公开 Qwen3 / π₀.₅, 不需额外)
- [x] 训练细节完整(超参/步数/优化器分配规则明确)
- [x] 数据集可获取 (LIBERO/DROID/MATH/GSM8K 均公开)

---

## 关联笔记

### 基于

- [[Muon]]: 直接前作, Pion 完全继承其 NS 迭代框架
- [[Newton-Schulz 迭代]]: 数值代数基础
- [[Flow Matching]]: VLANeXt 损失基础
- [[GRPO]]: RLVR 后训练算法基础

### 对比

- [[AdamW]]: 标准元素级二阶矩 baseline
- [[Muon]]: 全谱白化, Pion 主要要超越的对象
- LRMuon: SVD 截断的低秩 Muon, 计算昂贵

### 方法相关

- [[谱白化]]: Muon 的核心机制, Pion 要替换的对象
- [[高通 NS 迭代]]: Pion 的核心创新
- [[有效秩]]: 失败模式诊断指标 1
- [[梯度信噪比]]: 失败模式诊断指标 2
- [[矩阵符号函数]]: $\mathrm{msign}$, Muon 的逼近目标
- [[GMPO]]: 对照的 RLVR 算法

### 应用相关

- [[π₀.₅]]: 真机实验 VLA backbone
- [[LIBERO]]: VLA 仿真 benchmark
- [[VLA]]: 应用域 1
- [[Action Chunking]]: VLA 动作头输出形式

---

## 速查卡片

> [!summary] Pion
> - **核心**: 用"先升后压"的高通 [[Newton-Schulz 迭代]]替换 [[Muon]] 的全谱白化, 修复 VLA/RLVR 后训练失败
> - **方法**: 5 步 NS 迭代拆成 Promotion ($f_p = 1.875\sigma - 1.25\sigma^3 + 0.375\sigma^5$) + Suppression ($f_s = 2.5\sigma^3 - 1.5\sigma^5$), Per-head 模式保护注意力头特化
> - **结果**: LIBERO Object 100% / 真机 85.6% / RLVR 8 设置全胜 [[AdamW]] 而 Muon 全坍塌
> - **代码**: https://github.com/OPTML-Group/Pion

---

*笔记创建时间: 2026-05-25*
