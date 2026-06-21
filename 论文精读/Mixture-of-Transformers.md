---
type: deep-reading
title: "Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models"
method_name: "Mixture-of-Transformers"
authors: [Weixin Liang, Lili Yu, Liang Luo, Srinivasan Iyer, Ning Dong, Chunting Zhou, Gargi Ghosh, Mike Lewis, Wen-tau Yih, Luke Zettlemoyer, Xi Victoria Lin]
year: 2025
venue: TMLR
arxiv: https://arxiv.org/abs/2411.04996
tags: [multimodal, sparse-architecture, mixture-of-transformers, foundation-model, efficient-pretraining, modality-aware-sparsity]
image_source: online
source: websearch
created: 2026-06-21
---

# 精读：Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models

> **一句话**：在 Transformer 的每个非嵌入层按「模态」拆开权重（FFN/注意力投影/归一化各模态一套），但自注意力仍跨全序列共享，用一半左右的算力达到 dense 多模态模型的效果。

---

## TL;DR（30 秒）

- **解决什么问题**：原生多模态（文本+图像+语音）预训练算力爆炸——所有模态共用一套 dense 参数，但不同模态的统计特性差异巨大，强行共享既浪费又互相干扰。
- **核心 idea**：**modality-aware sparsity**——把每个 Transformer 层里除 embedding 外的全部参数（FFN、Q/K/V/O 投影、LayerNorm）按模态复制成多套，token 根据自身模态**确定性地**选用对应那套；唯独 self-attention 的「算注意力」这一步跨所有模态全局共享。
- **为什么有效**：路由是「看 token 是哪种模态」而不是「学一个 gate」，零路由开销、训练稳定；模态各自有专属参数避免梯度打架，但全局注意力保住了跨模态融合。
- **最强证据**：Chameleon 7B（文+图自回归）设定下，MoT 只用 **55.8% 的 FLOPs** 就追平 dense 基线；扩到含语音时 MoT (443M) 只用 **37.2% FLOPs** 达到 dense 级语音质量；系统实测 dense 级图像质量只花 **47% 的 wall-clock 时间**。
- **代码/主页**：https://github.com/facebookresearch/Mixture-of-Transformers （TMLR 2025）

---

## 0. 阅读前置（先懂这些再往下）

读懂本文你需要先理解：

- [[Transformer]] — 本文改的就是标准 decoder Transformer 块（注意力 + FFN + 归一化），需要清楚每个子层在哪、参数占比多少。
- [[MoE]] — Mixture-of-Experts：稀疏专家 + 学习式 gating router。MoT 是它的「对照组」——同样想稀疏化，但路由方式完全不同（按模态硬路由 vs 学习软路由）。
- [[Chameleon]] — Meta 的原生多模态模型：把图像也离散成 token，文本和图像在同一个自回归序列里混着训。本文的主实验设定之一。
- [[Discrete Tokenization]] / [[VQ Tokenizer]] — 图像/语音怎么变成离散 token 进自回归序列。
- [[DiT]] / [[Flow Matching]] — Transfusion 设定里图像走扩散/连续目标而非离散自回归，需要懂扩散 Transformer。
- [[RMSNorm]] / [[LayerNorm]] — 归一化层，本文也把它按模态拆开了。
- [[Autoregressive Modeling]] — 自回归下一 token 预测，文本和（离散）图像的训练目标。

> 如果上面有你不熟的，先点进概念笔记补一下。尤其 [[MoE]] 和 [[Chameleon]] 不熟的话 §2、§6 会读得吃力。

---

## 1. 问题是什么 & 为什么难

### 1.1 任务设定
- **输入**：一条混合模态序列——文本 token、图像 token（离散化或连续 patch）、可选语音 token，按任意顺序交错（interleaved）。
- **输出**：原生地生成任意模态——给文本续写文本、给提示生成图像、给图像生成描述，等等。是「一个模型统一生成所有模态」，不是分模型拼接。
- **评价指标**：各模态各自的验证损失 / 困惑度；图像用 COCO-30k 的 **FID**（越低越好）、**CLIP score**（越高越好）；语音用语音质量指标；统一对比「达到同等质量所需的 **FLOPs / 训练步数 / wall-clock 时间**」。

### 1.2 朴素做法为什么不行
最直接的做法就是 [[Chameleon]] 那套：**所有模态共用一套 dense Transformer 参数**，把图像也 tokenize 成离散 token，和文本拼成一条序列，统一做下一 token 预测。问题有两层：

1. **模态统计差异巨大**。文本 token 分布稀疏、语义离散；图像 token 局部相关性强、分布连续平滑；语音又有自己的节律。逼一套 FFN/注意力投影同时拟合三种分布，等于让一个人同时练三种互相打架的技能——出现 **模态间梯度冲突 / 负迁移**，模型容量被内耗。
2. **要补回性能只能堆参数和算力**。dense 模型想让每种模态都学好，只能整体放大，FLOPs 随总参数线性涨。多模态预训练本就吞算力，这条路 scaling 成本难以承受。

那为什么不直接用 [[MoE]] 稀疏化？MoE 用一个**学习出来的 gating router** 把 token 分给若干专家。但在多模态场景它有自己的麻烦：router 要学、容易**负载不均**（load imbalance）、训练不稳、还要 auxiliary loss 去平衡；而且 MoE 的专家分工是「涌现」出来的，不保证按模态干净地分开。

### 1.3 本文的切入点
作者的观察：**「token 属于哪个模态」这个信息是免费的、确定的——我们本来就知道每个 token 是文本/图像/语音**。既然如此，何必让 router 去「学」一个本就已知的路由？直接**按模态硬路由**：每种模态配一整套自己的非嵌入参数。这把 MoE 那套「学路由」的全部麻烦（不稳、不均、辅助损失）一笔勾销，同时拿到了「模态特化」的全部好处。这就是理解全文的钥匙。

---

## 2. 核心思想（用直觉讲透）

**一句话核心**：把 Transformer 层「按模态克隆」成若干并行的专属子网络，token 按自己的模态对号入座地走对应子网络，但**所有 token 仍在同一个全局自注意力里互相看见**。

**直觉/类比**：想象一家联合国同传机构。
- **dense 基线**：只雇一个万能翻译，让他同时处理中/英/法所有发言。他什么都会一点，但每种都不精，且脑子里几种语言互相串味。
- **MoT**：雇三个专职译员——中文译员、英文译员、法文译员，各有各的笔记本和习惯（= 各模态专属的 FFN、Q/K/V/O 投影、归一化）。每段发言按其语种交给对应译员处理（**确定性路由**，看一眼语种就知道交给谁，不用「猜」）。**但关键**：三个译员坐在同一张圆桌旁，能听到彼此正在处理的所有内容（= 全局共享的自注意力），所以中文发言里提到的概念，法文译员也能据此调整自己的翻译。这就是「模态特化处理 + 跨模态信息共享」的统一。

具体走一遍一个图像 token 的旅程：它进入第 $\ell$ 层，先用**图像专属**的 LayerNorm 归一化，再用**图像专属**的 $W_Q^{\text{img}}, W_K^{\text{img}}, W_V^{\text{img}}$ 投影出 query/key/value——到这里全是图像那套参数。然后它的 query 去和**整条序列所有模态**的 key 做注意力（这一步是全局共享、没有模态之分的 softmax 注意力），把文本/语音 token 的信息也聚合进来。聚合结果再用**图像专属**的输出投影 $W_O^{\text{img}}$ 和**图像专属**的 FFN 处理。文本 token 同理走文本那套。

**为什么这能解决 §1 的难点**：
- 对应 §1.2 的「梯度冲突」：每种模态有专属参数，反传时各管各的，**不再互相覆盖**，负迁移消失，所以用更少算力就能学好每种模态。
- 对应 §1.2 的「MoE 不稳/不均」：路由是模态身份决定的常量掩码，**没有 router 要学、天然按模态均衡、不需要辅助损失**，训练和 dense 一样稳。
- 对应「还要跨模态融合」：自注意力保持全局，**模态间该交流的照样交流**，没牺牲多模态对齐能力。

---

## 3. 方法逐层拆解

### 3.1 总览：数据流

设输入序列每个位置 $t$ 有一个已知模态标签 $m_t \in \{\text{text}, \text{image}, \text{speech}\}$。MoT 把标准 Transformer 块的每个组件都参数化为「按模态选一套」。第 $\ell$ 层对位置 $t$ 的隐状态 $h_t$：

- **输入** $h_t$（带模态标签 $m_t$）
  → 取出模态 $m_t$ 专属的归一化与投影参数
  → **模态专属** Pre-Norm + Q/K/V 投影
  → **全局共享** self-attention（所有模态 token 在同一注意力矩阵里互相可见）
  → **模态专属** 输出投影 $W_O^{m_t}$
  → **模态专属** Pre-Norm + FFN
  → **输出** $h_t'$

唯一全模态共享的是：**(a) 算注意力分数那一步**（softmax over 全序列），**(b) 输入/输出 embedding 表**（token embedding 本就按模态各有词表，但「非嵌入参数」指的是层内权重）。其余非嵌入参数全部模态解耦。

> 实现上（见官方 repo）整条序列用 `modality_masks`（二值掩码）把各模态 token 分组，分别送进各自的专家模块算完，再按原始位置 merge 回去，保证序列顺序不变。

### 3.2 模块 A：ModalityUntiedFeedForward（模态解耦前馈网络）

- **是什么**：每种模态一套独立的 FFN（两层 MLP + 激活）外加独立的 pre-FFN 归一化层。token 按 `modality_masks` 确定性路由到自己那套 FFN。
- **为什么这样设计**：FFN 是 Transformer 里参数大头——repo 明确指出前馈网络约占 **67% 的参数量**，「拆开 FFN 就完成了一大半的活」。模态间真正「各自记忆/各自变换特征」的能力主要在 FFN，把它解耦收益最大。
- **怎么实现**：用 [[MLP]] 给每个模态实例化一份权重 $\{W_1^m, W_2^m\}$ 和一份 [[RMSNorm]]/[[LayerNorm]]；前向时按模态分组计算。无 router、无 gating，路由开销为零。
- **去掉会怎样**：若 FFN 共享、只拆注意力，模态特化的主体（占比最大的容量）被砍掉，效率增益会大幅缩水——这是 MoT 增益的主来源。

### 3.3 模块 B：ModalityUntiedAttention（模态解耦注意力）

- **是什么**：每种模态有独立的 query/key/value/output 投影矩阵 $W_Q^m, W_K^m, W_V^m, W_O^m$ 和独立的 pre-attn 归一化；但**投影完之后的注意力计算（QK^T softmax 加权 V）是全局的**，跨所有模态。
- **为什么这样设计**：「怎么把一个 token 投影进注意力空间」是模态相关的（图像 query 和文本 query 该长得不一样），但「谁该关注谁」应当跨模态自由——所以投影解耦、注意力本身共享。这是 MoT 保留多模态融合能力的关键设计点。
- **怎么实现**：每个模态先用自己的 $W_Q^m/W_K^m/W_V^m$ 投影出 Q/K/V，把所有模态的 Q/K/V 在序列维度拼回原顺序，做一次**统一的**因果 [[softmax 注意力]]（含 [[RoPE]]），输出再按模态用各自 $W_O^m$ 投影回去。
- **去掉会怎样**：若把注意力也按模态隔离（即各模态只在自己内部做注意力），就退化成「几个独立单模态模型并排」，跨模态对齐崩掉——这正是和 §2 圆桌类比里「译员能互相听见」对应的、不能省的部分。

### 3.4 模块 C：模态解耦归一化（Modality-Untied LayerNorm）

- **是什么**：连 pre-attention 和 pre-FFN 的归一化层也按模态各一份。
- **为什么这样设计**：不同模态的激活尺度/分布不同，共享一套 norm 的 scale/shift 参数会被某个主导模态拉偏。解耦归一化让每模态有自己的缩放统计。
- **怎么实现**：每模态实例化独立 [[RMSNorm]]/[[LayerNorm]] 的可学习参数。
- **去掉会怎样**：是相对小的收益项，但作者把它一并解耦以求「非嵌入参数全部模态特化」的彻底性；实现建议把 norm 直接写进 attention/FFN 类内部而非单独的 block。

> 每个模块的统一逻辑：**凡是模态相关的变换都各给一套，凡是需要跨模态交互的计算（注意力打分）保持全局**。这条分界线就是 MoT 的全部设计哲学。

---

## 4. 关键公式逐步推导

> 本文是架构论文，核心不在新损失而在「权重如何按模态选取」。以下公式给出 MoT 块的形式化（符号沿用标准 Transformer + 模态上标 $m$；因 PDF 在沙箱中不可直接抓取，公式按论文方法重建，保持与官方实现一致）。

### 公式 1：[[Mixture-of-Transformers Routing|模态确定性路由]]

**直觉先行**：不像 [[MoE]] 要学一个 gate 决定 token 去哪个专家，MoT 直接「看 token 是什么模态」就决定用哪套参数——路由是输入自带的标签，不是学出来的。

$$
\theta_t \;=\; \Theta[\,m_t\,], \qquad m_t \in \{\text{text}, \text{image}, \text{speech}\}
$$

**逐符号解释**：
- $m_t$：位置 $t$ 的 token 的模态标签（已知、确定）。
- $\Theta = \{\theta^{\text{text}}, \theta^{\text{image}}, \theta^{\text{speech}}\}$：每个模态一整套非嵌入参数（含 $W_Q^m,W_K^m,W_V^m,W_O^m$、FFN 权重、norm 参数）。
- $\theta_t$：位置 $t$ 实际使用的那套参数，由 $m_t$ 索引选出。

**由来 / 推导**：把 MoE 的软路由 $g(\cdot)=\mathrm{softmax}(W_g x)$ 退化为「以模态标签为 one-hot 的硬选择」。等价于 gate 权重被钉死为 $\mathbb{1}[m_t = m]$，因此无需学习、无负载不均、无辅助平衡损失。

**自检**：若只有一种模态（$|\{m\}|=1$），$\Theta$ 退化为单套参数，MoT 完全等价于 dense Transformer——符合直觉：单模态时 MoT 不应有额外结构。

### 公式 2：[[扩展自注意力|模态解耦投影 + 全局自注意力]]

**直觉先行**：每个 token 用「自己模态那套」投影出 Q/K/V，但所有 token 的 Q/K/V 丢进**同一个**注意力里互相看。

$$
Q_t = h_t W_Q^{m_t}, \quad K_t = h_t W_K^{m_t}, \quad V_t = h_t W_V^{m_t}
$$

$$
A = \mathrm{softmax}\!\left( \frac{Q K^{\top}}{\sqrt{d}} + M_{\text{causal}} \right) V, \qquad o_t = A_t\, W_O^{m_t}
$$

**逐符号解释**：
- $W_Q^{m_t}, W_K^{m_t}, W_V^{m_t}, W_O^{m_t}$：模态 $m_t$ 专属的查询/键/值/输出投影。
- $Q, K, V$：把全序列所有模态的投影结果按原位置拼成的矩阵。
- $M_{\text{causal}}$：因果掩码（下三角），保证自回归；注意它**不区分模态**——任意 token 可看见其前面所有模态的 token。
- $A_t$：位置 $t$ 的注意力输出（聚合了全序列信息）。
- $o_t$：经模态专属输出投影后的结果。

**由来 / 推导**：标准多头注意力，唯一改动是投影矩阵从全模态共享变成按 $m_t$ 索引。注意力打分 $QK^\top$ 这一步**不带任何模态参数**，所以天然全局——这是「跨模态融合」的数学来源。

**自检**：取消模态上标（所有 $W^{m}$ 相同），公式塌回普通因果自注意力；把 $M_{\text{causal}}$ 改成「只允许同模态互看」则退化为多个独立单模态模型——后者正是作者反对的、会破坏融合的极端。

### 公式 3：[[MLP|模态解耦前馈]]

**直觉先行**：注意力之后，每个 token 再走「自己模态那套」MLP 做特征变换。

$$
\mathrm{FFN}^{m_t}(x) = W_2^{m_t}\,\sigma\!\left(W_1^{m_t}\, \mathrm{Norm}^{m_t}(x)\right)
$$

**逐符号解释**：
- $\mathrm{Norm}^{m_t}$：模态专属 pre-FFN 归一化（[[RMSNorm]]）。
- $W_1^{m_t}, W_2^{m_t}$：模态专属升维/降维矩阵。
- $\sigma$：激活（如 [[SiLU]]）。

**由来 / 推导**：标准 Transformer FFN，把权重与归一化按模态复制。因 FFN 占总参数约 67%，此处是模态特化容量的主体。

**自检**：单模态时各 $W^m$ 合并，退回标准 FFN——一致。

### 公式 4：[[预训练|总训练目标]]（各模态目标按模态聚合）

**直觉先行**：训练损失就是各模态 token 在各自目标下损失的和——文本/离散图像走自回归交叉熵，连续图像（Transfusion 设定）走扩散/流匹配损失。

$$
\mathcal{L} = \sum_{m} \;\sum_{t:\, m_t = m} \mathcal{L}_{m}\big(\hat{y}_t,\, y_t\big)
$$

**逐符号解释**：
- $\mathcal{L}_m$：模态 $m$ 的损失——文本与离散图像用 [[Cross-Entropy]] 下一 token 预测；Transfusion 设定下图像用 [[Diffusion Model]]/[[Flow Matching]] 损失。
- 外层对模态求和、内层对该模态所有位置求和。

**由来 / 推导**：MoT 不改训练目标，只改参数结构；目标与对应的 dense 基线（Chameleon 全自回归、Transfusion 自回归+扩散混合）完全一致，因此对比是「同目标、同数据、同总参数，只换 dense↔MoT」的公平比较。

**自检**：MoT 与 dense 基线在相同总参数下对比 → 任何性能差异都归因于「模态解耦」这一结构改动，而非目标或数据差异。这正是论文实验设计的公平性所在。

---

## 5. 关键图表精读

### Figure 1：MoT 架构图（dense vs MoT 对比）

![MoT Architecture](https://github.com/facebookresearch/Mixture-of-Transformers/raw/main/assets/MoT.png)

**这张图在说什么**：左侧是 dense 基线——所有模态共用一套注意力投影、FFN、归一化；右侧是 MoT——同一层里，文本/图像/语音各有一套彩色区分的 FFN、Q/K/V/O 投影和 norm（模态解耦部分），中间唯一交汇处是那块**全局 self-attention**（所有模态 token 在此互相可见）。它一图说清全文：**「非嵌入参数按模态拆开 + 自注意力全局共享」**。看这张图时盯两点：彩色块（模态专属、并行）和中央灰块（跨模态共享）的分工。

### Table 1：Chameleon 设定（文本+图像，全自回归）主结果

| 设定 | 模型 | 达到 dense 同等质量所需 FLOPs | wall-clock |
|------|------|------|------|
| Chameleon 7B (text+image) | Dense baseline | 100% | 100% |
| Chameleon 7B (text+image) | **MoT** | **55.8%** | — |
| Chameleon 443M (image 质量) | **MoT** | — | **47%（图像）/ 75.6%（文本）** |

**怎么读**：最该看「55.8%」这一格——MoT 用约一半多一点的算力追平 dense 的文+图自回归质量。443M 的 wall-clock（在 AWS p4de.24xlarge / A100 上实测）说明**FLOPs 节省能真实兑现成挂钟时间**，不是纸面 FLOPs，这点很重要（很多稀疏方法 FLOPs 省了但实际跑不快）。

### Table 2：扩展到语音（Chameleon + Speech，443M）

| 模态 | 模型 | 达到 dense 同等质量所需 FLOPs |
|------|------|------|
| Speech | Dense baseline | 100% |
| Speech | **MoT** | **37.2%** |

**怎么读**：加入第三种模态（语音）后增益更大——只用 37.2% FLOPs 达到 dense 级语音质量。模态越多、彼此越「打架」，dense 的内耗越严重，MoT 的解耦收益越大。这是 MoT 可 scale 到更多模态的有力证据。

### Table 3：Transfusion 设定（图像走扩散/连续目标）

| 模型 | 训练/推理 FLOPs | COCO-30k CLIP↑ | COCO-30k FID↓ |
|------|------|------|------|
| Dense 1.4B (Transfusion) | 100% | 0.206 | 24.688 |
| **MoT 760M** | **~50%** | **0.214** | **21.145** |
| MoT 7B（匹配 dense 图像质量） | **~1/3** | — | 8.14 @ guidance 1.6 |

**怎么读**：Transfusion 是「文本自回归 + 图像扩散」混合目标的设定。看 760M 那行——MoT 用 **dense 1.4B 一半的 FLOPs**，CLIP（0.214 vs 0.206）和 FID（21.145 vs 24.688）**双双反超** dense。7B 上更狠，只需 **约 1/3 FLOPs** 匹配 dense 图像质量。这证明 MoT 不只对全离散自回归（Chameleon）有效，对「不同模态用不同训练目标」的混合范式同样有效——通用性强。

### Figure 2：Step-matching / 训练曲线（消融）

> 🖼️ **Figure 2: 训练步数匹配曲线** — 图片暂缺（原图见 [arXiv HTML](https://arxiv.org/html/2411.04996)）

**这张图在说什么**：横轴训练步、纵轴验证损失，对比 MoT 与 dense 的收敛速度。关键读数：**MoT 只用约 30% 的训练步数就达到 dense 同等的图像预训练损失**——即收敛快 ~3 倍。这把「FLOPs 省」翻译成「同样步数下损失更低 / 同样损失下步数更少」，是效率主张的直接证据。

---

## 6. 实验：它到底证明了什么

### 主结果
三套设定一致显示「同总参数下 MoT 远比 dense 省」：
- **Chameleon（文+图，全自回归）**：MoT 7B 用 **55.8% FLOPs** 追平 dense。
- **Chameleon+Speech（三模态）**：MoT 443M 用 **37.2% FLOPs** 达 dense 级语音；图像质量只花 **47% wall-clock**、文本 **75.6% wall-clock**。
- **Transfusion（文自回归+图扩散）**：MoT 7B 用 **~1/3 FLOPs** 匹配 dense 图像质量；MoT 760M 用 dense 1.4B 一半 FLOPs 在 CLIP/FID 上**反超**。

数字背后的含义：节省不是靠砍质量换来的（质量持平甚至更好），而是靠**消除模态间负迁移 + 零开销路由**。且在 A100 真机上节省能兑现为 wall-clock。

### 消融
- **哪个组件最关键**：FFN 解耦——FFN 占 ~67% 参数，是模态特化容量主体；只拆注意力不拆 FFN 增益会大幅缩水。
- **收敛速度（step-matching）**：MoT 约 **30% 训练步**即达 dense 同等图像损失，反向印证 §2「解耦消除梯度冲突 → 每模态学得更快」的核心假设。
- 模态越多增益越大（语音设定 37.2% < 文图设定 55.8%），支持「dense 的内耗随模态数增长」这一动机。

### 局限实验 / 反例
- **参数量随模态线性增长**：每加一种模态就多一整套非嵌入参数，**总参数（显存）**随模态数线性涨——虽然每 token 的激活 FLOPs 不变，但存储成本上升。模态特别多时这是瓶颈。
- **依赖已知、离散的模态边界**：路由靠「token 属于哪个模态」的硬标签。模态界限模糊、或想在更细粒度（如「同一模态内不同子分布」）做特化时，硬路由不适用——此时 MoE 的学习式路由反而有空间。
- 评估集中在生成质量与 FLOPs/wall-clock，对超大规模（远超 7B）与更多模态组合的 scaling law 外推仍是开放问题。

---

## 7. 复现思路（动手向）

- **训练流程 / 伪代码**（MoT 块前向，单层）：
  ```python
  # h: [B, T, D] 隐状态; modality_id: [B, T] 每个 token 的模态标签
  def mot_block(h, modality_id, params):  # params[m] = 该模态的一整套权重
      out = torch.zeros_like(h)
      # ---- 注意力：模态专属投影，全局打分 ----
      Q = K = V = torch.zeros_like(h)
      for m in MODALITIES:
          mask = (modality_id == m)                 # 确定性路由，无 router
          x = params[m].attn_norm(h[mask])
          Q[mask] = x @ params[m].Wq
          K[mask] = x @ params[m].Wk
          V[mask] = x @ params[m].Wv
      A = causal_softmax_attention(Q, K, V)         # 跨所有模态全局共享
      for m in MODALITIES:
          mask = (modality_id == m)
          out[mask] = h[mask] + A[mask] @ params[m].Wo
      # ---- FFN：模态专属 ----
      for m in MODALITIES:
          mask = (modality_id == m)
          y = params[m].ffn_norm(out[mask])
          out[mask] = out[mask] + params[m].W2 @ silu(params[m].W1 @ y)
      return out
  ```
  关键点：用 `modality_masks` 分组算、按原位置 merge 回去；归一化写进 attn/ffn 内部而非单独 block。
- **关键超参 / 设定**：与对应 dense 基线（Chameleon / Transfusion）保持同 batch、同 lr、同步数、同总参数，只替换块结构——保证「只测结构差异」的公平性。
- **数据与算力**：多模态混合语料（文本 + 离散图像 token + 语音 token）；系统 profiling 在 **AWS p4de.24xlarge（NVIDIA A100）** 上做，验证 FLOPs 节省兑现为 wall-clock。
- **容易踩的坑**：
  1. 路由是硬掩码，**别引入 router / auxiliary balance loss**——那会把 MoT 退化成 MoE 并带回不稳定。
  2. 显存：N 模态 ≈ N 套非嵌入参数，预算显存时按总参数（非激活 FLOPs）算。
  3. 注意力一定要在**投影之后、跨全模态**统一计算；若不小心按模态分块算注意力，会悄悄破坏跨模态融合（性能崩但不报错）。

---

## 8. 批判与延伸

- **真创新 vs 包装**：剥掉术语，真正新的只有一点——**用「模态标签」这个免费的确定性信号替代 MoE 的学习式 router**。结构上 MoT ≈「按模态分组的 dense 多专家 + 全局注意力」，不复杂；但这个简化恰恰消掉了 MoE 全部的工程痛点（路由不稳、负载不均、辅助损失），且实验扎实——属于「简单但对」的好工作。
- **站不住 / 需警惕的点**：
  - 「省 FLOPs」是在**固定总参数**下比的；MoT 总参数随模态线性增长，若按「同显存预算」比较，结论可能没那么夸张——读者要分清是 compute-matched 还是 param-matched。
  - 主实验规模到 7B，更大规模与更多模态的 scaling 行为靠外推，证据未覆盖。
- **可以接着做什么**：
  1. **层级混合路由**：浅层用模态硬路由（MoT）保特化，深层叠 MoE 软路由捕捉模态内子分布——硬+软互补。
  2. **模态间参数共享/低秩化**：用 [[LoRA]] 让各模态共享一个底座、只学模态专属低秩增量，缓解总参数线性膨胀。
  3. **细粒度路由**：把「模态」推广到「语义子域」（如图像里的自然图 vs 文档图），在不引入完整 router 的前提下做更细的硬路由。
  4. 把 MoT 思想用到 [[VLA]] / 世界模型的「视频专家 + 动作专家」耦合（已有后续工作沿此方向，见 [[MoT]] 概念页）。

---

## 9. 常见疑问 Q&A

> **Q：MoT 和 MoE 到底差在哪？不都是「多专家」吗？**
> A：差在**路由**。MoE 用学习出来的 gating router 软路由 token（需训练、易负载不均、要辅助损失），专家分工是涌现的；MoT 用 **token 自带的模态标签硬路由**（零开销、天然均衡、无辅助损失），专家就是「每模态一套」。一句话：MoE 学路由，MoT 看标签。

> **Q：每个模态一套参数，那总参数不是暴涨吗？算力怎么还省了？**
> A：分清两件事——**总参数（存储）** 确实随模态线性涨；但**每个 token 的激活 FLOPs** 不变（它只走自己那一套，不会走别的模态的）。省的是 compute：同样总参数下，dense 要让一套参数硬扛所有模态、内耗严重，MoT 让每模态各得其所，所以同质量下需要的训练/推理 FLOPs 更少。

> **Q：自注意力为什么不也按模态拆开？**
> A：因为跨模态融合就发生在注意力里。若注意力也按模态隔离，模型就变成几个并排的单模态模型，图文/语音之间无法对齐——这正是 §3.3「去掉会怎样」说的崩点。投影可以模态专属（怎么进注意力空间是模态相关的），但「谁关注谁」必须全局。

> **Q：路由靠模态标签，那推理时混合模态序列怎么知道每个 token 的模态？**
> A：模态边界在多模态序列里是已知的——文本段、图像 token 段、语音段由 tokenizer/数据格式天然标好。所以模态标签在训练和推理时都是确定可得的，不需要预测，这也是 MoT「零路由开销」的前提。

> **Q：Chameleon 设定和 Transfusion 设定有什么区别，为什么都要测？**
> A：Chameleon 把图像**离散**成 token、全模态统一做自回归交叉熵；Transfusion 让图像走**连续扩散/流匹配**目标、文本走自回归，是「不同模态不同目标」的混合范式。两个都测是为了证明 MoT 的模态解耦**与训练目标无关**——无论全离散还是混合目标都有效，通用性更强。

---

## 关联

- **基于**：[[Chameleon]] — 主基线与实验骨架（原生混合模态自回归）；[[Transfusion]] 设定 — 文本自回归+图像扩散的混合范式基线。
- **对比**：[[MoE]] — 同为稀疏化但走学习式软路由，MoT 用确定性硬路由替代之，是核心对照。
- **概念**：[[Mixture-of-Transformers Routing]]、[[Transformer]]、[[MLP]]、[[RMSNorm]]、[[扩展自注意力]]、[[Discrete Tokenization]]、[[Flow Matching]]

---

## 速查卡

> [!summary] Mixture-of-Transformers (MoT)
> - **核心**：Transformer 每个非嵌入层按模态解耦权重（FFN/QKVO/Norm 各一套），自注意力全局共享，确定性模态路由替代 MoE 软路由。
> - **方法**：ModalityUntiedFeedForward + ModalityUntiedAttention + 模态专属归一化；token 按模态标签硬路由，FFN（占 67% 参数）解耦是增益主体。
> - **结果**：文+图 55.8% FLOPs 追平 dense；+语音 37.2% FLOPs；Transfusion ~1/3 FLOPs；图像质量 47% wall-clock；30% 训练步达 dense 同等损失。
> - **一句话评价**：简单但扎实、可直接落地的多模态稀疏化范式——值得深挖，尤其想做统一多模态/VLA backbone 时是默认对照与起点。

---

*精读创建时间：2026-06-21*
