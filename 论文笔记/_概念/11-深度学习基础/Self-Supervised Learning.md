---
type: concept
aliases: [SSL, Self-Supervised Learning, Self-Supervised Representation Learning, 自监督学习, 自监督表征学习]
---

# Self-Supervised Learning

## 定义

Self-Supervised Learning（自监督学习）指**不依赖人工标注**、通过构造数据自身的代理任务（pretext task）来学习通用表征的范式。在视觉领域典型 pretext 包括对比学习（[[CLIP]]、SimCLR）、掩码重建（MAE、BEiT）、Joint Embedding（[[JEPA]]、[[DINO]]、[[DINOv2]]）等。学到的表征可迁移到下游任务（分类、检测、世界模型）。

## 数学形式

通用形式：找一对相关样本 $(x, x^+)$ 与负样本 $\{x^-\}$，最大化正对相似度：

$$
\mathcal{L}_{\mathrm{SSL}} = -\log \frac{\exp(\mathrm{sim}(f(x), f(x^+))/\tau)}{\sum_{x' \in \{x^+\} \cup \{x^-\}} \exp(\mathrm{sim}(f(x), f(x'))/\tau)}
$$

或 JEPA 风格的非对比预测：

$$
\mathcal{L}_{\mathrm{JEPA}} = \big\| g_\psi(f_\theta(x_{\mathrm{ctx}})) - \mathrm{sg}(f_{\bar\theta}(x_{\mathrm{tgt}})) \big\|^2
$$

## 核心要点

1. **三条主线**：
   - 对比式：InfoNCE / SimCLR / MoCo
   - 掩码重建式：MAE / BEiT / SimMIM
   - Joint Embedding：[[JEPA]] / [[DINO]] / BYOL（无负样本但靠 stop-gradient 防坍塌）
2. **核心难点**：[[表征坍塌]] —— 模型可能学到常数表示满足 pretext 任务但毫无信息
3. **在世界模型中的角色**：[[Latent World Model]] 几乎都用 SSL 预训练 encoder（如 [[DINO-WM]] 用 [[DINOv2]]）
4. **优势**：可利用海量无标注数据（视频、网络图片），表征通用性强

## 代表工作

- [[DINO]] / [[DINOv2]]: 视觉 SSL 的代表，[[DINO-WM]] 直接复用
- [[JEPA]] / [[I-JEPA]] / [[V-JEPA]]: LeCun 力推的非对比 SSL
- [[CLIP]]: 文本-图像对比的多模态 SSL
- [[PLDM]] / LeWM: 把 SSL 直接嵌进 latent 世界模型训练
- [[StableWM]]: 为基于 SSL 的 WM 提供统一评测基础设施

## 相关概念

- [[JEPA]]
- [[DINO]]
- [[表征坍塌]]
- [[Latent World Model]]
- [[Continual Learning]]
