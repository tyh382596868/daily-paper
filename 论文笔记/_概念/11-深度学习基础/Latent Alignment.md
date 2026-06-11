---
type: concept
aliases: [潜空间对齐, 隐式对齐, Patch-Level Latent Alignment]
---

# Latent Alignment

## 定义
让两个模型（或同一模型的两条路径）在某个**潜在表征空间**上的特征相互逼近的技术。区别于输出层对齐：它直接操作 hidden state / patch token / CLS 等中间表征。

## 数学形式

通用形式：
$$
\mathcal{L}_{align} = d(\,\phi_A(\mathbf{x}),\; \phi_B(\mathbf{x})\,)
$$
其中 $d$ 可以是 [[Cosine Similarity|余弦距离]]、[[L2 损失|L2]]、KL、Wasserstein 等。

Patch-level 对齐（视觉常用）：
$$
\mathcal{L}_{patch} = \frac{1}{N}\sum_{i=1}^{N}\bigl(1 - \mathcal{S}(\phi_A^{(i)}, \phi_B^{(i)})\bigr)
$$

## 核心要点
1. **粒度可选**: 全图 / patch / token / 局部窗口
2. **常用 adapter 桥接**: 若 $\phi_A$ 与 $\phi_B$ 维度不同或来自不同模态，需要一个 projector
3. **可冻结一端**: 通常 teacher 端冻结（VGGT、DINOv2）做监督
4. **与 explicit modality 注入对比**: latent alignment 把外部知识"压"进表征，无需推理时再调用 teacher

## 代表工作
- [[3DThinkVLA]]: patch-level 对齐 ViT 中间层与 [[VGGT]]
- [[DINOv2]]: 自蒸馏 latent alignment
- LiT: 视觉 encoder 与冻结 LLM 文本 encoder 对齐
- [[LARA]]: LAM 潜在动作表示与扩散 VLA 动作隐空间的双向对齐，实现联合正则化

## 相关概念
- [[Latent Distillation]]
- [[Knowledge Distillation]]
- [[Cosine Similarity]]
