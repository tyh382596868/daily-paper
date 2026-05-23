---
type: concept
aliases: [SOMA 框架, SOMA Pipeline, Spatial Memory Construction-Refinement-Retrieval]
---

# SOMA 三阶段框架

## 定义

[[SOMA]] 论文提出的空间记忆处理流水线，由三个阶段组成：**Spatial Memory Construction → Dynamic Memory Refinement → Contextual Memory Retrieval**。

## 核心要点

1. **Construction（构建）**: 头部相机扫描序列 → [[VGGT]] 估位姿 + [[YOLO-World]] 检框 + [[DINOv3]] 提特征 → 跨视角实例关联 → 全局物体级 token $M_0$
2. **Refinement（刷新）**: 操作中持续观测，用 [[相似度-融合双权重]] 做 [[EMA]] 软更新，得到 $\tilde{M}^t$
3. **Retrieval（检索）**: [[VLM]] token 作 query，记忆 token 作 key/value，[[Cross-Attention]] 输出 boost 注入 [[DiT]]
4. 三个阶段缺一不可，消融显示 dynamic update 最关键（-7.8 pp）

## 数学形式

构建：

$$
m_k^0 = \Phi_{\text{mem}}(f_k) + p_k
$$

刷新：

$$
m_k^t = \alpha_{kj}^t m_j^t + (1 - \alpha_{kj}^t) m_k^{t-1}
$$

检索：

$$
X_{\text{boost}} = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{C}}\right) V
$$

## 代表工作

- [[SOMA]]: 提出三阶段框架

## 相关概念

- [[VGGT]]
- [[DINOv3]]
- [[YOLO-World]]
- [[Cross-Attention]]
- [[EMA]]
- [[相似度-融合双权重]]
- [[空间记忆专家]]
