---
type: concept
aliases: [t-SNE, t-Distributed Stochastic Neighbor Embedding]
---

# t-SNE

## 定义

t-Distributed Stochastic Neighbor Embedding：一种**非线性降维**方法，把高维点对的概率邻接关系投影到 2D / 3D，使得高维相似的点在低维也相邻；常用于潜表示可视化。

## 数学形式

高维相似度（高斯）：

$$
p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2/2\sigma_i^2)}{\sum_{k\ne i}\exp(-\|x_i - x_k\|^2/2\sigma_i^2)}
$$

低维相似度（学生 t 分布，自由度 1）：

$$
q_{ij} = \frac{(1+\|y_i - y_j\|^2)^{-1}}{\sum_{k\ne l}(1+\|y_k - y_l\|^2)^{-1}}
$$

最小化 KL 散度 $\sum_{ij} p_{ij}\log(p_{ij}/q_{ij})$ 得到低维嵌入 $\{y_i\}$。

## 核心要点

1. **保留局部结构**: 主要保留近邻关系，全局距离不可解释
2. **非确定性**: 不同初始化 / random seed 结果不同
3. **超参敏感**: perplexity 选择影响很大（一般 5–50）
4. **常见误用**: t-SNE 图上"簇间距离"不能解释为真实距离
5. **在世界模型中**: [[LeWM]] 用 t-SNE 可视化 PushT 的潜空间，证明 2D 物理位置的拓扑结构被保留

## 相关概念

- [[CLS Token]]
- [[ViT]]
