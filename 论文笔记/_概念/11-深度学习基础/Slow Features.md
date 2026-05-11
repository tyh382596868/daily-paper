---
type: concept
aliases: [Slow Feature Analysis, SFA, 慢特征]
---

# Slow Features

## 定义

由 Wiskott & Sejnowski (2002) 提出的 Slow Feature Analysis (SFA) 给出的概念：从快速变化的输入信号中提取**变化最慢的不变特征**。在自监督表示学习中，"慢特征"常作为对低频结构的简称——表示训练初期模型最先捕获的那些跨时间稳定的全局信息（如背景、布局）。

## 数学形式

SFA 求解（在线性 / 核化形式下）：

$$
\min_{f} \mathbb{E}\big[(\dot f(\mathbf{x}_t))^2\big] \quad \text{s.t.} \quad \mathbb{E}[f(\mathbf{x})] = 0,\; \mathbb{E}[f(\mathbf{x})^2] = 1
$$

## 核心要点

1. **训练动力学**：自监督模型常先收敛到慢特征（背景、布局），后逐步刻画细节
2. **与世界模型的关系**：latent decoder 在训练早期解码出"模糊但全局正确"的图像，体现慢特征优先
3. **理论联系**：SFA 与 [[JEPA]]、对比学习的目标在线性极限下有等价形式

## 代表工作

- Wiskott & Sejnowski, 2002: SFA 原始论文
- [[LeWM]]: 解码可视化中观察到训练早期 decoder 输出对应慢特征

## 相关概念

- [[JEPA]]
- [[表征坍塌]]
