---
type: concept
aliases: [槽 Token 化, Object Slot Tokenization, 对象槽 token]
---

# Slot Tokenization

## 定义

对象槽 Token 化（Object-Slot Tokenization）是 OA-WAM 提出的感知表示设计，将每帧图像分解为 $N+1$ 个结构化向量（1 个机器人槽 + $N$ 个对象槽），每个槽由冻结身份地址和时变内容两部分拼接构成。

## 数学形式

$$
\mathbf{s}_k^t = [\text{addr}_k(32) \| \text{cnt}_k^t(256) \| \pi^t(16) \| \rho_k(16)] \in \mathbb{R}^{320}
$$

其中：
- $\text{addr}_k = f_{\text{addr}}([\ell_k \| f_k^{(0)}])$：由语言标签和首帧视觉特征计算，每轮任务冻结
- $\text{cnt}_k^t = f_{\text{cnt}}(\text{raw}_k^t)$：每帧从 SAM 3 + DINOv3 重新计算
- $\pi^t$：正弦帧索引编码
- $\rho_k$：可学习角色 embedding

## 核心要点

1. **身份-内容分离**: addr 维度携带稳定身份（由语言 + 首帧视觉决定），cnt 维度携带时变状态
2. **N+1 槽设计**: 机器人槽（1 个）不参与世界损失；对象槽（最多 15 个）用掩码 padding
3. **感知管线**: Qwen3-VL 名词提取 → SAM 3 分割 → DINOv3 特征 → 槽适配器投影
4. **与 Slot Attention 的区别**: 不使用迭代竞争分配，而是通过语言提示的确定性分配保证槽与对象的稳定对应

## 代表工作

- [[OA-WAM]]: 提出此设计，槽容量 $N_{\max} = 16$，感知延迟约 95 ms/帧

## 相关概念

- [[Object-Addressable Attention]]
- [[SAM 3]]
- [[DINOv3]]
- [[VLA]]
