---
type: concept
aliases: [可寻址对象注意力, OA Attention, 对象可寻址性]
---

# Object-Addressable Attention

## 定义

对象可寻址注意力（Object-Addressable Attention）是 OA-WAM 提出的注意力机制变体，通过限制 cross-slot attention 的 key 仅读取槽向量的"地址"子空间（冻结身份标识符），使注意力路由由稳定的对象身份决定，与时变内容解耦。

## 数学形式

在槽 token 位置，第 $l$ 层注意力 key 修改为：

$$
\mathbf{K}_k^{(l)} = W_K^{(l)} \cdot \text{mask}_{\leq 32}(\mathbf{x}_k^{(l)})
$$

配合逐层地址重置钩子：

$$
\mathbf{x}_k^{(l+1)}[1:32] \leftarrow \text{addr}_k
$$

Query 和 Value 仍读取完整 token 隐状态。

## 核心要点

1. **Key-only 约束**: 仅限制 key 投影的输入维度，不改变 query 和 value，保留完整的信息流动能力
2. **地址-内容分离**: addr 子空间（前 32 维）只包含冻结的对象身份；content 子空间（后 288 维）携带时变状态
3. **逐层 Reset 钩子**: 每个 transformer block 后强制恢复 addr 维度，防止残差流污染身份子空间
4. **因果可验证**: Swap-Binding 测试可量化验证对象可寻址性（OA-WAM 获得 0.87 vs 基线 ≤0.09）

## 代表工作

- [[OA-WAM]]: 提出此机制，LIBERO-Plus 几何轴 +4.8%，swap-binding 0.87

## 相关概念

- [[Slot Tokenization]]
- [[VLA]]
- [[World Action Model]]
