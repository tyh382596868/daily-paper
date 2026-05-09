---
type: concept
aliases: [VLA JEPA, VLA-JEPA model]
---

# VLA-JEPA

## 定义

VLA-JEPA 是将 JEPA（Joint Embedding Predictive Architecture）思想引入视觉-语言-动作模型的方法，通过在潜在空间预测未来表示（而非像素重建）来学习对机器人操作有用的世界模型表示。

## 核心要点

1. **潜在空间预测**: 预测目标在特征空间的表示，避免像素级细节的过拟合
2. **JEPA 范式**: 遮蔽部分观测，在潜在空间预测被遮蔽区域，学习结构化表示
3. **对象可寻址性局限**: swap-binding 余弦仅 0.07，表明潜在空间的对象身份绑定仍不稳定

## 代表工作

- [[OA-WAM]]: 将 VLA-JEPA 作为 WAM 类方法的对比基线

## 相关概念

- [[World Action Model]]
- [[VLA]]
- [[World Model]]
