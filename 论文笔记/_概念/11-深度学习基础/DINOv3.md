---
type: concept
aliases: [DINOv3, DINO v3, DINOv2 successor]
---

# DINOv3

## 定义

DINOv3 是自监督视觉特征提取模型，继承 DINOv2 的自监督训练范式（知识蒸馏 + 对比学习），提供高质量的密集视觉特征，适用于物体识别、分割和机器人感知。

## 核心要点

1. **自监督训练**: 无需标签，通过 teacher-student 蒸馏在大规模图像数据上训练
2. **密集特征**: 输出每个 patch 的特征向量，适合下游分割和匹配任务
3. **高泛化性**: 在未见物体类别上保持稳定的特征空间，适合机器人操作中的新物体识别
4. **与 SAM 3 结合**: 在 OA-WAM 中，DINOv3 对 SAM 3 分割出的每个对象掩码区域提取 256 维内容特征

## 代表工作

- [[OA-WAM]]: 用 DINOv3 提取每个分割对象的 content 向量 $\text{cnt}_k^t \in \mathbb{R}^{256}$

## 相关概念

- [[SAM 3]]
- [[Slot Tokenization]]
- [[VLA]]
