---
type: concept
aliases: [KI, 知识隔离, Knowledge Insulation]
---

# Knowledge Insulation（知识隔离）

## 定义

Knowledge Insulation（KI）是 [[π₀.₅]] 提出的一种梯度截断技术：在 VLA 训练时，阻止从[[Action Expert|行动专家]]到 [[VLM]] 骨干的梯度反向传播，从而保护 VLM 中的语言表示不被动作学习信号污染。

## 核心要点

1. **动机**: VLA 行动专家的梯度若流回 VLM，会因数据不平衡（视觉-动作多样性远大于语言多样性）导致 VLM 语言能力退化
2. **实现**: 在行动专家与 VLM 接口处使用 `stop_gradient`（detach），切断反向传播路径
3. **局限**: 单独使用 KI 仅能保护分布内（in-distribution）语言能力，对 OOD 语言指令泛化收益有限（APT Table 2 显示单独 KI 无法显著改善 UO/UOUE 场景）
4. **与两阶段训练的协同**: [[APT]] 在 KI 基础上增加两阶段预训练，三者（KI + 2-Stage + Ft VLM）结合时在所有 OOD 维度取得最佳效果

## 代表工作

- [[π₀.₅]]: 首次提出 Knowledge Insulation，在 π₀ 框架上停止行动专家→VLM 梯度
- [[APT]]: 在 KI 基础上加入[[贝叶斯分解]]和两阶段训练，进一步突破 OOD 泛化瓶颈

## 相关概念

- [[π₀.₅]]
- [[Action Expert]]
- [[贝叶斯分解]]
- [[Action Shortcut]]
