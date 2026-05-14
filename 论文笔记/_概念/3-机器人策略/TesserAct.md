---
type: concept
aliases: []
---

# TesserAct

## 定义

Cascaded WAM 方法，把 4D（RGB + 深度 + normal + 动态）作为中间表征，提供比纯 RGB 更丰富的物理一致性监督。

## 核心要点

1. **4D 表征**: 不只生成像素，还生成深度 / normal / 运动，引入显式 3D 物理约束。
2. 属于 [[Cascaded WAM|Cascaded WAM]]；该方法对应的合成数据集也用于训练其它 WAM。

## 代表工作

- [[WAM-Survey]] 综述列为 Cascaded 4D-WAM 代表。

## 相关概念

- [[UniPi]]
- [[World Action Model]]
- [[视频扩散模型]]
