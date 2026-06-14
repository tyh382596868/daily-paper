---
type: concept
aliases: [Contactile 触觉传感器, 阵列型触觉传感器]
---

# Contactile

## 定义
Contactile 是一种阵列型触觉传感器，通过压力阵列矩阵捕捉接触分布信息，输出多维触觉信号矩阵。

## 核心要点
1. 输出为二维压力阵列（array-type），适合 CNN 处理
2. 多个功能区域可以共享相同的阵列形状，适合跨区域共享编码器
3. 在 [[FTP-1]] 中被用于 TactileUMI 平台（未见传感器实验）

## 代表工作
- [[FTP-1]]: 在 TactileUMI 配置中使用 Contactile，属于阵列型传感器，用 CNN 编码至 [[MTTS]] 令牌

## 相关概念
- [[触觉传感器]]
- [[MTTS]]
- [[GelSight]]
