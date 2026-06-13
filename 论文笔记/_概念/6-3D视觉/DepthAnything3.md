---
type: concept
aliases: [Depth Anything V3]
---

# DepthAnything3

## 定义
Depth Anything 系列的第三代单目深度估计模型，在超大规模无标注数据上预训练，支持任意分辨率输入，depth 估计精度和泛化能力大幅提升。

## 核心要点
1. 继承 Depth Anything V2 的 teacher-student 训练框架
2. 扩大训练数据规模，引入更多野外视频数据
3. 常用于视频世界模型的深度辅助（如 LSM-WM）

## 代表工作
- [[LSM-WM]] — 使用 DepthAnything3 做 latent 空间深度引导

## 相关概念
- [[深度估计]]
- [[时序深度预测]]
