---
type: concept
aliases: [VBench, VBench-I2V]
---

# VBench

## 定义

一套用于视频生成评测的多维基准协议，沿 8 个维度（subject consistency、background consistency、temporal flickering、motion smoothness、aesthetic quality、imaging quality、dynamic degree、overall consistency）分别打分并加权汇总为总分。VBench-I2V 是其图像到视频子集。

## 核心要点

1. 多维打分而非单一 FVD，能区分"画质好但动态差"等不同失败模式
2. 标准协议，便于跨论文比较
3. 主要面向短视频（5 秒），分钟级评测需配合时间稳定性指标（如 ΔIQ、PSNR 漂移）
4. 通常配合 FVD、相机精度（RotErr/TransErr/CamMC）一起报告

## 代表工作

- [[SANA-WM]]: 用 VBench Overall 衡量主基准结果，加精修器后 80.62（Simple）/ 81.89（Hard）
- 多数视频扩散模型评测的事实标准

## 相关概念

- [[相机投影]]
- [[10-数据集]]
