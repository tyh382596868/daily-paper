---
type: concept
aliases: [WorldVLA]
---

# WorldVLA

## 定义

WorldVLA 是一种整体式 World Action Model，将视觉语言动作模型（VLA）与世界模型联合训练，通过预测全局像素级未来状态来辅助动作生成。

## 核心要点

1. **整体式世界预测**: 预测下一帧全局图像，而非逐对象状态
2. **全局 token 表示**: 场景状态编码在统一的全局 token 中，对象身份与外观纠缠
3. **局限性**: 全局 token 在场景扰动（视角/布局/光照变化）下容易漂移，导致对象身份识别失误

## 性能参考

在 LIBERO-Plus 鲁棒性测试中（零样本）：
- Camera 扰动轴：0.1%（几乎完全失败）
- Overall Avg：25.0%
- Swap-Binding cosine：0.09（对象绑定极弱）

## 代表工作

- [[OA-WAM]]: 通过对象可寻址性解决 WorldVLA 类方法的身份漂移问题

## 相关概念

- [[World Action Model]]
- [[VLA]]
