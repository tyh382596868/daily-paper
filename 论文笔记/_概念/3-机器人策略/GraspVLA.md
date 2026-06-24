---
type: concept
aliases: [GraspVLA]
---

# GraspVLA

## 定义
专注于抓取任务的 VLA 模型，将 grasp-specific 几何先验融入视觉-语言-动作框架，在多物体抓取和新物体泛化上取得改进。

## 核心要点
1. 抓取几何先验：在 VLA token 序列中显式编码抓取位置和姿态的先验信息
2. 多物体场景：对目标物体进行精细定位，减少对语言描述的歧义
3. 在 LIBERO 和真实机器人上评测

## 代表工作
- [[GraspVLA]]: 原始论文
- [[UniFS]]: 将 GraspVLA 作为快慢系统 VLA 的对比方法

## 相关概念
- [[VLA]] — 上层概念
- [[OpenVLA]] — 同类基础模型
- [[Affordance]] — 相关概念
