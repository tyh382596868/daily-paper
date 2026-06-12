---
type: concept
aliases: [RoboGenesis Engine]
---

# RoboGenesis

## 定义
LabVLA 提出的三阶段仿真数据合成引擎，自动化生成实验室场景数据：文本驱动资产重建 → Agentic 流程生成 → 成功过滤导出，支持 16 种机器人平台，是现有机器人仿真引擎中功能最全的系统。

## 核心要点
1. **三阶段管道**: (1) 环境构建（text→3D 资产 LabAssetLibrary + 场景 LabSceneLibrary）；(2) Agentic 流程生成（原子技能组合 + 6 轴域随机化）；(3) LabEmbodied-Data 导出（成功过滤 + 15 类标注）
2. **规模**: 2,947 标注 3D 资产，10,000 程序化场景，16 机器人平台
3. **独特优势**: 自动资产生成 + 长时域任务 + 实验室协议（其他引擎均不完整支持）
4. **TRELLIS 2.0**: 使用该三维重建模型从参考图像生成 3D 资产

## 代表工作
- [[LabVLA]]: arXiv 2606.13578

## 相关概念
- [[LabVLA]]
- [[ManiSkill]]
- [[RoboTwin2]]
- [[Physics Simulator]]
- [[Sim-to-Real]]
