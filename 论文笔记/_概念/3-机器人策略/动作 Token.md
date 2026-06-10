---
type: concept
aliases: [Action Token, 动作离散Token, 动作词元]
---

# 动作 Token

## 定义

将连续机器人动作（关节角度、末端执行器位姿等）转换为离散符号（Token）的表示形式，使动作可以与语言/视觉 Token 在同一自回归框架中建模。

## 核心要点

1. **离散化方法**: 常见方案包括均匀量化（bin）、VQ-VAE 码本映射、DCT+BPE 频域量化等。
2. **共享词表**: 优质动作 Token 化使动作可与视觉/语言 Token 共享同一词表，实现统一 Next-Token Prediction。
3. **压缩权衡**: Token 数量越少推理越快，但精度损失越大；高精度操作（插孔）对 Token 粒度敏感。
4. **主流方案对比**:
   - 均匀 bin（OpenVLA）: 简单但忽略时序相关性
   - VQ-VAE 潜在动作（UniVLA-RSS2025）: 从视频学潜在动作码本
   - DCT+BPE（FAST、UniVLA-ICLR2026）: 频域量化，时序结构保留好

## 代表工作

- [[UniVLA-ICLR2026]]（BAAI, 2025）: DCT+BPE 动作 Token 化，集成入统一 VLA 框架
- [[OpenVLA]]: 均匀 bin 量化动作 Token
- [[FAST]]: 频域动作序列 Token 化

## 相关概念

- [[DCT 动作 Token 化]]
- [[VQ-VAE]]
- [[Action Chunking]]
- [[视觉 Token]]
- [[自回归 Transformer]]
