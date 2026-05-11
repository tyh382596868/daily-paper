---
type: concept
aliases: [ViT, Vision Transformer]
---

# ViT

## 定义

Vision Transformer 把图像切成不重叠的 patch（如 $16\times 16$ 或 $14\times 14$），每个 patch 线性映射为 token，加位置编码后送入标准 [[Transformer]] encoder，证明纯注意力架构在视觉任务上可以匹敌甚至超过 CNN。

## 数学形式

$$
z_0 = [x_{\text{cls}};\; x_p^1 E;\; \dots;\; x_p^N E] + E_{\text{pos}}
$$

其中 $x_p^i \in \mathbb{R}^{P^2 \cdot C}$ 是 flatten 后的 patch，$E$ 是 patch embedding 矩阵，$x_{\text{cls}}$ 是可学习的 [[CLS Token]]。

## 核心要点

1. **Patch tokenization**: 把图像视作 token 序列，长度 $N=HW/P^2$
2. **CLS token**: 末层 [[CLS Token]] 的输出常作为图像级表示
3. **数据需求大**: ViT 在 JFT-300M 这类大数据集上才完全显出优势
4. **变体**: ViT-tiny / small / base / large，区别在层数、heads、hidden dim
5. **在世界模型中**: [[LeWM]] 用 ViT-tiny + [[CLS Token]] 编码帧，token 数比 patch tokens 少 ~200×

## 代表工作

- Dosovitskiy et al., 2020: An Image is Worth 16x16 Words (原始 ViT)
- [[DINO]] / [[DINOv3]]: 自监督训练的 ViT
- [[LeWM]]: ViT-tiny encoder + SIGReg

## 相关概念

- [[Transformer]]
- [[CLS Token]]
- [[DINO]]
- [[自注意力]]
