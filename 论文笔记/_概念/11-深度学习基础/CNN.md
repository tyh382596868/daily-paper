---
type: concept
aliases: [CNN, Convolutional Neural Network, 卷积神经网络]
---

# CNN

## 定义

**Convolutional Neural Network (CNN)** 是用卷积层 + 池化层 + 非线性激活组合而成的神经网络架构，利用局部感受野和参数共享在图像 / 网格类输入上取得优势。

## 关键性质

- **平移等变**: 卷积核作用于滑动窗口，使输入平移导致输出对应平移。
- **参数共享**: 同一卷积核在空间上复用，参数量远小于全连接。
- **层级特征**: 浅层学边缘/纹理，深层学物体/语义。

## 在机器人感知中的常见用法

- 深度图 / 灰度图编码（[[MAD]] 用轻量 CNN 把 $18 \times 32$ 深度图编码到 latent）
- 占用栅格图、体素网格的 2D/3D 卷积处理
- 通常和 [[Transformer]] 或 [[MLP]] 串联

## 代表架构

- [[ResNet]]
- [[VGG]]
- [[UNet]]
- [[ViT]]（虽然名义不是 CNN，但常和 CNN patch embedding 混用）

## 关联概念

- [[Transformer]]
- [[MLP]]
- [[ResNet]]
