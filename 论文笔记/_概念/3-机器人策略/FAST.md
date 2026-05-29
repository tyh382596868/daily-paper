---
type: concept
aliases: [FAST tokenizer, Frequency-domain Action Sequence Tokenizer]
---

# FAST

## 定义

FAST 是一种**频域动作 tokenizer**：对动作 chunk 做 DCT 等频域变换、按频率截断保留低频系数，再离散量化得到一组 token，喂给自回归 [[VLA]]。

## 核心要点

1. **频域稀疏性**：人形机器人动作在低频处能量集中，DCT 系数极少；
2. **token 数显著降低**：相比 per-step token，FAST 大幅压缩序列长度；
3. **仍是离散表示**：频谱系数被量化成 token；
4. **与 [[NIAF]] 对比**：FAST 在频域离散化，NIAF 在频域**连续调制** [[SIREN]]。

## 代表工作

- **π₀-FAST** (Physical Intelligence, 2024)：首次大规模将 FAST 用于 VLA
- [[NIAF]]：用连续函数路线超过 FAST

## 相关概念

- [[Action Chunking]]
- [[BEAST]]
- [[Neural Implicit Action Field]]
- [[OpenVLA]]
