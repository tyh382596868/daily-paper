---
type: concept
aliases: [Hierarchical Data Format 5, h5]
---

# HDF5

## 定义

HDF5（Hierarchical Data Format version 5）是一种二进制层次化数据存储格式，常用于科学计算和机器学习数据集存储。它支持任意维度数组、嵌套 group、元数据 attribute、压缩与分块。

## 核心要点

1. **层次结构**：类似文件系统，group / dataset 两级抽象
2. **chunked + 压缩**：大数据集支持分块读取与 gzip/lz4 压缩
3. **跨语言**：Python (`h5py`)、C、MATLAB、Julia 通用
4. **机器人数据集事实标准**：[[DROID 数据集]]、Robomimic、SWM 默认存储格式

## 在 SWM 中的位置

[[StableWM]] 的 `world.record_dataset()` 默认以 HDF5 格式保存采集到的轨迹：
- key：`pixels`、`actions`、`states`、`rewards`
- 每个 episode 一个 group
- 同时支持 image folder / mp4 格式作为可选输出

## 代表工作

- [[StableWM]]: 默认数据采集格式
- 多数 robotics 数据集（[[BridgeV2]]、Robomimic、DROID）使用 HDF5

## 相关概念

- [[Push-T]]
- [[OGBench]]
- [[World 抽象]]
