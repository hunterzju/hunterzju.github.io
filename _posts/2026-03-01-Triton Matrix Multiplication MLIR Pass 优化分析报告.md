---
title: Triton Matrix Multiplication MLIR Pass
categories:
  - Compiller
  - DeepLearning
tags:
  - Triton
share: true
zhihu-title: 2026-03-01-Triton Matrix Multiplication MLIR Pass 优化分析报告
zhihu-topics: triton
zhihu-link: https://zhuanlan.zhihu.com/p/2036945602043459251
zhihu-created-at: 2026-05-10 23:09
---

## 概述

本文档分析了 Triton实例中`03_matmul` 日志文件中各个 MLIR Pass 对 Triton 矩阵乘法 kernel 的优化作用。

## Pass 优化作用分析

### 1. 前期优化 Pass

| Pass 名称 | 优化作用 |
|-----------|----------|
| **Inliner (inline)** | 函数内联，将 `cdiv`、`zeros` 等辅助函数内联到主函数，消除函数调用开销 |
| **Canonicalizer (canonicalize)** | 规范化 IR，简化常量表达式、死代码消除 |
| **CSE (cse)** | 公共子表达式消除，避免重复计算相同的表达式 |
| **SymbolDCE (symbol-dce)** | 符号死代码消除，移除未使用的函数和符号 |
| **TritonCombineOps (triton-combine)** | 组合多个 Triton 操作，减少操作数量 |
| **TritonReorderBroadcast** | 重排序广播操作，优化内存访问模式 |

### 2. 核心循环优化 Pass

| Pass 名称 | 优化作用 |
|-----------|----------|
| **TritonLoopUnroll (triton-loop-unroll)** | **重要**: 展开矩阵乘法循环，减少循环开销，增加指令级并行 |
| **TritonLoopAwareCSE** | 循环感知的公共子表达式消除 |
| **TritonLoopInvariantCodeMotion (triton-licm)** | 循环不变代码外提，将不变计算移到循环外 |
| **TritonGPUFuseNestedLoops** | 融合嵌套循环，减少循环嵌套层数 |

### 3. GPU 特定优化 Pass

| Pass 名称 | 优化作用 |
|-----------|----------|
| **ConvertTritonToTritonGPU** | 将 Triton IR 转换为 TritonGPU IR，添加 GPU 特定属性（block layout, warps等） |
| **TritonGPUCoalesce (tritongpu-coalesce)** | **重要**: 合并内存访问，提高全局内存访问效率 |
| **TritonGPUF32DotTC (tritongpu-F32DotTC)** | 启用 Tensor Core (FP32 -> TF32) 加速矩阵乘法 |
| **TritonGPUPlanCTAPass** | 计算 CTA (Cooperative Thread Array) 块划分策略 |
| **TritonGPUAccelerateMatmul** | **关键**: 应用矩阵乘法特定优化，包括 swizzle、mma 指令选择 |
| **TritonGPURemoveLayoutConversions** | 移除不必要的布局转换，减少 memory format 转换开销 |
| **TritonGPUOptimizeDotOperands** | 优化点积操作数的布局和精度 |
| **TritonGPUOptimizeThreadLocality** | 优化线程局部性，提高共享内存利用率 |
| **TritonGPUPrefetch** | 数据预取优化，隐藏内存访问延迟 |
| **TritonGPUCoalesceAsyncCopy** | 异步拷贝的合并优化 |
| **TritonGPUReduceDataDuplication** | 减少数据重复，优化寄存器使用 |
| **TritonGPUReorderInstructions** | 指令重排，提高指令级并行度 |

### 4. NVIDIA GPU 特定优化 Pass

| Pass 名称 | 优化作用 |
|-----------|----------|
| **TritonNvidiaGPUOptimizeDescriptorEncodingPass** | 优化描述符编码 |
| **TritonNvidiaGPUOptimizeTMemLayoutsPass** | Tensor Memory 布局优化 |
| **TritonNvidiaGPUInterleaveTMemPass** | Tensor Memory 交错优化 |
| **NVGPUWarpSpecialization** | Warp 特殊化，优化 warp 调度 |
| **TritonGPUFenceInsertion** | 插入内存屏障，确保内存操作顺序 |
| **TritonNvidiaGPUMMALoweringPass** | MMA (Matrix Multiply Accumulate) 指令 lowering |
| **TritonGPUAssignLatencies** | 分配操作延迟，用于指令调度 |
| **TritonGPUScheduleLoops** | 循环调度 |
| **TritonGPUPipeline** | 流水线化操作 |

### 5. 代码生成阶段 Pass

| Pass 名称 | 优化作用 |
|-----------|----------|
| **SCFToControlFlowPass** | 将 SCF (Structured Control Flow) 转换为标准控制流 |
| **GluonInline** | Gluon 内联 |
| **AllocateSharedMemoryNv** | 分配共享内存 |
| **ConvertTritonGPUToLLVM** | 将 TritonGPU IR 转换为 LLVM IR |
| **ConvertNVGPUToLLVM** | 将 NVGPU 操作转换为 LLVM |
| **ConvertNVVMToLLVMPass** | NVVM 到 LLVM 的转换 |

## 关键优化 Pass 详解

### TritonGPUAccelerateMatmul

这是最关键的优化 Pass，主要作用：
- 将矩阵乘法映射到 GPU 的 Tensor Core 指令 (MMA)
- 选择最优的 block layout 和 warp 划分
- 应用 swizzle 技术优化共享内存访问
- 配置 `inputPrecision = tf32` 使用 Tensor Core 加速

### TritonLoopUnroll

循环展开优化：
- 将 `scf.for` 循环展开
- 减少循环控制开销
- 增加指令级并行度 (ILP)
- 使编译器能够更好地进行寄存器分配和指令调度

### TritonGPUCoalesce

内存访问合并优化：
- 合并非连续的全局内存访问为合并访问
- 提高内存带宽利用率
- 减少内存访问延迟

### TritonGPURemoveLayoutConversions

消除不必要的布局转换：
- 减少 `convert_layout` 操作
- 降低寄存器压力
- 减少内存访问开销

## 优化效果总结

Triton 编译器通过多层优化实现高效的矩阵乘法 kernel：

1. **高层优化**: 函数内联、常量折叠、CSE、死代码消除
2. **循环优化**: 循环展开、循环融合、不变代码外提
3. **GPU 适配**: Layout 优化、Tensor Core 加速、内存访问合并
4. **指令级优化**: 指令重排、延迟隐藏、流水线化
5. **代码生成**: LLVM 后端优化、NVVM 转换

这些优化 Pass 共同作用，将高层的 Triton 矩阵乘法代码转换为高效的 CUDA 代码，充分利用 GPU 的 Tensor Core、共享内存和并行计算能力。
