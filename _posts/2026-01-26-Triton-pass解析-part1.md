---
title: Triton-pass解析-part1
categories:
  - Compiller
  - DeepLearning
tags:
  - Triton
share: true
---

前文以`vector_add`用例为例，分析了Triton在编译优化过程中，分别在MLIR层和LLVM层经历了哪些优化pass，以及各个优化pass中的简单作用；本文会针对具体pass产生的效果做直观分析，可能会分成几篇展开。

## MLIR优化pass

前文中列举的MLIR优化pass简单可以分为3个阶段：
```
1.TritonIR相关的通用优化：从`Inline`到`TritonLoopUnroll`
2.GPU相关优化：从`ConvertTritonToTritonGPU`到`TritonGPUProxyFenceInsertion`
3.降低到LLVM的Pass：从`ConvertTritonGPUToLLVM`到`LLVMDIScope`
```

写脚本统计了pass前后IR是否会发生变化，结果如下：
```json
{
  "Inliner": false,
  "Canonicalizer": false,
  "TritonRewriteTensorPointer": false,
  "TritonRewriteTensorDescriptorToPointer": false,
  "TritonCombineOps": false,
  "TritonReorderBroadcast": true,
  "CSE": false,
  "SymbolDCE": false,
  "TritonLoopUnroll": false,
  "ConvertTritonToTritonGPU": true,
  "TritonGPUCoalesce": true,
  "TritonGPUF32DotTC": false,
  "TritonGPUPlanCTAPass": false,
  "TritonGPURemoveLayoutConversions": false,
  "TritonGPUOptimizeThreadLocality": false,
  "TritonGPUAccelerateMatmul": false,
  "TritonGPUOptimizeDotOperands": false,
  "TritonNvidiaGPUOptimizeDescriptorEncodingPass": false,
  "TritonLoopAwareCSE": false,
  "TritonGPUFuseNestedLoops": false,
  "TritonLoopInvariantCodeMotion": false,
  "TritonGPUCombineTensorSelectAndIf": false,
  "NVGPUWarpSpecialization": false,
  "TritonGPUAssignLatencies": false,
  "TritonGPUScheduleLoops": true,
  "TritonGPUPipeline": true,
  "TritonGPUPrefetch": false,
  "TritonGPUCoalesceAsyncCopy": false,
  "TritonNvidiaGPUOptimizeTMemLayoutsPass": false,
  "TritonNvidiaGPUInterleaveTMemPass": false,
  "TritonGPUReduceDataDuplication": false,
  "TritonGPUReorderInstructions": false,
  "TritonGPUFenceInsertion": false,
  "TritonNvidiaGPUMMALoweringPass": false,
  "SCCP": true,
  "TritonGPUAllocateWarpGroups": true,
  "SCFToControlFlowPass": false,
  "GluonInline": false,
  "AllocateSharedMemoryNv": true,
  "TritonTensorMemoryAllocationPass": true,
  "TritonNvidiaGPUCheckMatmulTwoCTAPass": true,
  "TritonGPUGlobalScratchAllocationPass": true,
  "TritonGPUProxyFenceInsertion": false,
  "ConvertTritonGPUToLLVM": true,
  "ConvertNVGPUToLLVM": false,
  "ConvertWarpSpecializeToLLVM": false,
  "ReconcileUnrealizedCastsPass": false,
  "ConvertNVVMToLLVMPass": true,
  "LLVMDIScope": false
}
```

### TritonIR相关的通用优化

`vector_add`用例在TritonIR的通用优化阶段，主要有`TritonReorderBroadcast`优化生效：

```txt
TritonReorderBroadcast 之前（第 467-473 行）：
    %offsets_8 = tt.splat %block_start_6 : i32 -> tensor<1024xi32> loc(#loc20)
    %offsets_9 = arith.extsi %offsets_8 : tensor<1024xi32> to tensor<1024xi64> loc(#loc20)
    %offsets_10 = arith.extsi %offsets_7 : tensor<1024xi32> to tensor<1024xi64> loc(#loc20)
    %offsets_11 = arith.addi %offsets_9, %offsets_10 : tensor<1024xi64> loc(#loc20)
TritonReorderBroadcast 之后（第 539-547 行）：
    %offsets_8 = tt.splat %block_start_6 : i32 -> tensor<1024xi32> loc(#loc20)
    %offsets_9 = arith.extsi %block_start_6 : i32 to i64 loc(#loc20)
    %offsets_10 = tt.splat %offsets_9 : i64 -> tensor<1024xi64> loc(#loc20)
    %offsets_11 = arith.extsi %offsets_7 : tensor<1024xi32> to tensor<1024xi64> loc(#loc20)
    %offsets_12 = arith.addi %offsets_10, %offsets_11 : tensor<1024xi64> loc(#loc20)
```

#### TritonReorderBroadcast
该Pass完成的主要内容是将`offset`值从`i32`扩展到`i64`的方式做了修改：
1. 优化前：
   - 先将标量 %block_start_6 扩展为 tensor <1024xi32>
   - 再将整个 tensor 扩展为 <1024xi64>
   - 最后进行加法
2. 优化后：
   - 先将标量 %block_start_6 扩展为 i64
   - 再将 i64 扩展为 tensor <1024xi64>
   - 最后进行加法

##### 优化原理
tt.splat 是一个广播操作，将标量值复制到整个 tensor。优化关键在于：
- 减少计算量：先在标量上做类型转换，然后广播一次，而不是先广播再对整个 tensor 做类型转换
- 避免冗余操作：原代码需要先将 %block_start_6 splat 为 <1024xi32>，然后对 1024 个元素做 extsi i32->i64
- 改进后：只对单个标量做 extsi i32->i64，然后 splat 为 <1024xi64>
##### 性能影响
这种重排在 GPU 计算中特别重要，因为：
- 减少了类型转换指令数量（从 1024 次 → 1 次）
- 减少了中间 tensor 的内存占用
- 为后续的编译器优化提供更好的机会

TritonReorderBroadcast 通过重新排列广播和类型转换的顺序，将昂贵的 tensor-level 类型转换为更高效的 scalar-level 转换，显著提升了性能。