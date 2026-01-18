---
title: Trition-核心Dialect和Pass
categories:
  - Compiller
  - DeepLearning
tags:
  - Triton
share: true
---

## Triton中Dialect
Triton中Dialect包含和核心的IR设计，Dialect总览（目录 triton/lib/Dialect 下）：
- `Triton/`
- `TritonGPU/`
- `TritonNvidiaGPU/`
- `TritonInstrument/`
- `Gluon/`

下面按目录逐个说明。

1) `Triton`（路径：Triton）
- 作用（核心、高层 IR）  
  - 这是 Triton 的“前端 / 高层” dialect。承载了 Triton kernel 的主要抽象：线程/程序模型（如 get_program_id）、tensor-of-pointers 表示、`tt.make_range`、`tt.addptr`、`tt.load`/`tt.store`（高层 tensor/pointer ops）、广播/算术等。通常由 Python 前端生成或作为 early-stage IR，用于实现和验证 Triton kernel 的语义与高层优化。
- 关键实现/文件（你可以打开查看）  
  - Ops.cpp, `Types.cpp`, `Dialect.cpp`：定义 ops/type/attributes。  
  - Transforms 下的 transforms：  
    - `RewriteTensorDescriptorToPointer.cpp`、`RewriteTensorPointer.cpp`：把高层 tensor 描述符/descriptor 重写为指针/更低层表示（重要的 pointer 降级 pass）。  
    - `LoopUnroll.cpp`、`LoopPeeling.cpp`、`LoopInvariantCodeMotion.cpp`、`LoopAwareCSE.cpp`：循环优化相关（scf.for 的变换、iter_args 优化等）。  
    - `Combine.cpp`, `ReorderBroadcast.cpp`, `ArithTypeConversion.cpp`：通用 canonicalize/合并/类型调整等。
- 对应测试/pass 目的  
  - 在高层阶段做 canonicalize、简化、loop-level 变换和准备（例如 vecadd.mlir 中的`scf.for/iter_args`）。这些 pass 保证高层 IR 语义正确并为后续内存/编码 pass准备。

2) `TritonGPU`（路径：TritonGPU）
- 作用（GPU-target-independent GPU-dialect + transforms）  
  - 表示面向 GPU 的中间 dialect（资源分布、memdesc、warp/CTA 编码、异步 copy 的中间表示等），包含大量用于把 high-level Triton IR 降低成能映射到 GPU-specific memory/compute patterns 的变换与优化。它既包含中立于厂商的 GPU 优化（coalescing、prefetch/pipelining、warp specialization），也包含用于后端再降低前的准备工作（layout propagation、TMEM allocation hoisting 等）。
- 关键实现/文件和 transforms（可以直接看到很多与 vecadd/后端相关的 pass）
  - IR 包含 `Ops.cpp`, `Types.cpp`, `LinearLayoutConversions.cpp`：定义 GPU 相关的 ops/encoding 与 types。  
  - Transforms 重要文件：  
    - `AccelerateMatmul.cpp`：matmul 加速策略（MMA 转换、warps 分配、shared-memory preparation）。（你已提供该文件，里面有 `BlockedToMMA`、`BlockedToMMAv5` 等 pattern）  
    - `Coalesce.cpp` / `CoalesceAsyncCopy.cpp`：合并/组织内存访问，插入/合并异步 copy（`ttg.copy_async` 风格）以提高带宽。  
    - `Pipeliner/`：实现软件流水线/多阶段拷贝以隐藏内存延迟。  
    - `OptimizeThreadLocality.cpp`、`WarpSpecialization/`：改进线程/warp 的局部性与 specialized warp 生成。  
    - `DecomposeScaledBlocked.cpp`、`OptimizeAccumulatorInit.cpp`、`RemoveLayoutConversions.cpp` 等：针对具体算子（dot/accumulator）做优化和降级。
- 对应测试/pass 目的  
  - 将高层 tensor-of-ptrs/tt.load/tt.store 转为更“硬件友好”的访问模式（coalesced encodings、async copy、SMEM 多缓冲），以及对矩阵乘（dot family）应用 MMA/TCGen 优化路径（AccelerateMatmul）。这是 vecadd 注释版中看到 `ttg.copy_async` / `coalesced encoding` 风格 IR 的来源。

1) `TritonNvidiaGPU`（路径：TritonNvidiaGPU）
- 作用（NVIDIA-specific dialect / TMEM/TCGen 等）  
  - 包含面向 NVIDIA（具体 GPU 架构特性，如 TCGen、TMEM、特殊 memory encoding/placement）的 ops 和工具函数。它承载与 NVIDIA 目标直接相关的内存描述与特定操作，通常是把 TritonGPU 的更通用表示映射到 NVIDIA-specific 中间表示（例如 TCGen5 MMA、TMEM load/store、tmem alloc 等）。
- 关键实现/文件  
  - Ops.cpp, `TensorMemoryUtils.cpp`, `Dialect.cpp`：实现针对 NV 的 memory encoding/utility（TMEM、tensor memory descriptors、TCGen op helpers）。  
  - `Transforms/` 目录（存在）里会包含将 `TritonGPU` 表示进一步映射到 NV-specific constructs 的 passes。
- 对应测试/pass 目的  
  - 生成或准备 NV-specific lowering（例如最终生成 PTX/SASS 所需的 layout、TMEM 操作与 MMA 细节）。AccelerateMatmul 里与 `NvidiaMmaEncodingAttr`、`TCGen5MMAOp` 等的生成就与此层或相近层次相关。

2) `TritonInstrument`（路径：TritonInstrument）
- 作用（instrumentation / profiling / debug helpers）  
  - 提供用于插桩 / 性能计量 / instrumentation 的 ops 和工具。用于在 IR 中放置计时、计数、性能事件、GPU instrument helper calls 或辅助库接口来收集运行时信息或验证 runtime 行为。
- 关键实现/文件  
  - `IR/FunctionBuilder.cpp`, `Ops.cpp`, `Utility.cpp`：实现 instrumentation ops，和用于在 pass 中插入 instrumentation 的 builder。  
  - `Transforms/`：包含与 instrumentation 相关的转换/插桩 passes（例如把某些 ops 替换为 instrumented variants，或在 kernel 周围插入 hooks）。
- 对应测试/pass 目的  
  - 用于测试/验证 pipeline 的性能行为，或为单元测试（比如 instrumentation tests）提供 hook。

3) `Gluon`（路径：Gluon）
- 作用（前端/高层算子集 / Gluon 集成）  
  - 封装 Gluon（高层 fusion / frontend）相关的表达/ops。Gluon 在此处很可能表示用于将高层算子（例如来自某个前端或特定融合方案）映射到 Triton IR 的中间层表示，或包含特有的 layout 推断/auto-encoding pass（repo 中也存在 Gluon）。
- 关键实现/文件  
  - Dialect.cpp：定义 Gluon dialect。  
  - `Transforms/`：（存在）包含用于将 Gluon ops 转换/降低到 Triton/TritonGPU 的 pass。
- 对应测试/pass 目的  
  - 支持从更高层（框架/融合算子）到 Triton 的转换，或在前端阶段进行布局/编码推断以便生成高效 kernel。

## Triton的转换流程

本部分从Triton实现的`vector_add`操作出发，分析以下Triton的转换流程以及主要Pass。`vector_add`实现如下：
```python
@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)
```
实现逻辑非常简单，获取pid后计算偏移，加载x和y值，计算结束后store到对应偏移位置。
当前Triton实现的性能和torch原生基本一致，16M以上基本能到MemoryBound：
```text
vector-add-performance:
           size  Triton (GB/s)  Torch (GB/s)
0        4096.0       9.600000      9.600000
1        8192.0      16.253968     16.695652
2       16384.0      27.428571     26.597403
3       32768.0      48.000000     48.000000
4       65536.0      76.800002     76.800002
5      131072.0     118.153847    118.153847
6      262144.0     180.705879    170.666661
7      524288.0     186.181817    186.181817
8     1048576.0     223.418180    219.428568
9     2097152.0     224.566533    224.984118
10    4194304.0     237.556868    238.601945
11    8388608.0     243.854884    243.930513
12   16777216.0     246.530402    247.072571
13   33554432.0     251.033168    250.456057
14   67108864.0     251.256223    251.654468
15  134217728.0     253.483321    253.605926
```

在Triton优化过程中先后经过了`MLIR`层优化和`LLVM`层优化，下面分别看经历了那些优化pass。

### MLIR层优化
MLIR中经历的优化pass：
```text
// -----// IR Dump Before Inliner (inline) ('builtin.module' operation) //----- //
// -----// IR Dump Before Canonicalizer (canonicalize) ('tt.func' operation: @add_kernel) //----- //
// -----// IR Dump Before TritonRewriteTensorPointer (triton-rewrite-tensor-pointer) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonRewriteTensorDescriptorToPointer (triton-rewrite-tensor-descriptor-to-pointer) ('builtin.module' operation) //----- //
// -----// IR Dump Before Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonCombineOps (triton-combine) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonReorderBroadcast (triton-reorder-broadcast) ('builtin.module' operation) //----- //
// -----// IR Dump Before CSE (cse) ('builtin.module' operation) //----- //
// -----// IR Dump Before SymbolDCE (symbol-dce) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonLoopUnroll (triton-loop-unroll) ('builtin.module' operation) //----- //
// -----// IR Dump Before ConvertTritonToTritonGPU (convert-triton-to-tritongpu) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUCoalesce (tritongpu-coalesce) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUF32DotTC (tritongpu-F32DotTC) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUPlanCTAPass (triton-nvidia-gpu-plan-cta) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPURemoveLayoutConversions (tritongpu-remove-layout-conversions) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUOptimizeThreadLocality (tritongpu-optimize-thread-locality) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUAccelerateMatmul (tritongpu-accelerate-matmul) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPURemoveLayoutConversions (tritongpu-remove-layout-conversions) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUOptimizeDotOperands (tritongpu-optimize-dot-operands) ('builtin.module' operation) //----- //
// -----// IR Dump Before Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonNvidiaGPUOptimizeDescriptorEncodingPass (triton-nvidia-optimize-descriptor-encoding) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonLoopAwareCSE (triton-loop-aware-cse) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUFuseNestedLoops (tritongpu-fuse-nested-loops) ('builtin.module' operation) //----- //
// -----// IR Dump Before Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonLoopInvariantCodeMotion (triton-licm) ('builtin.module' operation) //----- //
// -----// IR Dump Before Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUCombineTensorSelectAndIf (tritongpu-combine-tensor-select-and-if) ('builtin.module' operation) //----- //
// -----// IR Dump Before NVGPUWarpSpecialization (nvgpu-warp-specialization) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUAssignLatencies (tritongpu-assign-latencies) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUScheduleLoops (tritongpu-schedule-loops) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUPipeline (tritongpu-pipeline) ('builtin.module' operation) //----- //
// -----// SoftwarePipeliner internal IR Dump After: LowerLoops
// -----// SoftwarePipeliner internal IR Dump After: ExpandLoops
// -----// IR Dump Before Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonLoopAwareCSE (triton-loop-aware-cse) ('builtin.module' operation) //----- //
// -----// IR Dump Be// -----// IR Dump Before Inliner (inline) ('builtin.module' operation) //----- //
// -----// IR Dump Before Canonicalizer (canonicalize) ('tt.func' operation: @add_kernel) //----- //
// -----// IR Dump Before TritonRewriteTensorPointer (triton-rewrite-tensor-pointer) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonRewriteTensorDescriptorToPointer (triton-rewrite-tensor-descriptor-to-pointer) ('builtin.module' operation) //----- //
// -----// IR Dump Before Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonCombineOps (triton-combine) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonReorderBroadcast (triton-reorder-broadcast) ('builtin.module' operation) //----- //
// -----// IR Dump Before CSE (cse) ('builtin.module' operation) //----- //
// -----// IR Dump Before SymbolDCE (symbol-dce) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonLoopUnroll (triton-loop-unroll) ('builtin.module' operation) //----- //
// -----// IR Dump Before ConvertTritonToTritonGPU (convert-triton-to-tritongpu) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUCoalesce (tritongpu-coalesce) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUF32DotTC (tritongpu-F32DotTC) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUPlanCTAPass (triton-nvidia-gpu-plan-cta) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPURemoveLayoutConversions (tritongpu-remove-layout-conversions) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUOptimizeThreadLocality (tritongpu-optimize-thread-locality) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUAccelerateMatmul (tritongpu-accelerate-matmul) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPURemoveLayoutConversions (tritongpu-remove-layout-conversions) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUOptimizeDotOperands (tritongpu-optimize-dot-operands) ('builtin.module' operation) //----- //
// -----// IR Dump Before Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonNvidiaGPUOptimizeDescriptorEncodingPass (triton-nvidia-optimize-descriptor-encoding) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonLoopAwareCSE (triton-loop-aware-cse) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUFuseNestedLoops (tritongpu-fuse-nested-loops) ('builtin.module' operation) //----- //
// -----// IR Dump Before Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonLoopInvariantCodeMotion (triton-licm) ('builtin.module' operation) //----- //
// -----// IR Dump Before Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUCombineTensorSelectAndIf (tritongpu-combine-tensor-select-and-if) ('builtin.module' operation) //----- //
// -----// IR Dump Before NVGPUWarpSpecialization (nvgpu-warp-specialization) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUAssignLatencies (tritongpu-assign-latencies) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUScheduleLoops (tritongpu-schedule-loops) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUPipeline (tritongpu-pipeline) ('builtin.module' operation) //----- //
// -----// SoftwarePipeliner internal IR Dump After: LowerLoops
// -----// SoftwarePipeliner internal IR Dump After: ExpandLoops
// -----// IR Dump Before Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonLoopAwareCSE (triton-loop-aware-cse) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUPrefetch (tritongpu-prefetch) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUOptimizeDotOperands (tritongpu-optimize-dot-operands) ('builtin.module' operation) //----- //
// -----// IR Dump Before Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUCoalesceAsyncCopy (tritongpu-coalesce-async-copy) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonNvidiaGPUOptimizeTMemLayoutsPass (triton-nvidia-optimize-tmem-layouts) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPURemoveLayoutConversions (tritongpu-remove-layout-conversions) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonNvidiaGPUInterleaveTMemPass (triton-nvidia-interleave-tmem) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUReduceDataDuplication (tritongpu-reduce-data-duplication) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUReorderInstructions (tritongpu-reorder-instructions) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonLoopAwareCSE (triton-loop-aware-cse) ('builtin.module' operation) //----- //
// -----// IR Dump Before SymbolDCE (symbol-dce) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUFenceInsertion (triton-nvidia-gpu-fence-insertion) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonNvidiaGPUMMALoweringPass (triton-nvidia-mma-lowering) ('builtin.module' operation) //----- //
// -----// IR Dump Before SCCP (sccp) ('builtin.module' operation) //----- //
// -----// IR Dump Before CSE (cse) ('builtin.module' operation) //----- //
// -----// IR Dump Before Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUCombineTensorSelectAndIf (tritongpu-combine-tensor-select-and-if) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUAllocateWarpGroups (tritongpu-allocate-warp-groups) ('builtin.module' operation) //----- //
// -----// IR Dump Before SCFToControlFlowPass (convert-scf-to-cf) ('builtin.module' operation) //----- //
// -----// IR Dump Before GluonInline (gluon-inline) ('builtin.module' operation) //----- //
// -----// IR Dump Before AllocateSharedMemoryNv (allocate-shared-memory-nv) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonTensorMemoryAllocationPass (triton-tensor-memory-allocation) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonNvidiaGPUCheckMatmulTwoCTAPass (triton-nvidia-check-matmul-two-cta) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUGlobalScratchAllocationPass (tritongpu-global-scratch-memory-allocation) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUProxyFenceInsertion (triton-nvidia-gpu-proxy-fence-insertion) ('builtin.module' operation) //----- //
// -----// IR Dump Before ConvertTritonGPUToLLVM (convert-triton-gpu-to-llvm) ('builtin.module' operation) //----- //
// -----// IR Dump Before Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
// -----// IR Dump Before CSE (cse) ('builtin.module' operation) //----- //
// -----// IR Dump Before ConvertNVGPUToLLVM (convert-nv-gpu-to-llvm) ('builtin.module' operation) //----- //
// -----// IR Dump Before ConvertWarpSpecializeToLLVM (convert-warp-specialize-to-llvm) ('builtin.module' operation) //----- //
// -----// IR Dump Before ReconcileUnrealizedCastsPass (reconcile-unrealized-casts) ('builtin.module' operation) //----- //
// -----// IR Dump Before Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
// -----// IR Dump Before CSE (cse) ('builtin.module' operation) //----- //
// -----// IR Dump Before SymbolDCE (symbol-dce) ('builtin.module' operation) //----- //
// -----// IR Dump Before ConvertNVVMToLLVMPass (convert-nvvm-to-llvm) ('builtin.module' operation) //----- //
// -----// IR Dump Before LLVMDIScope (enable-line-info) ('builtin.module' operation) //----- //
fore TritonGPUPrefetch (tritongpu-prefetch) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUOptimizeDotOperands (tritongpu-optimize-dot-operands) ('builtin.module' operation) //----- //
// -----// IR Dump Before Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUCoalesceAsyncCopy (tritongpu-coalesce-async-copy) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonNvidiaGPUOptimizeTMemLayoutsPass (triton-nvidia-optimize-tmem-layouts) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPURemoveLayoutConversions (tritongpu-remove-layout-conversions) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonNvidiaGPUInterleaveTMemPass (triton-nvidia-interleave-tmem) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUReduceDataDuplication (tritongpu-reduce-data-duplication) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUReorderInstructions (tritongpu-reorder-instructions) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonLoopAwareCSE (triton-loop-aware-cse) ('builtin.module' operation) //----- //
// -----// IR Dump Before SymbolDCE (symbol-dce) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUFenceInsertion (triton-nvidia-gpu-fence-insertion) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonNvidiaGPUMMALoweringPass (triton-nvidia-mma-lowering) ('builtin.module' operation) //----- //
// -----// IR Dump Before SCCP (sccp) ('builtin.module' operation) //----- //
// -----// IR Dump Before CSE (cse) ('builtin.module' operation) //----- //
// -----// IR Dump Before Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUCombineTensorSelectAndIf (tritongpu-combine-tensor-select-and-if) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUAllocateWarpGroups (tritongpu-allocate-warp-groups) ('builtin.module' operation) //----- //
// -----// IR Dump Before SCFToControlFlowPass (convert-scf-to-cf) ('builtin.module' operation) //----- //
// -----// IR Dump Before GluonInline (gluon-inline) ('builtin.module' operation) //----- //
// -----// IR Dump Before AllocateSharedMemoryNv (allocate-shared-memory-nv) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonTensorMemoryAllocationPass (triton-tensor-memory-allocation) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonNvidiaGPUCheckMatmulTwoCTAPass (triton-nvidia-check-matmul-two-cta) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUGlobalScratchAllocationPass (tritongpu-global-scratch-memory-allocation) ('builtin.module' operation) //----- //
// -----// IR Dump Before TritonGPUProxyFenceInsertion (triton-nvidia-gpu-proxy-fence-insertion) ('builtin.module' operation) //----- //
// -----// IR Dump Before ConvertTritonGPUToLLVM (convert-triton-gpu-to-llvm) ('builtin.module' operation) //----- //
// -----// IR Dump Before Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
// -----// IR Dump Before CSE (cse) ('builtin.module' operation) //----- //
// -----// IR Dump Before ConvertNVGPUToLLVM (convert-nv-gpu-to-llvm) ('builtin.module' operation) //----- //
// -----// IR Dump Before ConvertWarpSpecializeToLLVM (convert-warp-specialize-to-llvm) ('builtin.module' operation) //----- //
// -----// IR Dump Before ReconcileUnrealizedCastsPass (reconcile-unrealized-casts) ('builtin.module' operation) //----- //
// -----// IR Dump Before Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
// -----// IR Dump Before CSE (cse) ('builtin.module' operation) //----- //
// -----// IR Dump Before SymbolDCE (symbol-dce) ('builtin.module' operation) //----- //
// -----// IR Dump Before ConvertNVVMToLLVMPass (convert-nvvm-to-llvm) ('builtin.module' operation) //----- //
// -----// IR Dump Before LLVMDIScope (enable-line-info) ('builtin.module' operation) //----- //
```

各pass出现频次以及主要逻辑如下：
- `canonicalize`（Canonicalizer） — 10 次  
  目的：做局部的代数/结构等价变换以简化 IR（例如合并常量、消除零操作、简化嵌套结构）。  
  基本逻辑：应用一组等价规则把复杂表达式替换成更简单或更规则的形式，方便后续 pass 更有效地优化和匹配模式。

- `cse`（CSE, common subexpression elimination） — 4 次  
  目的：消除公共子表达式，复用已计算的值以减少重复计算。  
  基本逻辑：在作用域内查找语义相同且不变的表达式，把后者替换为先前生成的 SSA 值或临时，减少指令数和内存操作。

- `symbol-dce`（SymbolDCE） — 3 次  
  目的：删除未使用的全局符号（函数/全局变量等）。  
  基本逻辑：分析符号的引用图，移除没有外部或内部可到达引用的定义以减小模块体积。

- `tritongpu-remove-layout-conversions`（TritonGPURemoveLayoutConversions） — 3 次  
  目的：移除或合并多余的布局转换（tensor layout conversion）以减少不必要的数据重排。  
  基本逻辑：检测连续或等价的布局变换，将其消除或折叠到产生/消费点，减少内存/拷贝开销。

- `triton-loop-aware-cse`（TritonLoopAwareCSE） — 3 次  
  目的：在循环上下文中进行更智能的公共子表达式消除，避免破坏循环语义或寄存器使用模式。  
  基本逻辑：考虑循环不变性和循环范围，提取循环外可共享的计算或在循环体内复用表达式，同时避免跨迭代的错误合并。

- `tritongpu-optimize-dot-operands`（TritonGPUOptimizeDotOperands） — 2 次  
  目的：为点乘/矩阵乘（dot/matmul）优化操作数布局或预处理以提高性能（例如转置/重排以匹配硬件吞吐）。  
  基本逻辑：分析 dot 操作的维度与内存布局，做操作数重排、合并或形式变换以便后续的底层实现（如 MMA/WMMA）更高效。

- `tritongpu-combine-tensor-select-and-if`（TritonGPUCombineTensorSelectAndIf） — 2 次  
  目的：把相邻的 tensor select/if 之类的控制/选择模式合并为更简洁高效的形式。  
  基本逻辑：识别可合并的条件选择/分支结构，替换为单个更优的算子或减少分支以支持更好的向量化/并行化。

单次出现的 pass：
- `inline`（Inliner）  
  目的：函数内联，减少调用开销、增加后续优化机会。

- `triton-rewrite-tensor-pointer`  
  目的：把高层 tensor 表示改写为基于指针的表示，便于低层内存操作。

- `triton-rewrite-tensor-descriptor-to-pointer`  
  目的：将 tensor 描述符转换到指针/原始内存视图以便后端代码生成。

- `triton-combine`（TritonCombineOps）  
  目的：合并相邻的 Triton 操作以形成更大的复合/更高效的算子。

- `triton-reorder-broadcast`（TritonReorderBroadcast）  
  目的：重排广播/维度操作以减少内存移动或更好匹配后续算子。

- `triton-loop-unroll`（TritonLoopUnroll）  
  目的：循环展开以增加指令级并行与减少分支；也为向量化做准备。

- `convert-triton-to-tritongpu`（ConvertTritonToTritonGPU）  
  目的：将 Triton 高层 IR 转换到针对 GPU 的 TritonGPU dialect。

- `tritongpu-coalesce`（TritonGPUCoalesce）  
  目的：合并内存访问以形成更大的/更顺序的访问，改善内存带宽利用。

- `tritongpu-F32DotTC`  
  目的：用特定的高效核（例如 Tensor Core）实现或优化 float32 的点积/收敛路径。

- `triton-nvidia-gpu-plan-cta`（TritonGPUPlanCTAPass）  
  目的：规划线程块（CTA）维度与网格映射，决定线程组织和每个 CTA 的任务量。

- `tritongpu-optimize-thread-locality`（TritonGPUOptimizeThreadLocality）  
  目的：优化线程对数据的局部性，减少跨线程/跨 CTA 的数据竞争或远程访问。

- `tritongpu-accelerate-matmul`（TritonGPUAccelerateMatmul）  
  目的：用专门策略或替换为高效内核来加速矩阵乘。

- `triton-nvidia-optimize-descriptor-encoding`  
  目的：优化描述符（descriptor）编码以减少元数据大小或加快解析。

- `tritongpu-fuse-nested-loops`（TritonGPUFuseNestedLoops）  
  目的：融合嵌套循环以减少循环开销，提高并行体大小或内存访问连续性。

- `triton-licm`（TritonLoopInvariantCodeMotion）  
  目的：把循环不变的计算提取到循环外，减少重复计算。

- `nvgpu-warp-specialization`（NVGPUWarpSpecialization）  
  目的：对 warp 级并行策略做专门化以利用 GPU 的 warp 特性（例如分配不同角色给不同线程）。

- `tritongpu-assign-latencies`（TritonGPUAssignLatencies）  
  目的：给不同操作分配延迟估计，以便调度和管线化决策。

- `tritongpu-schedule-loops`（TritonGPUScheduleLoops）  
  目的：根据硬件特性和延迟/吞吐估计安排循环执行顺序和映射。

- `tritongpu-pipeline`（TritonGPUPipeline）  
  目的：对多个阶段（load/compute/store）进行软件流水线以掩盖延迟并提高吞吐。

- `SoftwarePipeliner` internal dumps（LowerLoops / ExpandLoops）  
  目的：循环软件流水线的内部阶段性转换（内部 IR dump 表明进行了流水线的重写/展开）。

- `tritongpu-prefetch`（TritonGPUPrefetch）  
  目的：插入预取指令以在需要前把数据加载到寄存器/共享内存，减少等待。

- `tritongpu-coalesce-async-copy`（TritonGPUCoalesceAsyncCopy）  
  目的：合并异步拷贝点以减少同步/分散的 DMA 操作次数。

- `triton-nvidia-optimize-tmem-layouts`  
  目的：优化线程内存（tmem）或共享内存布局以提高 bank conflict 和带宽表现。

- `triton-nvidia-interleave-tmem`  
  目的：对 tmem 数据做交错/重排以利用并行拷贝或避免冲突。

- `tritongpu-reduce-data-duplication`  
  目的：减少在多个线程/CTA 间不必要的数据拷贝或冗余存储。

- `tritongpu-reorder-instructions`（TritonGPUReorderInstructions）  
  目的：重新排序指令以改善调度、流水线或减少寄存器压力。

- `triton-nvidia-gpu-fence-insertion`（TritonGPUFenceInsertion）  
  目的：在必要点插入内存屏障/fence，保证异步拷贝/共享内存的可见性与同步。

- `triton-nvidia-mma-lowering`（TritonNvidiaGPUMMALoweringPass）  
  目的：把高层的矩阵乘相关操作下沉（lower）到对 MMA/Tensor Core 的具体调用或指令序列。

- `sccp`（SCCP：sparse conditional constant propagation）  
  目的：做条件常量传播和不确定性消除，替换可静态推断的值并删除死分支。  
  基本逻辑：结合控制流与数据流分析，推断在不同条件下变量是否常量，从而简化和消除不可达/恒定分支。

- `tritongpu-allocate-warp-groups`（TritonGPUAllocateWarpGroups）  
  目的：为 warp 组分配资源/任务以匹配硬件并行结构。

- `convert-scf-to-cf`（SCFToControlFlowPass）  
  目的：将结构化控制流（scf）转换为更低层的显式控制流（cf），便于后端生成。

- `gluon-inline`（GluonInline）  
  目的：类似函数内联，但可能针对框架/特定调用约定的特殊内联规则。

- `allocate-shared-memory-nv`  
  目的：在 NV（NVIDIA）目标为 kernel 分配共享内存资源并插入相应描述。

- `triton-tensor-memory-allocation`（TritonTensorMemoryAllocationPass）  
  目的：为 Triton 的 tensor 分配具体内存（寄存器/共享/全局）并产生相应指针/偏移。

- `triton-nvidia-check-matmul-two-cta`  
  目的：检查/决定是否将一个 matmul 划分为两个 CTA 执行（适用于特定硬件/问题规模）。

- `tritongpu-global-scratch-memory-allocation`  
  目的：分配全局 scratch（临时）内存区域给 GPU kernel。

- `triton-nvidia-gpu-proxy-fence-insertion`（TritonGPUProxyFenceInsertion）  
  目的：插入代理/辅助同步操作以协调不同内存模型或拷贝实现细节。

- `convert-triton-gpu-to-llvm`（ConvertTritonGPUToLLVM）  
  目的：把 TritonGPU dialect 降低到 LLVM-level IR（用于最终代码生成）。

- `convert-nv-gpu-to-llvm`（ConvertNVGPUToLLVM）  
  目的：把 NV（NVIDIA）特定 dialect/操作转换为 LLVM IR（例如 NVVM -> LLVM）。

- `convert-warp-specialize-to-llvm`（ConvertWarpSpecializeToLLVM）  
  目的：将 warp 专门化的中间表示转换成低层实现（LLVM）。

- `reconcile-unrealized-casts`（ReconcileUnrealizedCastsPass）  
  目的：清理/统一“未实现的”类型转换占位符，插入必要的实际 cast/convert 操作或移除冗余占位。

- `convert-nvvm-to-llvm`（ConvertNVVMToLLVMPass）  
  目的：将 NVVM-specific IR 转换到标准的 LLVM IR。

- `enable-line-info`（LLVMDIScope）  
  目的：在生成的 LLVM IR 中启用/保留调试/行信息（DI scope），便于调试/映射回源代码。


### LLVM层优化
llvm中经历的pass：
```text
Running pass: MemProfRemoveInfo on [module]
Running pass: Annotation2MetadataPass on [module]
Running pass: ForceFunctionAttrsPass on [module]
Running pass: InferFunctionAttrsPass on [module]
Running pass: CoroEarlyPass on [module]
Running pass: EntryExitInstrumenterPass on add_kernel (191 instructions)
Running pass: LowerExpectIntrinsicPass on add_kernel (191 instructions)
Running pass: SimplifyCFGPass on add_kernel (191 instructions)
Running pass: SROAPass on add_kernel (191 instructions)
Running pass: EarlyCSEPass on add_kernel (191 instructions)
Running pass: CallSiteSplittingPass on add_kernel (169 instructions)
Running pass: OpenMPOptPass on [module]
Running pass: IPSCCPPass on [module]
Running pass: CalledValuePropagationPass on [module]
Running pass: GlobalOptPass on [module]
Running pass: PromotePass on add_kernel (157 instructions)
Running pass: InstCombinePass on add_kernel (157 instructions)
Running pass: SimplifyCFGPass on add_kernel (84 instructions)
Running pass: AlwaysInlinerPass on [module]
Running pass: ModuleInlinerWrapperPass on [module]
Running pass: RequireAnalysisPass<llvm::GlobalsAA, llvm::Module> on [module]
Running pass: InvalidateAnalysisPass<llvm::AAManager> on add_kernel (82 instructions)
Running pass: RequireAnalysisPass<llvm::ProfileSummaryAnalysis, llvm::Module> on [module]
Running pass: DevirtSCCRepeatedPass on (add_kernel) (1 node)
Running pass: InlinerPass on (add_kernel) (1 node)
Running pass: PostOrderFunctionAttrsPass on (add_kernel) (1 node)
Running pass: ArgumentPromotionPass on (add_kernel) (1 node)
Running pass: OpenMPOptCGSCCPass on (add_kernel) (1 node)
Running pass: SROAPass on add_kernel (82 instructions)
Running pass: EarlyCSEPass on add_kernel (82 instructions)
Running pass: SpeculativeExecutionPass on add_kernel (78 instructions)
Running pass: JumpThreadingPass on add_kernel (78 instructions)
Running pass: CorrelatedValuePropagationPass on add_kernel (78 instructions)
Running pass: SimplifyCFGPass on add_kernel (78 instructions)
Running pass: InstCombinePass on add_kernel (78 instructions)
Running pass: AggressiveInstCombinePass on add_kernel (78 instructions)
Running pass: LibCallsShrinkWrapPass on add_kernel (78 instructions)
Running pass: TailCallElimPass on add_kernel (78 instructions)
Running pass: SimplifyCFGPass on add_kernel (78 instructions)
Running pass: ReassociatePass on add_kernel (78 instructions)
Running pass: ConstraintEliminationPass on add_kernel (78 instructions)
Running pass: LoopSimplifyPass on add_kernel (78 instructions)
Running pass: LCSSAPass on add_kernel (78 instructions)
Running pass: SimplifyCFGPass on add_kernel (78 instructions)
Running pass: InstCombinePass on add_kernel (78 instructions)
Running pass: LoopSimplifyPass on add_kernel (78 instructions)
Running pass: LCSSAPass on add_kernel (78 instructions)
Running pass: SROAPass on add_kernel (78 instructions)
Running pass: VectorCombinePass on add_kernel (78 instructions)
Running pass: MergedLoadStoreMotionPass on add_kernel (78 instructions)
Running pass: GVNPass on add_kernel (78 instructions)
Running pass: SCCPPass on add_kernel (77 instructions)
Running pass: BDCEPass on add_kernel (77 instructions)
Running pass: InstCombinePass on add_kernel (77 instructions)
Running pass: JumpThreadingPass on add_kernel (77 instructions)
Running pass: CorrelatedValuePropagationPass on add_kernel (77 instructions)
Running pass: ADCEPass on add_kernel (77 instructions)
Running pass: MemCpyOptPass on add_kernel (77 instructions)
Running pass: DSEPass on add_kernel (77 instructions)
Running pass: MoveAutoInitPass on add_kernel (77 instructions)
Running pass: LoopSimplifyPass on add_kernel (77 instructions)
Running pass: LCSSAPass on add_kernel (77 instructions)
Running pass: CoroElidePass on add_kernel (77 instructions)
Running pass: SimplifyCFGPass on add_kernel (77 instructions)
Running pass: InstCombinePass on add_kernel (77 instructions)
Running pass: PostOrderFunctionAttrsPass on (add_kernel) (1 node)
Running pass: RequireAnalysisPass<llvm::ShouldNotRunFunctionPassesAnalysis, llvm::Function> on add_kernel (77 instructions)
Running pass: CoroSplitPass on (add_kernel) (1 node)
Running pass: CoroAnnotationElidePass on (add_kernel) (1 node)
Running pass: InvalidateAnalysisPass<llvm::ShouldNotRunFunctionPassesAnalysis> on add_kernel (77 instructions)
Running pass: DeadArgumentEliminationPass on [module]
Running pass: CoroCleanupPass on [module]
Running pass: GlobalOptPass on [module]
Running pass: GlobalDCEPass on [module]
Running pass: EliminateAvailableExternallyPass on [module]
Running pass: ReversePostOrderFunctionAttrsPass on [module]
Running pass: RecomputeGlobalsAAPass on [module]
Running pass: DropUnnecessaryAssumesPass on add_kernel (77 instructions)
Running pass: Float2IntPass on add_kernel (77 instructions)
Running pass: LowerConstantIntrinsicsPass on add_kernel (77 instructions)
Running pass: ControlHeightReductionPass on add_kernel (77 instructions)
Running pass: BreakStructPhiNodesPass on add_kernel (77 instructions)
Running pass: InstCombinePass on add_kernel (77 instructions)
Running pass: LoopSimplifyPass on add_kernel (77 instructions)
Running pass: LCSSAPass on add_kernel (77 instructions)
Running pass: LoopDistributePass on add_kernel (77 instructions)
Running pass: InjectTLIMappings on add_kernel (77 instructions)
Running pass: LoopVectorizePass on add_kernel (77 instructions)
Running pass: DropUnnecessaryAssumesPass on add_kernel (77 instructions)
Running pass: InferAlignmentPass on add_kernel (77 instructions)
Running pass: LoopLoadEliminationPass on add_kernel (77 instructions)
Running pass: InstCombinePass on add_kernel (77 instructions)
Running pass: SimplifyCFGPass on add_kernel (77 instructions)
Running pass: SLPVectorizerPass on add_kernel (77 instructions)
Running pass: VectorCombinePass on add_kernel (77 instructions)
Running pass: InstCombinePass on add_kernel (77 instructions)
Running pass: LoopUnrollPass on add_kernel (77 instructions)
Running pass: WarnMissedTransformationsPass on add_kernel (77 instructions)
Running pass: SROAPass on add_kernel (77 instructions)
Running pass: InferAlignmentPass on add_kernel (77 instructions)
Running pass: InstCombinePass on add_kernel (77 instructions)
Running pass: LoopSimplifyPass on add_kernel (77 instructions)
Running pass: LCSSAPass on add_kernel (77 instructions)
Running pass: AlignmentFromAssumptionsPass on add_kernel (77 instructions)
Running pass: LoopSinkPass on add_kernel (77 instructions)
Running pass: InstSimplifyPass on add_kernel (77 instructions)
Running pass: DivRemPairsPass on add_kernel (77 instructions)
Running pass: TailCallElimPass on add_kernel (77 instructions)
Running pass: SimplifyCFGPass on add_kernel (77 instructions)
Running pass: GlobalDCEPass on [module]
Running pass: ConstantMergePass on [module]
Running pass: CGProfilePass on [module]
Running pass: RelLookupTableConverterPass on [module]
Running pass: AnnotationRemarksPass on add_kernel (77 instructions)
```


各个pass的主要作用：
- InstCombinePass  
  - 逐指令的合并与常量折叠（peephole 优化）：把多步计算合成更简单或更高效的指令序列，消除显式冗余运算。

- SimplifyCFGPass  
  - 简化控制流图：删除不可达块、合并基本块、简化条件分支和分叉路径，使 CFG 更规整便于后续优化。

- LoopSimplifyPass  
  - 将 loop 规范化（例如保证有 preheader、归一化 header 等），为后续循环优化（vectorize、unroll）做准备。

- LCSSAPass  
  - 将循环内的值转换为 Loop-Closed SSA 形式（LCSSA），确保从循环外部访问循环内部定义的值时的正确性与可分析性。

- SROAPass (Scalar Replacement of Aggregates)  
  - 把聚合类型（struct/array）的存取拆成标量，替换内存访问为寄存器/局部 SSA 值，从而启用更多标量优化与寄存器分配改进。

- VectorCombinePass  
  - 合并或重写向量表达式以减少指令数或产生更适合目标后端的向量操作。

- TailCallElimPass  
  - 检测并替换尾递归/尾调用为跳转以消除栈帧（尾调用消除）。

- PostOrderFunctionAttrsPass / ReversePostOrderFunctionAttrsPass  
  - 在特定遍历顺序（后序或逆后序）上推断或更新函数属性（如 readonly、nocapture 等），有助于跨函数优化决策。

- JumpThreadingPass  
  - 利用已知条件将跳转“穿线”（threading），把某些条件分支重写为更直接的控制流以启用进一步优化。

- InferAlignmentPass / AlignmentFromAssumptionsPass  
  - 从内存操作/assume 等信息推断指针对齐属性，允许后续生成更高效的对齐访问。

- GlobalOptPass  
  - 一系列模块级全局优化（删除未使用 globlas、折叠常量、简化全局初始器等）。

- GlobalDCEPass  
  - 删除不被使用的全局对象（函数/变量）。

- EarlyCSEPass  
  - 早期的公共子表达式消除：在 IR 早期阶段合并重复计算以减少冗余。

- DropUnnecessaryAssumesPass  
  - 去掉不会影响行为或优化结果的 assume 指示，减少不必要的约束。

- CorrelatedValuePropagationPass  
  - 基于条件信息传播值（例如 if-then-cond 推断），有助于常量传播与路径敏感优化。

- WarnMissedTransformationsPass  
  - 用于诊断，记录/警告编译器未奏效的转换或期望但未完成的优化点。

- SpeculativeExecutionPass  
  - 把某些可安全推测执行的操作提前执行，以减少关键路径上的控制依赖（需要保证语义或插入检查）。

- SLPVectorizerPass  
  - 基于超字（superword）并行性把标量操作转换为向量指令，常用于块级向量化（短矢量化）。

- SCCPPass (Sparse Conditional Constant Propagation)  
  - 稀疏条件常量传播：识别并传播在特定路径上为常量的值，可能删除条件分支或替换计算。

- RequireAnalysisPass<...> / InvalidateAnalysisPass<...>  
  - 框架类调用：RequireAnalysis 表示此处需要某个分析结果；InvalidateAnalysis 表示某些变换后失效或清理分析数据结构。

- RelLookupTableConverterPass  
  - 目标/后端相关：转换或重写某些查找表/重定位相关结构以适合目标格式（具体实现与目标相关）。

- RecomputeGlobalsAAPass / RecomputeGlobalsAAPass  
  - 重新计算全局别名分析（Globals AA），以便别名信息与最新 IR 保持一致。

- ReassociatePass  
  - 重新结合 (reassociate) 算术表达式以增加常量折叠或生成更高效的序列（注意浮点需小心）。

- PromotePass  
  - 将内存上的对象提升为寄存器/SSA 值（类似 mem2reg 或把某些 allocas/promotions 做到寄存器级别）。

- OpenMPOptPass / OpenMPOptCGSCCPass  
  - 针对 OpenMP 并行构造进行优化；CGSCC 版在调用图的 SCC（强连通分量）层面运行。

- MoveAutoInitPass  
  - 与自动初始化相关的重排/优化（改善初始化代码的位置或条件，以减少开销）。

- ModuleInlinerWrapperPass / InlinerPass / AlwaysInlinerPass  
  - 模块/函数级内联：把被调用函数复制到调用点，减少调用开销并启用更多优化；AlwaysInliner 强制内联标记为 always_inline 的函数。

- MergedLoadStoreMotionPass  
  - 将可合并的 load/store 移动以减少重复内存访问或延长可共用的加载范围。

- MemProfRemoveInfo  
  - 移除与内存分析/剖面（memory profiling）相关的辅助信息（清理非必要注记）。

- MemCpyOptPass  
  - 优化 memcpy/memmove 模式（用更高效的实现或折叠成更小/更少调用）。

- LowerExpectIntrinsicPass  
  - 将 llvm.expect（分支预测提示）降低为更直接的分支权重或元数据，供后端利用。

- LowerConstantIntrinsicsPass  
  - 将某些 constant intrinsics 用常量或更简单的操作替换（降低复杂内建常量表达）。

- LoopVectorizePass / LoopUnrollPass / LoopDistributePass / LoopLoadEliminationPass / LoopSinkPass  
  - 一组循环保守优化：分别负责循环向量化、循环展开、循环分配/拆分、消除循环内冗余 load、把不变代码下沉/上提以优化执行路径等。

- LibCallsShrinkWrapPass  
  - 针对库调用进行“shrink-wrap”优化（移动 prologue/epilogue 或把开销局部化以减少不必要的保存恢复等，具体与平台相关）。

- IPSCCPPass  
  - 跨过程的 SCCP（稀疏条件常量传播），用于在模块/函数间传播常量信息。

- InvalidateAnalysisPass<...>（具体化）  
  - 在变换后使相关分析失效/刷新（例如 AAManager 需要重建别名信息）。

- InstSimplifyPass  
  - 指令级简化，与 InstCombine 目标相近，常做更保守或不同阶段的简化。

- InjectTLIMappings  
  - 看起来是 Triton/目标特定的注入映射（把 TLI/TargetMapping 注入 IR），用于后续布局或代码生成。

- InferFunctionAttrsPass / ForceFunctionAttrsPass  
  - 推断或强制函数属性（如 noalias、readonly、nounwind 等），改善调用/内联/别名分析的决策。

- GVNPass (Global Value Numbering)  
  - 通过值编号消除等价计算，去除冗余表达式/计算。

- Float2IntPass  
  - 将某些浮点表达式转换为等价的整数运算（在安全/等价前提下），用于目标或优化上的需要。

- EntryExitInstrumenterPass  
  - 在函数入口/出口插入探针/计时或分析代码（用于 profiling 或 instrumentation）。

- EliminateAvailableExternallyPass  
  - 删除标记为 available_externally 的函数定义（保留声明），以减小模块或避免重复定义。

- DSEPass / ADCEPass  
  - DSE: Dead Store Elimination（删除未使用的写）；ADCE: Aggressive Dead Code Elimination（更激进地删除不影响程序语义的死代码）。

- DivRemPairsPass  
  - 识别并优化成对出现的 div/rem 操作，或用目标友好的扩展（例如把 div 和 rem 合并/重写以共用除法结果）。

- DevirtSCCRepeatedPass / CalledValuePropagationPass / CallSiteSplittingPass  
  - 一组与去虚函数/间接调用变为直接调用相关的优化：传播被调用值、分裂调用点、对 SCC 重复去虚化尝试以使内联成为可能。

- DeadArgumentEliminationPass  
  - 删除函数不使用的参数以简化签名与调用方。

- Coro* 系列（CoroEarlyPass / CoroSplitPass / CoroElidePass / CoroCleanupPass / CoroAnnotationElidePass）  
  - 协程相关变换：把 coroutine intrinsic 标准化、拆分协程状态机、在安全情况下消除协程开销、清理残留结构等。

- ControlHeightReductionPass  
  - 降低控制流“高度”或复杂性（把深嵌套/多分支结构扁平化或重写以利于后端映射）。

- ConstraintEliminationPass  
  - 去除多余的约束（某些分析得到的条件/约束不再必要时删除）。

- ConstantMergePass  
  - 合并重复常量以节省内存表项和加载成本。

- CGProfilePass  
  - 使用或生成与调用图/代码生成相关的剖面信息（Call-Graph/Profile 支持）。

- BreakStructPhiNodesPass  
  - 将涉及结构体的 phi 节点拆解为标量 phi，以便后续标量优化。

- BDCEPass  
  - Basic/Block-level Dead Code Elimination：删除无用基本块与其中的无效代码。

- ArgumentPromotionPass  
  - 把按值传递的参数（by-value aggregates）升格为寄存器/更直接的形式以减少拷贝。

- Annotation2MetadataPass / AnnotationRemarksPass  
  - 把源级注释或注记转换为 IR metadata，并在必要时生成诊断/备注信息供用户或后端参考。

- AlwaysInlinerPass  
  - 专门处理并强制内联那些标记为 always_inline 的函数（通常优先执行）。

- AlignmentFromAssumptionsPass  
  - 从 assume/约束中推导对齐信息并将其应用到内存操作或指针属性上。

- AggressiveInstCombinePass  
  - 比 InstCombine 更积极的合并/重写，可能进行更多有利于后端的复杂变换。


Trition基于MLIR和LLVM两级优化后生成GPU指令，整个优化流程还是相当复杂的，后面会针对一些性能相关的Pass做进一步分析。

