---
title: Triton-pass解析-part2
categories:
  - Compiller
  - DeepLearning
tags:
  - Triton
share: true
---


前文基于`vector_add`用例分析了triton中经历的优化pass，以及前端通用pass中具体生效的`TritonReorderBroadcast`优化的作用；本文主要结合GPU结构，分析Triton中GPU相关的优化Pass的作用。

## GPU结构
NVIDIA GPU 的基本计算架构基于 **SIMT**（Single Instruction, Multiple Thread，单指令多线程）模型，其核心组织单元包括 **SM**（Streaming Multiprocessor）、**CTA**（Cooperative Thread Array，也称为 Block）、**Warp** 和 **Thread**。它们之间具有清晰的层级关系，下面逐一介绍并说明其相互关系：
### 1. **Thread（线程）**

- 是 GPU 上最小的执行单元。
- 每个线程执行相同的 kernel 程序，但通常处理不同的数据（通过 threadIdx 等内置变量区分）。
- 线程不能独立调度，而是以 **Warp** 为单位进行调度。

---

### 2. **Warp（线程束）**

- **32 个连续的线程**组成一个 Warp（在当前所有 NVIDIA 架构中，Warp 大小固定为 32）。
- Warp 是 **硬件调度和执行的基本单位**。SM 一次发射一条指令，作用于整个 Warp 的 32 个线程。
- 如果 Warp 中的线程执行路径不同（如 if-else 分支），就会发生 **Warp Divergence**（线程束发散），导致性能下降（因为需要串行执行不同分支）。
- 所有线程共享相同的程序计数器（PC）和执行上下文。

> 举例：若一个 Block 包含 256 个线程，则会被划分为 256 / 32 = **8 个 Warps**。

---

### 3. **CTA（Cooperative Thread Array）/ Block（线程块）**

- CTA 是程序员在 CUDA 编程中定义的逻辑单元，也称为 **Block**。
- 一个 Block 包含多个线程（数量由程序员指定，如 128、256、512 等，需满足硬件限制）。
- 同一个 Block 内的线程可以：
    - 通过 **Shared Memory**（共享内存）高效通信；
    - 使用 **__syncthreads()** 进行同步。
- **Block 是调度到 SM 上的基本单位**：一个 Block 必须整体分配给某一个 SM，不能跨 SM 执行。
- 一个 SM 可以同时驻留多个 Blocks（取决于资源如寄存器、共享内存等）。

---

### 4. **SM（Streaming Multiprocessor）**

- 是 GPU 的物理计算核心。一块 GPU 芯片包含多个 SM（如 A100 有 108 个 SM，RTX 4090 有 128 个）。
- 每个 SM 包含：
    - 多个 **CUDA Core**（用于执行浮点/整数运算）；
    - **Warp Scheduler**（负责调度 Warps）；
    - **Register File**（大容量寄存器文件，供线程使用）；
    - **Shared Memory / L1 Cache**（可配置大小）；
    - 特殊功能单元（SFU）、加载/存储单元等。
- SM 以 **Warp** 为单位调度执行：每个时钟周期，Warp Scheduler 会选择一个“准备好”的 Warp 发射一条指令（可能是算术、内存访问等）。
- 为了隐藏延迟（如内存访问延迟），SM 需要维护 **大量活跃的 Warps**（称为 **Occupancy**，占用率）。

4者之间的关系总结：
```text
GPU
└── 多个 SM（Streaming Multiprocessors）
    └── 每个 SM 可同时驻留多个 CTA（Blocks）
        └── 每个 CTA（Block）包含多个 Threads
            └── Threads 被组织成多个 Warps（每 Warp = 32 Threads）
                └── Warp 是 SM 调度和执行的基本单位
```

下图为H100中1个SM的结构：
![[../assets/posts/2026-02-07-Triton-pass解析-part2-20260207171516136.png|2026-02-07-Triton-pass解析-part2-20260207171516136.png]]

## Triton中GPU相关优化

在`vector_add`用例中，GPU相关优化生效的pass主要包括：
- ConvertTritonToTritonGPU
- TritonGPUCoalesce
- TritonGPURemoveLayoutConversions
### ConvertTritonToTritonGPU
根据日志中ConvertTritonToTritonGPU前后的IR对比,该Pass完成了以下关键优化:
1. 添加GPU布局属性
转换前:
```
%offsets_8 = tt.splat %block_start_6 : i32 -> tensor<1024xi32>
```
转换后:
```
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
%offsets_8 = tt.splat %block_start_6 : i32 -> tensor<1024xi32, #blocked>
```
所有张量类型都添加了#blocked布局属性,定义了数据在GPU线程间的分布方式。

上一节中介绍了GPU的CTA和Warp参数被添加到`blocked`中，笔者用的Nvidia 4060ti GPU，硬件架构每个SM中包含了4各WarpScheduler，支持4路并发调度，这里的`warpsPerCTA=4`，单warp内包含32各线程。

1. 添加GPU模块属性
转换前:
```
module {
```
转换后:
```
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:89", "ttg.threads-per-warp" = 32 : i32} {
```

添加了GPU执行配置:
- ttg.num-ctas: CTA(线程块)数量为1
- ttg.num-warps: 每个CTA的warp数为4
- ttg.target: 目标设备为CUDA sm_89
- ttg.threads-per-warp: 每warp 32线程

2. 核心作用
ConvertTritonToTritonGPU 是Triton编译流程中的关键一步,它将高层的Triton IR(与硬件无关)转换为TritonGPU IR(与GPU硬件相关),为后续的GPU优化(如coalescing、pipeline等)奠定基础。

### TritonGPUCoalesce

根据日志对比，TritonGPUCoalesce 完成了以下关键优化：
1. 引入新的Coalesced布局
优化前: 仅有一种布局
```
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
```
优化后: 新增coalesced布局
```
#blocked  = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
```
2. 内存访问模式优化
为load/store操作插入布局转换，使内存访问更合并(coalesced):
优化前:
```
%x_19 = tt.load %x_18, %mask_17 : tensor<1024x!tt.ptr<f32>, #blocked>
```
优化后:
```
%x_19 = ttg.convert_layout %x_18 : tensor<1024x!tt.ptr<f32>, #blocked> -> tensor<1024x!tt.ptr<f32>, #blocked1>
%x_20 = ttg.convert_layout %mask_17 : tensor<1024xi1, #blocked> -> tensor<1024xi1, #blocked1>
%x_21 = tt.load %x_19, %x_20 : tensor<1024x!tt.ptr<f32>, #blocked1>
%x_22 = ttg.convert_layout %x_21 : tensor<1024xf32, #blocked1> -> tensor<1024xf32, #blocked>
```
3. 核心优化点
- sizePerThread从1增加到4: 每个线程一次处理4个连续元素
- 减少内存事务: 通过合并访问模式，减少全局内存事务数量
- 提高带宽利用率: 更好地利用GPU内存子系统的并行性
- 保持计算布局不变: 计算仍使用#blocked布局，仅在内存访问时使用#blocked1

### TritonGPURemoveLayoutConversions
#### 1. 优化前后IR对比
【优化前】存在9个冗余布局转换
```
// x加载路径
%x_18 = tt.addptr %x, %offsets_16 : tensor<1024x!tt.ptr<f32>, #blocked>
%x_19 = ttg.convert_layout %x_18 : #blocked -> #blocked1       // 转换1
%x_20 = ttg.convert_layout %mask_17 : #blocked -> #blocked1    // 转换2
%x_21 = tt.load %x_19, %x_20 : tensor<1024xf32, #blocked1>
%x_22 = ttg.convert_layout %x_21 : #blocked1 -> #blocked       // 转换3 (转回!)
// y加载路径
%y_23 = tt.addptr %y, %offsets_16 : tensor<1024x!tt.ptr<f32>, #blocked>
%y_24 = ttg.convert_layout %y_23 : #blocked -> #blocked1       // 转换4
%y_25 = ttg.convert_layout %mask_17 : #blocked -> #blocked1    // 转换5 (重复)
%y_26 = tt.load %y_24, %y_25 : tensor<1024xf32, #blocked1>
%y_27 = ttg.convert_layout %y_26 : #blocked1 -> #blocked       // 转换6 (转回!)
// store路径
%2 = ttg.convert_layout %1 : #blocked -> #blocked1             // 转换7
%3 = ttg.convert_layout %output : #blocked -> #blocked1        // 转换8
%4 = ttg.convert_layout %mask_17 : #blocked -> #blocked1       // 转换9 (重复)
tt.store %2, %3, %4 : tensor<1024xf32, #blocked1>
```
【优化后】0个转换，统一使用#blocked1
```
%x = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked1>
%mask = tt.splat %n_elements : tensor<1024xi32, #blocked1>
%x_19 = tt.addptr %x, %offsets_17 : tensor<1024x!tt.ptr<f32>, #blocked1>
%x_20 = tt.load %x_19, %mask_18 : tensor<1024xf32, #blocked1>     // 无需转换
%output = arith.addf %x_20, %y_22 : tensor<1024xf32, #blocked1>  // 计算也用#blocked1
tt.store %1, %output, %mask_18 : tensor<1024xf32, #blocked1>      // 直接存储
```

2. Pass优化策略
	1. 分析数据流依赖: 识别整个加载-计算-存储链路的布局需求
	2. 决策最优布局: 发现#blocked1可以满足所有操作
	3. 前置布局选择: 将splat、make_range等源头操作改为#blocked1
	4. 消除冗余转换: 移除所有中间convert_layout

3. 性能收益
	- 消除shuffle指令: warp内数据重排开销降为0
	- 减少指令数: 从45条减少到36条（减少20%）
	- 降低寄存器压力: 单一布局避免多版本数据维护


## 总结

1. 对于01-vector-add这类简单element-wise kernel:
   - 真正的GPU优化只有3个Pass: ConvertTritonToTritonGPU → Coalesce → TritonGPURemoveLayoutConversions.
   - ~95%的Pass因无循环/矩阵乘法而跳过
2. 优化流程:
```
   ┌────────────────────┐
   │  初始Triton IR     │  无GPU布局
   └────────┬───────────┘
            │ ConvertTritonToTritonGPU
   ┌────────▼───────────┐
   │  #blocked布局       │  所有tensor添加布局
   └────────┬───────────┘
            │ TritonGPUCoalesce
   ┌────────▼───────────┐
   │  #blocked + #blocked1│  内存访问coalesced
   └────────┬───────────┘
            │ TritonGPURemoveLayoutConversions
   ┌────────▼───────────┐
   │  统一使用#blocked1  │  消除冗余转换
   └────────┬───────────┘
            │ ConvertTritonGPUToLLVM
   ┌────────▼───────────┐
   │  LLVM IR           │  准备生成机器码
   └────────────────────┘
```

3. 关键洞见:
   - Triton的Pass设计针对通用场景，简单kernel无法触发高级优化
   - 布局优化(Coalesce + RemoveLayoutConverson是通用收益
   - 循环/矩阵乘法相关Pass是性能关键，但对简单kernel无效。