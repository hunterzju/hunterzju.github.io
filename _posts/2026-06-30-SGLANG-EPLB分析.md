---
title: SGLANG_EPLB
categories:
  - LLM
  - DeepLearning
tags:
  - LLM
  - Transformer
  - sglang
share: true
---
# SGLang EPLB (Expert Parallel Load Balancing) 机制分析

> 结合 sglang 源码分析 EPLB 的实现原理、数据流和运行时行为

---

## 目录

- [1. 架构概览](2026-06-30-SGLANG-EPLB%E5%88%86%E6%9E%90.md##1-%E6%9E%B6%E6%9E%84%E6%A6%82%E8%A7%88)
- [2. 核心数据结构](2026-06-30-SGLANG-EPLB%E5%88%86%E6%9E%90.md##2-%E6%A0%B8%E5%BF%83%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84)
- [3. 初始化流程](2026-06-30-SGLANG-EPLB%E5%88%86%E6%9E%90.md##3-%E5%88%9D%E5%A7%8B%E5%8C%96%E6%B5%81%E7%A8%8B)
- [4. 运行时 Token 路由与 Dispatch](2026-06-30-SGLANG-EPLB%E5%88%86%E6%9E%90.md##4-%E8%BF%90%E8%A1%8C%E6%97%B6-token-%E8%B7%AF%E7%94%B1%E4%B8%8E-dispatch)
- [5. 专家负载统计与反馈](2026-06-30-SGLANG-EPLB%E5%88%86%E6%9E%90.md##5-%E4%B8%93%E5%AE%B6%E8%B4%9F%E8%BD%BD%E7%BB%9F%E8%AE%A1%E4%B8%8E%E5%8F%8D%E9%A6%88)
- [6. EPLB Rebalancing 算法详解](2026-06-30-SGLANG-EPLB%E5%88%86%E6%9E%90.md##6-eplb-rebalancing-%E7%AE%97%E6%B3%95%E8%AF%A6%E8%A7%A3)
  - [6.1 算法选择机制](2026-06-30-SGLANG-EPLB%E5%88%86%E6%9E%90.md##61-%E7%AE%97%E6%B3%95%E9%80%89%E6%8B%A9%E6%9C%BA%E5%88%B6)
  - [6.2 DeepSeek 全局算法](2026-06-30-SGLANG-EPLB%E5%88%86%E6%9E%90.md##62-deepseek-%E5%85%A8%E5%B1%80%E7%AE%97%E6%B3%95)
  - [6.3 DeepSeek 层级算法 (Hierarchical)](2026-06-30-SGLANG-EPLB%E5%88%86%E6%9E%90.md##63-deepseek-%E5%B1%82%E7%BA%A7%E7%AE%97%E6%B3%95-hierarchical)
  - [6.4 DeepSeek Vec 分块向量化算法](2026-06-30-SGLANG-EPLB%E5%88%86%E6%9E%90.md##64-deepseek-vec-%E5%88%86%E5%9D%97%E5%90%91%E9%87%8F%E5%8C%96%E7%AE%97%E6%B3%95)
  - [6.5 弹性感知算法 (Elasticity-Aware)](2026-06-30-SGLANG-EPLB%E5%88%86%E6%9E%90.md##65-%E5%BC%B9%E6%80%A7%E6%84%9F%E7%9F%A5%E7%AE%97%E6%B3%95-elasticity-aware)
- [7. 权重迁移 (P2P Update)](2026-06-30-SGLANG-EPLB%E5%88%86%E6%9E%90.md##7-%E6%9D%83%E9%87%8D%E8%BF%81%E7%A7%BB-p2p-update)
- [8. 配置参数与环境变量](2026-06-30-SGLANG-EPLB%E5%88%86%E6%9E%90.md##8-%E9%85%8D%E7%BD%AE%E5%8F%82%E6%95%B0%E4%B8%8E%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F)
- [9. 完整端到端流程图](2026-06-30-SGLANG-EPLB%E5%88%86%E6%9E%90.md##9-%E5%AE%8C%E6%95%B4%E7%AB%AF%E5%88%B0%E7%AB%AF%E6%B5%81%E7%A8%8B%E5%9B%BE)
- [10. 源码文件索引](2026-06-30-SGLANG-EPLB%E5%88%86%E6%9E%90.md##10-%E6%BA%90%E7%A0%81%E6%96%87%E4%BB%B6%E7%B4%A2%E5%BC%95)

---

## 1. 架构概览

EPLB 是 SGLang 针对 MoE 模型在 Expert Parallelism (EP) 下的动态负载均衡机制。核心问题：当多个 GPU 各自持有一部分专家权重时，不同专家被选中的频率可能差异很大，导致某些 GPU 过载而其他 GPU 空闲。

**核心思想：** 在物理 GPU 上预分配若干"冗余专家槽位" (`ep_num_redundant_experts`)，根据运行时统计的专家调用频率，动态将这些冗余槽位分配给最繁忙的专家，使负载均衡。

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EPLB 系统架构                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  启动阶段                                                           │
│  ┌─────────────────┐   ┌──────────────────┐   ┌────────────────┐  │
│  │ ExpertLocation  │──▶│  ExpertLocation  │──▶│  ExpertLocation│  │
│  │ Metadata Init   │   │   Recorder Init  │   │    Manager     │  │
│  └─────────────────┘   └──────────────────┘   └───────┬────────┘  │
│                                                        │           │
│  运行阶段 (循环)                                       │           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Forward Pass:                                               │   │
│  │   Router 选专家(逻辑ID) → 转物理ID → EP All-to-All →       │   │
│  │   专家 FFN 计算 → All-to-All 返回 → 录制专家调用统计       │   │
│  └──────────────────────────────┬──────────────────────────────┘   │
│                                 │                                   │
│  每 N 步触发                   ▼                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Rebalancing:                                                │   │
│  │   统计累积 → EPLB 算法计算最优映射 → P2P 权重迁移 →         │   │
│  │   更新物理↔逻辑映射表                                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 关键配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_eplb` | `False` | 总开关 |
| `ep_num_redundant_experts` | `0` | 每 rank 的冗余专家数 |
| `eplb_algorithm` | `"auto"` | 算法选择 |
| `eplb_rebalance_num_iterations` | `1000` | 触发 rebalance 的 forward pass 间隔 |
| `eplb_rebalance_layers_per_chunk` | `None` | 分 chunk 更新层 (None=全量) |
| `eplb_min_rebalancing_utilization_threshold` | `1.0` | 利用率阈值 (高于则跳过 rebalance) |
| `expert_distribution_recorder_mode` | `"stat"` | 统计记录模式 |
| `ep_dispatch_algorithm` | `None` | Dispatch 策略: `static`/`dynamic`/`fake` |
| `init_expert_location` | `"trivial"` | 初始映射: `"trivial"` 或映射文件路径 |

---

## 2. 核心数据结构

### 2.1 ExpertLocationMetadata — 映射关系的载体

**文件:** `python/sglang/srt/eplb/expert_location.py:36-305`

```python
@dataclass
class ExpertLocationMetadata:
    # [layers, num_physical_experts] — 物理槽位 i 上存放的是哪个逻辑专家
    physical_to_logical_map: torch.Tensor
    physical_to_logical_map_cpu: torch.Tensor      # CPU 副本 (快速查询)

    # [layers, num_logical_experts, max_replicas] — 逻辑专家的所有物理副本位置 (-1 = 无效)
    logical_to_all_physical_map: torch.Tensor
    logical_to_all_physical_map_cpu: torch.Tensor
    logical_to_all_physical_map_num_valid: torch.Tensor  # [layers, num_logical_experts]

    # [layers, num_logical_experts] — 静态 dispatch 用的单值映射
    logical_to_rank_dispatch_physical_map: Optional[torch.Tensor]
```

**关键属性:**
```python
num_physical_experts = num_logical_experts + ep_num_redundant_experts
num_local_physical_experts = num_physical_experts // ep_size  # 当前 rank 持有的物理槽位数
```

**示例 (Qwen3.6-35B-A3B):**
```
num_logical_experts = 256
ep_num_redundant_experts = 64
num_physical_experts = 320
ep_size = 2 (2 GPU)
num_local_physical_experts = 160

每个 GPU 存放 160 个物理专家槽位
其中 256/2 = 128 个"基础"槽位 + 32 个"冗余"槽位
冗余槽位可根据负载统计动态分配给最繁忙的专家
```

### 2.2 ExpertLocationDispatchInfo — 每层 dispatch 信息

**文件:** `python/sglang/srt/eplb/expert_location_dispatch.py:24-61`

```python
@dataclass
class ExpertLocationDispatchInfo:
    ep_dispatch_algorithm: Literal["static", "dynamic", "fake"]
    partial_logical_to_rank_dispatch_physical_map: Optional[torch.Tensor]  # (num_logical_experts,)
    partial_logical_to_all_physical_map: torch.Tensor  # (num_logical_experts, X)
    partial_logical_to_all_physical_map_num_valid: torch.Tensor
    num_physical_experts: int
```

### 2.3 ModelConfigForExpertLocation — 模型元信息

**文件:** `python/sglang/srt/eplb/expert_location.py:564-580`

```python
@dataclass
class ModelConfigForExpertLocation:
    num_layers: int
    num_logical_experts: int
    num_groups: Optional[int] = None  # DeepSeek 的 n_group 路由组数
```

---

## 3. 初始化流程

### 3.1 ModelRunner 启动阶段

**文件:** `python/sglang/srt/model_executor/model_runner.py:632-646`

```python
def initialize(self, server_args, model_config):
    # 1. 计算初始专家映射
    set_global_expert_location_metadata(
        compute_initial_expert_location_metadata(server_args, model_config, self.moe_ep_rank)
    )

    # 2. 创建专家负载记录器
    set_global_expert_distribution_recorder(
        ExpertDistributionRecorder.init_new(server_args, metadata, rank=self.tp_rank)
    )

    # 3. 创建 EPLB 管理器 (启动 rebalance 生成器循环)
    self.eplb_manager = EPLBManager(self) if server_args.enable_eplb else None

    # 4. 创建权重更新器
    self.expert_location_updater = ExpertLocationUpdater()
```

### 3.2 初始映射计算

**文件:** `python/sglang/srt/eplb/expert_location.py:583-622`

三种初始化路径:

| init_expert_location | 行为 |
|---------------------|------|
| `"trivial"` | 恒等映射: 逻辑 i → 物理 i，冗余部分循环分配 |
| `.pt` / `.json` 文件 | 加载预计算的映射表 |
| JSON 字符串 | 直接解析映射表 |

`init_trivial()` 实现 (恒等映射):
```python
# 物理槽位 i 对应逻辑专家 i % num_logical_experts
# 如果有冗余槽位，循环分配给 0,1,2,...,num_logical_experts-1,0,1,...
for i in range(num_physical_experts):
    physical_to_logical_map[:, i] = i % num_logical_experts
```

### 3.3 物理专家数与 MoE 层

**文件:** `python/sglang/srt/layers/moe/ep_moe/layer.py:71-279`

MoE 层初始化时，实际创建的专家数包含冗余:
```python
self.experts = get_moe_impl_class(quant_config)(
    num_experts=num_experts_for_moe + server_args.ep_num_redundant_experts,  # 包含冗余
    ...
)
```

所以对于 Qwen3.6-35B-A3B:
```
num_experts_for_moe = 256  (逻辑专家)
ep_num_redundant_experts = 64  (冗余)
实际创建的权重张量维度 = 320 个专家
```

---

## 4. 运行时 Token 路由与 Dispatch

### 4.1 完整执行链

```
MoE Layer.forward()
  │
  ├── 1. Router: hidden → router_logits [num_tokens, num_logical_experts]
  │
  ├── 2. TopK: router_logits → topk_ids [num_tokens, topk]  (逻辑 ID 空间)
  │     └── ExpertDistributionRecorder.on_select_experts(topk_ids)  ← 统计入口
  │
  ├── 3. 逻辑ID → 物理ID 转换:
  │     └── topk_ids_logical_to_physical(topk_ids, dispatch_info)
  │
  ├── 4. EP All-to-All Dispatch:
  │     └── dispatcher.dispatch(hidden_states, topk_ids)
  │
  ├── 5. 专家 FFN 计算
  │
  └── 6. EP All-to-All Combine
```

### 4.2 逻辑 ID → 物理 ID 转换

**文件:** `python/sglang/srt/eplb/expert_location_dispatch.py:76-109`

```python
def topk_ids_logical_to_physical(topk_ids, info):
    if info.ep_dispatch_algorithm == "static":
        # 静态映射: 查表，每个逻辑 ID 固定映射到一个物理 ID
        # 映射表在 rebalance 时预计算，考虑了 GPU 亲和性
        return info.partial_logical_to_rank_dispatch_physical_map[topk_ids]
    elif info.ep_dispatch_algorithm in ["dynamic", "fake"]:
        # 动态映射: 随机选一个物理副本 (分散负载)
        chosen = torch.randint(0, 65536, topk_ids.shape) % num_valid[topk_ids]
        return info.partial_logical_to_all_physical_map[topk_ids, chosen]
```

**三种 Dispatch 模式:**

| 模式 | 实现 | 说明 |
|------|------|------|
| `static` | 直接查表 `logical_to_rank_dispatch_physical_map[topk_ids]` | 预计算，零开销，优先选同 GPU > 同节点 > 跨节点 |
| `dynamic` | 从 `logical_to_all_physical_map` 中随机选一个有效副本 | 运行时随机，更均衡但有方差 |
| `fake` | 将 router logits 均匀化 `router_logits.uniform_(5, 10)` | 仅用于测试 |

### 4.3 EP Token Dispatcher

**文件:** `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`

DeepEP 使用 GPU Direct RDMA 或 NVLink 进行 token 传输:

```python
# Normal 模式
buffer.get_dispatch_layout(topk_ids, ...)  # 计算每个 rank 需要接收的 token 数
buffer.dispatch(x, topk_idx=topk_ids, ...)  # All-to-All 传输

# Low-latency 模式 (更激进的 overlap)
buffer.low_latency_dispatch(x, topk_ids, ...)
```

---

## 5. 专家负载统计与反馈

### 5.1 ExpertDistributionRecorder

**文件:** `python/sglang/srt/eplb/expert_distribution.py:55-280`

统计记录器采用双层架构:

```
ExpertDistributionRecorder
  ├── SinglePassGatherer    (每次 forward pass 的临时计数)
  └── Accumulator           (累积多步的统计量)
```

**四种记录模式:**

| 模式 | Gatherer 类 | 存储内容 | 用途 |
|------|------------|---------|------|
| `stat` | `_SelectExpertsSinglePassGatherer` | GPU tensor 上 `scatter_add_` | **EPLB 标准模式** |
| `stat_approx` | `_DeepepNormalSinglePassGatherer` | DeepEP dispatch 报告的近似计数 | EP 场景 |
| `per_pass` | `_DetailSinglePassGatherer` | 完整 `topk_ids` | 调试 |
| `per_token` | `_DetailSinglePassGatherer` | 每 token 完整路由路径 | 调试/分析 |

### 5.2 统计数据流

```python
# 每步 forward pass:
with recorder.with_forward_pass(pass_id, forward_batch):
    with recorder.with_current_layer(layer_id):
        # 路由选择统计:
        recorder.on_select_experts(topk_ids=topk_ids)  # scatter_add 到 GPU tensor

    # 或者 DeepEP dispatch 统计:
    recorder.on_deepep_dispatch_normal(num_recv_tokens_per_expert, ...)

# 累积:
# _Accumulator 内部维护一个 circular buffer:
# _global_physical_count_of_buffered_step: Tensor [buffer_size, num_layers, num_physical_experts]
```

### 5.3 从物理计数到逻辑计数

Rebalance 时需要将**物理专家的调用计数**转换为**逻辑专家的调用计数**:

```python
# expert_distribution.py:1000-1020
logical_count = scatter_add(
    src=global_physical_count,       # [buffer, layers, num_physical_experts]
    index=physical_to_logical_map,   # [layers, num_physical_experts]
    dim=2,
    out=zeros([buffer, layers, num_logical_experts])
)
# 然后 all_reduce 跨所有 rank 求和
torch.distributed.all_reduce(logical_count, op=ReduceOp.SUM)
```

### 5.4 Utilization Rate (负载均衡度)

```python
# expert_distribution.py:1036-1051
utilization_rate = (mean_per_gpu_count + 1e-5) / (max_per_gpu_count + 1e-5)
# 1.0 = 完美均衡, 1/num_gpus = 最不均衡
```

当 `utilization_rate > eplb_min_rebalancing_utilization_threshold` 时，跳过 rebalance。

---

## 6. EPLB Rebalancing 算法详解

### 6.1 算法选择机制

**文件:** `python/sglang/srt/eplb/eplb_algorithms/__init__.py:1-87`

#### 算法枚举

```python
class EplbAlgorithm(Enum):
    deepseek = auto()                    # DeepSeek 全局算法
    deepseek_hierarchical = auto()       # DeepSeek 层级算法 (多节点)
    deepseek_vec = auto()                # DeepSeek 分块向量化
    deepseek_vec_hierarchical = auto()   # DeepSeek 分块向量化 + 层级
    elasticity_aware = auto()            # 弹性感知 (容错)
    elasticity_aware_hierarchical = auto() # 弹性感知 + 层级
```

#### 自动选择逻辑

```python
def compute_algorithm(raw_algorithm, num_groups, num_nodes):
    if raw_algorithm != "auto":
        return EplbAlgorithm[raw_algorithm]
    # auto: 如果 groups 与 nodes 对齐，用 hierarchical
    if (num_groups is not None) and (num_groups % num_nodes == 0):
        return EplbAlgorithm.deepseek_hierarchical
    else:
        return EplbAlgorithm.deepseek
```

**选择策略:**
- 单节点或 groups 与 nodes 不对齐 → `deepseek` (全局模式)
- 多节点且 groups 与 nodes 对齐 → `deepseek_hierarchical` (层级模式)
- 启用 Elastic EP → 强制使用 `elasticity_aware` 或 `elasticity_aware_hierarchical`

#### 算法调度入口

```python
def rebalance_experts(tokens_per_expert, num_physical_experts, ...):
    if algorithm in [EplbAlgorithm.deepseek, EplbAlgorithm.deepseek_hierarchical]:
        return deepseek.rebalance_experts(weight=tokens_per_expert.sum(dim=0), ...)
    if algorithm in [EplbAlgorithm.deepseek_vec, EplbAlgorithm.deepseek_vec_hierarchical]:
        return deepseek_vec.rebalance_experts(tokens_per_expert=tokens_per_expert, ...)
    if algorithm in [EplbAlgorithm.elasticity_aware, EplbAlgorithm.elasticity_aware_hierarchical]:
        return elasticity_aware.rebalance_experts(weight=tokens_per_expert.sum(dim=0), ..., active_ranks=...)
```

#### 算法对比总览

| 算法 | 输入维度 | 核心策略 | 适用场景 | 复杂度 |
|------|---------|---------|---------|--------|
| `deepseek` | `[layers, num_logical_experts]` | 贪心复制 + 全局装箱 | 单节点 / 通用 | O(R × L × E) |
| `deepseek_hierarchical` | `[layers, num_logical_experts]` | 三步层级: 组→节点→GPU | 多节点且 groups 对齐 | O(R × L × E) |
| `deepseek_vec` | `[steps, layers, num_logical_experts]` | 分块迭代 + zigzag GPU 均衡 | 大规模 / 多步统计 | O(R × L × E / C) |
| `deepseek_vec_hierarchical` | `[steps, layers, num_logical_experts]` | 层级 + 分块迭代 | 大规模多节点 | O(R × L × E / C) |
| `elasticity_aware` | `[layers, num_logical_experts]` | DeepSeek + rank 故障降级 | Elastic EP 容错 | O(R × L × E) |
| `elasticity_aware_hierarchical` | `[layers, num_logical_experts]` | 层级 + rank 故障降级 | Elastic EP 多节点容错 | O(R × L × E) |

> R = num_redundant_experts, L = num_layers, E = num_logical_experts, C = num_chunks

---

### 6.2 算法一: DeepSeek 全局算法 (`deepseek`)

**文件:** `python/sglang/srt/eplb/eplb_algorithms/deepseek.py`

从 [deepseek-ai/EPLB](https://github.com/deepseek-ai/EPLB/blob/main/eplb.py) 移植。

#### 6.2.1 设计思想

**核心问题:** 给定 N 个逻辑专家和 R 个冗余槽位，如何将冗余槽位分配给逻辑专家，使得所有物理槽位的最大负载最小化？

**数学建模:**
- 设逻辑专家 i 的负载为 `w_i`
- 设逻辑专家 i 被复制 `c_i` 次（`c_i ≥ 1`）
- 物理槽位 j 的负载为 `w_{phy2log[j]} / c_{phy2log[j]}`
- **目标:** `minimize max_j (w_{phy2log[j]} / c_{phy2log[j]})`
- **约束:** `Σ c_i = num_physical_experts`, `c_i ≥ 1`

**贪心策略:** 每次选择 `w_i / c_i` 最大的专家增加一个副本，逐步降低最大负载。

#### 6.2.2 核心子算法: `replicate_experts`

```python
def replicate_experts(weight, num_phy):
    """
    将 num_log 个逻辑专家扩展到 num_phy 个物理槽位
    使所有物理槽位的最大负载最小化

    weight: [X, num_log] — 每个逻辑专家的 token 调用次数
    返回: phy2log, rank, logcnt
    """
    n, num_log = weight.shape
    num_redundant = num_phy - num_log

    # 初始: 物理 i = 逻辑 i (identity mapping)
    phy2log = arange(num_phy).repeat(n, 1)   # [n, num_phy]
    logcnt = ones(n, num_log)                 # [n, num_log] 每个逻辑专家的副本数
    rank = zeros(n, num_phy)                  # [n, num_phy] 每个物理槽位是第几个副本

    # 核心: 贪心策略分配冗余槽位
    for i in range(num_log, num_phy):
        # 选择 "负载/副本数" 比值最高的逻辑专家
        target = (weight / logcnt).max(dim=-1).indices
        # 将冗余槽位 i 分配给该逻辑专家
        phy2log[:, i] = target
        rank[:, i] = logcnt[arange(n), target]  # 这是第 logcnt[target] 个副本
        logcnt[arange(n), target] += 1

    return phy2log, rank, logcnt
```

#### 6.2.3 逐步推演示例

```
场景: 4 个逻辑专家, 2 个冗余槽位 → 6 个物理槽位
      2 层 (L=2)

负载矩阵 weight:
  Layer 0: [100, 80, 60, 40]
  Layer 1: [90, 70, 50, 30]

初始状态:
  phy2log = [[0,1,2,3, -, -],
             [0,1,2,3, -, -]]
  logcnt  = [[1,1,1,1],
             [1,1,1,1]]

冗余槽位 4 (i=4):
  Layer 0: weight/logcnt = [100, 80, 60, 40] → max=100 → target=0
  Layer 1: weight/logcnt = [90, 70, 50, 30] → max=90  → target=0
  → phy2log[:,4] = [0, 0]
  → logcnt = [[2,1,1,1], [2,1,1,1]]

冗余槽位 5 (i=5):
  Layer 0: weight/logcnt = [50, 80, 60, 40] → max=80 → target=1
  Layer 1: weight/logcnt = [45, 70, 50, 30] → max=70 → target=1
  → phy2log[:,5] = [1, 1]
  → logcnt = [[2,2,1,1], [2,2,1,1]]

最终结果:
  phy2log = [[0,1,2,3,0,1],
             [0,1,2,3,0,1]]

  物理槽位负载 (每副本):
  Layer 0: [50, 40, 60, 40, 50, 40] → max=60
  Layer 1: [45, 35, 50, 30, 45, 35] → max=50

  vs 无冗余: max=100 (Layer 0 expert0)
  有冗余后: max=60 (Layer 0 expert2) → 降低 40%
```

#### 6.2.4 核心子算法: `balanced_packing`

```python
def balanced_packing(weight, num_packs):
    """
    将 n 个带权重的物品分配到 m 个包中,
    每个包恰好 n/m 个物品, 权重尽可能均衡

    贪心策略: 按权重降序排列, 每次分配给当前最轻的有空位的包

    参数:
      weight: [X, n] — 每个物品的权重
      num_packs: 包的数量 (必须整除 n)
    返回:
      pack_index: [X, n] — 每个物品所属的包
      rank_in_pack: [X, n] — 物品在包内的排名
    """
    num_layers, num_groups = weight.shape
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        # 每个包只放一个物品, 直接返回
        return arange(num_groups).expand(weight.shape), zeros_like(weight)

    # 贪心装箱
    for layer in range(num_layers):
        pack_weights = [0] * num_packs
        pack_items = [0] * num_packs
        # 按权重降序排列
        for group in sorted_indices_by_weight_desc[layer]:
            # 选最轻且未满的包
            pack = min(
                (j for j in range(num_packs) if pack_items[j] < groups_per_pack),
                key=pack_weights.__getitem__,
            )
            pack_index[layer][group] = pack
            rank_in_pack[layer][group] = pack_items[pack]
            pack_weights[pack] += weight[layer][group]
            pack_items[pack] += 1
```

#### 6.2.5 入口函数: `rebalance_experts`

```python
def rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus, enable_hierarchical):
    """
    参数:
      weight: [layers, num_logical_experts] — 负载统计
      num_replicas: 物理专家总数 (= num_logical + num_redundant)
      enable_hierarchical: 是否层级模式
    返回:
      physical_to_logical_map: [layers, num_replicas]
      logical_to_physical_map: [layers, num_logical_experts, max_replicas]
      expert_count: [layers, num_logical_experts] — 每个逻辑专家的副本数
    """
    weight = weight.float().cpu()
    if enable_hierarchical:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus
        )
    else:
        # 全局模式: 等价于 num_groups=1, num_nodes=1 的层级模式
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus
        )
    # 构建 log2phy 反向映射 (用于 dynamic dispatch)
    maxlogcnt = logcnt.max().item()
    log2phy = full((num_layers, num_logical_experts, maxlogcnt), -1)
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * maxlogcnt + phyrank,
        arange(num_replicas).expand(num_layers, -1),
    )
    return phy2log, log2phy, logcnt
```

---

### 6.3 算法二: DeepSeek 层级算法 (`deepseek_hierarchical`)

**文件:** `python/sglang/srt/eplb/eplb_algorithms/deepseek.py:86-168`

#### 6.3.1 设计思想

**核心问题:** 在多节点场景下，跨节点通信（InfiniBand）远慢于节点内通信（NVLink）。如何在保证负载均衡的同时最小化跨节点通信？

**关键洞察:** 专家组 (group) 是 DeepSeek V3 路由的基本单位。如果将整个组分配到同一个节点，可以避免组内跨节点通信。

**三步层级策略:**
1. **组→节点:** 用 `balanced_packing` 将专家组分配到节点，确保每节点 token 数均衡
2. **节点内冗余:** 在每个节点内独立执行 `replicate_experts`，创建冗余专家
3. **物理→GPU:** 用 `balanced_packing` 将节点内物理专家分配到各 GPU，确保 GPU 间均衡

#### 6.3.2 详细实现

```python
def rebalance_experts_hierarchical(weight, num_physical_experts, num_groups, num_nodes, num_gpus):
    """
    参数:
      weight: [num_moe_layers, num_logical_experts]
      num_physical_experts: 物理专家总数
      num_groups: 专家组数 (DeepSeek V3 = 8)
      num_nodes: 节点数
      num_gpus: GPU 总数
    """
    num_layers, num_logical_experts = weight.shape
    group_size = num_logical_experts // num_groups
    groups_per_node = num_groups // num_nodes
    phy_experts_per_gpu = num_physical_experts // num_gpus

    # ======== Step 1: 将 logical expert groups 打包到节点 ========
    # 将逻辑专家按组分组, 计算每组的 token 总数
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    #   shape: [num_layers, num_groups]

    # 贪心装箱: 将 groups 分配到 nodes
    group_pack_index, group_rank_in_pack = balanced_packing(tokens_per_group, num_nodes)
    #   group_pack_index: [num_layers, num_groups] — 每个 group 属于哪个 node
    #   group_rank_in_pack: [num_layers, num_groups] — group 在 node 内的排名

    # 构建 log2mlog: logical index → mapped logical index (按节点重排)
    log2mlog = (
        (group_pack_index * groups_per_node + group_rank_in_pack).unsqueeze(-1)
        + arange(group_size)
    ).flatten(-2)
    #   log2mlog: [num_layers, num_logical_experts]
    #   含义: 逻辑专家 i 在重排后的位置

    mlog2log = inverse(log2mlog)  # 逆映射

    # ======== Step 2: 在每个节点内构造冗余专家 ========
    # 按节点重排后的 token 分布
    tokens_per_mlog = weight.gather(-1, mlog2log).view(-1, num_logical_experts // num_nodes)
    #   shape: [num_layers * num_nodes, num_logical_experts_per_node]

    # 在每个节点内独立执行 replicate_experts
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes
    )
    #   phy2mlog: [num_layers * num_nodes, num_physical_experts_per_node]

    # ======== Step 3: 将物理专家打包到节点内的 GPU ========
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    #   每个物理槽位的负载 = 逻辑专家负载 / 副本数

    pack_index, rank_in_pack = balanced_packing(tokens_per_phy, num_gpus // num_nodes)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    # ======== 坐标变换: 将节点内映射转回全局映射 ========
    pphy2mlog = phy2mlog.gather(-1, pphy2phy)
    pphy2mlog = (
        pphy2mlog.view(num_layers, num_nodes, -1)
        + arange(0, num_logical_experts, num_logical_experts // num_nodes).view(1, -1, 1)
    ).flatten(-2)
    pphy2log = mlog2log.gather(-1, pphy2mlog)
    pphyrank = phyrank.gather(-1, pphy2phy).view(num_layers, -1)
    logcnt = mlogcnt.view(num_layers, -1).gather(-1, log2mlog)

    return pphy2log, pphyrank, logcnt
```

#### 6.3.3 数据流示意

```
Step 1: 组→节点
  原始: [group0, group1, group2, group3] (4 groups, 2 nodes)
  打包: node0=[group0, group2], node1=[group1, group3] (按 token 数均衡)

Step 2: 节点内冗余
  node0: 128 逻辑专家 → 160 物理专家 (32 冗余)
  node1: 128 逻辑专家 → 160 物理专家 (32 冗余)

Step 3: 物理→GPU
  node0: GPU0=[80个], GPU1=[80个] (按负载均衡)
  node1: GPU0=[80个], GPU1=[80个] (按负载均衡)

最终拓扑:
  node0/GPU0: 80 个物理专家 (主要来自 group0, group2)
  node0/GPU1: 80 个物理专家 (主要来自 group0, group2)
  node1/GPU0: 80 个物理专家 (主要来自 group1, group3)
  node1/GPU1: 80 个物理专家 (主要来自 group1, group3)
```

#### 6.3.4 优势分析

| 维度 | 全局模式 | 层级模式 |
|------|---------|---------|
| 跨节点通信 | 可能频繁 | 最小化 (组内不跨节点) |
| 负载均衡精度 | 全局最优 | 分层局部最优 |
| 适用场景 | 单节点 | 多节点 |
| 计算复杂度 | O(R×L×E) | O(R×L×E) (常数因子略高) |

---

### 6.4 算法三: DeepSeek Vec 分块向量化 (`deepseek_vec`)

**文件:** `python/sglang/srt/eplb/eplb_algorithms/deepseek_vec.py:1-276`

#### 6.4.1 设计思想

**核心问题:** 在大规模场景下（数百个逻辑专家、多步统计），需要更精细的负载均衡策略。`deepseek_vec` 引入两个关键改进：

1. **分块处理 (Chunkwise):** 将物理专家分成若干 chunk，每个 chunk 内独立分配冗余，降低计算复杂度
2. **多步统计 (Multi-step):** 利用多个 forward pass 的统计数据（`[num_steps, num_layers, num_logical_experts]`），进行更精确的负载估计
3. **Zigzag GPU 均衡:** 在 GPU 间实现交错分布，避免热点集中

#### 6.4.2 核心函数: `make_redundant_experts_chunkwise`

```python
def make_redundant_experts_chunkwise(
    tokens_per_expert,              # [num_steps, num_moe_layers, num_logical_experts]
    num_physical_experts,
    num_local_physical_experts,
    num_physical_experts_per_chunk, # 每个 chunk 的物理专家数
):
    """
    分块分配冗余专家, 支持多步统计
    """
    num_steps, num_moe_layers, num_logical_experts = tokens_per_expert.shape
    num_redundancy = num_physical_experts - num_logical_experts
    num_chunks = num_physical_experts // num_physical_experts_per_chunk
    num_logical_per_group = num_logical_experts // num_chunks
    num_redundancy_per_group = num_redundancy // num_chunks

    # ======== 初始化: 每个 chunk 内逻辑专家按顺序排列 ========
    # physical_to_logical_map: [num_layers, num_physical_experts]
    #   chunk 0: 物理 0~79 → 逻辑 0~39 (基础) + 冗余
    #   chunk 1: 物理 80~159 → 逻辑 40~79 (基础) + 冗余
    physical_to_logical_map.view(
        num_moe_layers, num_chunks, num_physical_experts_per_chunk
    )[:, :, :num_logical_per_group] = arange(num_logical_per_group).view(num_chunks, -1)

    # ======== 迭代分配冗余槽位 ========
    for i in range(num_redundancy_per_group):
        # 计算 score = tokens / logical_count (考虑多步统计)
        score = tokens_per_expert_all_diff / logical_count
        #   score: [num_steps, num_layers, num_chunks, num_logical_per_group]

        # 对每个 chunk, 选择 score 最高的专家
        values, indices = score.max(-1, keepdim=True)
        #   values: [num_steps, num_layers, num_chunks, 1]

        # 跨步累加 score, 选择总 score 最低的 chunk 中的专家
        redundancy_indices = values.sum(0).argmin(-1)
        #   redundancy_indices: [num_layers, num_chunks]

        # 更新映射
        physical_to_logical_map.view(
            num_moe_layers, num_chunks, num_physical_experts_per_chunk
        )[:, :, num_logical_per_group + i] = (
            redundancy_indices + arange_num_groups * num_logical_per_group
        )

        # 更新 logical_count
        logical_count[arange_num_moe_layers_num_groups, redundancy_indices.view(-1)] += 1

    # ======== GPU 级负载均衡: Zigzag 模式 ========
    if num_local_physical_experts > 1:
        # 计算每个物理槽位的负载
        counts = logical_count.gather(-1, physical_to_logical_map_int64)
        score = tokens_per_expert.sum(0).gather(-1, physical_to_logical_map_int64) / counts
        #   score: [num_layers, num_physical_experts]

        # 按 score 降序排列
        indices = score.view(num_layers, num_chunks, num_physical_experts_per_chunk).argsort(-1, descending=True)

        # Zigzag: 奇数 chunk 翻转顺序, 实现交错分布
        indices[:, :, 1::2, :] = indices[:, :, 1::2, :].flip(-1)
        indices = indices.transpose(2, 3)  # GPU 维度转置
        #   效果: GPU0 拿高负载专家, GPU1 拿低负载专家, 交替分布

        # 应用重排
        physical_to_logical_map = physical_to_logical_map.gather(-1, indices)
```

#### 6.4.3 Zigzag GPU 均衡原理

```
假设 2 个 GPU, 每个 chunk 4 个物理专家:

重排前 (按 score 降序):
  chunk0: [exp0(100), exp1(80), exp2(60), exp3(40)]
  chunk1: [exp4(90), exp5(70), exp6(50), exp7(30)]

Zigzag 重排:
  chunk0 (偶数, 不翻转): [exp0, exp1] [exp2, exp3]
                           GPU0         GPU1
  chunk1 (奇数, 翻转):   [exp7, exp6] [exp5, exp4]
                           GPU0         GPU1

结果:
  GPU0: [exp0(100), exp7(30)] → 平均 65
  GPU1: [exp3(40), exp4(90)]  → 平均 65
  → GPU 间负载均衡
```

#### 6.4.4 入口函数: `rebalance_experts`

```python
def rebalance_experts(tokens_per_expert, ..., enable_hierarchical):
    """
    tokens_per_expert: [num_steps, num_layers, num_logical_experts]
    注意: 与 deepseek 不同, 这里保留了 num_steps 维度
    """
    if enable_hierarchical:
        # prefill 模式: 层级 + 分块
        return prefill_rebalance_experts(
            tokens_per_expert, num_physical_experts, num_local_physical_experts,
            num_groups, num_nodes
        )
    else:
        # decode 模式: 简单分块 (chunk = 全部物理专家)
        return decode_rebalance_experts(
            tokens_per_expert, num_physical_experts, num_local_physical_experts
        )
```

#### 6.4.5 `prefill_rebalance_experts` (层级 + 分块)

```python
def prefill_rebalance_experts(tokens_per_expert, num_physical_experts, num_local_physical_experts, num_groups, num_nodes):
    """
    层级模式: 先将 groups 打包到 nodes, 再在节点内分块分配冗余
    """
    tokens_per_group = tokens_per_expert.sum(0).unflatten(-1, (num_groups, -1)).sum(-1)

    # Step 1: pack groups to nodes
    group_perm = pack_groups(tokens_per_group, num_nodes)

    # Step 2: 构建 log2mlog 映射
    log2mlog = (group_perm * group_size).unsqueeze(-1) + arange(group_size)
    log2mlog = log2mlog.flatten(-2)

    # Step 3: 按节点重排 token 分布
    tokens_per_mlog = tokens_per_expert.gather(2, mlog2log.unsqueeze(0).expand(num_steps, -1, -1))

    # Step 4: 在节点内分块分配冗余
    phy2mlog, mlog2phy, mlog_count = make_redundant_experts_chunkwise(
        tokens_per_mlog, num_physical_experts, num_local_physical_experts,
        num_physical_experts // num_nodes  # chunk 大小 = 节点内物理专家数
    )

    # Step 5: 坐标变换回全局
    phy2log = mlog2log.gather(1, phy2mlog.to(int64))
    log2phy = mlog2phy.gather(1, log2mlog.unsqueeze(-1).expand(-1, -1, mlog2phy.size(-1)))
    log_count = mlog_count.gather(1, log2mlog)
    return phy2log, log2phy, log_count
```

#### 6.4.6 `decode_rebalance_experts` (简单分块)

```python
def decode_rebalance_experts(tokens_per_expert, num_physical_experts, num_local_physical_experts):
    """
    decode 模式: chunk = 全部物理专家 (即不分块)
    """
    return make_redundant_experts_chunkwise(
        tokens_per_expert, num_physical_experts, num_local_physical_experts,
        num_physical_experts  # chunk 大小 = 全部物理专家
    )
```

#### 6.4.7 与 deepseek 的对比

| 维度 | deepseek | deepseek_vec |
|------|----------|-------------|
| 输入维度 | `[L, E]` (求和后) | `[S, L, E]` (保留多步) |
| 冗余分配 | 全局贪心 | 分块贪心 + 跨步累加 |
| GPU 均衡 | 无专门处理 | Zigzag 交错分布 |
| 大规模性能 | O(R×L×E) | O(R×L×E/C) (C=chunk 数) |
| 多步统计 | 不支持 | 支持 (num_steps) |

---

### 6.5 算法四: DeepSeek Vec 层级算法 (`deepseek_vec_hierarchical`)

**文件:** `python/sglang/srt/eplb/eplb_algorithms/deepseek_vec.py:197-252`

#### 6.5.1 设计思想

结合 `deepseek_hierarchical` 的层级策略和 `deepseek_vec` 的分块策略：

1. 先用 `pack_groups` 将专家组打包到节点（同 `deepseek_hierarchical` 的 Step 1）
2. 在节点内用 `make_redundant_experts_chunkwise` 分配冗余（同 `deepseek_vec` 的核心）
3. 最后做坐标变换回全局映射

#### 6.5.2 详细实现

```python
def prefill_rebalance_experts(tokens_per_expert, num_physical_experts, num_local_physical_experts, num_groups, num_nodes):
    """
    完整流程:
    1. pack_groups → 节点分配
    2. log2mlog 坐标变换
    3. make_redundant_experts_chunkwise → 节点内分块冗余
    4. 坐标变换回全局
    """
    tokens_per_group = tokens_per_expert.sum(0).unflatten(-1, (num_groups, -1)).sum(-1)

    # Step 1: pack groups to nodes
    group_perm = pack_groups(tokens_per_group, num_nodes)
    #   group_perm: [num_layers, num_groups] — 每个 group 属于哪个 node

    # Step 2: 构建 log2mlog 映射
    log2mlog = (group_perm * group_size).unsqueeze(-1) + arange(group_size)
    log2mlog = log2mlog.flatten(-2)
    #   log2mlog: [num_layers, num_logical_experts]

    # Step 3: mlog2log (逆映射)
    mlog2log = scatter_inverse(log2mlog)

    # Step 4: 按节点重排 token 分布
    tokens_per_mlog = tokens_per_expert.gather(2, mlog2log.unsqueeze(0).expand(num_steps, -1, -1))
    #   tokens_per_mlog: [num_steps, num_layers, num_logical_per_node]

    # Step 5: 在节点内分块分配冗余
    #   chunk 大小 = num_physical_experts // num_nodes
    phy2mlog, mlog2phy, mlog_count = make_redundant_experts_chunkwise(
        tokens_per_mlog,
        num_physical_experts,
        num_local_physical_experts,
        num_physical_experts // num_nodes,  # 关键: chunk 大小
    )

    # Step 6: 坐标变换回全局
    phy2log = mlog2log.gather(1, phy2mlog.to(int64))
    log2phy = mlog2phy.gather(1, log2mlog.unsqueeze(-1).expand(-1, -1, mlog2phy.size(-1)).to(int64))
    log_count = mlog_count.gather(1, log2mlog)

    return phy2log, log2phy, log_count
```

#### 6.5.3 与 deepseek_hierarchical 的对比

| 维度 | deepseek_hierarchical | deepseek_vec_hierarchical |
|------|----------------------|--------------------------|
| 冗余分配 | `replicate_experts` (全局贪心) | `make_redundant_experts_chunkwise` (分块贪心) |
| GPU 均衡 | `balanced_packing` | Zigzag 交错分布 |
| 多步统计 | 不支持 | 支持 |
| 大规模性能 | O(R×L×E) | O(R×L×E/C) |

---

### 6.6 算法五: 弹性感知算法 (`elasticity_aware`)

**文件:** `python/sglang/srt/eplb/eplb_algorithms/elasticity_aware.py:1-87`

#### 6.6.1 设计思想

**核心问题:** 在 Elastic EP 场景下，GPU rank 可能随时故障或恢复。如何在 rank 故障时仍能正确计算负载均衡？

**关键洞察:**
- rank 故障时，该 rank 持有的专家权重不可用
- 需要将总物理专家数缩减为 `num_local_experts × num_active_ranks`
- 仅使用活跃 rank 计算映射，然后将结果"展开"回原始 rank 维度

#### 6.6.2 详细实现

```python
def rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus, enable_hierarchical, active_ranks):
    """
    参数:
      weight: [num_layers, num_logical_experts]
      active_ranks: [num_gpus] — 1=活跃, 0=故障
    """
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
    num_active_ranks = active_ranks.sum().item()
    num_local_experts = num_replicas // num_gpus

    # ======== 情况 1: 有 rank 故障 ========
    if num_active_ranks < num_gpus:
        # 关键: 将总物理专家数缩减为活跃 rank 能容纳的数量
        # 强制使用全局模式 (num_groups=1, num_nodes=1)
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight,
            num_local_experts * num_active_ranks,  # 缩减后的物理专家数
            1, 1, num_active_ranks,                # 全局模式, 仅活跃 rank
        )
    # ======== 情况 2: 所有 rank 正常 ========
    elif enable_hierarchical:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus
        )
    else:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus
        )

    # ======== 构建 log2phy 反向映射 ========
    maxlogcnt = logcnt.max().item()
    log2phy = full((num_layers, num_logical_experts, maxlogcnt), -1, dtype=int64)
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * maxlogcnt + phyrank,
        arange(num_local_experts * num_active_ranks).expand(num_layers, -1),
    )

    # ======== 关键: 将紧凑映射展开回原始 rank 维度 ========
    if num_active_ranks < num_gpus:
        # 将 phy2log 按 rank 拆分
        phy2log_slices = list(phy2log.view(num_layers, num_active_ranks, -1).unbind(dim=1))

        # 遍历所有 rank, 为故障 rank 插入零张量占位
        active_ranks_list = active_ranks.tolist()
        for idx, active_rank in enumerate(active_ranks_list):
            if not active_rank:
                # 插入全零占位 (故障 rank 不持有任何专家)
                phy2log_slices.insert(idx, zeros_like(phy2log_slices[0]))
                # 调整 log2phy 中的索引偏移
                log2phy = torch.where(
                    log2phy >= idx * num_local_experts,
                    log2phy + num_local_experts,  # 向后偏移
                    log2phy,
                )

        # 重新拼接为完整的 [num_layers, num_gpus, num_local_experts] 形状
        phy2log = torch.stack(phy2log_slices, dim=1).contiguous().view(num_layers, -1)

    return phy2log, log2phy, logcnt
```

#### 6.6.3 故障场景推演

```
场景: 4 个 GPU, 2 个故障 (GPU 2, GPU 3)
      num_local_experts = 80, num_logical_experts = 256

active_ranks = [1, 1, 0, 0]
num_active_ranks = 2

Step 1: 计算紧凑映射
  总物理专家数 = 80 × 2 = 160 (仅活跃 rank)
  执行 rebalance_experts_hierarchical(weight, 160, 1, 1, 2)
  → phy2log_compact: [num_layers, 160]

Step 2: 构建 log2phy
  log2phy: [num_layers, 256, max_replicas]
  索引范围: [0, 159]

Step 3: 展开回原始维度
  phy2log_slices = [slice_gpu0(80个), slice_gpu1(80个)]
  遍历 active_ranks = [1, 1, 0, 0]:
    idx=0, active=1 → 不操作
    idx=1, active=1 → 不操作
    idx=2, active=0 → 插入 zeros(80), log2phy += 80 (偏移)
    idx=3, active=0 → 插入 zeros(80), log2phy += 80 (偏移)

  最终 phy2log: [num_layers, 320]
    GPU0: 活跃专家 0~79
    GPU1: 活跃专家 80~159
    GPU2: 全零 (故障)
    GPU3: 全零 (故障)

  log2phy 索引调整后:
    物理槽位 0~79 → GPU0
    物理槽位 80~159 → GPU1
    物理槽位 160~239 → GPU2 (故障, 不会被选中)
    物理槽位 240~319 → GPU3 (故障, 不会被选中)
```

#### 6.6.4 与 Elastic EP 的集成

```python
# 在 __init__.py 中:
if algorithm in [EplbAlgorithm.elasticity_aware, EplbAlgorithm.elasticity_aware_hierarchical]:
    from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
    return elasticity_aware.rebalance_experts(
        ...,
        active_ranks=(
            ElasticEPStateManager.instance().active_ranks
            if ElasticEPStateManager.instance() is not None
            else ElasticEPStateManager.healthy_rank_state()  # 全部健康
        ),
    )
```

#### 6.6.5 Rank 恢复处理

```python
# model_runner.py:3621-3622
# 当 rank 恢复时, 立即触发 rebalance
if elastic_ep_backend is not None and self.eplb_manager is not None:
    # rank 恢复后, 重新计算映射
    self.eplb_manager.rebalance()
    # 广播新的元数据到所有 rank
    broadcast_global_expert_location_metadata(src_rank=0)
```

---

### 6.7 算法六: 弹性感知层级算法 (`elasticity_aware_hierarchical`)

**文件:** `python/sglang/srt/eplb/eplb_algorithms/elasticity_aware.py` (复用同一文件)

#### 6.7.1 设计思想

与 `elasticity_aware` 完全相同的容错逻辑，但在所有 rank 正常时使用层级模式（而非全局模式）。

```python
def rebalance_experts(..., enable_hierarchical, active_ranks):
    num_active_ranks = active_ranks.sum().item()

    if num_active_ranks < num_gpus:
        # 有故障: 强制全局模式 (与 elasticity_aware 相同)
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_local_experts * num_active_ranks, 1, 1, num_active_ranks
        )
    elif enable_hierarchical:
        # 无故障 + 层级模式 (区别在这里!)
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus
        )
    else:
        # 无故障 + 全局模式
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus
        )
    ...
```

#### 6.7.2 与 `elasticity_aware` 的唯一区别

| 条件 | elasticity_aware | elasticity_aware_hierarchical |
|------|-----------------|------------------------------|
| 所有 rank 正常 | `enable_hierarchical` 参数决定 | 强制层级模式 |
| 有 rank 故障 | 强制全局模式 | 强制全局模式 |

---

### 6.8 算法选择决策树

```
用户指定 --eplb-algorithm?
  │
  ├── 是 → 使用指定算法
  │
  └── 否 ("auto") →
        │
        ├── 启用 Elastic EP? ──是──→ elasticity_aware_hierarchical
        │                              (或 elasticity_aware)
        │
        └── 否 →
              │
              ├── num_groups % num_nodes == 0? ──是──→ deepseek_hierarchical
              │
              └── 否 → deepseek
```

### 6.9 算法性能对比

| 算法 | 适用规模 | 负载均衡质量 | 计算开销 | 通信优化 |
|------|---------|------------|---------|---------|
| `deepseek` | 通用 | 优 | 低 | 无 |
| `deepseek_hierarchical` | 多节点 | 优 | 低 | 跨节点最小化 |
| `deepseek_vec` | 大规模 | 极优 | 中 | Zigzag GPU 均衡 |
| `deepseek_vec_hierarchical` | 大规模多节点 | 极优 | 中 | 跨节点 + Zigzag |
| `elasticity_aware` | 容错场景 | 良 | 低 | 无 |
| `elasticity_aware_hierarchical` | 容错多节点 | 良 | 低 | 跨节点最小化 |

---

## 7. 权重迁移 (P2P Update)

### 7.1 ExpertLocationUpdater

**文件:** `python/sglang/srt/eplb/expert_location_updater.py:37-618`

当 rebalance 计算出新的物理→逻辑映射后，需要将 GPU 上的专家权重张量迁移到正确位置。

### 7.2 五种更新场景

对每个目标物理槽位 `dst_phy`:

| 场景 | 条件 | 操作 |
|------|------|------|
| **1. Unchanged** | `old_map[dst] == new_map[dst]` | 无操作 |
| **2. Same-GPU** | 逻辑专家在同 GPU 的另一个物理槽位 | GPU 内本地拷贝 (memcpy) |
| **3. Free-rider** | 另一个目标槽位已经加载了相同的权重 | 复用已有拷贝 |
| **4. Same-node P2P** | 逻辑专家在同节点的另一个 GPU 上 | NVLink P2P recv |
| **5. Cross-node P2P** | 逻辑专家在不同节点的 GPU 上 | InfiniBand P2P recv |

### 7.3 P2P 通信模式

```python
# 对每个源物理槽位, 创建 isend 操作:
for src_phy in source_experts_of_this_logical:
    for dst_rank in all_ranks_needing_this_expert:
        p2p_ops.append(P2POp(isend, weight[src_phy], dst_rank))

# 对每个目标物理槽位, 创建 irecv 操作:
for dst_phy in destination_experts:
    src_rank = find_source_rank(dst_phy)
    p2p_ops.append(P2POp(irecv, buffer[dst_phy], src_rank))

# 批量执行:
torch.distributed.batch_isend_irecv(p2p_ops)
for req in reqs:
    req.wait()
```

### 7.4 Load-Balanced P2P 分发

`_ChunkUtils` 类确保当多个目标 rank 需要同一个源 rank 的权重时，通过分块 (chunk) 方式将传输负载分散到多个源上:

```python
# 逻辑: 如果源 rank A 需要向 rank B 和 rank C 各发送一个专家,
# 则让 A 发送前半部分给 B, 后半部分给 C (而非全部发给 B)

class _ChunkUtils:
    def chunk_value_from_element_value(self, element_value):
        # 将 element_value 映射到对应的 chunk_value
        chunk_index = self._chunk_index_from_element_index(...)
        return self.chunk_values[chunk_index]

    def element_values_from_chunk_value(self, chunk_value):
        # 返回该 chunk 需要处理的所有 element_values
        element_slice = self._element_slice_from_chunk_index(...)
        return self.element_values[element_slice]
```

### 7.5 Canary 验证

当环境变量 `SGLANG_EXPERT_LOCATION_UPDATER_CANARY=True` 时，启用金丝雀验证:

```python
# 在权重列表末尾追加一个包含旧映射值的特殊张量
canary_tensor = old_physical_to_logical_map[local_range].clone()
routed_experts_weights_of_layer[layer_id].append(canary_tensor)

# 执行 P2P 传输后，验证金丝雀值是否等于期望的新映射值
expect_value = new_physical_to_logical_map[local_range]
actual_value = routed_experts_weights_of_layer[layer_id][-1].cpu()
assert torch.all(expect_value == actual_value)
```

---

## 8. 配置参数与环境变量

### 8.1 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable-eplb` | `False` | EPLB 总开关 |
| `--eplb-algorithm` | `"auto"` | 算法: `auto`/`deepseek`/`deepseek_hierarchical`/`deepseek_vec`/`deepseek_vec_hierarchical`/`elasticity_aware`/`elasticity_aware_hierarchical` |
| `--eplb-rebalance-num-iterations` | `1000` | 两次 rebalance 间的 forward 次数 |
| `--eplb-rebalance-layers-per-chunk` | `None` | 分 chunk 更新的 layer 数 (None=全量) |
| `--eplb-min-rebalancing-utilization-threshold` | `1.0` | 利用率阈值，超过则跳过 rebalance |
| `--ep-num-redundant-experts` | `0` | 每 rank 的冗余 expert 数 |
| `--ep-dispatch-algorithm` | `None` | Dispatch 策略: `static`/`dynamic`/`fake` |
| `--init-expert-location` | `"trivial"` | 初始映射: `"trivial"` 或映射文件路径 |
| `--expert-distribution-recorder-mode` | `None` | 记录模式: `stat`/`stat_approx`/`per_pass`/`per_token` |
| `--expert-distribution-recorder-buffer-size` | auto | 循环缓冲区大小 |
| `--enable-expert-distribution-metrics` | `False` | 启用 Prometheus 指标 |

### 8.2 自动推断逻辑

当 `--enable-eplb` 设置时:
1. `expert_distribution_recorder_mode` 自动设为 `"stat"` (如未指定)
2. `ep_dispatch_algorithm` 自动设为 `"static"` (如未指定)
3. 启用 Elastic EP 时，`eplb_algorithm` 强制为 `elasticity_aware` 或 `elasticity_aware_hierarchical`

### 8.3 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SGLANG_EXPERT_LOCATION_UPDATER_LOG_INPUT` | `False` | 记录 P2P updater 输入数据 |
| `SGLANG_EXPERT_LOCATION_UPDATER_CANARY` | `False` | 启用金丝雀验证 |
| `SGLANG_EXPERT_LOCATION_UPDATER_LOG_METRICS` | `False` | 记录 P2P 通信指标 (每 GPU/节点的字节数) |
| `SGLANG_LOG_EXPERT_LOCATION_METADATA` | `False` | 记录启动时的初始元数据 |
| `SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR` | `/tmp` | 分布记录保存目录 |
| `SGLANG_EPLB_HEATMAP_COLLECTION_INTERVAL` | `0` | Prometheus heatmap 收集间隔 (0=禁用) |
| `SGLANG_ENABLE_EPLB_BALANCEDNESS_METRIC` | `False` | 将 balancedness 暴露为 Prometheus 指标 |

---

## 9. 完整端到端流程图

```
时间线 ──────────────────────────────────────────────────────────────────▶

┌─ 启动阶段 ─────────────────────────────────────────────────────────┐
│                                                                     │
│  compute_initial_expert_location_metadata()                        │
│    └─ trivial: identity mapping (逻辑 i → 物理 i)                  │
│                                                                     │
│  ExpertDistributionRecorder.init_new()                             │
│    └─ 创建 circular buffer, 开始记录                               │
│                                                                     │
│  EPLBManager.__init__()                                             │
│    └─ 创建 generator, 每 N 步 yield 一次                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ Forward Pass #1 ─────────────────────────────────────────────────┐
│                                                                    │
│  ┌─ Router ─────────────────────────────────────────────────────┐  │
│  │  hidden[1, 2048] × gate_weight[2048, 256] → logits[1, 256] │  │
│  │  topk(logits, k=8) → topk_ids[1, 8]  (逻辑 ID)            │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│  ┌─ ID 转换 ──────────────────────────────────────────────────┐   │
│  │  topk_ids_logical_to_physical(topk_ids, dispatch_info)     │   │
│  │  静态: 查表 → 直接映射到物理 ID                            │   │
│  │  动态: 从副本中随机选一个                                  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│  ┌─ EP All-to-All Dispatch ─────────────────────────────────────┐  │
│  │  DeepEP: GPU Direct RDMA / NVLink 传输 token 到目标 GPU      │  │
│  │  Recorder: on_select_experts() 记录每个物理专家的 token 数    │  │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│  ┌─ Expert FFN ─────────────────────────────────────────────────┐  │
│  │  对每个收到的 token 执行专家前向:                            │  │
│  │  hidden → w13_gemm → silu_and_mul → w2_gemm → sum_reduce   │  │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│  ┌─ EP All-to-All Combine ──────────────────────────────────────┐  │
│  │  DeepEP: 将计算结果发回原始 GPU                              │  │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│  EPLBManager.on_forward_pass_end()  ← generator advance            │
│  (如果达到 N 步, 触发 rebalance)                                    │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ Rebalancing (每 N 步) ──────────────────────────────────────────┐
│                                                                    │
│  1. dump_record()                                                  │
│     └─ physical_count → logical_count (scatter_add + all_reduce)  │
│                                                                    │
│  2. _check_rebalance_needed()                                      │
│     └─ if utilization_rate > threshold: skip                      │
│                                                                    │
│  3. ExpertLocationMetadata.init_by_eplb()                          │
│     └─ 执行算法 (以 deepseek_hierarchical 为例):                  │
│        a. balanced_packing(groups → nodes)                        │
│        b. replicate_experts(分配冗余槽位)                          │
│        c. balanced_packing(physical → GPUs)                       │
│        → 返回新的 physical_to_logical_map                          │
│                                                                    │
│  4. ExpertLocationUpdater.update()                                 │
│     └─ P2P 权重迁移:                                              │
│        - Unchanged: 无操作                                         │
│        - Same-GPU: 本地拷贝                                       │
│        - Same-node: NVLink P2P                                    │
│        - Cross-node: InfiniBand P2P                               │
│                                                                    │
│  5. ExpertLocationMetadata.update()                                │
│     └─ 更新全局映射表, 新映射即时生效                              │
│                                                                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 10. 源码文件索引

| 文件 | 核心函数/类 | 行号 |
|------|------------|------|
| `eplb_manager.py` | `EPLBManager`, `rebalance()`, `_entrypoint()` | 16-121 |
| `expert_location.py` | `ExpertLocationMetadata`, `init_trivial()`, `init_by_eplb()`, `update()` | 36-622 |
| `expert_location_dispatch.py` | `ExpertLocationDispatchInfo`, `topk_ids_logical_to_physical()` | 24-109 |
| `expert_distribution.py` | `ExpertDistributionRecorder`, `_StatAccumulator.dump()`, `_on_forward_pass_end()` | 55-1051 |
| `expert_location_updater.py` | `ExpertLocationUpdater`, `update_expert_weights_single_layer()`, P2P ops | 37-618 |
| `eplb_algorithms/__init__.py` | `compute_algorithm()`, `EplbAlgorithm` enum, `rebalance_experts()` | 1-87 |
| `eplb_algorithms/deepseek.py` | `replicate_experts()`, `rebalance_experts()`, `balanced_packing()`, `rebalance_experts_hierarchical()` | 7-224 |
| `eplb_algorithms/deepseek_vec.py` | `make_redundant_experts_chunkwise()`, `prefill_rebalance_experts()`, `decode_rebalance_experts()` | 35-276 |
| `eplb_algorithms/elasticity_aware.py` | `rebalance_experts()` with `active_ranks` | 1-87 |
| `layers/moe/ep_moe/layer.py` | `DeepEPMoE.forward_impl()` | 71-279 |
| `layers/moe/token_dispatcher/deepep.py` | `_DeepEPDispatcherImplNormal._dispatch_core()` | 442-502 |
| `layers/moe/topk.py` | `select_experts()`, `_biased_grouped_topk_postprocess()` | 996, 1362 |
| `model_executor/model_runner.py` | `initialize()`, `update_expert_location()`, `on_forward_pass_end()` | 632-646, 1648-1720 |
| `models/deepseek_v2.py` | `get_model_config_for_expert_location()`, `routed_experts_weights_of_layer` | 2385-2527 |

---

## 11. 设计亮点总结

1. **生成器模式:** `EPLBManager` 用 Python generator 精确控制 rebalance 时机，不阻塞 forward pass
2. **分层更新:** `eplb_rebalance_layers_per_chunk` 支持将权重迁移分摊到多个 forward pass，避免大延迟尖峰
3. **5-case P2P 优化:** 避免不必要的通信——同 GPU 本地拷贝、Free-rider 复用、同节点优先
4. **算法可扩展:** 通过 `EplbAlgorithm` 枚举注册新算法，现有 6 种变体覆盖从单机到弹性多机场景
5. **统计反馈闭环:** 从 token 路由统计 → 物理计数 → 逻辑计数 → 算法计算 → 权重迁移 → 映射更新，形成完整的自适应循环
