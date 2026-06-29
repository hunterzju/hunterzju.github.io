---
title: SGLANG_MULTI_GPU
categories:
  - LLM
  - DeepLearning
tags:
  - LLM
  - Transformer
  - sglang
share: true
---

# sglang 多 GPU 并行推理分析 (以 Qwen3.5-35B-A3B 为例)

基于 sglang 源码 (`/home/hunter/workspace/sglang/python/sglang/srt/`) 分析。

---

## 一、并行策略总览

sglang 支持**三层并行**的任意组合，通过 `initialize_model_parallel` 统一初始化：

```
                    ┌─────────────────────────────────────┐
                    │         世界进程组 (WORLD)            │
                    │         World size = N GPU           │
                    └──────────┬──────────────────────────┘
                               │
              ┌────────────────┼──────────────────┐
              ▼                ▼                   ▼
      ┌────────────┐  ┌──────────────┐  ┌────────────────┐
      │  TP Group   │  │   EP Group   │  │   PP Group     │
      │ 张量并行组   │  │  专家并行组   │  │ 流水线并行组    │
      │ size=tp_size │  │ size=ep_size │  │ size=pp_size   │
      └────────────┘  └──────────────┘  └────────────────┘
```

**约束:** `tp_size × ep_size × pp_size = world_size` (不含 DP 时)

**Qwen3.5-35B-A3B 典型部署 (8×GPU 示例):**

| 配置 | tp | ep | pp | 总 GPU | 说明 |
|------|----|----|----|--------|------|
| single GPU | 1 | 1 | 1 | 1 | 单卡推理 |
| TP only | 8 | 1 | 1 | 8 | 纯张量并行 |
| TP+EP | 4 | 2 | 1 | 8 | 常见 MoE 部署 |
| TP+EP+PP | 2 | 2 | 2 | 8 | 兼顾长序列 |

---

## 二、进程组初始化

### 2.1 核心入口

`parallel_state.py:1755` — `initialize_model_parallel(tensor_model_parallel_size, expert_model_parallel_size, pipeline_model_parallel_size, ...)`

### 2.2 各组创建逻辑

以 **8 GPU (tp=4, ep=2, pp=1)** 为例，rank 编号 `g0..g7`:

```
TP Group (size=4):     [g0,g1,g2,g3], [g4,g5,g6,g7]     # 张量切分
EP Group (size=2):     [g0,g2], [g1,g3], [g4,g6], [g5,g7]  # 专家分布
MoE_TP Group (size=2): [g0,g1], [g2,g3], [g4,g5], [g6,g7]  # MoE 内 TP
MoE_DP Group (size=2): [g0,g4], [g1,g5], [g2,g6], [g3,g7]  # MoE 数据并行
PP Group (size=1):     每个 GPU 单独一组                    # 流水线
```

**分组公式 (`parallel_state.py:1755-2040`):**

```
# TP 组: 连续 tp_size 个 rank 为一组
for tp_group_idx in range(num_tp_groups):
    ranks = range(tp_group_idx * tp_size, (tp_group_idx + 1) * tp_size)

# EP 组: 按 stride=moe_tp 间隔取 rank
for tp_group_idx, moe_dp_idx, moe_tp_idx:
    st = tp*idx + dp*ep*tp + tp_idx
    ranks = range(st, st + ep*tp, tp_idx)

# MoE_TP 组: 连续 moe_tp 个 rank 为一组
for tp_group_idx, ep_dp_combined_idx:
    st = tp*idx + ep_dp_idx * moe_tp
    ranks = range(st, st + moe_tp)

# PP 组: 按 stride=pp_size 间隔取 rank
for pp_group_idx:
    ranks = range(pp_idx, world_size, num_pp_groups)
```

### 2.3 GroupCoordinator

`parallel_state.py:199` — 每个通信组被封装为 `GroupCoordinator`，内部包含:

```python
class GroupCoordinator:
    device_group: ProcessGroup    # NCCL 通信组 (GPU-GPU)
    cpu_group: ProcessGroup       # Gloo 通信组 (CPU 控制)
    pynccl_comm: PyNcclCommunicator  # 自定义 PyNccl (CUDA graph 友好)
    ca_comm: CustomAllReduce      # 自定义 all-reduce (小张量优化)
```

---

## 三、张量并行 (TP) — 模型权重的切分

### 3.1 权重切分类型

sglang 定义了两类 TP-aware 线性层 (`linear.py`):

| 层类型 | 切分方式 | forward 输出 | 通信 |
|--------|---------|-------------|------|
| `ColumnParallelLinear` | 沿 output_dim 切分: `W → [W_0..W_p]` | 每卡持有部分输出 | 可选 `all_gather` |
| `RowParallelLinear` | 沿 input_dim 切分: `W → [W_0; ..; W_p]` | 每卡计算部分 → `all_reduce` 合并 | 必须 all_reduce |
| `QKVParallelLinear` | 按 head 维度切分 Q/K/V | 每卡持有部分头 | 可选 |
| `VocabParallelEmbedding` | 沿 vocab 维切分 | token lookup → `all_reduce` 合并 | 必须 all_reduce |
| `ParallelLMHead` | 沿 vocab 维切分 | 输出部分 logits | sampler 中 all-gather |

### 3.2 Qwen3.5 各组件 TP 切分细则

#### Embedding (`VocabParallelEmbedding`, `vocab_parallel_embedding.py:163`)

```python
class VocabParallelEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, ...):
        # 词表补齐到 64 的倍数
        self.org_vocab_size_padded = pad_vocab_size(org_vocab_size, 64)
        self.num_embeddings_per_partition = padded_vocab_size // tp_size

    def forward(self, input_):
        if tp_size > 1:
            # 只查属于自己的词表段落，其余 token 置 0
            masked_input, input_mask = get_masked_input_and_mask(...)
        output_parallel = F.embedding(masked_input, weight)
        if tp_size > 1:
            output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
            output_parallel = tensor_model_parallel_all_reduce(output_parallel)
        return output_parallel
```

**TP=4 时切分 (vocab_size=248320, padded=248384):**
```
GPU0: weight [62096, 2048]  tokens 0~62095
GPU1: weight [62096, 2048]  tokens 62096~124191
GPU2: weight [62096, 2048]  tokens 124192~186287
GPU3: weight [62096, 2048]  tokens 186288~248383 (含 padding)
```

#### Attention QKV (`QKVParallelLinear`)

QKV 按 head 维度切分:

```
Q: [4096, 2048]  →  每卡 [1024, 2048]   (16 heads ÷ 4 = 4 heads)
K: [512, 2048]   →  每卡 [128, 2048]    (2 heads ÷ 4 -> 每个 head 是 256 dim, 每卡 1/4 即 128 dim -> 实际为 ceil,需要按 num_kv_heads 整除)
V: [512, 2048]   →  每卡 [128, 2048]
O: [2048, 4096]  →  RowParallel, input 切分 → all_reduce 合并
```

**TP=4, num_key_value_heads=2 时的处理:**
K/V 头的数量可能小于 tp_size。sglang 的策略是:
- 如果 `num_kv_heads % tp_size != 0`, KV heads 会**复制**（replicate）而非切分
- Q heads 始终按 tp_size 整除切分（Qwen3.5: 16 heads / 4 = 4 heads/GPU）
- K/V heads 数量少时采用 broadcast/replicate 策略

#### MoE Gate/Up/Down (`FusedMoE`, `fused_moe_triton/layer.py:141`)

```
gate_proj: [512, 2048]  →  ColumnParallel, 每卡 [128, 2048] (TP=4)
up_proj:   [512, 2048]  →  ColumnParallel, 每卡 [128, 2048]
down_proj: [2048, 512]  →  RowParallel, 每卡输入 128 维, all_reduce 合并
```

#### LM Head (`ParallelLMHead`, `vocab_parallel_embedding.py:514`)

继承自 `VocabParallelEmbedding`，但与 `embed_tokens` 权重解耦 (`tie_word_embeddings=false`)。

```python
class ParallelLMHead(VocabParallelEmbedding):
    def forward(self, input_):
        raise RuntimeError("LMHead's weights should be used in the sampler.")
```

LM Head 的权重在 `Sampler` 中被直接用于 `torch.nn.functional.linear(hidden_states, weight.T)`，然后在 TP 组内 `all-gather` 得到完整 logits。

### 3.3 TP 前向综合通信流

```
输入 tokens [B, L]
  │
  ├─ Embedding: 每卡查局部表 → all_reduce → [B, L, 2048]
  │
  ├─ [每层 Transformer]
  │   │
  │   ├─ QKV Projection: ColumnParallel → 每卡 [B, L, local_qkv]
  │   │   (Attention 计算在 local heads 上, 不需要通信)
  │   │
  │   ├─ Output Projection (O): RowParallel → all_reduce → [B, L, 2048]
  │   │   (如果是 Mamba2, out_proj 同理)
  │   │
  │   ├─ MoE 部分:
  │   │   ├─ gate (Router): ReplicatedLinear → 每卡拿到相同的 routing 结果
  │   │   ├─ gate_proj/up_proj: ColumnParallel → 每卡 [B, L, 128] (TP=4)
  │   │   ├─ 专家计算: 在 local expert 上执行
  │   │   ├─ down_proj: RowParallel → all_reduce → [B, L, 2048]
  │   │
  │   ├─ 残差连接 + LayerNorm: 不需要通信
  │
  ├─ Final Norm: 不需要通信
  │
  └─ LM Head: ColumnParallel → all-gather 合并 logits → sampler
```

---

## 四、专家并行 (EP) — MoE Expert 的分布

### 4.1 Expert 分组策略

当 `ep_size > 1` 时，256 个 expert 被均匀分布到 EP 组内的各 GPU 上:

```
Qwen3.5: num_experts=256, ep_size=2
  GPU0 (EP rank 0): experts 0~127      (128 experts)
  GPU1 (EP rank 1): experts 128~255    (128 experts)

Qwen3.5: num_experts=256, ep_size=4
  GPU0: experts 0~63
  GPU1: experts 64~127
  GPU2: experts 128~191
  GPU3: experts 192~255
```

### 4.2 Token 分发 — All-to-All 通信

`FusedMoE` 的 `forward_impl` (`fused_moe_triton/layer.py:1064`):

```python
def forward_impl(self, hidden_states, topk_output):
    # 1) 路由: 选择 top-8 experts → topk_ids [B*L, num_experts_per_tok]
    dispatch_output = self.dispatcher.dispatch(hidden_states, topk_output)
    
    # 2) MoE 核心计算: 在 local expert 上执行
    combine_input = self.run_moe_core(dispatch_output)
    
    # 3) Combine: 收集所有 expert 的输出
    final_hidden_states = self.dispatcher.combine(combine_input)
    
    # 4) 如果 TP>1 或 EP>1, 需要 all-reduce
    if self.reduce_results and (self.moe_tp_size > 1 or self.moe_ep_size > 1):
        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
    return final_hidden_states
```

**Token 分发的三种后端 (`create_moe_dispatcher`, `fused_moe_triton/layer.py:83`):**

| 后端 | 通信方式 | 适用场景 |
|------|---------|---------|
| `StandardDispatcher` | `all_gatherv` + 本地 scatter | 标准 EP，非融合路径 |
| `FlashinferDispatcher` | FlashInfer 内置 all-to-all | 融合 cutlass MoE kernel |
| `DeepEPDispatcher` | DeepEP 专用通信库 | 高性能 EP (NVLink) |

**StandardDispatcher 的分发流程 (`standard.py:115`):**

```python
def dispatch(self, hidden_states, topk_output):
    # Step 1: All-gather tokens 到所有 EP rank (通过 TP group)
    # 输入: [local_tokens, hidden_size]
    # 输出: [total_tokens, hidden_size] (所有 DP rank 的 token 拼接)
    hidden_states = get_tp_group().all_gatherv(hidden_states, sizes=dp_global_num_tokens)
    
    # Step 2: 根据路由结果，将 token 分发到对应的 expert 所在 GPU
    # 使用 local_expert_mapping 确定哪些 token 去本地 GPU
    for each expert i assigned to this GPU:
        expert_mask = (topk_ids == local_expert_id)
        # 提取需要该 expert 计算的 token
        expert_input = hidden_states[expert_mask]

    # Step 3: 在 local expert 上执行 gate_proj → SiLU → up_proj → down_proj
    ...

    # Combine 阶段: 反向操作，将结果收集回原 token 位置
```

### 4.3 EP 下的 MoE_TP 协同

当 **TP 和 EP 同时启用**时，MoE 部分的计算分为 `moe_tp_size` 和 `moe_ep_size`:

```
8 GPU 配置: moe_tp_size=2, moe_ep_size=4, moe_dp_size=1

GPU 编号: g0 g1 | g2 g3 | g4 g5 | g6 g7
         EP=0  EP=0  EP=1  EP=1  EP=2  EP=2  EP=3  EP=3
         └─TP─┘  └─TP─┘  └─TP─┘  └─TP─┘
         
每个 EP 组内:
  - g0/expert0-w13 和 g1/expert0-w13 做 column-parallel (TP 切分 intermediate)
  - 结果在 g0-g1 之间需要 all-reduce 才能拿到完整输出
```

**`FusedMoE` 初始化中的 expert 映射 (`fused_moe_triton/layer.py:214-217`):**

```python
self._num_global_routed = num_experts - num_shared_slots   # 256 routed experts
self._num_local_routed = self._num_global_routed // self.moe_ep_size  # 每卡 expert 数
self.num_local_experts = self._num_local_routed + num_fused_shared_experts
```

---

## 五、流水线并行 (PP) — 层切分

### 5.1 层分配

`Qwen3_5ForCausalLM.__init__` (`qwen3_5.py:1004`):

```python
self.layers, self._start_layer, self._end_layer = make_layers(
    config.num_hidden_layers,
    get_layer,
    pp_rank=self.pp_group.rank_in_group,
    pp_size=self.pp_group.world_size,
)
```

每层 Transformer 通过 `make_layers` 函数均分到 pp_size 个 GPU:

```
PP=2, 40 层:
  GPU0 (PP rank 0): layers 0~19
  GPU1 (PP rank 1): layers 20~39
```

### 5.2 PP 前向传递

`forward` (`qwen3_5.py:1079`):

```python
def forward(self, input_ids, positions, forward_batch, ...):
    if self.pp_group.is_first_rank:
        hidden_states = self.embed_tokens(input_ids)   # 第一卡: embedding
        residual = None
    else:
        hidden_states = pp_proxy_tensors["hidden_states"]  # 后续卡: 接收上一卡输出
        residual = pp_proxy_tensors["residual"]
    
    for layer_idx in range(self.start_layer, self.end_layer):
        hidden_states, residual = self.layers[layer_idx](...)
    
    if self.pp_group.is_last_rank:
        hidden_states = self.norm(hidden_states)  # 最后一卡: final norm + lm_head
    return hidden_states
```

PP 通信通过 `pp_proxy_tensors` 在相邻 PP rank 之间传递 hidden states（使用 p2p send/recv）。

### 5.3 PP 下的 Embedding/LMHead 分布

```python
# 只有 PP 第一卡持有 embedding
if self.pp_group.is_first_rank:
    self.embed_tokens = VocabParallelEmbedding(...)
else:
    self.embed_tokens = PPMissingLayer()

# 只有 PP 最后一卡持有 norm + lm_head
if self.pp_group.is_last_rank:
    self.norm = GemmaRMSNorm(...)
else:
    self.norm = PPMissingLayer()

# LM Head 通过 get_embed_and_head() / set_embed_and_head() 传递
```

---

## 六、完整前向传播示例 (8 GPU: tp=4, ep=2, pp=1)

### 6.1 通信拓扑

```
GPU 0 ──TP── GPU 1 ──TP── GPU 2 ──TP── GPU 3
  │                    │
  EP                  EP
  │                    │
GPU 4 ──TP── GPU 5 ──TP── GPU 6 ──TP── GPU 7
```

### 6.2 前向流程分步

```
Step 1: Embedding (所有 GPU)
 ・每个 GPU 查局部 embedding 表 (padded_vocab/4 = 62096 行)
 ・tensor_model_parallel_all_reduce → 所有 GPU 拿到完整 [B, L, 2048]

Step 2-39: Transformer Layers (x40)
  Layer 1-40 每层流程相同:
  
  Step 2a: Input LayerNorm (RMSNorm)
    ・每卡独立计算, 不需要通信
  
  Step 2b: Attention (Mamba2 or Softmax)
    Mamba2 路径:
      ・in_proj_qkv: ColumnParallel (tp=4, 每卡 10240/4=2560 dim)
      ・卷积 + SSM: 每卡独立, 无通信
      ・out_proj: RowParallel → all_reduce
    
    Softmax Attention 路径:
      ・q_proj: ColumnParallel (每卡 4096/4=1024 dim)
      ・k/v_proj: K/V head 复制
      ・Self-Attention: 每卡在自己的 Q/K/V heads 上计算
      ・o_proj: RowParallel → all_reduce
  
  Step 2c: Post-Attention LayerNorm (RMSNorm)
    ・每卡独立
  
  Step 2d: MoE FFN
    ・Router (gate): 每卡全量计算 (ReplicatedLinear)
    ・结果: 每卡拿到相同的 top-8 expert 选择
    
    [EP=2 时分发]
      ・GPU0/1 负责 experts 0-127 (EP rank 0)
      ・GPU2/3 负责 experts 128-255 (EP rank 1)
      ・GPU4/5 与 GPU0/1 同 EP rank (MoE_DP)
      ・GPU6/7 与 GPU2/3 同 EP rank (MoE_DP)
    
      ・all_gatherv: 收集所有 DP rank 的 token 到同一 EP 组
      ・本地专家计算: gate/up/down projection
          gate/up: ColumnParallel (每卡 intermediate_size/4=128)
          down: RowParallel → local all-reduce (在同 TP 组内)
      ・combine: 将 expert 输出返还给原 token
    
    ・如果 moe_tp_size>1 or moe_ep_size>1:
        tensor_model_parallel_all_reduce 合并结果 → [B, L, 2048]

Step 40: Final Norm (所有 GPU)
  ・每卡独立 RMSNorm

Step 41: LM Head
  ・每卡计算局部 logits (vocab/4 = 62096 类)
  ・all-gather 合并完整 logits → [B, L, 248320]
  ・Sampler 计算下一个 token (每卡相同)
```

### 6.3 每步通信次数统计 (tp=4, ep=2, 1 层)

| 操作 | 通信原语 | 数据量 | 次数/层 |
|------|---------|--------|--------|
| Embedding → all_reduce | all-reduce | B×L×2048×2B | 1 |
| Attention out_proj → all_reduce | all-reduce | B×L×2048×2B | 1 |
| MoE 分发 → all_gatherv | all-gather | B×L×2048×2B | 1 |
| MoE down_proj → all_reduce | all-reduce | B×L×2048×2B | 1 |
| MoE 合并 → all_reduce | all-reduce | B×L×2048×2B | 1 |
| LM Head → all-gather | all-gather | B×L×248320×2B | 1 |
| **每层总计** | | | **4~5 次 all-reduce** |
| **40 层总计** | | | **~160 次 all-reduce** |

---

## 七、通信优化策略

### 7.1 多级 AllReduce 后端选择

`GroupCoordinator.all_reduce` (`parallel_state.py:559`) 根据张量大小动态选择:

```
张量大小 → 选择策略 (按优先级):
  极小块 (< 1KB) → NCCL all_reduce (torch.distributed)
  小块 (< 512KB) → CustomAllReduce (自定义 RDMA)
  中块          → PyNccl (CUDA graph 兼容)
  大块          → TorchSymmMemAllReduce (对称内存优化)
  ROCm GPU      → QuickAllReduce (AMD 专用)
```

### 7.2 算子融合

sglang 针对多 GPU 做了算子融合以减少通信量:

- **Fused AllReduce + RMSNorm** (`GroupCoordinator.fused_allreduce_rmsnorm`): 在 RowParallel 的 all-reduce 后立即接 LayerNorm，合并为一个 kernel
- **MoE All-to-All 与 GEMM 重叠**: `down_gemm_overlap_args` 允许 expert 计算的 down-proj GEMM 与 all-to-all 通信重叠执行
- **CUDA Graph**: 所有通信操作在 CUDA graph 捕获模式下被记录，推理时零开销 launch

### 7.3 Symmetric Memory

sglang 使用 **对称内存 (symmetric memory)** 技术优化 `all-gather`:

```python
with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):
    output_parallel = self.quant_method.embedding(self, masked_input.long())
```

对称内存允许 all-gather 的**接收缓冲区**在 TP 组内所有 GPU 上具有相同的虚拟地址，从而启用更高效的 NVLink 直通。

---

## 八、针对 Qwen3.5-35B-A3B 的推荐配置

### 8.1 显存占用估算

| 并行配置 | 每卡 Expert 数 | 每卡显存 (bf16 权重) | 每卡显存 (4-bit expert) | 总计 |
|---------|---------------|---------------------|----------------------|------|
| TP=8, EP=1 | 256 (全部) | ~2.5 GB | ~7.5 GB | ~10 GB |
| TP=4, EP=2 | 128 | ~2.5 GB | ~3.75 GB | ~6.25 GB |
| TP=2, EP=4 | 64 | ~2.5 GB | ~1.875 GB | <5 GB |
| TP=1, EP=8 | 32 | ~2.5 GB | ~0.94 GB | <4 GB |

**注意:** 以上仅为权重，还需加上 KV cache、activation memory、optimizer states 等。

### 8.2 通信-计算比权衡

```
TP 越大 → 通信量越大 (每次 linear 都要 all-reduce)
EP 越大 → 通信集中在 MoE (All-to-All 变大)
PP 越大 → 流水线气泡增加 (GPU 利用率下降)

推荐 (8×A100/H100):
  长序列 (32K+)： TP=4, EP=2 (减少 TP 通信，发挥 EP 优势)
  短序列 (<4K)：  TP=8, EP=1 (TP 通信占比小，利用 NVIDIA 集体通信)
  多模态 (视觉)：  TP=4, EP=2, PP=2 (PP 容纳视觉编解码器)
```

### 8.3 sglang 启动命令示例

```bash
# TP=4, EP=2 部署
python -m sglang.launch_server \
    --model-path /path/to/Qwen3.5-35B-A3B-GPTQ-Int4 \
    --tp 4 \
    --ep 2 \
    --host 0.0.0.0 \
    --port 30000

# TP=8 (纯张量并行)
python -m sglang.launch_server \
    --model-path /path/to/Qwen3.5-35B-A3B-GPTQ-Int4 \
    --tp 8 \
    --host 0.0.0.0 \
    --port 30000
```

---

## 九、关键源码映射表

| 功能 | 文件路径 | 关键类/函数 |
|------|---------|------------|
| 并行组初始化 | `.../distributed/parallel_state.py` | `initialize_model_parallel()`, `GroupCoordinator` |
| 通信原语 | `.../distributed/communication_op.py` | `tensor_model_parallel_all_reduce/gather` |
| 嵌入并行 | `.../layers/vocab_parallel_embedding.py` | `VocabParallelEmbedding`, `ParallelLMHead` |
| 线性层 TP | `.../layers/linear.py` | `ColumnParallelLinear`, `RowParallelLinear`, `QKVParallelLinear` |
| MoE Expert | `.../layers/moe/fused_moe_triton/layer.py` | `FusedMoE`, `create_moe_dispatcher` |
| Token 分发 | `.../layers/moe/token_dispatcher/standard.py` | `StandardDispatcher.dispatch()` |
| MoE runner | `.../layers/moe/moe_runner/runner.py` | `MoeRunner.run()` |
| Qwen3.5 模型 | `.../models/qwen3_5.py` | `Qwen3_5MoeForCausalLM`, `Qwen3_5MoeForConditionalGeneration` |
| PP 层分配 | `.../models/qwen3_5.py` | `make_layers()` + `self.pp_group` |
| 量化方法 | `.../layers/quantization/` | `GPTQLinearMethod` (4-bit expert) |

