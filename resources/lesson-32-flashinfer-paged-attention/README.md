# Lesson 32：FlashInfer Paged Attention——从 Page Table 到 `plan()` / `run()` 的完整 GPU 路径

> 源码基线：`c335497c6bf67a4dc8cb5ba748ace7b7c1cb77af`
>
> 这一课把 `python/aios/attention/flashinfer.py` 的关键代码直接放进 README。学完后，你不需要打开源码，也能解释 Prefill/Decode 为什么使用不同 Wrapper、`cu_seqlens` 和 `indices` 分别是什么、为什么 Metadata 只 Plan 一次却能被 14 层共享。

## 1. FlashInfer 替 AIOS 做了什么

AIOS 自己负责：

```text
哪些请求进入 Batch
每个请求 q_len / kv_len
Page Table 行
物理 Page ID
Flat q/k/v
KV 写入位置
```

FlashInfer 负责：

```text
根据 Metadata 选择/配置 Attention Kernel
从 Paged KV Cache 按请求逻辑顺序读取 K/V
计算 Causal Prefill 或单 Token Decode Attention
把输出写回 Flat Tensor
```

边界很清晰：AIOS 管“请求与内存身份”，FlashInfer 管“高性能 GPU Attention”。

## 2. Metadata 结构

```python
@dataclass
class FlashInferAttentionMetadata:
    cu_seqlens_q_cpu: torch.Tensor
    cu_seqlens_k_cpu: torch.Tensor
    cu_seqlens_q_gpu: torch.Tensor
    indices: torch.Tensor
    last_page_len_cpu: torch.Tensor
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    pos_encoding_mode: str
    seq_lens_cpu: torch.Tensor
    dtype: torch.dtype
    wrapper: Any
    initialized: bool = False
```

它不是 Attention 输出，而是“如何解释本 Batch 的 Flat Q 与 Paged K/V”的计划输入。

## 3. 手算两个 Prefill 请求

请求 A：

```text
extend_len = 3
device_len = 3
physical pages = [10,11,12]
```

请求 B：

```text
extend_len = 2
device_len = 5
physical pages = [20,21,22,23,24]
```

Flat Q：

```text
[Aq0,Aq1,Aq2,Bq0,Bq1]
```

所以：

```math
cu\_q=[0,3,5]
```

Paged K/V 展开边界：

```math
cu\_k=[0,3,8]
```

物理 Page Index：

```text
indices=[10,11,12,20,21,22,23,24]
```

`cu_q` 告诉 Kernel Flat Q 的请求边界；`cu_k` 告诉 Page Index 数组的请求边界；`indices` 告诉每个逻辑 KV Token 实际在哪个物理 Page。

## 4. `prepare_metadata()` 完整关键代码

```python
reqs = batch.reqs
batch_size = len(reqs)
seqlens_q = [req.extend_len for req in reqs]
seqlens_k = [req.device_len for req in reqs]

seq_lens_cpu = torch.tensor(seqlens_k, device="cpu", dtype=torch.int32, pin_memory=True)
cu_seqlens_k_cpu = torch.tensor(
    [0] + seqlens_k,
    device="cpu",
    dtype=torch.int32,
    pin_memory=True,
).cumsum_(dim=0)
cu_seqlens_q_cpu = torch.tensor(
    [0] + seqlens_q,
    device="cpu",
    dtype=torch.int32,
    pin_memory=True,
).cumsum_(dim=0)

metadata = FlashInferAttentionMetadata(
    cu_seqlens_q_cpu=cu_seqlens_q_cpu,
    cu_seqlens_k_cpu=cu_seqlens_k_cpu,
    cu_seqlens_q_gpu=cu_seqlens_q_cpu.to(device, non_blocking=True),
    indices=torch.cat([
        page_table[req.table_idx, :req.device_len]
        for req in reqs
    ]),
    last_page_len_cpu=self._get_ones_cpu(batch_size),
    ...
)
```

## 5. 为什么有些 Metadata 在 CPU，有些在 GPU

FlashInfer `plan()` 的部分接口需要 CPU Tensor 描述长度/边界，内部用它生成计划；Kernel `run()` 或 AIOS 选最后位置时需要 GPU Tensor。

```python
cu_seqlens_q_gpu = cu_seqlens_q_cpu.to(device, non_blocking=True)
```

`pin_memory=True` 让 CPU Buffer 可进行真正异步的 non-blocking H2D Copy；普通 Pageable Memory 往往需要额外 staging/同步。

当前 Metadata 很小，性能收益未必巨大，但语义正确且避免显式阻塞。

## 6. Wrapper 与 Workspace

Prefill：

```python
flashinfer.BatchPrefillWithPagedKVCacheWrapper(
    workspace,
    kv_layout="NHD",
    backend="fa2",
)
```

Decode：

```python
flashinfer.BatchDecodeWithPagedKVCacheWrapper(
    workspace,
    use_tensor_cores=use_tensor_cores,
    kv_layout="NHD",
    backend="fa2",
)
```

Workspace 是预分配的临时 GPU Byte Buffer：

```python
self._workspace = torch.empty(
    workspace_size,
    dtype=torch.uint8,
    device=device,
)
```

复用 Workspace 避免每层/每步重复分配临时显存。

## 7. `NHD` 是什么

FlashInfer 接收的 Paged Cache View：

```text
[num_pages, page_size, num_kv_heads, head_dim]
```

`NHD` 可理解为：

```text
N = token/page position
H = head
D = head dimension
```

当前 Page Size=1，所以 AIOS 将：

```python
k_cache.view(-1, 1, num_kv_heads, head_dim)
```

传给 FlashInfer。

## 8. 为什么 Prefill 与 Decode Wrapper 分开

Prefill：

```text
每请求 q_len 可以 >1
需要 causal attention 处理多个新 Query Token
```

Decode：

```text
每请求正好 1 个新 Query
但 K/V 历史很长
```

两者矩阵形状和最佳 Kernel 策略不同。Decode 常更 Memory-bound，需要高效从 Paged KV 读取历史；Prefill 有更多 Q，可利用更大的矩阵块。

当前新版 FlashInfer 还有统一 `BatchAttention` 接口可混合 Prefill/Decode，但 AIOS 当前源码仍显式分开 Wrapper；教材以固定源码为准，不把新 API 写成已实现。

## 9. `plan()` 为什么只执行一次

```python
def _initialize_metadata_once(metadata):
    if metadata.initialized:
        return
    metadata.initialized = True
    metadata.wrapper.plan(...)
```

一个 Batch 会经过 14 层 Attention，但：

```text
请求边界
Page Table
Q/K 长度
Head 数
Page Size
DType
```

在各层相同。

所以第一层 Plan 后，后续层只：

```text
store current layer K/V
→ wrapper.run(q, layer_cache)
```

避免 14 次重复计划。

不能跨 Batch 无条件复用 Plan，因为下一 Batch 的 active requests、lengths、indices 可能变化。

## 10. Prefill 完整执行

```python
self._initialize_metadata_once(metadata)
paged_kv_cache.store_kv(k, v, batch.out_loc.view(-1), layer_id)
attn_output = metadata.wrapper.run(
    q,
    self._kv_cache_for_flashinfer(paged_kv_cache, layer_id),
)
return attn_output.reshape(q.size(0), -1)
```

顺序很重要：

1. Plan 当前 Batch；
2. 把当前新 Token 的 K/V 写入 Page；
3. Attention 读取包含新旧 Token 的完整 K/V；
4. Causal Mask 保证 Query 不能看未来 Query Token。

## 11. Decode 完整执行

```python
bsz = batch.size
assert q.size(0) == bsz
self._initialize_metadata_once(metadata)
paged_kv_cache.store_kv(k, v, batch.out_loc.view(-1), layer_id)
attn_output = metadata.wrapper.run(
    q,
    self._kv_cache_for_flashinfer(paged_kv_cache, layer_id),
)
return attn_output.reshape(bsz, -1)
```

Decode 不变量：

```text
每个请求每轮恰好一个新 Token
→ total q tokens = batch size
```

若 `q.size(0) != bsz`，说明 Scheduler/Batch 构造错误，不应让 Backend 猜。

## 12. 为什么先写 K/V 再 Attention

当前 Query Token 也应看到自己的 K/V（因果 Attention 包含当前位置）：

```text
位置 t 可以看 0..t
```

所以必须先把位置 t 的 K/V 写入 Cache，再让 Q_t Attention 到长度 `t+1` 的 K/V。

## 13. `use_tensor_cores` 的判断

```python
use_tensor_cores = num_heads // num_kv_heads >= 4
```

当前：

```text
12 // 4 = 3
```

所以 Decode Wrapper 不走这个 GQA Ratio≥4 的 Tensor Core 路径。

这是一条库策略，不代表 Tensor Core 在整个模型中不用；Linear GEMM 仍可能使用 Tensor Core。

## 14. FlashInfer 为什么难自己重写

高性能 Attention 涉及：

- Online Softmax；
- Paged Gather；
- GQA Head Mapping；
- Causal Varlen；
- Tensor Core Tile；
- Shared Memory/Register Blocking；
- 不同 GPU 架构与 Shape Dispatch；
- 数值稳定与低精度。

AIOS 选择把控制面做清楚，调用成熟 Kernel 库，比为课程展示而重写一个低性能 Attention 更合理。

## 15. 常见错误理解

### 错误：`plan()` 已经执行 Attention

错。Plan 只根据 Metadata 准备 Kernel 运行计划；实际计算在 `run()`。

### 错误：Page Table 本身就是 K/V

错。`indices` 是物理 Page ID；真实 K/V 在 `MHAKVCache` Buffer。

### 错误：Metadata Plan 可以在模型加载时一次永久完成

错。请求数、长度、Page Index 每个 Batch 都可能变化，只能在同一个 Batch 的层间复用。

## 16. 运行实验

```bash
python resources/lesson-32-flashinfer-paged-attention/run_lesson32.py
```

CPU 实验会构造两个请求的 `cu_q/cu_k/indices`，逐位置恢复每个请求的 Page 列表。

## 17. 检验问题与参考答案

### 问题 1：`cu_seqlens_q=[0,3,5]` 为什么能表示两个请求？

**参考答案：** Prefix Sum 的相邻差给出长度：请求 0 的 Flat Q 区间是 `[0,3)`，长度 3；请求 1 是 `[3,5)`，长度 2。这样不需要 Padding，也能在一维 Flat Tensor 中恢复请求边界。

### 问题 2：为什么 `indices` 必须按每个请求的逻辑 Token 顺序拼接？

**参考答案：** Paged KV 的物理 Page 可以散乱，但 Attention 的时间顺序必须是逻辑位置 0、1、2……。`cu_k` 划分请求，`indices` 在每个区间内按逻辑顺序列物理 Page，Kernel 才能恢复正确序列。

### 问题 3：为什么 Metadata 可以跨 14 层共享，KV Cache Tensor却不能？

**参考答案：** 一个 Batch 的请求边界和 Page 身份在各层相同，所以 Plan Metadata 可共享；但每层有自己独立的 K/V 数值，`layer_id` 选择不同 Cache Slice，因此数值 Tensor 不能跨层共用。

### 问题 4：为什么 Decode 要断言 `q.size(0)==batch_size`？

**参考答案：** Decode 定义为每个运行请求本轮处理一个新 Token，所以 Flat Q 行数必须等于请求数。若不相等，Page 写入、cu_seqlens 和输出对应关系都会错位，这是上游状态机错误而不是 Backend 可修复情况。

## 18. 一句话复述

AIOS 把每个 Batch 的 Q/K 长度和 Page Table 转成 `cu_seqlens + indices`，FlashInfer `plan()` 据此准备 Paged Attention，一次 Plan 被 14 层共享；每层先 Scatter 当前 K/V，再用对应层 Cache `run()` Attention。Prefill 与 Decode 因 Q Shape 不同使用专用 Wrapper。

## 19. 一手参考

- FlashInfer Attention API：Paged Prefill/Decode、`plan()`/`run()`、NHD Layout。
- 当前 AIOS 固定依赖 `flashinfer-python==0.5.3`；阅读新文档时需区分版本 API。
