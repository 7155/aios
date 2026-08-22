# Lesson 5：Paged KV Cache —— 从动态扩容到页式 KV 内存管理

这一课解决的不是“KV Cache 为什么存在”，而是更进一步的问题：

> **已经决定缓存历史 K/V 以后，怎样管理这些不断增长、长度不同、生命周期不同的 KV 内存？**

Lesson 4 解决的是“不要重复计算历史 K/V”；Lesson 5 解决的是“不要为了保存这些 K/V，不断重拷贝或长期浪费显存”。

---

## 学习目标

完成本课后，你应该能够解释：

1. `torch.cat` 动态 KV Cache 为什么正确但越来越慢；
2. 为什么旧 Cache 实际会释放，但仍然存在 O(n²) 的累计内存搬运；
3. Preallocated KV Cache 怎样用“提前预留 + 按位置写入”消除重拷贝；
4. 为什么给每个请求都按 `max_seq_len` 预留会浪费显存；
5. Paged KV Cache 怎样把“每请求预分配”升级成“全 Runtime 共享物理 KV Pool”；
6. `page_table`、`free_slots`、`MHAKVCache` 分别负责什么；
7. 当前 AIOS 为什么没有 `Page` 类，所谓 page 实际只是大 Tensor 中的一个物理 slot；
8. 当前 AIOS 的 `page_size=1` 到底意味着什么。

---

# 1. 第一代：Dynamic KV Cache

![Dynamic KV Cache 示意图](dynamic_kv_cache.svg)

最容易想到的实现是：每次生成一个新 token，就把新 K/V 拼到旧 K/V 后面。

概念代码：

```python
if past_key_value is not None:
    k = torch.cat([past_key_value[0], k], dim=2)
    v = torch.cat([past_key_value[1], v], dim=2)
```

假设当前：

```text
old_k = [K0 K1 K2 K3]
new_k = [K4]
```

`torch.cat` 不能直接让旧 Tensor “原地长大”。它必须：

```text
旧 Storage：
[K0 K1 K2 K3]

新 token：
[K4]

        ↓ torch.cat

申请一个更大的新 Storage：
[_  _  _  _  _]

复制：
[K0 K1 K2 K3 K4]

        ↓
旧 Tensor 不再被引用后可以释放
```

## 一个容易误解的点：Dynamic Cache 并不会永久保存所有旧版本

它不是：

```text
[K0]
[K0 K1]
[K0 K1 K2]
[K0 K1 K2 K3]
```

这些版本永久同时存在。

正常情况下，新 `cat` 完成、引用切换后，旧 Tensor 就可以释放。真正的问题有两个：

### 问题 A：每一步都要重新搬全部历史 KV

```text
Step 1：复制 1 份
Step 2：复制 2 份
Step 3：复制 3 份
...
Step n：复制 n 份
```

累计：

```text
1 + 2 + 3 + ... + n = n(n+1)/2 = O(n²)
```

所以 KV 虽然“不重算”了，却仍然在每一步“重搬家”。

### 问题 B：cat 期间需要旧 Storage 和新 Storage 同时存在

必须先：

```text
旧 Cache 还活着
    ↓
分配新 Cache
    ↓
把旧内容复制进去
    ↓
复制完成
    ↓
旧 Cache 才能释放
```

因此会出现临时峰值显存和 allocator 抖动。

> PyTorch 的 CUDA caching allocator 会缓存已释放的显存块，所以“释放”也不等于 `nvidia-smi` 立刻下降；但无论 allocator 是否复用，`cat` 的历史数据重拷贝成本都存在。

### Dynamic Cache 的本质

> **历史 K/V 不再重复计算，但历史 K/V 仍在每一步重复复制。**

这就是下一步为什么要做 Preallocated Cache。

---

# 2. 第二代：Preallocated KV Cache

![Preallocated KV Cache 示意图](preallocated_kv_cache.svg)

如果我们知道一个请求最大可能有 `max_seq_len` 个 token，就可以一开始直接申请整块连续空间：

```python
k_cache = torch.empty(
    max_seq_len,
    num_kv_heads,
    head_dim,
    device="cuda",
    dtype=dtype,
)
```

假设：

```text
max_seq_len = 8
```

初始化：

```text
[_  _  _  _  _  _  _  _]
```

生成 K0：

```text
[K0 _  _  _  _  _  _  _]
```

生成 K1：

```text
[K0 K1 _  _  _  _  _  _]
```

每一步只是：

```python
k_cache[position] = new_k
v_cache[position] = new_v
```

因此：

```text
Dynamic cat：
第 n 步还要复制 K0...K(n-1)

Preallocated：
第 n 步只写 Kn
```

KV **写入**从累计 O(n²) 搬运，变成累计 O(n) 写入。

## 注意：Attention 本身仍需要读取历史 KV

Preallocated Cache 消除的是：

```text
❌ 为了“追加 Kn”而复制 K0...K(n-1)
```

它不能消除：

```text
✅ Qn 做 Attention 时读取 K0...Kn
```

后者是自回归 Attention 算法本身需要的。

## Preallocated 的缺点

如果：

```text
max_seq_len = 4096
实际请求只用了 120 token
```

那么仍然有：

```text
[使用 120 | 空着 3976]
```

这些没用到的槽位已经被这个请求预留，别的请求不能使用。

单请求问题不大，但多请求并发时会造成严重浪费：

```text
Request A：预留 4096，实际使用 100
Request B：预留 4096，实际使用 80
Request C：预留 4096，实际使用 200
...
```

于是问题从：

> “怎么避免每次扩容？”

进一步变成：

> **“怎么既不扩容重拷贝，又不让每个请求独占一大块最大长度连续空间？”**

答案就是分页。

---

# 3. 第三代：Paged KV Cache

![操作系统分页内存管理示意图](os_paging_memory.svg)

Paged KV Cache 借鉴操作系统的分页思想：

1. 请求看到的 token 序列在**逻辑上连续**；
2. KV 在物理显存中可以放在**不连续的 page/slot**；
3. 用一个 Page Table 记录“逻辑位置 → 物理 page id”。

概念上可以把 page 设成多个 token，例如：

```text
page_size = 512 tokens

逻辑 token 0~511      → Virtual Page 0
逻辑 token 512~1023   → Virtual Page 1
逻辑 token 1024~1535  → Virtual Page 2
```

物理页不需要连续：

```text
Virtual Page 0 → Physical Page 17
Virtual Page 1 → Physical Page 3
Virtual Page 2 → Physical Page 91
```

于是页表：

```text
block/page table = [17, 3, 91]
```

## 一个关键修正：Paged KV 不是“每缺一页就 cudaMalloc 一页”

成熟实现一般不会在 Decode 热路径里反复：

```text
需要新 page
    ↓
cudaMalloc(...)
```

因为频繁申请/释放 GPU 内存本身就很贵。

更常见、也是当前 AIOS 的思想是：

```text
Runtime 启动
    ↓
一次性建立一个大的 Physical KV Pool
    ↓
切成很多固定 page/slot
    ↓
请求只领取 page id
    ↓
请求结束归还 page id
```

所以 Paged KV 可以理解成：

> **Preallocated 的“提前申请”思想没有消失，只是从“每个请求各自预留一大块”升级成了“整个 Runtime 共享一个大物理池”。**

---

# 4. 当前 AIOS 的关键事实：1 token = 1 page slot

![Paged KV Cache Prefill 流程图](paged_kv_cache_prefill.svg)

![Paged KV Cache Decode 流程图](paged_kv_cache_decode.svg)

当前实现明确限制：

```python
assert page_size == 1, "AIOS FlashInfer backend only supports page_size=1"
```

所以当前 AIOS 里：

```text
1 logical token
    ↓
1 physical page slot
```

不要把“512 token 一页”的概念例子机械套到当前源码。

当前实现更准确地说是：

> **token-level paged KV slot management。**

---

# 5. 当前 AIOS 没有 `Page` 类：Page 就是大 Tensor 中的一个索引

当前物理池在：

```text
python/aios/kvcache/mha_pool.py
MHAKVCache
```

核心代码：

```python
self._kv_buffer = torch.empty(
    (
        2,
        num_layers,
        num_pages,
        page_size,
        local_kv_heads,
        head_dim,
    ),
    device=device,
    dtype=dtype,
)
```

各维含义：

```text
2              → K / V
num_layers     → Transformer layer
num_pages      → 物理 page 数量
page_size      → 当前固定为 1
local_kv_heads → KV heads
head_dim       → 每个 head 的维度
```

所以所谓：

```text
Physical Page 17
```

并不是一个 Python `Page` 对象，也不是单独一次 `cudaMalloc`。

它只是大 Tensor 里面：

```text
_kv_buffer[:, :, 17, ...]
```

对应的物理槽位。

可以把它理解成：

```text
Physical KV Pool

Page 0
Page 1
Page 2
...
Page 17
...
Page N
```

但是这些 page 在底层实际上属于同一个预分配 Tensor Storage。

---

# 6. `CacheManager`：管理哪些物理 page 还空着

当前代码：

```text
python/aios/scheduler/cache.py
```

初始化：

```python
self._free_slots = torch.arange(
    num_pages,
    dtype=torch.int32,
    device=device,
)
```

例如：

```text
num_pages = 8

_free_slots = [0,1,2,3,4,5,6,7]
```

请求需要 3 个新 token：

```python
def allocate(self, needed_len: int) -> torch.Tensor:
    if needed_len <= len(self._free_slots):
        allocated = self._free_slots[:needed_len]
        self._free_slots = self._free_slots[needed_len:]
        return allocated
    raise NotImplementedError("CacheManager eviction is not implemented.")
```

于是：

```text
allocate(3)

allocated   = [0,1,2]
_free_slots = [3,4,5,6,7]
```

这里没有新申请一块 GPU KV Storage。

只是：

> **从空闲物理 page id 中拿走几个号码。**

请求结束：

```python
def _free(self, indices: torch.Tensor) -> None:
    if len(indices) > 0:
        self._free_slots = torch.cat([self._free_slots, indices])
```

page id 被重新放回 free list，后续请求可以复用。

注意这里的 `torch.cat` 只是拼接一个很小的 **page id 管理数组**，不是在重拷贝巨大 KV Tensor，因此和 Dynamic KV Cache 对 K/V 本体做 `cat` 不是一个量级的问题。

---

# 7. `page_table`：逻辑 token → 物理 page id

当前代码不再把 `block_table` 直接塞进每个 `Req`。

现在由：

```text
TableManager.page_table
```

统一维护：

```python
class TableManager:
    def __init__(self, max_running_reqs: int, page_table: torch.Tensor) -> None:
        self._free_slots = list(range(max_running_reqs))
        self.page_table = page_table
        # shape: (max_running_reqs, max_seq_len)
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)
```

每个请求只保存：

```python
@dataclass(eq=False)
class Req:
    ...
    table_idx: int | None = None
```

所以可以把 `page_table` 想成：

```text
              logical token position
             0    1    2    3    4
           ┌────┬────┬────┬────┬────┐
Req row 0  │ 17 │  3 │ 91 │ 22 │  8 │
           ├────┼────┼────┼────┼────┤
Req row 1  │  5 │ 42 │  6 │    │    │
           └────┴────┴────┴────┴────┘
```

表示：

```text
请求 0：
logical token 0 → physical Page 17
logical token 1 → physical Page 3
logical token 2 → physical Page 91
logical token 3 → physical Page 22
```

逻辑 token 连续，物理页完全可以不连续。

---

# 8. 分页真正发生在哪里：`allocate_paged()`

当前代码：

```python
def allocate_paged(self, reqs: list[Req], page_table: torch.Tensor) -> None:
    for req in reqs:
        assert req.table_idx is not None
        extend_len = req.extend_len
        if extend_len == 0:
            continue

        indices = self.allocate(extend_len)

        page_table[
            req.table_idx,
            req.cached_len : req.device_len,
        ] = indices
```

这几行就是 Paged KV 内存管理的核心之一。

逐行理解：

```text
extend_len
    ↓
“这个请求本轮有几个新 token 没写入 KV？”

allocate(extend_len)
    ↓
“从共享物理池领取同样数量的 page id”

page_table[request, logical positions] = page ids
    ↓
“建立逻辑 token → 物理 slot 映射”
```

例如：

```text
req.table_idx = 2
cached_len    = 4
device_len    = 5
extend_len    = 1
```

`allocate(1)` 返回：

```text
[37]
```

于是：

```text
page_table[2, 4] = 37
```

意思就是：

```text
请求 2 的逻辑 token 4
        ↓
写到 Physical Page 37
```

---

# 9. Scheduler 怎样把 page id 变成真正的写入地址

当前顶层路径在：

```text
python/aios/scheduler/scheduler.py
```

准备一个 batch 时：

```python
def _prepare_batch(self, batch: Batch) -> Batch:
    self.cache_manager.allocate_paged(
        batch.reqs,
        self.table_manager.page_table,
    )

    batch.positions = _make_positions(batch, self.device)
    input_mapping = _make_input_tuple(batch, self.device)

    batch.input_ids = self.table_manager.token_pool[input_mapping].long()
    batch.out_loc = self.table_manager.page_table[input_mapping]

    self.attn_backend.prepare_metadata(batch)
    return batch
```

这里最关键的是：

```python
batch.out_loc = self.table_manager.page_table[input_mapping]
```

`out_loc` 就是：

> **本轮这些新 token 的 KV 应该写入哪些 physical page。**

例如：

```text
out_loc = [17, 3, 91, 22]
```

表示本轮 4 个 token：

```text
token 0 → Page 17
token 1 → Page 3
token 2 → Page 91
token 3 → Page 22
```

---

# 10. `MHAKVCache.store_kv()`：真正把 KV scatter 到物理页

当前代码：

```python
def store_kv(
    self,
    k: torch.Tensor,
    v: torch.Tensor,
    out_loc: torch.Tensor,
    layer_id: int,
) -> None:
    from aios.kernel import store_cache

    store_cache(
        k_cache=self._k_buffer[layer_id].view(self._storage_shape),
        v_cache=self._v_buffer[layer_id].view(self._storage_shape),
        indices=out_loc,
        k=k,
        v=v,
    )
```

概念上相当于：

```python
for token_idx, physical_page in enumerate(out_loc):
    k_cache[physical_page] = k[token_idx]
    v_cache[physical_page] = v[token_idx]
```

真实实现当然不是 Python `for`，而是由 GPU kernel 做并行 scatter。

因此 Paged KV 的写入链可以压缩成：

```text
逻辑 token
    ↓
page_table 查到 physical page id
    ↓
out_loc
    ↓
store_cache kernel
    ↓
写进共享 MHAKVCache 大 Tensor
```

---

# 11. 一个完整例子

假设：

```text
num_pages = 8
page_size = 1
```

初始物理池：

```text
Physical KV Pool

P0 P1 P2 P3 P4 P5 P6 P7
```

空闲表：

```text
free_slots = [0,1,2,3,4,5,6,7]
```

请求 A 的 prompt：

```text
[t0, t1, t2, t3]
```

### Step A：请求 A 获得一个 `table_idx`

例如：

```text
A.table_idx = 0
```

### Step B：Prefill 需要 4 个 page

```text
extend_len = 4
allocate(4)
→ [0,1,2,3]
```

写入：

```text
page_table[0, 0:4] = [0,1,2,3]
```

因此：

```text
A logical token 0 → P0
A logical token 1 → P1
A logical token 2 → P2
A logical token 3 → P3
```

KV Pool：

```text
P0 = KV(t0)
P1 = KV(t1)
P2 = KV(t2)
P3 = KV(t3)
P4 = free
P5 = free
P6 = free
P7 = free
```

### Step C：Decode 生成 n0

请求状态推进：

```python
def complete_one(self) -> None:
    self.cached_len = self.device_len
    self.device_len += 1
```

下一轮：

```text
cached_len = 4
device_len = 5
extend_len = 1
```

申请：

```text
allocate(1) → [4]
```

页表：

```text
page_table[0, 4] = 4
```

所以：

```text
A logical token 4 → P4
```

写入：

```text
P4 = KV(n0)
```

### Step D：A 结束

Scheduler 当前释放逻辑：

```python
def _free_req_resources(self, req: Req) -> None:
    used_pages = self.table_manager.page_table[
        req.table_idx,
        : req.cached_len,
    ]
    self.table_manager.free(req.table_idx)
    self.cache_manager._free(used_pages)
```

于是 A 使用过的 page id 被放回 `_free_slots`，可以马上给后续请求复用。

---

# 12. 为什么物理不连续仍然能工作？

假设多个请求交错运行后，请求 C 获得：

```text
C logical token 0 → P6
C logical token 1 → P1
C logical token 2 → P7
C logical token 3 → P3
```

它的页表可以是：

```text
[6, 1, 7, 3]
```

逻辑上仍然是：

```text
t0 → t1 → t2 → t3
```

Attention backend 根据 page table / metadata 找到对应 KV 即可。

因此 Paged KV 的核心不是“内存连续”，而是：

> **逻辑连续性和物理连续性解耦。**

这就是它和操作系统虚拟内存非常像的地方。

---

# 13. 三种 KV Cache 放在一起比较

| 方案 | KV 增长方式 | 历史 KV 是否重拷贝 | 未使用显存浪费 | 多请求复用 | 关键问题 |
|---|---|---:|---:|---:|---|
| Dynamic / `cat` | 每步重新建更大 Tensor | 是 | 小 | 差 | O(n²) 累计搬运 |
| Preallocated | 每请求按最大长度预留连续块 | 否 | 大 | 差 | 空槽长期占用 |
| Paged KV | 全局共享 Pool，按 page/slot 分配 | 否 | 小 | 好 | 需要页表与调度管理 |

它们的演进逻辑可以记成：

```text
Dynamic
“需要多少就扩多少”
    ↓
问题：每次扩容都搬旧数据

Preallocated
“一次给一个请求留够最大空间”
    ↓
问题：大量预留空间根本没用

Paged
“整个 Runtime 一次建共享 Pool，
 请求只拿自己真正需要的 page”
```

---

# 14. 当前 AIOS 的组件边界

![AIOS Paged KV 整体架构图](paged_kv_architecture_aios.svg)

当前 `main` 的真实结构是：

```text
Req
├── cached_len
├── device_len
└── table_idx
        │
        ▼
TableManager.page_table
逻辑 token → physical page id
        │
        ▼
CacheManager
allocate / free physical page id
        │
        ▼
Scheduler._prepare_batch
生成 batch.out_loc
        │
        ▼
MHAKVCache.store_kv
        │
        ▼
store_cache GPU kernel
        │
        ▼
Physical KV Pool
```

对应源码：

- 请求状态：`python/aios/core.py` (`Req`)
- 页表 / 请求槽：`python/aios/scheduler/table.py` (`TableManager`)
- page id 分配释放：`python/aios/scheduler/cache.py` (`CacheManager`)
- batch 映射：`python/aios/scheduler/scheduler.py` (`_prepare_batch`)
- 物理 KV Pool：`python/aios/kvcache/mha_pool.py` (`MHAKVCache`)
- GPU KV 写入：`python/aios/kernel` 中的 `store_cache`
- Attention backend：当前由 FlashInfer 路径消费 paged metadata

当前代码里：

- 没有 `Page` 类；
- 没有 `PagedKVCache` 类；
- 没有把完整 `block_table` 存进每个 `Req`；
- page table 是共享二维 Tensor；
- 当前 `page_size=1`；
- eviction 尚未实现，page 不够时当前会 `NotImplementedError`；
- prefix-cache handle / lock / unlock 相关接口当前仍处于注释或后续扩展边界。

---

# 15. 四行代码记住 Paged KV

如果把所有工程细节压缩掉，Paged KV 就是：

```python
# 1. Runtime 启动时建立共享物理 KV Pool
kv_pool = torch.empty(num_pages, ...)

# 2. 请求新增 token 时领取空闲 physical page id
pages = allocate(num_new_tokens)

# 3. 建立 logical token → physical page 映射
page_table[request, token_positions] = pages

# 4. 按 page id 把新 K/V scatter 进物理池
kv_pool[pages] = new_kv
```

当前 AIOS 的真实代码只是把这四步拆分到了 `MHAKVCache`、`CacheManager`、`TableManager` 和 `Scheduler` 中。

---

# 16. 面试版总结

可以这样回答：

> Dynamic KV Cache 用 `torch.cat` 动态增长。旧 Tensor 最终会释放，但每次增长前仍必须分配新的更大 Storage，并把全部历史 K/V 重新复制进去，所以累计产生 O(n²) 的内存搬运。Preallocated KV Cache 通过一次性给请求预留最大连续空间，消除了历史 KV 重拷贝，但大量未使用槽位会长期占显存。Paged KV Cache 再进一步，把整个 Runtime 的 KV 显存预分配成共享物理 Pool，请求只按实际 token 数领取 page，并通过 page table 将逻辑 token 映射到不连续的物理页；请求结束后 page 归还 free list，从而同时获得稳定写入和更高的并发显存利用率。

当前 AIOS 还可以补一句：

> AIOS 当前 `page_size=1`，因此一个 token 对应一个 physical slot；所谓 page 并不是独立 Python 对象或每次单独 `cudaMalloc` 的显存，而是预分配 `MHAKVCache` 大 Tensor 中由整数 page id 索引的槽位。

---

# 17. 练习题

### 练习 1

为什么 `torch.cat` 后旧 Cache 明明会释放，Dynamic KV Cache 仍然是 O(n²) 累计搬运？

**参考答案：** 因为释放只能发生在新 Tensor 创建并完成旧数据复制以后。第 n 步仍然必须读取并复制前 n-1 个历史 KV，因此总搬运量是 `1+2+...+n`。

### 练习 2

Preallocated KV Cache 为什么不能简单作为高并发 Serving 的最终方案？

**参考答案：** 它解决了扩容重拷贝，但通常需要按最大可能长度给每个请求预留连续空间。短请求会留下大量无法被其他请求使用的空槽，导致显存利用率低。

### 练习 3

Paged KV Cache 为什么需要 page table？

**参考答案：** 因为逻辑 token 顺序连续，但物理 KV page 可以离散。page table 保存逻辑位置到 physical page id 的映射，使 Attention 能按逻辑顺序找到真实 KV。

### 练习 4

当前 AIOS 中 `allocate(1)` 是否意味着执行一次新的 `cudaMalloc`？

**参考答案：** 不是。物理 KV Pool 已经由 `MHAKVCache` 预分配。`CacheManager.allocate(1)` 只是从 `_free_slots` 中取出一个已有 physical page id。

### 练习 5

为什么 `page_size=1` 仍然可以称为 paged KV？

**参考答案：** 分页的本质是“逻辑地址到离散物理槽位的映射”，而不是一页必须包含多个 token。当前 AIOS 把粒度降到 token 级，每个 token 一个 physical slot，仍然通过共享 Pool、free list 和 page table 完成逻辑/物理解耦。
