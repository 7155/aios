# Lesson 12：CandidateGroup——一次 Prefix，八条分支，共享物理 KV

> 源码基线：`bfc72896bbadab5c897672506d237c070900412e`
>
> 前置知识：Lesson 4 的 KV Cache、Lesson 5 的 Paged KV、Lesson 8 的 Varlen Prefill。
>
> 本课目标：看懂 `page_table` 的每一格到底指向什么，并能解释为什么八条候选在逻辑上是八个序列，在物理上却只保存一份 Prefix KV。

![CandidateGroup 与 Top-3 流程](../../docs/images/aios-ime-candidate-group.svg)

## 1. 先看最浪费的实现

前缀：

```text
没关系，你先忙你的，
```

要生成 8 条候选。最直接的代码是循环 8 次：

```python
for seed in range(8):
    model.generate(prefix, seed=seed)
```

这样会重复：

```text
Tokenizer（可能可缓存）
Prefix Embedding
14 层 Prefix Attention / MLP
Prefix K/V 写入
```

但八条候选在产生第一个新 token 前，所有计算完全相同。

真正需要分叉的时刻是：

```text
同一个 prefix_logits
→ 8 个独立随机数
→ 8 个 first token
```

所以正确结构是：

```text
Prefix Prefill 一次
        ↓
prefix_logits [1,V]
        ↓ expand，不重新计算
[8,V]
        ↓ 每行独立采样
8 个 first token
        ↓
之后才为各行分配独立 suffix KV
```

## 2. Page Table 是逻辑序列到物理 Page 的映射

假设：

```text
Prefix 长度 = 4
候选分支 = 3（为了手算）
每条最多再生成 4 token
Page size = 1 token
```

Prefix Prefill 分到物理 Page：

```text
prefix_pages = [10, 11, 12, 13]
```

Page Table 初始：

```text
row 0: [10,11,12,13, 0,0,0,0]
row 1: [10,11,12,13, 0,0,0,0]
row 2: [10,11,12,13, 0,0,0,0]
```

前三行的前四格数字相同，不代表复制了三份 K/V。它们只是都指向同一组物理 Page：

```text
Page 10: prefix token 0 的 K/V
Page 11: prefix token 1 的 K/V
Page 12: prefix token 2 的 K/V
Page 13: prefix token 3 的 K/V
```

因此：

```text
逻辑 Page 引用数 = 3 × 4 = 12
物理唯一 Prefix Page = 4
```

这是共享的关键。

## 3. 为什么第一个候选 token 不需要先分配 suffix Page

Prefix Forward 已经返回最后位置的：

```text
prefix_logits [1,V]
```

它直接定义：

```math
P(x_1 \mid prefix)
```

所以首 token 可以从该分布采样：

```python
logits = prefix_logits.expand(attempts, -1)
```

这只是创建 Batch 视图/广播语义，不重新执行模型。

采出第一个 token 后，若分支还要生成第二个 token，才需要把第一个 token 送入 Decode，并给它分配新的 K/V Page。

因此最坏 Page 预算是：

```math
\text{prefix\_len} + \text{attempts}\times(\text{max\_new\_tokens}-1)
```

不是：

```math
\text{attempts}\times(\text{prefix\_len}+\text{max\_new\_tokens})
```

当前源码正按这个公式提前检查 KV Budget。

### 真实默认值示例

假设 Prefix 20 token、首轮 8 路、最多 12 token：

```math
20 + 8\times(12-1)=108\text{ pages}
```

如果复制 Prefix：

```math
8\times(20+11)=248\text{ pages}
```

共享 Prefix 在这个例子中少使用：

```text
248 - 108 = 140 pages
```

每 Page 约 14 KiB，理论 KV 差约：

```math
140\times14\text{ KiB}\approx1.91\text{ MiB}
```

对 0.1B 模型看似不大，但这还影响 Page 分配、Page Table、Cache 可用容量和后续 Prefix 复用。

## 4. 源码如何写共享引用

核心操作：

```python
page_table = torch.zeros(
    (max_attempts, max_total_len),
    dtype=torch.int32,
    device=device,
)

page_table[:, :len(token_ids)] = prefix_pages.unsqueeze(0)
```

看懂 Shape：

```text
prefix_pages              [prefix_len]
unsqueeze(0)              [1,prefix_len]
赋给所有 row              [max_attempts,prefix_len]
```

这里复制的是很小的 Page ID，不是 K/V Tensor。

真实 K/V 存在：

```text
MHAKVCache[num_layers, K/V, num_pages, num_kv_heads, head_dim]
```

Attention Backend 通过 Page Table 找到每条逻辑序列需要读取的物理 Page。

## 5. Prefix 与 Suffix 的所有权不同

### Prefix Page

```text
所有候选 row 共享
CandidateGroup 完成后仍可保留
下一次按键可能继续复用
```

### Suffix Page

```text
只属于一条候选分支
分支结束后不再有后续用途
候选文本/分数物化后立即释放
```

源码在每轮候选批次完成后：

```python
self.llm.cache_manager._free(torch.cat(allocated_pages))
allocated_pages.clear()
```

然后才判断是否需要补采样。

这意味着：

> refill 的 4 路不会和已经完成的 8 路 suffix KV 同时占显存。

## 6. 为什么不能简单复制 `prefix_pages` 的 K/V Tensor

复制 Tensor 会产生：

- 8 倍 Prefix KV 写入与显存；
- 分叉前不必要的带宽；
- 更低的 Cache 容量；
- 跨按键复用时更复杂的多副本生命周期。

Page Table 共享把“相同内容”表示为“相同物理身份”，而不只是值相等。

这是系统设计里常见的区别：

```text
value equality   值一样
identity sharing 真正指向同一个资源
```

## 7. 为什么 Page Size 选择 1 token

当前 AIOS-IME 使用：

```text
page_size = 1
```

好处：

- token-LCP 可以精确到单 token；
- 任意分支每步只分配一个 Page；
- 不需要处理半满 Block；
- 0.1B、短上下文下 Page Table 成本可接受。

代价：

- Page Table 更长；
- 分配元数据次数更多；
- 大模型长上下文下通常更希望一个 Block 包含多个 token。

所以这不是 PagedAttention 的通用最佳值，而是当前 workload 的取舍。

## 8. 常见错误解释

### 错误：八行 Page Table 相同，所以 K/V 被复制了八次

错。表里是物理 Page ID；相同 ID 代表共享同一份 K/V。

### 错误：首 token 也需要一页 suffix KV 才能采样

错。首 token 从 Prefix 最后位置 Logits 采样；只有为预测第二个 token 执行 Decode 时，才写首 token 的 K/V。

### 错误：CandidateGroup 就是 Static Batch

不完整。它除了 Batch 计算，还拥有共享 Prefix、共同取消、共同 refill 和共同 Top-3 选择语义。

## 9. 运行实验

```bash
python resources/lesson-12-candidate-group/run_lesson12.py
```

实验会构造一个小 Page Allocator，打印三行 Page Table，验证前缀物理 Page 只分配一次、后缀各自独占，并在完成后释放 suffix。

正式 GPU 验证：

```bash
AIOS_IME_MODEL=/path/to/model pytest -q \
  tests/test_ime_gpu.py::test_candidate_group_returns_three_and_recycles_branch_pages
```

## 10. 检验问题与参考答案

### 问题 1：为什么共享 Prefix Page 不需要 Copy-on-write？

**参考答案：** 因为 Decode 只会把新 Token 的 K/V 写到新的 suffix Page，不会原地修改历史 Prefix 的 K/V。所有候选都只是只读同一份 Prefix Cache，所以没有“某个分支写坏共享页”的风险。若未来算法允许修改历史 K/V，才需要 Copy-on-write 或另一种版本化机制。

### 问题 2：为什么最坏 Page 预算是 `prefix + attempts × (max_new_tokens - 1)`？

**参考答案：** Prefix 只保存一份；第一个新 Token 直接从 Prefix 最后位置的 Logits 采样，还没有执行该 Token 的 Decode，所以不需要它的 KV Page。只有为了预测第二个及之后的 Token，才需要把已生成 Token 写进 KV。因此每个分支最多额外占 `max_new_tokens - 1` 页。

### 问题 3：Page Size 变成 16 后，token-LCP 复用会遇到什么问题？

**参考答案：** LCP 可能落在一个 16-token Block 中间。若 Cache 只能以整块共享，就要么丢弃这个半块重新计算，产生内部碎片和重复 Prefill；要么支持部分 Block 复用/Copy-on-write，元数据和所有权都会更复杂。`page_size=1` 避免了这个边界问题，但增加 Page Table 与分配开销。

### 问题 4：`expand` 与真实复制 Prefix Logits 有什么区别？

**参考答案：** `expand` 让多个逻辑 Batch Row 读取同一块底层数据，不重新复制 `[1,V]` 的内容；真实复制会产生多份内存。这里每个候选只需要基于同一个分布使用不同随机数采样，因此广播视图足够。

### 问题 5：为什么 CandidateGroup 不能只理解成 Static Batch？

**参考答案：** Static Batch 只描述“多行一起算”。CandidateGroup 还定义了共享 Prefix KV、共同 generation 生命周期、ragged row 退出、候选池统一过滤与 MMR，以及不足三条时的组级 refill。它是带业务语义和资源所有权的批处理单位。

## 11. 一句话复述

CandidateGroup 用多行 Page Table 表示八条逻辑序列，但让这些行的 Prefix 区间指向同一组物理 KV Page；只在首 token 之后为各分支分配独占 Suffix Page，从而把“八次完整生成”变成“一次 Prefill + 多路 Decode”。
