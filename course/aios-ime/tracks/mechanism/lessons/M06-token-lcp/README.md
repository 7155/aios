# M06：token-LCP Prefix KV 复用

> 源码固定点：`9f53740753de36899aa7694cf7dcb5304e58ea54`
>
> Canonical concept：C07
>
> 对应实战课：[P03](../../../practice/lessons/P03-continuous-typing/README.md)
>
> 核心源码：`python/aios/ime.py::token_longest_common_prefix`、`_prepare_prefix`、`_extend_prefill`

## 字符相同不等于模型输入相同

```text
旧文本：研究
旧 Token：[研究]

新文本：研究生
新 Token：[研究生]
```

字符串满足 `new.startswith(old)`，Token 序列却没有可复用的旧尾 Token。

安全条件：

```math
old\_ids[:k] = new\_ids[:k]
```

最大的 `k` 就是 Token Longest Common Prefix。

```mermaid
flowchart LR
    A[旧 Token IDs + Pages] --> C[token-LCP]
    B[新 Token IDs] --> C
    C --> D[保留 old_pages[:k]]
    C --> E[释放 old_pages[k:]]
    C --> F[为 new_ids[k:] 分配 Page]
    F --> G[positions 从 k 开始增量 Prefill]
```

## 1. LCP 实现

```python
def token_longest_common_prefix(left, right):
    length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        length += 1
    return length
```

例子：

```text
old = [BOS, 31, 45, 90, 12]
new = [BOS, 31, 45, 77, 81]
LCP = 3
```

保留前三页，释放旧尾两页，为新尾两 Token 分配 Page。

## 2. 四种状态转移

### 完全相同

```python
if reused == len(old_ids) == len(new_ids):
    page_table[:, :len(new_ids)] = old_pages.unsqueeze(0)
    return old_prefix_logits, old_pages, reused
```

同时复用全部 K/V 与最后位置 Logits，不执行 Forward。

### 只追加

```text
old = [1,2,3]
new = [1,2,3,4,5]
```

保留 3 页，只 Prefill `[4,5]`。

### 尾部重切或改写

```text
old = [1,2,3,4]
new = [1,2,8,9]
```

LCP=2，旧页 3/4 释放，新页 8/9 重新写入。

### 严格 Backspace

```text
old = [1,2,3,4]
new = [1,2,3]
```

当前实现把 `reused` 置 0，重新 Prefill。

## 3. Backspace 为什么不能只截 Page Table

系统缓存：

```text
所有历史 K/V
旧完整 Prefix 最后位置的 next-token logits
```

Backspace 后需要：

```text
Token 3 作为最后位置时的 next-token logits
```

K/V 不是最终 Logits。恢复任意历史末位 Logits还需要 Attention 输出、MLP、Final Norm 和 LM Head，当前并未逐位置缓存这些结果。

## 4. 增量 Prefill 的位置不能从零开始

```python
batch.input_ids = token_ids[cached_len:]
batch.positions = torch.arange(
    cached_len,
    len(token_ids),
    dtype=torch.int32,
    device=device,
)
batch.out_loc = extension_pages
```

若新增 Token 从 position 0 编号，它的 RoPE 坐标与历史 K/V 不在同一坐标系，结果会静默错误。

复用至少同时要求：

```text
Token identity
Position
Page Table
cached_len
模型与 dtype
```

## 5. Page 生命周期

```python
kept_pages = old_pages[:reused]
released_pages = old_pages[reused:]
free(released_pages)

extension_pages = allocate(len(new_ids) - reused)
prefix_pages = cat(kept_pages, extension_pages)
```

所以连续输入不会把每次按键的 Prefix Page 全部累积。只有当前最新 Prefix 的 Page 持久存在。

## 6. 正确不等于必然加速

短 Prefix 可能被固定成本主导：

```text
Python
Page allocator
FlashInfer metadata / plan
Kernel launch
同步
```

减少 Prefill Token 是算法收益；墙钟收益必须用长短 Prefix A/B 证明。课程不把 FLOPs 减少直接写成已测加速。

## 7. 与全局 Prefix Cache 的区别

当前是：

```text
单用户
只保存上一次 Prefix
精确 token-LCP
无全局 Hash / LRU / 多租户引用计数
```

通用跨用户 Prefix Cache 需要额外的隔离、Hash、引用计数、Eviction 和并发一致性，不能直接套用当前所有权。

## 验收

给出四组 Token IDs，写出：

```text
reused
kept_pages
released_pages
extension_pages
是否需要重新 Prefill
positions 起点
```

## 练习题

### 1. 为什么 K/V 不能恢复任意历史位置的 Logits？

<details>
<summary>参考答案</summary>

K/V 只是 Attention 的历史键和值；最终 Logits 还经过当前 Query Attention、残差、MLP、Final Norm 和 LM Head。缓存 K/V 不等于缓存每个位置的最终输出。
</details>

### 2. 若额外缓存每个位置的 Logits，可以优化 Backspace 吗？

<details>
<summary>参考答案</summary>

可以在严格 Token 边界回退时直接取历史末位 Logits，但每位置保存 `vocab_size` 向量很昂贵，还要处理 Tokenizer 重切与失效范围。
</details>

### 3. Page Size=16 时，LCP 落在 Block 中间怎么办？

<details>
<summary>参考答案</summary>

只能复用到上一个完整 Block 并重算稳定尾部，或实现部分 Block/Copy-on-write。当前 page_size=1 用更多元数据换取精确 Token 粒度。
</details>

### 4. 为什么复用后 `cached_len` 与 `positions` 必须一致？

<details>
<summary>参考答案</summary>

`cached_len` 决定逻辑历史长度，`positions` 决定 RoPE 坐标。两者不一致会让新 Token 在错误位置读取历史 KV。
</details>
