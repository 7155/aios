# Lesson 14：跨按键 token-LCP Prefix KV 复用

> 源码基线：`bfc72896bbadab5c897672506d237c070900412e`
>
> 同一 CandidateGroup 内共享 Prefix 只解决“一次按键”。输入法还会连续输入：下一次按键的 Prefix 通常是上一次的延长。本课解释怎样跨按键保留稳定 Token 的 KV，以及为什么必须以 Token ID 而不是字符做判断。

![跨按键 token-LCP Prefix KV 复用](../../docs/images/aios-ime-prefix-kv.svg)

## 1. 先看连续输入

```text
t0：没关系，你先忙
t1：没关系，你先忙你的，
t2：没关系，你先忙你的，我
```

每次从头 Prefill 都会重复计算大部分历史 Token。

理想情况：

```text
旧 Prefix K/V 保留
只为新增 Token 执行增量 Prefill
```

但“新增字符”不能直接等同于“新增 Token”。

## 2. 为什么字符前缀相同不等于 Token 前缀相同

Unigram/BPE Tokenizer 会根据更长上下文重新切分尾部。

示意：

```text
旧文本：研究
旧 tokens: [研究]

新文本：研究生
新 tokens: [研究生]
```

虽然新字符串以旧字符串开头，但旧 token `[研究]` 并不是新 token 序列的前缀。

如果直接复用它的 KV：

```text
Cache 语义：模型看过 token “研究”
新输入语义：模型本应看 token “研究生”
```

Shape 全部合法，结果却错误。

因此安全条件是：

```math
\text{old\_ids}[:k] = \text{new\_ids}[:k]
```

最大这样的 `k` 就是 token Longest Common Prefix。

## 3. `token_longest_common_prefix` 做了什么

```python
def token_longest_common_prefix(left, right):
    length = 0
    for a, b in zip(left, right):
        if a != b:
            break
        length += 1
    return length
```

示例：

```text
old = [BOS, 31, 45, 90, 12]
new = [BOS, 31, 45, 77, 81]
LCP = 3
```

Page 操作：

```text
保留 old_pages[:3]
释放 old_pages[3:]
为 new_ids[3:] 分配新 Page
从 position=3 增量 Prefill
```

## 4. 四种 Prefix 变化分别怎样处理

### A. 完全相同

```text
old_ids == new_ids
```

可以同时复用：

- 全部 Prefix Page；
- 上次保存的 `prefix_logits`。

无需任何模型 Forward。

### B. 只在尾部追加

```text
old = [1,2,3]
new = [1,2,3,4,5]
```

保留前三页，只对 `[4,5]` 执行 `_extend_prefill`，位置从 3 开始。

### C. 尾部重切/改写

```text
old = [1,2,3,4]
new = [1,2,8,9]
```

LCP=2。释放旧 `[3,4]` 对应 Page，保留 `[1,2]`，对 `[8,9]` 增量 Prefill。

### D. 只 Backspace，变成旧序列的严格短前缀

```text
old = [1,2,3,4]
new = [1,2,3]
```

看似可以保留前三页，但当前实现会令 `reused=0`，重新 Prefill 短 Prefix。

为什么？

系统保存的是：

```text
旧 Prefix 全部 K/V
+ 旧最后位置 token 4 对应的 next-token logits
```

Backspace 后需要的是：

```text
token 3 作为最后位置时的 next-token logits
```

它并没有单独保存每个历史位置的 Logits；K/V 本身不能直接恢复该 Logits。因此当前选择重新 Prefill，保证分数精确。

这是一个重要检验点：

> KV Cache 保存历史 K/V，不等于保存每个历史位置的输出 Logits。

## 5. `_extend_prefill` 怎样保持位置正确

增量部分：

```python
batch.input_ids = token_ids[cached_len:]
batch.positions = arange(cached_len, len(token_ids))
```

如果错误地从 0 开始：

```text
新增 token 会获得错误 RoPE position
→ Attention 读到相同 K/V，却以错误位置编码查询
→ 结果静默错误
```

因此复用 Cache 时至少要同时保证：

```text
Token Identity 正确
Position 正确
Page Table 正确
Cached Length 正确
```

## 6. Prefix Page 生命周期

`_prepare_prefix` 处理旧 Page：

```text
kept_pages     = old_pages[:reused]
released_pages = old_pages[reused:]
free(released_pages)
allocate(new_len - reused)
prefix_pages = kept + extension
```

所以 Cache 不会随着每次按键无限增长；不再属于新 Prefix 的尾页立即释放。

`reset_prefix_cache()` 则释放当前全部持久 Prefix Page，用于：

- 输入上下文切换；
- Engine 关闭；
- GPU 测试验证无泄漏。

## 7. 为什么短 Prefix 几乎没有加速

报告：

| 输入 | 完整 Prefill | 增量 Prefill | 累计加速 |
|---|---:|---:|---:|
| 22 字短输入 | 7.69 ms/键 | 7.71 ms/键 | 1.00x |
| 95 字长上下文 | 7.96 ms/键 | 7.53 ms/键 | 1.056x |

短输入中，真正矩阵计算很少，固定开销占主导：

```text
Python 调度
Page 分配
FlashInfer metadata/plan
Kernel launch
同步与计时
```

所以少算几个 Token 并不一定降低墙钟。

这说明：

> 算法上减少 FLOPs，不等于在当前规模上必然得到可测加速。

## 8. 与通用 Prefix Caching 的区别

通用服务常用：

```text
对完整 Token Block 做 Hash
→ 在不同请求/用户之间复用公共 Prefix
```

当前 IME 更简单也更专用：

```text
只维护本地当前用户上一次 Prefix
→ 每次重新 Tokenize
→ 计算相邻输入精确 token-LCP
→ Page size=1，精确保留稳定 Token
```

没有全局 Hash Table、LRU 或跨用户 Cache Eviction。

## 9. 常见错误解释

### 错误：`new_text.startswith(old_text)` 就能复用全部旧 Cache

错。Tokenizer 可能重切尾部，必须比较 Token IDs。

### 错误：Backspace 只要截断 Page Table 就够了

当前不够，因为还需要新最后位置的 Logits；实现没有保存历史每格 Logits。

### 错误：减少 Prefill Token 一定能显著加速

错。短模型/短前缀可能被固定元数据和 launch 成本主导。

## 10. 运行实验

```bash
python resources/lesson-14-prefix-kv-reuse/run_lesson14.py
```

它会模拟相同、追加、尾部重切、Backspace 四种情况，并输出保留/释放/新分配的 Page。

GPU 测试：

```bash
AIOS_IME_MODEL=/path/to/model pytest -q \
  tests/test_ime_gpu.py::test_incremental_prefix_reuses_token_lcp
```

## 11. 检验问题与参考答案

### 问题 1：为什么 KV Cache 不能直接恢复历史任意位置的 next-token Logits？

**参考答案：** K/V 是 Attention 为后续 Query 准备的历史键和值，它们不是经过后续 Attention、MLP、Final Norm 和 LM Head 后得到的最终 Logits。要恢复“某个历史位置作为最后位置时”的 Logits，要么重新运行对应后缀计算，要么额外缓存当时的输出 Logits。

### 问题 2：若保存每个 Prefix 位置的 Logits，可以优化 Backspace 吗？代价是什么？

**参考答案：** 可以。严格 Backspace 到一个已有 Token 边界时，可以直接取该位置保存的 next-token Logits，无需重新 Prefill。但这会增加每个位置一份 `vocab_size` 级别的存储，远比 KV Page 昂贵；也要处理 Tokenizer 重切后哪些历史 Logits 仍然有效。

### 问题 3：Page Size=16 时，LCP 落在 Page 中间怎样处理？

**参考答案：** 若只能复用完整 Page，就只能保留到上一个完整 16-token 边界，Page 内剩余稳定 Token 需要重新计算；若想精确复用，就要支持部分 Page 或 Copy-on-write，复杂度上升。当前 page_size=1 用更多元数据换取了精确 Token 粒度。

### 问题 4：为什么增量 Prefill 仍需给新增 Token 构造正确 `positions`？

**参考答案：** 历史 K/V 已经带着原位置对应的 RoPE 信息。新增 Token 若从 position 0 重新编号，其 Q/K 会使用错误旋转角度，破坏与历史 K/V 的相对位置关系。复用内容的同时必须复用同一坐标系。

### 问题 5：跨用户 Prefix Cache 与本地相邻按键 LCP 的所有权有何不同？

**参考答案：** 本地 LCP 只有一个当前用户、一个上一 Prefix，所有权简单，可以直接替换尾页。跨用户 Prefix Cache 是共享全局资源，需要 Hash、引用计数、LRU/eviction、隔离和并发一致性，不能沿用单用户“只有一份当前 Prefix”的假设。

## 12. 一句话复述

跨按键复用必须在重新分词后比较 Token ID 的最长公共前缀，只保留语义与位置都稳定的 Page；追加和尾部改写可增量 Prefill，而严格 Backspace 因缺少新末位 Logits 当前重新 Prefill。优化对长 Prefix 有收益，对短 Prefix 可能被固定开销淹没。
