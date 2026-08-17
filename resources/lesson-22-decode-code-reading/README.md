# Lesson 22：`DecodeManager` 与逐 Token 推进

> 源码基线：`db343cbe07075c619d2519cb499c401f9edf895a`
>
> 目标：看懂为什么 Decode Batch 每个请求恰好贡献一个 Flat Token、采样 Token 与 KV 写入相差一轮，以及 EOS 时哪些 Page 实际存在。

## 1. Prefill 后的状态

Prompt：

```text
[BOS,A,B]
```

Prefill Forward 计算了这 3 个 Token 的 K/V，并采样 `C`。

状态：

```text
Token Pool: [BOS,A,B,C]
cached_len = 3
device_len = 4
extend_len = 1
```

`C` 已经知道，但还没有 K/V；下一轮 Decode 输入正是 `C`。

## 2. Decode Batch 为什么每请求一行

所有 Running Req 通常满足：

```math
E_i = D_i - C_i = 1
```

所以 `_make_positions` 为每条请求产生一个位置，Flat `input_ids` Shape：

```text
[B]
```

模型 Embedding 后：

```text
[B,H]
```

而不是 `[B,1,H]`。AIOS 的执行格式从一开始就是 Flat Token。

## 3. 一轮 Decode 的精确时间线

假设当前输入 `C`：

```text
1. 为 position=3 分配 Page
2. Model(C) 写入 C 的 K/V
3. C 作为 Query 读取 Prefix+C 的 K/V
4. LM Head 得到 P(next | BOS,A,B,C)
5. Sampler 采样 D
6. complete_one：cached_len=4, device_len=5
7. Token Pool position 4 写 D
```

下一轮才把 `D` 写进 KV。

## 4. EOS 为什么通常不需要写 KV

若步骤 5 采样到 EOS：

```text
EOS 被写入 Token Pool 和 generated
但请求立刻完成
不会有下一轮 Forward
所以 EOS 不需要 K/V Page
```

资源回收使用：

```python
used_pages = page_table[slot, :cached_len]
```

此时 `cached_len` 只覆盖已经实际 Forward 过的 Token，不包含刚采样的 EOS。

这是 `cached_len/device_len` 分开的直接价值。

## 5. Output Budget 如何结束请求

初始：

```math
M = T + O
```

每采样一个 Token，`D` 增 1；当：

```math
D = M
```

则：

```math
R = M - D = 0
```

`can_decode=False`，即使没有 EOS 也结束。

## 6. `inflight_tokens` 为什么要预留未来

```python
sum(req.remain_len for req in running_reqs)
```

表示所有 Running 请求最多还会新增多少逻辑 Token。接纳新 Prefill 时把它作为已承诺容量，避免运行到一半才发现 Page 不够。

`page_size>1` 时还额外考虑每请求最后 Block 可能未满的保留；当前 Page Size=1，所以额外项为 0。

## 7. Running Set 使用 Set 的影响

优点：

- 按对象身份快速添加/删除；
- 自然去重。

边界：

- 迭代顺序不构成 API 保证；
- 通用 Sampler 使用全局 RNG 时，执行顺序可能影响请求对应随机数；
- 最终结果通过 UID 排序只恢复输出顺序，不恢复采样随机流。

IME CandidateGroup 专门用 Stateless Stream 解决了自己的分支确定性；通用 Scheduler 当前没有同等级 request-local RNG 合同。

## 8. 运行实验

```bash
python resources/lesson-22-decode-code-reading/run_lesson22.py
```

实验逐轮显示 Token Pool、Cached/Device Length、当前输入、采样输出和 EOS 时实际释放的 Page 区间。

## 9. 常见错误解释

### 错误：采样出 Token 后，它已经在 KV Cache 中

错。它只进入 Token Pool；下一轮作为输入 Forward 后才有 K/V。

### 错误：EOS 也必须先写 KV 才能结束

错。EOS 已经是当前分布的决策结果，不需要再用它预测下一 Token。

### 错误：Decode Batch Shape 一定是 `[B,1]`

AIOS 使用 Flat Token，输入是 `[B]`，Head 维度在模型内部建立。

## 10. 检验问题与参考答案

### 问题 1：为什么 `complete_one()` 后 `extend_len` 又变成 1？

**参考答案：** 本轮 Forward 结束后，原来的未缓存 Token 全部进入 KV，所以先令 `cached_len=device_len`；随后 Sampler 又产生一个新 Token，`device_len` 增 1。于是新的 `device_len-cached_len=1`，正好表示“下一轮只有刚采出的这个 Token 需要 Forward”。

### 问题 2：若 Sampling 后立即把 `cached_len` 也增加到包括新 Token，会导致什么？

**参考答案：** 系统会误以为新 Token 的 K/V 已经存在，下一轮 `extend_len` 会变成 0，从而跳过该 Token 的 Forward。后续 Attention 会缺失它的 K/V，Page Table 也不会为它分配写入位置，状态机语义被破坏。

### 问题 3：Set 顺序如何影响随机采样可复现性？

**参考答案：** 若每轮按 Set 当前遍历顺序从一个全局 RNG 连续取随机数，请求顺序变化会让不同请求拿到不同随机数，即使它们的 UID 和输入没变。要获得稳定复现，应把随机流绑定到请求身份/step，或使用每请求独立 Generator，而不是依赖 Batch Row 顺序。

### 问题 4：Beam Search 下 `Req` 与 Page 所有权要怎样扩展？

**参考答案：** 一条逻辑请求会分成多个 Beam 分支，它们共享 Prefix KV、随后拥有各自 Suffix KV，并可能在每轮重新排序、复制或淘汰。`Req` 需要从“单序列状态”扩展为 request + beam states，Page 管理还要支持共享引用和分支释放，必要时引入引用计数或 Copy-on-write。

### 问题 5：Speculative Decoding 一轮接纳多个 Token 时，`extend_len` 是否仍为 1？

**参考答案：** 不一定。若一次接受了多个 Draft Token，这些已进入逻辑序列但尚未写入主模型 KV 的 Token 可以形成一个长度大于 1 的 extension，主模型下一轮可能一次验证/写入多个位置。因此当前“一轮领先一格”的不变量是普通单 Token 自回归 Decode 的特例。

## 11. 一句话复述

Decode 的每轮输入是上一轮刚采样但尚未缓存的一个 Token；Forward 写入其 K/V 后再采样下一个 Token，因此 Token Pool 总比 KV Cache 领先一格，EOS/预算完成时最后采样 Token 无需进入 Cache。
