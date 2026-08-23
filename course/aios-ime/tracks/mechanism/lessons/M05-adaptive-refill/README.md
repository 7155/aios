# M05：过滤后有效产出驱动的自适应补采样

> 源码固定点：`9f53740753de36899aa7694cf7dcb5304e58ea54`
>
> Canonical concept：C06
>
> 对应实战课：[P02](../../../practice/lessons/P02-candidate-budget/README.md)
>
> 核心源码：`python/aios/ime.py::candidate_pool_stats`、`adaptive_refill_attempts`、`_complete_locked`

## 从旧思路到当前思路

旧思路：

```text
首轮 8 路不足
→ 固定再补 4 路
```

当前思路：

```text
首轮或上一轮结束
→ 看过滤与显示去重后的 unique-valid yield
→ 根据 Top-3 deficit 估算本轮需要多少新分支
→ 受 2～8 路、总预算 24 和 deadline 限制
```

```mermaid
flowchart TD
    A[当前候选池] --> B[统计 attempts / valid / unique / invalid / duplicate]
    B --> C[deficit = 3 - unique_valid]
    C --> D[估算 predicted = ceil(deficit / yield)]
    D --> E[加入 diversity headroom]
    E --> F[裁到 2~8 路和剩余总预算]
    F --> G[新 seed + 更高探索 + 重复规避]
    G --> H[生成下一轮]
    H --> A
```

## 1. 源码核心

```python
def adaptive_refill_attempts(candidates, config, attempts_used):
    remaining = config.max_sampling_attempts - attempts_used
    if remaining <= 0:
        return 0

    stats = candidate_pool_stats(candidates)
    deficit = (
        config.display_candidates
        - stats.valid_unique_candidates
    )
    if deficit <= 0:
        return 0

    yield_floor = 1.0 / max(1, config.refill_batch_size)
    observed_yield = (
        stats.valid_unique_candidates / stats.attempts
        if stats.attempts
        else yield_floor
    )
    effective_yield = max(yield_floor, observed_yield)
    predicted = math.ceil(deficit / effective_yield)
    diversity_headroom = deficit * 2

    planned = max(
        config.min_refill_batch_size,
        diversity_headroom,
        predicted,
    )
    return min(
        planned,
        config.refill_batch_size,
        remaining,
    )
```

## 2. 手算三个例子

默认：

```text
display=3
refill_batch_size=8
min_refill_batch_size=2
```

### 例 A：8 路得到 2 条不同有效候选

```math
deficit = 1
```

```math
yield = 2/8 = 0.25
```

```math
predicted = \lceil 1/0.25 \rceil = 4
```

```math
diversity\ headroom = 2
```

本轮：

```text
max(2, 2, 4) = 4 路
```

### 例 B：8 路只得到 1 条不同有效候选

```math
deficit=2,\quad yield=1/8
```

```math
predicted=\lceil2/(1/8)\rceil=16
```

受单轮上限裁成 8 路。

### 例 C：8 路零有效候选

实际 yield 为 0，但用：

```math
yield\ floor = 1/8
```

避免除零。缺口 3 的预测是 24，仍裁成 8 路。

## 3. 为什么要有 `diversity_headroom`

若只按“缺一条就补一条”，新分支可能：

```text
非法
与旧候选显示重复
与已有候选高度同质
```

`deficit * 2` 给治理留出选择空间。它不是概率理论最优值，而是有界工程余量，必须靠冻结数据调整。

## 4. 每轮探索会逐步增强

```python
temperature = min(
    max_refill_temperature,
    refill_temperature
    + (refill_rounds - 1) * refill_temperature_step,
)

top_k = min(
    max_refill_top_k,
    refill_top_k
    + (refill_rounds - 1) * refill_top_k_step,
)
```

默认趋势：

```text
temperature: 0.75 → 0.85 → 0.95
top-k:       96   → 128  → 160
```

首轮仍使用更稳定的 `0.35 / top-k 50`。只有候选覆盖不足时才扩大探索。

## 5. 重复规避不是禁止相同首字

已有候选会被 Tokenize 成完整序列。新分支只有在已生成前缀与旧序列相同，且走到需要分叉的位置时，才屏蔽“继续完整复现旧序列”的下一个 Token。

```text
允许：
“我晚点给你发消息”
“我晚点再回复你”

禁止：
第二轮再次逐 token 完整复现
“我晚点给你发消息”
```

这样保留自然的共同开头，不粗暴强迫首 token 不同。

## 6. 三种停止原因

```text
filled        已选出三条不同有效候选
max_attempts  用完总探索预算
deadline      到达墙钟恢复预算
cancelled     新 generation 使旧组失效
```

`refill_stop_reason` 必须进入结果和 Benchmark。没有停止原因，就无法区分模型失败、预算策略和用户新输入。

## 7. 为什么每轮后先释放 suffix Page

当前流程：

```text
生成本轮
→ 文本/分数物化
→ 释放本轮 suffix KV
→ 治理候选池
→ 决定是否下一轮
```

所以总预算 24 路不会让 24 路 suffix 同时占显存。若未来要跨轮复用分支隐藏状态，所有权模型必须重新设计。

## 验收

手算下面三种输入的下一轮路数：

| 已用 attempts | unique-valid | remaining | 计划路数 |
|---:|---:|---:|---:|
| 8 | 2 | 16 |  |
| 8 | 1 | 16 |  |
| 16 | 2 | 8 |  |

并说明何时返回 0。

## 练习题

### 1. 为什么 observed yield 使用 `unique_valid / attempts`，而不是 `valid / attempts`？

<details>
<summary>参考答案</summary>

最终显示要求不同候选。多个合法但显示等价的分支不能填补 Top-3 缺口，因此产出率必须在显示去重之后计算。
</details>

### 2. `yield_floor` 太高会怎样？

<details>
<summary>参考答案</summary>

系统会过度乐观，零或低产出候选池只补很少分支，难以恢复完整 Top-3。太低则容易每轮直接打满上限，增加尾延迟。
</details>

### 3. 为什么探索参数逐轮增加，而不是首轮就用最高 temperature？

<details>
<summary>参考答案</summary>

首轮希望稳定、常见且高概率的短补全；只有覆盖不足时才支付更高随机性。最高温度作为恢复路径能减少普通样本的质量波动。
</details>

### 4. 为什么修改硬过滤规则会改变 refill 性能？

<details>
<summary>参考答案</summary>

过滤规则决定 unique-valid yield 和 deficit，从而直接决定是否补采样、补多少轮及完整 Top-3 尾延迟。
</details>
