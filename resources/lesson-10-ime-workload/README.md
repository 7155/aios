# Lesson 10：输入法不是缩短版聊天服务

> 源码基线：`bfc72896bbadab5c897672506d237c070900412e`
>
> 这一课不急着读 `ImeCompletionEngine`。先解决一个更上层的问题：**为什么已经拥有 Continuous Batching、Paged KV Cache 的通用推理引擎，仍然需要为本地输入法重新定义调度对象、生命周期和指标？**

![AIOS-IME 与通用 serving 的 workload 差异](../../docs/images/aios-ime-vllm-comparison.svg)

## 1. 先明确产品输出，不要从类名开始

聊天服务通常收到一条请求，输出一段文本：

```text
用户 A：解释 AdamW
用户 B：写一段代码
用户 C：总结文章
```

系统关心：

- 多个用户的总吞吐；
- GPU 利用率；
- 请求排队、首 token、每 token 延迟；
- 多模型与长上下文支持。

输入法的一次按键则是：

```text
当前唯一用户输入：没关系，你先忙你的，

候选栏必须同时显示：
1. 我晚点给你发消息。
2. 我回去以后再回复你。
3. 等我忙完再联系你。
```

这里的交付单位不是“一条生成序列”，而是：

\[
\boxed{\text{一次按键的完整 Top-3 候选栏}}
\]

因此优化目标也必须改成：

```text
一次按键的 Top-3 p50 / p95
+ 满三条率
+ 三条互异率
+ 峰值显存
+ 新按键到来后旧结果是否立即失效
```

## 2. 为什么跨请求 Continuous Batching 不是当前核心收益

Continuous Batching 的典型问题是：

```text
GPU 上已经有请求 A、B
请求 C 到来
→ 不等待 A/B 全部完成
→ 在下一个 token step 加入 C
```

它提高的是**多个独立请求之间**的利用率。

但本地输入法的真实情况通常是：

```text
同一时刻只有当前用户的一次有效按键
旧按键一旦过时就应取消
不存在“用户 B 等待合批”
```

所以不能为了展示引擎能力，故意等待其他请求来组成 Batch。等待本身会增加候选栏延迟。

当前 IME 的主要并行性来自：

```text
一个 Prefix
→ 8 条候选分支
```

也就是**组内批处理**，而不是跨用户批处理。

### 面试追问

> 为什么不仍然把 8 条候选当成 8 个普通 request，交给通用 Scheduler？

因为它们拥有额外的共同不变量：

- Prefix 完全相同；
- 生命周期相同；
- 旧 generation 应整体作废；
- Prefix KV 应物理共享；
- 结果最终要共同参与去重与 MMR；
- 补采样是否发生取决于整个组是否已得到三条有效候选。

通用 request abstraction 看不到这些组级语义，因此难以进行最直接的资源复用和一次性治理。

## 3. 为什么只生成 3 路通常不够

候选栏需要 3 条，并不意味着只采样 3 路。

假设三条原始输出：

```text
row 0：我晚点给你发消息。        合法
row 1：作为一个AI，我建议……       助手模板，过滤
row 2：我晚点给你发消息           与 row 0 显示等价，去重
```

过滤后只剩 1 条。

所以系统区分：

```text
sampling_attempts = 8      # 首轮探索池
max_sampling_attempts = 12 # 最多补 4 路
display_candidates = 3     # 最终显示数
```

这是“候选池大小”和“显示数量”的区别。

当前 A/B 也证明：

| 方案 | Top-3 p50 / p95 | 满三条 |
|---|---:|---:|
| 固定 8 路、12 token | 87.72 / 140.90 ms | 93.33% |
| 8 路 + 不足时补 4 路 | **81.98 / 109.97 ms** | **100%** |
| 延长到 16 token | 115.55 / 225.71 ms | 86.67% |

结论不是“越多越好”，而是：

> 保持短输出上限；先用 8 路获得覆盖；只有结果不足时，才以更探索性的分布补少量分支。

## 4. 为什么 latest-wins 是产品语义，不只是性能优化

用户输入速度可能快于一次完整 Top-3 生成：

```text
时刻 t0：前缀 = “没关系”
时刻 t1：前缀 = “没关系，你”
时刻 t2：前缀 = “没关系，你先忙”
```

若 t0 的候选在 t2 后才返回，它即使模型算得完全正确，也已经是**错误产品结果**。

因此输入法请求不是普通 FIFO：

```text
new_generation()
→ 旧 CancellationToken.cancel()
→ generation_id + 1
→ 只有最新 generation 有资格返回
```

这里的 correctness 是：

\[
\text{returned generation id} = \text{latest generation id}
\]

而不是“所有请求最终都得到回答”。

## 5. 为什么完整墙钟比 token throughput 更重要

候选栏路径包括：

```text
Tokenizer
→ Prefix Prefill
→ 多路 Decode
→ batch_decode
→ 合法性过滤
→ 显示去重
→ 软惩罚
→ MMR Top-3
→ 必要时 refill
```

只测：

```text
一次 Prefix Prefill + 首 token ≈ 8 ms
```

不能得出：

```text
完整 Top-3 ≈ 8 ms
```

当前报告中，完整 Top-3 p50/p95 是：

```text
81.98 / 109.97 ms
```

这才接近用户看到候选栏的等待时间。

## 6. 当前系统的产品合同

| 约束 | 运行时决定 |
|---|---|
| 单用户、本地按键 | 不等待跨请求合批 |
| 一次必须显示三条 | CandidateGroup 先 8 路，必要时补 4 路 |
| 多路拥有同一 Prefix | 只 Prefill 一次并共享 Prefix Page |
| 分支结束时间不同 | Ragged Decode，完成行退出 active rows |
| 新按键使旧结果过时 | latest-wins 取消整个旧组 |
| 相邻按键 Prefix 高度重合 | token-LCP 增量 Prefill |
| 候选必须可直接显示 | 中文过滤、去重、原始分数与 MMR |

## 7. 源码地图

| 问题 | 入口 |
|---|---|
| CandidateGroup 主流程 | `python/aios/ime.py::ImeCompletionEngine.complete` |
| 组内采样 | `ImeCompletionEngine._generate_branch_batch` |
| Prefix Cache | `ImeCompletionEngine._prepare_prefix` |
| latest-wins | `new_generation`、`CancellationToken` |
| 原始 logprob 采样 | `python/aios/engine/sample.py` |
| 完整 Top-3 Benchmark | `benchmark/bench_ime.py` |
| 冻结质量 Lane | `scripts/eval_aios_ime_frozen.py` |

## 8. 常见错误解释

### 错误 1：输入法就是把聊天输出长度改成 12

错。候选组、结果完整率、latest-wins、组内共享 KV 和候选共同排序，都不是 `max_tokens=12` 能表达的。

### 错误 2：AIOS 有 Continuous Batching，所以输入法一定从它获益

错。当前单用户场景没有等待合批的第二个用户。收益主要来自组内分支并行和 Prefix 复用。

### 错误 3：满三条率 100% 就说明模型准确

错。它只说明结构完整。语义仍需冻结排序指标和人工 accept/reject。

## 9. 运行实验

```bash
python resources/lesson-10-ime-workload/run_lesson10.py
```

它会比较“3 路直接显示”和“8 路候选池后选 3 条”的结构可靠性，并打印一次按键完整路径的计时边界。

## 10. 一句话复述

AIOS-IME 不是把通用 serving 缩短，而是把调度单位从“多个用户请求”改成“当前按键的 CandidateGroup”，把正确性改成 latest-wins，把性能改成完整 Top-3 尾延迟，把内存复用改成同组 Prefix Page 与跨按键 token-LCP。
