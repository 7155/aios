# Lesson 19：`SamplingParams`、`Req`、`Batch` 与 `Context`

> 源码基线：`db343cbe07075c619d2519cb499c401f9edf895a`
>
> 目标：看懂 AIOS 运行时最重要的四个数据对象。它们不是普通“字段集合”，而是把请求生命周期、一次 Forward 和全局执行环境分成三个不同层次。

## 1. `SamplingParams`：策略，不是请求进度

```python
@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024
```

它描述“如何从 Logits 选 Token”和“最多生成多少”，本身不记录已经生成几步。

`is_greedy`：

```python
return (temperature <= 0.0 or top_k == 1) and top_p == 1.0
```

注意 `top_k=1` 即使 temperature 大于 0，候选集合仍只有一个 Token，所以结果仍是 Greedy。

## 2. `Req`：一条请求的可变运行时状态

最关键字段：

```text
input_ids       初始 Prompt Tensor（主要保留身份/长度）
cached_len      已经写入 KV Cache 的 Token 数
device_len      当前 Token Pool 中已经存在的 Token 数
max_device_len  Prompt + 最大输出长度
created         generated 保存已采样 Token，Page 释放后仍存在
```

必须始终满足：

```math
0 \le C \le D \le M
```

其中：

- `C = cached_len`；
- `D = device_len`；
- `M = max_device_len`。

两个派生长度：

```math
R = M - D
```

```math
E = D - C
```

- `R` 是还能采样多少 Token；
- `E` 是这次 Forward 还需要把多少 Token 写进 KV。

### 完整状态例子

Prompt 长 4，最多输出 3：

```text
刚创建： C=0, D=4, M=7 → extend=4, remain=3
Prefill 后采到 x1：C=4, D=5 → extend=1, remain=2
Decode x1 后采到 x2：C=5, D=6 → extend=1, remain=1
Decode x2 后采到 x3：C=6, D=7 → extend=1, remain=0
```

这里非常关键：**新采样 Token 已写入 Token Pool，但它的 K/V 要到下一次 Forward 才生成。** 所以运行中常见 `device_len = cached_len + 1`。

## 3. `complete_one()` 为什么先改 cached，再增加 device

```python
self.cached_len = self.device_len
self.device_len += 1
```

这表示：

1. 刚刚 Forward 的输入部分全部已经进入 KV；
2. Sampler 又产生了一个新 Token；
3. 新 Token 进入 Token Pool，但尚未进入 KV。

若先 `device_len += 1` 再赋给 `cached_len`，就会错误声称新采样 Token 已经有 K/V。

## 4. `Batch`：只代表一次 Forward

`Batch` 不是请求队列。它只描述当前这一轮：

```text
reqs          哪些请求参加
phase         prefill / decode
input_ids     本次真正送模型的 Flat Token
positions     这些 Token 的绝对位置
out_loc       本次 K/V 要写到哪些物理 Page
attn_metadata Attention Kernel 所需边界与 Page 索引
```

Prefill Batch：每个请求 `extend_len` 可以大于 1。

Decode Batch：每个请求通常 `extend_len=1`。

## 5. `Context`：为什么 `model.forward()` 不需要参数

`Context.forward_batch(batch)`：

```python
try:
    self._batch = batch
    yield
finally:
    self._batch = None
```

Engine：

```python
with ctx.forward_batch(batch):
    logits = model.forward()
```

Model 内部：

```python
ctx = get_global_ctx()
input_ids = ctx.batch.input_ids
```

所以 Batch 通过 Context 隐式传递。

优点：所有 Layer 不必层层传 `input_ids/page_table/metadata`。

代价：

- 当前是进程级 Singleton；
- 不支持嵌套 Forward；
- 多模型/多线程需要更明确的 Context 所有权；
- 测试必须先安装 Global Context。

## 6. 为什么 `Req` 使用 `@dataclass(eq=False)`

`DecodeManager.running_reqs` 是 `Set[Req]`。

普通 dataclass 若生成字段相等比较，包含 Tensor 时既不适合作 Hash，也可能触发 Tensor 布尔歧义。`eq=False` 保留对象身份语义：

```text
两个字段值相同的 Req
仍是两个不同运行请求
```

因此可以放入 Set，并按对象身份移除。

## 7. 运行实验

```bash
python resources/lesson-19-runtime-data-model/run_lesson19.py
```

它会逐步打印 `C/D/M/E/R`，并故意演示错误更新顺序怎样让系统误以为新 Token 已经缓存。

## 8. 常见错误解释

### 错误：`device_len` 是 GPU 已缓存长度

错。它是 Token Pool 中当前序列长度；真正已缓存的是 `cached_len`。

### 错误：Batch 会跨多个 Forward 保存

错。每轮 Scheduler 都会重新构造 Batch；长期状态在 Req、Manager 和 KV Pool。

### 错误：Global Context 只是方便写代码，没有约束

错。它形成单实例、不可嵌套的执行假设，是未来并发扩展的重要边界。

## 9. 面试追问

1. 为什么 `generated` 必须独立保存，不能最终从 Token Pool 读取？
2. 请求完成后 Page/Slot 立即释放，结果为什么仍能返回？
3. `remain_len=0` 时，最后采样 Token 是否已经写入 KV？为什么通常没有必要？
4. 如果把 `running_reqs` 从 Set 改成 List，有哪些确定性与复杂度变化？
5. 怎样把 Global Context 改造成多模型、多线程安全？

## 10. 一句话复述

`Req` 保存跨轮生命周期，`Batch` 只保存一次 Forward 的 Flat 输入和写入位置，`Context` 在执行期间把当前 Batch 暴露给所有算子；`cached_len` 与 `device_len` 的一格差异正是自回归 Decode 状态机的核心。
