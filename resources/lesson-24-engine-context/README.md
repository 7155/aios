# Lesson 24：`Engine.run_batch()`、Global `Context` 与 LM Head

> 源码基线：`db343cbe07075c619d2519cb499c401f9edf895a`
>
> 目标：解释一个看起来奇怪的设计：`model.forward()` 没有任何参数，为什么所有层仍知道当前 Input、Position、Page 和 Attention Metadata？

## 1. Engine 是很薄的执行边界

```python
ctx = get_global_ctx()
with ctx.forward_batch(batch):
    logits = model.forward()
```

它没有重组 Batch，也不分配 Page；这些已经由 Scheduler 完成。

随后按每个 Req 的 `SamplingParams` 创建 Sampler，采样一个 Token。

## 2. Context Manager 保证“只在 Forward 期间可见”

```python
@contextmanager
def forward_batch(self, batch):
    assert self._batch is None
    try:
        self._batch = batch
        yield
    finally:
        self._batch = None
```

三个不变量：

1. Forward 前安装当前 Batch；
2. 正常/异常退出后都清空；
3. 不允许 Nested Forward 覆盖外层 Batch。

若没有 `finally`，模型异常后旧 Batch 可能残留，下一次 Forward 读取陈旧状态。

## 3. Model 如何读取数据

`Qwen3Model.forward()`：

```python
ctx = get_global_ctx()
input_ids = ctx.batch.input_ids
paged_kv_cache = ctx.kv_cache
batch = ctx.batch
```

Attention Backend 也用同一个 Context 读取：

```text
page_table
kv_cache dtype/device
```

这使算子签名更短，但把依赖转成隐式全局状态。

## 4. 为什么 LM Head 在 Prefill 只取 Last Position

Flat Prefill Hidden：

```text
请求 A 长 3
请求 B 长 5
→ hidden.shape = [8,H]
```

服务生成只需要每条 Prompt 最末位置：

```text
indices = [2,7]
```

LM Head：

```python
if batch.is_prefill and not return_all_logits:
    x = x[last_indices]
return linear(x, vocab_weight)
```

词表投影成本从：

```math
8 \times H \times V
```

降到：

```math
2 \times H \times V
```

对于大词表，这个节省很显著。

诊断/Teacher-forced Scoring 需要每个位置 Logits 时，设置 `return_all_logits=True`。

## 5. `last_logits = logits[:batch.size]` 为什么成立

- Prefill Serving：LM Head 已经输出 `[B,V]`；
- Decode：Flat 输入每请求一个 Token，本来就是 `[B,V]`；
- `return_all_logits=True` 的特殊评测路径不走通用 `Engine.run_batch()` 采样逻辑，而是直接消费全部 Logits。

所以通用 Engine 中切前 `B` 行符合当前契约，但这个隐式约定应由测试保护。

## 6. 每请求不同 SamplingParams

```python
for i, req in enumerate(batch.reqs):
    sampler = Sampler(req.sampling_params)
    tok = sampler.sample(last_logits[i:i+1])
```

同一 Forward 可以同时支持：

```text
请求 A：Greedy
请求 B：temperature=0.8, top_p=0.9
```

代价是 Python 循环逐请求采样；大 Batch 下可进一步融合或向量化。

## 7. Global Context 的并发边界

当前：

```text
_GLOBAL_CTX 进程级单实例
Context._batch 单一活动 Batch
单 CUDA Stream
```

若两个线程同时进入不同模型 Forward：

- `Nested forward_batch` 断言可能触发；
- 或需要外部锁串行化；
- 不适合自然的多模型、多 Device 并发。

可演进方向：

- `contextvars` / thread-local；
- 每个 Model 显式持有 Context；
- 将 Batch/Cache 参数显式传入；
- Runtime Host 为每 Device 管理独立执行上下文。

## 8. 运行实验

```bash
python resources/lesson-24-engine-context/run_lesson24.py
```

实验会验证正常、异常和嵌套 Context，证明 `finally` 清理与 Nested Guard 的意义。

## 9. 常见错误解释

### 错误：`model.forward()` 没参数，所以模型使用固定输入

错。输入来自当前 Context 的 Batch。

### 错误：Global Context 只是一种语法简化，不影响架构

错。它定义了单实例、不可嵌套、外部需串行化的并发边界。

### 错误：Prefill 所有位置 Logits 都必须计算

Serving 只需要各请求最后位置；完整 Logits 仅训练/评分/诊断需要。

## 10. 检验问题与参考答案

### 问题 1：为什么 Context 清理必须放在 `finally`？

**参考答案：** Forward 过程中任何一层都可能抛异常。如果只在正常返回后清空 `_batch`，异常会让旧 Batch 永久残留，下一次 Forward 可能误读陈旧输入，或因为 Nested Guard 认为已有活动 Batch 而无法继续。`finally` 保证正常、异常两条路径都恢复 Context 不变量。

### 问题 2：如何让两个 GPU 各有独立 Context？

**参考答案：** 可以让每个 Device/Engine 持有自己的 Context，并让 Model 显式引用对应实例；或者使用 thread/task-local `contextvars`，避免一个进程级 `_GLOBAL_CTX` 被所有模型共享。关键是把“当前 Batch 属于哪个 Device/Model”变成明确所有权。

### 问题 3：LM Head Last-index 优化在训练阶段为什么不能直接用？

**参考答案：** 自回归训练需要几乎每个有效位置都预测下一个 Token并计算 Loss，因此历史位置的 Logits 都是监督信号。Serving Prefill 只需要每条 Prompt 最后位置来采样首个输出 Token，才可以丢弃其他位置的词表投影。

### 问题 4：每请求 Python Sampler 循环何时会成为瓶颈？

**参考答案：** 当 Batch 很大、模型本身很小或 Decode 每步计算很短时，逐请求 Python 对象创建、分支判断和 Kernel 调用的固定开销会占比上升。此时可以把相同 Sampling 配置的请求分组，或将 Temperature/Top-k/Top-p 与抽样向量化到 GPU。

### 问题 5：隐式 Context 与显式参数传递怎样权衡？

**参考答案：** 隐式 Context 让每层函数签名简洁，减少大量 `batch/cache/metadata` 参数穿透；代价是依赖不显眼、测试需准备全局状态、并发所有权更难。显式传递更易推理和并发，但接口更长。选择取决于 Runtime 是否单实例以及未来并发需求。

## 11. 一句话复述

Engine 通过 Context Manager 临时绑定当前 Batch，Model/Attention/LM Head 从进程级 Context 读取执行状态；这让算子接口简洁并支持 Prefill Last-logit 优化，但形成单实例和并发串行化边界。
