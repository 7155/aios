# Lesson 33：异步执行、Pinned Memory、正确计时与 Profiling

> 源码基线：`c335497c6bf67a4dc8cb5ba748ace7b7c1cb77af`
>
> 优化前必须先会测。否则你可能把“CPU 很快把 Kernel 排进队列”误认为“GPU 已经完成”，或者把首次 JIT、模型加载、Tokenizer、候选治理混进某个不相干的指标。

## 1. CUDA 为什么默认异步

CPU 与 GPU 是两个执行器：

```text
CPU：准备下一项工作
GPU：执行已经排队的 Kernel
```

若每发一个 Kernel 都等待，CPU/GPU 无法重叠。CUDA Stream 允许：

```python
y = x @ w1.T       # enqueue GEMM 1
z = y @ w2.T       # enqueue GEMM 2，依赖由 Stream 保序
```

Python 可以继续准备后续 Metadata，而 GPU 按顺序执行。

## 2. 何时发生隐式同步

常见同步点：

```python
value = cuda_tensor.item()        # CPU 要得到标量
print(cuda_tensor)                # 可能需要读 Device 数据
cuda_tensor.cpu()                 # D2H 拷贝后 CPU 使用
numpy_array = cuda_tensor.numpy() # CUDA Tensor 不能直接转
```

以及显式：

```python
torch.cuda.synchronize()
```

隐式同步散落在热路径中，会让 CPU 被迫等待并破坏 Pipeline。

## 3. AIOS 中的 Pinned Memory

FlashInfer Metadata：

```python
cpu_kwargs = {
    "device": "cpu",
    "dtype": torch.int32,
    "pin_memory": True,
}

cu_seqlens_q_gpu = cu_seqlens_q_cpu.to(
    device,
    non_blocking=True,
)
```

### 普通 CPU Memory

操作系统可以把页面换出或移动。GPU DMA 前通常需要先复制到页锁定 staging buffer。

### Pinned / Page-locked Memory

物理页面固定，CUDA 可直接 DMA，因此：

```python
.to(device, non_blocking=True)
```

才更可能与 CPU 工作重叠。

代价：Pinned Memory 是稀缺资源，过多会影响系统内存管理；小 Tensor 的固定开销可能大于收益。

## 4. AIOS Benchmark 的正确结构

当前 `bench_ime.py` 关键逻辑：

```python
torch.cuda.empty_cache()
llm = LLM(...)
engine = ImeCompletionEngine(llm)
torch.cuda.synchronize()
load_allocated = torch.cuda.memory_allocated() / 2**20

for prefix in warmup_prefixes:
    engine.complete(prefix, config)

torch.cuda.reset_peak_memory_stats()

for prefix in measured_prefixes:
    result = engine.complete(prefix, config)
    latencies.append(result.latency_ms)
```

它明确分开：

1. 模型加载；
2. 首次 Wrapper/JIT/Allocator；
3. Warmup；
4. Peak Memory Reset；
5. 正式样本。

报告中也写明完整 Top-3 计时包含：

```text
Prefix Prefill
Candidate Decode
Raw Logprob
CPU Decode
Filter/Dedup/MMR
Optional Refill
```

不包含模型加载与首次 JIT。

## 5. 为什么 Warmup 必须存在

第一次运行可能触发：

- Triton JIT Compile；
- FlashInfer Wrapper 初始化/Plan Cache；
- cuBLAS Algorithm/Handle 初始化；
- CUDA Context 创建；
- Memory Allocator 扩容；
- CPU 文件页缓存；

把第一次延迟放进普通 p50 会测到“冷启动 + 稳态”的混合值。

应该分别报告：

```text
Cold Start Latency
Steady-state p50/p95
```

当前 AIOS 的公开表是 Warmup 后稳态，不是冷启动。

## 6. CUDA Event 计时

用于纯 GPU 区间：

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
y = x @ w.T
end.record()
end.synchronize()
print(start.elapsed_time(end), "ms")
```

Event 被记录在 CUDA Stream 时间线上，`elapsed_time` 是两个 Event 之间 GPU 工作的时间。

但完整 IME Top-3 包含 CPU 候选治理，因此应使用端到端墙钟，而不是只用 GPU Event。

## 7. p50、p95 与 Mean

假设延迟：

```text
80,81,82,83,85,90,100,110,180,300 ms
```

Mean 会被两个慢样本显著拉高；p50 表示典型体验；p95 表示尾部风险。

输入法按键频繁，偶发 300ms 卡顿很明显，所以不能只看平均值或 tokens/s。

## 8. GPU Memory 三个概念

```python
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
torch.cuda.max_memory_allocated()
```

- Allocated：当前活跃 Tensor 实际占用；
- Reserved：PyTorch Caching Allocator 已向 CUDA 申请、可供复用的内存；
- Peak Allocated：Reset 后出现过的最大活跃占用。

`empty_cache()` 主要把未使用 Reserved Block 归还给 CUDA Driver，不会释放仍被 Tensor 引用的内存，也不会让模型本身变小。

## 9. Profiling 的三层工具

### PyTorch Profiler

回答：

```text
Python Op / ATen Op / CUDA Kernel 各耗时多少？
哪个调用栈发射了它？
```

示例：

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
) as prof:
    engine.complete(prefix, config)

print(prof.key_averages().table(
    sort_by="self_cuda_time_total",
    row_limit=30,
))
```

### Nsight Systems

回答时间线问题：

```text
CPU 是否来不及发射 Kernel？
Kernel 之间是否有大空洞？
H2D Copy 与 Kernel 是否重叠？
哪里发生同步？
```

典型命令：

```bash
nsys profile -t cuda,nvtx,osrt \
  -o reports/aios-ime \
  python benchmark/bench_ime.py --model /path/to/model
```

### Nsight Compute

回答单 Kernel 微架构问题：

```text
DRAM 带宽多少？
Tensor Core 利用率？
Occupancy？
Register/Shared Memory 是否限制？
Warp Stall 原因？
```

不要一开始就对整个程序用 Nsight Compute；先用 Systems 找 Top Kernel，再定点分析。

## 10. 如何判断三种瓶颈

### CPU / Launch-bound

迹象：

- Nsight 时间线 Kernel 很短，中间空洞明显；
- CPU 线程发射占比高；
- 增大 Batch 后效率大幅改善；
- CUDA Graph/Fusion 可能有效。

### Memory-bound

迹象：

- DRAM Throughput 高；
- Arithmetic Intensity 低；
- Elementwise/Scatter/Norm；
- Fusion、连续访问、减少中间 Tensor可能有效。

### Compute-bound

迹象：

- Tensor Core/SM 运算利用率高；
- 大 GEMM；
- 量化、更高效 GEMM、Tensor Parallel 可能有效。

一个完整请求可以在不同阶段分别属于三类。

## 11. 为什么短 Prefix LCP 几乎没加速

报告中：

```text
22 字短输入：1.00x
95 字长输入：1.056x
```

短 Prefix 少算几个 Token，但固定成本仍在：

```text
Tokenizer
Python 调度
Page 操作
Metadata 构造
FlashInfer plan/launch
```

这就是 Profile 指导决策：算法减少 FLOPs，并不保证墙钟下降。

## 12. README 内置代码：Profile 包装器

```python
from contextlib import contextmanager
import time
import torch

@contextmanager
def wall_clock_cuda(label):
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        print(label, (time.perf_counter() - start) * 1000, "ms")

with wall_clock_cuda("complete top3"):
    result = engine.complete(prefix, config)
```

完整产品路径适合这种墙钟。纯 Kernel A/B 则优先 CUDA Event 或 Triton `do_bench`。

## 13. 常见错误理解

### 错误：`non_blocking=True` 一定异步

只有来源/目标和 Stream 条件允许时才真正重叠；CPU 来源通常需 Pinned Memory。并且后续若立即读取结果，仍会同步。

### 错误：`empty_cache()` 会减少模型运行显存

它只释放 Allocator 中未使用的 Reserved Block，不会释放活跃 Tensor。

### 错误：Profiler 显示某 Kernel 最慢，就直接重写它

应先看它占端到端比例、调用次数和替代方案。优化一个 5% Kernel 两倍，理论总收益上限也很小。

## 14. 运行实验

```bash
python resources/lesson-33-async-benchmark-profiling/run_lesson33.py
```

CPU 实验展示 percentile 和 Amdahl 上限；CUDA 可用时额外比较同步/不同步计时。

## 15. 检验问题与参考答案

### 问题 1：什么时候使用 CUDA Event，什么时候用端到端墙钟？

**参考答案：** CUDA Event 适合纯 GPU 区间和 Kernel A/B，排除 CPU 工作；端到端墙钟适合用户实际路径，包括 Tokenizer、Python 调度、GPU、D2H、候选治理。AIOS Top-3 应以墙钟为主，Kernel 优化再用 Event。

### 问题 2：为什么 Pinned Memory 不是越多越好？

**参考答案：** 页锁定内存不能被正常换出，过多会压迫操作系统内存管理；分配也更贵。它适合反复复用的传输 Buffer，不应为每个很小、一次性对象无限创建。

### 问题 3：如何用时间线判断 CUDA Graph 可能有价值？

**参考答案：** 若大量 Kernel 很短，CPU 发射与 Kernel 间空洞占比明显，Shape/控制流又可固定，则 Graph Replay 减少 Launch 层级可能有价值；若单个大 GEMM 已占绝大多数时间，Graph 收益较小。

### 问题 4：为什么只报告 Mean 可能掩盖输入法体验问题？

**参考答案：** 少量高延迟请求可能不显著改变中位数，却造成明显卡顿。输入法是高频交互，尾部 p95/p99 直接影响体验，因此应同时报告 p50、p95 和失败/补采样分组。

## 16. 一句话复述

CUDA 异步让 CPU 与 GPU 重叠，但也让朴素 Python 计时失真。AIOS 优化必须区分冷启动与稳态、Kernel 时间与完整 Top-3 墙钟，并用 PyTorch Profiler→Nsight Systems→Nsight Compute 逐层定位 CPU Launch、显存带宽或计算瓶颈。

## 17. 用 NVTX 给时间线加业务标签

只看到一串 Kernel 名很难知道它属于 Prefix、Decode 还是 Refill。可在 Python 热路径增加 NVTX Range：

```python
with torch.cuda.nvtx.range("ime/prefix_prefill"):
    prefix_logits = engine.prefill(...)

with torch.cuda.nvtx.range("ime/candidate_decode"):
    raw = engine.generate_branch_batch(...)

with torch.cuda.nvtx.range("ime/governance_cpu"):
    selected = select_top_candidates(...)
```

Nsight Systems 时间线会显示这些区间，帮助把 GPU Kernel 与产品阶段对齐。

注意：NVTX 标记本身也有少量开销，适合 Profiling Build；正式极低延迟路径可通过开关控制。

## 18. 一个实际分析顺序

```text
1. 端到端 Benchmark：确认问题确实存在
2. PyTorch Profiler：找大类 Op 和 CPU/CUDA 比例
3. Nsight Systems：看 Launch 空洞、同步、H2D、Kernel 顺序
4. 选 Top 1～3 Kernel
5. Nsight Compute：看带宽、Tensor Core、Occupancy、Stall
6. 提出一个只改变单一机制的优化
7. Correctness + Frozen Eval
8. 重新测完整 Top-3 p50/p95
```

这能避免“看到一个 CUDA 技术就往项目里塞”的无证据优化。
