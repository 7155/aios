# AIOS CUDA 与推理优化长篇课程

> 固定源码基线：`c335497c6bf67a4dc8cb5ba748ace7b7c1cb77af`（`main`）
>
> 这条路线默认读者**没有学过 CUDA，也不想来回跳源码**。每节 README 直接包含必要源码、逐行解释、张量 Shape、内存地址、运行顺序、可运行实验、常见错误理解以及带答案的检验问题。

## 先说清楚：AIOS 里到底有哪些“CUDA 代码”

当前 AIOS 没有手写 `.cu` CUDA C++ 文件。GPU 路径分三层：

```text
AIOS Python 控制面
├─ PyTorch / F.linear
│  └─ 底层调用 cuBLAS / CUDA kernels
├─ FlashInfer
│  └─ 提供 RMSNorm、SwiGLU、Paged Attention 等高性能 kernels
└─ Triton
   └─ AIOS 自己写了 KV Cache scatter kernel：python/aios/kernel/store.py
```

所以课程不会假装“读完 CUDA C++”。它会分别解释：

1. Python 语句如何异步发射 GPU Kernel；
2. Triton Kernel 每个 Program 实例处理什么数据；
3. FlashInfer `plan()` / `run()` 如何把 Page Table 变成 Paged Attention；
4. Fusion 为什么减少 Kernel Launch 与显存往返；
5. CUDA Graph、量化和 Speculative Decoding 为什么是下一阶段，而不是当前已实现能力。

## 课程顺序

| 课次 | 教材 | 最终能回答的问题 |
|---:|---|---|
| 29 | [CUDA 从零：Host、Device、Kernel、Block、Warp 与显存](../lesson-29-cuda-foundations/README.md) | 一行 PyTorch 为什么不是“一行就算完”，CPU 与 GPU 谁在等待谁？ |
| 30 | [逐行读懂 Triton `store_cache` Kernel](../lesson-30-triton-kv-store/README.md) | `program_id/arange/mask/stride/load/store` 各自到底做什么？ |
| 31 | [Operator Fusion 与显存流量](../lesson-31-fusion-memory-traffic/README.md) | QKV、Gate-Up、SwiGLU、Residual+RMSNorm 为什么要融合？ |
| 32 | [FlashInfer Paged Attention 完整代码路径](../lesson-32-flashinfer-paged-attention/README.md) | `cu_seqlens/indices/workspace/plan/run` 怎样进入 GPU Attention？ |
| 33 | [异步执行、Pinned Memory、正确计时与 Profiling](../lesson-33-async-benchmark-profiling/README.md) | 为什么不 `synchronize()` 会测到假延迟，如何判断 CPU-bound/Memory-bound/Compute-bound？ |
| 34 | [CUDA Graph：为什么适合小模型，又为什么 AIOS 还没接](../lesson-34-cuda-graphs/README.md) | Capture/Replay 的固定 Shape、固定地址约束如何与 Ragged Decode 冲突？ |
| 35 | [量化：BF16、FP8、INT8、INT4 与 KV Cache](../lesson-35-quantization/README.md) | 权重、激活、KV 分别怎样量化，Scale/Zero-point 在哪里使用？ |
| 36 | [Speculative Decoding、MTP 与下一步优化路线](../lesson-36-speculative-roadmap/README.md) | Draft/Verify 为什么可能一次接受多个 Token，何时反而更慢？ |

## 推荐学习方法

```text
先看图和固定例子
→ 阅读 README 内的完整核心代码
→ 对照逐行解释
→ 运行 CPU 实验
→ 有 CUDA 时运行可选 GPU 分支
→ 完成检验问题
→ 对照参考答案修正理解
```

## 一键验证课程结构与 CPU 实验

```bash
python resources/cuda-optimization/validate_cuda_course.py
```

## 事实边界

- 当前实现：PyTorch CUDA、FlashInfer、Triton KV scatter、Fused Projection/Norm/Activation、Paged KV。
- 当前未实现：CUDA Graph、权重量化、KV 量化、Speculative Decoding、Tensor Parallel。
- README 中这些未实现部分都是**带代码骨架的设计课程**，不是性能声明。
- 真实 GPU 收益必须在固定模型、输入、硬件、Warmup 和完整 Top-3 计时口径下重新测。
