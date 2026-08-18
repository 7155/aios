# AIOS CUDA 与推理优化长篇课程

> 固定源码基线：`1d63bca4cf24885a1b15897003e3481db53d8ada`（`main`）
>
> 默认读者没有学过 CUDA，也不想来回跳源码。每节 README 直接包含必要代码、逐行解释、Tensor Shape、地址、GPU 资源、可运行实验、常见错误理解和带答案的检验问题。

## 当前 AIOS 的 GPU 路径

```text
AIOS Python 控制面
├─ PyTorch / F.linear → cuBLAS / CUDA kernels
├─ FlashInfer → RMSNorm、SwiGLU、Paged Attention kernels
└─ Triton → AIOS 自定义 KV Cache scatter kernel
```

## 学习阶段

### A. 先建立 GPU 执行与内存直觉

| 课 | 内容 |
|---:|---|
| 29 | CUDA Host/Device/Kernel/Block/Warp/Stream |
| 37 | SM、Warp Scheduler、Occupancy、寄存器压力 |
| 38 | HBM/L2/Shared/Register、Coalescing、Bank Conflict |

### B. 看懂当前 AIOS 的核心 Kernel 和算子

| 课 | 内容 |
|---:|---|
| 30 | Triton KV Scatter 完整代码 |
| 31 | QKV/Gate-Up/SwiGLU/Residual Fusion |
| 32 | FlashInfer Paged Attention `plan/run` |
| 39 | `F.linear`、GEMM、Tensor Core、Prefill vs Decode |
| 41 | FlashAttention Online Softmax 算法 |
| 42 | Triton Matmul Tile 与 Autotune |

### C. 学会 Profile 与管理运行时

| 课 | 内容 |
|---:|---|
| 33 | 异步、Pinned Memory、计时、Nsight |
| 40 | Roofline、Latency/Memory/Compute 瓶颈 |
| 43 | Caching Allocator、碎片、Peak Memory |

### D. 未来优化与扩展

| 课 | 内容 |
|---:|---|
| 34 | CUDA Graph（当前未实现） |
| 35 | 量化（当前未实现） |
| 36 | Speculative Decoding（当前未实现） |
| 44 | Tensor Parallel/NCCL（当前未实现） |

## 学习方法

```text
先看图和固定例子
→ 阅读 README 内代码
→ 手算 Shape/地址/字节/FLOPs
→ 运行 CPU 机制实验
→ 有 CUDA 时按课运行可选 GPU 路径
→ 完成检验问题并对照答案
→ 最后回到完整 Top-3 p95 与质量门禁
```

## 一键验证

```bash
python resources/cuda-optimization/validate_cuda_course.py
```

## 事实边界

- 当前实现：PyTorch CUDA、FlashInfer、Triton KV Scatter、Fused Projection/Norm/Activation、Paged KV。
- 当前未实现：CUDA Graph、权重量化、KV 量化、Speculative Decoding、Tensor Parallel。
- FlashAttention/Matmul/TP 课程用于解释当前依赖和未来设计，不是新增运行时代码。
- 真实收益必须在固定硬件、模型、输入、Warmup 和完整 Top-3 口径下重新测。
