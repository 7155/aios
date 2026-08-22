# AIOS CUDA 与推理优化长篇课程

> 固定 AIOS 源码基线：`1d63bca4cf24885a1b15897003e3481db53d8ada`（`main`）
>
> Triton FlashAttention 专题固定外部源码：`hkproj/triton-flash-attention@296ee44c8a238cd2192d13e22e9082251f1c1289`。
>
> 默认读者没有学过 CUDA，也不想来回跳源码。每节 README 直接包含必要代码、逐行解释、Tensor Shape、地址、GPU 资源、可运行实验、常见错误理解和带答案的练习题。

## 当前 AIOS 的 GPU 路径

```text
AIOS Python 控制面
├─ PyTorch / F.linear → cuBLAS / CUDA kernels
├─ FlashInfer → RMSNorm、SwiGLU、Paged Attention kernels
└─ Triton → AIOS 自定义 KV Cache scatter kernel
```

Lesson 45～50 分析的是一个独立的 Dense FlashAttention 2 教学实现：

```text
连续 Q/K/V
→ Triton Forward Online Softmax
→ Autograd Wrapper
→ Triton Backward 重算
```

它用于学习底层机制，**没有接入 AIOS 当前 Runtime，也不能直接替代带 Varlen、Paged KV、GQA 与 Decode Wrapper 的 FlashInfer 路径**。

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

### E. Triton FlashAttention 源码实战

> 推荐先完成 Lesson 41 的 Online Softmax 与 Lesson 42 的 Triton Tile/Autotune，再进入本专题。

| 课 | 内容 | 学完后能回答 |
|---:|---|---|
| 45 | 项目总览、调用链、显存与边界 | 一个约 700 行文件怎样串起 Forward、Backward 与 PyTorch Autograd？ |
| 46 | `_attn_fwd_inner` 源码带读 | `m_i/l_i/O_block/alpha` 怎样实现精确分块 Softmax？ |
| 47 | Grid、Stride、Block Pointer、Causal Stage、Autotune | 每个 Program 负责哪块 Q，怎样找到 K/V，并跳过完整未来 Tile？ |
| 48 | Backward 数学与 `D_i` | 只保存 O 与 LSE，怎样重算 P 并推导 `dQ/dK/dV`？ |
| 49 | `dQ`、`dK/dV` Kernel 与 Causal Pruning | 为什么两个所有权方向能避免 Atomic，怎样收缩循环又不漏梯度？ |
| 50 | 正确性矩阵、显存、Benchmark、AIOS 边界 | 怎样把“默认 PASSED”升级为可信、可复现且不过度宣传的工程证据？ |

专题学习链：

```text
普通 Attention / Online Softmax
→ Forward Tile 与片上状态
→ Grid / Pointer / Causal Stage
→ LSE 与 Backward 重算
→ 输出 Tile 所有权
→ 正确性、性能和产品边界
```

## 学习方法

```text
先看图和固定例子
→ 阅读 README 内关键代码与教学化伪代码
→ 手算 Shape/地址/字节/FLOPs
→ 运行 CPU 机制实验
→ 有匹配 CUDA/Triton 环境时再跑固定上游 GPU 路径
→ 完成练习题并对照答案
→ 最后回到完整 Top-3 p95 与质量门禁
```

## 一键验证

```bash
python resources/cuda-optimization/validate_cuda_course.py
```

验证器会检查 Lesson 29～50：

- 每课目录、README 与运行脚本存在；
- README 包含代码、常见错误理解、练习题/参考答案和一句话复述；
- 代码围栏与公式分隔符符合 GitHub Markdown；
- Python 实验能编译并在 CPU 环境运行；
- SVG（若存在）可解析。

## 事实边界

- 当前 AIOS 已实现：PyTorch CUDA、FlashInfer、Triton KV Scatter、Fused Projection/Norm/Activation、Paged KV。
- 当前 AIOS 未实现：CUDA Graph、权重量化、KV 量化、Speculative Decoding、Tensor Parallel。
- Lesson 45～50 固定分析外部教学仓库的 Dense FlashAttention Forward/Backward；没有把该代码复制为 AIOS Runtime Backend。
- 外部固定实现的安全学习合同较窄：连续且一致的 Q/K/V Layout、受控 Sequence Tile 与 Head Dimension；课程明确测试和解释这些边界。
- 六个新增 `run_lesson45.py`～`run_lesson50.py` 是 CPU 机制实验，不执行 Triton，不冒充 GPU 正确性或性能数据。
- 真实收益必须在固定 GPU、Driver、PyTorch、Triton、Shape、Dtype、Causal Flag、Warmup 与同步协议下重新测。
- AIOS 最终仍以完整输入法请求、CandidateGroup Top-3、p50/p95、显存与质量门禁判断优化，而不是只看单个 Dense Attention Kernel 微基准。
