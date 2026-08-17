# Lesson 31：Operator Fusion 与显存流量——为什么少一个 Kernel 可能比少一点 FLOPs 更重要

> 源码基线：`c335497c6bf67a4dc8cb5ba748ace7b7c1cb77af`
>
> 这一课直接讲 AIOS 目前已经实现的融合：QKV Projection、Gate/Up Projection、SwiGLU、Residual+RMSNorm。目标是看懂“融合”到底融合了什么，以及为什么它通常减少的是 Kernel Launch 和 Global Memory 往返，而不是模型参数或数学量。

## 1. 先区分两种融合

### 计算图融合

把多个 Elementwise/Reduction 操作放进一个 Kernel：

```text
add → square → mean → rsqrt → multiply weight
```

变成一个 Fused RMSNorm Kernel。

### 权重 Packing / GEMM 合并

三个共享输入的 Linear：

```text
Q = X Wqᵀ
K = X Wkᵀ
V = X Wvᵀ
```

把权重沿输出维拼起来：

```math
W_{qkv}=\operatorname{Concat}_0(W_q,W_k,W_v)
```

一次 GEMM：

```math
[Q|K|V]=XW_{qkv}^{T}
```

数学结果相同，但从 3 次 GEMM Launch 变成 1 次。

## 2. AIOS QKV 融合的完整代码

定义：

```python
class LinearQKVMerged(Linear):
    def __init__(self, hidden_size, q_size, kv_size, has_bias=False):
        super().__init__(hidden_size, q_size + 2 * kv_size, has_bias)
        self.q_size = q_size
        self.kv_size = kv_size
```

Forward：

```python
qkv = self.qkv_proj.forward(hidden_states)
q, k, v = qkv.split(
    [self.q_size, self.kv_size, self.kv_size],
    dim=-1,
)
```

真实 0.1B 配置：

```text
hidden_size = 768
q_size      = 12×64 = 768
kv_size     = 4×64  = 256
```

Packed Weight：

```text
Wq [768,768]
Wk [256,768]
Wv [256,768]
→ Wqkv [1280,768]
```

输入 `X.shape=[N,768]`：

```text
一次 F.linear
→ qkv [N,1280]
→ split 为 [N,768]、[N,256]、[N,256]
```

### 为什么沿 `dim=0` 拼权重

PyTorch Linear 权重 Shape：

```text
[out_features, in_features]
```

Q/K/V 共享 `in_features=768`，不同的是输出行数，所以沿输出维 `dim=0` 拼接。

若沿 `dim=1`：

```text
[768,768] 与 [256,768]
```

第一维不同，根本不能直接拼；即使构造别的布局，也会改变输入语义。

## 3. QKV 合并节省什么

未合并：

```text
Launch GEMM Q
Launch GEMM K
Launch GEMM V
```

合并：

```text
Launch one GEMM QKV
```

收益：

- CPU/Driver Launch 从 3 次降到 1 次；
- 输入 `hidden_states` 理论上只需按一个 GEMM 流程读取；
- cuBLAS/Tensor Core 得到更大的输出矩阵，可能更易提高利用率；
- 后续 Split 只是 View/元数据操作时，成本很小。

不节省：

- 参数量：三张权重只是拼接，元素总数不变；
- 主要乘加量：仍需计算所有 Q/K/V 输出；
- 输出内存总字节：Q/K/V 结果仍存在。

## 4. Gate/Up 融合

SwiGLU 数学：

```math
y = W_{down}\left(\operatorname{SiLU}(W_gx)\odot W_ux\right)
```

AIOS：

```python
self.gate_up_proj = LinearColParallelMerged(
    hidden_size,
    [intermediate_size, intermediate_size],
)
```

真实 Shape：

```text
Wgate [2048,768]
Wup   [2048,768]
→ Wgate_up [4096,768]
```

Forward：

```python
gate_up = self.gate_up_proj.forward(x)
return self.down_proj.forward(self._act_fn(gate_up))
```

其中 `_act_fn` 是：

```python
def silu_and_mul(x, out=None):
    from flashinfer import silu_and_mul
    return silu_and_mul(x, out=out)
```

FlashInfer Kernel 内部逻辑等价于：

```python
gate, up = x.chunk(2, dim=-1)
out = torch.nn.functional.silu(gate) * up
```

但避免中间：

```text
silu_output 写 Global Memory
→ 下一 Kernel 再读回来与 up 相乘
```

## 5. 手算显存流量：SwiGLU 为什么适合融合

假设一共有 `N=128` 个 Token，Intermediate=2048，BF16 每元素 2 字节。

未融合粗略中间流量：

```text
读取 gate：128×2048×2
写 silu(gate)：128×2048×2
读取 silu(gate)：128×2048×2
读取 up：128×2048×2
写 product：128×2048×2
```

总约：

```math
5\times128\times2048\times2
=2{,}621{,}440\ \text{bytes}
\approx2.5\ \text{MiB}
```

融合后可近似：

```text
读取 gate
读取 up
写 product
```

```math
3\times128\times2048\times2
\approx1.5\ \text{MiB}
```

这是概念估算，真实 Cache/Kernel 实现会影响流量，但它说明融合的方向：少写一次大中间 Tensor、少读一次、少一个 Launch。

## 6. Residual + RMSNorm 融合

普通 Pre-Norm 逻辑：

```python
residual = residual + x
normalized = rmsnorm(residual)
```

若拆成两个 Kernel：

```text
Kernel A：读 residual + x，写 new_residual
Kernel B：再读 new_residual，算 RMS，写 normalized
```

AIOS：

```python
class RMSNormFused(BaseOP):
    def forward(self, x, residual=None):
        if residual is None:
            return self.rmsnorm(x, self.weight, self.eps), x
        self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual
```

关键语义：

```text
residual 输入/输出：长期未归一化主线
x 输入：本子层刚产生的增量
x 输出：更新主线后的 normalized branch input
```

也就是 Kernel 原地完成：

```math
r' = r + x
```

```math
x' = \operatorname{RMSNorm}(r')
```

调用方继续：

```python
hidden_states, residual = self.post_attention_layernorm.forward(
    hidden_states,
    residual,
)
hidden_states = self.mlp.forward(hidden_states)
```

这里 `hidden_states` 从“Attention 增量”变成“已加入主线后的归一化 MLP 输入”；`residual` 保留未归一化主线。

## 7. 为什么原地写安全

原地操作要求旧值之后不再被其他分支使用。

在当前 Decoder Layer：

```text
Attention 输出 x
+ residual 主线
→ 下一步只需要：
   更新后的 residual
   以及它的 normalized 版本
```

所以 FlashInfer 可以原地改 `x/residual`。

若 Autograd 训练需要保存旧 Tensor 做 backward，原地写会更复杂；AIOS 是 Inference-only，不需要反向图，因此空间更大。

## 8. Fusion 的代价

- Kernel 更专用，Shape/DType/Layout 不匹配时不能复用；
- 调试不如分离算子直观；
- 可能增加 Register 使用，降低 Occupancy；
- 小 Tensor 时收益可能被 Wrapper/Launch 抵消；
- 数值归约顺序改变，BF16 近平局排序可能发生细小变化；
- 依赖 FlashInfer 版本与支持矩阵。

所以不是“能融合就全融合”，而是 Profile 后选择高流量、连续相邻、无中间复用的操作。

## 9. 为什么 Linear 本身仍调用 `F.linear`

```python
class Linear(BaseOP):
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
```

AIOS 没有自己写 GEMM Triton Kernel，因为矩阵乘优化非常复杂，cuBLAS/Tensor Core 已高度优化。当前项目的自定义价值在：

- 合并布局；
- 权重加载时 Packing；
- 调用高性能库；
- 专用 Elementwise/Attention 路径。

“自己写 Kernel”不等于一定更快。

## 10. 常见错误理解

### 错误：QKV Fusion 把参数量减少了三分之二

错。只是把同样的参数拼到一张矩阵，参数总数不变。

### 错误：Fusion 的收益主要是少算数学

多数上述融合的 FLOPs 基本不变，主要减少 Launch、中间 Tensor 和 Global Memory 往返。

### 错误：Fused Kernel 一定比多个 Kernel 快

不一定。Shape 太小、Register Pressure 太高、已有库优化更好时可能不划算，必须 Benchmark。

## 11. 运行实验

```bash
python resources/lesson-31-fusion-memory-traffic/run_lesson31.py
```

实验验证：分离 Q/K/V Linear 与 Packed Linear 后 Split 数值相同，并计算 SwiGLU 中间流量概念值。

## 12. 检验问题与参考答案

### 问题 1：QKV Packing 为什么不减少参数量？

**参考答案：** `Wq/Wk/Wv` 的所有元素都仍然保存在 `Wqkv` 中，只是沿输出维连续排列。参数总数与乘加量等于三者之和；收益来自一次更大的 GEMM、减少 Launch 和输入读取流程。

### 问题 2：`silu_and_mul` 为什么适合做成单 Kernel？

**参考答案：** SiLU 输出只用于立即与 Up Branch 逐元素相乘，没有其他消费者。拆分会把 SiLU 中间结果写入 Global Memory，再由乘法 Kernel 读回；融合可在寄存器/片上状态中完成后只写最终 Product。

### 问题 3：Residual+RMSNorm 原地融合为什么在推理中更容易？

**参考答案：** 推理没有 Autograd Backward 需要保存旧激活。当前执行顺序确认旧 `x/residual` 不再有其他消费者，因此可安全复用存储。训练中原地改写可能破坏 backward 所需版本，需要额外保存或定制 backward。

### 问题 4：如何判断一个融合是 Memory-bound 还是 Compute-bound？

**参考答案：** 比较每读取/写入字节对应的运算量（Arithmetic Intensity），并用 Profiler/Nsight 观察 DRAM Throughput、Tensor Core 利用率和 Kernel 时间。Elementwise Norm/Activation 常更偏带宽；大 GEMM 更偏计算，但小 Shape 也可能受 Launch 限制。

## 13. 一句话复述

AIOS 通过 QKV/Gate-Up Packing 把共享输入的多个 Linear 合并为一次 GEMM，通过 FlashInfer 融合 SwiGLU 与 Residual+RMSNorm，主要减少 CPU Launch 和 Global Memory 中间读写；数学和参数量大多不变，收益必须由具体 Shape 的 Profile 证明。

## 14. 用 Roofline 直觉判断“该优化计算还是内存”

Arithmetic Intensity：

```math
I=\frac{\text{FLOPs}}{\text{bytes moved}}
```

- `I` 很低：每搬很多字节只做少量运算，通常 Memory-bound；
- `I` 很高：大量乘加复用数据，可能 Compute-bound。

`store_cache` 近似没有 FLOPs，所以极低；SwiGLU Elementwise 也偏低；大 Linear GEMM 中同一 Weight/Input Tile 被多次复用，Intensity 高得多。

Fusion 的价值正是提高有效 Intensity：

```text
原来：中间结果写 DRAM，再读回来
融合：中间值留在 Register/片上，只写最终结果
```

分母中的 bytes moved 下降，而数学 FLOPs 基本不变。

## 15. 如何验证融合没有改数学

每一种 Fusion 都应有 Reference：

```python
# Reference
q_ref = F.linear(x, wq)
k_ref = F.linear(x, wk)
v_ref = F.linear(x, wv)

# Fused layout
qkv = F.linear(x, torch.cat([wq, wk, wv], dim=0))
q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

assert torch.allclose(q, q_ref, atol=..., rtol=...)
```

Residual+Norm 也应比较：

```python
r_ref = residual + x
x_ref = rmsnorm(r_ref)

x_fused, r_fused = fused_add_rmsnorm(x, residual)
```

低精度允许合理误差，但离散排序任务还必须跑冻结 Lane，因为“allclose 通过”不保证近平局名次不变。
