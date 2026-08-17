# Lesson 35：量化——BF16、FP8、INT8、INT4 与 KV Cache 到底怎样变小

> 源码基线：`c335497c6bf67a4dc8cb5ba748ace7b7c1cb77af`
>
> 当前 AIOS 以 BF16 推理，**没有实现权重量化、激活量化或 KV Cache 量化**。本课把量化数学、Kernel 边界、显存收益和精度风险完整讲清，并给出接入 AIOS 时要改哪些层。

## 1. 为什么量化可能有价值

当前在线参数：

```text
100,687,360 parameters
BF16 = 2 bytes/parameter
```

权重理论大小：

```math
100{,}687{,}360\times2
=
201{,}374{,}720\ \text{bytes}
\approx192.05\ \text{MiB}
```

若仅看权重：

| 格式 | 理论字节/参数 | 约权重大小 |
|---|---:|---:|
| BF16 | 2 | 192 MiB |
| FP8 / INT8 | 1 | 96 MiB + Scale |
| INT4 | 0.5 | 48 MiB + Scale/Metadata |

但真实延迟不会按文件大小同比下降。还取决于：

- GPU 是否有对应低精度 Tensor Core；
- Dequant 是否融合进 GEMM；
- 矩阵 Shape 是否足够大；
- Kernel Launch 与 Attention 是否才是瓶颈；
- CPU 候选治理占比；
- 精度回退是否导致更多 refill/更差候选。

## 2. 最基本的对称 INT8 量化

给浮点向量 `x`，选择最大绝对值：

```math
m = \max_i |x_i|
```

INT8 范围约 `[-127,127]`，Scale：

```math
s = \frac{m}{127}
```

量化：

```math
q_i = \operatorname{round}\left(\frac{x_i}{s}\right)
```

反量化：

```math
\hat{x}_i = s q_i
```

例子：

```text
x = [-1.0, -0.4, 0.2, 0.9]
m = 1.0
s = 1/127 ≈ 0.007874
q ≈ [-127, -51, 25, 114]
```

反量化存在舍入误差，但存储减半。

## 3. 非对称量化与 Zero Point

若数据范围不是围绕 0 对称，可用：

```math
q = \operatorname{round}\left(\frac{x}{s}\right)+z
```

其中 `z` 是 Zero Point。

神经网络 Weight 常用对称量化，因为权重通常正负都有、矩阵乘更容易融合；Activation 可能使用动态/非对称方案。

## 4. Per-Tensor、Per-Channel 与 Group-wise

### Per-Tensor

整张 Weight 一个 Scale：

```text
简单、Metadata 少
但大 Outlier 决定全矩阵精度
```

### Per-Output-Channel

Linear 每个输出行一个 Scale：

```text
W[out,in]
scale[out]
```

更能适应不同 Channel 范围，常用于 INT8 Weight。

### Group-wise INT4

每 32/64/128 个权重一组 Scale：

```text
更精细
但 Scale/Metadata 更多
Kernel 必须高效解包
```

INT4 的实际收益高度依赖是否有成熟 W4A16/W4A8 GEMM Kernel；若先把 INT4 解压回 BF16 再调用普通 GEMM，显存带宽可能省了，但临时成本和 Kernel 不一定划算。

## 5. Weight-only 与 Weight+Activation

### W8A16 / W4A16

```text
Weight 存 INT8/INT4
Activation 保持 BF16
GEMM Kernel 内部读取/反量化 Weight
```

优势：改动较小，Activation 精度保留；适合模型权重带宽是瓶颈。

### W8A8 / FP8

```text
Weight 与 Activation 都低精度
Accumulator 常用 FP16/FP32
```

可能充分利用低精度 Tensor Core，但 Calibration/Scale 和精度风险更高。

## 6. AIOS 哪些层受影响

当前：

```python
class Linear(BaseOP):
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
```

量化后不能只把 `weight.dtype` 换成 `int8`，因为 `F.linear(BF16,int8)` 不是期望的量化 GEMM。

需要新的算子合同：

```python
class QuantizedLinear(BaseOP):
    qweight: Tensor       # int8/int4 packed
    scales: Tensor        # per-channel/group
    zero_points: Tensor | None

    def forward(self, x):
        return quantized_gemm(x, qweight, scales, zero_points)
```

还需修改：

- Exporter：量化、保存 Scale/Format；
- Manifest：记录量化格式、Group Size、Calibration；
- Weight Loader：识别 packed layout；
- QKV/Gate-Up Packing：先量化还是先 Packing；
- LM Head/Tied Embedding：是否量化，如何共享；
- Tests：与 BF16 Reference 比较；
- Frozen Eval：候选/排序/近平局。

## 7. Packing 与量化的顺序

两种策略：

### 先拼 Q/K/V，再统一量化

```text
Wq/Wk/Wv FP
→ concat Wqkv
→ 按 Wqkv 输出 Channel 量化
```

布局直接符合 Runtime，但每段可能需要不同 Calibration/Scale 解释。

### 各自量化，再拼 Packed Buffer

```text
quantize Wq
quantize Wk
quantize Wv
→ 拼 qweight 和对应 scales
```

更容易保持原模块 Scale，但 Loader/Kernel 必须知道三段 Scale 布局。

选择取决于 Quantized GEMM Kernel 接受的 Layout，而不是纯数学偏好。

## 8. KV Cache 量化

当前每 Token KV：

```math
2(K/V)\times14\times4\times64\times2\text{ bytes}
=14\text{ KiB}
```

若 INT8 KV 理论减半：

```text
≈ 7 KiB/token + scales
```

256 Page：

```text
BF16 ≈ 3.5 MiB
INT8 ≈ 1.75 MiB + metadata
```

对于当前 0.1B、256 Page，这部分绝对值不大；量化 KV 的复杂度和 Attention Kernel 支持可能不值得。长上下文/更多 Page 时才更有吸引力。

KV 量化还直接影响 Attention Score，必须处理：

- K/V Scale 粒度；
- 每 Token/每 Head Scale；
- Decode 追加时 Scale 写入；
- FlashInfer 是否支持该 Layout；
- BF16 近平局排序稳定性。

## 9. Calibration 是什么

Post-Training Quantization 需要样本观察 Activation/Weight 分布，选择 Scale/Clip。

若 Calibration 只用：

```text
极短日常 Prefix
```

可能在长上下文、技术词、同拼音排序上出现 Outlier 未覆盖。

AIOS 至少应使用冻结多 Lane 的代表样本：

```text
日常开放 Prefix
上下文固定候选
同拼音候选
长 Prefix
特殊字符/高频短语
```

Calibration Set 不是 Test Set；不能反复调到 Test 最优。

## 10. 如何验收量化版本

硬门禁：

```text
模型/Tokenizer/Quant Format Manifest 正确
所有 Linear Shape 与 Scale Shape 正确
BF16 vs Quantized Logits 误差在预期范围
Top-3 满三条/互异不退化
冻结 Context/Pinyin 排序不明显下降
人工语义不过度退化
p50/p95 真正改善
Peak Memory 真正下降
```

特别要看：

> 若量化导致候选质量差、refill 次数上升，模型单步更快也可能让完整 Top-3 不快。

## 11. README 内置 INT8 教学代码

```python
def quantize_symmetric_int8(values):
    max_abs = max(abs(v) for v in values)
    scale = max_abs / 127 if max_abs else 1.0
    q = [max(-127, min(127, round(v / scale))) for v in values]
    return q, scale

values = [-1.0, -0.4, 0.2, 0.9]
q, scale = quantize_symmetric_int8(values)
reconstructed = [value * scale for value in q]
```

它展示量化误差，但不是高性能 GPU Quantized GEMM。

## 12. 常见错误理解

### 错误：把 Tensor cast 成 int8 就完成推理量化

错。需要 Scale、可能的 Zero Point、Packed Layout，以及能在 GEMM 中高效反量化/累加的专用 Kernel。

### 错误：INT4 一定比 INT8 快两倍

错。受硬件支持、解包、Scale 粒度、矩阵 Shape 和内存/计算瓶颈影响；有时只省存储，不提高延迟。

### 错误：权重量化通过，就不需重新测候选质量

错。量化误差贯穿 Logits；近平局排序和采样路径可能离散翻转，必须跑冻结 Lane 与人工评测。

## 13. 运行实验

```bash
python resources/lesson-35-quantization/run_lesson35.py
```

实验对一个小向量执行对称 INT8 量化、反量化并打印误差，同时估算 BF16/INT8/INT4 权重大小。

## 14. 检验问题与参考答案

### 问题 1：为什么 INT8 Weight 不能直接交给普通 `F.linear` 与 BF16 Activation？

**参考答案：** 普通 `F.linear` 期望兼容的浮点/库路径，不知道 INT8 Weight 的 Scale/Zero Point 和 Packed Layout。需要 Quantized GEMM Kernel 在读取 Weight 时应用 Scale、用合适 Accumulator 计算并输出目标 DType。

### 问题 2：Per-Channel 为什么比 Per-Tensor 更准确？

**参考答案：** 不同输出 Channel 的数值范围可能差异大。整张矩阵一个 Scale 会被最大 Outlier 主导，使小范围 Channel 的量化步长过粗；每 Channel 独立 Scale 能更贴合局部分布，代价是更多 Metadata 和 Kernel 读取。

### 问题 3：当前 AIOS 为什么未必优先做 KV INT8？

**参考答案：** 低显存 Profile 只有约 3.5 MiB BF16 KV，绝对节省较小；接入需要 Attention Kernel、Scale Layout、追加写入和稳定性验证。相比权重或 Launch 优化，工程收益可能不够，应先 Profile 长上下文场景。

### 问题 4：量化后单 Token 更快，为什么完整 Top-3 可能不快？

**参考答案：** 量化误差可能降低候选合法率/多样性，增加 Refill 或生成长度；Dequant/不匹配 Kernel 也可能有额外成本。最终需测完整 CandidateGroup 墙钟和质量门禁，而非单 GEMM。

## 15. 一句话复述

量化用 Scale/Zero Point 把浮点 Weight、Activation 或 KV 映射到更窄格式，真正高性能依赖专用 Packed GEMM/Attention Kernel。AIOS 当前仍是 BF16；接入量化需要新 Linear/Exporter/Loader/Manifest，并以完整 Top-3 延迟和冻结质量共同验收。

## 16. Quantized GEMM 运行时到底做什么

以 Weight-only INT8、Activation BF16 为例，概念 Kernel：

```python
# 不是真实高性能实现，只展示数据流。
for output_channel in channels:
    scale = scales[output_channel]
    accumulator = 0.0
    for input_channel in inputs:
        w = qweight[output_channel, input_channel] * scale
        accumulator += activation[input_channel] * w
    output[output_channel] = accumulator
```

高性能 Kernel 不会先物化整张 BF16 Weight，而是在 Tile 载入/矩阵乘过程中应用 Scale，并用 FP16/FP32 Accumulator。这决定了量化是否真的节省带宽。

若代码是：

```python
weight_bf16 = qweight.float() * scales
return F.linear(x, weight_bf16)
```

每次都展开完整权重，通常失去主要内存与延迟收益，只适合正确性原型。

## 17. Outlier 为什么麻烦

例如一组权重：

```text
[-0.2, 0.1, 0.15, 12.0]
```

Per-Tensor Scale 被 12.0 决定，小值只能落在很少几个量化格点，误差巨大。处理方式可能包括：

- Per-Channel/Group Scale；
- Clip Outlier；
- 保留少量 Outlier Channel 高精度；
- SmoothQuant 类方法把 Activation/Weight 尺度重新分配；

每种都会改变 Export/Kernel/Calibration 合同，不能只改一行 dtype。
