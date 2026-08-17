# Lesson 28：权重加载、Fused Packing 与 `BaseOP`

> 源码基线：`db343cbe07075c619d2519cb499c401f9edf895a`
>
> 目标：理解训练 checkpoint 里分离的 Q/K/V、Gate/Up 权重怎样被推理引擎打包成融合矩阵，以及 `BaseOP` 为什么靠“公开 Tensor/子 OP”递归建立自己的 State Dict。

## 1. 为什么 AIOS 不直接继承 `torch.nn.Module`

`BaseOP` 提供最小能力：

```text
forward
state_dict
load_state_dict
```

规则：

- 不以下划线开头的 `torch.Tensor` 是权重；
- 不以下划线开头的 `BaseOP` 递归收集；
- `_backend/_scale/_layer_idx` 等运行时字段不进入 State Dict。

这是一个轻量推理算子系统，但依赖严格命名约定。

## 2. `state_dict()` 怎样递归命名

例如：

```text
model.layers.0.self_attn.qkv_proj.weight
```

来自：

```text
Qwen3ForCausalLM.model
→ Qwen3Model.layers (OPList index 0)
→ DecoderLayer.self_attn
→ Attention.qkv_proj
→ weight
```

`OPList` 必须覆写递归逻辑，为列表元素加入数字路径。

## 3. 为什么权重加载先遍历目标模型

```python
for target_name in model.state_dict():
```

目标模型决定需要哪些 Tensor；Loader 再从 Safetensors Index 查找来源。

好处：

- 缺少目标 Tensor 立即报错；
- 不把 checkpoint 中无关训练状态加载进来；
- Fused Module 的目标名字由当前 Runtime 结构定义。

## 4. QKV Packing 怎样计算

HF 来源：

```text
q_proj.weight [768,768]
k_proj.weight [256,768]
v_proj.weight [256,768]
```

沿输出维 `dim=0` 拼接：

```math
W_{qkv} = \operatorname{Concat}_0(W_q,W_k,W_v)
```

Shape：

```math
(768 + 256 + 256, 768) = (1280,768)
```

Forward 一次矩阵乘：

```text
[N,768] × [1280,768]^T → [N,1280]
```

再按 `[768,256,256]` Split。

数学上等价于三次独立 Linear，但减少 Kernel Launch 与输入读取。

## 5. Gate/Up Packing

```text
gate_proj [2048,768]
up_proj   [2048,768]
```

打包：

```math
W_{gate\_up} = \operatorname{Concat}_0(W_g,W_u)
```

得到：

```text
[4096,768]
```

一次 GEMM 后 `silu_and_mul` 融合激活与逐元素乘法。

## 6. 为什么 Packing 放在 Loader 边界

如果 Qwen3Attention 自己知道 HF 分离命名：

- 模型层耦合 Checkpoint 格式；
- 换导出格式需要改算子；
- 运行时每次 Forward 可能处理拼接；
- 单元测试边界模糊。

当前：

```text
Checkpoint Format Knowledge → Loader
Fused Tensor Layout          → Runtime Model
```

只在加载时转换一次。

## 7. `load_state_dict` 怎样防错

对每个目标 Tensor：

```text
必须存在
必须是 Tensor
Shape 必须完全相同
```

加载结束仍有多余 Key：报 `Unexpected keys`。

但 `BaseOP.load_state_dict` 直接 `pop` 输入 Dict，所以调用者应传可消费副本；当前 Loader 构造新的 `fused_state_dict`，符合这个契约。

## 8. Tied LM Head 为什么 State Dict 为空

若 Tied：

```text
LMHead._tied_embedding → Embedding.weight
```

`LMHead.state_dict()` 返回空，避免同一 Tensor 重复要求。

加载时若 checkpoint 仍含 `lm_head.weight`，LMHead 会 Pop；但正式导出器已验证它与 Embedding 一致，不能把这个 Pop 当成“无条件相信 tied”。

## 9. 从 CPU 到 GPU/DType

每个来源 Tensor：

```python
tensor.to(device=device, dtype=dtype)
```

然后赋给 Meta Device 上构建的目标字段。这样避免先分配随机真实权重再覆盖。

边界：Loader 当前按 Tensor 逐个读 Safetensors 并搬运，未实现并行 IO、异步流或量化转换。

## 10. 运行实验

```bash
python resources/lesson-28-weight-loading/run_lesson28.py
```

实验用小矩阵验证“三次 Linear”与“拼接一次 Linear 后 Split”数值相同，并打印 Shape。

## 11. 常见错误解释

### 错误：Fused QKV 改变了 Attention 数学

错。它只把三个共享输入的 Linear 沿输出维打包。

### 错误：所有以下划线开头的字段都只是缓存

它们是不进入 State Dict 的运行时字段，可能是 Backend、Scale、配置或函数引用，不一定是缓存。

### 错误：Shape 对上就证明权重语义正确

不够。顺序若从 `[Q,K,V]` 错成 `[K,Q,V]`，Shape 仍正确但结果错误；Packing Mapping 必须有测试/对照。

## 12. 检验问题与参考答案

### 问题 1：为什么 QKV 沿 `dim=0` 拼接，而不是 `dim=1`？

**参考答案：** PyTorch Linear 权重 Shape 是 `[out_features, in_features]`。Q/K/V 共享同一个输入维 `hidden_size`，区别在各自输出通道数，因此要沿 `out_features` 的第 0 维拼接。若沿 `dim=1`，等于把输入维拼长，要求输入也跟着扩展，已经不是三次共享输入 Linear 的等价变换。

### 问题 2：Fused Packing 节省的是参数量还是运行开销？

**参考答案：** 参数量基本不变，只是把同样的 Q/K/V 或 Gate/Up 权重重新排成一张大矩阵。收益主要来自减少 GEMM/Kernel Launch 次数、重复读取同一输入 Hidden 的带宽，以及更适合融合算子的内存布局。

### 问题 3：`BaseOP` 命名约定相对 `nn.Module` 有什么脆弱点？

**参考答案：** 它依赖“公开 Tensor=权重、下划线字段=非权重”的约定，没有 PyTorch Parameter/Module 的注册机制保护。开发者如果给一个真正需要保存的权重加了下划线，或把运行时 Tensor 暴露为公开字段，都可能让 State Dict 漏项/多项，因此更依赖测试和编码纪律。

### 问题 4：Sharded Safetensors 怎样避免重复 Key？

**参考答案：** Loader 先遍历所有 shard，建立 `tensor_name → file_path` 索引；同一个 Tensor 名如果在两个 shard 同时出现就立即报 Duplicate Key，而不是采用“后读覆盖前读”。这样能避免 checkpoint 打包错误被静默吞掉。

### 问题 5：量化权重接入时，Packing 应发生在量化前还是后？取决于什么？

**参考答案：** 取决于量化格式和 Kernel 期望布局。如果量化是逐输出通道、各 Q/K/V 使用兼容 Scale，可能先量化后按 Kernel 格式打包；若量化算法需要整张 Fused 矩阵共同计算 Group/Scale，则应先 Packing 再量化。不能只拼 Int4 字节而忽略对应 Scale/Zero-point 的布局。

## 13. 一句话复述

AIOS 用目标模型的 State Dict 定义所需权重，在 Loader 边界把 HF 分离 Q/K/V 与 Gate/Up 沿输出维打包，搬到目标 Device/DType 后由 `BaseOP` 严格按路径和 Shape 赋值；融合减少运行时 GEMM/内存访问，不改变模型数学。
