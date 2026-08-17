# Lesson 11：把 MiniMind-IME 安全接入 AIOS

> 源码基线：`bfc72896bbadab5c897672506d237c070900412e`
>
> 这课解决的不是“怎样新写一套 Transformer”，而是：**一个训练仓库产生的 checkpoint，凭什么可以交给另一个推理引擎加载？哪些结构相似可以复用，哪些契约必须独立验证？**

![MiniMind-IME 模型结构](../../docs/images/minimind-ime-model-architecture.svg)

## 1. 为什么模型适配器只有几行代码，却不代表工作很少

当前适配类：

```python
class MiniMindIMEForCausalLM(Qwen3ForCausalLM):
    """Inference-only MiniMind-IME adapter."""
```

它复用 Qwen3 Backend，是因为最终 MiniMind-IME v3 在线主干使用相同的 dense decoder 算子族：

```text
RMSNorm
Q/K/V Linear
QK-Norm
RoPE
GQA Attention
SwiGLU
Residual
Tied LM Head
```

但“算子名字相同”不意味着任意 checkpoint 都能安全复用。

推理引擎真正依赖的是一组严格不变量：

```text
层数连续
hidden_size = num_attention_heads × head_dim
K projection 行数 = num_kv_heads × head_dim
激活函数是 SiLU/SwiGLU
RMSNorm/QK-Norm 权重存在
Embedding 与 LM Head 的 tied 状态真实一致
没有未支持的 MoE/YaRN
训练期 MTP 已剥离
Tokenizer 与词表完全对应
```

所以适配类很薄，**导出器和验证器才是模型边界的主要工作**。

## 2. 从真实配置还原在线结构

部署模型当前值：

| 参数 | 值 |
|---|---:|
| Decoder layers | 14 |
| Hidden size | 768 |
| Intermediate size | 2,048 |
| Q heads | 12 |
| KV heads | 4 |
| Head dim | 64 |
| Vocabulary | 16,384 |
| Context | 512 |
| Precision | BF16 |
| 在线参数 | 100,687,360 |

首先检查：

\[
12 \times 64 = 768
\]

K/V Projection 输出宽度：

\[
4 \times 64 = 256
\]

因此典型权重形状应是：

```text
q_proj: [768, 768]
k_proj: [256, 768]
v_proj: [256, 768]
o_proj: [768, 768]
```

若配置写 `num_kv_heads=8`，但 `k_proj.shape[0]=256`：

```text
inferred_kv_heads = 256 / 64 = 4
```

导出器必须拒绝，而不是相信 JSON 后继续加载。

## 3. 为什么不能“缺 LM Head 就自动绑到 Embedding”

有两种合法模型：

### Tied

```text
lm_head.weight 与 embed_tokens.weight 是同一语义矩阵
```

### Untied

```text
输入 Embedding 与输出 LM Head 分别训练
```

危险做法：

```text
配置说 untied
checkpoint 缺 lm_head
→ 推理引擎为了加载成功，偷偷拿 embedding 当 lm_head
```

这样 Shape 完全合法，但语义错误，不一定立刻报错。

当前导出器的规则：

- 配置说 tied 且 checkpoint 同时有 LM Head：两者必须逐值相等；
- 配置说 untied：必须真的存在 `lm_head.weight`；
- 只有验证 tied 后，部署包才可以省掉重复 LM Head。

这体现一个重要工程原则：

> **宁可导出失败，也不要通过“猜测”生成一个看起来能运行的错误模型。**

## 4. 为什么训练期 MTP 必须剥离

MiniMind-IME 训练可能包含 Next-2 MTP 辅助模块。它的目标是训练时增加更远一步监督，并不是当前 AIOS Decode 所需的在线结构。

导出器会检测：

```text
mtp_module.*
model.mtp_head.*
任何 path part 为 mtp 或 mtp_*
```

然后：

```text
剥离训练辅助 Tensor
config.mtp_enabled = false
```

这不是简单省显存，还避免在线运行时误把训练专用状态当成主干依赖。

## 5. 为什么导出为 Safetensors + Manifest

部署目录：

```text
model.safetensors
config.json
tokenizer.json
tokenizer_config.json
aios_manifest.json
```

Manifest 应回答：

- 权重来自哪个 checkpoint；
- checkpoint SHA256；
- 导出后模型 SHA256；
- dtype；
- 参数量；
- 哪些 MTP Tensor 被剥离；
- Tokenizer 文件 SHA；
- tied embedding 是否验证；
- model type 与结构值。

Hash 证明的是**字节完整性和身份**，不是模型质量或许可证。

## 6. 为什么 AIOS 的 KV Page 大小可以从配置直接算

当前 Page Size 为 1 token。一个 token 的 K/V 字节：

\[
2(K/V)
\times14\text{ layers}
\times4\text{ KV heads}
\times64\text{ head dim}
\times2\text{ bytes(BF16)}
\]

\[
= 14,336\text{ bytes} = 14\text{ KiB}
\]

因此低显存 profile：

```text
256 token pages × 14 KiB ≈ 3.5 MiB KV storage
```

注意这只是 KV 本体，不包含 Page Table、FlashInfer Workspace、临时激活和 allocator overhead。

## 7. 加载路径的真实执行顺序

```text
LLM(model_path)
→ AutoConfig / config.json
→ ModelConfig.from_hf / from_json
→ create_model(model_type)
→ meta device 建结构，不分配完整权重
→ load_weights 到 CUDA/BF16
→ 设置 RoPE device
→ 加载 Tokenizer
→ 根据配置和显存计算 num_pages
→ 创建 MHAKVCache
→ 创建 CacheManager / Context
```

`meta` device 的意义：先构造参数 Shape 和模块关系，不先在 CPU/GPU 分配一份完整随机权重，再覆盖它。

## 8. 当前复用 Qwen3 Backend 的边界

可以复用：

- Dense GQA Decoder；
- QK Norm；
- RoPE；
- RMSNorm；
- SwiGLU；
- 已有融合算子与 FlashInfer 接口。

不能静默复用：

- MoE Expert Routing；
- YaRN/其他 RoPE scaling；
- 不同激活函数；
- 缺失 QK Norm 的结构；
- 不同权重命名/Shape；
- 训练期 MTP 作为在线分支。

## 9. 测试如何把“模型合同”变成证据

`tests/test_ime_export.py` 覆盖：

```text
假 tied 声明必须失败
untied 但缺 LM Head 必须失败
MoE 必须失败
非 SiLU 必须失败
导出 dtype 必须正确
只有验证后的 tied LM Head 才能省略
MTP 命名空间必须被识别
```

这些测试比“成功加载一次模型”更重要，因为它们覆盖错误模型也可能 Shape 合法的情况。

## 10. 常见错误解释

### 错误：MiniMind 与 Qwen 都是 Decoder，所以直接换权重就行

错。需要验证每个算子的布局、Head 数、Norm、激活、Tied 状态和训练辅助边界。

### 错误：模型能输出中文，就说明导出正确

错。错误绑头、漏 Tensor、Tokenizer 漂移都可能仍输出“像中文”的文本。

### 错误：Manifest Hash 证明模型效果

错。Hash 只证明加载的是记录中的文件。

## 11. 运行实验

```bash
python resources/lesson-11-minimind-ime-adapter/run_lesson11.py
```

它会用真实在线配置计算 Q/K/V Shape、每 Token KV 字节和 256 页预算，并故意构造一个 K Head 配置冲突。

正式测试：

```bash
pytest -q tests/test_ime_export.py
```

## 12. 面试追问

1. 为什么模型适配类可以很薄，但 Exporter 不能很薄？
2. `hidden_size` 与 `head_dim` 对不上时，为什么不能自动向下取整？
3. Tied Embedding 节省了什么，又需要验证什么？
4. 为什么 `state_dict` Shape 全对，Tokenizer 错了仍可能无法被单元测试发现？
5. `meta` device 对模型加载峰值内存有什么意义？

## 13. 一句话复述

MiniMind-IME 接入 AIOS 的核心不是重写 Decoder，而是把“算子兼容”转化成可验证的部署合同：结构、权重 Shape、Tied 状态、Tokenizer、MTP 剥离和 Manifest 身份都必须明确，任何无法证明的猜测都应让导出失败。
