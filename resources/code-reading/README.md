# AIOS 核心源码带读篇

> 固定源码：`db343cbe07075c619d2519cb499c401f9edf895a`（`main`）
>
> 这条路线不按文件夹逐个翻译，而是追踪一次请求的真实运行顺序：从 `LLM.generate()` 进入 Scheduler，经 Prefill/Decode、Page Table、Engine、Model、FlashInfer，再回到 Token 与结果。

## 阅读顺序

| 课次 | 教材 | 你最终能回答的问题 |
|---:|---|---|
| 18 | [一次请求怎样穿过整个 AIOS](../lesson-18-request-lifecycle/README.md) | `LLM.generate()` 里面究竟创建了哪些对象，谁驱动循环？ |
| 19 | [Req、Batch 与 Context](../lesson-19-runtime-data-model/README.md) | `cached_len/device_len/extend_len` 分别代表什么？ |
| 20 | [Scheduler 状态机](../lesson-20-scheduler-state-machine/README.md) | Pending、Running、Finished 怎样迁移？为什么 Prefill-first？ |
| 21 | [PrefillManager 与 Flat Batch](../lesson-21-prefill-code-reading/README.md) | 变长 Prompt 怎样不 Padding 地拼成一次 Forward？ |
| 22 | [DecodeManager 与逐 Token 推进](../lesson-22-decode-code-reading/README.md) | 为什么每个运行请求每轮正好贡献一个 Token？ |
| 23 | [TableManager、CacheManager 与 Page 所有权](../lesson-23-page-table-cache-manager/README.md) | 逻辑请求槽与物理 KV Page 为什么必须分开？ |
| 24 | [Engine、Context 与 LM Head](../lesson-24-engine-context/README.md) | 为什么 `model.forward()` 没参数，仍知道当前 Batch？ |
| 25 | [Qwen3/MiniMind Decoder Forward](../lesson-25-model-forward/README.md) | Fused RMSNorm/Residual 代码怎样等价于 Pre-Norm Block？ |
| 26 | [FlashInfer Metadata 与 Paged Attention](../lesson-26-flashinfer-backend/README.md) | `cu_seqlens`、`indices`、Prefill/Decode Wrapper 各负责什么？ |
| 27 | [ImeCompletionEngine 主状态机](../lesson-27-ime-engine-code-reading/README.md) | 一次按键怎样完成 Prefix 复用、8 路生成、补采样和取消？ |
| 28 | [权重加载、Fused Packing 与 BaseOP](../lesson-28-weight-loading/README.md) | HF 分离权重怎样变成 QKV/Gate-Up 融合权重？ |

## 每课阅读方法

```text
先看调用图
→ 用 Tiny 数值手算状态
→ 只读本节 5～15 行关键源码
→ 写出进入前/执行后状态
→ 运行 CPU 小实验
→ 回到真实 CUDA 路径
→ 回答面试追问
```

## 公式渲染约定

所有块级公式统一使用 GitHub 支持的 fenced math：

````markdown
```math
R = M - D
```
````

不会再使用旧式方括号数学定界符；统一使用 fenced math。变量会在公式前后用中文定义，不让公式单独悬空。

## 一键验证

```bash
python resources/code-reading/validate_code_reading.py
```

它检查：

- 11 节 README 与实验是否存在；
- 数学代码块是否成对；
- 不允许使用旧式方括号数学定界符；
- SVG 是否能解析；
- Python 示例是否能编译并运行。
