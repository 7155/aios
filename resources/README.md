# AIOS 课程导航：从通用推理引擎到本地输入法 Runtime

> 当前教材扩展基线：`db343cbe07075c619d2519cb499c401f9edf895a`（`main`）
>
> 学习目标：先从 0～9 课建立推理引擎机制，再用 10～17 课理解输入法专项设计，最后用 18～28 课沿真实函数调用链完成核心源码带读。

## 第一篇：通用推理引擎基础

| 课次 | 教材 | 建立的能力 |
|---:|---|---|
| 0 | [Introduction](lesson-0-introduction/README_CN.md) | 知道 LLM 推理引擎解决什么问题 |
| 1 | [LLM Basics](lesson-1-llm-basics/README_CN.md) | 理解 Tokenizer、Embedding、Attention、RoPE、MLP 与 Logits |
| 2 | [Run Qwen3](lesson-2-run-qwen3/README.md) | 跑通完整 Decoder-only Forward |
| 3 | [Refactor to Package](lesson-3-refactor-to-package/README.md) | 把脚本重构为可扩展 Runtime |
| 4 | [KV Cache](lesson-4-kv-cache/README_CN.md) | 区分 Prefill 与 Decode |
| 5 | [Paged KV Cache](lesson-5-paged-kv-cache/README_CN.md) | 用 Page Table 管理非连续 KV |
| 6 | [Static Batching](lesson-6-static-batching/TECHNICAL_CN.md) | 多请求固定批处理 |
| 7 | [Continuous Batching](lesson-7-continuous-batching/README_CN.md) | 请求在 Token Step 动态进出 |
| 8 | [Flat Varlen Prefill](lesson-8-flat-varlen-prefill/README_CN.md) | 不 Padding 地并行变长 Prefix |
| 9 | [Fused Layers](lesson-9-fused-layers/README_CN.md) | 小算子与显存往返为什么成为瓶颈 |

## 第二篇：AIOS-IME 输入法专项

| 课次 | 教材 | 关键问题 |
|---:|---|---|
| 10 | [输入法不是缩短版聊天服务](lesson-10-ime-workload/README.md) | 为什么通用 Serving 指标不够？ |
| 11 | [MiniMind-IME 安全适配](lesson-11-minimind-ime-adapter/README.md) | 导出器必须验证哪些结构不变量？ |
| 12 | [CandidateGroup](lesson-12-candidate-group/README.md) | 八路怎样共享物理 Prefix KV？ |
| 13 | [Ragged Decode](lesson-13-ragged-candidate-decode/README.md) | Row 压缩怎样不改变随机结果？ |
| 14 | [token-LCP Prefix KV](lesson-14-prefix-kv-reuse/README.md) | 为什么字符 Prefix 不能直接复用？ |
| 15 | [Latest-wins](lesson-15-latest-wins/README.md) | 取消与 KV 生命周期怎样闭环？ |
| 16 | [候选治理](lesson-16-candidate-governance/README.md) | Raw Branch 怎样变成 Top-3？ |
| 17 | [性能与部署验收](lesson-17-evaluation-deployment/README.md) | 怎样建立可发布证据？ |

## 第三篇：核心源码带读

> [打开独立导航与验证说明](code-reading/README.md)

| 课次 | 教材 | 核心入口 |
|---:|---|---|
| 18 | [一次请求生命周期](lesson-18-request-lifecycle/README.md) | `LLM.generate()` |
| 19 | [运行时数据模型](lesson-19-runtime-data-model/README.md) | `core.py` |
| 20 | [Scheduler 状态机](lesson-20-scheduler-state-machine/README.md) | `scheduler.py` |
| 21 | [Prefill 代码带读](lesson-21-prefill-code-reading/README.md) | `prefill.py` + `_prepare_batch` |
| 22 | [Decode 代码带读](lesson-22-decode-code-reading/README.md) | `decode.py` + `_advance` |
| 23 | [Page/Slot 所有权](lesson-23-page-table-cache-manager/README.md) | `table.py` + `cache.py` |
| 24 | [Engine 与 Context](lesson-24-engine-context/README.md) | `engine.py` + `core.Context` |
| 25 | [Model Forward](lesson-25-model-forward/README.md) | `models/qwen3.py` + `norm.py` |
| 26 | [FlashInfer Backend](lesson-26-flashinfer-backend/README.md) | `attention/flashinfer.py` |
| 27 | [IME 主状态机](lesson-27-ime-engine-code-reading/README.md) | `ime.py` |
| 28 | [权重加载与 Packing](lesson-28-weight-loading/README.md) | `weight.py` + `BaseOP` |

## 公式与图示标准

新增源码带读篇统一使用 GitHub fenced math：

````markdown
```math
E = D - C
```
````

难懂状态机、Flat Batch、Page 所有权、Fused Residual、FlashInfer Metadata 和 IME 主流程均配有 SVG。图只负责展示关系，关键解释仍完整写在 README。

## 运行全部核心源码实验

```bash
for script in resources/lesson-{18..28}-*/run_lesson*.py; do
  python "$script"
done

python resources/code-reading/validate_code_reading.py
```

CPU 实验验证机制，不冒充 CUDA 性能；真实 GPU 行为仍由项目测试、Benchmark 与冻结评测负责。
