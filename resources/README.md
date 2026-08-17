# AIOS 课程导航：从通用推理引擎到本地输入法 Runtime

> 课程源码基线：`bfc72896bbadab5c897672506d237c070900412e`（`feat/aios-ime`）
>
> 学习目标：完成课程后，能够沿着一次中文按键请求，解释它如何从 Tokenizer 进入 Prefix Prefill、如何分叉为 CandidateGroup、如何共享 Paged KV、如何取消旧 generation、如何过滤并选出 Top-3，以及为什么最终性能必须按完整候选栏墙钟而不是单 token 吞吐验收。

## 课程主线

### 通用推理引擎基础（现有课程）

| 课次 | 教材 | 建立的能力 |
|---:|---|---|
| 0 | [Introduction](lesson-0-introduction/README_CN.md) | 知道 LLM 推理引擎解决什么问题 |
| 1 | [LLM Basics](lesson-1-llm-basics/README_CN.md) | 理解 Tokenizer、Embedding、Attention、RoPE、MLP 与 Logits |
| 2 | [Run Qwen3](lesson-2-run-qwen3/README.md) | 跑通一个完整 Decoder-only Forward |
| 3 | [Refactor to Package](lesson-3-refactor-to-package/README.md) | 把脚本重构为可扩展运行时 |
| 4 | [KV Cache](lesson-4-kv-cache/README_CN.md) | 区分 Prefill 与 Decode，理解为什么缓存 K/V |
| 5 | [Paged KV Cache](lesson-5-paged-kv-cache/README_CN.md) | 用页表管理非连续 KV 内存 |
| 6 | [Static Batching](lesson-6-static-batching/TECHNICAL_CN.md) | 多请求如何组成固定 Batch |
| 7 | [Continuous Batching](lesson-7-continuous-batching/README_CN.md) | 请求如何在 token step 边界动态进出 |
| 8 | [Flat Varlen Prefill](lesson-8-flat-varlen-prefill/README_CN.md) | 不 Padding 地并行不同长度 Prefix |
| 9 | [Fused Layers](lesson-9-fused-layers/README_CN.md) | 为什么小算子和显存往返成为瓶颈 |

### 输入法专项篇（新增课程）

| 课次 | 教材 | 关键问题 |
|---:|---|---|
| 10 | [输入法不是缩短版聊天服务](lesson-10-ime-workload/README.md) | 为什么通用 serving 指标不能直接指导本地 IME？ |
| 11 | [把 MiniMind-IME 安全接入 AIOS](lesson-11-minimind-ime-adapter/README.md) | 为什么只有算子相同还不够，导出器必须验证哪些不变量？ |
| 12 | [CandidateGroup：一次 Prefix，八条分支](lesson-12-candidate-group/README.md) | 如何让八条候选共享物理 Prefix KV，而不是复制八份？ |
| 13 | [Ragged Decode、独立随机流与按需补采样](lesson-13-ragged-candidate-decode/README.md) | 分支提前结束后为什么要压缩 active rows，怎样避免随机流漂移？ |
| 14 | [跨按键 token-LCP Prefix KV 复用](lesson-14-prefix-kv-reuse/README.md) | 为什么字符前缀相同不等于 Token Cache 可以安全复用？ |
| 15 | [Latest-wins：取消、锁与 KV 生命周期](lesson-15-latest-wins/README.md) | 用户继续输入时，旧请求怎样停止且不泄漏 Page？ |
| 16 | [候选治理与稳定重排序](lesson-16-candidate-governance/README.md) | 原始采样如何变成可显示、互异、可解释的 Top-3？ |
| 17 | [完整 Top-3 的性能、冻结评测与部署](lesson-17-evaluation-deployment/README.md) | 为什么微基准很快仍不代表候选栏快，怎样建立发布证据？ |

## 固定学习例子

专项篇尽量复用同一个请求：

```text
用户前缀：没关系，你先忙你的，
目标输出：三条可以直接进入候选栏的短后缀
```

它会依次经历：

```text
中文前缀
→ [BOS] + raw token ids
→ token-LCP 与旧 Prefix Cache 对齐
→ 一次 Prefix Prefill
→ 8 行 CandidateGroup 共享 Prefix Page
→ Ragged batched Decode
→ EOS/标点分支退出 active rows
→ 过滤、归一化、去重、原始 logprob、MMR
→ 不足三条时补 4 路
→ 返回 Top-3
```

## 学习方式

每课都按以下顺序：

```text
为什么需要
→ 没有它会具体坏在哪里
→ 手算/状态机小例子
→ 当前仓库的真实值与真实函数
→ 可运行 CPU 实验
→ GPU/正式路径验证命令
→ 常见错误解释
→ 面试追问
```

课程不会把 Tiny CPU 实验冒充 CUDA 端到端性能，也不会把满三条率、非空率或 LCP 写成语义准确率。

## 专项篇本地机制实验

新课中的 `run_lesson10.py`～`run_lesson17.py` 均为不依赖模型权重的 CPU 小实验：

```bash
for script in resources/lesson-1*/run_lesson*.py; do
  python "$script"
done
```

它们用于验证 CandidateGroup Page 引用、Ragged Row 压缩、Stateless Random Stream、token-LCP、latest-wins、MMR 与发布门禁等机制；真实 CUDA 性能和模型质量仍以各课列出的 GPU 测试、Benchmark 与冻结评测为准。
