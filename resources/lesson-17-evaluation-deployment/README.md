# Lesson 17：完整 Top-3 的性能、冻结评测与部署证据

> 源码基线：`bfc72896bbadab5c897672506d237c070900412e`
>
> 这是专项篇的验收课。目标不是再加一个优化，而是学会回答：**怎样证明运行时真的更快、没有破坏排序质量、没有泄漏 KV、部署包身份明确，同时不把结构指标包装成模型语义准确率？**

![AIOS-IME 性能结果](../../docs/images/aios-ime-performance.svg)

## 1. 先定义评测对象

一次完整开放生成包含：

```text
Tokenizer / BOS
→ Prefix Cache 处理
→ Prefix Prefill 或增量 Prefill
→ 8 路 CandidateGroup Decode
→ Decode 成文本
→ Hard Filter
→ Dedup / Soft Penalty / MMR
→ 可选 refill 4 路
→ Top-3 Result
```

端到端 Benchmark 必须覆盖上述完整墙钟，不包括：

- 模型加载；
- 首次 JIT 编译；
- 与本次按键无关的初始化。

## 2. 为什么首 token 微基准不能代表候选栏

当前候选组首步：

```text
一次 Prefix Prefill + 8 路首 token 采样
p50/p95 ≈ 7.58/8.26 ms
```

完整 Top-3：

```text
p50/p95 = 81.98/109.97 ms
```

差异来自后续多个 Decode Step、分支长度、CPU 解码、治理和 refill。

面试中若只报告 8ms，属于更换指标对象。

## 3. 正确的性能对照

固定：

```text
同一模型权重
同一 Tokenizer
同一输入集
同一采样配置与 seed
同一 GPU / BF16
相同预热次数
相同计时边界
```

结果：

| Runtime | Top-3 p50 | Top-3 p95 | Peak allocated |
|---|---:|---:|---:|
| MiniMind PyTorch | 258.64 ms | 279.43 ms | 216.44 MiB |
| AIOS-IME | **81.98 ms** | **109.97 ms** | **227.10 MiB** |

可以说：

```text
p50 加速 3.15x
p95 加速 2.54x
峰值 allocated 增加 10.66 MiB
```

不能只说“快 3 倍”而不说明是哪个分位、哪个计时范围。

## 4. 为什么峰值显存略增仍可能是好取舍

AIOS 增加：

- 有界 Paged KV；
- FlashInfer Workspace；
- Page Table / Metadata。

它换来更低尾延迟。评价应是 Pareto 关系：

```text
+10.66 MiB
↔ p95 从 279.43 降到 109.97 ms
```

对本地 4080 Laptop 和 0.1B 模型，这个取舍可接受；对极端低显存设备需要重新测。

## 5. 结构质量 Lane 与语义质量 Lane

### 开放生成结构

```text
满三条率
三条互异率
invalid reason 分布
候选长度
```

### 固定候选排序质量

```text
上下文 acceptable Top-1 / pairwise
同拼音 acceptable Top-1 / pairwise
```

### 人工语义

```text
真实 Prefix 下候选是否自然、贴切、可接受
```

当前冻结结果：

| Lane | AIOS-IME | 原最终 0.1B |
|---|---:|---:|
| 上下文 acceptable Top-1 | 76.55% | 76.55% |
| 上下文 pairwise | 71.31% | 71.31% |
| 同拼音 acceptable Top-1 | 87.50% | 87.50% |
| 同拼音 pairwise | 94.35% | 94.35% |

这证明：运行时改造没有破坏固定排序 Lane。

但 15 条开放真实消息虽然满三条 100%，唯一 Reference 的 Top-3 LCP 均为 0。这个结果不能解释为“准确率 0”，也不能解释为“已生产可用”，因为开放补全存在多个合理答案，必须人工 accept/reject。

## 6. 为什么测试必须覆盖资源不变量

CPU 测试：

- Filter / Dedup / MMR；
- raw logprob 不受 temperature 影响；
- stateless random stream；
- token LCP；
- Exporter 拒绝错误模型。

GPU 测试：

- 返回三条互异候选；
- 分支 Suffix Page 回收后只剩 Prefix Page；
- 相邻 Prefix 复用；
- stable 排序不发生已知近平局翻转；
- latest-wins 取消旧组；
- reset 后所有 Page 回收。

这里 Page 可用数量相等是一项**内存生命周期正确性断言**，不是性能指标。

## 7. 部署导出流程

```text
训练 checkpoint
→ 读取并规范化 state_dict 名称
→ 剥离 MTP
→ 验证 dense / SiLU / QK Norm / Head Shape
→ 验证 tied/untied LM Head
→ 转 BF16
→ 保存 model.safetensors
→ 复制 Tokenizer
→ 写 config.json
→ 写 aios_manifest.json 与 SHA
→ AIOS LLM 加载
```

运行：

```bash
python scripts/export_minimind_ime.py \
  --checkpoint /path/to/best_validation.pt \
  --config /path/to/ime_100m_v1.json \
  --tokenizer-dir /path/to/tokenizer \
  --output-dir /path/to/minimind-ime-0.1b-aios
```

## 8. 端到端验收命令

### CPU

```bash
pytest -q tests/test_ime.py tests/test_ime_export.py
```

### GPU

```bash
AIOS_IME_MODEL=/path/to/minimind-ime-0.1b-aios pytest -q tests/test_ime_gpu.py
```

### 完整 Top-3 Benchmark

```bash
python benchmark/bench_ime.py \
  --model /path/to/minimind-ime-0.1b-aios
```

### Prefix 复用 A/B

```bash
python benchmark/bench_ime_prefix_reuse.py \
  --model /path/to/minimind-ime-0.1b-aios
```

### 冻结评测

```bash
python scripts/eval_aios_ime_frozen.py \
  --model /path/to/minimind-ime-0.1b-aios \
  --eval-dir /path/to/frozen_eval
```

## 9. 发布决策表

| 问题 | 必须给出的证据 |
|---|---|
| 模型身份正确吗 | Checkpoint/导出 Hash、结构 Manifest |
| 推理结果结构正确吗 | 满三条、互异、Filter 测试 |
| 原排序质量保留吗 | 冻结 acceptable/pairwise Lane |
| latest-wins 正确吗 | 并发取消 GPU Test |
| KV 泄漏吗 | reset 后全部 Page 可用 |
| 用户实际等多久 | 完整 Top-3 p50/p95 |
| 峰值资源可接受吗 | 同设备 Peak allocated |
| 语义可发布吗 | 独立人工 accept/reject，不由运行时指标替代 |

## 10. 当前明确没有实现什么

课程和 README 必须保留范围边界：

- 没有把跨用户 Continuous Batching 算作单用户收益；
- 没有 CUDA Graph；
- 没有量化；
- 没有 Speculative Decoding；
- 没有 Candidate Branch Promotion；
- 没有拼音词典；
- CacheManager 没有通用 eviction；
- 当前 Context 是进程级单实例。

这些不是“以后一定要做”，而是后续优化必须先由 Profile 和产品约束证明价值。

## 11. 常见错误解释

### 错误：p50 快就说明输入法稳定

输入法对尾延迟敏感，必须同时看 p95 和满三条率。

### 错误：冻结排序持平就说明开放生成质量持平

固定候选排序与开放采样是不同任务，需要不同 Lane。

### 错误：19 passed 就说明可生产

测试证明实现不变量；真实语义、设备兼容、长期稳定和用户接受仍需额外证据。

## 12. 运行实验

```bash
python resources/lesson-17-evaluation-deployment/run_lesson17.py
```

它会对多个候选版本执行硬门禁：任何一个版本即使 p50 更快，只要满三条率、冻结排序或 Page 回收不合格，就不能晋级。

## 13. 面试追问

1. 为什么 p95 对输入法候选栏比 tokens/s 更重要？
2. 如何设计 Benchmark 避免把首次 JIT 编译算进每次请求？
3. 为什么固定候选排序与开放生成必须拆 Lane？
4. 如果显存多 10 MiB 但 p95 减半，你怎样判断值不值？
5. 为什么“测试全部通过”与“模型生产可用”是两层结论？

## 14. 全课程最终复述

请不看教材讲清一次按键：

```text
prefix
→ tokenize + BOS
→ token-LCP
→ prefix page keep/free/extend
→ one prefill
→ prefix_logits expand
→ 8 independent stateless streams
→ active-row ragged decode
→ suffix page allocate/free
→ raw logprob + hard filter + dedup + MMR
→ optional refill
→ latest generation check
→ Top-3 result
```

并能说出每一步的 Owner、状态、失败路径与验证证据。

## 15. 一句话复述

AIOS-IME 的交付证据必须同时覆盖模型身份、运行时不变量、完整 Top-3 尾延迟、峰值显存、冻结排序质量和人工语义；任何单一微基准、满三条率或测试通过数都不能单独代表生产可用。
