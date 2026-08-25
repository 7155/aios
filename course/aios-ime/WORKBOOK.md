# AIOS-IME 阶段产物工作簿

> 课程版本：`v5.1`
>
> 源码固定点：`9f53740753de36899aa7694cf7dcb5304e58ea54`
>
> 目标：把“我看懂了”改造成可追踪的命令、原始结果、解释、反例和下一步。

## 建议的个人工作目录

不要把实验输出直接混进课程源码。可以在仓库根目录建立：

```text
course-work/aios-ime/
├── 00-environment/
├── P01-first-top3/
├── P02-candidate-budget/
├── P03-continuous-typing/
├── P04-model-matrix/
├── P05-profile-attnres/
├── P06-release-gate/
├── mechanism-notes/
└── final-review/
```

建议加入自己的 `.gitignore`：

```gitignore
course-work/aios-ime/**/raw/
course-work/aios-ime/**/*.json
course-work/aios-ime/**/*.trace
course-work/aios-ime/**/*.log
```

是否提交聚合 Markdown，由你自己决定；模型、原始数据与大 Trace 不应默认入库。

## 每次实验的统一头部

在每个阶段目录保存 `RUN.md`：

```markdown
# Run Record

- Date:
- Source SHA:
- Model path / model SHA:
- Tokenizer SHA:
- GPU:
- PyTorch:
- CUDA:
- Triton:
- FlashInfer:
- dtype:
- Eval data / SHA:
- Seed:
- ImeGenerationConfig:
- Command:
- Warmup:
- Timing boundary:
- Raw output:
```

结论前必须先填身份与边界。否则两个同名 `p95` 可能根本不可比较。

## 统一判断模板

每个阶段都按下面格式收尾：

```markdown
## Observation

只写直接观察到的字段或现象。

## Explanation

把观察连接到状态、Owner、Shape 或公式。

## Counterexample

写一个会推翻当前过度结论的例子。

## Decision

keep / change / investigate / blocked

## Evidence

列出原始 JSON、Trace、Test 或报告路径。

## Next smallest action

只写一个最小下一步。
```

一个合格例子：

```text
Observation:
8→24 后满三条，但 p95 上升。

Explanation:
困难 Prefix 触发额外 refill rounds 与 active_model_tokens。

Counterexample:
满三条可能仍然三条都不自然，因此不能写“质量提升”。

Decision:
保留为恢复路径，不把 24 路改成无条件默认。

Next smallest action:
钻取 p95 前五个 Prefix 的 invalid / duplicate / refill_stop_reason。
```

## P01 工作簿：第一组 Top-3

### 运行

```bash
python course/aios-ime/scripts/run_top3.py \
  --model /path/to/minimind-ime-aios \
  --prefix "没关系，你先忙你的，" \
  --seed 20260814 \
  > course-work/aios-ime/P01-first-top3/result.json
```

### 记录表

| 字段 | 值 | 解释 |
|---|---:|---|
| `generation_id` |  | 本次按键身份 |
| `prefix_tokens` |  | 模型看到的 Token 数 |
| `reused_prefix_tokens` |  | 单次新 Engine 应为 0 |
| `sampling_attempts` |  | 实际探索路数 |
| `refill_rounds` |  | 是否进入恢复路径 |
| `valid_unique_candidates` |  | 最终可显示的不同候选 |
| `invalid_candidates` |  | 硬过滤数量 |
| `duplicate_candidates` |  | 显示等价重复 |
| `refill_stop_reason` |  | filled / budget / deadline / cancelled |
| `latency_ms` |  | 完整墙钟 |
| `gpu_latency_ms` |  | GPU 区间 |
| `unique_kv_pages` |  | 本轮资源观测 |

### 验收签字

```text
[ ] 原始 JSON 已保存
[ ] 三条最终候选显示 key 不重复
[ ] 能区分 raw_candidates 与 candidates
[ ] 没把第一次运行写成正式性能
[ ] reset_prefix_cache 正常完成
```

## P02 工作簿：候选预算对比

### 运行

```bash
python course/aios-ime/scripts/compare_candidate_budget.py \
  --model /path/to/model \
  --prefix "回头" \
  > course-work/aios-ime/P02-candidate-budget/compare.json
```

### 对照表

| 方案 | 最终条数 | unique-valid | invalid | duplicate | 实际 attempts | refill rounds | stop reason | wall ms | active model tokens |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| fixed 3 |  |  |  |  |  |  |  |  |  |
| fixed 8 |  |  |  |  |  |  |  |  |  |
| adaptive 8→24 |  |  |  |  |  |  |  |  |  |

### 失败模式分类

只选主要原因：

```text
[ ] 原始覆盖不足
[ ] hard invalid 较多
[ ] display duplicate 较多
[ ] 达到 attempts 上限
[ ] 达到 deadline
[ ] 样本太少，尚不能判断
```

### 不允许写的结论

```text
“24 路更好”
```

应改成：

```text
“在这条 Prefix、固定 seed 与当前规则下，自适应恢复把候选从 X 条补到 Y 条，代价是 attempts 与 wall time 增加。是否值得需要多样本 p95 和人工语义。”
```

## P03 工作簿：连续输入

### 运行

```bash
python course/aios-ime/scripts/trace_typing.py \
  --model /path/to/model \
  > course-work/aios-ime/P03-continuous-typing/trace.json
```

### Trace 表

| step | Prefix | generation | prefix tokens | reused tokens | cancelled | latency | Top-3 |
|---:|---|---:|---:|---:|---|---:|---|
| 0 |  |  |  |  |  |  |  |
| 1 |  |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |  |

### 另外保存一条 Backspace 或尾部改写

```text
旧 Prefix:
新 Prefix:
旧 Token IDs:
新 Token IDs:
token-LCP:
保留 Page:
释放 Page:
重新 Prefill:
```

### latest-wins 单独验收

串行 Trace 不能证明并发取消。记录 GPU Test：

```markdown
- Test:
  tests/test_ime_gpu.py::test_latest_generation_cancels_old_group_and_frees_pages
- Result:
- Old generation cancelled:
- Suffix pages returned:
- New generation completed:
- Pages after reset:
```

## P04 工作簿：模型档位

### 身份表

| 模型 | 参数量 | residual type | layers | hidden | tokenizer SHA | attention backend |
|---|---:|---|---:|---:|---|---|
| 0.06B |  |  |  |  |  |  |
| 0.1B |  |  |  |  |  |  |
| 0.214B |  |  |  |  |  |  |

### 同合同结果表

| 模型 | fixed-8 p95 | adaptive p95 | 满三条 | 首轮满三条 | 违规 | 平均 attempts | peak allocated | 人工接受度 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.06B |  |  |  |  |  |  |  |  |
| 0.1B |  |  |  |  |  |  |  |  |
| 0.214B |  |  |  |  |  |  |  |  |

### 三档决策

```markdown
## 低延迟档

- Model:
- Why:
- Evidence:
- Missing evidence:
- Reject condition:

## 默认档

- Model:
- Why:
- Evidence:
- Missing evidence:
- Reject condition:

## 高容量实验档

- Model:
- Why:
- Evidence:
- Missing evidence:
- Promotion condition:
```

参数量不能单独出现在 `Why` 中。

## P05 工作簿：AttnRes Profile

### 微内核

| Backend | 65-Mixer ms | active tokens/s | peak allocated | Kernel 形态 | dynamic N 覆盖 |
|---|---:|---:|---:|---|---|
| reference |  |  |  |  |  |
| eager |  |  |  |  |  |
| compiled |  |  |  |  |  |
| triton |  |  |  |  |  |

### 端到端固定 8 路

| Backend | Top-3 p50 | Top-3 p95 | GPU p95 | active model tokens/s | candidates equal |
|---|---:|---:|---:|---:|---|
| eager |  |  |  |  | 基准 |
| triton |  |  |  |  |  |

### 数值与离散行为

| 项目 | 阈值 | 结果 | 结论 |
|---|---:|---:|---|
| logits max abs diff |  |  |  |
| logits mean abs diff |  |  |  |
| cosine |  |  |  |
| Top-1/3/10/50 set |  |  |  |
| 固定 seed 文本 | 一致或明确容差 |  |  |
| stop reason | 一致 |  |  |
| invalid reasons | 一致 |  |  |
| average logprob diff |  |  |  |

### Amdahl 解释

```text
Mixer 加速:
完整 Top-3 加速:
未被 Mixer 覆盖的主要耗时:
为什么两者比例不同:
```

## P06 工作簿：发布门禁

### 决策头部

```text
Candidate version:
Source SHA:
Model SHA:
Tokenizer SHA:
Device:
Dependencies:
Eval SHA:
Config:
Baseline:
```

### 门禁表

| Lane | 阈值 | Baseline | Candidate | Evidence | 状态 |
|---|---|---:|---:|---|---|
| 模型身份 | 无静默兼容 |  |  |  |  |
| CPU 不变量 | 全部通过 |  |  |  |  |
| GPU Page 回收 | reset 后全部归还 |  |  |  |  |
| latest-wins | 旧组停止且不交付 |  |  |  |  |
| 完整 Top-3 p95 | 产品预算内 |  |  |  |  |
| 峰值显存 | 目标设备预算内 |  |  |  |  |
| 满三条且互异 | 目标阈值 |  |  |  |  |
| 契约违规 | 不超过阈值 |  |  |  |  |
| 冻结排序 | 基线容差内 |  |  |  |  |
| 人工语义 | 达到阈值 |  |  |  |  |

结论只允许：

```text
release
experimental only
reject
blocked — missing evidence
```

`blocked` 比没有证据时写“应该没问题”更可信。

## 机制课手算模板

保存在 `mechanism-notes/<lesson-id>.md`。

### M03 Page Budget

```text
prefix_len:
sampling_attempts:
refill_batch_size:
max_new_tokens:
max_concurrent_attempts:
required_pages:
为什么不是 max_sampling_attempts × suffix:
```

### M04 Ragged Trace

```text
candidate lengths:
step:
active_local:
page_table rows:
uniform identity:
output write positions:
```

### M05 Refill

```text
attempts:
unique-valid:
deficit:
observed yield:
yield floor:
predicted:
diversity headroom:
remaining:
planned:
stop reason:
```

### M06 token-LCP

```text
old token ids:
new token ids:
LCP:
kept pages:
released pages:
extension pages:
positions start:
prefix logits reused:
```

### M07 Cancellation

```text
generation invalidated at:
last CUDA step:
next-step suppression:
suffix page finalizer:
prefix page decision:
new group enters at:
```

### M08 Governance

```text
raw text:
raw avg logprob:
hard reasons:
soft penalty:
display key:
dedup winner:
MMR similarity:
final rank:
```

### M09/M10 AttnRes

```text
bank [S,N,D]:
partial [N,D] or None:
total sources:
score grid:
value grid:
score dtype:
weight dtype:
output dtype:
scratch owner:
dynamic shapes tested:
```

## 最终复盘文件

在 `final-review/PROJECT_REVIEW.md` 中完成：

1. 用一张图说明一次按键；
2. 用一张表说明所有状态 Owner；
3. 用一个反例说明为什么 Top-3 完整率不是语义质量；
4. 用一个时间线说明 latest-wins；
5. 用一个 Shape Ledger 说明 AttnRes；
6. 用一张发布表说明一个优化版本为何能或不能发布；
7. 列出仍未运行的验证及其影响。

这份复盘比“全部课程已读”更能证明你真正掌握了系统。
