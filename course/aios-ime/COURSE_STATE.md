# AIOS-IME Course State

## Source pin

- Repository: `7155/aios`
- Canonical revision: `9f53740753de36899aa7694cf7dcb5304e58ea54`
- Legacy specialty lesson pin: `bfc72896bbadab5c897672506d237c070900412e`
- Course version: `v5.0`
- Last rebuilt: `2026-08-23`

## Frozen decisions

1. 使用双轨：`实战上手` 与 `源码学习`。
2. 两轨共享概念 ID、源码 revision 与 evidence ledger，但不逐课镜像。
3. 实战轨由真实任务和可观察问题推进；源码轨沿真实 runtime spine 打开黑盒。
4. 关键代码必须直接进入 README；源码路径只作为证据来源。
5. 学习者问题区统一命名为 `练习题`。
6. 旧 `resources/lesson-10`～`lesson-17` 保留为历史切片，不伪装成当前 revision。
7. 当前默认候选合同是首轮 8 路、最多 24 路、显示 Top-3。
8. 0.214B 的 AttnRes 性能数字只引用冻结报告；本次课程构建未重新运行 GPU 基准。

## Completion status

| Area | Status | Evidence |
|---|---|---|
| Root navigation | complete | `README.md` |
| Curriculum | complete | `curriculum.yaml` |
| Practice track P01–P06 | complete | track READMEs |
| Mechanism track M01–M11 | complete | track READMEs |
| Evidence ledger | complete | `evidence.jsonl` |
| Structural validator | complete | `scripts/validate_course.py` |
| GPU reproduction | not run in course build environment | requires CUDA + exported model |

## Known black-box debt

| Debt | Opened in |
|---|---|
| `engine.complete()` initially treated as one call | M01 |
| Candidate rows initially treated as ordinary batch | M03 |
| Row compaction initially treated as pure optimization | M04 |
| Refill initially treated as “再多生成几条” | M05 |
| Prefix reuse initially treated as string prefix reuse | M06 |
| Cancellation initially treated as dropping output | M07 |
| AttnRes initially treated as a model label | M09 |
| Triton initially treated as a generic speed switch | M10 |
| “100% 满三条” initially treated as quality | M11 |

## Exact next action after source changes

1. Compare new source against `9f53740753de36899aa7694cf7dcb5304e58ea54`.
2. Update `evidence.jsonl` first.
3. Repair the smallest affected lessons and bridge entries.
4. Run `python course/aios-ime/scripts/validate_course.py`.
5. Re-run GPU tests/benchmarks only when behavior or performance claims changed.
6. Record new revision and reproduced evidence; never silently carry old numbers forward.
