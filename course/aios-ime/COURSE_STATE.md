# AIOS-IME Course State

## Source pin

- Repository: `7155/aios`
- Canonical revision: `9f53740753de36899aa7694cf7dcb5304e58ea54`
- Legacy specialty lesson pin: `bfc72896bbadab5c897672506d237c070900412e`
- Course version: `v5.1`
- Last rebuilt: `2026-08-23`

## Frozen decisions

1. 使用双轨：`实战上手` 与 `源码学习`。
2. 两轨共享概念 ID、源码 revision、source symbol registry 与 evidence ledger，但不逐课镜像。
3. 实战轨由真实任务和可观察问题推进；源码轨沿真实 Runtime Spine 打开黑盒。
4. 关键代码必须直接进入 README；源码路径只作为证据来源。
5. 学习者问题区统一命名为 `练习题`。
6. 旧 `resources/lesson-10`～`lesson-17` 保留为历史切片，不伪装成当前 revision。
7. 当前默认候选合同是首轮 8 路、最多 24 路、显示 Top-3。
8. 0.214B 的 AttnRes 性能数字只引用冻结报告；本次课程构建未重新运行 GPU 基准。
9. Markdown 是课程事实源；`index.html` 只提供渲染、导航、搜索和本地进度。
10. 实战课完成必须产生工作簿产物，网页“已完成”状态只作个人提示。
11. canonical source anchor 使用 Symbol，而不是脆弱的固定行号。
12. 网页依赖固定版本的 Marked、Mermaid 与 KaTeX CDN；离线时仍可直接阅读 Markdown。

## Completion status

| Area | Status | Evidence |
|---|---|---|
| Root navigation | complete | `README.md` |
| Learning path diagnosis | complete | `LEARNING_PATHS.md` |
| Curriculum | complete | `curriculum.yaml` |
| Practice track P01–P06 | complete | track READMEs |
| Mechanism track M01–M11 | complete | track READMEs |
| Visual atlas | complete | `VISUAL_ATLAS.md` |
| Source map | complete | `SOURCE_MAP.md` |
| Stage artifact workbook | complete | `WORKBOOK.md` |
| Troubleshooting | complete | `TROUBLESHOOTING.md` |
| Evidence ledger | complete | `evidence.jsonl` |
| Source symbol registry | complete | `source-symbols.json` |
| Static web portal | complete | `index.html`, `assets/`, `site-manifest.json` |
| Structural validator | complete | `scripts/validate_course.py` |
| Pull-request CI | complete | `.github/workflows/aios-ime-course.yml` |
| GPU reproduction | not run in course build environment | requires CUDA + exported model |
| Browser smoke test on deployed Pages | not run | requires serving branch content |

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

课程结束前，上述债务均由 canonical mechanism lesson 打开。后续课程新增黑盒时，必须同步记录关闭位置。

## Current learner artifacts

| Practice lesson | Required artifact |
|---|---|
| P01 | 可复现 `ImeCompletionResult` JSON |
| P02 | fixed-3 / fixed-8 / adaptive-8→24 对照 |
| P03 | 连续 Prefix Trace + latest-wins GPU Test 记录 |
| P04 | 低延迟 / 默认 / 高容量实验三档决策 |
| P05 | 65-Mixer、固定 8 路端到端与等价性记录 |
| P06 | 多 Lane 发布门禁与明确结论 |

模板位于 `WORKBOOK.md`。

## Exact next action after source changes

1. Compare new source against `9f53740753de36899aa7694cf7dcb5304e58ea54`.
2. Update `evidence.jsonl` and `source-symbols.json` first.
3. Run `python course/aios-ime/scripts/validate_course.py`，让 Symbol Drift 先失败。
4. Repair the smallest affected canonical owner lessons.
5. Repair practice bridge、`SOURCE_MAP.md`、`VISUAL_ATLAS.md` 与 `site-manifest.json`.
6. Re-run GPU tests/benchmarks only when behavior or performance claims changed.
7. Record new revision and reproduced evidence; never silently carry old numbers forward.
8. Serve `course/aios-ime/` and smoke-test the web portal after navigation changes.

## Validation boundaries

当前结构校验可以证明：

```text
17 课存在且 ID 唯一
Manifest 与课程文件一致
相对链接可解析
代码围栏成对
每课包含图、验收、练习题和关键代码/命令
M01～M11 的 source symbols 在当前工作树存在
YAML / JSON / JSONL 与 Python/JavaScript 基础语法成立
```

它不能证明：

```text
目标 GPU 性能
模型自然度
Triton 在所有设备上的数值行为
浏览器 CDN 永久可用
生产发布结论
```
