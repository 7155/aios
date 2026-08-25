# Changelog

## v5.1 — 2026-08-23

- 新增无构建步骤的课程网页入口：双轨导航、搜索、进度、上一篇/下一篇、Mermaid、KaTeX 与代码复制。
- 新增 `site-manifest.json`，让网页导航与 17 课清单由同一结构化数据驱动。
- 新增 `LEARNING_PATHS.md`：起点诊断、90 分钟总览、一日实战、三日源码和 AttnRes 专题。
- 新增 `VISUAL_ATLAS.md`：14 个聚焦 Mermaid 图块，覆盖 CandidateGroup、物理 KV、Ragged RNG、Refill、token-LCP、latest-wins、治理、AttnRes 与证据。
- 新增 `SOURCE_MAP.md` 与 `source-symbols.json`，按问题、Owner 与 canonical symbol 定位源码。
- 新增 `WORKBOOK.md`，把 P01～P06 从阅读目录升级为可累积阶段产物链。
- 新增 `TROUBLESHOOTING.md`，从症状进入观测字段、机制课和最小修复。
- 扩展课程校验器：Manifest、源码 Symbol、代码围栏、支持文档与网页资产进入同一门禁。
- 新增 GitHub Actions，在课程或导航变化时自动运行结构校验与 JavaScript 语法检查。
- 明确 Markdown 是事实源，网页进度只是个人提示；性能与 GPU 结论仍需目标环境复测。

## v5.0 — 2026-08-23

- 固定当前源码 revision `9f53740753de36899aa7694cf7dcb5304e58ea54`。
- 将 AIOS-IME 从单线性目录重建为 `实战上手 + 源码学习` 双轨。
- 加入渐进披露、概念 canonical owner、跨轨 bridge 和 black-box debt。
- 将当前默认候选合同更新为 `8 → 最多 24 → Top-3`。
- 新增自适应补采样、候选池可观测字段和逐轮重复规避课程。
- 新增 0.06B / 0.1B / 0.214B 五模型矩阵阅读课。
- 新增 0.214B Block AttnRes 与 Triton 热路径课程。
- 所有新课统一使用 `练习题`，并把关键代码直接嵌入 README。
- 保留旧 `resources/lesson-10`～`lesson-17` 为历史切片，不删除源码演进证据。
