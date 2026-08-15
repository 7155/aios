# AIOS-IME 跨按键 Prefix KV 复用 A/B

- 输入：从 `2` 字开始，逐字增长到 `22` 字；重复 `5` 轮。
- 每次完整 Prefill：p50/p95 `7.58/8.26 ms`，累计 `807.79 ms`。
- token-LCP 增量 Prefill：p50/p95 `7.61/8.75 ms`，累计 `809.27 ms`。
- 整段按键序列累计加速：`1.00x`。

该微基准把生成限制为一个 token，用来隔离 Prefill；完整 Top-3 延迟还包含多路 decode、过滤和排序。
