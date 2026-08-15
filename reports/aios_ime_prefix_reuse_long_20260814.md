# AIOS-IME 跨按键 Prefix KV 复用 A/B

- 输入：从 `20` 字开始，逐字增长到 `95` 字；重复 `3` 轮。
- 每次完整 Prefill：p50/p95 `7.56/9.37 ms`，累计 `1814.22 ms`。
- token-LCP 增量 Prefill：p50/p95 `7.37/8.33 ms`，累计 `1717.83 ms`。
- 整段按键序列累计加速：`1.06x`。

该微基准把生成限制为一个 token，用来隔离 Prefill；完整 Top-3 延迟还包含多路 decode、过滤和排序。
