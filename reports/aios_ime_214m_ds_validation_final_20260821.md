# AIOS-IME 单用户 CandidateGroup 基准

- GPU：`NVIDIA GeForce RTX 4080 Laptop GPU`；模型：`ime_214m_block_attnres_v1` BF16；AttnRes backend：`triton`；显示 Top-3，内部先并行 `8` 路，有效候选不足时最多补到 `24` 路。
- 预热：`5`；计时样本：`30`；不含首次 JIT/wrapper 规划。
- Top-3 p50/p95：`505.54/557.24 ms`。
- 实际模型工作量：`347.95 active tokens/s`（只计 Prefix 新算 token 与 Decode 活跃 row）。
- 模型加载后/基准峰值 allocated：`433.69/468.19 MiB`。
- 返回满三条比例：`100.00%`；三条互异比例：`100.00%`。
- 冻结参考诊断：Top-1/Top-3 exact `30.00%/33.33%`，Top-1/Top-3 首字方向 `66.67%/70.00%`，mean best LCP `5.63/6.27`。

计时口径包含一次 Prefix Prefill、同组分支共享 Prefix KV 的 decode、GPU 原始 logprob、CPU
统一解码、过滤、去重和 MMR Top-3；不包含模型加载和首次 JIT 编译。
