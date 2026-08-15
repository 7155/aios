# AIOS-IME 单用户 CandidateGroup 基准

- GPU：`NVIDIA GeForce RTX 4080 Laptop GPU`；模型：MiniMind-IME 0.1B BF16；显示 Top-3，内部并行 8 路。
- 预热：`5`；计时样本：`30`；不含首次 JIT/wrapper 规划。
- Top-3 p50/p95：`87.72/140.90 ms`。
- 模型加载后/基准峰值 allocated：`206.81/240.85 MiB`。
- 返回满三条比例：`93.33%`；三条互异比例：`93.33%`。

计时口径包含一次 Prefix Prefill、8 路共享 Prefix KV 的 decode、GPU 原始 logprob、CPU
统一解码、过滤、去重和 MMR Top-3；不包含模型加载和首次 JIT 编译。
