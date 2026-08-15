# AIOS-IME 优化与验收结论

日期：2026-08-14；GPU：NVIDIA GeForce RTX 4080 Laptop GPU；精度：BF16。

## 最终运行时配置

- 模型：最终 MiniMind-IME 0.1B，在线参数 100,687,360，BF16 权重 192.05 MiB。
- 导出源：`out/ime_100m_hq100_ab_user_daily_core_v1_20260814/best_validation.pt`。
- checkpoint SHA256：`9933f2418e058ff78e4f79d023d33e4a5d4986b169de187a965a999fc31b86b9`。
- 输入：裸中文上下文，前置 BOS；不加 chat template，不把拼音串直接喂给语言模型。
- 生成：先并行 8 路，最多 12 token；不足三条时以更高温度补 4 路，最多 12 路。
- 选择：硬过滤、显示归一化去重、原始平均 token logprob、软重复惩罚、字符 bigram MMR Top-3。
- 资源：低显存配置 256 个 KV token page、1 MiB FlashInfer workspace；每个 KV token 为 14 KiB。

## 端到端结果

同一冻结数据文件、5 次预热、30 条计时样本：

| Runtime | Top-3 p50 | Top-3 p95 | Peak allocated | 满三条 | 三条互异 |
|---|---:|---:|---:|---:|---:|
| 原 MiniMind PyTorch | 258.64 ms | 279.43 ms | 216.44 MiB | — | — |
| AIOS-IME 低显存配置 | 81.98 ms | 109.97 ms | 227.10 MiB | 100% | 100% |

AIOS-IME 的 p50 为原路径的 3.15x，p95 为 2.54x；峰值 allocated 增加 10.66 MiB，来源是有界 Paged KV 与 FlashInfer workspace，而不是模型权重膨胀。若业务需要更长上下文，可把 KV profile 调到 384/2 MiB 或 768/8 MiB；低显存 profile 的可用上下文容量由 KV page 数决定。

候选组首步微基准同时完成一次 Prefix Prefill 和 8 路首 token 采样，短输入 p50/p95 约 7.58/8.26 ms；它不是完整 Top-3 墙钟。

## 冻结质量门禁

| Lane | AIOS-IME | 原最终 0.1B | 结论 |
|---|---:|---:|---|
| 上下文 acceptable Top-1 | 76.55% | 76.55% | 持平 |
| 上下文 pairwise | 71.31% | 71.31% | 持平 |
| 同拼音 acceptable Top-1 | 87.50% | 87.50% | 持平 |
| 同拼音 pairwise | 94.35% | 94.35% | 持平 |

共享 decode 在“世间/时间”这种约 0.004 nat 的 BF16 近平局上发生过 kernel 数值翻转。因此同拼音最终排序使用整序列 varlen Prefill 稳定复评分；上下文生成仍使用共享 Prefix KV 的 decode 分数。这个分流恢复了原最终 0.1B 的冻结排序指标。

15 条 checkpoint 完成后真实消息中，AIOS 返回满三条为 100%，但唯一参考的 Top-3 LCP 均为 0，逐条输出仍有明显语义错位。LCP 本来也不是开放候选的接受率；这些结果必须人工 accept/reject。结论是：推理运行时已正确、快速且未破坏排序质量，但模型语义质量尚不能宣称生产可用，后续瓶颈仍是最终模型实际 Top-3 的人工反馈与定向监督，而不是继续美化运行时指标。

## 关键 A/B

| 试验 | p50 / p95 | 满三条 | 决策 |
|---|---:|---:|---|
| 12 token，固定低温补采样 | 87.72 / 140.90 ms | 93.33% | 有两组不足三条 |
| 16 token | 115.55 / 225.71 ms | 86.67% | 更长病句触发 16 字过滤，不保留 |
| 12 token，高温 refill + active-row compaction | 81.98 / 109.97 ms | 100% | 最终默认 |

“多生成几个 token”没有自动提升质量。0.1B 会把短候选拖成长句，导致过长和截断；更有效的是保持 12-token 上限，只在候选不足时改变采样分布并补少量分支。

## 单用户系统行为

- `new_generation()` 原子地取消旧 generation；GPU 每个 token step 检查一次，旧组不与新组排队混批。
- 实测取消发生在旧组生成 8 个 token 后，新组正常返回三条；`reset_prefix_cache()` 后 768/768 测试页全部回收。
- 分支提前 EOS/标点后从 active rows 移除，后续不再为空分支执行 decode。
- 初始分支完成并物化文本/分数后立即释放其 suffix KV，再决定是否补采样；补采样不与旧分支同时占页。
- 同一候选组的所有 page-table row 指向同一组 Prefix page，suffix page 独占。

## 跨按键 Prefix KV

token-LCP 增量 Prefill 已实现并通过增量/完整排序一致性检查。22 字短输入逐键 A/B 的累计加速为 1.00x，说明短 Prefix 被 metadata/plan 开销主导；95 字长上下文的累计加速为 1.056x。该优化主要服务长上下文，短输入不报告额外加速收益。

## 功能范围

- 没有把跨用户 Continuous Batching 计入单用户 IME 收益。
- 没有实现或宣称 CUDA Graph、量化、Speculative Decoding、候选分支晋升。
- 没有实现拼音词典；运行时边界是“词典提供同拼音中文候选，AIOS 按中文上下文稳定排序”。
- 没有把 100% 满三条、LCP、非空率写成语义准确率。

## 证据

- `reports/aios_ime_benchmark_final_20260814.md`
- `reports/aios_ime_frozen_eval_v2_20260814.md`
- `reports/aios_ime_prefix_reuse_20260814.md`
- `reports/aios_ime_prefix_reuse_long_20260814.md`
- `reports/aios_ime_benchmark_max16_20260814.md`
- `scripts/eval_aios_ime_frozen.py`
- `benchmark/bench_ime.py`

逐条 JSON 含冻结输入或真实历史 prefix，仅在本地生成并由 `.gitignore` 排除；公开仓库保留
聚合报告与可复现脚本。
