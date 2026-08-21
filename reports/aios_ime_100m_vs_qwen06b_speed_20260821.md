# MiniMind-IME 0.1B 与 Qwen3-0.6B BF16 延迟对比

## 结论

在同一块 NVIDIA GeForce RTX 4080 Laptop GPU、同一 AIOS CandidateGroup 运行时、同一批
冻结中文前缀和相同生成合同下，MiniMind-IME 0.1B 的完整固定 8 路 Top-3 p50/p95 为
`83.24/96.24 ms`，Qwen3-0.6B 为 `170.85/197.96 ms`。

| BF16 模型 | 参数量 | Top-3 p50 | Top-3 p95 | 峰值 allocated | 满三条且互异 |
|---|---:|---:|---:|---:|---:|
| MiniMind-IME 0.1B Standard | 100.69M | **83.24 ms** | **96.24 ms** | **238.60 MiB** | **96.67%** |
| Qwen3-0.6B | 596.05M | 170.85 ms | 197.96 ms | 1,283.99 MiB | 86.67% |

在这组固定 8 路基准中，Qwen3-0.6B 的 p50/p95 分别是 MiniMind-IME 0.1B 的
`2.05×/2.06×`；MiniMind-IME 的峰值 allocated 低 `81.42%`。实际模型工作量吞吐分别为
`1,061.35` 和 `485.23 active tokens/s`。

## 测试合同

- 精度：两侧均为 BF16。
- GPU：NVIDIA GeForce RTX 4080 Laptop GPU。
- 数据：`ime_eval_v2_frozen_20260814/all_200.jsonl` 中相同顺序的唯一前缀。
- 预热：5 条；计时：随后 30 条。
- 候选：固定 8 路，显示 Top-3，不进行第二轮补采样。
- 生成：最多 12 tokens，Top-p 0.9，固定 seed 规则。
- KV/Attention：512-token Paged KV，8 MiB FlashInfer workspace。
- 计时范围：Prefix Prefill、候选组 Decode、原始 logprob、统一解码、过滤、去重和 MMR
  Top-3；不包含模型加载和首次 CUDA JIT。

该结果用于比较同一推理合同下的延迟、显存和候选完整形态，不替代模型质量评测。质量结论
仍应使用各模型适配后的冻结验证集和人工盲评，不能从这张速度表推导。

## 复现命令

```bash
source scripts/activate_aios.sh

python benchmark/bench_ime.py \
  --model /home/codex/ai/models/minimind-ime-0.1b-aios \
  --samples 30 --warmup 5 \
  --sampling-attempts 8 --max-sampling-attempts 8 \
  --max-new-tokens 12 \
  --kv-cache-max-tokens 512 --attention-workspace-mib 8 \
  --output-json reports/aios_ime_100m_fixed8_speed_20260821.json \
  --output-markdown /tmp/aios_ime_100m_fixed8_speed.md

python benchmark/bench_ime.py \
  --model /home/codex/ai/cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --samples 30 --warmup 5 \
  --sampling-attempts 8 --max-sampling-attempts 8 \
  --max-new-tokens 12 \
  --kv-cache-max-tokens 512 --attention-workspace-mib 8 \
  --output-json reports/aios_ime_qwen3_06b_fixed8_speed_20260821.json \
  --output-markdown /tmp/aios_ime_qwen3_06b_fixed8_speed.md
```

原始 JSON 保留为本机实验产物并由 `.gitignore` 排除；本报告保存协议和汇总数字。
