# Block AttnRes 真实模型运行时等价性

- 模型：`/mnt/d/codex-artifacts/minimind-ime/aios_ime_214m_foundation50m_daily_sft_20260821`
- 架构：`ime_214m_block_attnres_v1`
- 对比：`eager` → `triton`
- Prefix logits max/mean absolute diff：`0.09375` / `0.01377584`
- Cosine similarity：`0.9999508262`
- Top-1/3/10/50 overlap：`1/1`、`3/3`、`10/10`、`50/50`
- 固定 8 路候选文本/停止状态完全一致：`True`
- 候选 average logprob 最大差值：`0.00013712173`

该检查加载同一份导出权重，在相同裸中文 Prefix、KV 配置、采样参数和 seed 下分别执行
Direct/Eager 与优化后端。它同时比较 Prefix 最后位置的完整词表 logits，以及固定 8 路生成
得到的原始候选文本、token 数、停止原因和 logprob。
