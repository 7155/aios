# AIOS-IME Frozen Eval v2

三条 lane 保持独立，不合并成一个容易误导的总分。上下文候选使用共享 Prefix KV 的
decode 打分；同拼音候选使用整序列 Prefill 稳定复评分；两者都按原始平均 token
logprob 排序。

15 条真实生成的 LCP 只作唯一参考诊断，不等于语义接受率；其人工 accept/reject 仍待
标注。

| Lane | Rows | Primary | Secondary | p95 |
|---|---:|---:|---:|---:|
| Post-training generation | 15 | Full Top-3 100.00% | Best Top-3 LCP 0.000 | 106.70 ms |
| Context ranking | 145 | Acceptable Top-1 76.55% | Pairwise 71.31% | 59.86 ms |
| Same-pinyin ranking | 40 | Acceptable Top-1 87.50% | Pairwise 94.35% | 7.91 ms |

逐条 prefix、参考答案、候选文本和人工标注不进入公开仓库。完整记录由
`scripts/eval_aios_ime_frozen.py` 在本地生成；公开报告只保留评测协议和聚合指标。
