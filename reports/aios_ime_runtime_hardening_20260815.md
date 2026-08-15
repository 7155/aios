# AIOS-IME 采样与导出增强 A/B

日期：2026-08-15；GPU：NVIDIA GeForce RTX 4080 Laptop GPU；模型：最终 MiniMind-IME
0.1B BF16。

## 运行时改动

1. 达到 `min_new_tokens` 前，在采样副本屏蔽 EOS 与句末 stop token；排序继续读取原始
   logits 的 token logprob。
2. 候选随机数由 `(candidate_seed, token_step)` 唯一确定，并预计算为小型 GPU 矩阵；
   active-row compaction 不改变其余候选的随机流。
3. 导出器校验 dense/MoE、YaRN、层连续性、Q/K/V、QK-Norm、激活函数、KV head 形状和
   tied/untied LM head，同时将 FP32 训练权重转换为 BF16 部署权重。

CandidateGroup 的 Prefix page 由候选组统一持有，候选 row 只引用 page ID，suffix page
独立释放，因此当前路径不额外维护候选级 Prefix KV 引用计数。

## 同机 A/B

固定同一模型、冻结数据文件、5 次预热和 30 条计时样本：

| 指标 | 修改前 | 修改后 |
|---|---:|---:|
| Top-3 p50 | 82.66 ms | 81.98 ms |
| Top-3 p95 | 111.70 ms | 109.97 ms |
| 满三条 | 100% | 100% |
| 三条互异 | 100% | 100% |
| Raw invalid | 9.52% | 6.97% |
| 平均尝试路数 | 8.40 | 8.13 |
| Peak allocated | 226.85 MiB | 227.10 MiB |

随机算法改变后生成文本会随之改变，因此 raw-invalid 与尝试数只作为固定协议下的运行
指标，不解释为语义准确率变化。完整 Top-3 p95、候选完整率和显存均未退化。

## 冻结评测

| Lane | 修改后 | 原门禁 | 结论 |
|---|---:|---:|---|
| Context acceptable Top-1 | 76.55% | 76.55% | 持平 |
| Context pairwise | 71.31% | 71.31% | 持平 |
| Same-pinyin acceptable Top-1 | 87.50% | 87.50% | 持平 |
| Same-pinyin pairwise | 94.35% | 94.35% | 持平 |
| Post-training generation 满三条 | 100% | 100% | 持平 |

## 导出与测试

- 在线参数：100,687,360。
- 训练期 MTP：剥离 15 个 tensor。
- Tied LM head：逐值确认等于 embedding 后省略。
- 权重：全部 BF16，192.05 MiB。
- 导出部署包已重新加载并通过真实 GPU 测试。
- 测试结果：15 项 CPU 测试与 4 项 GPU 集成测试，共 `19 passed`。
