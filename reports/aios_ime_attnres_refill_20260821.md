# AIOS-IME 0.214B AttnRes 推理与无效候选补采样

## 结果总览

本轮为 `MiniMind-IME-0.214B-Block-AttnRes-v1` 增加原生 AIOS 推理路径，并将候选恢复从
固定补采样改为按过滤后有效产出动态规划。

测试环境：NVIDIA GeForce RTX 4080 Laptop GPU、BF16、CUDA、FlashInfer；模型为
32 层、8 个 AttnRes block、每 block 4 层、16K 词表、`attnres_alpha=1`。

| 结果 | 优化前 | 优化后 |
|---|---:|---:|
| 65 Mixer pipeline | Eager 17.34 ms | Triton 4.02 ms |
| 65 Mixer active tokens/s | 461.47 | 1,989.46 |
| Mixer 临时 peak allocated | 9.48 MiB | 0.94 MiB |
| 固定 8 路 Top-3 p50 | Reference 389.47 ms | Triton 254.99 ms |
| 固定 8 路 Top-3 p95 | Reference 422.43 ms | Triton 261.44 ms |
| 固定 8 路 active tokens/s | 237.35 | 366.70 |
| DS validation 满三条且互异 | 固定 8 路 23.33% | 自适应最多 24 路 100% |
| DS validation Top-3 exact | 30.00% | 33.33% |
| DS validation Top-3 首字方向 | 60.00% | 70.00% |

AttnRes 算子 pipeline 相对训练所用 Eager 路径提速 `4.31×`，算子临时显存下降约
`10.04×`。完整 CandidateGroup 包含 Attention、MLP、采样和候选后处理，因此端到端
p50/p95 提速分别为 `1.53×/1.62×`，active tokens/s 提升 `1.54×`。

## Block AttnRes 推理路径

AIOS 保留原 Qwen3/MiniMind Attention、RoPE、QK Norm、SwiGLU、Paged KV 与 LM Head，
只按配置切换 residual trunk。旧配置没有 `residual_type` 时继续走原 standard residual，
0.214B 则执行：

```text
embedding bank
  ↓
8 blocks × 4 Transformer layers
  ↓ 每层 pre-attention / pre-MLP 各一次 depth mix
8 个 block-local residual delta
  ↓
final AttnRes mixer
  ↓
final RMSNorm → tied LM Head
```

每次 Mixer 严格依赖刚产生的 Attention/MLP delta，65 次调用不能跨层提前合并。本轮优化
集中在单个 Mixer 内：

1. `alpha=1` 只计算 AttnRes，不再构造无效的 standard residual sum；
2. 接口直接接收 `bank + partial`，不物化大尺寸 `torch.cat`；
3. 将 `q · RMSNorm(source)` 展开为 RMS reduction 与 query dot，不生成完整 normalized bank；
4. 预分配 depth bank、score 和 mixed-output scratch；
5. Triton score kernel 完成 RMS 与打分，value kernel 完成 depth softmax 与加权聚合；
6. `num_tokens` 不参与 Triton specialization，避免短前缀长度变化触发运行时 JIT 抖动。

### 65 Mixer CUDA profiler

固定 `N=8, D=768, 65 mixers`，5 次预热、50 次计时：

| Backend | Pipeline latency | Active tokens/s | Mixer tokens/s | Peak allocated |
|---|---:|---:|---:|---:|
| Materialized Reference | 12.67 ms | 631.53 | 41,049.29 | 9.61 MiB |
| Direct Eager（训练语义） | 17.34 ms | 461.47 | 29,995.33 | 9.48 MiB |
| `torch.compile` hot path | 15.87 ms | 504.12 | 32,767.94 | 9.27 MiB |
| Triton two-kernel | **4.02 ms** | **1,989.46** | **129,314.68** | **0.94 MiB** |

Profiler 中 Triton 每次 Mixer 只启动两个 kernel；一次 65-Mixer pipeline 的 score/value CUDA
self time 分别约 `111.57/125.82 μs`。`torch.compile` 能减少部分 CUDA 工作，但短序列下仍
受动态形状、Python 调用与 kernel launch 影响，所以部署包默认选择 Triton。

### 真实模型数值合同

同一导出权重、裸中文 Prefix、BF16、相同 KV 与固定 8 路 seed，对比训练所用 Direct Eager
和 Triton：

- 完整词表 logits max/mean absolute diff：`0.09375 / 0.01377584`；
- cosine similarity：`0.999950827`；
- Top-1/3/10/50 token 集合全部一致；
- 固定 8 路候选文本、token 数、停止原因和过滤结果全部一致；
- average logprob 最大差：`0.00013712`。

这是 65 次 BF16 reduction 次序不同产生的数值差异；部署验收使用离散候选一致、Top-k
一致和明确数值阈值，不把 BF16 路径包装成 bitwise FP32 等价。可重复检查见
[运行时等价性记录](aios_ime_attnres_214m_equivalence_20260821.md)。

## 自适应无效候选补采样

首轮仍并行生成 8 路。只有硬过滤和显示级去重后不足三条时才进入恢复路径：

```text
8 路首轮
  ↓
过滤 + 显示归一化去重
  ↓
统计 unique-valid yield 与 Top-3 deficit
  ↓
估算本轮所需分支数（2～8 路）
  ↓
fresh seed + 渐进提高 temperature/top-k
  ↓
禁止再次生成已经见过的完整 token sequence
  ↓
满三条 / deadline / 24 路总预算
```

当前补采样参数为 `temperature=0.75→0.95`、`top-k=96→160`、`top-p=0.95`。重复规避不强迫
首 token 不同；候选共享两个 token 后，仅在下一步屏蔽会完整复现旧序列的 token。这样既
允许自然候选拥有相同开头，也能修复“24 路全部复读同一条未完成句”的失败模式。

此外，若 tokenizer 把句号和下一句开头放在同一个 token 中，后处理会截到第一个完整句，
并按实际保留的 token 前缀重新计算平均 logprob，避免把未显示部分计入排序。

### DS Daily validation A/B

数据：`ds_daily_completion_eval_v2_20260817/validation.jsonl`；5 次预热、30 条计时；相同模型、
Triton backend、seed、12-token decode 与过滤排序流程。

| 指标 | 固定 8 路 | 自适应最多 24 路 |
|---|---:|---:|
| 满三条 | 23.33% | **100%** |
| 三条互异 | 23.33% | **100%** |
| 平均实际路数 | 8.00 | 13.60 |
| 平均 refill rounds | 0 | 0.77 |
| Top-1 exact | 30.00% | 30.00% |
| Top-3 exact | 30.00% | **33.33%** |
| Top-3 首字方向 | 60.00% | **70.00%** |
| Mean best Top-3 LCP | 5.77 | **6.27** |
| p50 | 251.70 ms | 505.54 ms |
| p95 | 285.78 ms | 557.24 ms |
| Active tokens/s | 411.34 | 347.95 |
| Peak allocated | 468.18 MiB | 468.19 MiB |

补采样将候选栏完整率和互异率补到 100%，没有降低 Top-1 exact；代价只发生在首轮不足的
请求上。平均实际生成 13.6 路而不是无条件 24 路，峰值显存基本不变，因为每轮候选文本
物化后立即释放 suffix KV，再启动下一轮。

## 部署包

```text
source checkpoint SHA-256:
9f05b15f07f171ef64d46e2a9e927e466ee2c6188f6c5deef3466ad33f2f8008

model.safetensors SHA-256:
39b555564bf5d768599f7ac787d1244b7055781db00a24d342c5edd9a27fa38a

inference parameters: 214,063,360
inference tensors:     484
BF16 weight bytes:     428,126,720
residual_type:         block_attnres
blocks/layers:         8 × 4
attnres_alpha:         1.0
attnres_backend:       triton
MTP exported:          false
```

导出器会拒绝非 `alpha=1` 快照、错误 block 划分、缺少 query/key norm、错误 tied embedding、
MoE/YaRN 或不受支持的激活函数。

## 验证

```text
pytest -q
29 passed, 4 skipped

旧 0.1B GPU compatibility
4 passed

新 0.214B GPU CandidateGroup / Prefix LCP / latest-wins
3 passed
```

全量测试还覆盖 AttnRes operator、零 query 均匀权重、配置门禁、导出权重名称、采样行级
token ban、候选池统计、动态 refill 规划、句末截断和 KV page 回收。

## 复现命令

```bash
source scripts/activate_aios.sh

python benchmark/profile_attnres.py \
  --backend triton \
  --active-tokens 8 \
  --hidden-size 768 \
  --warmup 5 \
  --iterations 50 \
  --output-json reports/aios_ime_attnres_triton_final_profile.json

python scripts/check_attnres_runtime_equivalence.py \
  --model /path/to/minimind-ime-0.214b-aios \
  --reference-backend eager \
  --candidate-backend triton

python benchmark/bench_ime.py \
  --model /path/to/minimind-ime-0.214b-aios \
  --attnres-backend triton \
  --eval-data /path/to/ds_daily_completion_eval_v2/validation.jsonl \
  --warmup 5 \
  --samples 30 \
  --sampling-attempts 8 \
  --max-sampling-attempts 24 \
  --refill-batch-size 8 \
  --max-new-tokens 12 \
  --max-candidate-chars 32
```
