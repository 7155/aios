# Lesson 36：Speculative Decoding、MTP 与 AIOS 下一步优化路线

> 源码基线：`c335497c6bf67a4dc8cb5ba748ace7b7c1cb77af`
>
> 当前 AIOS **没有实现 Speculative Decoding，也没有把训练期 MTP 当作在线多 Token 生成器**。这一课解释 Draft/Verify 数学、接受率、为什么短候选可能不划算，并给出 AIOS-IME 的证据优先级路线图。

## 1. 普通自回归为什么每步只能提交一个 Token

```text
Prefix
→ 主模型 Forward
→ sample token 1
→ token 1 写 KV
→ 主模型 Forward
→ sample token 2
```

虽然一次 Forward 计算整个词表 Logits，但下一位置分布依赖刚采出的 Token，所以不能直接跳过因果依赖。

## 2. Speculative Decoding 的核心

使用较便宜 Draft Model 一次提议多个 Token：

```text
Draft: d1,d2,d3,d4
```

Target Model 一次并行验证这些位置：

```text
P_target(d1|prefix)
P_target(d2|prefix,d1)
P_target(d3|prefix,d1,d2)
P_target(d4|...)
```

根据接受规则，可能一次提交多个 Token；遇到第一个拒绝位置后，由 Target 分布修正并重新开始。

核心加速条件：

```text
Draft 足够便宜
+ 接受率足够高
+ Target 一次 Verify 多 Token 比逐 Token Decode 划算
```

## 3. 简化接受率直觉

若每个 Draft Token 独立平均接受概率 `a`，一次提议 `k` 个，预期连续接受数近似：

```math
E[L]
=
\sum_{i=1}^{k} a^i
```

例如 `a=0.8,k=4`：

```math
0.8+0.64+0.512+0.4096
=2.3616
```

平均每轮可能提交约 2.36 个提议 Token（真实算法还包括修正 Token 和非独立性）。

若 `a=0.4`：

```math
0.4+0.16+0.064+0.0256
=0.6496
```

大部分提议很快拒绝，Draft 与 Verify 开销可能得不偿失。

## 4. 训练期 MTP 为什么不等于在线 Speculative

MiniMind-IME 的 Next-2 MTP 训练辅助：

```text
hidden(t) + embedding(真实 token t+1)
→ 预测 token t+2
```

训练使用 Teacher-forced 真实下一 Token。线上没有免费真实 `t+1`。

Speculative 需要：

- 在线 Draft 产生候选 Token；
- Target Model 计算验证分布；
- 接受/拒绝算法保证输出分布正确或定义近似边界；
- KV Cache 对已接受/拒绝分支做提交/回滚。

因此不能因为 checkpoint 训练过 MTP，就说线上“一次预测两个 Token”。

## 5. AIOS-IME 为什么未必优先做 Speculative

当前候选最多 12 Token，完整 Top-3 p50/p95 约 82/110ms。

Speculative 的额外固定成本：

- Draft Model 权重/显存；
- Draft Forward；
- Verify Batch 构造；
- 接受/拒绝逻辑；
- KV 暂存与回滚；
- 8 路 CandidateGroup 每路的 Draft 状态；
- Ragged 分支接受长度不同。

短输出可能来不及摊薄固定成本。先做 Profile：

```text
当前 Decode 纯 GPU 占多少？
每候选平均实际长度？
Kernel Launch 是否主瓶颈？
Draft 接受率预估多少？
```

若 CUDA Graph/Fusion 已能显著降低 Launch，Speculative 的优先级可能更低。

## 6. CandidateGroup 下的两种方案

### 每分支独立 Draft

```text
8 branches
→ 每行 Draft k tokens
→ Target Ragged Verify
```

优点：保持候选随机独立；缺点：Draft Batch/KV 更复杂。

### 共享主干后 Branch Promotion

从多候选首步中选择高概率/代表性分支，优先给少数分支更多计算，再按需补充分支。

这更像候选资源调度，不是严格 Speculative；可能减少低价值分支 Decode，但会改变候选覆盖，需要结构/语义评测。

## 7. KV 提交与回滚

若 Draft 提议 4 Token，Target 一次 Verify 后只接受前 2：

```text
临时 KV：d1,d2,d3,d4
提交：d1,d2
丢弃/释放：d3,d4
```

Paged KV 适合做临时 Page：

```text
allocate speculative pages
→ verify
→ accepted pages 进入正式 Page Table
→ rejected pages free
```

但必须保证：

- Page Identity 不重复；
- 不让 rejected K/V 被后续 Attention 读取；
- Cancellation/异常全部回收；
- 多分支峰值 Page Budget 可控。

## 8. Verify 为什么能并行

Draft 已提供整段 `d1..dk`，Target Forward 可以像训练 Teacher Forcing 一样，对整个候选片段做 Causal Prefill：

```text
Query d1 只看 prefix+d1
Query d2 看 prefix+d1+d2
...
```

一次 Flat/Varlen Prefill 产生所有位置 Logits。它仍遵守 Causal Mask，只是输入 Token 已由 Draft 提前给出。

## 9. README 内置代码：AIOS 接入骨架

```python
class SpeculativeBatch:
    branch_ids: list[int]
    draft_tokens: Tensor       # [total_draft_tokens]
    cu_draft: Tensor           # varlen boundaries
    temp_pages: Tensor

# 1. Draft 生成 k 个
proposal = draft_model.propose(prefix_state, k)

# 2. Target 一次 Varlen Verify
raw_logits = target_model.verify(proposal)

# 3. 计算每分支 accepted length
accepted = acceptance_rule(proposal, raw_logits)

# 4. Page commit/free
commit_prefix_pages(temp_pages, accepted)
free_rejected_pages(temp_pages, accepted)
```

这是教学骨架，不是当前可运行实现。

## 10. 后续优化应该按什么顺序

建议按证据，而不是潮流：

### 第 1 层：已有路径 Profile 与低风险优化

```text
修正 Benchmark
定位 CPU Launch/Metadata/Memory
现有 Fusion/FlashInfer/Triton correctness
减少不必要同步和分配
```

### 第 2 层：CUDA Graph Bucket

适合当前小模型、短 Kernel，但先解决 Active-row Shape/Metadata 静态地址。

### 第 3 层：量化

若设备显存或 GEMM 带宽是约束，优先 Weight-only A/B；不先碰 KV INT8。

### 第 4 层：Speculative / Branch Promotion

只有 Decode 时间占主导、候选长度足够、接受率高时再做。

### 第 5 层：更大系统能力

```text
多 Session Context
Tensor Parallel
全局 Prefix Cache/Eviction
跨请求 Mixed Prefill/Decode
```

这些与当前单用户本地 IME 目标不同，不应为了“像 vLLM”而提前引入。

## 11. Promotion Gate

任何后续优化必须同时比较：

```text
完整 Top-3 p50/p95
Peak allocated/reserved
满三条率/互异率
Frozen Context/Pinyin 排序
Latest-wins 取消延迟
Page 回收
人工 accept/reject
维护复杂度
```

如果优化只让微 Kernel 快，但完整 Top-3 不变，就不应合入默认路径。

## 12. 常见错误理解

### 错误：Speculative 一次生成 k 个 Token，所以必然 k 倍快

错。Draft、Verify、拒绝和 KV 管理都有成本；收益由接受率和 Target Verify 效率决定。

### 错误：训练 MTP 可以直接当 Draft Head 用

错。训练时用真实下一 Token 条件；线上需要独立提议、验证和 Cache 提交机制。

### 错误：技术越新越应优先接入

错。当前 0.1B 单用户短候选可能更受 Launch/CPU/候选质量限制。应按端到端 Profile 和产品门禁排序。

## 13. 运行实验

```bash
python resources/lesson-36-speculative-roadmap/run_lesson36.py
```

实验计算不同接受率和 Draft 长度下的预期连续接受 Token，并用简单成本模型展示什么时候 Speculative 反而更慢。

## 14. 检验问题与参考答案

### 问题 1：为什么 Target 可以一次 Verify 多个 Draft Token，却不能在没有 Draft 时直接一次生成多个？

**参考答案：** Verify 时 Draft 已提供每个未来位置的输入 Token，Target 可以用 Causal Mask 并行计算各位置条件分布；普通生成时下一位置输入尚未知，必须先采出前一 Token 才能确定后续条件。

### 问题 2：接受率低为什么会让 Speculative 变慢？

**参考答案：** Draft Forward 和 Target Verify 固定发生，但很快拒绝时只提交很少 Token，额外工作无法摊薄；还需处理临时 KV 与回滚。最终每提交一个 Token 的成本可能高于普通 Decode。

### 问题 3：Paged KV 对 Speculative 有什么帮助？

**参考答案：** 提议 Token 可写入独立临时 Page；验证后把接受的 Page 纳入正式逻辑序列，拒绝 Page 直接归还 Pool，无需搬移大块连续 Cache。但所有权、Page Table 和异常回收必须严格。

### 问题 4：AIOS 当前更应先尝试 CUDA Graph 还是 Speculative？

**参考答案：** 需看 Profile。对于 0.1B、短 Decode、Kernel 多而短的路径，若 Launch 空洞明显且 Batch 可 Bucket，CUDA Graph 通常风险更低；Speculative 还引入 Draft、接受率和 KV 回滚。不能脱离数据绝对排序。

## 15. 一句话复述

Speculative Decoding 让便宜 Draft 先给出未来 Token，再由 Target 一次 Causal Verify 并提交连续接受前缀；它不等于训练 MTP，也不保证加速。AIOS-IME 应先用端到端 Profile 决定 CUDA Graph、量化或 Speculative 的优先级，并通过完整 Top-3 与质量门禁晋级。

## 16. 接受算法的一个简化代码示意

严格 Speculative Sampling 会根据 Draft 与 Target 概率比决定接受，并在拒绝位置从修正分布采样。为理解控制流，可先看简化 Greedy Verify：

```python
def greedy_verify(draft_tokens, target_argmax_tokens):
    accepted = 0
    for draft, target in zip(draft_tokens, target_argmax_tokens):
        if draft != target:
            break
        accepted += 1
    return accepted
```

例如：

```text
Draft  = [A,B,C,D]
Target = [A,B,X,...]
accepted = 2
```

提交 A/B，C/D 临时 Page 释放，再提交 Target 的 X。

随机采样下不能只比较 Argmax，否则会改变 Target 原始分布；需要使用正式概率接受/修正算法。这也是为什么工程不能把“Draft 一致就提交”随意用于带 Temperature 的候选生成。

## 17. 与输入法多候选的特殊冲突

普通 Speculative 关注单条输出分布；AIOS-IME 同时要求 8 条候选多样性。若 Draft 太强地把所有分支推向相同高概率 Token：

```text
单分支接受率提高
但 Top-3 互异率可能下降
```

因此需要同时测：

```text
accepted tokens / verify
candidate diversity
raw invalid rate
refill rate
完整 Top-3 p95
```

不能只以接受率晋级。
