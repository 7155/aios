# 实战上手轨

这一轨先让系统工作，再让每个可观察问题推动下一步。

```mermaid
flowchart LR
    P01[跑出 Top-3] --> P02[候选池与补采样]
    P02 --> P03[连续输入]
    P03 --> P04[模型档位选择]
    P04 --> P05[AttnRes Profile]
    P05 --> P06[发布门禁]
```

| 课次 | 产物 | 新观察 | 对应源码课 |
|---|---|---|---|
| [P01](lessons/P01-first-top3/README.md) | 一次可复现 Top-3 JSON | `complete()` 不只返回文本 | M01 |
| [P02](lessons/P02-candidate-budget/README.md) | 3/8/8→24 对比 | 显示数量与探索数量不同 | M03～M05 |
| [P03](lessons/P03-continuous-typing/README.md) | 连续 Prefix trace | 新按键同时触发复用与失效 | M06～M07 |
| [P04](lessons/P04-model-matrix/README.md) | 模型选择记录 | 参数量不是唯一决策变量 | M02、M09、M11 |
| [P05](lessons/P05-profile-attnres/README.md) | 65-Mixer Profile | 微内核与端到端收益不同 | M09～M10 |
| [P06](lessons/P06-release-gate/README.md) | 发布证据清单 | “更快”不能越过质量与资源门禁 | M08、M11 |

每课都必须保留上一阶段的可运行产物。不要在 P02 为了展示新机制而让 P01 的最小路径失效。
