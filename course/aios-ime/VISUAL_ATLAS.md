# AIOS-IME 机制图谱

> 课程版本：`v5.1`
>
> 源码固定点：`9f53740753de36899aa7694cf7dcb5304e58ea54`
>
> 用法：先看一张图回答图下的问题，再进入对应课程读代码。图谱负责建立关系，不替代 README 中的源码、公式和验收。

## 图 1：输入法不是缩短版聊天生成

```mermaid
flowchart LR
    subgraph General["通用生成"]
      G1[多个独立 Request] --> G2[Scheduler]
      G2 --> G3[分别 Prefill / Decode]
      G3 --> G4[分别返回]
    end

    subgraph IME["AIOS-IME"]
      I1[一个当前按键] --> I2[一个 CandidateGroup]
      I2 --> I3[共享 Prefix KV<br/>独立随机流]
      I3 --> I4[共同过滤 / 去重 / MMR]
      I4 --> I5[一次交付完整 Top-3]
    end
```

看图时回答：

```text
为什么“同时生成八条”仍不足以定义 CandidateGroup？
```

关键是八条分支还共享 generation、取消边界、候选池与最终交付。

对应：P01、M01。

## 图 2：逻辑八行，物理一份 Prefix KV

```mermaid
flowchart TB
    P1[Physical Page 10<br/>Prefix token 1]
    P2[Physical Page 11<br/>Prefix token 2]
    P3[Physical Page 12<br/>Prefix token 3]

    R0[row 0: 10 11 12 + suffix A]
    R1[row 1: 10 11 12 + suffix B]
    R2[row 2: 10 11 12 + suffix C]
    R3[row ...: 10 11 12 + suffix ...]

    P1 --> R0
    P1 --> R1
    P1 --> R2
    P1 --> R3
    P2 --> R0
    P2 --> R1
    P2 --> R2
    P2 --> R3
    P3 --> R0
    P3 --> R1
    P3 --> R2
    P3 --> R3
```

看图时回答：

```text
Page Table 里 Prefix Page ID 出现八次，为什么显存里仍只有一份 K/V？
```

Page Table 复制的是物理 Page ID，不是 K/V Tensor。

对应：P02、M03。

## 图 3：首 token 的真实分叉点

```mermaid
flowchart LR
    A[Prefix tokens] --> B[Prefill 一次]
    B --> C[prefix_logits 1×V]
    C --> D[expand 8×V]
    D --> S0[seed 0]
    D --> S1[seed 1]
    D --> S2[seed 2]
    D --> S7[seed 7]
    S0 --> T0[first token]
    S1 --> T1[first token]
    S2 --> T2[first token]
    S7 --> T7[first token]
    T0 --> K0[需要预测第二 token 时<br/>才分配 suffix Page]
    T1 --> K1[独立 suffix Page]
    T2 --> K2[独立 suffix Page]
    T7 --> K7[独立 suffix Page]
```

看图时回答：

```text
为什么首 token 若立即 EOS，可以不分配任何 suffix Page？
```

首 token 直接从 Prefix 最后位置 Logits 采样；只有继续预测下一 token 时才需要保存它的 K/V。

对应：M03。

## 图 4：Ragged Row 收缩，但候选身份不漂移

```mermaid
flowchart TB
    S0["step 0<br/>dense rows: [c0,c1,c2,c3]<br/>active_local: [0,1,2,3]"]
    S1["c0 结束"]
    S2["step 1<br/>dense rows: [c1,c2,c3]<br/>active_local: [1,2,3]"]
    S3["c2 结束"]
    S4["step 2<br/>dense rows: [c1,c3]<br/>active_local: [1,3]"]

    S0 --> S1 --> S2 --> S3 --> S4
```

同时保留：

```mermaid
flowchart LR
    C[candidate identity] --> R[page_table row]
    C --> U[uniform candidate, token_step]
    C --> O[output write position]
    C --> L[logprob accumulator]
```

看图时回答：

```text
如果 RNG 绑定当前 Dense Row，c1 在 c0 结束后会发生什么？
```

c1 会拿到本属于另一行的随机数，优化改变候选文本。

对应：P02、M04。

## 图 5：自适应补采样是反馈控制

```mermaid
flowchart TD
    A[当前 Raw Candidate Pool] --> B[Hard Filter]
    B --> C[Display-key Dedup]
    C --> D[unique-valid count]
    D --> E[deficit = 3 - unique-valid]
    E --> F{deficit <= 0?}
    F -- yes --> G[filled]
    F -- no --> H[observed yield]
    H --> I[ceil deficit / yield]
    I --> J[加 diversity headroom]
    J --> K[裁到 2~8 路<br/>剩余总预算<br/>deadline]
    K --> L[新 seed / 更高探索 / 旧序列规避]
    L --> A
```

看图时回答：

```text
为什么不能用 raw valid count 规划 refill？
```

最终缺口发生在显示去重之后；合法但显示重复的候选不能填满 Top-3。

对应：P02、M05。

## 图 6：token-LCP 决定 Page 去留

```mermaid
flowchart LR
    O[旧 Token IDs<br/>1 2 3 4] --> L[token-LCP = 2]
    N[新 Token IDs<br/>1 2 8 9] --> L
    L --> K[保留 Page 1 2]
    L --> F[释放旧 Page 3 4]
    L --> A[为 Token 8 9 分配新 Page]
    A --> P[positions 从 2 开始增量 Prefill]
```

四种变化：

```mermaid
stateDiagram-v2
    [*] --> Same: Token IDs 完全相同
    [*] --> Append: 只追加 Token
    [*] --> Retokenize: 尾部重切或改写
    [*] --> Backspace: 严格回退

    Same --> ReuseAll: KV + 末位 Logits 全复用
    Append --> Extend: 保留 LCP + 新尾 Prefill
    Retokenize --> RebuildTail: 释放旧尾 + 重算新尾
    Backspace --> ExactRefill: 当前实现重新 Prefill
```

看图时回答：

```text
为什么严格 Backspace 不能只截断 Page Table？
```

旧 Cache 保存 K/V 与旧完整 Prefix 的末位 Logits，没有保存任意历史位置重新成为末位时的最终 Logits。

对应：P03、M06。

## 图 7：latest-wins 的三个层次

```mermaid
sequenceDiagram
    participant U as 新按键
    participant G as generation lock
    participant O as 旧 CandidateGroup
    participant R as run lock
    participant P as Page Pool
    participant N as 新 CandidateGroup

    U->>G: new_generation()
    G->>O: cancel token
    G->>G: generation_id += 1
    Note over O: 旧结果立即失去交付资格
    O->>O: 当前 CUDA step 完成
    O->>O: step 边界检查 cancelled
    O->>P: finally 释放 suffix Pages
    O-->>R: 退出 Runtime
    N->>R: 获得独占
    N->>P: token-LCP 决定 Prefix Page 去留
```

三个层次：

```text
交付资格：旧结果不能显示
计算止损：旧组不再发射下一 token step
资源终结：临时 suffix Page 在 finally 归还
```

对应：P03、M07。

## 图 8：候选治理不能混淆三种分数

```mermaid
flowchart LR
    L[Original logits] --> R[Raw token logprob]
    L --> S[Sampling policy<br/>temperature / top-k / top-p / stop mask]
    S --> T[Sampled token sequence]
    T --> D[Decode + terminal trim]
    R --> B[Average raw logprob]
    D --> H[Hard invalid reasons]
    D --> P[Soft penalty]
    B --> Q[Base score]
    H --> Q
    P --> Q
    Q --> K[Display-key dedup]
    K --> M[MMR: quality - similarity]
    M --> O[Stable Top-3]
```

看图时回答：

```text
为什么 refill 每轮温度不同，候选仍能在同一池里排序？
```

排序保存修改前的 raw model logprob；Temperature 只改变探索分布。

对应：P06、M08。

## 图 9：Block AttnRes 的 Bank 与 Partial

```mermaid
flowchart TB
    E[Embedding source] --> B0[Bank S=1]
    B0 --> MX1[Mix before Attention]
    MX1 --> A[Attention delta]
    A --> P1[partial]
    B0 --> MX2[Mix bank + partial before MLP]
    P1 --> MX2
    MX2 --> M[MLP delta]
    M --> P2[updated partial]
    P2 --> END[Block end]
    END --> B1[Bank append one block delta]
    B1 --> NEXT[Next block]
    NEXT --> FINAL[Final Mixer]
```

调用次数：

```mermaid
flowchart LR
    B[8 Blocks] --> L[每 Block 4 Layers]
    L --> X[每 Layer 2 Mixers]
    X --> N[64 Mixers]
    N --> F[Final Mixer]
    F --> T[总计 65]
```

看图时回答：

```text
为什么不能把 65 次 Mixer 合成一次？
```

每次后续 Mixer 的输入依赖刚产生的 Attention 或 MLP delta，存在真实数据依赖。

对应：P04、P05、M09。

## 图 10：两段 Triton Kernel 与证据回路

```mermaid
flowchart LR
    subgraph Kernel["AttnRes Hot Path"]
      A[bank S×N×D<br/>partial N×D<br/>weighted query D]
      A --> SK[Score Kernel<br/>沿 D 归约]
      SK --> SC[scores S_total×N FP32]
      SC --> VK[Value Kernel<br/>沿 S Softmax / 聚合]
      A --> VK
      VK --> O[mixed N×D model dtype]
    end

    subgraph Evidence["发布证据"]
      E1[logits diff / cosine]
      E2[Top-k token set]
      E3[固定 seed 候选文本]
      E4[65-Mixer latency / memory]
      E5[完整 Top-3 p50 / p95]
      E6[结构与人工语义]
    end

    O --> E1
    O --> E2
    O --> E3
    O --> E4
    O --> E5
    O --> E6
```

看图时回答：

```text
为什么“Kernel 快且 max diff 小”仍不足以上线？
```

离散 token、候选排序、动态 active-row Shape、Page 生命周期和人工语义仍可能退化。

对应：P05、P06、M10、M11。

## 一张总图：从按键到证据

```mermaid
flowchart TD
    A[中文 Prefix] --> B[token-LCP Prefix State]
    B --> C[CandidateGroup]
    C --> D[Ragged Decode + Stateless RNG]
    D --> E[Raw Candidate Pool]
    E --> F[Filter / Dedup / MMR]
    F --> G{Top-3?}
    G -- no --> H[Adaptive Refill]
    H --> D
    G -- yes --> I{Latest generation?}
    I -- no --> J[取消交付 / 回收]
    I -- yes --> K[Top-3]
    K --> L[完整性能 Lane]
    K --> M[结构质量 Lane]
    K --> N[固定排序 Lane]
    K --> O[人工语义 Lane]
    C --> P[Page / Cancellation Lane]
    L --> Q[发布判定]
    M --> Q
    N --> Q
    O --> Q
    P --> Q
```

能独立解释这张总图，并指出每个节点的 canonical source owner，才算真正建立了项目地图。
