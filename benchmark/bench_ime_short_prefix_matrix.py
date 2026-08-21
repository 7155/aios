#!/usr/bin/env python3
"""Benchmark the five BF16 IME comparison profiles on one frozen lane.

Each profile is loaded into slot A by itself.  The service is unloaded between
models, so the 4B comparison cannot inherit memory pressure from a previous
worker.  Model loading and warmup are outside the measured request latency.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench_ime_short_prefix_compare import (
    DEFAULT_EXAMPLE_PREFIXES,
    PROJECT_ROOT,
    post_json,
    read_json,
    summarize_slot,
)


DEFAULT_PROFILES = (
    "MiniMind-IME 0.06B",
    "MiniMind-IME 0.1B 极速版",
    "0.214B AttnRes 质量版",
    "Qwen3-0.6B",
    "Qwen3-4B",
)


def compare_model_once(
    server_url: str,
    config: dict[str, Any],
    prefix: str,
    *,
    seed: int,
    max_sampling_attempts: int,
) -> dict[str, Any]:
    generation = {
        **config["generation"],
        "sampling_attempts": 8,
        "max_sampling_attempts": max_sampling_attempts,
        "max_new_tokens": 12,
        "seed": seed,
    }
    return post_json(
        f"{server_url}/api/compare",
        {
            "prefix": prefix,
            "slots": config["slots"],
            "generation": generation,
            "targets": ["a"],
            "order": "a_then_b",
            "reset_prefix_cache": True,
        },
    )


def run_model_protocol(
    server_url: str,
    config: dict[str, Any],
    cases: list[dict[str, Any]],
    *,
    label: str,
    name: str,
    seed: int,
    max_sampling_attempts: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        response = compare_model_once(
            server_url,
            config,
            str(case["prefix"]),
            seed=seed + index * 100_003,
            max_sampling_attempts=max_sampling_attempts,
        )
        result = response["results"].get("a", {})
        if not result.get("ok"):
            raise RuntimeError(
                f"{label} failed for prefix {case['prefix']!r}: "
                f"{result.get('error', 'unknown error')}"
            )
        rows.append({"case": case, "response": response})
        print(
            f"[{label} | {name} {index + 1:02d}/{len(cases)}] "
            f"{case['prefix']}",
            flush=True,
        )
    return {
        "name": name,
        "sampling_attempts": 8,
        "max_sampling_attempts": max_sampling_attempts,
        "summary": summarize_slot(rows, "a"),
        "results": rows,
    }


def format_parameters(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    return f"{value / 1_000_000:.2f}M"


def model_shape(summary: dict[str, Any]) -> str:
    model = summary["model"]
    residual = (
        "Block AttnRes"
        if model["residual_type"] == "block_attnres"
        else "Standard"
    )
    return f"{model['layers']}L · {residual}"


def render_table(models: list[dict[str, Any]], protocol: str) -> str:
    lines = [
        "| 模型 | 参数量 | 架构 | Top-3 p50 / p95 | 满三条且互异 | 契约违规候选 | 受影响前缀 | 平均采样路数 | 峰值显存 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_result in models:
        item = model_result["protocols"][protocol]["summary"]
        model = item["model"]
        lines.append(
            f"| {item['label']} | {format_parameters(model['parameter_count'])} | "
            f"{model_shape(item)} | {item['latency_ms']['p50']:.2f} / "
            f"{item['latency_ms']['p95']:.2f} ms | "
            f"{item['all_distinct_rate']:.2%} | "
            f"{item['contract_violation_candidates']}/{item['shown_candidates']} "
            f"({item['contract_violation_rate']:.2%}) | "
            f"{item['affected_prefixes']}/{item['rows']} | "
            f"{item['mean_sampling_attempts']:.2f} | "
            f"{item['peak_allocated_mib']:.2f} MiB |"
        )
    return "\n".join(lines)


def candidate_texts(protocol: dict[str, Any], prefix: str) -> list[str]:
    for row in protocol["results"]:
        if row["case"]["prefix"] == prefix:
            result = row["response"]["results"]["a"]["result"]
            return [str(candidate["text"]) for candidate in result["candidates"]]
    return []


def render_markdown(report: dict[str, Any]) -> str:
    models = report["models"]
    adaptive_table = render_table(models, "adaptive24")
    fixed_table = render_table(models, "fixed8")
    speed = next(
        model for model in models if model["profile_label"] == "MiniMind-IME 0.1B 极速版"
    )["protocols"]["adaptive24"]["summary"]
    examples: list[str] = []
    for prefix in report["example_prefixes"]:
        examples.extend((f"### `{prefix}`", "", "```text"))
        for model_result in models:
            texts = candidate_texts(model_result["protocols"]["adaptive24"], prefix)
            examples.append(f"{model_result['profile_label']}：")
            for index, text in enumerate(texts, 1):
                examples.append(f"  {index}. {prefix}{text}")
        examples.extend(("```", ""))

    backend_lines = []
    for model_result in models:
        summary = model_result["protocols"]["adaptive24"]["summary"]
        backend_lines.append(
            f"- `{summary['label']}`：`{summary['model']['attention_backend']}`"
        )

    return f"""# AIOS-IME 五模型短前缀统一对比

## 结果总览

以下五个模型全部使用 BF16、同一张 GPU、同一批 {report['rows']} 条冻结短前缀、同一 seed
和同一 CandidateGroup 合同。表中延迟是完整 Top-3 墙钟时间，包含 Prefill、多路 Decode、
过滤、去重、排序与必要的补采样，不包含模型加载和一次性预热。

### 默认自适应 Top-3（首轮 8 路，最多 24 路）

{adaptive_table}

0.1B 极速版在该合同下的 p50/p95 为
`{speed['latency_ms']['p50']:.2f}/{speed['latency_ms']['p95']:.2f} ms`。参数更小不自动等于
候选更好，参数更大也不自动适配裸中文前缀；“契约违规候选”只统计可确定复现的失败，
包括冻结禁词、元文本/指令文本、重度拉丁字符以及日文或韩文脚本，不等同于完整人工自然度。

### 固定 8 路

{fixed_table}

固定 8 路用于分离模型本身的候选有效率；自适应表更接近产品实际，因为候选不足时会在
24 路总预算内补采样。两个表必须一起看，不能用补采样后的“满三条”掩盖额外延迟。

## 评测合同

- 数据：`{report['eval_data']}` 的全部 `category=short_prefix` 行，不挑样例。
- 输入：`[BOS] + 裸中文前缀`，不套聊天模板。
- 生成：Top-3，首轮 8 路，最多 12 个新 token，固定 seed；自适应上限 24 路。
- 执行：模型逐个加载、逐个预热、串行测量；每个模型结束后卸载并释放显存。
- 显存：单次请求期间 `torch.cuda.max_memory_allocated`，不是整机显存占用。

### 实际注意力后端

{chr(10).join(backend_lines)}

0.06B 的旧模型使用 `head_dim=96`。FlashInfer 0.5.x 强制 FA2 paged-prefill 会产生 NaN，
因此 AIOS 对该尺寸使用 PyTorch SDPA Prefill，并继续使用 FlashInfer paged Decode；其余已支持
尺寸保留 FA2 快路径。这个后端差异在表中如实保留，没有把空输出计成模型质量。

## 同前缀输出

{chr(10).join(examples)}
## 复现

启动本地对比前端后执行：

```bash
python benchmark/bench_ime_short_prefix_matrix.py \\
  --server-url http://127.0.0.1:7860
```
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:7860")
    parser.add_argument(
        "--eval-data",
        type=Path,
        default=Path(
            "/home/codex/ai/projects/minimind/dataset/ime_eval_daily_200.jsonl"
        ),
    )
    parser.add_argument("--category", default="short_prefix")
    parser.add_argument(
        "--profile",
        action="append",
        dest="profiles",
        help="Profile label from /api/config; repeat to override the default five",
    )
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--warmup-prefix", default="测试")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "reports/aios_ime_short_prefix_matrix_20260821.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=PROJECT_ROOT / "reports/aios_ime_short_prefix_matrix_20260821.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server_url = args.server_url.rstrip("/")
    config = read_json(f"{server_url}/api/config")
    profiles_by_label = {
        str(profile["label"]): profile for profile in config.get("profiles", [])
    }
    profile_labels = tuple(args.profiles or DEFAULT_PROFILES)
    missing = [label for label in profile_labels if label not in profiles_by_label]
    if missing:
        raise ValueError(
            f"Profiles not found: {missing}; available: {sorted(profiles_by_label)}"
        )

    rows = [
        json.loads(line)
        for line in args.eval_data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cases = [row for row in rows if row.get("category") == args.category]
    if not cases:
        raise ValueError(f"No {args.category!r} rows found in {args.eval_data}")

    model_results: list[dict[str, Any]] = []
    try:
        for model_index, label in enumerate(profile_labels):
            post_json(f"{server_url}/api/unload", {})
            model_config = {
                **config,
                "slots": {
                    **config["slots"],
                    "a": profiles_by_label[label],
                },
            }
            print(f"\nLoading and warming {label}...", flush=True)
            warmup = compare_model_once(
                server_url,
                model_config,
                args.warmup_prefix,
                seed=args.seed - 1,
                max_sampling_attempts=8,
            )
            if not warmup["results"].get("a", {}).get("ok"):
                raise RuntimeError(
                    f"{label} warmup failed: "
                    f"{warmup['results'].get('a', {}).get('error')}"
                )
            protocols = {
                "fixed8": run_model_protocol(
                    server_url,
                    model_config,
                    cases,
                    label=label,
                    name="fixed8",
                    seed=args.seed,
                    max_sampling_attempts=8,
                ),
                "adaptive24": run_model_protocol(
                    server_url,
                    model_config,
                    cases,
                    label=label,
                    name="adaptive24",
                    seed=args.seed,
                    max_sampling_attempts=24,
                ),
            }
            model_results.append(
                {
                    "profile_label": label,
                    "profile": profiles_by_label[label],
                    "protocols": protocols,
                    "model_order": model_index,
                }
            )
    finally:
        post_json(f"{server_url}/api/unload", {})

    report = {
        "schema_version": "aios.ime_short_prefix_matrix.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "server_url": server_url,
        "eval_data": str(args.eval_data.resolve()),
        "category": args.category,
        "rows": len(cases),
        "seed": args.seed,
        "dtype": "bfloat16",
        "example_prefixes": [
            prefix
            for prefix in DEFAULT_EXAMPLE_PREFIXES
            if any(case["prefix"] == prefix for case in cases)
        ],
        "models": model_results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    print("\n" + render_table(model_results, "adaptive24"), flush=True)


if __name__ == "__main__":
    main()
