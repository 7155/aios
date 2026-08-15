#!/usr/bin/env python3
"""A/B benchmark single-user incremental token-LCP prefix KV reuse."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from aios import ImeCompletionEngine, ImeGenerationConfig, LLM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEXT = "没关系，你先忙你的，等你忙完以后再给我发消息"


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * quantile) - 1)]


def stats(values: list[float]) -> dict[str, float]:
    return {
        "p50": statistics.median(values),
        "p95": percentile(values, 0.95),
        "mean": statistics.mean(values),
        "total": sum(values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--start-chars", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--kv-cache-max-tokens", type=int, default=256)
    parser.add_argument("--attention-workspace-mib", type=int, default=1)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "reports/aios_ime_prefix_reuse_20260814.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=PROJECT_ROOT / "reports/aios_ime_prefix_reuse_20260814.md",
    )
    return parser.parse_args()


def run_sequence(
    engine: ImeCompletionEngine,
    prefixes: list[str],
    config: ImeGenerationConfig,
    *,
    reset_each: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    engine.reset_prefix_cache()
    for index, prefix in enumerate(prefixes):
        if reset_each:
            engine.reset_prefix_cache()
        run_config = ImeGenerationConfig(**{
            **config.__dict__, "seed": config.seed + index * 100_003
        })
        result = engine.complete(prefix, run_config)
        rows.append({
            "prefix": prefix,
            "prefix_tokens": result.prefix_tokens,
            "reused_prefix_tokens": result.reused_prefix_tokens,
            "latency_ms": result.latency_ms,
        })
    return rows


def render_markdown(report: dict[str, Any]) -> str:
    cold = report["full_prefill_each_key"]
    reuse = report["incremental_prefix_reuse"]
    speedup = cold["total"] / reuse["total"]
    return f"""# AIOS-IME 跨按键 Prefix KV 复用 A/B

- 输入：从 `{report['protocol']['start_chars']}` 字开始，逐字增长到 `{report['protocol']['text_chars']}` 字；重复 `{report['protocol']['rounds']}` 轮。
- 每次完整 Prefill：p50/p95 `{cold['p50']:.2f}/{cold['p95']:.2f} ms`，累计 `{cold['total']:.2f} ms`。
- token-LCP 增量 Prefill：p50/p95 `{reuse['p50']:.2f}/{reuse['p95']:.2f} ms`，累计 `{reuse['total']:.2f} ms`。
- 整段按键序列累计加速：`{speedup:.2f}x`。

该微基准把生成限制为一个 token，用来隔离 Prefill；完整 Top-3 延迟还包含多路 decode、过滤和排序。
"""


def main() -> None:
    args = parse_args()
    prefixes = [args.text[:length] for length in range(args.start_chars, len(args.text) + 1)]
    llm = LLM(
        str(args.model),
        kv_cache_max_tokens=args.kv_cache_max_tokens,
        attention_workspace_size=args.attention_workspace_mib * 2**20,
    )
    engine = ImeCompletionEngine(llm)
    config = ImeGenerationConfig(
        display_candidates=3,
        sampling_attempts=8,
        max_sampling_attempts=8,
        max_new_tokens=1,
        min_new_tokens=1,
        seed=20260814,
    )

    # Compile wrappers and stabilize clocks before either side of the A/B.
    run_sequence(engine, prefixes[:3], config, reset_each=False)
    full_rows: list[dict[str, Any]] = []
    reuse_rows: list[dict[str, Any]] = []
    for _ in range(args.rounds):
        full_rows.extend(run_sequence(engine, prefixes, config, reset_each=True))
        reuse_rows.extend(run_sequence(engine, prefixes, config, reset_each=False))
    engine.reset_prefix_cache()

    full_stats = stats([row["latency_ms"] for row in full_rows])
    reuse_stats = stats([row["latency_ms"] for row in reuse_rows])
    report = {
        "schema_version": "aios.ime_prefix_reuse_benchmark.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model.resolve()),
        "gpu": torch.cuda.get_device_name(),
        "protocol": {
            "text": args.text,
            "text_chars": len(args.text),
            "start_chars": args.start_chars,
            "keystrokes_per_round": len(prefixes),
            "rounds": args.rounds,
            "generated_tokens_per_key": 1,
        },
        "full_prefill_each_key": full_stats,
        "incremental_prefix_reuse": {
            **reuse_stats,
            "mean_reused_tokens": statistics.mean(
                row["reused_prefix_tokens"] for row in reuse_rows
            ),
        },
        "sequence_total_speedup": full_stats["total"] / reuse_stats["total"],
        "sample_incremental_rows": reuse_rows[: len(prefixes)],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({
        key: value for key, value in report.items() if key != "sample_incremental_rows"
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
