#!/usr/bin/env python3
"""Benchmark AIOS-IME CandidateGroup latency, quality shape and GPU memory."""

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


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def summary(values: list[float]) -> dict[str, float]:
    return {
        "p50": statistics.median(values),
        "p95": percentile(values, 0.95),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
    }


def read_eval_cases(path: Path) -> list[tuple[str, list[str]]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    seen: set[str] = set()
    cases: list[tuple[str, list[str]]] = []
    for row in rows:
        prefix = str(row.get("model_prefix") or row.get("prefix") or "").strip()
        if prefix and prefix not in seen:
            seen.add(prefix)
            references = row.get("reference_completions")
            if not isinstance(references, list) or not references:
                single = row.get("reference_completion") or row.get("completion")
                references = [str(single)] if single else []
            cases.append((prefix, [str(item) for item in references if item]))
    return cases


def longest_common_prefix_length(left: str, right: str) -> int:
    length = 0
    for left_char, right_char in zip(left, right):
        if left_char != right_char:
            break
        length += 1
    return length


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--eval-data",
        type=Path,
        default=Path("/home/codex/ai/projects/minimind/dataset/ime_eval_v2_frozen_20260814/all_200.jsonl"),
    )
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--display-candidates", type=int, default=3)
    parser.add_argument("--sampling-attempts", type=int, default=8)
    parser.add_argument("--max-sampling-attempts", type=int, default=24)
    parser.add_argument("--refill-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--max-candidate-chars", type=int, default=32)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--refill-top-p", type=float, default=0.95)
    parser.add_argument("--refill-temperature", type=float, default=0.75)
    parser.add_argument("--refill-temperature-step", type=float, default=0.10)
    parser.add_argument("--max-refill-temperature", type=float, default=0.95)
    parser.add_argument("--refill-top-k", type=int, default=96)
    parser.add_argument("--refill-top-k-step", type=int, default=32)
    parser.add_argument("--max-refill-top-k", type=int, default=160)
    parser.add_argument("--kv-cache-max-tokens", type=int, default=256)
    parser.add_argument("--attention-workspace-mib", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--attnres-backend",
        choices=("reference", "eager", "compiled", "triton"),
        default="triton",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "reports/aios_ime_benchmark_20260814.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=PROJECT_ROOT / "reports/aios_ime_benchmark_20260814.md",
    )
    return parser.parse_args()


def render_markdown(report: dict[str, Any]) -> str:
    latency = report["latency_ms"]
    memory = report["memory_mib"]
    quality = report["candidate_shape"]
    throughput = report["throughput"]
    reference_quality = report["reference_quality"]
    return f"""# AIOS-IME 单用户 CandidateGroup 基准

- GPU：`{report['gpu']}`；模型：`{report['architecture_revision']}` BF16；AttnRes backend：`{report['protocol']['attnres_backend']}`；显示 Top-3，内部先并行 `{report['protocol']['sampling_attempts']}` 路，有效候选不足时最多补到 `{report['protocol']['max_sampling_attempts']}` 路。
- 预热：`{report['protocol']['warmup']}`；计时样本：`{report['protocol']['samples']}`；不含首次 JIT/wrapper 规划。
- Top-3 p50/p95：`{latency['p50']:.2f}/{latency['p95']:.2f} ms`。
- 实际模型工作量：`{throughput['active_model_tokens_per_second']:.2f} active tokens/s`（只计 Prefix 新算 token 与 Decode 活跃 row）。
- 模型加载后/基准峰值 allocated：`{memory['after_model_load_allocated']:.2f}/{memory['benchmark_peak_allocated']:.2f} MiB`。
- 返回满三条比例：`{quality['full_top3_rate']:.2%}`；三条互异比例：`{quality['all_distinct_rate']:.2%}`。
- 冻结参考诊断：Top-1/Top-3 exact `{reference_quality['top1_exact']:.2%}/{reference_quality['top3_exact']:.2%}`，Top-1/Top-3 首字方向 `{reference_quality['top1_first_character']:.2%}/{reference_quality['top3_first_character']:.2%}`，mean best LCP `{reference_quality['mean_top1_lcp']:.2f}/{reference_quality['mean_best_top3_lcp']:.2f}`。

计时口径包含一次 Prefix Prefill、同组分支共享 Prefix KV 的 decode、GPU 原始 logprob、CPU
统一解码、过滤、去重和 MMR Top-3；不包含模型加载和首次 JIT 编译。
"""


def main() -> None:
    args = parse_args()
    cases = read_eval_cases(args.eval_data)
    prefixes = [prefix for prefix, _ in cases]
    required = args.warmup + args.samples
    if len(prefixes) < required:
        raise ValueError(f"Need {required} unique prefixes, found {len(prefixes)}")

    torch.cuda.empty_cache()
    llm = LLM(
        str(args.model),
        attnres_backend=args.attnres_backend,
        kv_cache_max_tokens=args.kv_cache_max_tokens,
        attention_workspace_size=args.attention_workspace_mib * 2**20,
    )
    engine = ImeCompletionEngine(llm)
    torch.cuda.synchronize()
    load_allocated = torch.cuda.memory_allocated() / 2**20
    load_reserved = torch.cuda.memory_reserved() / 2**20
    config = ImeGenerationConfig(
        display_candidates=args.display_candidates,
        sampling_attempts=args.sampling_attempts,
        max_sampling_attempts=args.max_sampling_attempts,
        refill_batch_size=args.refill_batch_size,
        max_new_tokens=args.max_new_tokens,
        max_candidate_chars=args.max_candidate_chars,
        top_p=args.top_p,
        refill_top_p=args.refill_top_p,
        refill_temperature=args.refill_temperature,
        refill_temperature_step=args.refill_temperature_step,
        max_refill_temperature=args.max_refill_temperature,
        refill_top_k=args.refill_top_k,
        refill_top_k_step=args.refill_top_k_step,
        max_refill_top_k=args.max_refill_top_k,
        seed=args.seed,
    )

    for index, prefix in enumerate(prefixes[: args.warmup]):
        engine.complete(prefix, ImeGenerationConfig(**{
            **config.__dict__, "seed": args.seed + index * 100_003
        }))
    torch.cuda.reset_peak_memory_stats()

    latencies: list[float] = []
    gpu_latencies: list[float] = []
    rows: list[dict[str, Any]] = []
    full_top3 = 0
    all_distinct = 0
    invalid_raw = 0
    total_attempts = 0
    total_active_model_tokens = 0
    total_generated_tokens = 0
    total_refill_rounds = 0
    top1_exact = 0
    top3_exact = 0
    top1_first_character = 0
    top3_first_character = 0
    top1_lcp_sum = 0
    top3_lcp_sum = 0
    for index, prefix in enumerate(prefixes[args.warmup : required]):
        run_config = ImeGenerationConfig(**{
            **config.__dict__, "seed": args.seed + (args.warmup + index) * 100_003
        })
        result = engine.complete(prefix, run_config)
        texts = [candidate.text for candidate in result.candidates]
        references = cases[args.warmup + index][1]
        first_characters = {item[0] for item in references if item}
        top1 = texts[:1]
        top1_exact += int(any(text in references for text in top1))
        top3_exact += int(any(text in references for text in texts))
        top1_first_character += int(
            any(text and text[0] in first_characters for text in top1)
        )
        top3_first_character += int(
            any(text and text[0] in first_characters for text in texts)
        )
        top1_lcp_sum += max(
            (
                longest_common_prefix_length(text, reference)
                for text in top1
                for reference in references
            ),
            default=0,
        )
        top3_lcp_sum += max(
            (
                longest_common_prefix_length(text, reference)
                for text in texts
                for reference in references
            ),
            default=0,
        )
        latencies.append(result.latency_ms)
        gpu_latencies.append(result.gpu_latency_ms)
        full_top3 += int(len(texts) == args.display_candidates)
        all_distinct += int(len(texts) == args.display_candidates and len(set(texts)) == len(texts))
        invalid_raw += sum(bool(candidate.invalid_reasons) for candidate in result.raw_candidates)
        total_attempts += result.sampling_attempts
        total_active_model_tokens += result.active_model_tokens
        total_generated_tokens += result.generated_tokens
        total_refill_rounds += result.refill_rounds
        rows.append({
            "prefix": prefix,
            "candidates": texts,
            "latency_ms": result.latency_ms,
            "raw_invalid": sum(bool(candidate.invalid_reasons) for candidate in result.raw_candidates),
            "sampling_attempts": result.sampling_attempts,
            "active_model_tokens": result.active_model_tokens,
            "refill_rounds": result.refill_rounds,
            "refill_stop_reason": result.refill_stop_reason,
            "valid_unique_candidates": result.valid_unique_candidates,
            "invalid_candidates": result.invalid_candidates,
            "duplicate_candidates": result.duplicate_candidates,
            "reused_prefix_tokens": result.reused_prefix_tokens,
        })

    report = {
        "schema_version": "aios.ime_benchmark.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model.resolve()),
        "gpu": torch.cuda.get_device_name(),
        "architecture_revision": llm.config.architecture_revision,
        "protocol": {
            "eval_data": str(args.eval_data.resolve()),
            "warmup": args.warmup,
            "samples": args.samples,
            "display_candidates": args.display_candidates,
            "sampling_attempts": args.sampling_attempts,
            "max_sampling_attempts": args.max_sampling_attempts,
            "refill_batch_size": args.refill_batch_size,
            "max_new_tokens": args.max_new_tokens,
            "max_candidate_chars": args.max_candidate_chars,
            "top_p": args.top_p,
            "refill_top_p": args.refill_top_p,
            "refill_temperature": args.refill_temperature,
            "refill_temperature_step": args.refill_temperature_step,
            "max_refill_temperature": args.max_refill_temperature,
            "refill_top_k": args.refill_top_k,
            "refill_top_k_step": args.refill_top_k_step,
            "max_refill_top_k": args.max_refill_top_k,
            "kv_cache_max_tokens": args.kv_cache_max_tokens,
            "attention_workspace_mib": args.attention_workspace_mib,
            "attnres_backend": args.attnres_backend,
        },
        "latency_ms": summary(latencies),
        "gpu_latency_ms": summary(gpu_latencies),
        "throughput": {
            "active_model_tokens": total_active_model_tokens,
            "generated_tokens": total_generated_tokens,
            "active_model_tokens_per_second": (
                total_active_model_tokens / (sum(gpu_latencies) / 1000.0)
            ),
        },
        "memory_mib": {
            "after_model_load_allocated": load_allocated,
            "after_model_load_reserved": load_reserved,
            "benchmark_peak_allocated": torch.cuda.max_memory_allocated() / 2**20,
            "benchmark_peak_reserved": torch.cuda.max_memory_reserved() / 2**20,
        },
        "candidate_shape": {
            "full_top3_rate": full_top3 / args.samples,
            "all_distinct_rate": all_distinct / args.samples,
            "raw_invalid_rate": invalid_raw / total_attempts,
            "mean_sampling_attempts": total_attempts / args.samples,
            "mean_refill_rounds": total_refill_rounds / args.samples,
        },
        "reference_quality": {
            "top1_exact": top1_exact / args.samples,
            "top3_exact": top3_exact / args.samples,
            "top1_first_character": top1_first_character / args.samples,
            "top3_first_character": top3_first_character / args.samples,
            "mean_top1_lcp": top1_lcp_sum / args.samples,
            "mean_best_top3_lcp": top3_lcp_sum / args.samples,
        },
        "results": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
