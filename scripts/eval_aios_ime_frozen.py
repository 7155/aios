#!/usr/bin/env python3
"""Evaluate AIOS-IME on the immutable three-lane MiniMind-IME eval v2."""

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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "p50": statistics.median(values),
        "p95": percentile(values, 0.95),
        "mean": statistics.mean(values),
    }


def compact(text: str) -> str:
    return "".join(str(text or "").split()).strip("，。！？；：,.!?;:")


def lcp_chars(left: str, right: str) -> int:
    count = 0
    for left_char, right_char in zip(compact(left), compact(right)):
        if left_char != right_char:
            break
        count += 1
    return count


def pairwise_accuracy(scored: list[dict[str, Any]]) -> tuple[int, int]:
    correct = 0
    total = 0
    for left_index, left in enumerate(scored):
        for right in scored[left_index + 1 :]:
            if int(left["grade"]) == int(right["grade"]):
                continue
            total += 1
            correct += int(int(left["grade"]) > int(right["grade"]))
    return correct, total


def evaluate_ranking_lane(
    engine: ImeCompletionEngine,
    rows: list[dict[str, Any]],
    scoring_mode: str,
) -> dict[str, Any]:
    acceptable_top1 = 0
    best_top1 = 0
    pair_correct = 0
    pair_total = 0
    latencies: list[float] = []
    results: list[dict[str, Any]] = []
    for row in rows:
        # Frozen rows are independent examples. Resetting prevents an unrelated
        # previous row from changing the numerical prefill path.
        engine.reset_prefix_cache()
        metadata = {str(item["text"]): item for item in row["candidates"]}
        result = engine.score_candidates(
            str(row["model_prefix"]),
            [str(item["text"]) for item in row["candidates"]],
            mode=scoring_mode,
        )
        enriched = [
            {
                **metadata[item.text],
                "text": item.text,
                "average_log_probability": item.average_logprob,
                "sum_log_probability": item.sum_logprob,
                "token_count": item.token_count,
            }
            for item in result.candidates
        ]
        top = enriched[0]
        acceptable_top1 += int(bool(top["acceptable"]))
        best_top1 += int(int(top["grade"]) == 3)
        correct, total = pairwise_accuracy(enriched)
        pair_correct += correct
        pair_total += total
        latencies.append(result.latency_ms)
        results.append({
            "id": row["id"],
            "category": row["category"],
            "input_environment": row["input_environment"],
            "model_prefix": row["model_prefix"],
            "typed_pinyin": row.get("typed_pinyin"),
            "ranked_candidates": enriched,
            "top1_acceptable": bool(top["acceptable"]),
            "top1_best_grade": int(top["grade"]) == 3,
            "latency_ms": result.latency_ms,
        })
    return {
        "rows": len(rows),
        "acceptable_top1_rate": acceptable_top1 / max(1, len(rows)),
        "best_grade_top1_rate": best_top1 / max(1, len(rows)),
        "pairwise_grade_accuracy": pair_correct / max(1, pair_total),
        "pairwise_correct": pair_correct,
        "pairwise_total": pair_total,
        "latency_ms": latency_summary(latencies),
        "scoring_mode": scoring_mode,
        "results": results,
    }


def evaluate_generation_lane(
    engine: ImeCompletionEngine,
    rows: list[dict[str, Any]],
    config: ImeGenerationConfig,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    latencies: list[float] = []
    lcp_top1: list[int] = []
    lcp_top3: list[int] = []
    nonempty = 0
    full_top3 = 0
    for index, row in enumerate(rows):
        engine.reset_prefix_cache()
        run_config = ImeGenerationConfig(**{
            **config.__dict__,
            "seed": config.seed + index * 100_003,
        })
        result = engine.complete(str(row["model_prefix"]), run_config)
        rendered = [item.text for item in result.candidates]
        reference = str(row["reference_completion"])
        top1 = lcp_chars(rendered[0], reference) if rendered else 0
        best3 = max((lcp_chars(item, reference) for item in rendered), default=0)
        lcp_top1.append(top1)
        lcp_top3.append(best3)
        latencies.append(result.latency_ms)
        nonempty += int(bool(rendered))
        full_top3 += int(len(rendered) == config.display_candidates)
        results.append({
            "id": row["id"],
            "category": row["category"],
            "prefix": row["model_prefix"],
            "reference_completion": reference,
            "top_candidates": rendered,
            "top1_lcp_chars": top1,
            "best_top3_lcp_chars": best3,
            "human_top1_acceptable": None,
            "human_top3_has_acceptable": None,
            "human_notes": "",
            "latency_ms": result.latency_ms,
            "sampling_attempts": result.sampling_attempts,
        })
    return {
        "rows": len(rows),
        "nonempty_rate": nonempty / max(1, len(rows)),
        "full_top3_rate": full_top3 / max(1, len(rows)),
        "mean_top1_lcp_chars": statistics.mean(lcp_top1),
        "mean_best_top3_lcp_chars": statistics.mean(lcp_top3),
        "latency_ms": latency_summary(latencies),
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path(
            "/home/codex/ai/projects/minimind/dataset/ime_eval_v2_frozen_20260814"
        ),
    )
    parser.add_argument("--kv-cache-max-tokens", type=int, default=256)
    parser.add_argument("--attention-workspace-mib", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "reports/aios_ime_frozen_eval_v2_20260814.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=PROJECT_ROOT / "reports/aios_ime_frozen_eval_v2_20260814.md",
    )
    return parser.parse_args()


def render_markdown(report: dict[str, Any]) -> str:
    generation = report["posttrain_user_generation"]
    context = report["context_candidate_ranking"]
    pinyin = report["pinyin_candidate_ranking"]
    lines = [
        "# AIOS-IME Frozen Eval v2",
        "",
        "三条 lane 保持独立，不合并成一个容易误导的总分。上下文候选使用共享 Prefix KV 的 decode 打分；同拼音候选使用整序列 Prefill 稳定复评分；两者都按原始平均 token logprob 排序。",
        "",
        "15 条真实生成的 LCP 只作唯一参考诊断，不等于语义接受率；其人工 accept/reject 仍待标注。",
        "",
        "| Lane | Rows | Primary | Secondary | p95 |",
        "|---|---:|---:|---:|---:|",
        f"| Post-training generation | {generation['rows']} | Full Top-3 {generation['full_top3_rate']:.2%} | Best Top-3 LCP {generation['mean_best_top3_lcp_chars']:.3f} | {generation['latency_ms']['p95']:.2f} ms |",
        f"| Context ranking | {context['rows']} | Acceptable Top-1 {context['acceptable_top1_rate']:.2%} | Pairwise {context['pairwise_grade_accuracy']:.2%} | {context['latency_ms']['p95']:.2f} ms |",
        f"| Same-pinyin ranking | {pinyin['rows']} | Acceptable Top-1 {pinyin['acceptable_top1_rate']:.2%} | Pairwise {pinyin['pairwise_grade_accuracy']:.2%} | {pinyin['latency_ms']['p95']:.2f} ms |",
        "",
        "## 15 条真实后训练输入",
        "",
        "| Prefix | Reference | Top 1 | Top 2 | Top 3 |",
        "|---|---|---|---|---|",
    ]
    for row in generation["results"]:
        candidates = row["top_candidates"] + [""] * 3
        cells = [row["prefix"], row["reference_completion"], *candidates[:3]]
        lines.append(
            "| "
            + " | ".join(str(cell).replace("|", "\\|") for cell in cells)
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    llm = LLM(
        str(args.model),
        kv_cache_max_tokens=args.kv_cache_max_tokens,
        attention_workspace_size=args.attention_workspace_mib * 2**20,
    )
    engine = ImeCompletionEngine(llm)
    config = ImeGenerationConfig(seed=args.seed)

    # Compile both FlashInfer paths before measuring the frozen rows.
    engine.complete("这是预热输入", ImeGenerationConfig(**{
        **config.__dict__, "seed": args.seed - 1
    }))
    engine.score_candidates("明天", ["继续处理", "早点出发", "再确认一下"])

    generation_rows = read_jsonl(args.eval_dir / "posttrain_user_generation_15.jsonl")
    context_rows = read_jsonl(args.eval_dir / "context_candidate_ranking_145.jsonl")
    pinyin_rows = read_jsonl(args.eval_dir / "pinyin_candidate_ranking_40.jsonl")
    generation = evaluate_generation_lane(engine, generation_rows, config)
    context = evaluate_ranking_lane(engine, context_rows, "shared_decode")
    pinyin = evaluate_ranking_lane(engine, pinyin_rows, "stable")
    engine.reset_prefix_cache()

    report = {
        "schema_version": "aios.minimind_ime.frozen_eval_result.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model.resolve()),
        "eval_dir": str(args.eval_dir.resolve()),
        "gpu": torch.cuda.get_device_name(),
        "input_contract": "raw Chinese context + BOS; no chat template",
        "aggregate_policy": "lanes must remain separate",
        "ranking_score": "candidate raw average token log probability; context uses shared_decode, same-pinyin uses numerically stable full prefill",
        "posttrain_user_generation": generation,
        "context_candidate_ranking": context,
        "pinyin_candidate_ranking": pinyin,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "posttrain_user_generation": {
            key: value for key, value in generation.items() if key != "results"
        },
        "context_candidate_ranking": {
            key: value for key, value in context.items() if key != "results"
        },
        "pinyin_candidate_ranking": {
            key: value for key, value in pinyin.items() if key != "results"
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
