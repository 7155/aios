#!/usr/bin/env python3
"""Compare two loaded AIOS-IME models on the frozen short-prefix lane.

The benchmark talks to ``run_ime_compare_frontend.py`` so each model remains
isolated in its own CUDA process.  Both slots receive the same prefix, seed and
generation contract, and the parent service executes them serially on one GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
META_OR_INSTRUCTION_PATTERN = re.compile(
    r"question|answer|assistant|system|user|prompt|instruction|response|"
    r"publication|shipping|translation|json|markdown|author|date|name\d|"
    r"multiple|comparison|explanation|以下是|作为(?:一个)?AI|我来回答|"
    r"请用中文|请帮我|请根据|翻译成|摘要",
    re.IGNORECASE,
)
FOREIGN_SCRIPT_PATTERN = re.compile(r"[\u3040-\u30ff\uac00-\ud7af]")
DEFAULT_EXAMPLE_PREFIXES = ("先把", "回头", "不用急", "我觉得", "这个", "暂时")


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "p50": statistics.median(values),
        "p95": percentile(values, 0.95),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
    }


def read_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def latin_heavy(text: str) -> bool:
    latin = len(re.findall(r"[A-Za-z]", text))
    chinese = len(re.findall(r"[\u3400-\u9fff]", text))
    return latin >= 4 and latin > chinese / 2


def candidate_contract_flags(text: str, forbidden: list[str]) -> list[str]:
    """Return deterministic structural flags, not a semantic quality score."""
    folded = text.casefold()
    flags: list[str] = []
    if any(term.casefold() in folded for term in forbidden):
        flags.append("frozen_forbidden_term")
    if META_OR_INSTRUCTION_PATTERN.search(text):
        flags.append("meta_or_instruction_text")
    if latin_heavy(text):
        flags.append("latin_heavy")
    if FOREIGN_SCRIPT_PATTERN.search(text):
        flags.append("foreign_script")
    return flags


def compare_once(
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
            "targets": ["a", "b"],
            "order": "a_then_b",
            "reset_prefix_cache": True,
        },
    )


def summarize_slot(rows: list[dict[str, Any]], slot: str) -> dict[str, Any]:
    latencies: list[float] = []
    gpu_latencies: list[float] = []
    full_top3 = 0
    all_distinct = 0
    shown_candidates = 0
    raw_candidates = 0
    invalid_candidates = 0
    duplicate_candidates = 0
    sampling_attempts = 0
    refill_rounds = 0
    contract_violations = 0
    affected_prefixes = 0
    reason_counts = {
        "frozen_forbidden_term": 0,
        "meta_or_instruction_text": 0,
        "latin_heavy": 0,
        "foreign_script": 0,
    }
    audit_rows: list[dict[str, Any]] = []
    peak_allocated_mib = 0.0
    runtime: dict[str, Any] | None = None

    for row in rows:
        case = row["case"]
        result = row["response"]["results"][slot]["result"]
        candidates = result["candidates"]
        raw = result["raw_candidates"]
        runtime = result["runtime"]
        prefix_violated = False
        candidate_audit: list[dict[str, Any]] = []

        latencies.append(float(result["latency_ms"]))
        gpu_latencies.append(float(result["gpu_latency_ms"]))
        full_top3 += int(len(candidates) == 3)
        all_distinct += int(
            len(candidates) == 3
            and len({candidate["text"] for candidate in candidates}) == 3
        )
        shown_candidates += len(candidates)
        raw_candidates += len(raw)
        invalid_candidates += int(result["invalid_candidates"])
        duplicate_candidates += int(result["duplicate_candidates"])
        sampling_attempts += int(result["sampling_attempts"])
        refill_rounds += int(result["refill_rounds"])
        peak_allocated_mib = max(
            peak_allocated_mib,
            float(runtime["cuda_peak_allocated_mib"]),
        )

        forbidden = [str(term) for term in case.get("forbidden", [])]
        for candidate in candidates:
            text = str(candidate["text"])
            flags = candidate_contract_flags(text, forbidden)
            for flag in flags:
                reason_counts[flag] += 1
            if flags:
                contract_violations += 1
                prefix_violated = True
            candidate_audit.append({"text": text, "flags": flags})
        affected_prefixes += int(prefix_violated)
        audit_rows.append({"prefix": case["prefix"], "candidates": candidate_audit})

    if runtime is None:
        raise ValueError("No benchmark rows were summarized")
    row_count = len(rows)
    return {
        "label": runtime["label"],
        "model_path": runtime["model_path"],
        "model": runtime["model"],
        "rows": row_count,
        "latency_ms": distribution(latencies),
        "gpu_latency_ms": distribution(gpu_latencies),
        "peak_allocated_mib": peak_allocated_mib,
        "full_top3_rate": full_top3 / row_count,
        "all_distinct_rate": all_distinct / row_count,
        "shown_candidates": shown_candidates,
        "raw_candidates": raw_candidates,
        "raw_invalid_rate": invalid_candidates / sampling_attempts,
        "raw_duplicate_rate": duplicate_candidates / sampling_attempts,
        "mean_sampling_attempts": sampling_attempts / row_count,
        "mean_refill_rounds": refill_rounds / row_count,
        "contract_violation_candidates": contract_violations,
        "contract_violation_rate": contract_violations / shown_candidates,
        "affected_prefixes": affected_prefixes,
        "affected_prefix_rate": affected_prefixes / row_count,
        "contract_violation_reason_counts": reason_counts,
        "contract_audit": audit_rows,
    }


def run_protocol(
    server_url: str,
    config: dict[str, Any],
    cases: list[dict[str, Any]],
    *,
    name: str,
    seed: int,
    max_sampling_attempts: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        response = compare_once(
            server_url,
            config,
            str(case["prefix"]),
            seed=seed + index * 100_003,
            max_sampling_attempts=max_sampling_attempts,
        )
        if not all(response["results"].get(slot, {}).get("ok") for slot in ("a", "b")):
            raise RuntimeError(f"Comparison failed for prefix: {case['prefix']}")
        rows.append({"case": case, "response": response})
        print(f"[{name} {index + 1:02d}/{len(cases)}] {case['prefix']}", flush=True)

    slots = {slot: summarize_slot(rows, slot) for slot in ("a", "b")}
    left = slots["a"]
    right = slots["b"]
    comparison = {
        "p50_latency_ratio_b_over_a": (
            right["latency_ms"]["p50"] / left["latency_ms"]["p50"]
        ),
        "p95_latency_ratio_b_over_a": (
            right["latency_ms"]["p95"] / left["latency_ms"]["p95"]
        ),
        "peak_memory_reduction_a_vs_b": (
            1.0 - left["peak_allocated_mib"] / right["peak_allocated_mib"]
        ),
    }
    return {
        "name": name,
        "sampling_attempts": 8,
        "max_sampling_attempts": max_sampling_attempts,
        "slots": slots,
        "comparison": comparison,
        "results": rows,
    }


def example_candidates(
    protocol: dict[str, Any],
    prefix: str,
    slot: str,
) -> list[str]:
    for row in protocol["results"]:
        if row["case"]["prefix"] == prefix:
            result = row["response"]["results"][slot]["result"]
            return [str(candidate["text"]) for candidate in result["candidates"]]
    return []


def render_markdown(report: dict[str, Any]) -> str:
    fixed = report["protocols"]["fixed8"]
    adaptive = report["protocols"]["adaptive24"]
    label_a = adaptive["slots"]["a"]["label"]
    label_b = adaptive["slots"]["b"]["label"]
    row_count = report["rows"]

    def table(protocol: dict[str, Any]) -> str:
        lines = [
            "| 模型 | Top-3 p50 / p95 | 满三条且互异 | 契约违规候选 | 受影响前缀 | 平均采样路数 | 峰值 allocated |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for slot in ("a", "b"):
            item = protocol["slots"][slot]
            lines.append(
                f"| {item['label']} | {item['latency_ms']['p50']:.2f} / "
                f"{item['latency_ms']['p95']:.2f} ms | {item['all_distinct_rate']:.2%} | "
                f"{item['contract_violation_candidates']}/{item['shown_candidates']} "
                f"({item['contract_violation_rate']:.2%}) | "
                f"{item['affected_prefixes']}/{item['rows']} "
                f"({item['affected_prefix_rate']:.2%}) | "
                f"{item['mean_sampling_attempts']:.2f} | "
                f"{item['peak_allocated_mib']:.2f} MiB |"
            )
        return "\n".join(lines)

    examples: list[str] = []
    for prefix in report["example_prefixes"]:
        examples.extend([f"### `{prefix}`", "", "```text"])
        for slot in ("a", "b"):
            label = adaptive["slots"][slot]["label"]
            examples.append(f"{label}：")
            for index, text in enumerate(example_candidates(adaptive, prefix, slot), 1):
                examples.append(f"  {index}. {prefix}{text}")
        examples.extend(["```", ""])
    examples_text = "\n".join(examples)

    return f"""# AIOS-IME 短前缀批量对比

## 结论

在冻结的 {row_count} 条真实短前缀上，`{label_a}` 与 `{label_b}` 使用完全相同的输入和
候选合同。默认自适应候选合同下，前者的完整 Top-3 p50/p95 为
`{adaptive['slots']['a']['latency_ms']['p50']:.2f}/{adaptive['slots']['a']['latency_ms']['p95']:.2f} ms`，
后者为
`{adaptive['slots']['b']['latency_ms']['p50']:.2f}/{adaptive['slots']['b']['latency_ms']['p95']:.2f} ms`；
后者分别为前者的
`{adaptive['comparison']['p50_latency_ratio_b_over_a']:.2f}×/{adaptive['comparison']['p95_latency_ratio_b_over_a']:.2f}×`。

`{label_b}` 虽然经过补采样也能填满 Top-3，但 120 条显示候选中有
`{adaptive['slots']['b']['contract_violation_candidates']}` 条命中确定性输入法契约违规规则，
影响 `{adaptive['slots']['b']['affected_prefixes']}/{row_count}` 个短前缀；`{label_a}` 为
`{adaptive['slots']['a']['contract_violation_candidates']}/120`。

## 评测合同

- 数据：`{report['eval_data']}` 中全部 `category=short_prefix` 行，共 40 条，不挑样例。
- 硬件：单张 GPU；两个模型串行执行；均为 BF16。
- 输入：`[BOS] + 裸中文前缀`，不套聊天模板。
- 生成：首轮 8 路、最多 12 个新 token、固定 seed；分别测试固定 8 路和最多 24 路自适应补采样。
- 延迟：完整 CandidateGroup 墙钟时间，包含 Prefill、组内 Decode、过滤、去重和 Top-3 排序，不含模型加载和预热。

“契约违规”是可复现的结构门禁：命中冻结禁词、Question/Answer/摘要等元文本或指令文本、
重度拉丁字符、韩文或日文脚本。它不是人工自然度，也不把单参考 exact 当作开放短前缀的
唯一正确答案；因此这里报告的是明确失败的下界。

## 固定 8 路

{table(fixed)}

固定预算下，`{label_a}` 有 `{fixed['slots']['a']['all_distinct_rate']:.2%}` 的前缀直接返回
完整互异 Top-3；`{label_b}` 有
`{fixed['slots']['b']['affected_prefixes']}` 个前缀出现契约违规，且只有
`{fixed['slots']['b']['all_distinct_rate']:.2%}` 的前缀能直接返回完整互异 Top-3。

## 默认自适应补采样

{table(adaptive)}

`{label_b}` 通过平均 `{adaptive['slots']['b']['mean_sampling_attempts']:.2f}` 路生成填满了
候选栏，但违规候选并未被“多采几路”解决，p95 反而因补采样拉长到
`{adaptive['slots']['b']['latency_ms']['p95']:.2f} ms`。

## 同前缀原始输出

{examples_text}
## 复现

先启动推理前端，再执行：

```bash
python benchmark/bench_ime_short_prefix_compare.py \\
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
        "--slot-a-profile",
        help="Use a profile label returned by /api/config for comparison slot A",
    )
    parser.add_argument(
        "--slot-b-profile",
        help="Use a profile label returned by /api/config for comparison slot B",
    )
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--warmup-prefix", default="测试")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "reports/aios_ime_short_prefix_compare_20260821.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=PROJECT_ROOT / "reports/aios_ime_short_prefix_compare_20260821.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server_url = args.server_url.rstrip("/")
    config = read_json(f"{server_url}/api/config")
    for slot, profile_label in (
        ("a", args.slot_a_profile),
        ("b", args.slot_b_profile),
    ):
        if not profile_label:
            continue
        matches = [
            profile
            for profile in config.get("profiles", [])
            if profile.get("label") == profile_label
        ]
        if len(matches) != 1:
            available = ", ".join(
                str(profile.get("label")) for profile in config.get("profiles", [])
            )
            raise ValueError(
                f"Profile {profile_label!r} was not found exactly once; available: {available}"
            )
        config["slots"][slot] = matches[0]
    rows = [
        json.loads(line)
        for line in args.eval_data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cases = [row for row in rows if row.get("category") == args.category]
    if not cases:
        raise ValueError(f"No {args.category!r} rows found in {args.eval_data}")

    print("Warming both model workers on the fixed-8 path...", flush=True)
    compare_once(
        server_url,
        config,
        args.warmup_prefix,
        seed=args.seed - 1,
        max_sampling_attempts=8,
    )
    protocols = {
        "fixed8": run_protocol(
            server_url,
            config,
            cases,
            name="fixed8",
            seed=args.seed,
            max_sampling_attempts=8,
        ),
        "adaptive24": run_protocol(
            server_url,
            config,
            cases,
            name="adaptive24",
            seed=args.seed,
            max_sampling_attempts=24,
        ),
    }
    report = {
        "schema_version": "aios.ime_short_prefix_compare.v1",
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
        "protocols": protocols,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(
        {
            name: {
                "slots": {
                    slot: {
                        key: value
                        for key, value in summary.items()
                        if key != "contract_audit"
                    }
                    for slot, summary in protocol["slots"].items()
                },
                "comparison": protocol["comparison"],
            }
            for name, protocol in protocols.items()
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
