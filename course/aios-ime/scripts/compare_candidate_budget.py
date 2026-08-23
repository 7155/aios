#!/usr/bin/env python3
"""Compare fixed 3, fixed 8 and adaptive 8-to-24 CandidateGroup budgets."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from aios import ImeCompletionEngine, ImeGenerationConfig, LLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--kv-cache-max-tokens", type=int, default=256)
    parser.add_argument("--attention-workspace-mib", type=int, default=1)
    parser.add_argument(
        "--attnres-backend",
        choices=("reference", "eager", "compiled", "triton"),
        default="triton",
    )
    return parser.parse_args()


def summarize(name: str, result: Any) -> dict[str, Any]:
    return {
        "name": name,
        "candidates": [candidate.text for candidate in result.candidates],
        "candidate_count": len(result.candidates),
        "valid_unique_candidates": result.valid_unique_candidates,
        "invalid_candidates": result.invalid_candidates,
        "duplicate_candidates": result.duplicate_candidates,
        "sampling_attempts": result.sampling_attempts,
        "refill_rounds": result.refill_rounds,
        "refill_stop_reason": result.refill_stop_reason,
        "generated_tokens": result.generated_tokens,
        "active_model_tokens": result.active_model_tokens,
        "latency_ms": result.latency_ms,
        "gpu_latency_ms": result.gpu_latency_ms,
        "unique_kv_pages": result.unique_kv_pages,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("AIOS-IME requires a CUDA device")

    llm = LLM(
        str(args.model),
        attnres_backend=args.attnres_backend,
        kv_cache_max_tokens=args.kv_cache_max_tokens,
        attention_workspace_size=args.attention_workspace_mib * 2**20,
    )
    engine = ImeCompletionEngine(llm)

    base = ImeGenerationConfig(
        display_candidates=3,
        max_new_tokens=12,
        max_candidate_chars=32,
        seed=args.seed,
    )
    cases = [
        (
            "fixed_3",
            replace(
                base,
                sampling_attempts=3,
                max_sampling_attempts=3,
                refill_batch_size=3,
                min_refill_batch_size=2,
            ),
        ),
        (
            "fixed_8",
            replace(
                base,
                sampling_attempts=8,
                max_sampling_attempts=8,
                refill_batch_size=8,
            ),
        ),
        (
            "adaptive_8_to_24",
            replace(
                base,
                sampling_attempts=8,
                max_sampling_attempts=24,
                refill_batch_size=8,
                min_refill_batch_size=2,
            ),
        ),
    ]

    rows: list[dict[str, Any]] = []
    try:
        for name, config in cases:
            # Keep this one-prefix observation fair: no later case may inherit
            # the previous case's persistent prefix KV.
            engine.reset_prefix_cache()
            result = engine.complete(args.prefix, config)
            rows.append(summarize(name, result))
    finally:
        engine.reset_prefix_cache()

    print(
        json.dumps(
            {
                "note": (
                    "Single-prefix observation only; use benchmark/bench_ime.py "
                    "for warmed multi-sample performance evidence."
                ),
                "prefix": args.prefix,
                "seed": args.seed,
                "results": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
