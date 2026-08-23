#!/usr/bin/env python3
"""Trace prefix-token reuse across a sequence of simulated IME keystrokes."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from aios import ImeCompletionEngine, ImeGenerationConfig, LLM


DEFAULT_PREFIXES = [
    "没关系",
    "没关系，你",
    "没关系，你先忙",
    "没关系，你先忙你的，",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--prefix",
        action="append",
        dest="prefixes",
        help="Repeat this option to supply an ordered typing sequence.",
    )
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--kv-cache-max-tokens", type=int, default=256)
    parser.add_argument("--attention-workspace-mib", type=int, default=1)
    parser.add_argument(
        "--attnres-backend",
        choices=("reference", "eager", "compiled", "triton"),
        default="triton",
    )
    return parser.parse_args()


def row_from_result(step: int, result: Any) -> dict[str, Any]:
    return {
        "step": step,
        "prefix": result.prefix,
        "generation_id": result.generation_id,
        "prefix_tokens": result.prefix_tokens,
        "reused_prefix_tokens": result.reused_prefix_tokens,
        "cancelled": result.cancelled,
        "sampling_attempts": result.sampling_attempts,
        "refill_rounds": result.refill_rounds,
        "refill_stop_reason": result.refill_stop_reason,
        "latency_ms": result.latency_ms,
        "candidates": [candidate.text for candidate in result.candidates],
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("AIOS-IME requires a CUDA device")

    prefixes = args.prefixes or DEFAULT_PREFIXES
    llm = LLM(
        str(args.model),
        attnres_backend=args.attnres_backend,
        kv_cache_max_tokens=args.kv_cache_max_tokens,
        attention_workspace_size=args.attention_workspace_mib * 2**20,
    )
    engine = ImeCompletionEngine(llm)
    base_config = ImeGenerationConfig(seed=args.seed)

    rows: list[dict[str, Any]] = []
    try:
        for step, prefix in enumerate(prefixes):
            config = replace(
                base_config,
                seed=args.seed + step * 100_003,
            )
            result = engine.complete(prefix, config)
            rows.append(row_from_result(step, result))
    finally:
        engine.reset_prefix_cache()

    print(
        json.dumps(
            {
                "note": (
                    "This sequential trace demonstrates token-LCP reuse. "
                    "Run the GPU latest-wins test for concurrent cancellation evidence."
                ),
                "prefixes": prefixes,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
