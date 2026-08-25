#!/usr/bin/env python3
"""Run one AIOS-IME CandidateGroup and print the complete result as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aios import ImeCompletionEngine, ImeGenerationConfig, LLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--sampling-attempts", type=int, default=8)
    parser.add_argument("--max-sampling-attempts", type=int, default=24)
    parser.add_argument("--refill-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--max-candidate-chars", type=int, default=32)
    parser.add_argument("--refill-deadline-ms", type=float, default=0.0)
    parser.add_argument("--kv-cache-max-tokens", type=int, default=256)
    parser.add_argument("--attention-workspace-mib", type=int, default=1)
    parser.add_argument(
        "--attnres-backend",
        choices=("reference", "eager", "compiled", "triton"),
        default="triton",
    )
    return parser.parse_args()


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
    config = ImeGenerationConfig(
        display_candidates=3,
        sampling_attempts=args.sampling_attempts,
        max_sampling_attempts=args.max_sampling_attempts,
        refill_batch_size=args.refill_batch_size,
        max_new_tokens=args.max_new_tokens,
        max_candidate_chars=args.max_candidate_chars,
        refill_deadline_ms=args.refill_deadline_ms,
        seed=args.seed,
    )

    try:
        result = engine.complete(args.prefix, config)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    finally:
        engine.reset_prefix_cache()


if __name__ == "__main__":
    main()
