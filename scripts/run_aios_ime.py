#!/usr/bin/env python3
"""Run the AIOS-IME single-user CandidateGroup pipeline."""

from __future__ import annotations

import argparse
import json

from aios import ImeCompletionEngine, ImeGenerationConfig, LLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prefix", action="append", required=True)
    parser.add_argument("--display-candidates", type=int, default=3)
    parser.add_argument("--sampling-attempts", type=int, default=8)
    parser.add_argument("--max-sampling-attempts", type=int, default=12)
    parser.add_argument("--refill-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--refill-temperature", type=float, default=0.55)
    parser.add_argument("--refill-top-k", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--kv-cache-max-tokens", type=int, default=256)
    parser.add_argument("--attention-workspace-mib", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm = LLM(
        args.model,
        kv_cache_max_tokens=args.kv_cache_max_tokens,
        attention_workspace_size=args.attention_workspace_mib * 2**20,
    )
    engine = ImeCompletionEngine(llm)
    config = ImeGenerationConfig(
        display_candidates=args.display_candidates,
        sampling_attempts=args.sampling_attempts,
        max_sampling_attempts=args.max_sampling_attempts,
        refill_batch_size=args.refill_batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        refill_temperature=args.refill_temperature,
        refill_top_k=args.refill_top_k,
        seed=args.seed,
    )
    for prefix in args.prefix:
        result = engine.complete(prefix, config)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
