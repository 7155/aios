#!/usr/bin/env python3
"""Profile the 65-mixer Block AttnRes inference hot path on CUDA."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from aios.models.minimind_ime import BlockAttnResMixer


MIXERS_PER_FORWARD = 65


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("reference", "eager", "compiled", "triton"),
        required=True,
    )
    parser.add_argument("--active-tokens", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--trace",
        type=Path,
        help="Optional Chrome trace path for one warmed pipeline iteration.",
    )
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def make_mixer(
    hidden_size: int,
    backend: str,
    generator: torch.Generator,
) -> BlockAttnResMixer:
    mixer = BlockAttnResMixer(hidden_size, 1e-6, backend)
    mixer.query = torch.randn(
        hidden_size,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    mixer.key_norm.weight = torch.randn(
        hidden_size,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    return mixer


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.active_tokens < 1 or args.hidden_size < 1:
        raise ValueError("active-tokens and hidden-size must be positive")
    if args.iterations < 1 or args.warmup < 1:
        raise ValueError("warmup and iterations must be positive")

    generator = torch.Generator(device="cuda").manual_seed(20260821)
    banks = [
        torch.randn(
            depth,
            args.active_tokens,
            args.hidden_size,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        for depth in range(1, 10)
    ]
    partial = torch.randn(
        args.active_tokens,
        args.hidden_size,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    mixers = [
        make_mixer(args.hidden_size, args.backend, generator)
        for _ in range(MIXERS_PER_FORWARD)
    ]

    @torch.inference_mode()
    def run_pipeline() -> torch.Tensor:
        output: torch.Tensor | None = None
        mixer_index = 0
        for block_index in range(8):
            bank = banks[block_index]
            for layer_in_block in range(4):
                output = mixers[mixer_index].forward(
                    bank, None if layer_in_block == 0 else partial
                )
                mixer_index += 1
                output = mixers[mixer_index].forward(bank, partial)
                mixer_index += 1
        output = mixers[mixer_index].forward(banks[-1], None)
        assert mixer_index + 1 == MIXERS_PER_FORWARD
        return output

    for _ in range(args.warmup):
        run_pipeline()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    for _ in range(args.iterations):
        run_pipeline()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function(f"attnres_{args.backend}_65_mixers"):
            run_pipeline()
        torch.cuda.synchronize()

    if args.trace is not None:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(args.trace))

    result = {
        "backend": args.backend,
        "gpu": torch.cuda.get_device_name(),
        "active_tokens": args.active_tokens,
        "hidden_size": args.hidden_size,
        "mixers_per_forward": MIXERS_PER_FORWARD,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "elapsed_seconds": elapsed,
        "pipeline_latency_ms": elapsed * 1000.0 / args.iterations,
        "active_tokens_per_second": (
            args.active_tokens * args.iterations / elapsed
        ),
        "mixer_tokens_per_second": (
            args.active_tokens
            * MIXERS_PER_FORWARD
            * args.iterations
            / elapsed
        ),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "trace": str(args.trace) if args.trace is not None else None,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))


if __name__ == "__main__":
    main()
