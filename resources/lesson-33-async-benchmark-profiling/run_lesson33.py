#!/usr/bin/env python3
"""Percentiles, Amdahl limits, and optional CUDA synchronization timing."""

from __future__ import annotations

import math
import statistics
import time


def percentile(values, q):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def main() -> None:
    values = [80, 81, 82, 83, 85, 90, 100, 110, 180, 300]
    print("mean:", statistics.mean(values))
    print("p50: ", statistics.median(values))
    print("p95: ", percentile(values, 0.95))

    fraction = 0.05
    kernel_speedup = 2.0
    total_speedup = 1 / ((1 - fraction) + fraction / kernel_speedup)
    print(f"Amdahl: a 5% kernel made 2x faster -> total {total_speedup:.3f}x")

    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        print("CUDA unavailable; skipped enqueue timing")
        return
    x = torch.randn(4_000_000, device="cuda")
    for _ in range(5):
        _ = x * 2
    torch.cuda.synchronize()
    start = time.perf_counter()
    _ = x * 2
    enqueue = (time.perf_counter() - start) * 1000
    torch.cuda.synchronize()
    complete = (time.perf_counter() - start) * 1000
    print(f"enqueue={enqueue:.4f} ms complete={complete:.4f} ms")


if __name__ == "__main__":
    main()
