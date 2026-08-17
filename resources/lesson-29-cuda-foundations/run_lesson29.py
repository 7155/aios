#!/usr/bin/env python3
"""CUDA enqueue vs completion demo. Works without CUDA as a conceptual simulation."""

from __future__ import annotations

import time


def cpu_queue_simulation() -> None:
    queue = []
    queue.append("random kernel")
    queue.append("multiply kernel")
    print("CPU has enqueued:", queue)
    while queue:
        print("GPU executes:", queue.pop(0))


def cuda_demo() -> None:
    import torch

    if not torch.cuda.is_available():
        print("CUDA unavailable; skipped real GPU timing.")
        return
    x = torch.randn(4_000_000, device="cuda")
    for _ in range(5):
        _ = x * 2
    torch.cuda.synchronize()
    start = time.perf_counter()
    _ = x * 2
    enqueue_ms = (time.perf_counter() - start) * 1000
    torch.cuda.synchronize()
    complete_ms = (time.perf_counter() - start) * 1000
    print(f"enqueue return: {enqueue_ms:.4f} ms")
    print(f"GPU complete:   {complete_ms:.4f} ms")


if __name__ == "__main__":
    cpu_queue_simulation()
    cuda_demo()
