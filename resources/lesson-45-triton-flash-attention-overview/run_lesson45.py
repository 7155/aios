#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Shape:
    batch: int
    heads: int
    seq: int
    dim: int


def tensor_bytes(elements: int, bytes_per_element: int) -> int:
    return elements * bytes_per_element


def gib(value: int) -> float:
    return value / (1024 ** 3)


def naive_score_elements(shape: Shape) -> int:
    return shape.batch * shape.heads * shape.seq * shape.seq


def flash_state_elements(shape: Shape) -> int:
    # O [B,H,N,D] + LSE/M [B,H,N]. Q/K/V are inputs, not extra attention matrices.
    return shape.batch * shape.heads * shape.seq * shape.dim + shape.batch * shape.heads * shape.seq


def program_trace(shape: Shape, block_q: int) -> list[tuple[int, int, int, range]]:
    trace = []
    q_blocks = (shape.seq + block_q - 1) // block_q
    for pid_q in range(q_blocks):
        for pid_bh in range(shape.batch * shape.heads):
            batch = pid_bh // shape.heads
            head = pid_bh % shape.heads
            q_range = range(pid_q * block_q, min((pid_q + 1) * block_q, shape.seq))
            trace.append((pid_q, batch, head, q_range))
    return trace


def main() -> None:
    original = Shape(batch=8, heads=16, seq=4096, dim=64)
    score_elems = naive_score_elements(original)
    fp16_score = tensor_bytes(score_elems, 2)
    fp32_score = tensor_bytes(score_elems, 4)

    assert score_elems == 2_147_483_648
    assert round(gib(fp16_score), 2) == 4.00
    assert round(gib(fp32_score), 2) == 8.00

    print("Original repository test shape:", original)
    print(f"One [B,H,N,N] score/probability tensor: {score_elems:,} elements")
    print(f"FP16 storage only: {gib(fp16_score):.2f} GiB")
    print(f"FP32 storage only: {gib(fp32_score):.2f} GiB")
    print("The naive reference can need several such buffers at once; this is not a safe smoke test.")

    smoke = Shape(batch=1, heads=2, seq=128, dim=64)
    naive = tensor_bytes(naive_score_elements(smoke), 4)
    state = tensor_bytes(flash_state_elements(smoke), 4)
    print("\nSuggested teaching smoke shape:", smoke)
    print(f"Naive FP32 score matrix: {naive / 1024:.1f} KiB")
    print(f"Flash output + LSE state (FP32 upper-bound illustration): {state / 1024:.1f} KiB")

    trace = program_trace(Shape(1, 2, 16, 64), block_q=4)
    print("\nFirst six Triton program assignments for grid=(ceil(N/BQ), B*H, 1):")
    for pid_q, batch, head, q_range in trace[:6]:
        print(f"  pid_q={pid_q}, batch={batch}, head={head}, queries={list(q_range)}")

    assert len(trace) == 8
    assert list(trace[0][3]) == [0, 1, 2, 3]
    assert list(trace[-1][3]) == [12, 13, 14, 15]
    print("\nPASSED: memory estimate and program-grid mapping are internally consistent.")


if __name__ == "__main__":
    main()
