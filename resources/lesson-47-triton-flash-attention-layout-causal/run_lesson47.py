#!/usr/bin/env python3
from __future__ import annotations


def contiguous_strides(batch: int, heads: int, seq: int, dim: int) -> tuple[int, int, int, int]:
    del batch
    return heads * seq * dim, seq * dim, dim, 1


def base_offset(batch_index: int, head_index: int, strides: tuple[int, int, int, int]) -> int:
    return batch_index * strides[0] + head_index * strides[1]


def causal_cells(seq: int) -> set[tuple[int, int]]:
    return {(q, k) for q in range(seq) for k in range(seq) if k <= q}


def forward_stage_cells(seq: int, block_q: int, block_kv: int, causal: bool) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    q_blocks = (seq + block_q - 1) // block_q
    for block_index_q in range(q_blocks):
        q_start = block_index_q * block_q
        q_end = min(q_start + block_q, seq)
        q_positions = range(q_start, q_end)
        if not causal:
            kv_ranges = [(0, seq, False)]
        else:
            # Stage 1: blocks strictly left of the diagonal; Stage 2: diagonal transition block.
            kv_ranges = [(0, q_start, False), (q_start, min(q_start + block_q, seq), True)]
        for lo, hi, diagonal in kv_ranges:
            for kv_start in range(lo, hi, block_kv):
                for q in q_positions:
                    for k in range(kv_start, min(kv_start + block_kv, hi)):
                        if not diagonal or k <= q:
                            cells.add((q, k))
    return cells


def main() -> None:
    batch, heads, seq, dim = 2, 3, 16, 64
    strides = contiguous_strides(batch, heads, seq, dim)
    print("Contiguous [B,H,N,D] strides:", strides)
    print("base offset for batch=1, head=2:", base_offset(1, 2, strides))
    assert base_offset(1, 2, strides) == (1 * heads + 2) * seq * dim

    # Demonstrate the repository's implicit same-layout contract.
    q_strides = strides
    k_strides = (heads * seq * (dim + 8), seq * (dim + 8), dim + 8, 1)
    q_based = base_offset(1, 2, q_strides)
    k_based = base_offset(1, 2, k_strides)
    print(f"Q-based offset={q_based}, true K offset with another layout={k_based}")
    assert q_based != k_based

    noncausal = forward_stage_cells(seq=16, block_q=4, block_kv=2, causal=False)
    causal = forward_stage_cells(seq=16, block_q=4, block_kv=2, causal=True)
    assert noncausal == {(q, k) for q in range(16) for k in range(16)}
    assert causal == causal_cells(16)
    print(f"non-causal covered cells={len(noncausal)} (expected 256)")
    print(f"causal covered cells={len(causal)} (expected 136)")

    configs = [
        (bq, bkv, stages, warps)
        for bq in (64, 128)
        for bkv in (32, 64)
        for stages in (3, 4, 7)
        for warps in (2, 4)
    ]
    print(f"forward autotune candidates={len(configs)}")
    assert len(configs) == 24
    print("PASSED: stride contract, stage coverage, and autotune search size are explicit.")


if __name__ == "__main__":
    main()
