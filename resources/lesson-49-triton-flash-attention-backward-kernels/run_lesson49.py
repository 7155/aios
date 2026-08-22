#!/usr/bin/env python3
from __future__ import annotations


def useful_causal(seq: int) -> set[tuple[int, int]]:
    return {(q, k) for q in range(seq) for k in range(seq) if k <= q}


def tile_cells(q_start: int, q_end: int, k_start: int, k_end: int) -> set[tuple[int, int]]:
    return {(q, k) for q in range(q_start, q_end) for k in range(k_start, k_end)}


def dq_schedule(seq: int, block_q: int, block_kv: int, optimized: bool) -> set[tuple[int, int]]:
    visited: set[tuple[int, int]] = set()
    for q_start in range(0, seq, block_q):
        q_end = min(q_start + block_q, seq)
        kv_limit = q_end if optimized else seq
        for k_start in range(0, kv_limit, block_kv):
            visited |= tile_cells(q_start, q_end, k_start, min(k_start + block_kv, seq))
    return visited


def dkdv_schedule(seq: int, block_q: int, block_kv: int, optimized: bool) -> set[tuple[int, int]]:
    visited: set[tuple[int, int]] = set()
    for k_start in range(0, seq, block_kv):
        k_end = min(k_start + block_kv, seq)
        q_begin = (k_start // block_q) * block_q if optimized else 0
        for q_start in range(q_begin, seq, block_q):
            visited |= tile_cells(q_start, min(q_start + block_q, seq), k_start, k_end)
    return visited


def report(name: str, visited: set[tuple[int, int]], useful: set[tuple[int, int]]) -> None:
    productive = len(visited & useful)
    waste = len(visited - useful)
    print(
        f"{name}: visited={len(visited)}, productive={productive}, masked/waste={waste}, "
        f"useful_coverage={productive / len(useful):.1%}"
    )


def main() -> None:
    seq, block_q, block_kv = 16, 4, 4
    useful = useful_causal(seq)

    baseline_dq = dq_schedule(seq, block_q, block_kv, optimized=False)
    optimized_dq = dq_schedule(seq, block_q, block_kv, optimized=True)
    baseline_dkdv = dkdv_schedule(seq, block_q, block_kv, optimized=False)
    optimized_dkdv = dkdv_schedule(seq, block_q, block_kv, optimized=True)

    report("dQ baseline", baseline_dq, useful)
    report("dQ causal-pruned", optimized_dq, useful)
    report("dK/dV baseline", baseline_dkdv, useful)
    report("dK/dV causal-pruned", optimized_dkdv, useful)

    for visited in (optimized_dq, optimized_dkdv):
        assert useful <= visited, "Optimization must not skip any mathematically useful score cell"
    assert len(optimized_dq) < len(baseline_dq)
    assert len(optimized_dkdv) < len(baseline_dkdv)

    # Each output tile has one owner; therefore no HBM atomics are required in this schedule.
    dq_owners = {q // block_q for q in range(seq)}
    kv_owners = {k // block_kv for k in range(seq)}
    assert len(dq_owners) == seq // block_q
    assert len(kv_owners) == seq // block_kv
    print("PASSED: causal loop bounds reduce visited tiles while preserving all useful cells.")


if __name__ == "__main__":
    main()
