#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    batch: int
    heads: int
    seq: int
    dim: int
    causal: bool


def mib(value: int) -> float:
    return value / (1024 ** 2)


def naive_probability_bytes(case: Case, bytes_per_element: int = 4) -> int:
    return case.batch * case.heads * case.seq * case.seq * bytes_per_element


def satisfies_pinned_kernel_contract(case: Case) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if case.seq % 128:
        reasons.append("SEQ_LEN is not divisible by the backward macro tile 128")
    if case.seq % 64:
        reasons.append("SEQ_LEN is not divisible by all forward candidate Q/KV tiles")
    if case.dim not in (64, 128):
        reasons.append("HEAD_DIM is outside the teaching/tested 64/128 set")
    return not reasons, reasons


def benchmark_phases() -> list[str]:
    return [
        "compile/autotune outside user latency",
        "warm up until clocks and caches stabilize",
        "synchronize before starting the timer",
        "run many steady-state repetitions",
        "synchronize before stopping the timer",
        "report median and tail, shape, dtype, GPU, versions, causal flag",
    ]


def main() -> None:
    cases = [
        Case(1, 2, 128, 64, False),
        Case(1, 2, 128, 64, True),
        Case(2, 8, 512, 64, True),
        Case(1, 4, 1024, 128, False),
        Case(1, 2, 192, 64, True),  # Deliberate contract failure.
        Case(1, 2, 256, 80, False),  # Deliberate teaching-boundary failure.
    ]

    print("Validation matrix and naive FP32 probability memory:")
    valid_count = 0
    for case in cases:
        valid, reasons = satisfies_pinned_kernel_contract(case)
        valid_count += int(valid)
        print(
            f"  {case}: P={mib(naive_probability_bytes(case)):.2f} MiB, "
            f"contract={'PASS' if valid else 'SKIP'}"
        )
        for reason in reasons:
            print("    -", reason)

    assert valid_count == 4
    assert len(cases) - valid_count == 2

    print("\nBenchmark protocol:")
    for index, phase in enumerate(benchmark_phases(), 1):
        print(f"  {index}. {phase}")
    assert benchmark_phases()[0].startswith("compile/autotune")

    original = Case(8, 16, 4096, 64, True)
    print(
        f"\nOriginal default naive FP32 P alone: "
        f"{naive_probability_bytes(original) / (1024 ** 3):.2f} GiB"
    )
    assert naive_probability_bytes(original) == 8 * 1024 ** 3
    print("PASSED: tests are bucketed by contract and benchmark phases exclude first-use work.")


if __name__ == "__main__":
    main()
