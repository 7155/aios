#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Iterable


def naive_attention(scores: list[float], values: list[list[float]]) -> list[float]:
    maximum = max(scores)
    weights = [math.exp(score - maximum) for score in scores]
    denominator = sum(weights)
    return [
        sum(weight * value[d] for weight, value in zip(weights, values)) / denominator
        for d in range(len(values[0]))
    ]


def online_attention(
    scores: list[float], values: list[list[float]], block_size: int, trace: bool = False
) -> list[float]:
    m_i = float("-inf")
    l_i = 1.0  # The first alpha becomes zero, so this sentinel does not affect the result.
    out = [0.0 for _ in values[0]]

    for start in range(0, len(scores), block_size):
        block_scores = scores[start : start + block_size]
        block_values = values[start : start + block_size]
        m_ij = max(m_i, max(block_scores))
        alpha = 0.0 if math.isinf(m_i) and m_i < 0 else math.exp(m_i - m_ij)
        probabilities = [math.exp(score - m_ij) for score in block_scores]

        out = [alpha * item for item in out]
        for probability, value in zip(probabilities, block_values):
            for d, item in enumerate(value):
                out[d] += probability * item
        l_i = alpha * l_i + sum(probabilities)
        m_i = m_ij

        if trace:
            print(
                f"block={start // block_size} scores={block_scores} "
                f"m={m_i:.6f} alpha={alpha:.6f} l={l_i:.6f} out_acc={out}"
            )

    return [item / l_i for item in out]


def max_abs_error(left: Iterable[float], right: Iterable[float]) -> float:
    return max(abs(a - b) for a, b in zip(left, right))


def main() -> None:
    scores = [1.0, 2.0, 5.0, -1.0, 3.5]
    values = [
        [1.0, 0.0],
        [0.0, 1.0],
        [2.0, -1.0],
        [3.0, 2.0],
        [-2.0, 4.0],
    ]

    reference = naive_attention(scores, values)
    print("Naive result:", reference)
    for block_size in (1, 2, 3, 4, 8):
        actual = online_attention(scores, values, block_size, trace=(block_size == 2))
        error = max_abs_error(reference, actual)
        print(f"block_size={block_size}: result={actual}, max_abs_error={error:.3e}")
        assert error < 1e-12

    # LSE identity used by the backward pass.
    maximum = max(scores)
    lse = maximum + math.log(sum(math.exp(score - maximum) for score in scores))
    probabilities = [math.exp(score - lse) for score in scores]
    assert abs(sum(probabilities) - 1.0) < 1e-12
    print(f"\nLSE={lse:.6f}; sum(exp(score-LSE))={sum(probabilities):.12f}")
    print("PASSED: block order changes the execution trace, not the mathematical result.")


if __name__ == "__main__":
    main()
