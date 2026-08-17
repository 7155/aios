#!/usr/bin/env python3
"""Packed projection equivalence and conceptual memory-traffic calculation."""

from __future__ import annotations


def matvec(weight, x):
    return [sum(a * b for a, b in zip(row, x)) for row in weight]


def main() -> None:
    x = [1.0, 2.0, -1.0]
    wq = [[1, 0, 0], [0, 1, 0]]
    wk = [[0, 0, 1]]
    wv = [[1, 1, 1]]
    separate = matvec(wq, x) + matvec(wk, x) + matvec(wv, x)
    packed = matvec(wq + wk + wv, x)
    print("separate:", separate)
    print("packed:  ", packed)
    assert separate == packed

    n, intermediate, bytes_per = 128, 2048, 2
    unfused = 5 * n * intermediate * bytes_per
    fused = 3 * n * intermediate * bytes_per
    print(f"conceptual SwiGLU traffic: {unfused/2**20:.2f} MiB -> {fused/2**20:.2f} MiB")


if __name__ == "__main__":
    main()
