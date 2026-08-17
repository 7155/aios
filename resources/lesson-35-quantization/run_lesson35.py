#!/usr/bin/env python3
"""Symmetric INT8 quantization mechanics and storage estimates."""

from __future__ import annotations


def quantize(values):
    max_abs = max(abs(value) for value in values)
    scale = max_abs / 127 if max_abs else 1.0
    q = [max(-127, min(127, round(value / scale))) for value in values]
    return q, scale


def main() -> None:
    values = [-1.0, -0.4, 0.2, 0.9]
    q, scale = quantize(values)
    reconstructed = [value * scale for value in q]
    errors = [abs(a - b) for a, b in zip(values, reconstructed)]
    print("values:       ", values)
    print("int8:        ", q)
    print("scale:       ", scale)
    print("reconstructed", reconstructed)
    print("max error:   ", max(errors))

    params = 100_687_360
    for name, bytes_per in [("BF16", 2), ("INT8/FP8", 1), ("INT4", 0.5)]:
        print(name, f"{params * bytes_per / 2**20:.2f} MiB (before scale metadata)")


if __name__ == "__main__":
    main()
