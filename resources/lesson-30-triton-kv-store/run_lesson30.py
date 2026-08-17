#!/usr/bin/env python3
"""CPU reference for the Triton KV scatter kernel."""

from __future__ import annotations


def scatter(cache, indices, values):
    for token_idx, physical_index in enumerate(indices):
        cache[physical_index] = list(values[token_idx])


def main() -> None:
    width = 8
    indices = [17, 4, 9]
    values = [[token * 100 + offset for offset in range(width)] for token in range(3)]
    cache = [[-1] * width for _ in range(20)]
    scatter(cache, indices, values)
    for token_idx, physical_index in enumerate(indices):
        print(f"token row {token_idx} -> cache row {physical_index}: {cache[physical_index]}")
        assert cache[physical_index] == values[token_idx]
    untouched = {row for row in range(len(cache)) if cache[row][0] == -1}
    assert 17 not in untouched and 4 not in untouched and 9 not in untouched
    print("all scatter assertions passed")


if __name__ == "__main__":
    main()
