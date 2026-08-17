#!/usr/bin/env python3
"""Build and decode FlashInfer-style cu_seqlens and page indices on CPU."""

from __future__ import annotations


def prefix_sum(lengths):
    result = [0]
    for length in lengths:
        result.append(result[-1] + length)
    return result


def main() -> None:
    q_lengths = [3, 2]
    kv_pages = [[10, 11, 12], [20, 21, 22, 23, 24]]
    k_lengths = [len(pages) for pages in kv_pages]
    cu_q = prefix_sum(q_lengths)
    cu_k = prefix_sum(k_lengths)
    indices = [page for pages in kv_pages for page in pages]
    print("cu_q:", cu_q)
    print("cu_k:", cu_k)
    print("indices:", indices)
    for request in range(len(q_lengths)):
        q_range = (cu_q[request], cu_q[request + 1])
        pages = indices[cu_k[request] : cu_k[request + 1]]
        print(f"request {request}: flat q range={q_range}, pages={pages}")
    assert cu_q == [0, 3, 5]
    assert cu_k == [0, 3, 8]


if __name__ == "__main__":
    main()
