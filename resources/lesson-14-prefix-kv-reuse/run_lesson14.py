#!/usr/bin/env python3
"""Token-LCP cache lifecycle walkthrough."""


def lcp(left, right):
    n = 0
    for a, b in zip(left, right):
        if a != b:
            break
        n += 1
    return n


def plan(old_ids, old_pages, new_ids):
    reused = lcp(old_ids, new_ids)
    reason = "normal"
    if old_ids and reused == len(new_ids) < len(old_ids):
        reused = 0
        reason = "strict backspace: re-prefill to recover exact final logits"
    kept = old_pages[:reused]
    released = old_pages[reused:]
    allocated = list(range(100, 100 + len(new_ids) - reused))
    return reused, kept, released, allocated, reason


cases = [
    ("same", [1,2,3], [1,2,3]),
    ("append", [1,2,3], [1,2,3,4,5]),
    ("retokenized tail", [1,2,3,4], [1,2,8,9]),
    ("backspace", [1,2,3,4], [1,2,3]),
]
for name, old_ids, new_ids in cases:
    old_pages = list(range(10, 10 + len(old_ids)))
    print("\n", name)
    print("old/new:", old_ids, new_ids)
    print("reuse, kept, released, allocated, reason =")
    print(plan(old_ids, old_pages, new_ids))
