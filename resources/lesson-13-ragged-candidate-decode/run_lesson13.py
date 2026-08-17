#!/usr/bin/env python3
"""Why row compaction needs candidate-local random streams."""

import random


def sequential_demo(compact: bool):
    rng = random.Random(7)
    active = [0, 1, 2]
    result = {i: [] for i in active}
    for step in range(3):
        rows = active if compact else [0, 1, 2]
        for row in rows:
            value = round(rng.random(), 6)
            if row in active:
                result[row].append(value)
        if step == 0:
            active.remove(0)
    return result


def stateless(candidate, step):
    return round(random.Random(candidate * 100003 + step).random(), 6)


print("sequential RNG, no compaction:", sequential_demo(False))
print("sequential RNG, compaction:   ", sequential_demo(True))
print("row 1 changed because row 0 stopped consuming random numbers\n")

full = {c: [stateless(c, s) for s in range(3)] for c in range(3)}
compacted = {c: [stateless(c, s) for s in range(3)] for c in (1, 2)}
print("stateless full:      ", full)
print("stateless compacted: ", compacted)
assert compacted[1] == full[1]
assert compacted[2] == full[2]
