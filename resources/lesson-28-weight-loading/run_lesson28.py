#!/usr/bin/env python3
"""Show that fused QKV packing preserves the three separate linear projections."""


def matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(weight * value for weight, value in zip(row, vector)) for row in matrix]


x = [1.0, 2.0]
q_weight = [[1.0, 0.0], [0.0, 1.0]]
k_weight = [[1.0, 1.0]]
v_weight = [[2.0, -1.0]]

q = matvec(q_weight, x)
k = matvec(k_weight, x)
v = matvec(v_weight, x)

# The inference loader concatenates the output rows (dim 0).
packed_weight = [*q_weight, *k_weight, *v_weight]
packed_output = matvec(packed_weight, x)
q2 = packed_output[:2]
k2 = packed_output[2:3]
v2 = packed_output[3:4]

print("packed rows:", len(packed_weight), "input width:", len(packed_weight[0]))
print("separate Q/K/V:", q, k, v)
print("packed then split:", q2, k2, v2)
assert q == q2 and k == k2 and v == v2
