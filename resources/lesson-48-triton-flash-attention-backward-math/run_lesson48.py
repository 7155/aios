#!/usr/bin/env python3
from __future__ import annotations

import math
from copy import deepcopy

Matrix = list[list[float]]


def transpose(matrix: Matrix) -> Matrix:
    return [list(column) for column in zip(*matrix)]


def matmul(left: Matrix, right: Matrix) -> Matrix:
    right_t = transpose(right)
    return [[sum(a * b for a, b in zip(row, column)) for column in right_t] for row in left]


def scale_matrix(matrix: Matrix, scale: float) -> Matrix:
    return [[scale * value for value in row] for row in matrix]


def softmax_rows(scores: Matrix, causal: bool) -> Matrix:
    result: Matrix = []
    for q, row in enumerate(scores):
        masked = [value if (not causal or k <= q) else float("-inf") for k, value in enumerate(row)]
        maximum = max(masked)
        weights = [0.0 if math.isinf(value) and value < 0 else math.exp(value - maximum) for value in masked]
        denominator = sum(weights)
        result.append([weight / denominator for weight in weights])
    return result


def forward(q: Matrix, k: Matrix, v: Matrix, scale: float, causal: bool) -> tuple[Matrix, Matrix]:
    scores = scale_matrix(matmul(q, transpose(k)), scale)
    p = softmax_rows(scores, causal)
    return matmul(p, v), p


def loss(q: Matrix, k: Matrix, v: Matrix, do: Matrix, scale: float, causal: bool) -> float:
    output, _ = forward(q, k, v, scale, causal)
    return sum(output[i][d] * do[i][d] for i in range(len(output)) for d in range(len(output[0])))


def analytic_grads(
    q: Matrix, k: Matrix, v: Matrix, do: Matrix, scale: float, causal: bool
) -> tuple[Matrix, Matrix, Matrix, list[float]]:
    output, p = forward(q, k, v, scale, causal)
    dp = matmul(do, transpose(v))
    delta = [sum(do_i[d] * out_i[d] for d in range(len(out_i))) for do_i, out_i in zip(do, output)]
    ds = [[p[i][j] * (dp[i][j] - delta[i]) for j in range(len(p[0]))] for i in range(len(p))]
    dq = scale_matrix(matmul(ds, k), scale)
    dk = scale_matrix(matmul(transpose(ds), q), scale)
    dv = matmul(transpose(p), do)
    return dq, dk, dv, delta


def finite_difference(
    name: str, q: Matrix, k: Matrix, v: Matrix, do: Matrix, scale: float, causal: bool, eps: float = 1e-6
) -> Matrix:
    target = {"Q": q, "K": k, "V": v}[name]
    gradient = [[0.0 for _ in row] for row in target]
    for i in range(len(target)):
        for j in range(len(target[0])):
            plus_q, plus_k, plus_v = deepcopy(q), deepcopy(k), deepcopy(v)
            minus_q, minus_k, minus_v = deepcopy(q), deepcopy(k), deepcopy(v)
            {"Q": plus_q, "K": plus_k, "V": plus_v}[name][i][j] += eps
            {"Q": minus_q, "K": minus_k, "V": minus_v}[name][i][j] -= eps
            plus = loss(plus_q, plus_k, plus_v, do, scale, causal)
            minus = loss(minus_q, minus_k, minus_v, do, scale, causal)
            gradient[i][j] = (plus - minus) / (2 * eps)
    return gradient


def max_error(left: Matrix, right: Matrix) -> float:
    return max(abs(left[i][j] - right[i][j]) for i in range(len(left)) for j in range(len(left[0])))


def main() -> None:
    q = [[0.2, -0.3], [0.7, 0.1], [-0.4, 0.8]]
    k = [[-0.5, 0.4], [0.6, -0.2], [0.1, 0.9]]
    v = [[1.0, -1.0], [0.3, 0.7], [-0.6, 0.2]]
    do = [[0.4, -0.1], [-0.2, 0.5], [0.8, -0.3]]
    scale = 1.0 / math.sqrt(2.0)

    for causal in (False, True):
        dq, dk, dv, delta = analytic_grads(q, k, v, do, scale, causal)
        fd_q = finite_difference("Q", q, k, v, do, scale, causal)
        fd_k = finite_difference("K", q, k, v, do, scale, causal)
        fd_v = finite_difference("V", q, k, v, do, scale, causal)
        errors = (max_error(dq, fd_q), max_error(dk, fd_k), max_error(dv, fd_v))
        print(f"causal={causal} D/delta={delta}")
        print(f"  max errors: dQ={errors[0]:.3e}, dK={errors[1]:.3e}, dV={errors[2]:.3e}")
        assert max(errors) < 2e-9

    print("PASSED: D = rowsum(dO*O) yields finite-difference-correct dQ/dK/dV.")


if __name__ == "__main__":
    main()
