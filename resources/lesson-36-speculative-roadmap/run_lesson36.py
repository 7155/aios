#!/usr/bin/env python3
"""Simple expected-acceptance and cost model for speculative decoding."""

from __future__ import annotations


def expected_accepted(probability, draft_length):
    return sum(probability ** index for index in range(1, draft_length + 1))


def cost_per_committed_token(probability, draft_length, draft_cost, verify_cost):
    committed = expected_accepted(probability, draft_length)
    return (draft_cost + verify_cost) / max(committed, 1e-9)


def main() -> None:
    for probability in (0.4, 0.6, 0.8, 0.9):
        for length in (2, 4, 8):
            accepted = expected_accepted(probability, length)
            cost = cost_per_committed_token(
                probability,
                length,
                draft_cost=0.35 * length,
                verify_cost=1.2,
            )
            print(f"a={probability:.1f} k={length}: E[accepted]={accepted:.3f}, cost/token={cost:.3f}")
        print()


if __name__ == "__main__":
    main()
