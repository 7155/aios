#!/usr/bin/env python3
"""Toy workload comparison for Lesson 10. No GPU dependency."""

from dataclasses import dataclass


@dataclass
class RawCandidate:
    text: str
    valid: bool = True


def displayable_count(items: list[RawCandidate]) -> int:
    seen: set[str] = set()
    for item in items:
        key = item.text.rstrip("，。！？；：,.!?;:").replace(" ", "")
        if item.valid and key:
            seen.add(key)
    return len(seen)


three_rows = [
    RawCandidate("我晚点给你发消息。"),
    RawCandidate("作为一个AI，我建议", valid=False),
    RawCandidate("我晚点给你发消息"),
]

eight_rows = three_rows + [
    RawCandidate("等我回来再联系你。"),
    RawCandidate("处理完以后回复你。"),
    RawCandidate("我回去以后再认真回复你。"),
    RawCandidate("好的好的好的", valid=False),
    RawCandidate("我晚一点给你发消息。"),
]

print("3 raw branches -> displayable:", displayable_count(three_rows))
print("8 raw branches -> displayable:", displayable_count(eight_rows))
print()
print("Full keystroke latency boundary:")
for stage in [
    "tokenize + token-LCP",
    "prefix prefill",
    "candidate-group decode",
    "decode text",
    "filter + deduplicate + MMR",
    "optional refill",
]:
    print(" -", stage)
