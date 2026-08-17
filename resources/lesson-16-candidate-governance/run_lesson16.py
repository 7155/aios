#!/usr/bin/env python3
"""Hard filtering, display dedup and MMR selection."""

from dataclasses import dataclass


@dataclass
class Candidate:
    text: str
    score: float


def key(text):
    return ''.join(text.split()).rstrip('，。！？；：,.!?;:').casefold()


def invalid(text):
    compact = ''.join(text.split())
    if not compact:
        return 'empty'
    if '作为一个AI' in compact or '以下是' in compact:
        return 'assistant_template'
    if '好的好的好的' in compact:
        return 'repeated_ngram'
    return ''


def bigrams(text):
    t = key(text)
    return {t[i:i + 2] for i in range(max(0, len(t) - 1))}


def sim(a, b):
    x, y = bigrams(a), bigrams(b)
    return len(x & y) / len(x | y) if x and y else 0.0


raw = [
    Candidate('我晚点给你发消息。', -0.10),
    Candidate('我晚点给你发消息', -0.12),
    Candidate('我晚一点给你发消息。', -0.11),
    Candidate('等我回来再联系你。', -0.16),
    Candidate('处理完以后回复你。', -0.18),
    Candidate('作为一个AI，我建议', -0.02),
]

valid = []
for candidate in raw:
    reason = invalid(candidate.text)
    print('filter', candidate.text, '->', reason or 'valid')
    if not reason:
        valid.append(candidate)

best = {}
for candidate in valid:
    if key(candidate.text) not in best or candidate.score > best[key(candidate.text)].score:
        best[key(candidate.text)] = candidate
remaining = list(best.values())
selected = []
diversity_lambda = 0.8
while remaining and len(selected) < 3:
    chosen = max(
        remaining,
        key=lambda candidate: candidate.score - diversity_lambda * max(
            [sim(candidate.text, item.text) for item in selected] or [0]
        ),
    )
    remaining.remove(chosen)
    selected.append(chosen)

print('\nTop-3:')
for candidate in selected:
    print(candidate)
