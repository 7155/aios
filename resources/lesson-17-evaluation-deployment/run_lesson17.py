#!/usr/bin/env python3
"""A tiny promotion gate: fastest is not automatically shippable."""

candidates = [
    dict(name='fixed-8', p50=87.72, p95=140.90, full3=0.9333, distinct=1.0, ranking_ok=True, pages_recycled=True),
    dict(name='adaptive-8+4', p50=81.98, p95=109.97, full3=1.0, distinct=1.0, ranking_ok=True, pages_recycled=True),
    dict(name='fast-but-broken', p50=70.0, p95=90.0, full3=0.80, distinct=0.7, ranking_ok=False, pages_recycled=True),
]


def passes(item):
    reasons = []
    if item['full3'] < 1.0:
        reasons.append('full-3 gate')
    if item['distinct'] < 0.97:
        reasons.append('diversity gate')
    if not item['ranking_ok']:
        reasons.append('frozen ranking gate')
    if not item['pages_recycled']:
        reasons.append('KV lifecycle gate')
    return reasons


for item in candidates:
    reasons = passes(item)
    print(item['name'], 'REJECT ' + ', '.join(reasons) if reasons else 'PROMOTABLE')

promotable = [item for item in candidates if not passes(item)]
best = min(promotable, key=lambda item: item['p95'])
print('\nselected by lowest p95 among gate-passing builds:', best['name'])
