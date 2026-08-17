#!/usr/bin/env python3
from dataclasses import dataclass

@dataclass
class Req:
    uid: int
    prompt: int
    remain: int
    state: str = 'pending'

reqs = [Req(0, 3, 2), Req(1, 5, 1)]
round_id = 0
while any(r.state != 'finished' for r in reqs):
    print(f'round {round_id}')
    for r in reqs:
        if r.state == 'pending':
            r.state = 'running'
            r.remain -= 1  # prefill also samples first token
            print(' prefill uid', r.uid)
        elif r.state == 'running':
            r.remain -= 1
            print(' decode  uid', r.uid)
        if r.state == 'running' and r.remain == 0:
            r.state = 'finished'
            print(' finish/free uid', r.uid)
    round_id += 1
print('API order:', [r.uid for r in sorted(reqs, key=lambda r: r.uid)])
