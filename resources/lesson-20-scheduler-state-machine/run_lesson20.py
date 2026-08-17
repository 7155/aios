#!/usr/bin/env python3
pending = [('long', 9), ('short-a', 2), ('short-b', 2)]
running = ['old-decode']
budget = 4
selected = []
used = 0
for name, prompt_len in pending:
    if selected and used + prompt_len > budget:
        break
    if prompt_len > budget and selected:
        break
    if prompt_len > budget and not selected:
        # Mirrors current code: oversized first request can still be selected if capacity permits.
        selected.append(name); used += prompt_len; break
    selected.append(name); used += prompt_len
print('prefill-first selected:', selected)
print('decode delayed this round:', running)
print('short requests behind head remain pending:', [x[0] for x in pending[len(selected):]])
