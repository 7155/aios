#!/usr/bin/env python3
raw=[]; live_suffix=[]; cancelled=False
batches=[
    [('A',True),('A.',True),('',False),('bad',False),('B',True),('B',True),('x',False),('y',False)],
    [('C',True),('D',True),('C.',True),('z',False)],
]
for round_id,batch in enumerate(batches):
    suffix=list(range(round_id*10,round_id*10+len(batch)))
    live_suffix.extend(suffix)
    raw.extend(batch)
    print('round',round_id,'allocated suffix',suffix)
    live_suffix.clear()
    print('materialized -> suffix freed')
    selected=[]
    seen=set()
    for text,valid in raw:
        key=text.rstrip('.')
        if valid and key and key not in seen:
            seen.add(key); selected.append(text)
    print('selected:',selected[:3])
    if len(selected)>=3: break
print('live suffix pages:',live_suffix)
assert len(selected[:3])==3 and not live_suffix
