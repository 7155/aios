#!/usr/bin/env python3
pool = {
    2: ['A0','A1','A2'],
    0: ['B0','B1','B2','B3','B4'],
}
pages = {
    2: [10,11,12],
    0: [20,21,22,23,24],
}
reqs = [(2,0,3),(0,0,5)]  # slot,cached,device
mapping=[]; positions=[]
for slot,cached,device in reqs:
    for pos in range(cached,device):
        mapping.append(slot); positions.append(pos)
flat=[pool[s][p] for s,p in zip(mapping,positions)]
out=[pages[s][p] for s,p in zip(mapping,positions)]
lengths=[d-c for _,c,d in reqs]
cu=[0]
for length in lengths: cu.append(cu[-1]+length)
print('mapping:',mapping)
print('positions:',positions)
print('flat input:',flat)
print('out_loc:',out)
print('cu_seqlens_q:',cu)
print('last indices:',[x-1 for x in cu[1:]])
