#!/usr/bin/env python3
q_lengths=[3,2]; k_lengths=[3,5]
pages=[[7,4,9],[2,8,1,6,3]]
def cumulative(lengths):
    out=[0]
    for n in lengths: out.append(out[-1]+n)
    return out
cu_q=cumulative(q_lengths); cu_k=cumulative(k_lengths)
indices=sum(pages,[])
last=[x-1 for x in cu_q[1:]]
print('cu_q:',cu_q)
print('cu_k:',cu_k)
print('indices:',indices)
print('last query indices:',last)
for i in range(len(q_lengths)):
    print('req',i,'q slice',slice(cu_q[i],cu_q[i+1]),'kv pages',indices[cu_k[i]:cu_k[i+1]])
assert last==[2,4]
