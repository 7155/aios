import math

def naive(scores, values):
    m=max(scores); p=[math.exp(x-m) for x in scores]; z=sum(p)
    return [sum(w*v[j] for w,v in zip(p,values))/z for j in range(len(values[0]))]

def online(scores, values, block):
    m=float('-inf'); l=0.0; out=[0.0]*len(values[0])
    for start in range(0,len(scores),block):
        s=scores[start:start+block]; v=values[start:start+block]
        new_m=max(m,max(s)); alpha=0.0 if m==float('-inf') else math.exp(m-new_m)
        p=[math.exp(x-new_m) for x in s]
        out=[alpha*x for x in out]
        for w,row in zip(p,v):
            for j,x in enumerate(row): out[j]+=w*x
        l=alpha*l+sum(p); m=new_m
        print('block',start,'m=',round(m,5),'l=',round(l,5),'alpha=',round(alpha,5))
    return [x/l for x in out]

scores=[1.0,2.0,5.0,-1.0]
values=[[1,0],[0,1],[2,2],[-1,3]]
print('naive :',naive(scores,values))
for block in (1,2,3):
    print('\nblock size',block,online(scores,values,block))
