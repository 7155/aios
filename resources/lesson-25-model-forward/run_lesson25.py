#!/usr/bin/env python3
import math

def norm(x):
    rms=math.sqrt(sum(v*v for v in x)/len(x))
    return [v/rms for v in x]
def add(a,b): return [x+y for x,y in zip(a,b)]

e=[1.0,2.0]
d_attn=[0.5,-0.5]
d_mlp=[0.2,0.3]
explicit_h1=add(e,d_attn)
explicit_h2=add(explicit_h1,d_mlp)

residual=e[:]
hidden=norm(e)
# attention returns d_attn
residual=add(residual,d_attn); hidden=norm(residual)
# mlp returns d_mlp, next fused norm adds it
residual=add(residual,d_mlp); hidden=norm(residual)
print('explicit residual stream:',explicit_h2)
print('fused residual stream:   ',residual)
print('next normalized input:   ',[round(v,6) for v in hidden])
assert all(abs(a-b)<1e-9 for a,b in zip(explicit_h2,residual))
