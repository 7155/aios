import numpy as np
rng=np.random.default_rng(1)
X=rng.normal(size=(3,8)); W=rng.normal(size=(6,8)); ref=X@W.T

# Column parallel: split output rows
W0,W1=np.split(W,2,axis=0)
Y0,Y1=X@W0.T,X@W1.T
column=np.concatenate([Y0,Y1],axis=-1)

# Row parallel: split input columns and add partial outputs
X0,X1=np.split(X,2,axis=1); R0,R1=np.split(W,2,axis=1)
P0,P1=X0@R0.T,X1@R1.T
row=P0+P1

print('column max error',np.max(np.abs(column-ref)))
print('row/allreduce max error',np.max(np.abs(row-ref)))

def communication_us(bytes_, latency_us=8, bandwidth_GBs=50):
    return latency_us + bytes_/(bandwidth_GBs*1e9)*1e6
for M in (1,8,128):
    bytes_=M*768*2
    print(f'M={M:3} allreduce payload={bytes_:6} bytes rough one-way model={communication_us(bytes_):.3f} us')
