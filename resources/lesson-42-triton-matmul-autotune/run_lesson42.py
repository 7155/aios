import numpy as np

def blocked(A,B,BM,BN,BK):
    M,K=A.shape; _,N=B.shape; C=np.zeros((M,N),dtype=np.float64); loads=0
    for m in range(0,M,BM):
        for n in range(0,N,BN):
            acc=np.zeros((min(BM,M-m),min(BN,N-n)))
            for k in range(0,K,BK):
                a=A[m:m+BM,k:k+BK]; b=B[k:k+BK,n:n+BN]
                acc += a@b; loads += 2
            C[m:m+BM,n:n+BN]=acc
    return C,loads

rng=np.random.default_rng(0); A=rng.normal(size=(9,7)); B=rng.normal(size=(7,11)); ref=A@B
for tile in ((2,2,2),(4,4,4),(8,8,4)):
    C,loads=blocked(A,B,*tile)
    print(tile,'loads=',loads,'max error=',np.max(np.abs(C-ref)))
