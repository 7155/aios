def stats(M,N,K,bpe=2):
    flops=2*M*N*K
    moved=bpe*(M*K+N*K+M*N)
    return flops, moved, flops/moved

ops=[('QKV',1280,768),('GateUp',4096,768),('Down',768,2048)]
for name,N,K in ops:
    print('\n',name)
    for M in (1,8,128):
        f,b,i=stats(M,N,K)
        print(f'M={M:3} FLOPs={f/1e6:9.2f}M bytes={b/2**20:7.3f}MiB intensity={i:7.2f}')
