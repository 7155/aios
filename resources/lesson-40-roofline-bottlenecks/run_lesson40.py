def classify(flops, bytes_moved, peak=100e12, bandwidth=1e12, parallel=True):
    intensity=flops/bytes_moved
    balance=peak/bandwidth
    if not parallel: kind='latency candidate'
    elif intensity < balance: kind='memory candidate'
    else: kind='compute candidate'
    return intensity,balance,kind

cases=[('decode qkv',15.7e6,2.0e6,True),('prefill qkv',251.7e6,2.36e6,True),
       ('tiny launch',1e6,1e3,False),('rmsnorm',1e6,2e6,True)]
for name,f,b,p in cases:
    print(name, classify(f,b,parallel=p))

def amdahl(fraction,speedup): return 1/((1-fraction)+fraction/speedup)
print('5% kernel accelerated 2x -> total',amdahl(.05,2))
print('60% kernel accelerated 2x -> total',amdahl(.60,2))
