from math import ceil

def estimate(threads, regs, shared=0, max_threads=2048, max_warps=64,
             max_blocks=32, regs_sm=65536, shared_sm=100*1024):
    warps = ceil(threads/32)
    limits = [max_blocks, max_threads//threads, max_warps//warps,
              regs_sm//(threads*regs)]
    limits.append(max_blocks if shared == 0 else shared_sm//shared)
    blocks = max(0, min(limits))
    active_warps = blocks*warps
    return blocks, active_warps, active_warps/max_warps

for threads in (128,256,512):
    for regs in (32,64,128):
        print(f'threads={threads:3} regs={regs:3} -> blocks,warps,occ=', estimate(threads,regs))
print('\nOccupancy is a resource estimate, not a speed guarantee.')
