class Pool:
    def __init__(self,size): self.blocks=[['free',0,size]]
    def allocate(self,size):
        for i,(state,start,length) in enumerate(self.blocks):
            if state=='free' and length>=size:
                self.blocks[i]=['used',start,size]
                if length>size: self.blocks.insert(i+1,['free',start+size,length-size])
                return start
        raise MemoryError(f'cannot allocate {size}; blocks={self.blocks}')
    def free(self,start):
        for block in self.blocks:
            if block[1]==start and block[0]=='used': block[0]='free'; break
        merged=[]
        for b in self.blocks:
            if merged and b[0]=='free' and merged[-1][0]=='free': merged[-1][2]+=b[2]
            else: merged.append(b)
        self.blocks=merged
    def __repr__(self): return repr(self.blocks)

p=Pool(100); a=p.allocate(20); b=p.allocate(30); c=p.allocate(25); d=p.allocate(15)
print('allocated',p); p.free(b); p.free(d); print('free total 45 but split:',p)
try: p.allocate(40)
except MemoryError as e: print('fragmentation example:',e)
p.free(c); print('after adjacent merge:',p); print('allocate 40 at',p.allocate(40),p)
