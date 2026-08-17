#!/usr/bin/env python3
class Pool:
    def __init__(self,n): self.free=list(range(n))
    def alloc(self,n):
        result=self.free[:n]; self.free=self.free[n:]; return result
    def release(self,x): self.free.extend(x)

slots=Pool(2); pages=Pool(8)
slot=slots.alloc(1)[0]; owned=pages.alloc(3)
print('allocated slot/page:',slot,owned)
print('capacity:',len(slots.free),len(pages.free))
slots.release([slot])
print('only slot freed -> page leak:',len(slots.free),len(pages.free))
pages.release(owned)
print('both freed:',len(slots.free),len(pages.free))
assert len(slots.free)==2 and len(pages.free)==8
