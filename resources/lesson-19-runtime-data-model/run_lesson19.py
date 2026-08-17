#!/usr/bin/env python3
class Req:
    def __init__(self, prompt, output):
        self.cached_len = 0
        self.device_len = prompt
        self.max_device_len = prompt + output
    @property
    def extend_len(self): return self.device_len - self.cached_len
    @property
    def remain_len(self): return self.max_device_len - self.device_len
    def complete_one(self):
        self.cached_len = self.device_len
        self.device_len += 1

def show(label, req):
    print(label, dict(C=req.cached_len, D=req.device_len, M=req.max_device_len,
                      E=req.extend_len, R=req.remain_len))

req = Req(4, 3)
show('created', req)
for name in ['prefill sampled x1', 'decode sampled x2', 'decode sampled x3']:
    req.complete_one(); show(name, req)

print('\nwrong ordering would claim sampled token is already cached:')
D = 4
D += 1
C = D
print(dict(C=C, D=D, wrong_extend=D-C))
