#!/usr/bin/env python3
from contextlib import contextmanager

class Context:
    def __init__(self): self.batch=None
    @contextmanager
    def forward_batch(self,batch):
        assert self.batch is None, 'nested batch forbidden'
        try:
            self.batch=batch; yield
        finally:
            self.batch=None
ctx=Context()
with ctx.forward_batch('A'):
    print('inside:',ctx.batch)
print('after:',ctx.batch)
try:
    with ctx.forward_batch('B'):
        raise RuntimeError('model failed')
except RuntimeError:
    pass
print('after exception:',ctx.batch)
try:
    with ctx.forward_batch('outer'):
        with ctx.forward_batch('inner'): pass
except AssertionError as e:
    print('nested rejected:',e)
