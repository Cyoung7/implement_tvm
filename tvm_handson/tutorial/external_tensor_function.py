# -*-coding:utf-8-*-
import tvm
import numpy as np
from tvm.contrib import cblas

n = 1024
l = 128
m = 235
# var
bias = tvm.var('bias', dtype=tvm.float32)
# Tensor
A = tvm.placeholder((n, l), name='A')
B = tvm.placeholder((l, m), name='B')
# 添加一个额外的函数，这个函数更多可以是自定义的
# Tensor
# C = tvm.extern((n, m), [A, B],
#                lambda ins, outs: tvm.call_packed('tvm.contrib.cblas.matmul',
#                                                  ins[0], ins[1], outs[0], False, False),
#                name='C')
# # Tensor
# D = tvm.compute(C.shape, lambda i, j: C[i, j] + bias, name='D')
# s = tvm.create_schedule(D.op)

# 另外一种写法
C = cblas.matmul(A, B)
D = tvm.compute(C.shape, lambda i, j: C[i, j] + bias, name='D')
s = tvm.create_schedule(D.op)

# verify the result
ctx = tvm.cpu(0)
# Module
f = tvm.build(s, [A, B, D, bias], 'llvm')
# 为什么不行
# f = tvm.build(s, [A, B, bias,D], 'llvm')
a = tvm.nd.array(np.random.uniform(size=(n, l)).astype(A.dtype), ctx=ctx)
b = tvm.nd.array(np.random.uniform(size=(l, m)).astype(B.dtype), ctx=ctx)
d = tvm.nd.array(np.zeros(shape=(n, m), dtype=D.dtype), ctx=ctx)
bb = 10.0
f(a, b, d, bb)
np.testing.assert_allclose(
    d.asnumpy(), np.dot(a.asnumpy(), b.asnumpy()) + 10, rtol=1e-5)
print(d.shape)


@tvm.register_func('tvm.contrib.my_tvm_add_one')
def my_tvm_add_one(x, y):
    print('my tvm add one signatures :%s, %s' % (type(x), type(y)))
    tvm.nd.array(x.asnumpy() + 1).copyto(y)


A = tvm.placeholder((n,), name='A')
B = tvm.extern(A.shape, [A],
               lambda ins, outs:
               tvm.call_packed('tvm.contrib.my_tvm_add_one', ins[0], outs[0]),
               name='C')
s = tvm.create_schedule(B.op)
f = tvm.build(s, [A, B], 'llvm')
a = tvm.nd.array(np.random.uniform(size=(n,)).astype(A.dtype), ctx=ctx)
b = tvm.nd.array(np.random.uniform(size=(n,)).astype(B.dtype), ctx=ctx)
f(a, b)
np.testing.assert_allclose(b.asnumpy(), a.asnumpy() + 1, rtol=1e-5)
print(b.shape)
