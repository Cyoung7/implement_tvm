# -*-coding:utf-8-*-
import tvm
import numpy as np

tgt_host = 'llvm'
tgt = 'cuda'

n = tvm.var('n')
A = tvm.placeholder((n,), name='A')
B = tvm.placeholder((n,), name='B')
# Tensor
C = tvm.compute(A.shape, lambda i: A[i] + B[i], name='C')
print(type(C))

# s:Schedule
s = tvm.create_schedule(C.op)
# bx:IterVar tx:IterVal s[C]:Stage
bx, tx = s[C].split(C.op.axis[0], factor=64)

if tgt == 'cuda':
    s[C].bind(bx, tvm.thread_axis('blockIdx.x'))
    s[C].bind(tx, tvm.thread_axis('threadIdx.x'))

# Module
fadd = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name='my_add')

# run the function
ctx = tvm.context(tgt, 0)
n = 1024
a = tvm.nd.array(np.random.uniform(-1, 1, size=n).astype(A.dtype), ctx)
b = tvm.nd.array(np.random.uniform(-1, 1, size=n).astype(B.dtype), ctx)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
fadd(a, b, c)

np.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

if tgt == 'cuda':
    dev_module = fadd.imported_modules[0]
    print('-----GPU code-----')
    print(dev_module.get_source())
else:
    print(fadd.get_source())

# save compiling module
from tvm.contrib import cc

fadd.save('./model/myadd.o')
if tgt == 'cuda':
    fadd.imported_modules[0].save('./model/myadd.ptx')
cc.create_shared('./model/myadd.so', ['./model/myadd.o'])

# load compiling module
fadd_load = tvm.module.load('./model/myadd.so')
if tgt == 'cuda':
    fadd_dev = tvm.module.load('./model/myadd.ptx')
    fadd_load.import_module(fadd_dev)
fadd_load(a, b, c)
np.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

# Pack Everything into One Library
fadd.export_library('./model/myadd_pack.so')
fadd_load2 = tvm.module.load('./model/myadd_pack.so')
fadd_load2(a, b, c)
np.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

# tgt = 'opencl'
if tgt == "opencl":
    fadd_cl = tvm.build(s, [A, B, C], "opencl", name="myadd")
    print("------opencl code------")
    print(fadd_cl.imported_modules[0].get_source())
    ctx = tvm.cl(0)
    n = 1024
    a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
    fadd_cl(a, b, c)
    np.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())
