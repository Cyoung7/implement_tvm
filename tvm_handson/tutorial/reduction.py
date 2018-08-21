import tvm
import numpy as np

n = tvm.var("n")
m = tvm.var("m")
A = tvm.placeholder((n, m), name='A')
# IterVar:一个迭代变量
k = tvm.reduce_axis((0, m), name='k')
# sum 计算k声明范围内所有值的和
B = tvm.compute((n,), lambda i: tvm.sum(A[i, k], axis=k), name='B')

s = tvm.create_schedule(B.op)
# print(tvm.lower(s, [A, B], simple_mode=True))
# B.op.reduce_axis:B的缩减轴,行方向
# ko, ki = s[B].split(B.op.reduce_axis[0], factor=16)

# B.op.axis:B自身的轴方向,B一维,所以就是列方向
# xo,xi:IterVar
# xo, xi:分别为外部迭代,内部迭代
xo, xi = s[B].split(B.op.axis[0], factor=32)
# print(tvm.lower(s, [A, B], simple_mode=True))

# tvm.schedule.Stage.split()
# bind the rows of B to GPU threads.
s[B].bind(xo, tvm.thread_axis('blockIdx.x'))
s[B].bind(xi, tvm.thread_axis("threadIdx.x"))
# print(tvm.lower(s, [A, B], simple_mode=True))

# Reduction Factoring and Parallelization
s = tvm.create_schedule(B.op)
ko, ki = s[B].split(B.op.reduce_axis[0], factor=16)
# print('yuan:')
# print(tvm.lower(s, [A, B], simple_mode=True))
# 被分解的维度编程了BF的第一维
BF = s.rfactor(B, ki)
# print('xian:')
# print(tvm.lower(s, [A, B], simple_mode=True))
# print(s[B].op.body)

# Cross Thread Reduction
xo, xi = s[B].split(s[B].op.axis[0], factor=32)
s[B].bind(xo, tvm.thread_axis('blockIdx.x'))
s[B].bind(xi, tvm.thread_axis('threadIdx.y'))
tx = tvm.thread_axis('threadIdx.x')
s[B].bind(s[B].op.reduce_axis[0], tx)

# 这个有点没懂
s[BF].compute_at(s[B], s[B].op.reduce_axis[0])
# 设置断言
s[B].set_store_predicate(tx.var.equal(0))
fcuda = tvm.build(s, [A, B], 'cuda')
print('cuda:')
print(fcuda.imported_modules[0].get_source())

nn = 128
ctx = tvm.gpu(0)
a = tvm.nd.array(np.random.uniform(size=(nn, nn)).astype(A.dtype), ctx)
b = tvm.nd.array(np.zeros(nn, dtype=B.dtype), ctx)
fcuda(a, b)

np.testing.assert_allclose(
    b.asnumpy(), np.sum(a.asnumpy(), axis=1), rtol=1e-4)

# Describe Convolution via 2D Reduction
n = tvm.var('n')
Input = tvm.placeholder((n, n), name='Input')
Filter = tvm.placeholder((3, 3), name='Filter')
di = tvm.reduce_axis((0, 3), name='di')
dj = tvm.reduce_axis((0, 3), name='dj')
output = tvm.compute((n - 2, n - 2),
                     lambda i, j: tvm.sum(Input[i + di, j + dj] * Filter[di, dj],
                                          axis=[di, dj]), name='output')
s = tvm.create_schedule(output.op)
print(tvm.lower(s, [Input, Filter, output], simple_mode=True))

# Define General Commutative Reduction Operation
n = tvm.var('n')
m = tvm.var('m')
product = tvm.comm_reducer(lambda x, y: x * y,
                           lambda t: tvm.const(1, dtype=t), name='product')

A = tvm.placeholder((n, m), name='A')
k = tvm.reduce_axis((0, m), name='k')
B = tvm.compute((n,), lambda i: product(A[i, k], axis=k), name='B')
