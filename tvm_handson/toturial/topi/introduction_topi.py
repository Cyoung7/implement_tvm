import tvm
import topi
import numpy as np

n = tvm.var('n')
m = tvm.var('m')
A = tvm.placeholder((n, m), name='A')
k = tvm.reduce_axis((0, m), 'k')
# 原始的计算方式
B = tvm.compute((n,), lambda i: tvm.sum(A[i, k], axis=k), name='B')
s = tvm.create_schedule(B.op)
# print(tvm.lower(s, [A], simple_mode=True))

# 更加现代的计算
C = topi.sum(A, axis=1)
ts = tvm.create_schedule(C.op)
# print(tvm.lower(ts, [A], simple_mode=True))

# numpy style
x, y = 100, 10
a = tvm.placeholder((x, y, y), name='a')
b = tvm.placeholder((y, y), name='b')
c = a + b
d = a * b
e = topi.elemwise_sum([c, d])
f = e / 2.0
g = topi.sum(f)
with tvm.target.cuda():
    # 这个有点问题
    sg = topi.generic.schedule_reduce(g)
    # print(tvm.lower(sg, [a, b], simple_mode=True))
# print(sg.stages)

func = tvm.build(sg, [a, b, g], 'cuda')
ctx = tvm.gpu(0)
a_np = np.random.uniform(size=(x, y, y)).astype(a.dtype)
b_np = np.random.uniform(size=(y, y)).astype(b.dtype)
g_np = np.sum(np.add(a_np + b_np, a_np * b_np) / 2.0)

a_nd = tvm.nd.array(a_np, ctx)
b_nd = tvm.nd.array(b_np, ctx)
g_nd = tvm.nd.array(np.zeros(g_np.shape, dtype=g_np.dtype), ctx)
func(a_nd, b_nd, g_nd)
np.testing.assert_allclose(g_nd.asnumpy(), g_np, rtol=1e-5)

# common neural nets
tarray = tvm.placeholder((512, 512), name="tarray")
softmax_topi = topi.nn.softmax(tarray)
with tvm.target.cuda():
    sst = topi.generic.schedule_softmax(softmax_topi)
    # print(tvm.lower(sst, [tarray], simple_mode=True))

# fusing conv
# fuse topi.nn.conv2d and topi.nn.relu together
# -----------------------old vision---------------------------
# data = tvm.placeholder((1, 3, 224, 224))
# kernel = tvm.placeholder((10, 3, 5, 5))
# conv = topi.nn.conv2d(data, kernel, strides=1, padding=2)
# out = topi.nn.relu(conv)
# with tvm.target.create('cuda'):
#     # 难道每种操作都有一个专门的调度
#     sconv = topi.generic.nn.schedule_conv2d_nchw(out)
#     print(tvm.lower(sconv, [data, kernel], simple_mode=True))
# ------------------------------------------------------------

data = tvm.placeholder((1, 3, 224, 224))
kernel = tvm.placeholder((10, 3, 5, 5))
with tvm.target.create("cuda"):
    conv = topi.nn.conv2d(data, kernel, strides=1, padding=2, dilation=1)
    out = topi.nn.relu(conv)
    sconv = topi.generic.nn.schedule_conv2d_nchw(out)
    print(tvm.lower(sconv, [data, kernel], simple_mode=True))