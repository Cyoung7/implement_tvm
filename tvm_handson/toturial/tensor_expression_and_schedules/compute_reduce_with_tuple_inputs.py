# -*-coding:utf-8-*-
import tvm
import numpy as np

n = tvm.var('n', dtype=tvm.int32)
m = tvm.var('m', dtype=tvm.int32)
A0 = tvm.placeholder((m, n), name='A0')
A1 = tvm.placeholder((m, n), name='A1')
# B0,B1是同一个操作输出的结果
B0, B1 = tvm.compute((m, n), lambda i, j: (A0[i, j] + 2, A1[i, j] * 3), name='B')

# 生成中间代码
# 参数为B0.op or B1.op,两者等价
s = tvm.create_schedule(B1.op)
print(tvm.lower(s, [A0, A1, B0, B1], simple_mode=True))


def fcombine(x, y):
    lhs = tvm.select((x[1] >= y[1]), x[0], y[0])
    rhs = tvm.select((x[1] >= y[1]), x[1], y[1])
    return lhs, rhs


def fidentity(t0, t1):
    return tvm.const(-1, t0), tvm.min_value(t1)


# function
argmax = tvm.comm_reducer(fcombine, fidentity, name='argmax')

# describe the reduction computation
# 看不懂
m = tvm.var('m')
n = tvm.var('n')
idx = tvm.placeholder((m, n), name='idx', dtype=tvm.int32)
val = tvm.placeholder((m, n), name='val', dtype=tvm.int32)

# IterVar
k = tvm.reduce_axis((0, n), name='k')
T0, T1 = tvm.compute((m,), lambda i: argmax((idx[i, k], val[i, k]), axis=k), name='T')

s = tvm.create_schedule(T0.op)
print(tvm.lower(s, [idx, val, T0, T1], simple_mode=True))

# Schedule Operation with Tuple Input
n = tvm.var("n")
m = tvm.var("m")
A0 = tvm.placeholder((m, n), name='A0')
B0, B1 = tvm.compute((m, n), lambda i, j: (A0[i, j] + 2, A0[i, j] * 3), name='B')
A1 = tvm.placeholder((m, n), name='A1')
C = tvm.compute((m, n), lambda i, j: A1[i, j] + B0[i, j], name='C')

s = tvm.create_schedule(C.op)
# 将B0 stage附加到C stage上面，在计算C的过程中先计算B0
s[B0].compute_at(s[C], C.op.axis[0])

print(tvm.lower(s, [A0, A1, C], simple_mode=True))
