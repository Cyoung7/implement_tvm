import tvm
import numpy as np

# 同一个计算有多种不同的计算方式,更会有不同的性能
# Schedule来决定如何计算,schedule是一组计算转换,用于转化程序中的循环计算
# schedule 是由一组opts组成
# 默认情况下,以行优先的串行方式计算
n = tvm.var('n')
m = tvm.var('m')
A = tvm.placeholder((m, n), name='A')
B = tvm.placeholder((m, n), name='B')
C = tvm.compute((m, n), lambda i, j: A[i, j] * B[i, j], name='C')

s = tvm.create_schedule([C.op])
# lower会将计算从定义转换为真正的可调用函数。 使用参数`simple_mode = True`，
# 它将返回一个可读的C like语句，我们在此处使用它来打印计划结果。
# print(tvm.lower(s, [A, B, C], simple_mode=True))

# 一个schedule由多个stage组成,一个stage代表一个opt
# 每个stage提供多种方法

# split
# 将特定的一维拆成两维
A = tvm.placeholder((m,), name='A')
B = tvm.compute((m,), lambda i: A[i] * 2, name='B')
s = tvm.create_schedule(B.op)
xo, xi = s[B].split(B.op.axis[0], factor=32)
# print(tvm.lower(s, [A, B], simple_mode=True))

s = tvm.create_schedule(B.op)
bx, tx = s[B].split(B.op.axis[0], nparts=32)
# print(tvm.lower(s, [A, B], simple_mode=True))

# tile
# 在两个维度上执行平铺
A = tvm.placeholder((m, n), name='A')
B = tvm.compute((m, n), lambda i, j: A[i, j] * 2, name='B')
s = tvm.create_schedule(B.op)
# tvm.schedule.Stage.tile()
xo, yo, xi, yi = s[B].tile(B.op.axis[0], B.op.axis[1], x_factor=10, y_factor=5)
# print(tvm.lower(s, [A, B], simple_mode=True))

# fuse
# 可以融合一个计算的两个连续轴
s = tvm.create_schedule(B.op)
xo, yo, xi, yi = s[B].tile(B.op.axis[0], B.op.axis[1], x_factor=10, y_factor=5)
fused = s[B].fuse(xi, yi)
# print(tvm.lower(s, [A, B], simple_mode=True))

# reorder
# 重排axis的顺序
s = tvm.create_schedule(B.op)
xo, yo, xi, yi = s[B].tile(B.op.axis[0], B.op.axis[1], x_factor=10, y_factor=5)
s[B].reorder(xi, yo, xo, yi)
# 注意只能融合两个联系的轴
fused = s[B].fuse(xo, yi)
# print(tvm.lower(s, [A, B], simple_mode=True))

# bind
# 在gpu程序中使用,将一个轴与一个gpu线程轴进行绑定
s = tvm.create_schedule(B.op)
s[B].bind(B.op.axis[0], tvm.thread_axis('blockIdx.x'))
s[B].bind(B.op.axis[1], tvm.thread_axis('threadIdx.x'))
# print(tvm.lower(s, [A, B], simple_mode=True))

# compute_at
# 一个schedule有多个opts组成,默认情况下,tvm将从最底层的opt开始计算
A = tvm.placeholder((m, n), name='A')
B = tvm.compute((m, n), lambda i, j: A[i, j] + 1, name='B')
C = tvm.compute((m, n), lambda i, j: B[i, j] * 2, name='C')

s = tvm.create_schedule(C.op)
# 从代码来看是将B全部计算完毕才开始计算C,也就是每个opt单独计算
print(tvm.lower(s, [A, B, C], simple_mode=True))

# 可以将计算B这一步变到计算C的inner维上去
s = tvm.create_schedule(C.op)
s[B].compute_at(s[C], C.op.axis[0])
print(tvm.lower(s, [A, B, C], simple_mode=True))

# compute_inline
# A将一个stage设为内联,然后stage的计算体将会被展开,会被插入到tensor被需要的地址
s = tvm.create_schedule(C.op)
s[B].compute_inline()
# print(tvm.lower(s, [A, B, C], simple_mode=True))

# compute_root
# 会将一个stage的计算移到最低层
s = tvm.create_schedule(C.op)
s[B].compute_at(s[C], C.op.axis[0])
# 可以理解成还原到默认计算顺序
s[B].compute_root()
print(tvm.lower(s, [A, B, C], simple_mode=True))
