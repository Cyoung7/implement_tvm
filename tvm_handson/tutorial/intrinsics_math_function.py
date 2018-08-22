import tvm
import numpy as np

# direct declare extern math call
n = tvm.var('n')
A = tvm.placeholder((n,), name='A')
B = tvm.compute(A.shape,
                lambda i: tvm.call_pure_extern('float32', '__expf', A[i]),
                name='B')
s = tvm.create_schedule(B.op)
num_thread = 64
bx, tx = s[B].split(B.op.axis[0], factor=num_thread)
s[B].bind(bx, tvm.thread_axis('blockIdx.x'))
s[B].bind(tx, tvm.thread_axis('threadIdx.x'))
f = tvm.build(s, [A, B], 'cuda', name='my_exp')
# print(f.imported_modules[0].get_source())

# unified intrinsic call
n = tvm.var('n')
A = tvm.placeholder((n,), name='A')
B = tvm.compute(A.shape, lambda i: tvm.exp(A[i]), name='B')
s = tvm.create_schedule(B.op)
num_thread = 64
bx, tx = s[B].split(B.op.axis[0], factor=num_thread)
s[B].bind(bx, tvm.thread_axis('blockIdx.x'))
s[B].bind(tx, tvm.thread_axis('threadIdx.x'))
f_cuda = tvm.build(s, [A, B], 'cuda', name='my_exp')
print(f_cuda.imported_modules[0].get_source())

fopencl = tvm.build(s, [A, B], "opencl", name="myexp")


# print(fopencl.imported_modules[0].get_source())


def my_cuda_math_rule(op):
    """Customized CUDA intrinsic lowering rule"""
    assert isinstance(op, tvm.expr.Call)
    if op.dtype == "float32":
        # call float function
        return tvm.call_pure_extern("float32", "%sf" % op.name, op.args[0])
    elif op.dtype == "float64":
        # call double function
        return tvm.call_pure_extern("float32", op.name, op.args[0])
    else:
        # cannot do translation, return self.
        return op


# 调用cuda的tvm.exp会采用此计算规则
tvm.register_intrin_rule("cuda", "exp", my_cuda_math_rule, override=True)
fcuda = tvm.build(s, [A, B], "cuda", name="my_exp")
print(fcuda.imported_modules[0].get_source())


# 添加自己的内建函数
def my_log(x):
    # 通过内建函数构建表达式 my_log:内建函数名
    return tvm.call_intrin(x.dtype, 'my_log', x)


# 内建函数的计算规则
def my_cuda_log_rule(op):
    if op.dtype == "float32":
        # logf:外部函数名,应该是cuda的内部函数
        return tvm.call_pure_extern("float32", "logf", op.args[0])
    elif op.dtype == "float64":
        return tvm.call_pure_extern("float64", "log", op.args[0])
    else:
        return op


tvm.register_intrin_rule('cuda', 'my_log', my_cuda_log_rule, override=True)

n = tvm.var("n")
A = tvm.placeholder((n,), dtype='float64', name='A')
B = tvm.compute(A.shape, lambda i: my_log(A[i]), name='B')
s = tvm.create_schedule(B.op)
num_thread = 64
bx, tx = s[B].split(B.op.axis[0], factor=num_thread)
s[B].bind(bx, tvm.thread_axis("blockIdx.x"))
s[B].bind(tx, tvm.thread_axis("threadIdx.x"))
fcuda = tvm.build(s, [A, B], "cuda", name="mylog")
print(fcuda.imported_modules[0].get_source())
