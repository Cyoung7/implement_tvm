import tvm
import numpy as np

m = tvm.var('m')
n = tvm.var('n')
X = tvm.placeholder((m, n), name='X')
s_state = tvm.placeholder((m, n))
s_init = tvm.compute((1, n), lambda _, i: X[0, i])
s_update = tvm.compute((m, n), lambda t, i: s_state[t - 1, i] + X[t, i])
s_scan = tvm.scan(s_init, s_update, s_state, inputs=[X])

# Schedule the Scan Cell
s = tvm.create_schedule(s_scan.op)
num_thread = 256
block_x = tvm.thread_axis('blockIdx.x')
thread_x = tvm.thread_axis('threadIdx.x')
xo, xi = s[s_init].split(s_init.op.axis[1], factor=num_thread)
s[s_init].bind(xo, block_x)
s[s_init].bind(xi, thread_x)
xo, xi = s[s_update].split(s_update.op.axis[1], factor=num_thread)
s[s_update].bind(xo, block_x)
s[s_update].bind(xi, thread_x)
print(tvm.lower(s, [X, s_scan], simple_mode=True))

# Build and Verify
f_scan = tvm.build(s, [X, s_scan], 'cuda', name='my_scan')
ctx = tvm.gpu(0)
n = 1024
m = 10
a_np = np.random.uniform(size=(m, n)).astype(s_scan.dtype)
a = tvm.nd.array(a_np, ctx=ctx)
b = tvm.nd.array(np.zeros((m, n), dtype=s_scan.dtype), ctx=ctx)
f_scan(a, b)
np.testing.assert_allclose(b.asnumpy(), np.cumsum(a_np, axis=0))

# Multi-Stage Scan Cell
m = tvm.var("m")
n = tvm.var("n")
X = tvm.placeholder((m, n), name="X")
s_state = tvm.placeholder((m, n))
s_init = tvm.compute((1, n), lambda _, i: X[0, i])
s_update_s1 = tvm.compute((m, n), lambda t, i: s_state[t - 1, i] * 2, name='s1')
s_update_s2 = tvm.compute((m, n), lambda t, i: s_update_s1[t, i] + X[t, i], name='s2')
s_scan = tvm.scan(s_init, s_update_s2, s_state, inputs=[X])

s = tvm.create_schedule(s_scan.op)
xo, xi = s[s_update_s2].split(s_update_s2.op.axis[1], factor=32)
# compute_at没懂
s[s_update_s1].compute_at(s[s_update_s2], xo)
print(tvm.lower(s, [X, s_scan], simple_mode=True))
