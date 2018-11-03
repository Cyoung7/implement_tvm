import tvm
import numpy as np

from tvm.contrib import graph_runtime as runtime
import nnvm.symbol as sym
import nnvm.compiler
from nnvm.testing import utils
import logging


# create a simple network
out_channel = 16
data = sym.Variable(name='data')
simple_net = sym.conv2d(data=data, kernel_size=(3, 3), channels=out_channel,
                        padding=(1, 1), use_bias=True)
simple_net = sym.batch_norm(data=simple_net, )
simple_net = sym.relu(data=simple_net)

batch_size = 1
data_shape = (batch_size, 3, 224, 224)
net, params = utils.create_workload(simple_net, batch_size, data_shape[1:])

# build and run with cuda bankend
logging.basicConfig(level=logging.DEBUG)
target = 'cuda'
graph, lib, params = nnvm.compiler.build(net, target, shape={'data': data_shape},
                                         params=params)

ctx = tvm.context(target, 0)
data = np.random.uniform(-1, 1, size=data_shape).astype(np.float32)
module = runtime.create(graph, lib, ctx=ctx)
module.set_input(**params)
module.set_input('data', data)
module.run()
out_shape = (batch_size, out_channel, 224, 224)
out = module.get_output(0, tvm.nd.empty(out_shape))
out_cuda = out.asnumpy()
print(out_cuda.shape)

# use cudnn for a conv layer
net, params = utils.create_workload(simple_net, batch_size=batch_size,
                                    image_shape=data_shape[1:])

target = 'cuda -libs=cudnn'
graph, lib, params = nnvm.compiler.build(net, target, shape={'data': data_shape},
                                         params=params)
ctx = tvm.context(target, 0)
data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
module = runtime.create(graph, lib, ctx)
module.set_input(**params)
module.set_input('data', data)
module.run()
out_shape = (batch_size, out_channel, 224, 224)
out = module.get_output(0, tvm.nd.empty(out_shape))
out_cudnn = out.asnumpy()
print(out_cudnn.shape)

np.testing.assert_allclose(out_cuda, out_cudnn, rtol=1e-5)
