# -*-coding:utf-8-*-
import numpy as np
import nnvm.compiler
import nnvm.testing
import tvm
from tvm.contrib import graph_runtime

# 定义神经网络
batch_size = 1
num_class = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)
# net:Symbol
net, params = nnvm.testing.resnet.get_workload(num_layers=18, batch_size=batch_size,
                                               image_shape=image_shape)
# 输出网络的定义
print(net.debug_str())

# 编译
opt_level = 3
target = tvm.target.cuda()
# graph:Graph lib:Module params:dict()
with nnvm.compiler.build_config(opt_level=opt_level):
    graph, lib, params = nnvm.compiler.build(net, target, shape={'data': data_shape},
                                             params=params)

# 运行生成的库
ctx = tvm.gpu()
data = np.random.uniform(-1, 1, size=data_shape).astype(np.float32)
# 创建一个module
# module:GraphModule
module = graph_runtime.create(graph, lib, ctx)
module.set_input('data', data)
module.set_input(**params)

module.run()
# out:tvm.NDarray
out = module.get_output(0, tvm.nd.empty(out_shape))
print(out.asnumpy().flatten()[0:10])

# save and load compiled module
from tvm.contrib import util

temp = util.tempdir()

# path_lib = temp.relpath('deploy_lib.so')
# lib.export_library(path_lib)
# with open(temp.relpath('deploy_graph.json'),'w')as fo:
#     fo.write(graph.json())
# with open(temp.relpath('deploy_params.params'), 'wb') as fo:
#     fo.write(nnvm.compiler.save_param_dict(params))
# print(temp.listdir())


lib.export_library('./model/deploy_lib.so')
with open('./model/deploy_graph.json', 'w')as fo:
    fo.write(graph.json())
with open('./model/deploy_params.params', 'wb') as fo:
    fo.write(nnvm.compiler.save_param_dict(params))

loaded_json = open('./model/deploy_graph.json').read()
loaded_lib = tvm.module.load('./model/deploy_lib.so')
load_params = bytearray(open('./model/deploy_params.params', 'rb').read())
input_data = tvm.nd.array(np.random.uniform(size=data_shape).astype(np.float32))
module = graph_runtime.create(loaded_json, loaded_lib, tvm.gpu())
module.load_params(load_params)
module.run(data=input_data)
out = module.get_output(0, out=tvm.nd.empty(out_shape))
print(out.asnumpy().flatten()[0:10])
