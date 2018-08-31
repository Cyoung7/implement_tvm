# nnvm workflow
import nnvm.compiler
import nnvm.symbol as sym

x = sym.Variable('x')
y = sym.Variable('y')
z = sym.elemwise_add(x, sym.sqrt(y))
compute_graph = nnvm.graph.create(z)
print("--------compute graph---------")
print(compute_graph.ir())

shape = (4,)
# 输入是一个z:nnvm.symbol,输出deply_graph:nnvm.symbol,这是经过nnvm优化之后的图
# 也就是说图优化这个步骤实在nnvm里面进行的
deploy_graph, lib, params = nnvm.compiler.build(z, target='cuda',
                                                shape={'x': shape}, dtype='float32')

print('---------deploy graph----------')
print(deploy_graph.ir())

# lib:host module lib.imported_modules[0]:a device module
# 现在的疑问是:这个fuse opt是 nnvm 中手写的优化算子,还是自动生成的优化算子?
print('---------deploy lib------------')
print(lib.imported_modules[0].get_source())

# deploy and run
import tvm
import numpy as np
from tvm.contrib import graph_runtime
module = graph_runtime.create(deploy_graph,lib,ctx=tvm.gpu())

x_np = np.array([1, 2, 3, 4]).astype("float32")
y_np = np.array([4, 4, 4, 4]).astype("float32")
# set input to the graph module
module.set_input(x=x_np, y=y_np)
module.run()
out = module.get_output(0,tvm.nd.empty(shape))
print(out.asnumpy())

# provide model parameters
d_graph,lib,params = nnvm.compiler.build(z)