# tvm

[TOC]

## TVM

### python

#### `/python/tvm/api.py`

```
// tensor计算源语具体执行函数
tvm.compute() --> src/api/api_lang.cc(TVM_REGISTER_API("_TensorComputeOp"),TVM_REGISTER_API("_ComputeOp"))
```

```
// tvm层 变量声明
tvm.var() --> src/api/api_ir.cc(TVM_REGISTER_API("_Var"))
```

```
// tvm层 占位符声明
tvm.placeholder() --> src/api/api_lang.cc(TVM_REGISTER_API("_Placeholder"))
```

#### `/python/tvm/build_module.py`

```
//从LowerFunc build成TVM底层runtime的Module
tvm.build() --> codegen.build_module（） --> src/api/api_codegen.cc(TVM_REGISTER_API("codegen._Build")) --> runtime::Module Build()
```

```
/python/tvm/build_module.py
//将schedule lower到 LowerFunc
tvm.lower(sch) --> 
{
// 推断循环边界及缓冲区大小等等
bounds = schedule.InferBound(sch) --> src/schedule/bound.cc(Map<IterVar, Range> InferBound)
//表示初始的循环嵌套结构，如果已将reorder,split等应用，则循环嵌套已反映其更改
stmt = schedule.ScheduleOps(sch, bounds) --> src/schedule/schedule_ops.cc(Stmt ScheduleOps())
ir_pass.StorageFlatten --> src/api/api_pass.cc(TVM_REGISTER_API("ir_pass.StorageFlatten")),
ir_pass.MakeAPI(stmt) --> src/api/api_pass.cc(REGISTER_PASS5(MakeAPI))
}
```

#### `/python/tvm/Container.py`:

Container data structures used in TVM DSL，传给runtime(Module)的计算流图, `TVM`层都是用`LoweredFunc` 进行代码生成，pass优化等

```
/python/tvm/container.py
class LoweredFunc(NodeBase) --> include/tvm/lowered_func.h(class LoweredFunc : public FunctionRef)
```

#### `/python/tvm/schedule.py`

对TVM做计算调度

```
def create_schedule() --> src/api/api_lang.cc(TVM_REGISTER_API("_CreateSchedule"))
```

```
//
@register_node
class Schedule(NodeBase):
```

```
//
@register_node
class Stage(NodeBase):

def fuse(self, *args): --> src/api/api_lang.cc(TVM_REGISTER_API("_StageFuse"))
```

```
//
@register_node
class Buffer(NodeBase)
```

#### `/python/tvm/expr.py`

tvm层 Expressions的所有基类

```

```



### pass

```

```

### codegen

#### `/src/codegen/codegen.cc` 

```
// 创建一个特定后端的runtime，供上层的tvm.build()调用
runtime::Module Build() --> runtime::Registry::Get(build_f_name)(得到特定后端的runtime)
```

```
// 各个不同后端的注册方式
{
src/codegen/opt/build_cuda_on.cc(TVM_REGISTER_API("codegen.build_cuda")),
src/codegen/llvm/llvm_module.cc(TVM_REGISTER_API("codegen.build_llvm"))...
}

// src/runtime/regitry.cc
// 具体的注册类
Registry& Registry::Register()
```

#### cuda target

`src/codegen/opt/build_cuda_on.cc`

```

```

`src/codegen/codege_cuda.cc`

```

```

#### llvm target

`src/codegen/llvm/llvm_module.cc`

```

```





### runtime

#### `/src/runtime/graph/graph_runtime.cc`

```
// 创建一个graph runtime(Wrapper runtime module)
// tvm.build()根据已有的计算流图来创建一个特定后端的runtime(Module)，create可以创建以空的runtime
// python/contrib/graph_runtime.py
def create() --> class GraphModule() --> src/runtime/graph/graph_runtime.cc(TVM_REGISTER_GLOBAL("tvm.graph_runtime.create"))
```

#### `/src/runtime/module.cc`

TVM runtime的计算图(整个计算流程)都存储在 class Module(include/tvm/runtime/module.h)中

```
// 创建一个runtime的功能函数
TVM_REGISTER_GLOBAL("module._LoadFromFile")...
```

**runtime各个不同的后端target是提供在runtime阶段，各个后端的一些功能函数，来达到创建对应后端runtime(Module)的目的**.

例：

```
//src/runtime/opengl/opengl_module.cc 
// 提供loadfile_gl等功能函数，用这些后端创建runtime
{
TVM_REGISTER_GLOBAL("module.loadfile_gl")，
TVM_REGISTER_GLOBAL("module.loadfile_glbin")，
TVM_REGISTER_GLOBAL("module.loadbinary_opengl")...
}
//其他后端同理
```

但是，tvm.build() 也可以针对特定后端创建runtime(Module)

## Relay

Expr,Call,Function，Module 等等是relay层的ir，全部定义在  /src/relay/ir/*cc :

例：

```
TVM_REGISTER_API("relay._make.Var")
TVM_REGISTER_API("relay._make.GlobalVar")
TVM_REGISTER_API("relay._make.Function")
//等等 均定义在 /src/relay/ir/expr.cc 中
```

### python

#### `python/tvm/relay/build_module.py`

`relay.build()`

```
// python/tvm/relay/build_module.py
// 先将relay层的Func lower到TVM层的LowerFunc，在调用 tvm.build()生成runtime(Module)
relay.build() --> 
{
graph_gen.codegen()(python/tvm/relay/backend/graph_runtime_codegen.py/def codegen),
_tvm_build_module(tvm.build())
}
```

`create_executor()`

```
// 从relay层来创建一个relay的interpreter或者graph runtime 
// python/tvm/relay/build_module.py
def create_executor() -->
{
_interpreter.Interpreter --> _backend.CreateInterpreter --> src/relay/backend/interpreter.cc(TVM_REGISTER_API("relay.backend.CreateInterpreter")),
GraphExecutor() --> def _make_executor() --> _graph_rt.create() --> class GraphModule() --> get_global_func("tvm.graph_runtime.create") --> src/runtime/graph/graph_runtime.cc(TVM_REGISTER_GLOBAL("tvm.graph_runtime.create"))
}
```

`optimize()` :优化relay层的ir

```
// 对relay层的ir做优化(对Func优化)
// python/tvm/relay/build_module.py
def optimize() -->
{ _bind_params_by_name() --> relay/expr.py(def bind()) --> src/relay/ir/expr_functor.cc(TVM_REGISTER_API("relay._expr.Bind")),
 ir_pass.infer_type --> src/relay/pass/type_infer.cc(TVM_REGISTER_API("relay._ir_pass.infer_type"))
}
```

### frontend

**relay 现阶段支持的所有前端，看源码建议从此切入**

`python/tvm/relay/frontend/*.py`

代码简单，在此忽略

### backend

`python/tvm/relay/backend/graph_runtime_codegen.py`

```
// 从relay层的Func到tvm的LowerFunc
// python/tvm/relay/backend/graph_runtime_codegen.py
def codegen() --> 
{
_backend.GraphPlanMemory --> src/relay/backend/graph_plan_memory.cc(TVM_REGISTER_GLOBAL("relay.backend.GraphPlanMemory")) ,
self.visit() --> def visit_call() --> self.compile_engine.lower()--> src/relay/backend/compile_engine.cc(TVM_REGISTER_GLOBAL("relay.backend._CompileEngineLower")) --> CachedFunc Lower(具体实现在class CompileEngineImpl里)
}
```

### op

**relay层所支持的所有op**

#### `python/tvm/relay/op/op.py`

```
// 根据名字获取一个op
// python/tvm/relay/op/op.py
def get(op_name) --> src/relay/ir/op.cc(TVM_REGISTER_API("relay.op._GetOp"))
```

#### `src/relay/op/*` ：进入topi

注册了所有relay支持的op，绝大数op通过调用topi的具体实现完成计算

例： 

c++调c++的op实现

```
// src/relay/op/nn/nn.cc
// 注册
RELAY_REGISTER_OP("nn.leaky_relu")
// 调用的具体实现
topi::leaky_relu
```

c++调python的op实现

```
// src/relay/op/nn/convolution.cc
// 注册
RELAY_REGISTER_OP("nn.conv2d")

// python端注册compute
// python/tvm/relay/op/nn/_nn.py
@reg.register_compute("nn.conv2d")
def compute_conv2d(attrs, inputs, out_type, target):
	//调用topi python端的具体实现
	topi.nn.conv2d()

// topi实现
// topi/python/topi/nn/conv2d.py
@tvm.target.generic_func
def conv2d(input, filter, strides, padding, dilation, layout='NCHW', out_dtype=None)
```



## AutoTVM

## TOPI

**使用tvm::compute() 原语实现了所有tvm中的计算op ,供relay使用**

```

```



## CLASS:核心类

### class Schedule

tvm计算调度类

```
// tvm/python/schedule.py
class Schedule(NodeBase)
// tvm/include/tvm/schedule.h
class Schedule : public NodeRef
// tvm/src/schedule/schedule_lang.cc:实现
```

### class PackedFunc

tvm中所有函数的包装类，使得c++实现的函数在任何前端语言均可调用执行

```
// tvm/include/tvm/runtime/packed_func.h
class PackedFunc
```

### class LoweredFunc

LoweredFunc 代表function lowering之后的结构，codegen之前的最后IR表征

```
// tvm/include/tvm/lowered_func.h
class LoweredFunc : public FunctionRef
```

