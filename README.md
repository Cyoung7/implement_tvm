# implement_tvm

[TOC]

## 1.安装

```shell
git clone --recursive https://github.com/dmlc/tvm
cd tvm
mkdir build
cp cmake/config.cmake build
cd build 
```

安装 llvm

- https://apt.llvm.org/

```
# 安装稳定版本
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key|sudo apt-key add -
sudo apt-get install clang-6.0 lldb-6.0 lld-6.0
```

编辑config.cmake

```shell
vi config.cmake
```

- set(USE_CUDA ON)
- set(USE_LLVM ON)

编译

```shell
cmake ..
make -j6
```

安装python包

```
cd python; /home/cyoung/miniconda3/bin/python setup.py install ; cd ..
cd topi/python; /home/cyoung/miniconda3/bin/python setup.py install; cd ../..
cd nnvm/python; /home/cyoung/miniconda3/bin/python setup.py install; cd ../..
```

local模式(--user)默认安装的路径: /home/cyoung/.miniconda3/lib/python3.6/site-packages/

## 2.踩坑记录

### 1.运行docs.tvm.ai第一个demo报错(NVCC)

Quick Start Tutorial for Compiling Deep Learning Models

```python
 # 本句报错
 graph, lib, params = nnvm.compiler.build(
        net, target, shape={"data": data_shape}, params=params)
        
 FileNotFoundError: [Errno 2] No such file or directory: 'nvcc' 
 # 说明系统找不到nvcc指令
```

nvcc是cuda的一个编译器，实际存在与/usr/local/cuda/bin中，.bashrc中实际已经添加export PATH=$PATH:/usr/local/cuda/bin,同样报错

解决方案:将/usr/local/cuda/bin添加到系统环境变量文件(etc/environment )中，重启！

说明:

- etc/profile: 此文件为系统的每个用户设置环境信息。当用户登录时，该文件被执行一次，并从 /etc/profile.d 目录的配置文件中搜集shell 的设置。一般用于设置所有用户使用的全局变量。
- /etc/bashrc: 当 bash shell 被打开时，该文件被读取。也就是说，每次新打开一个终端 shell，该文件就会被读取。
- ~/.bash_profile 或 ~/.profile: 只对单个用户生效，当用户登录时该文件仅执行一次。用户可使用该文件添加自己使用的 shell 变量信息。另外在不同的LINUX操作系统下，这个文件可能是不同的，可能是 ~/.bash_profile， ~/.bash_login 或 ~/.profile 其中的一种或几种，如果存在几种的话，那么执行的顺序便是：~/.bash_profile、 ~/.bash_login、 
- ~/.bashrc: 只对单个用户生效，当登录以及每次打开新的 shell 时，该文件被读取。
- 系统先读取 etc/profile 再读取 /etc/environment（还是反过来？）
- /etc/environment 中不能包含命令，即直接通过 `VAR="..."` 的方式设置，不使用 export 。
- 使用 `source /etc/environment` 可以使变量设置在当前窗口立即生效，需注销/重启之后，才能对每个新终端窗口都生效。

## 如何学习TVM源码

工具:Clion + pycharm plugin

 学习路线:

copy [蓝色大大](https://www.zhihu.com/people/lan-se-52-30/activities)在知乎上的[回答](https://www.zhihu.com/question/268423574)：

作者：蓝色

链接：https://www.zhihu.com/question/268423574/answer/506008668

来源：知乎

或许和很多人不同，以我的经验来看，觉得理解TVM，或者推理框架一定要从前端开始。即你从一个Tensorflow模型 / MXNet模型等，是如何转为NNVM的，然后再应该是后续的图优化，以及后续的TVM Tensor，LLVM代码生成等东西。

为什么我会这么强调从前端开始呢？因为不理解前端模型，就很难理解后续TVM为什么是这样，而且出了错以后很难知道到底是什么原因，比如很多时候找了半天，其实只是你忘记了模型输入图片的预处理，却误认为是后续卷积的调度优化做的有问题，**所以我强烈建议先从一个模型前端开始，在tvm/nnvm/frontend里面选取一个前端。**而选取前端开始不应该仅仅是看，Bug / 需求驱动永远是最好学习源代码的方式，建议从一个固化好的模型开始，然后补足NNVM算子，比如Mobilenet / Resnet50等，这里也是让你熟悉工具，熟悉NNVM的开始，可能会遇到很多问题，但是一个一个克服会收获很多，这里面推荐一个看模型的好工具: [https://github.com/lutzroeder/Netron](https://link.zhihu.com/?target=https%3A//github.com/lutzroeder/Netron) 我也是看苹果公司一个人用了以后发现的，确实是好东西。

接下来你应该首先理解TOPI，这是架设在NNVM与TVM之间的东西(首先忽略图优化，你后面再去看)，因为你需要理解NNVM Symbol (其它模型在转为NNVM前端表示时会以Symbol形式的Api表示)(注: 现TVM已用relay替代nnvm，relay可以直接lower到tvm层，通过relay.build()实现，无需topi作为中间层) 如何与TVM之间是如何连接起来的，在这里面你会有点迷糊，因为TVM是C++和Python混合的工程，这里面你需要在这两者跳来跳去，但是你这一步你最重要的是抓住两个核心: FTVMCompute (@reg.register_compute) / @reg.register_schedule，这一个你需要分别在nnvm/top里面的C++ / Python去找，top里面会告诉你是如何从NNVM进入topi的。

这一步完成以后，你则需要进入topi里面的任意一个后端Target去看，我暂时推荐x86后端，因为这一个后端还没有被AutoTVM改造。对于你来说，理解起来更容易。在这里你会遇到topi/nn里面的@tvm.target.generic_func到类似具体@generic.schedule_conv2d_nchw.register(["cpu"])的改变，这是TVM的核心所在，对于卷积这样的数据负载处理，为了优化而沿用Halide的思想: 计算与调度分离。为了理解这个，你最好参考一下这个文档: [https://docs.tvm.ai/tutorials/optimize/opt_gemm.html#sphx-glr-tutorials-optimize-opt-gemm-py](https://link.zhihu.com/?target=https%3A//docs.tvm.ai/tutorials/optimize/opt_gemm.html%23sphx-glr-tutorials-optimize-opt-gemm-py)

到这一步理解好以后，后续的TVM底层API大部分情况下你都不需要去动，包括后续的LLVM自动生成，优化等你也大部分不需要去动，因为类似CNN这样的网络，大部分你要做的工作就是在调度上，如何减少Cache Miss ，如何更好的让数据Locality是更关键的地方。

到这一步以后，你可以再回过头去理解图优化的部分，如Operator Fusion / FoldScaleAxis等，以及包括TVM目前最核心最与众不同的地方: AutoTVM([https://docs.tvm.ai/tutorials/autotvm/tune_nnvm_arm.html#sphx-glr-tutorials-autotvm-tune-nnvm-arm-py](https://link.zhihu.com/?target=https%3A//docs.tvm.ai/tutorials/autotvm/tune_nnvm_arm.html%23sphx-glr-tutorials-autotvm-tune-nnvm-arm-py))，这是TVM去击败NCNN等用手写汇编的推理框架的关键一环，用机器学习去解决机器学习的问题，让你从调度参数的设置中解放出来，而专心写调度算法。这里面目前ARM CPU的调度算法并非是最优的，但是从测试来看，至少在测试中使用硬件和环境来看，已经超过能找到的推理框架。后续我将撰写一篇文章到TVM社区，将我在ARM CPU的工作写出来，这将改善目前ARM CPU的官方调度版本，这将在Mobilenet等模型中有很好的提升，敬请关注！

TVM是很好的一个项目，这种基于编译优化思想的深度学习推理框架正是我赞同的，虽然还有很多工作需要做，但是我认为它已经走在一个很好的方向上了。

## 3.源码阅读小记

[tvm source code](https://github.com/Cyoung7/implement_tvm/blob/master/tvm source code.md)

