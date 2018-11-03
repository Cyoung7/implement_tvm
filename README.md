# implement_tvm

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

默认安装的路径: /home/cyoung/.miniconda3/lib/python3.6/site-packages/

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



