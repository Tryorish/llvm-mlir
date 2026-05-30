# Toy MatMul GPU Run

本文记录第七阶段：把 `-emit=llvm-gpu` 生成的 LLVM IR 编译、链接成可执行文件，并在 CUDA GPU 上运行。

前面阶段已经生成了 host LLVM IR，里面包含：

```llvm
call ptr @mgpuModuleLoad(...)
call ptr @mgpuMemAlloc(...)
call void @mgpuMemcpy(...)
call void @mgpuLaunchKernel(...)
call void @mgpuMemFree(...)
```

这些 `mgpu*` 符号不是 CUDA 原生 API 名字，而是 MLIR 的 CUDA runtime wrapper，实现在：

```text
mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp
```

所以第七阶段需要两件事：

```text
1. 生成 CUDA driver 能加载的 GPU binary。
2. 链接 MLIR CUDA runtime wrapper。
```

## 为什么不能继续用 format=llvm

第四阶段为了方便观察，我们把 `gpu-module-to-binary` 设置成：

```cpp
binaryOptions.compilationTarget = "llvm";
```

这会生成 LLVM bitcode offload object。你之前看到的：

```text
BC C0 DE ...
```

就是 LLVM bitcode 的开头。

但第七阶段要真正运行，`mgpuModuleLoad` 内部会调用 CUDA driver API 加载模块。它需要 PTX/cubin/fatbin 这类 CUDA 可加载对象，而不是 LLVM bitcode。

所以本阶段新增了命令行选项：

```bash
-gpu-binary-format=<llvm|isa|bin|fatbin>
```

默认仍然是 `llvm`，这样前面教程阶段的输出保持不变。真正运行时使用：

```bash
-gpu-binary-format=fatbin
```

对应代码在 `toyc.cpp`：

```cpp
static cl::opt<std::string>
    gpuBinaryFormat("gpu-binary-format", cl::init("llvm"),
                    cl::desc("GPU binary format ..."));

mlir::GpuModuleToBinaryPassOptions binaryOptions;
binaryOptions.compilationTarget = gpuBinaryFormat;
pm.addPass(mlir::createGpuModuleToBinaryPass(binaryOptions));
```

## 云端构建

先确保 `toyc-ch7` 和 CUDA runtime wrapper 都编译出来：

```bash
ninja -C build toyc-ch7 mlir_cuda_runtime
```

如果 `mlir_cuda_runtime` target 不存在，说明你的 LLVM/MLIR CMake 配置没有打开 CUDA runner/runtime。需要重新配置 CMake，确保能找到 CUDA toolkit 和 CUDA driver library。

## 生成 LLVM IR

```bash
./build/bin/toyc-ch7 mlir/test/Examples/Toy/Ch7/matmul-codegen.toy \
  -emit=llvm-gpu \
  -gpu-binary-format=fatbin \
  > /tmp/matmul-gpu.ll
```

可以快速确认 IR 里有 runtime 调用：

```bash
grep -E "mgpu(ModuleLoad|MemAlloc|Memcpy|LaunchKernel|MemFree)" /tmp/matmul-gpu.ll
```

还可以确认不是 LLVM bitcode offload：

```bash
grep main_kernel_binary /tmp/matmul-gpu.ll
```

如果还是看到大量 `BC\C0\DE`，说明没有传 `-gpu-binary-format=fatbin`，或者 pass 仍然使用了 `format=llvm`。

## 链接可执行文件

通常库会在：

```text
build/lib/libmlir_cuda_runtime.so
```

先确认：

```bash
ls build/lib/libmlir_cuda_runtime.*
```

用 clang 链接：

```bash
clang++ /tmp/matmul-gpu.ll \
  -L build/lib \
  -lmlir_cuda_runtime \
  -Wl,-rpath,$PWD/build/lib \
  -ldl -lpthread -lm \
  -o /tmp/matmul-gpu
```

如果链接时报找不到 CUDA driver 符号，例如 `cuModuleLoadData`，再加 CUDA driver library 路径：

```bash
clang++ /tmp/matmul-gpu.ll \
  -L build/lib \
  -lmlir_cuda_runtime \
  -L /usr/local/cuda/lib64 \
  -lcuda \
  -Wl,-rpath,$PWD/build/lib \
  -ldl -lpthread -lm \
  -o /tmp/matmul-gpu
```

有些云平台的 `libcuda.so` 在 driver 目录，例如：

```bash
ldconfig -p | grep libcuda
```

然后把对应目录加到 `-L`。

## 运行

```bash
/tmp/matmul-gpu
```

`matmul-codegen.toy` 是 2x2：

```text
A = [[1, 2],
     [3, 4]]

B = [[5, 6],
     [7, 8]]
```

所以期望输出：

```text
19.000000 22.000000
43.000000 50.000000
```

## 常见错误

### 1. 找不到 libmlir_cuda_runtime.so

说明没有构建 runtime：

```bash
ninja -C build mlir_cuda_runtime
```

如果 target 不存在，需要重新 CMake 配置 CUDA runner。

### 2. 运行时报找不到 libmlir_cuda_runtime.so

设置：

```bash
export LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH
```

或者链接时保留：

```bash
-Wl,-rpath,$PWD/build/lib
```

### 3. CUDA driver 加载失败

先检查：

```bash
nvidia-smi
ldconfig -p | grep libcuda
```

如果 `nvidia-smi` 正常，但链接或运行找不到 CUDA driver，通常是 `libcuda.so` 路径没有传给 linker 或 loader。

### 4. 仍然输出 BC C0 DE

这是 LLVM bitcode，不是运行格式。重新生成：

```bash
./build/bin/toyc-ch7 mlir/test/Examples/Toy/Ch7/matmul-codegen.toy \
  -emit=llvm-gpu \
  -gpu-binary-format=fatbin \
  > /tmp/matmul-gpu.ll
```

## 本阶段边界

第七阶段解决的是“能跑起来”：

```text
Toy -> LLVM IR -> native executable -> CUDA GPU execution
```

它还没有做性能优化。kernel 仍然是 naive one-thread-one-output，下一阶段才适合做 tiling、shared memory、coalescing 等优化。

## 直接 GPU JIT 运行

也可以不手动生成 `.ll` 和可执行文件，而是让 `toyc-ch7` 直接 JIT 运行：

```bash
./build/bin/toyc-ch7 mlir/test/Examples/Toy/Ch7/matmul-codegen.toy \
  -emit=gpu-jit \
  -shared-libs=$PWD/build/lib/libmlir_cuda_runtime.so
```

`-emit=gpu-jit` 内部走的仍然是同一条 lowering 链：

```text
Toy
-> gpu.launch
-> gpu.module
-> NVVM
-> gpu.binary
-> GPU host runtime lowering
-> LLVM dialect
-> ExecutionEngine JIT
```

区别只是最后一步不再把 LLVM IR 打印出来，而是交给 MLIR `ExecutionEngine` JIT 编译 host 侧 IR。GPU kernel 仍然通过 `gpu.binary` 和 `mgpuModuleLoad` 交给 CUDA driver 加载。

`-emit=gpu-jit` 默认使用：

```bash
-gpu-binary-format=fatbin
```

如果你显式传了 `-gpu-binary-format=llvm`，它仍然会生成 LLVM bitcode offload object，通常不能真正运行。

如果 runtime 库路径不同，先找一下：

```bash
find build -name 'libmlir_cuda_runtime.*'
```

然后把找到的路径传给：

```bash
-shared-libs=/path/to/libmlir_cuda_runtime.so
```
