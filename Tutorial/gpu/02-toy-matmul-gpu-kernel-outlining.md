# Toy MatMul GPU Kernel Outlining

本文记录第二阶段改动：把第一阶段生成的 `gpu.launch` 继续 outline 成独立的 GPU kernel。

第一阶段输出形态是：

```text
func.func @main() {
  gpu.launch ... {
    ...
    gpu.terminator
  }
}
```

第二阶段输出形态是：

```text
func.func @main() {
  gpu.launch_func @...::@... blocks in (...) threads in (...) args(...)
}

gpu.module @... {
  gpu.func @... kernel {
    ...
    gpu.return
  }
}
```

也就是：

```text
gpu.launch -> gpu.launch_func + gpu.module + gpu.func kernel
```

这一阶段仍然不是最终可执行 CUDA binary。它只是把内联在 host 函数里的 launch body 分离成 device kernel，为后续 NVVM lowering、binary serialization、runtime launch 做准备。

## 为什么需要 outlining

`gpu.launch` 是一种比较高层的写法，kernel body 直接嵌在 host 函数里：

```mlir
gpu.launch blocks(...) in (...) threads(...) in (...) {
  // kernel body
  gpu.terminator
}
```

后续真正生成 GPU 代码时，device 端代码需要成为一个独立 kernel 函数。MLIR GPU dialect 用下面这种形式表达：

```mlir
gpu.module @main_kernel {
  gpu.func @main_kernel(...) kernel {
    // device code
    gpu.return
  }
}
```

host 侧不再包含 kernel body，只保留一次 kernel 调用：

```mlir
gpu.launch_func @main_kernel::@main_kernel
  blocks in (...)
  threads in (...)
  args(...)
```

所以第二阶段的核心 pass 是：

```cpp
mlir::createGpuKernelOutliningPass()
```

## toyc.cpp 的改动

新增头文件：

```cpp
#include "mlir/Dialect/GPU/Transforms/Passes.h"
```

这个头文件提供：

```cpp
mlir::createGpuKernelOutliningPass()
```

### 新增 emit action

`Action` enum 新增：

```cpp
DumpMLIRGPUOutlined,
```

命令行新增：

```cpp
clEnumValN(DumpMLIRGPUOutlined, "mlir-gpu-outlined",
           "output the MLIR dump after gpu kernel outlining")
```

所以现在可以运行：

```bash
./build/bin/toyc-ch7 mlir/test/Examples/Toy/Ch7/matmul64-gpu-outlining.mlir \
  -emit=mlir-gpu-outlined -x=mlir
```

### GPU pipeline 拆成两个阶段

现在的判断逻辑是：

```cpp
bool isLoweringToGPU = emitAction == Action::DumpMLIRGPU ||
                       emitAction == Action::DumpMLIRGPUOutlined;
bool isOutliningGPU = emitAction == Action::DumpMLIRGPUOutlined;
bool isLoweringToAffine = emitAction == Action::DumpMLIRAffine ||
                          emitAction >= Action::DumpMLIRLLVM;
bool isLoweringToLLVM = emitAction >= Action::DumpMLIRLLVM;
```

含义：

```text
-emit=mlir-gpu
  只执行 Toy -> gpu.launch。

-emit=mlir-gpu-outlined
  先执行 Toy -> gpu.launch，
  再执行 gpu.launch -> gpu.launch_func + gpu.module。

-emit=mlir-llvm / -emit=llvm / -emit=jit
  仍走原来的 Toy -> affine -> LLVM 路线。
```

注意 `mlir-gpu-outlined` 不能走原来的 affine lowering。否则 `toy.matmul` 会先被 lower 成 CPU affine loop，就没有 `gpu.launch` 可以 outline。

### 第二阶段 pass pipeline

第一段仍然是 Toy 到 GPU：

```cpp
if (isLoweringToGPU) {
  pm.addPass(mlir::toy::createLowerToGPUPass());

  mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
  optPM.addPass(mlir::createCanonicalizerPass());
  optPM.addPass(mlir::createCSEPass());
}
```

第二段只在 `-emit=mlir-gpu-outlined` 时执行：

```cpp
if (isOutliningGPU) {
  pm.addPass(mlir::createGpuKernelOutliningPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}
```

完整 pipeline 是：

```text
load Toy/MLIR
  -> inline
  -> canonicalize
  -> shape inference
  -> canonicalize
  -> cse
  -> createLowerToGPUPass()
  -> func.func(canonicalize)
  -> func.func(cse)
  -> createGpuKernelOutliningPass()
  -> canonicalize
  -> cse
  -> dump MLIR
```

### MLIR 输出判断

`main` 中也要把 `DumpMLIRGPUOutlined` 加进 MLIR dump 条件：

```cpp
bool isOutputingMLIR = emitAction == Action::DumpMLIR ||
                       emitAction == Action::DumpMLIRAffine ||
                       emitAction == Action::DumpMLIRGPU ||
                       emitAction == Action::DumpMLIRGPUOutlined ||
                       emitAction == Action::DumpMLIRLLVM;
```

否则 pass 跑完后不会直接打印 MLIR。

## CMakeLists.txt 的改动

第二阶段调用了 GPU transforms 里的 pass，所以 `toyc-ch7` 需要链接：

```cmake
MLIRGPUTransforms
```

对应改动：

```cmake
target_link_libraries(toyc-ch7
  PRIVATE
    ...
    MLIRGPUTransforms
    ...
)
```

如果不加这个库，云平台编译时可能出现类似问题：

```text
undefined reference to mlir::createGpuKernelOutliningPass()
```

## 新增测试文件

第二阶段测试文件：

```text
mlir/test/Examples/Toy/Ch7/matmul64-gpu-outlining.mlir
```

它使用 MLIR 输入，而不是 `.toy` 输入：

```mlir
%0 = toy.constant dense<1.000000e+00> : tensor<64x64xf64>
%1 = toy.constant dense<1.000000e+00> : tensor<64x64xf64>
%2 = toy.matmul %0, %1 : tensor<64x64xf64>, tensor<64x64xf64> to tensor<64x64xf64>
toy.print %2 : tensor<64x64xf64>
```

这里用 `dense<1.000000e+00> : tensor<64x64xf64>` 是 splat constant，表示 64x64 的所有元素都是 1.0。不要在 Toy 源文件里用一个标量强行 reshape 成 64x64，因为 `DenseElementsAttr::reshape` 要求 reshape 前后元素数量相同。

测试检查重点：

```text
1. host 函数里出现 gpu.launch_func。
2. device 侧出现 gpu.module。
3. device 侧出现带 kernel 属性的 gpu.func。
4. kernel 内还保留 naive matmul 的 scf.if/scf.for/memref.load/arith.mulf/arith.addf/memref.store。
5. 原来的 toy.matmul 不应再出现。
6. 原来的 gpu.launch blocks 不应再出现。
```

## 预期 IR 结构

第二阶段输出大致如下：

```mlir
module attributes {gpu.container_module} {
  func.func @main() {
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    ...
    %out = memref.alloc() : memref<64x64xf64>

    gpu.launch_func @main_kernel::@main_kernel
      blocks in (%c4, %c4, %c1)
      threads in (%c16, %c16, %c1)
      args(%lhs : memref<64x64xf64>,
           %rhs : memref<64x64xf64>,
           %out : memref<64x64xf64>)

    toy.print %out : memref<64x64xf64>
    return
  }

  gpu.module @main_kernel {
    gpu.func @main_kernel(
      %lhs: memref<64x64xf64>,
      %rhs: memref<64x64xf64>,
      %out: memref<64x64xf64>) kernel {
      %bid_x = gpu.block_id x
      %tid_x = gpu.thread_id x
      ...
      scf.if ... {
        %sum = scf.for ... -> (f64) {
          %a = memref.load %lhs[...] : memref<64x64xf64>
          %b = memref.load %rhs[...] : memref<64x64xf64>
          %p = arith.mulf %a, %b : f64
          %s = arith.addf ..., %p : f64
          scf.yield %s : f64
        }
        memref.store %sum, %out[...] : memref<64x64xf64>
      }
      gpu.return
    }
  }
}
```

实际 SSA 名字和 kernel 名字可能不同，所以测试不要依赖具体 `%0/%1` 或具体 symbol 名。

## 云平台编译命令

本地没有 GPU 时，不需要在本机跑。到云平台后建议这样配置：

```bash
cmake -S llvm -B build -G Ninja \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
  -DMLIR_ENABLE_EXECUTION_ENGINE=ON \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=Release
```

然后编译：

```bash
ninja -C build toyc-ch7 mlir-opt mlir-translate FileCheck
```

## 云平台验证命令

第一阶段验证：

```bash
./build/bin/toyc-ch7 mlir/test/Examples/Toy/Ch7/matmul64-gpu-lowering.mlir \
  -emit=mlir-gpu -x=mlir
```

应该看到：

```text
gpu.launch blocks(...) ... threads(...)
```

第二阶段验证：

```bash
./build/bin/toyc-ch7 mlir/test/Examples/Toy/Ch7/matmul64-gpu-outlining.mlir \
  -emit=mlir-gpu-outlined -x=mlir
```

应该看到：

```text
gpu.launch_func
gpu.module
gpu.func ... kernel
```

并且不应该再看到：

```text
gpu.launch blocks
toy.matmul
```

如果要跑 lit 测试：

```bash
./build/bin/llvm-lit mlir/test/Examples/Toy/Ch7/matmul64-gpu-lowering.mlir
./build/bin/llvm-lit mlir/test/Examples/Toy/Ch7/matmul64-gpu-outlining.mlir
```

也可以直接用 `FileCheck`：

```bash
./build/bin/toyc-ch7 mlir/test/Examples/Toy/Ch7/matmul64-gpu-outlining.mlir \
  -emit=mlir-gpu-outlined -x=mlir 2>&1 | \
  ./build/bin/FileCheck mlir/test/Examples/Toy/Ch7/matmul64-gpu-outlining.mlir
```

## 当前阶段还缺什么

第二阶段之后，IR 已经有独立 GPU kernel，但还没有到“可以直接在 GPU 上运行”：

```text
1. 还没有把 gpu.module 里的 gpu.func lowering 到 NVVM/ROCDL。
2. 还没有生成 GPU binary 或 fatbin。
3. 还没有 lower gpu.launch_func 到 CUDA/HIP runtime 调用。
4. 还没有显式处理 host/device memory allocation 和 memcpy。
5. toy.print 仍然是 Toy op，只是输入从 tensor 变成了 memref。
```

因此第二阶段的验收标准不是运行结果，而是 IR 结构正确：

```text
gpu.launch_func + gpu.module + gpu.func kernel
```

下一阶段先接 device 侧 NVVM lowering，见：

```text
Tutorial/gpu/03-toy-matmul-gpu-nvvm-lowering.md
```

runtime 调用和真正执行会放到后续阶段。
