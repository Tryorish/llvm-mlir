# Toy MatMul GPU Binary Lowering

本文记录第四阶段改动：在第三阶段 device kernel 已经 lower 到 NVVM/LLVM IR 之后，把 `gpu.module` 继续转换成 `gpu.binary`。

第四阶段目标是观察 IR：

```text
gpu.module @main_kernel {
  llvm.func @main_kernel(...) attributes {nvvm.kernel} { ... }
}

  -> gpu.binary @main_kernel [#gpu.object<#nvvm.target<...>, offload = "...">]
```

host 侧仍然保留：

```text
gpu.launch_func
toy.print
```

所以第四阶段仍然不是最终运行。它只完成 device module 的 binary/offloading object 包装。

## 为什么先用 format=llvm

`gpu-module-to-binary` 支持几种输出格式：

```text
llvm / offloading  生成 offloading 表示
isa / assembly     生成 PTX/ISA 文本
bin / binary       生成目标二进制
fatbin / fatbinary 生成 fatbin
```

本阶段先使用：

```cpp
binaryOptions.compilationTarget = "llvm";
```

原因是 `format=llvm` 更适合作为教学和 CI 的第四阶段验收：

```text
1. 不需要马上依赖 ptxas。
2. 不需要 CUDA toolkit 完整可用。
3. 可以先验证 gpu.module -> gpu.binary 的 MLIR 链路。
```

如果改成：

```cpp
binaryOptions.compilationTarget = "isa";
```

或：

```cpp
binaryOptions.compilationTarget = "fatbin";
```

云平台就需要更完整的 NVIDIA 工具链。

## toyc.cpp 的改动

新增 target 注册头文件：

```cpp
#include "mlir/Target/LLVM/NVVM/Target.h"
```

并在 `DialectRegistry` 上注册 NVVM target external model：

```cpp
mlir::NVVM::registerNVVMTargetInterfaceExternalModels(registry);
```

这一步让 `#nvvm.target` 实现 `gpu::TargetAttrInterface`，也就是让 `gpu-module-to-binary` 知道如何把 NVVM target 的 `gpu.module` 序列化成 object。如果缺少这一步，`gpu-module-to-binary` 可能无法处理 `#nvvm.target`。

### 新增 emit action

`Action` enum 新增：

```cpp
DumpMLIRGPUBinary,
```

命令行新增：

```cpp
clEnumValN(DumpMLIRGPUBinary, "mlir-gpu-binary",
           "output the MLIR dump after gpu binary lowering")
```

使用方式：

```bash
./build/bin/toyc-ch7 mlir/test/Examples/Toy/Ch7/matmul64-gpu-binary.mlir \
  -emit=mlir-gpu-binary -x=mlir
```

### pass 阶段判断

第四阶段复用前三阶段：

```cpp
bool isLoweringToGPU = emitAction == Action::DumpMLIRGPU ||
                       emitAction == Action::DumpMLIRGPUOutlined ||
                       emitAction == Action::DumpMLIRGPUNVVM ||
                       emitAction == Action::DumpMLIRGPUBinary;

bool isOutliningGPU = emitAction == Action::DumpMLIRGPUOutlined ||
                      emitAction == Action::DumpMLIRGPUNVVM ||
                      emitAction == Action::DumpMLIRGPUBinary;

bool isLoweringGPUToNVVM = emitAction == Action::DumpMLIRGPUNVVM ||
                           emitAction == Action::DumpMLIRGPUBinary;

bool isLoweringGPUToBinary = emitAction == Action::DumpMLIRGPUBinary;
```

含义：

```text
-emit=mlir-gpu
  Toy -> gpu.launch

-emit=mlir-gpu-outlined
  Toy -> gpu.launch
      -> gpu.launch_func + gpu.module/gpu.func

-emit=mlir-gpu-nvvm
  Toy -> gpu.launch
      -> gpu.launch_func + gpu.module/gpu.func
      -> gpu.module 内部 lower 到 NVVM/LLVM

-emit=mlir-gpu-binary
  Toy -> gpu.launch
      -> gpu.launch_func + gpu.module/gpu.func
      -> gpu.module 内部 lower 到 NVVM/LLVM
      -> gpu.binary
```

### 第四阶段 pipeline

新增：

```cpp
if (isLoweringGPUToBinary) {
  mlir::GpuNVVMAttachTargetOptions nvvmTargetOptions;
  nvvmTargetOptions.triple = "nvptx64-nvidia-cuda";
  nvvmTargetOptions.chip = "sm_86";
  nvvmTargetOptions.features = "+ptx60";
  pm.addPass(mlir::createGpuNVVMAttachTarget(nvvmTargetOptions));

  mlir::GpuModuleToBinaryPassOptions binaryOptions;
  binaryOptions.compilationTarget = "llvm";
  pm.addPass(mlir::createGpuModuleToBinaryPass(binaryOptions));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}
```

这里有两个关键 pass。

## nvvm-attach-target

`createGpuNVVMAttachTarget()` 给 `gpu.module` 加 target：

```mlir
gpu.module @main_kernel [#nvvm.target<chip = "sm_86", features = "+ptx60">] {
  ...
}
```

`gpu-module-to-binary` 需要这个 target attribute。没有 target 时会报：

```text
the module has no target attributes
```

当前配置：

```text
triple   = nvptx64-nvidia-cuda
chip     = sm_86
features = +ptx60
```

这里按当前云平台 NVIDIA A10 配置选择 `sm_86`。如果换到其他 GPU，可以改成对应的 `sm_80`、`sm_90` 等。

## gpu-module-to-binary

`createGpuModuleToBinaryPass()` 会查找 `gpu.module`，然后调用 target 的 serializer，把它替换成：

```mlir
gpu.binary @main_kernel [
  #gpu.object<#nvvm.target<...>, offload = "...">
]
```

也就是说，第四阶段之后：

```text
gpu.module 消失
gpu.binary 出现
```

但 host 侧的：

```text
gpu.launch_func
```

仍然存在，还没有变成 CUDA runtime call。

## CMakeLists.txt 的改动

第四阶段显式补了：

```cmake
MLIRNVVMTarget
MLIRTargetLLVM
```

原因：

```text
MLIRNVVMTarget  提供 NVVM target serialization 能力。
MLIRTargetLLVM  支撑 LLVM/offloading object 生成。
```

之前已有：

```cmake
MLIRGPUTransforms
MLIRGPUToNVVMTransforms
MLIRNVVMToLLVM
```

它们分别提供 GPU transform pass、GPU->NVVM lowering、NVVM->LLVM conversion interface。

## 新增测试文件

测试文件：

```text
mlir/test/Examples/Toy/Ch7/matmul64-gpu-binary.mlir
```

测试命令：

```text
toyc-ch7 %s -emit=mlir-gpu-binary -x=mlir
```

它检查：

```text
1. host 侧仍有 gpu.launch_func。
2. host 侧仍有 toy.print memref<64x64xf64>。
3. 出现 gpu.binary。
4. gpu.binary 内有 #gpu.object<#nvvm.target ... offload = "...">。
5. 不再出现 toy.matmul。
6. 不再出现 gpu.module / gpu.func。
```

## 预期 IR 形态

第四阶段输出大致如下：

```mlir
module attributes {gpu.container_module} {
  func.func @main() {
    ...
    gpu.launch_func @main_kernel::@main_kernel
      blocks in (...)
      threads in (...)
      args(...)
    toy.print %out : memref<64x64xf64>
    return
  }

  gpu.binary @main_kernel [
    #gpu.object<#nvvm.target<chip = "sm_86", features = "+ptx60">,
                offload = "...">
  ]
}
```

注意：`gpu.launch_func` 仍然引用原来的 kernel symbol。后续 lowering 到 LLVM/offloading runtime 时，MLIR 的 GPU offloading translation 会使用 `gpu.binary` 里的对象来生成 runtime 侧调用。

## 云平台验证命令

只编译 Toy Ch7：

```bash
ninja -C build toyc-ch7
```

手动看第四阶段 IR：

```bash
./build/bin/toyc-ch7 mlir/test/Examples/Toy/Ch7/matmul64-gpu-binary.mlir \
  -emit=mlir-gpu-binary -x=mlir
```

如果要跑测试：

```bash
ninja -C build FileCheck llvm-lit
./build/bin/llvm-lit mlir/test/Examples/Toy/Ch7/matmul64-gpu-binary.mlir
```

或者：

```bash
./build/bin/toyc-ch7 mlir/test/Examples/Toy/Ch7/matmul64-gpu-binary.mlir \
  -emit=mlir-gpu-binary -x=mlir 2>&1 | \
  ./build/bin/FileCheck mlir/test/Examples/Toy/Ch7/matmul64-gpu-binary.mlir
```

## 下一阶段

第四阶段之后，device 侧已经被包装成 `gpu.binary`，但还不能直接运行。

下一阶段要处理：

```text
1. gpu.launch_func -> CUDA runtime 调用。
2. host 侧 memref/arith/func lowering 到 LLVM。
3. host/device 内存分配和拷贝。
4. toy.print 的处理。
```

也就是从“有 GPU binary object”进入“host 能调用 GPU kernel”的阶段。
