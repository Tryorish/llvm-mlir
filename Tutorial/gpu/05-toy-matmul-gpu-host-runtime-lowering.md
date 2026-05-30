# Toy MatMul GPU Host Runtime Lowering

本文记录第五阶段改动：在第四阶段已经生成 `gpu.binary` 之后，继续 lower host 侧代码，并在 LLVM IR 输出中生成 GPU runtime 调用。

第五阶段新增两个输出：

```text
-emit=mlir-gpu-host
  输出 host lowering 后的 MLIR。

-emit=llvm-gpu
  输出最终 LLVM IR，可以看到 mgpuModuleLoad / mgpuLaunchKernel。
```

## 这一阶段做什么

第四阶段结束时 IR 形态大致是：

```mlir
module attributes {gpu.container_module} {
  func.func @main() {
    ...
    gpu.launch_func @main_kernel::@main_kernel ...
    toy.print %out : memref<64x64xf64>
    return
  }

  gpu.binary @main_kernel [
    #gpu.object<#nvvm.target<chip = "sm_86">, offload = "...">
  ]
}
```

第五阶段继续处理 host 侧：

```text
1. toy.print -> scf.for + llvm.call @printf
2. scf.for/scf.if -> cf.br/cf.cond_br
3. func/memref/arith/cf -> LLVM dialect
4. 保留 gpu.binary 和 gpu.launch_func，交给 LLVM IR translation 做 offloading 翻译
```

注意：`gpu-to-llvm` 这个 MLIR pass 会把 host 侧类型和普通 op 转成 LLVM dialect，但它不会在 MLIR 文本中直接删除 `gpu.launch_func`。真正把 `gpu.launch_func` 翻译成 `mgpuLaunchKernel`，发生在：

```cpp
mlir::translateModuleToLLVMIR(module, llvmContext)
```

也就是 `-emit=llvm-gpu` 阶段。

## 新增 print-only lowering pass

原来的 `createLowerToLLVMPass()` 是 CPU codegen 路径使用的完整 lowering pass，它会处理 Toy + Affine + SCF + Func + MemRef。

GPU host 路径不能直接复用它，因为 GPU 路径里已经有：

```text
gpu.binary
gpu.launch_func
```

所以新增一个更小的 pass：

```cpp
mlir::toy::createLowerPrintToLLVMPass()
```

它只做一件事：

```text
toy.print -> scf.for + memref.load + llvm.call @printf
```

这样后续标准 `gpu-to-llvm` pass 可以继续 lower `scf` / `func` / `memref` / `arith`。

## toyc.cpp 的第五阶段 pipeline

第五阶段复用前四阶段：

```text
Toy -> gpu.launch
    -> gpu.launch_func + gpu.module
    -> gpu.module 内部 lower 到 NVVM/LLVM
    -> gpu.binary
```

然后新增 host lowering：

```cpp
if (isLoweringGPUHost) {
  pm.addPass(mlir::toy::createLowerPrintToLLVMPass());
  pm.addPass(mlir::createSCFToControlFlowPass());

  mlir::GpuToLLVMConversionPassOptions gpuToLLVMOptions;
  pm.addPass(mlir::createGpuToLLVMConversionPass(gpuToLLVMOptions));
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}
```

`createGpuToLLVMConversionPass()` 的作用：

```text
1. lower host 侧 func / memref / arith / cf 到 LLVM dialect。
2. 保留 gpu.binary。
3. 保留 operand 已经合法化后的 gpu.launch_func。
```

为什么保留 `gpu.launch_func`：

```text
gpu.launch_func 需要和 gpu.binary 配合。
LLVM IR translation 会查找对应的 gpu.binary，把 binary 嵌入 LLVM module，
并把 launch_func 翻译成 mgpuModuleGetFunction + mgpuLaunchKernel。
```

## 新增 registry

第五阶段新增：

```cpp
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUCommon/GPUToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
```

并注册：

```cpp
mlir::gpu::registerConvertGpuToLLVMInterface(registry);
```

这个 interface 让 GPU 相关 op 能参与 `gpu-to-llvm` 的 host lowering。

## CMakeLists.txt 的改动

新增链接库：

```cmake
MLIRGPUToGPURuntimeTransforms
MLIRReconcileUnrealizedCasts
```

原因：

```text
MLIRGPUToGPURuntimeTransforms
  提供 createGpuToLLVMConversionPass() 和 GPU runtime call lowering。

MLIRReconcileUnrealizedCasts
  清理 conversion 过程中产生的 unrealized_conversion_cast。
```

## 预期 MLIR 形态

运行：

```bash
./build/bin/toyc-ch7 mlir/test/Examples/Toy/Ch7/matmul-codegen.toy \
  -emit=mlir-gpu-host
```

预期还能看到：

```mlir
gpu.binary @main_kernel [...]
gpu.launch_func @main_kernel::@main_kernel ...
```

但 host 侧应该已经变成 LLVM dialect，例如：

```mlir
llvm.func @main() {
  ...
  gpu.launch_func @main_kernel::@main_kernel ...
  ...
  llvm.call @printf(...)
  llvm.return
}
```

## 预期 LLVM IR 形态

运行：

```bash
./build/bin/toyc-ch7 mlir/test/Examples/Toy/Ch7/matmul-codegen.toy \
  -emit=llvm-gpu
```

预期能看到：

```llvm
@main_kernel_binary = internal constant ...
@main_kernel_module = internal global ptr null
@llvm.global_ctors = ...
@llvm.global_dtors = ...
```

以及 runtime 调用：

```llvm
call ptr @mgpuModuleLoad(...)
call ptr @mgpuModuleGetFunction(...)
call ptr @mgpuStreamCreate()
call void @mgpuLaunchKernel(...)
call void @mgpuStreamSynchronize(...)
call void @mgpuStreamDestroy(...)
call void @mgpuModuleUnload(...)
```

这些 `mgpu*` 函数不是 CUDA Driver API 原函数，而是 MLIR GPU runtime wrapper 的稳定 ABI。真正链接运行时，需要链接 MLIR 的 CUDA runtime wrapper 库。

## 云平台验证命令

只编译 Toy Ch7：

```bash
ninja -C build toyc-ch7
```

看 host lowering 后的 MLIR：

```bash
./build/bin/toyc-ch7 mlir/test/Examples/Toy/Ch7/matmul-codegen.toy \
  -emit=mlir-gpu-host
```

看最终 LLVM IR：

```bash
./build/bin/toyc-ch7 mlir/test/Examples/Toy/Ch7/matmul-codegen.toy \
  -emit=llvm-gpu
```

快速检查 runtime call：

```bash
./build/bin/toyc-ch7 mlir/test/Examples/Toy/Ch7/matmul-codegen.toy \
  -emit=llvm-gpu 2>&1 | grep -E "mgpu(ModuleLoad|ModuleGetFunction|LaunchKernel|StreamCreate|StreamSynchronize|StreamDestroy|ModuleUnload)"
```

## 这一阶段还不能保证直接运行

第五阶段已经能生成 host 调用 GPU runtime 的 LLVM IR，但要真正执行，还需要处理运行时链接：

```text
1. 链接 MLIR CUDA runtime wrapper。
2. 链接 CUDA driver/runtime 依赖。
3. 确认当前 memref 指针对于 device kernel 是否可访问。
```

当前 Toy GPU lowering 是教学用 naive 版本，它直接把 host memref 作为 kernel 参数传入。真实 CUDA 程序通常还需要：

```text
host alloc -> gpu.alloc
host->device gpu.memcpy
kernel launch
device->host gpu.memcpy
gpu.dealloc
```

所以第六阶段应该补显式 device memory 管理，而不是只依赖 host 指针。
