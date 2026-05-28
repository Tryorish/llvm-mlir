# Toy MatMul GPU to NVVM Lowering

本文记录第三阶段改动：在第二阶段 `gpu.launch_func + gpu.module + gpu.func kernel` 的基础上，继续把 device kernel 内部 lower 到 NVVM/LLVM 相关 IR。

第三阶段目标是观察 IR：

```text
gpu.func kernel
  -> llvm.func attributes {nvvm.kernel}

gpu.block_id / gpu.thread_id
  -> nvvm.read.ptx.sreg.ctaid.* / nvvm.read.ptx.sreg.tid.*

arith / memref / cf
  -> llvm dialect ops
```

这一阶段仍然不是最终运行。host 侧仍保留：

```text
gpu.launch_func
toy.print
```

也就是说，第三阶段是 partial NVVM lowering，不是完整的 CUDA runtime/fatbin pipeline。

## 为什么不用完整 gpu-lower-to-nvvm-pipeline

MLIR 自带完整 pipeline：

```text
gpu-lower-to-nvvm-pipeline
```

它会继续做：

```text
gpu.module -> gpu.binary/fatbin
gpu.launch_func -> CUDA runtime 调用
host 侧 dialect -> LLVM
```

但 Toy Ch7 当前还有一个教学限制：

```text
toy.print 仍然保留在 host 侧
```

如果直接跑完整 pipeline，host 侧会被继续 lower，而 `toy.print` 不是标准 GPU runtime pipeline 的一部分，容易把第三阶段和第四/第五阶段混在一起。

因此本阶段先只做 device kernel 的 NVVM lowering：

```text
host:
  func.func @main
  gpu.launch_func
  toy.print

device:
  gpu.module {
    llvm.func ... attributes {nvvm.kernel}
    nvvm.read.ptx.sreg.*
    llvm.load/store/fmul/fadd
  }
```

## toyc.cpp 的改动

新增 include：

```cpp
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
```

这些头文件提供第三阶段需要的 pass 和转换接口。

### 新增 emit action

`Action` enum 新增：

```cpp
DumpMLIRGPUNVVM,
```

命令行新增：

```cpp
clEnumValN(DumpMLIRGPUNVVM, "mlir-gpu-nvvm",
           "output the MLIR dump after gpu to nvvm lowering")
```

使用方式：

```bash
./build/bin/toyc-ch7 mlir/test/Examples/Toy/Ch7/matmul64-gpu-nvvm.mlir \
  -emit=mlir-gpu-nvvm -x=mlir
```

### pass 阶段判断

现在 GPU 相关 emit 分三层：

```cpp
bool isLoweringToGPU = emitAction == Action::DumpMLIRGPU ||
                       emitAction == Action::DumpMLIRGPUOutlined ||
                       emitAction == Action::DumpMLIRGPUNVVM;

bool isOutliningGPU = emitAction == Action::DumpMLIRGPUOutlined ||
                      emitAction == Action::DumpMLIRGPUNVVM;

bool isLoweringGPUToNVVM = emitAction == Action::DumpMLIRGPUNVVM;
```

含义：

```text
-emit=mlir-gpu
  Toy -> gpu.launch

-emit=mlir-gpu-outlined
  Toy -> gpu.launch -> gpu.launch_func + gpu.module/gpu.func

-emit=mlir-gpu-nvvm
  Toy -> gpu.launch
      -> gpu.launch_func + gpu.module/gpu.func
      -> gpu.module 内部 lower 到 NVVM/LLVM
```

### 注册 LLVM conversion interfaces

`convert-gpu-to-nvvm` 会顺带调用各 dialect 的 `ConvertToLLVMPatternInterface`，把 device kernel 内部的 arith/memref/func/cf 等 op 转成 LLVM dialect。

所以在 `DialectRegistry` 中注册：

```cpp
mlir::arith::registerConvertArithToLLVMInterface(registry);
mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
mlir::registerConvertFuncToLLVMInterface(registry);
mlir::registerConvertMemRefToLLVMInterface(registry);
mlir::registerConvertNVVMToLLVMInterface(registry);
mlir::ub::registerConvertUBToLLVMInterface(registry);
```

否则 pass 可能无法把 kernel 内部的非 GPU op 一起降干净。
其中 `NVVMToLLVM` 和 `UBToLLVM` 是为了处理 `nvvm` / `ub` dialect 承诺的 LLVM conversion interface；如果不注册，`convert-gpu-to-nvvm` 遍历已加载 dialect 时可能报：

```text
LLVM ERROR: checking for an interface (`mlir::ConvertToLLVMPatternInterface`)
that was promised by dialect 'nvvm' but never implemented
```

### 第三阶段 pipeline

第三阶段新增：

```cpp
if (isLoweringGPUToNVVM) {
  pm.addPass(mlir::createSCFToControlFlowPass());

  mlir::ConvertGpuOpsToNVVMOpsOptions gpuToNVVMOptions;
  gpuToNVVMOptions.indexBitwidth = 64;
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(
      mlir::createConvertGpuOpsToNVVMOps(gpuToNVVMOptions));
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(
      mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createCSEPass());
}
```

`createSCFToControlFlowPass()` 先把：

```text
scf.if / scf.for
```

转成：

```text
cf.cond_br / cf.br
```

然后 `createConvertGpuOpsToNVVMOps()` 在 `gpu.module` 内部把：

```text
gpu.func
gpu.block_id
gpu.thread_id
gpu.return
arith.*
memref.*
cf.*
func.*
```

转换到：

```text
llvm.func
nvvm.read.ptx.sreg.*
llvm.return
llvm.*
```

这里 `indexBitwidth = 64` 表示 device 侧 index 使用 64 位整数 lowering，和当前 Toy/host 侧 index 默认习惯保持一致。

## CMakeLists.txt 的改动

第三阶段需要显式链接这些库：

```cmake
MLIRArithToLLVM
MLIRControlFlowToLLVM
MLIRFuncToLLVM
MLIRGPUToNVVMTransforms
MLIRMemRefToLLVM
MLIRNVVMToLLVM
MLIRSCFToControlFlow
MLIRUBToLLVM
```

其中最关键的是：

```text
MLIRGPUToNVVMTransforms
```

它提供：

```cpp
createConvertGpuOpsToNVVMOps(...)
```

## 新增测试文件

测试文件：

```text
mlir/test/Examples/Toy/Ch7/matmul64-gpu-nvvm.mlir
```

测试命令：

```text
toyc-ch7 %s -emit=mlir-gpu-nvvm -x=mlir
```

它检查：

```text
1. host 侧仍有 gpu.launch_func。
2. host 侧仍有 toy.print memref<64x64xf64>。
3. gpu.module 内部出现 llvm.func。
4. kernel 标记变成 nvvm.kernel。
5. block/thread id 变成 nvvm.read.ptx.sreg.*。
6. load/store/fmul/fadd 变成 llvm.*。
7. 不再出现 toy.matmul。
8. 不再出现 gpu.func / gpu.block_id / gpu.thread_id。
```

## 预期 IR 形态

第三阶段输出会类似：

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

  gpu.module @main_kernel {
    llvm.func @main_kernel(...) attributes {nvvm.kernel} {
      %bid_x = nvvm.read.ptx.sreg.ctaid.x : i32
      %tid_x = nvvm.read.ptx.sreg.tid.x : i32
      ...
      %a = llvm.load ...
      %b = llvm.load ...
      %p = llvm.fmul %a, %b : f64
      %s = llvm.fadd %acc, %p : f64
      llvm.store %s, ...
      llvm.return
    }
  }
}
```

实际 IR 中会有很多 memref descriptor、指针计算和 block 参数，这是正常的。

## 云平台验证命令

只编译 Toy Ch7：

```bash
ninja -C build toyc-ch7
```

手动看第三阶段 IR：

```bash
./build/bin/toyc-ch7 mlir/test/Examples/Toy/Ch7/matmul64-gpu-nvvm.mlir \
  -emit=mlir-gpu-nvvm -x=mlir
```

如果要跑测试：

```bash
ninja -C build FileCheck llvm-lit
./build/bin/llvm-lit mlir/test/Examples/Toy/Ch7/matmul64-gpu-nvvm.mlir
```

或者：

```bash
./build/bin/toyc-ch7 mlir/test/Examples/Toy/Ch7/matmul64-gpu-nvvm.mlir \
  -emit=mlir-gpu-nvvm -x=mlir 2>&1 | \
  ./build/bin/FileCheck mlir/test/Examples/Toy/Ch7/matmul64-gpu-nvvm.mlir
```

## 下一阶段

第三阶段后，device kernel 已经接近 NVVM/LLVM 层，但还没有变成可运行程序。

下一阶段要处理：

```text
1. gpu.module -> gpu.binary / fatbin。
2. gpu.launch_func -> CUDA runtime 调用。
3. host/device 内存分配和拷贝。
4. toy.print 的 host 侧 lowering 或替换成普通 runtime 输出。
```

也就是从“能看 NVVM IR”进入“能真正运行”的阶段。
