# toy.matmul lower 到 GPU 的步骤总结

本文针对 `mlir/examples/toy/Ch7` 当前代码结构，说明如何把 `toy.matmul` 从 Toy IR lower 到 GPU。当前工程里 `toy.matmul` 已经存在，并且已有一条 CPU 路径：

```text
Toy AST
  -> Toy MLIR
  -> shape inference
  -> toy-to-affine
  -> lower-to-llvm
  -> JIT
```

如果要上 GPU，关键点是：`toy.matmul` 必须在当前 `MatMulOpLowering` 把它改成 `affine.for` 之前被 GPU 路径接管。

## 当前 toy.matmul 的状态

相关文件：

- `include/toy/Ops.td`
  - `MatMulOp` 名字是 `toy.matmul`。
  - 输入是两个 `F64Tensor`。
  - 输出是一个 `F64Tensor`。
  - 语义是 `(M x K) * (K x N) -> (M x N)`。
- `mlir/Dialect.cpp`
  - `MatMulOp::build` 初始返回 `tensor<*xf64>`。
  - `MatMulOp::inferShapes` 根据 lhs/rhs 推出 `M x N`。
  - `MatMulOp::verify` 检查 rank-2、`lhs.shape[1] == rhs.shape[0]`、result 是 `M x N`。
- `mlir/MLIRGen.cpp`
  - Toy 源码里的 `matmul(a, b)` 会生成 `toy.matmul`。
- `mlir/LowerToAffineLoops.cpp`
  - 当前 `MatMulOpLowering` 生成三层 `affine.for i/j/k`。
  - 每个 `C[i, j]` 由内层 `k` 循环累加。

当前 CPU lowering 形状大致是：

```mlir
%C = memref.alloc() : memref<MxNxf64>
affine.for %i = 0 to M {
  affine.for %j = 0 to N {
    %sum = affine.for %k = 0 to K iter_args(%acc = %c0) -> f64 {
      %a = affine.load %A[%i, %k] : memref<MxKxf64>
      %b = affine.load %B[%k, %j] : memref<KxNxf64>
      %p = arith.mulf %a, %b : f64
      %s = arith.addf %acc, %p : f64
      affine.yield %s : f64
    }
    affine.store %sum, %C[%i, %j] : memref<MxNxf64>
  }
}
```

## 推荐实现路线

有两条路线：

1. 教学/最小实现路线：`toy.matmul -> gpu.launch -> NVVM/ROCDL/SPIR-V`
2. 工程化路线：`toy.matmul -> linalg.matmul -> tiling/vector/gpu -> NVVM/ROCDL/SPIR-V`

建议先做路线 1，因为 Ch7 的 Toy 示例规模小，已有 Toy-to-Affine conversion 可以复用大部分 tensor-to-memref、constant、print、return 的 lowering。等 naive GPU kernel 跑通后，再迁移到 Linalg 路线。

## 路线 1：直接 lower 到 gpu.launch

### 1. 新增 GPU lowering pass

新增文件：

```text
mlir/examples/toy/Ch7/mlir/LowerToGPU.cpp
```

建议不要从零写完整 conversion。更稳的方式是以 `LowerToAffineLoops.cpp` 为模板：

- 复用 tensor 到 memref 的 type conversion。
- 复用 `toy.constant`、`toy.print`、`toy.return`、`toy.transpose`、elementwise op 的 lowering。
- 替换 `MatMulOpLowering`：不再生成三层 `affine.for`，而是生成 `gpu.launch`。

新增声明：

```cpp
// include/toy/Passes.h
std::unique_ptr<mlir::Pass> createLowerToGPUPass();
```

### 2. 注册 GPU 相关 dialect

GPU pass 至少要注册这些 dialect：

```cpp
registry.insert<mlir::arith::ArithDialect,
                mlir::func::FuncDialect,
                mlir::gpu::GPUDialect,
                mlir::memref::MemRefDialect,
                mlir::scf::SCFDialect>();
```

如果 pass 里继续使用 affine load/store，需要保留 `affine::AffineDialect`。但 naive GPU kernel 内部建议用 `memref.load` / `memref.store`，后续 lower 到 GPU 后端更直接。

### 3. MatMulOpLowering 的目标 IR

naive 版本采用“一条 GPU thread 计算一个 C[i, j]”：

```text
thread global id x -> j
thread global id y -> i
每个 thread 串行循环 k，计算 C[i, j]
```

目标 IR 形态：

```mlir
%C = memref.alloc() : memref<MxNxf64>

%blockX = arith.constant 16 : index
%blockY = arith.constant 16 : index
%gridX = ceildiv(N, 16)
%gridY = ceildiv(M, 16)
%one = arith.constant 1 : index

gpu.launch
  blocks(%bx, %by, %bz) in (%gridX, %gridY, %one)
  threads(%tx, %ty, %tz) in (%blockX, %blockY, %one) {
    %j0 = arith.muli %bx, %blockX : index
    %j = arith.addi %j0, %tx : index
    %i0 = arith.muli %by, %blockY : index
    %i = arith.addi %i0, %ty : index

    %inM = arith.cmpi ult, %i, %M : index
    %inN = arith.cmpi ult, %j, %N : index
    %inBounds = arith.andi %inM, %inN : i1
    scf.if %inBounds {
      %sum = scf.for %k = %c0 to %K step %c1
              iter_args(%acc = %zero) -> f64 {
        %a = memref.load %A[%i, %k] : memref<MxKxf64>
        %b = memref.load %B[%k, %j] : memref<KxNxf64>
        %p = arith.mulf %a, %b : f64
        %s = arith.addf %acc, %p : f64
        scf.yield %s : f64
      }
      memref.store %sum, %C[%i, %j] : memref<MxNxf64>
    }
    gpu.terminator
  }
```

### 4. 计算 grid/block

固定 block 大小先用：

```text
blockX = 16
blockY = 16
blockZ = 1
```

grid 维度：

```text
gridX = ceildiv(N, blockX)
gridY = ceildiv(M, blockY)
gridZ = 1
```

在 MLIR 里可以用整数表达式生成 ceildiv：

```text
ceildiv(x, b) = (x + b - 1) / b
```

对应 ops：

```mlir
%tmp0 = arith.addi %x, %block : index
%tmp1 = arith.subi %tmp0, %c1 : index
%grid = arith.divui %tmp1, %block : index
```

如果 M/N/K 都是静态 shape，也可以先直接生成 `arith.constant` grid 值。Ch7 当前 Toy 更适合先要求静态 shape。

### 5. pass 管线接入 toyc

在 `toyc.cpp` 增加一个新的 emit action，例如：

```cpp
DumpMLIRGPU,
```

命令行选项：

```cpp
clEnumValN(DumpMLIRGPU, "mlir-gpu",
           "output the MLIR dump after GPU lowering")
```

pipeline 逻辑上放在 shape inference 之后、CPU affine lowering 之前：

```cpp
bool isLoweringToGPU = emitAction >= Action::DumpMLIRGPU;

if (enableOpt || isLoweringToAffine || isLoweringToGPU) {
  pm.addPass(mlir::createInlinerPass());
  mlir::OpPassManager &optPM = pm.nest<mlir::toy::FuncOp>();
  optPM.addPass(mlir::createCanonicalizerPass());
  optPM.addPass(mlir::toy::createShapeInferencePass());
  optPM.addPass(mlir::createCanonicalizerPass());
  optPM.addPass(mlir::createCSEPass());
}

if (isLoweringToGPU) {
  pm.addPass(mlir::toy::createLowerToGPUPass());
  mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
  optPM.addPass(mlir::createCanonicalizerPass());
  optPM.addPass(mlir::createCSEPass());
}
```

注意：如果 `DumpMLIRGPU` 的 enum 顺序放在 `DumpMLIRAffine` 后面，原来的 `isLoweringToAffine = emitAction >= DumpMLIRAffine` 会误触发 CPU affine lowering。需要拆开判断，避免 GPU 路径又进入 `createLowerToAffinePass()`。

### 6. CMake 接入

在 `mlir/examples/toy/Ch7/CMakeLists.txt` 里把新文件加到 `add_toy_chapter`：

```cmake
mlir/LowerToGPU.cpp
```

当前 CMake 已经链接了：

```cmake
${dialect_libs}
${conversion_libs}
${extension_libs}
```

通常足够覆盖 GPU dialect 和 conversion 依赖。如果改成显式依赖，可以关注：

```cmake
MLIRGPUDialect
MLIRGPUTransforms
MLIRGPUToNVVMTransforms
MLIRGPUToROCDLTransforms
MLIRGPUToLLVMTransforms
MLIRSCFToGPU
```

实际库名以当前 LLVM 21.1.8 源码里的 CMake target 为准。

## 路线 1 的完整 lowering 管线

### 只看 GPU dialect IR

```bash
toyc-ch7 input.toy -emit=mlir-gpu
```

期望看到：

```text
gpu.launch
memref.load
memref.store
scf.for
arith.mulf
arith.addf
```

并且不应该再看到：

```text
toy.matmul
```

### lower 到 NVVM/CUDA

如果 GPU IR 已经包含 `gpu.launch`，可以继续走 MLIR 自带 NVVM pipeline：

```bash
mlir-opt input-gpu.mlir \
  -gpu-lower-to-nvvm-pipeline="cubin-chip=sm_80 cubin-features=+ptx70 opt-level=3"
```

`gpu-lower-to-nvvm-pipeline` 会做这些事：

```text
gpu-kernel-outlining
gpu.module(convert-gpu-to-nvvm)
nvvm-attach-target
gpu-to-llvm
gpu-module-to-binary
```

也可以手动写成：

```bash
mlir-opt input-gpu.mlir \
  --pass-pipeline='builtin.module(
    gpu-kernel-outlining,
    nvvm-attach-target{chip=sm_80 features=+ptx70 O=3},
    gpu.module(convert-gpu-to-nvvm),
    gpu-to-llvm,
    gpu-module-to-binary
  )'
```

然后转 LLVM IR：

```bash
mlir-translate input-nvvm.mlir --mlir-to-llvmir -o input.ll
```

### lower 到 ROCDL/AMD

AMD 路线类似，但使用 ROCDL：

```text
gpu-kernel-outlining
rocdl-attach-target{chip=gfx90a}
gpu.module(convert-gpu-to-rocdl)
gpu-to-llvm
gpu-module-to-binary
```

### lower 到 SPIR-V

如果目标是 Vulkan/SYCL/OpenCL 风格后端，可以走：

```text
spirv-attach-target
convert-gpu-to-spirv
gpu.module(spirv.module(...))
gpu-to-llvm
gpu-module-to-binary
```

Toy 教学路径建议先做 CUDA/NVVM，因为 `mlir/docs/Dialects/GPU.md` 里已经有 `gpu-lower-to-nvvm-pipeline` 的默认管线。

## Host/device 内存问题

`gpu.launch` 本身表达的是 kernel launch，但数据在哪里取决于后续 runtime lowering。

最小教学版本可以先让 lowering 生成普通 memref，然后依赖 `gpu-to-llvm` 和 runner/runtime 处理 launch。要真正 JIT 运行，需要额外考虑：

- host memref 是否要 `gpu.host_register`
- 是否显式生成 `gpu.alloc`
- 是否显式生成 `gpu.memcpy` host -> device
- kernel 结束后是否 `gpu.memcpy` device -> host
- 是否链接 CUDA/ROCm runner wrapper library

如果只是 `-emit=mlir-gpu` 或 `mlir-opt` 验证 IR，不必一开始解决完整运行时。

完整运行时 IR 更像：

```mlir
%A_dev = gpu.alloc() : memref<MxKxf64>
%B_dev = gpu.alloc() : memref<KxNxf64>
%C_dev = gpu.alloc() : memref<MxNxf64>
gpu.memcpy %A_dev, %A : memref<MxKxf64>, memref<MxKxf64>
gpu.memcpy %B_dev, %B : memref<KxNxf64>, memref<KxNxf64>
gpu.launch ... args use %A_dev, %B_dev, %C_dev ...
gpu.memcpy %C, %C_dev : memref<MxNxf64>, memref<MxNxf64>
```

## 路线 2：toy.matmul lower 到 linalg.matmul

如果目标不是教学，而是后续优化能力，推荐把 `toy.matmul` 先 lower 成 `linalg.matmul`。

优势：

- `linalg.matmul` 是 MLIR 标准结构化 op。
- 可以直接使用 tiling、fusion、vectorization、bufferization。
- 可以接 Transform dialect/NVGPU/Tensor Core 路线。
- 更容易复用已有优化，而不是维护手写 matmul kernel。

目标 IR：

```mlir
%init = tensor.empty() : tensor<MxNxf64>
%zero = arith.constant 0.0 : f64
%filled = linalg.fill ins(%zero : f64)
                      outs(%init : tensor<MxNxf64>) -> tensor<MxNxf64>
%result = linalg.matmul
  ins(%lhs, %rhs : tensor<MxKxf64>, tensor<KxNxf64>)
  outs(%filled : tensor<MxNxf64>) -> tensor<MxNxf64>
```

随后 pipeline：

```text
toy.matmul
  -> linalg.matmul
  -> one-shot-bufferize
  -> tile/fuse/vectorize
  -> map parallel loops to gpu
  -> gpu.launch
  -> gpu-kernel-outlining
  -> NVVM/ROCDL/SPIR-V
```

一个简化版 pass 顺序：

```text
shape-inference
toy-to-linalg
one-shot-bufferize
linalg-tile
convert-linalg-to-parallel-loops
gpu-map-parallel-loops
convert-parallel-loops-to-gpu
gpu-kernel-outlining
gpu-lower-to-nvvm-pipeline
```

实际 pass 名称和参数要根据 LLVM 21.1.8 中 `mlir-opt --help` 和本地构建启用的组件确认。

## naive GPU MatMul 的正确性检查

建议新增一个 Toy 输入：

```toy
def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b<3, 2> = [[7, 8], [9, 10], [11, 12]];
  var c = matmul(a, b);
  print(c);
}
```

期望输出：

```text
[[58, 64], [139, 154]]
```

检查顺序：

1. `-emit=mlir`：确认生成 `toy.matmul`。
2. `-emit=mlir-gpu`：确认 `toy.matmul` 消失，出现 `gpu.launch`。
3. `mlir-opt -verify-each`：确认 GPU IR 合法。
4. `gpu-lower-to-nvvm-pipeline`：确认可以 lower 到 NVVM/LLVM。
5. 如果要运行，再处理 CUDA/ROCm runtime、设备内存和 wrapper 链接。

## 常见坑

- enum 顺序导致 GPU 路径意外执行 CPU affine lowering。
- `toy.matmul` 的 shape 还是 unranked，导致无法取 M/N/K。必须先跑 shape inference。
- kernel 内用了 `affine.load/store`，后续 GPU lowering 不一定顺畅。naive 版本优先用 `memref.load/store`。
- 没有处理边界条件：当 M/N 不是 block size 的整数倍，越界 thread 会访问非法地址。
- `f64` 在部分 GPU 上性能差，甚至需要设备能力支持。Toy 当前使用 `f64`，教学上可以保留；性能实验建议扩展 Toy 类型系统支持 `f32`。
- `gpu.launch` 只是并行 IR，不等于已经生成可运行 cubin/hsaco。还需要 outlining、target attach、device dialect conversion、host runtime lowering、binary serialization。
- 如果目标是高性能 matmul，不要长期维护 naive kernel，应转向 `linalg.matmul + tiling/vector/nvgpu` 或直接调用 BLAS/cuBLAS 类库。

## 最小改动清单

```text
include/toy/Passes.h
  + createLowerToGPUPass()

mlir/LowerToGPU.cpp
  + ToyToGPU pass
  + MatMulOpLowering -> gpu.launch
  + 复用/移植 ToyToAffine 的其他 conversion patterns

toyc.cpp
  + DumpMLIRGPU action
  + -emit=mlir-gpu
  + shape inference 后运行 createLowerToGPUPass()
  + 避免 GPU 路径误进 createLowerToAffinePass()

CMakeLists.txt
  + mlir/LowerToGPU.cpp
  + 必要 GPU/SCF/MemRef/Conversion 依赖

测试文件
  + matmul.toy
  + FileCheck: toy.matmul 消失，gpu.launch 出现
```

## 建议推进顺序

1. 先做 `-emit=mlir-gpu`，只验证 IR。
2. 再接 `gpu-kernel-outlining`，确认生成 `gpu.module` / `gpu.func` / `gpu.launch_func`。
3. 再接 `gpu-lower-to-nvvm-pipeline`，确认能生成 NVVM/LLVM IR。
4. 最后处理真正运行：CUDA/ROCm runtime、device memory、JIT/link wrapper。
5. naive 版本稳定后，再做 `toy.matmul -> linalg.matmul` 路线，用标准 MLIR 优化替换手写 kernel。
