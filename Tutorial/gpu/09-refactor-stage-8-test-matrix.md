# Stage 8: Test Matrix Cleanup

目标：为前面每个阶段建立清晰、可维护的测试矩阵。

## 阶段测试

每个阶段应有对应测试：

```text
matmul64-scf-lowering.mlir
matmul64-tiled-loops.mlir
matmul64-reordered-tiled-loops.mlir
matmul64-gpu-naive.mlir
matmul64-gpu-workgroup.mlir
matmul64-gpu-device-memory.mlir
matmul64-gpu-outlining.mlir
matmul64-gpu-nvvm.mlir
matmul64-gpu-binary.mlir
matmul64-gpu-host.mlir
matmul64-llvm-gpu.mlir
```

保留已有历史测试时，可以逐步重命名，不要一次性打乱所有 FileCheck。

## 覆盖矩阵

建议覆盖：

```text
2x2 小矩阵：
  用于 gpu-jit 输出检查。

64x64 整除 16：
  用于 shared memory tiling 主路径。

非 16 倍数矩阵：
  用于边界判断。
```

建议新增：

```text
matmul2-codegen.toy
matmul64.toy
matmul70.toy
```

## FileCheck 重点

SCF 阶段：

```text
scf.for 三层 loop
memref.load
memref.store
没有 gpu.launch
```

Tiling 阶段：

```text
outer loop step 16
inner loop step 1
边界判断
没有 gpu.launch
```

GPU mapping 阶段：

```text
gpu.launch
block/thread id
没有 workgroup
没有 gpu.barrier
```

Workgroup 阶段：

```text
workgroup memref<16x16xf64, #gpu.address_space<workgroup>>
两个 gpu.barrier
global load 在 barrier 前
shared load 在 barrier 后
```

Host/runtime 阶段：

```text
gpu.binary
host runtime calls
没有 builtin.unrealized_conversion_cast
```

## 执行建议

每一步完成后至少跑：

```bash
ninja -C build toyc-ch7
./build/bin/llvm-lit mlir/test/Examples/Toy/Ch7/matmul64-gpu-lowering.mlir
```

完整链路稳定后再跑：

```bash
./build/bin/toyc-ch7 mlir/test/Examples/Toy/Ch7/matmul-codegen.toy -emit=mlir-gpu
./build/bin/toyc-ch7 mlir/test/Examples/Toy/Ch7/matmul-codegen.toy -emit=mlir-gpu-nvvm
./build/bin/toyc-ch7 mlir/test/Examples/Toy/Ch7/matmul-codegen.toy -emit=llvm-gpu
```

如果 CUDA runtime 可用，再跑：

```bash
./build/bin/toyc-ch7 mlir/test/Examples/Toy/Ch7/matmul-codegen.toy \
  -emit=gpu-jit \
  -shared-libs=$PWD/build/lib/libmlir_cuda_runtime.so
```
