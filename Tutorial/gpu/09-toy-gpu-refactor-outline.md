# Toy GPU Lowering Refactor Outline

本文是 GPU lowering 后续重构路线的索引页。每个阶段已经拆成单独文档，便于逐阶段执行、验证和记录状态。

前八个阶段已经证明了完整链路：

```text
toy.matmul
  -> gpu.launch
  -> gpu.module / gpu.func
  -> NVVM
  -> gpu.binary
  -> host runtime calls
  -> gpu-jit
  -> shared memory tiling
```

重构前实现主要集中在 `LowerToGPU.cpp` 的 `MatMulOpLowering` 里。它适合教学和快速验证，但把太多职责写在一个 rewrite pattern 中：

```text
识别 toy.matmul
创建 host/device buffer
插入 gpu.alloc / gpu.memcpy / gpu.dealloc
创建 gpu.launch
计算 block/thread mapping
生成 shared memory tiling
处理边界判断
维护 async token 依赖链
```

当前重构已经把 matmul GPU lowering 拆成 Stage 2-7，并且完整 GPU action
已经统一复用这条 staged pipeline：

```text
toy-matmul-to-scf
  -> toy-matmul-tile-loops
  -> toy-matmul-reorder-tiled-loops
  -> toy-matmul-map-to-gpu
  -> toy-matmul-promote-workgroup-memory
  -> toy-gpu-insert-device-memory
  -> gpu outlining / NVVM / binary / host runtime / JIT
```

也就是说，`-emit=mlir-gpu`、`-emit=mlir-gpu-nvvm`、`-emit=mlir-gpu-host`、
`-emit=llvm-gpu` 和 `-emit=gpu-jit` 现在不再走旧的 monolithic matmul
lowering，而是先走分阶段 pipeline，再接 MLIR 标准 GPU 后端。

## 总体原则

每次只做一类变化，并保证每一步都可以单独验证。

```text
先拆 helper，不改 IR
再拆 pass，不改最终行为
最后再引入更正规的 linalg / transform 路线
```

不要一开始就把当前实现全部替换成标准 MLIR pipeline。当前手写 lowering 仍然有价值，因为它能清楚展示 GPU dialect、NVVM lowering 和 runtime lowering 的机制。

## 阶段文档

```text
Stage 1:
  09-refactor-stage-1-helper-split.md
  函数级拆分，保持 IR 完全不变。

Stage 2:
  09-refactor-stage-2-naive-scf-matmul.md
  toy.matmul 先降成 naive scf matmul loop。

Stage 3:
  09-refactor-stage-3-loop-tiling.md
  loop split / tiling。

Stage 4:
  09-refactor-stage-4-loop-reorder.md
  loop reorder。

Stage 5:
  09-refactor-stage-5-gpu-mapping.md
  GPU block/thread mapping。

Stage 6:
  09-refactor-stage-6-workgroup-promotion.md
  shared memory / workgroup memory promotion。

Stage 7:
  09-refactor-stage-7-device-memory-management.md
  拆 device memory 管理。

Stage 8:
  09-refactor-stage-8-test-matrix.md
  整理测试矩阵。

Stage 9:
  09-refactor-stage-9-linalg-route.md
  考虑 linalg-based pipeline。
```

## 推荐执行顺序

```text
1. 先把 MatMulOpLowering helper 化
2. toy.matmul 先降成 naive scf matmul loop
3. loop split / tiling
4. loop reorder
5. GPU mapping
6. shared memory promotion
7. device memory 管理
8. 扩展测试覆盖非 16 倍数矩阵
9. 最后再调研 linalg-based pipeline
```

每一步完成后，都至少跑：

```bash
ninja -C build toyc-ch7
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

本轮只做文档拆分和 Stage 1 helper 化，不在本地执行编译。
