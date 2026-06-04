# Stage 6: Promote Matmul Tiles To Workgroup Memory

目标：把 global memory tile 显式提升到 workgroup/shared memory。

## 已新增 Pass

```text
toy-matmul-promote-workgroup-memory
```

实现文件：

```text
mlir/examples/toy/Ch7/mlir/MatMulPromoteWorkgroupMemory.cpp
mlir/examples/toy/Ch7/include/toy/Passes.h
mlir/examples/toy/Ch7/toyc.cpp
mlir/examples/toy/Ch7/CMakeLists.txt
```

新增 emit action：

```text
-emit=mlir-gpu-workgroup-matmul
```

pipeline 形状：

```text
Toy/MLIR input
  -> inliner
  -> canonicalizer
  -> toy shape inference
  -> canonicalizer
  -> CSE
  -> toy-matmul-to-scf
  -> toy-matmul-tile-loops
  -> toy-matmul-reorder-tiled-loops
  -> toy-matmul-map-to-gpu
  -> toy-matmul-promote-workgroup-memory
  -> func-level canonicalizer
  -> func-level CSE
  -> dump MLIR
```

## 输入

阶段 5 生成的 naive GPU launch。Stage 6 不再直接匹配阶段 4 的 reordered
SCF loop nest，而是在 Stage 5 的 `gpu.launch` 上做 workgroup memory promotion。

```mlir
gpu.launch blocks(...) threads(16, 16, 1) {
  row = blockIdx.y * 16 + threadIdx.y
  col = blockIdx.x * 16 + threadIdx.x
  if row < M && col < N
    for ko step 16
      for ki step 1
        k = ko + ki
        load A[row, k]
        load B[k, col]
        accumulate
    store C[row, col]
}
```

这一点是阶段边界：

```text
Stage 5:
  reordered tiled loop -> naive gpu.launch

Stage 6:
  naive gpu.launch -> workgroup gpu.launch
```

Stage 6 不绕过 Stage 5，也不重复做 loop-to-GPU mapping。

## 输出

```mlir
gpu.launch workgroup(%tileA, %tileB) {
  for ko step 16
    guarded load A[row, ko + threadIdx.x] -> tileA[threadIdx.y, threadIdx.x]
    guarded load B[ko + threadIdx.y, col] -> tileB[threadIdx.y, threadIdx.x]
    gpu.barrier

    for ki = 0 to 16
      load tileA[threadIdx.y, ki]
      load tileB[ki, threadIdx.x]
      accumulate

    gpu.barrier
}
```

## 实施步骤

```text
1. 已在 gpu.launch 上新增两个 workgroup attribution。
2. tile 类型固定为 memref<16x16xf64, #gpu.address_space<workgroup>>。
3. 已用 threadIdx.x/threadIdx.y 协作加载 A/B tile。
4. 已对 A/B tile load 分别生成边界判断。
5. 越界线程写 0 到 tile。
6. shared tile store 后已插入 gpu.barrier。
7. inner ki loop 已改为读取 tileA/tileB。
8. inner reduction 后已再插入 gpu.barrier。
9. 最终 C store 仍保留在 row/col inBounds guard 内。
```

## 注意事项

```text
barrier 必须所有线程都执行。
边界线程要写 0 到 tile，不能跳过 store。
workgroup attribution 在 gpu.launch/gpu.func 头部声明。
global load 必须在第一个 barrier 前。
shared load 必须在第一个 barrier 后。
```

## 验证

必须能看到：

```text
workgroup memref<16x16xf64, #gpu.address_space<workgroup>>
两个 gpu.barrier
global load 在 barrier 前
shared load 在 barrier 后
tileA[threadIdx.y, ki]
tileB[ki, threadIdx.x]
```

新增测试：

```text
mlir/test/Examples/Toy/Ch7/matmul64-gpu-workgroup.mlir
mlir/test/Examples/Toy/Ch7/matmul70-gpu-workgroup.mlir
```

## 当前实现范围

`MatMulPromoteWorkgroupMemory.cpp` 是一个 launch-level rewrite pass：

```text
只匹配：
  Stage 5 输出的 naive gpu.launch：
    - blocks/grid 已由 Stage 5 计算。
    - threads 为 16 x 16 x 1。
    - launch 内部保留 ko/ki reduction。
    - launch 尚未包含 workgroup attribution。

改写为：
  gpu.launch blocks(gridX, gridY, 1) threads(16, 16, 1)
    workgroup(tileA: memref<16x16xf64, #gpu.address_space<workgroup>>,
              tileB: memref<16x16xf64, #gpu.address_space<workgroup>>)
    for ko step 16:
      guarded global load A[row, ko + threadIdx.x] or 0 -> tileA[threadIdx.y, threadIdx.x]
      guarded global load B[ko + threadIdx.y, col] or 0 -> tileB[threadIdx.y, threadIdx.x]
      gpu.barrier
      for ki = 0 to 16:
        acc += tileA[threadIdx.y, ki] * tileB[ki, threadIdx.x]
      gpu.barrier
    if row < M && col < N:
      store C[row, col]
```

当前不处理：

```text
device memory allocation/copy
gpu.module outlining
NVVM lowering
host runtime lowering
```

这些仍留给后续阶段或旧 `toy-to-gpu` 路径。

## 验证状态

已新增测试：

```text
mlir/test/Examples/Toy/Ch7/matmul64-gpu-workgroup.mlir
mlir/test/Examples/Toy/Ch7/matmul70-gpu-workgroup.mlir
```

测试命令：

```bash
./build/bin/llvm-lit mlir/test/Examples/Toy/Ch7/matmul64-gpu-workgroup.mlir
./build/bin/llvm-lit mlir/test/Examples/Toy/Ch7/matmul70-gpu-workgroup.mlir
```

或手动查看：

```bash
./build/bin/toyc-ch7 mlir/test/Examples/Toy/Ch7/matmul64-gpu-workgroup.mlir \
  -emit=mlir-gpu-workgroup-matmul -x=mlir
```

当前执行状态：

```text
已完成：
  - 新增 MatMulPromoteWorkgroupMemory.cpp。
  - 新增 createMatMulPromoteWorkgroupMemoryPass() 声明。
  - 新增 -emit=mlir-gpu-workgroup-matmul。
  - CMake 已加入新源文件。
  - 新增 matmul64-gpu-workgroup.mlir。
  - 新增 matmul70-gpu-workgroup.mlir。

未在本地执行：
  - ninja -C build toyc-ch7
  - toyc-ch7 -emit=mlir-gpu-workgroup-matmul
  - llvm-lit matmul64-gpu-workgroup.mlir
  - llvm-lit matmul70-gpu-workgroup.mlir

未执行原因：
  - 延续前置要求：本地不要编译。
```
