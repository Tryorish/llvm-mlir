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
  -> toy-matmul-promote-workgroup-memory
  -> func-level canonicalizer
  -> func-level CSE
  -> dump MLIR
```

## 输入

阶段 4 的 reordered tiled loop nest。当前实现直接从 reordered loop 生成 workgroup 版本的 `gpu.launch`，不先经过 Stage 5 的 naive launch。

```mlir
for io
  for jo
    for ii
      for ji
        for ko
          for ki
            accumulate
```

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

`MatMulPromoteWorkgroupMemory.cpp` 是一个 loop-level rewrite pass：

```text
只匹配：
  Stage 4 输出的 io/jo/ii/ji/ko/ki reordered tiled matmul loop nest。

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
