# Stage 6: Promote Matmul Tiles To Workgroup Memory

目标：把 global memory tile 显式提升到 workgroup/shared memory。

## 建议新增 Pass

```text
toy-matmul-promote-workgroup-memory
```

建议实现文件：

```text
mlir/examples/toy/Ch7/mlir/MatMulPromoteWorkgroupMemory.cpp
mlir/examples/toy/Ch7/include/toy/Passes.h
mlir/examples/toy/Ch7/toyc.cpp
mlir/examples/toy/Ch7/CMakeLists.txt
```

## 输入

阶段 5 的 naive GPU kernel：

```mlir
gpu.launch {
  for ko step 16
    for ki step 1
      load A[row, ko + ki]
      load B[ko + ki, col]
      accumulate
}
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
1. 在 gpu.launch 上新增两个 workgroup attribution。
2. tile 类型固定为 memref<16x16xf64, #gpu.address_space<workgroup>>。
3. 用 threadIdx.x/threadIdx.y 协作加载 A/B tile。
4. 对 A/B tile load 分别生成边界判断。
5. 越界线程写 0 到 tile。
6. shared tile store 后插入 gpu.barrier。
7. inner ki loop 改为读取 tileA/tileB。
8. inner reduction 后再插入 gpu.barrier。
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

建议新增测试：

```text
mlir/test/Examples/Toy/Ch7/matmul64-gpu-workgroup.mlir
mlir/test/Examples/Toy/Ch7/matmul70-gpu-workgroup.mlir
```
