# Stage 5: Map Tiled Matmul To GPU

目标：把 tile loop 映射到 GPU block，把 point loop 映射到 GPU thread。

这一阶段可以先生成 naive GPU kernel，不引入 workgroup memory。

## 建议新增 Pass

```text
toy-matmul-map-to-gpu
```

建议实现文件：

```text
mlir/examples/toy/Ch7/mlir/MatMulMapToGPU.cpp
mlir/examples/toy/Ch7/include/toy/Passes.h
mlir/examples/toy/Ch7/toyc.cpp
mlir/examples/toy/Ch7/CMakeLists.txt
```

## 映射关系

```text
jo / 16 -> blockIdx.x
io / 16 -> blockIdx.y
ji      -> threadIdx.x
ii      -> threadIdx.y
```

输出形状：

```mlir
gpu.launch blocks(...) threads(...) {
  %row = blockIdx.y * 16 + threadIdx.y
  %col = blockIdx.x * 16 + threadIdx.x
  ...
}
```

naive kernel 形状：

```text
for k = 0..K
  load A[row, k]
  load B[k, col]
  acc += ...
store C[row, col]
```

## 实施步骤

```text
1. 匹配阶段 4 的 reordered tiled loop nest。
2. 用 outer tile loop bounds 计算 gridX/gridY。
3. 创建 gpu.launch。
4. threads 固定为 16 x 16 x 1。
5. 在 launch body 中用 block/thread id 重建 row/col。
6. 保留 row < M 和 col < N 边界判断。
7. 生成 naive k reduction，直接读 global memory。
8. 在 inBounds 内 store C。
```

## 边界

```text
允许：
  - 生成 gpu.launch。
  - 使用 blockIdx/threadIdx。
  - 生成 naive global-memory matmul kernel。

不允许：
  - 生成 workgroup memory。
  - 生成 gpu.barrier。
  - 做 shared memory promotion。
  - 拆 device memory 管理。
```

## 验证

必须能看到：

```text
gpu.launch
block id
thread id
row = blockIdx.y * 16 + threadIdx.y
col = blockIdx.x * 16 + threadIdx.x
global memref.load A[row, k]
global memref.load B[k, col]
```

不能出现：

```text
workgroup
gpu.barrier
```

建议新增测试：

```text
mlir/test/Examples/Toy/Ch7/matmul64-gpu-naive.mlir
```
