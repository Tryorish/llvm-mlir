# Stage 5: Map Tiled Matmul To GPU

目标：把 tile loop 映射到 GPU block，把 point loop 映射到 GPU thread。

这一阶段可以先生成 naive GPU kernel，不引入 workgroup memory。

## 已新增 Pass

```text
toy-matmul-map-to-gpu
```

实现文件：

```text
mlir/examples/toy/Ch7/mlir/MatMulMapToGPU.cpp
mlir/examples/toy/Ch7/include/toy/Passes.h
mlir/examples/toy/Ch7/toyc.cpp
mlir/examples/toy/Ch7/CMakeLists.txt
```

新增 emit action：

```text
-emit=mlir-gpu-naive-matmul
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
  -> func-level canonicalizer
  -> func-level CSE
  -> dump MLIR
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
1. 已匹配阶段 4 的 reordered tiled loop nest。
2. 已用 outer tile loop bounds 计算 gridX/gridY。
3. 已创建 gpu.launch。
4. threads 固定为 16 x 16 x 1。
5. 已在 launch body 中用 block/thread id 重建 row/col。
6. 已保留 row < M 和 col < N 边界判断。
7. 已生成 naive k reduction，直接读 global memory。
8. 已在 inBounds 内 store C。
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

新增测试：

```text
mlir/test/Examples/Toy/Ch7/matmul64-gpu-naive.mlir
```

## 当前实现范围

`MatMulMapToGPU.cpp` 是一个 loop-level rewrite pass：

```text
只匹配：
  Stage 4 输出的 io/jo/ii/ji/ko/ki reordered tiled matmul loop nest。

改写为：
  gpu.launch blocks(gridX, gridY, 1) threads(16, 16, 1)
    row = blockIdx.y * 16 + threadIdx.y
    col = blockIdx.x * 16 + threadIdx.x
    if row < M && col < N:
      for k = 0 to K step 1:
        acc += A[row, k] * B[k, col]
      store C[row, col]
```

当前不处理：

```text
device memory allocation/copy
gpu.module outlining
NVVM lowering
workgroup memory
gpu.barrier
```

这些仍留给后续阶段或旧 `toy-to-gpu` 路径。

## 验证状态

已新增测试：

```text
mlir/test/Examples/Toy/Ch7/matmul64-gpu-naive.mlir
```

测试命令：

```bash
./build/bin/llvm-lit mlir/test/Examples/Toy/Ch7/matmul64-gpu-naive.mlir
```

或手动查看：

```bash
./build/bin/toyc-ch7 mlir/test/Examples/Toy/Ch7/matmul64-gpu-naive.mlir \
  -emit=mlir-gpu-naive-matmul -x=mlir
```

当前执行状态：

```text
已完成：
  - 新增 MatMulMapToGPU.cpp。
  - 新增 createMatMulMapToGPUPass() 声明。
  - 新增 -emit=mlir-gpu-naive-matmul。
  - CMake 已加入新源文件。
  - 新增 matmul64-gpu-naive.mlir。

未在本地执行：
  - ninja -C build toyc-ch7
  - toyc-ch7 -emit=mlir-gpu-naive-matmul
  - llvm-lit matmul64-gpu-naive.mlir

未执行原因：
  - 延续前置要求：本地不要编译。
```
