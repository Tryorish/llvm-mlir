# Stage 2: toy.matmul To Naive SCF Matmul

目标：不要直接从 `toy.matmul` 生成最终 GPU kernel，而是先生成一个普通、清晰、可匹配的 matmul loop nest。

这一阶段不涉及 GPU，也不涉及 shared memory。

## 建议新增 Pass

```text
toy-matmul-to-scf
```

建议实现文件：

```text
mlir/examples/toy/Ch7/mlir/MatMulToSCF.cpp
mlir/examples/toy/Ch7/include/toy/Passes.h
mlir/examples/toy/Ch7/toyc.cpp
mlir/examples/toy/Ch7/CMakeLists.txt
```

## 输入

```mlir
%C = toy.matmul %A, %B : tensor<MxKxf64>, tensor<KxNxf64> to tensor<MxNxf64>
```

输入前置条件：

```text
- shape inference 已经运行。
- lhs/rhs/result 都有 ranked tensor type。
- 本阶段只先支持 rank-2 f64 matmul。
```

## 输出

核心输出形状：

```mlir
%C = memref.alloc() : memref<MxNxf64>
scf.for %i = %c0 to %M step %c1 {
  scf.for %j = %c0 to %N step %c1 {
    %sum = scf.for %k = %c0 to %K step %c1 iter_args(%acc = %zero) -> (f64) {
      %a = memref.load %A[%i, %k]
      %b = memref.load %B[%k, %j]
      %mul = arith.mulf %a, %b
      %next = arith.addf %acc, %mul
      scf.yield %next : f64
    }
    memref.store %sum, %C[%i, %j]
  }
}
```

## 实施步骤

```text
1. 新增 createMatMulToSCFPass() 声明。
2. 新增 pass 实现，只匹配 toy.matmul。
3. 为 result tensor 创建 host memref.alloc。
4. 把 lhs/rhs operand 转成当前 lower pipeline 可使用的 memref。
5. 生成 i/j/k 三层 scf.for。
6. 用 scf.for iter_args 表达 sum accumulation。
7. 用输出 memref 替换 toy.matmul。
8. 接入一个只停在 SCF 的 emit action 或测试入口。
```

## 边界

```text
允许：
  - 生成 memref.alloc。
  - 生成 scf.for / memref.load / memref.store / arith.mulf / arith.addf。

不允许：
  - 生成 gpu.launch。
  - 生成 workgroup memory。
  - 做 tiling。
  - 做 block/thread mapping。
  - 插入 gpu.alloc/gpu.memcpy/gpu.dealloc。
```

## 验证

必须能看到：

```text
i/j/k 三层 scf.for
memref.load A[i, k]
memref.load B[k, j]
arith.mulf
arith.addf
memref.store C[i, j]
```

不能出现：

```text
gpu.launch
workgroup
gpu.barrier
gpu.alloc
gpu.memcpy
```

建议新增测试：

```text
mlir/test/Examples/Toy/Ch7/matmul64-scf-lowering.mlir
```
