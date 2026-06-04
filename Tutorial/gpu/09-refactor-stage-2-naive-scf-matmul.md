# Stage 2: toy.matmul To Naive SCF Matmul

目标：不要直接从 `toy.matmul` 生成最终 GPU kernel，而是先生成一个普通、清晰、可匹配的 matmul loop nest。

这一阶段不涉及 GPU，也不涉及 shared memory。

## 已新增 Pass

```text
toy-matmul-to-scf
```

实现文件：

```text
mlir/examples/toy/Ch7/mlir/MatMulToSCF.cpp
mlir/examples/toy/Ch7/include/toy/Passes.h
mlir/examples/toy/Ch7/toyc.cpp
mlir/examples/toy/Ch7/CMakeLists.txt
```

新增 emit action：

```text
-emit=mlir-scf-matmul
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
  -> func-level canonicalizer
  -> func-level CSE
  -> dump MLIR
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
1. 已新增 createMatMulToSCFPass() 声明。
2. 已新增 MatMulToSCF.cpp。
3. 已在 CMakeLists.txt 加入 MatMulToSCF.cpp。
4. 已在 toyc.cpp 新增 -emit=mlir-scf-matmul。
5. 已让 -emit=mlir-scf-matmul 跑 inline + shape inference。
6. 已为 result tensor 创建 host memref.alloc。
7. 已把 lhs/rhs operand 转成当前 lowering pipeline 可使用的 memref。
8. 已生成 i/j/k 三层 scf.for。
9. 已用 scf.for iter_args 表达 sum accumulation。
10. 已用输出 memref 替换 toy.matmul。
```

## 当前实现范围

`MatMulToSCF.cpp` 当前不是只转换单个 `toy.matmul`，而是做一个可停在 SCF/memref 层的 Toy partial lowering：

```text
toy.constant   -> memref.alloc + memref.store
toy.add        -> scf loop + memref.load/store + arith.addf
toy.mul        -> scf loop + memref.load/store + arith.mulf
toy.neg        -> scf loop + memref.load/store + arith.negf
toy.transpose  -> scf loop + memref.load/store
toy.matmul     -> naive i/j/k scf.for matmul
toy.func main  -> func.func main
toy.return     -> func.return
toy.print      -> 保留 toy.print，但 operand 更新为 memref
```

这样做是为了让 `toy.matmul` 的输入在同一个 pass 内已经是 memref，输出也能被 `toy.print` 接住。

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

已新增测试：

```text
mlir/test/Examples/Toy/Ch7/matmul64-scf-lowering.mlir
```

测试命令：

```bash
./build/bin/llvm-lit mlir/test/Examples/Toy/Ch7/matmul64-scf-lowering.mlir
```

或手动查看：

```bash
./build/bin/toyc-ch7 mlir/test/Examples/Toy/Ch7/matmul64-scf-lowering.mlir \
  -emit=mlir-scf-matmul -x=mlir
```

当前执行状态：

```text
已完成：
  - 新增 MatMulToSCF.cpp。
  - 新增 createMatMulToSCFPass() 声明。
  - 新增 -emit=mlir-scf-matmul。
  - CMake 已加入新源文件。
  - 新增 matmul64-scf-lowering.mlir FileCheck 测试。

未在本地执行：
  - ninja -C build toyc-ch7
  - toyc-ch7 -emit=mlir-scf-matmul
  - llvm-lit matmul64-scf-lowering.mlir

未执行原因：
  - 本轮延续前置要求：本地不要编译。
```
