# Stage 3: Matmul Loop Split And Tiling

目标：把 naive matmul loop 拆成 tile loop 和 point loop。

这一阶段仍然不生成 GPU，只把 loop nest 整理成后续 mapping 可识别的结构。

## 已新增 Pass

```text
toy-matmul-tile-loops
```

实现文件：

```text
mlir/examples/toy/Ch7/mlir/MatMulTileLoops.cpp
mlir/examples/toy/Ch7/include/toy/Passes.h
mlir/examples/toy/Ch7/toyc.cpp
mlir/examples/toy/Ch7/CMakeLists.txt
```

新增 emit action：

```text
-emit=mlir-tiled-matmul
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
  -> func-level canonicalizer
  -> func-level CSE
  -> dump MLIR
```

## 核心变换

```text
i -> io, ii
j -> jo, ji
k -> ko, ki
```

输入形状：

```text
for i
  for j
    for k
```

输出形状：

```text
for io step 16
  for jo step 16
    for ko step 16
      for ii step 1
        for ji step 1
          for ki step 1
```

索引关系：

```text
i = io + ii
j = jo + ji
k = ko + ki
```

## 边界处理

从本阶段开始显式出现边界判断：

```text
i < M
j < N
k < K
```

建议先使用 `arith.cmpi ult` 和 `scf.if`，保证非 16 倍数矩阵能表达安全访问。

当前实现：

```text
i/j 外层边界：
  inM = i < M
  inN = j < N
  inMN = inM && inN
  scf.if inMN { ... compute and store partial C[i, j] ... }

k 内层边界：
  inK = k < K
  lhsInBounds = inM && inK
  rhsInBounds = inK && inN
  越界 load yield 0.0
```

注意：当前 loop 顺序是 `io/jo/ko/ii/ji/ki`。这一阶段只负责 split/tile，不做 reorder。为了保持跨 `ko` tile 的累加语义，每个 `(i, j)` 在 `ko == 0` 时从 `0.0` 开始，否则从当前输出 `C[i, j]` 读回 partial sum 继续累加，并在每个 `ko` tile 后写回 partial sum。

## 实施步骤

```text
1. 已定义 tile size，第一版固定为 16。
2. 已匹配阶段 2 生成的 canonical matmul loop nest。
3. 已拆分 i/j/k 三个 induction variable。
4. 已用 io+ii、jo+ji、ko+ki 替换原 load/store index。
5. 已在 load/store 前插入边界判断。
6. 已保留 accumulation 语义不变：跨 `ko` tile 通过输出 memref 暂存 partial sum。
7. 输出仍是 host-side SCF/memref IR。
8. 已新增 -emit=mlir-tiled-matmul。
9. 已新增 64x64 和 70x70 FileCheck 测试。
```

## 当前实现范围

`MatMulTileLoops.cpp` 是一个保守的 rewrite pass：

```text
只匹配第二阶段输出的标准形状：
  scf.for i step 1
    scf.for j step 1
      scf.for k step 1 iter_args(acc)
        memref.load lhs[i, k]
        memref.load rhs[k, j]
        arith.mulf
        arith.addf
        scf.yield
      memref.store sum, out[i, j]

不匹配：
  - 非 matmul 的 scf.for。
  - 已经 tiled 的 loop。
  - 输出 memref 和输入 memref alias 的情况。
  - 不是 step 1 的 naive loop。
```

## 边界

```text
允许：
  - 生成 tile loop。
  - 生成 point loop。
  - 生成边界判断。

不允许：
  - 生成 gpu.launch。
  - 生成 workgroup memory。
  - 生成 gpu.barrier。
  - 改 device memory 管理。
```

## 验证

必须能看到：

```text
外层 i/j/k tile loop step 16
内层 i/j/k point loop step 1
i = io + ii
j = jo + ji
k = ko + ki
显式 i < M, j < N, k < K 边界判断
每个 `ko` tile 后写回 partial C[i, j]
```

不能出现：

```text
gpu.launch
workgroup
gpu.barrier
```

已新增测试：

```text
mlir/test/Examples/Toy/Ch7/matmul64-tiled-loops.mlir
mlir/test/Examples/Toy/Ch7/matmul70-tiled-loops.mlir
```

测试命令：

```bash
./build/bin/llvm-lit mlir/test/Examples/Toy/Ch7/matmul64-tiled-loops.mlir
./build/bin/llvm-lit mlir/test/Examples/Toy/Ch7/matmul70-tiled-loops.mlir
```

或手动查看：

```bash
./build/bin/toyc-ch7 mlir/test/Examples/Toy/Ch7/matmul64-tiled-loops.mlir \
  -emit=mlir-tiled-matmul -x=mlir
```

当前执行状态：

```text
已完成：
  - 新增 MatMulTileLoops.cpp。
  - 新增 createMatMulTileLoopsPass() 声明。
  - 新增 -emit=mlir-tiled-matmul。
  - CMake 已加入新源文件。
  - 新增 matmul64-tiled-loops.mlir。
  - 新增 matmul70-tiled-loops.mlir。

未在本地执行：
  - ninja -C build toyc-ch7
  - toyc-ch7 -emit=mlir-tiled-matmul
  - llvm-lit matmul64-tiled-loops.mlir
  - llvm-lit matmul70-tiled-loops.mlir

未执行原因：
  - 本轮延续前置要求：本地不要编译。
```
