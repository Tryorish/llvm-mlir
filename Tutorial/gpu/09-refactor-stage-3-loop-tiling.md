# Stage 3: Matmul Loop Split And Tiling

目标：把 naive matmul loop 拆成 tile loop 和 point loop。

这一阶段仍然不生成 GPU，只把 loop nest 整理成后续 mapping 可识别的结构。

## 建议新增 Pass

```text
toy-matmul-tile-loops
```

建议实现文件：

```text
mlir/examples/toy/Ch7/mlir/MatMulTileLoops.cpp
mlir/examples/toy/Ch7/include/toy/Passes.h
mlir/examples/toy/Ch7/toyc.cpp
mlir/examples/toy/Ch7/CMakeLists.txt
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

## 实施步骤

```text
1. 定义 tile size，第一版固定为 16。
2. 匹配阶段 2 生成的 canonical matmul loop nest。
3. 拆分 i/j/k 三个 induction variable。
4. 用 io+ii、jo+ji、ko+ki 替换原 load/store index。
5. 在 load/store 前插入边界判断。
6. 保留 accumulation 语义不变。
7. 输出仍是 host-side SCF/memref IR。
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
```

不能出现：

```text
gpu.launch
workgroup
gpu.barrier
```

建议新增测试：

```text
mlir/test/Examples/Toy/Ch7/matmul64-tiled-loops.mlir
mlir/test/Examples/Toy/Ch7/matmul70-tiled-loops.mlir
```
