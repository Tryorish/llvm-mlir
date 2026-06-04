# Stage 4: Reorder Tiled Matmul Loops

目标：把 tiled loop nest 固定成更适合 GPU mapping 和 shared memory promotion 的顺序。

这一阶段重点不是生成 GPU，而是把 loop 结构整理成后续 mapping 更容易处理的形状。

## 已新增 Pass

```text
toy-matmul-reorder-tiled-loops
```

实现文件：

```text
mlir/examples/toy/Ch7/mlir/MatMulReorderTiledLoops.cpp
mlir/examples/toy/Ch7/include/toy/Passes.h
mlir/examples/toy/Ch7/toyc.cpp
mlir/examples/toy/Ch7/CMakeLists.txt
```

新增 emit action：

```text
-emit=mlir-reordered-tiled-matmul
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
  -> func-level canonicalizer
  -> func-level CSE
  -> dump MLIR
```

## 输入

阶段 3 输出的 tiled loop nest：

```text
for io
  for jo
    for ko
      for ii
        for ji
          for ki
```

## 输出

每个 output point 独立 accumulation 的形状：

```text
for io
  for jo
    for ii
      for ji
        acc = 0
        for ko
          for ki
            acc += ...
        store C
```

注意：第三阶段只做 split/tile，输出顺序是 `io/jo/ko/ii/ji/ki`。第四阶段当前实现会真正重建 loop nest，把 `ko/ki` 移到每个 output point 的 accumulation 内部，并把 store 移到完整 K 维 reduction 之后。

后续 shared memory promotion 更希望看到：

```text
for io
  for jo
    for ii/ji mapped to threads
      acc = 0
      for ko
        load A/B tile
        barrier
        for ki
          acc += tileA * tileB
        barrier
      store C
```

## 实施步骤

```text
1. 已匹配阶段 3 的 split tiled loop nest。
2. 已把 loop 顺序从 io/jo/ko/ii/ji/ki 重建为 io/jo/ii/ji/ko/ki。
3. 已让 ko 成为带 iter_args 的 tile reduction loop。
4. 已让 ki 成为带 iter_args 的 point reduction loop。
5. 已把 store C 移到完整 ko accumulation 之后。
6. 已保留边界判断。
7. 已保持 IR 不引入 GPU op。
```

## 边界

```text
允许：
  - 重排 loop 顺序。
  - 调整 acc 的作用域。
  - 移动 store C 到完整 reduction 之后。

不允许：
  - 生成 gpu.launch。
  - 生成 block/thread id。
  - 生成 workgroup memory。
  - 生成 gpu.barrier。
```

## 验证

必须能看到：

```text
outer tile loops 在前。
k tile loop 包在每个 output tile 的 accumulation 内。
store C 在 k tile loop 之后。
acc 初始化不在 ko loop 内重复清零。
```

不能出现：

```text
gpu.launch
workgroup
gpu.barrier
```

新增测试：

```text
mlir/test/Examples/Toy/Ch7/matmul64-reordered-tiled-loops.mlir
```

## 当前实现范围

`MatMulReorderTiledLoops.cpp` 是一个 rewrite pass：

```text
只匹配：
  scf.for io step 16
    scf.for jo step 16
      scf.for ko step 16
        scf.for ii = 0 to 16 step 1
          scf.for ji = 0 to 16 step 1
            scf.if inBounds
              scf.for ki = 0 to 16 step 1 iter_args(acc)
              memref.store partial C

改写为：
  scf.for io step 16
    scf.for jo step 16
      scf.for ii = 0 to 16 step 1
        scf.for ji = 0 to 16 step 1
          scf.if inBounds
            scf.for ko step 16 iter_args(acc)
              scf.for ki = 0 to 16 step 1 iter_args(acc)
            memref.store final C
```

## 验证状态

已新增测试：

```text
mlir/test/Examples/Toy/Ch7/matmul64-reordered-tiled-loops.mlir
```

测试命令：

```bash
./build/bin/llvm-lit mlir/test/Examples/Toy/Ch7/matmul64-reordered-tiled-loops.mlir
```

或手动查看：

```bash
./build/bin/toyc-ch7 mlir/test/Examples/Toy/Ch7/matmul64-reordered-tiled-loops.mlir \
  -emit=mlir-reordered-tiled-matmul -x=mlir
```

当前执行状态：

```text
已完成：
  - 新增 MatMulReorderTiledLoops.cpp。
  - 新增 createMatMulReorderTiledLoopsPass() 声明。
  - 新增 -emit=mlir-reordered-tiled-matmul。
  - CMake 已加入新源文件。
  - 新增 matmul64-reordered-tiled-loops.mlir。

未在本地执行：
  - ninja -C build toyc-ch7
  - toyc-ch7 -emit=mlir-reordered-tiled-matmul
  - llvm-lit matmul64-reordered-tiled-loops.mlir

未执行原因：
  - 延续前置要求：本地不要编译。
```
