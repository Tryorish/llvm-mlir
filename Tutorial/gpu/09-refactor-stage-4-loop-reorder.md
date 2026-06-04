# Stage 4: Reorder Tiled Matmul Loops

目标：把 tiled loop nest 重排成更适合 GPU mapping 和 shared memory promotion 的顺序。

这一阶段重点不是生成 GPU，而是把 loop 结构整理成后续 mapping 更容易处理的形状。

## 建议新增 Pass

```text
toy-matmul-reorder-tiled-loops
```

建议实现文件：

```text
mlir/examples/toy/Ch7/mlir/MatMulReorderTiledLoops.cpp
mlir/examples/toy/Ch7/include/toy/Passes.h
mlir/examples/toy/Ch7/toyc.cpp
mlir/examples/toy/Ch7/CMakeLists.txt
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

先整理成每个 output point 独立 accumulation 的形状：

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
1. 匹配阶段 3 的 loop nest。
2. 确认 store C 当前在正确的 accumulation 之后。
3. 把 ko 移到 ii/ji 内部的 accumulation 区域。
4. 让 acc 初始化接近 output point。
5. 保证 store C 位于完整 ko accumulation 之后。
6. 保留边界判断。
7. 不引入 GPU op。
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

建议新增测试：

```text
mlir/test/Examples/Toy/Ch7/matmul64-reordered-tiled-loops.mlir
```
