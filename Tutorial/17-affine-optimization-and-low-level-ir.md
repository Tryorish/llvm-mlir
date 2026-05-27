# 第 17 课：Affine 优化与低层 IR 观察

## 本节定位

第 16 课已经讲过 Toy operation 如何 lowering 到 Affine/MemRef/Func/Arith：

```text
toy.constant
  -> memref.alloc + arith.constant + affine.store

toy.transpose / toy.mul
  -> affine.for + affine.load + arith op + affine.store
```

本节继续观察 lowering 后的低层 IR，并重点比较：

```text
toyc-ch5 input.mlir -emit=mlir-affine
toyc-ch5 input.mlir -emit=mlir-affine -opt
```

也就是：

- 不开启 affine 优化时，IR 长什么样。
- 开启 affine 优化后，哪些中间 buffer 和 loop 被合并或消除。

本节不会深入 affine pass 的内部算法，而是先训练一个能力：

```text
能读懂 affine lowering 后的 IR 层级，
能看出优化前后多了什么、少了什么、为什么。
```

## 本节目标

- 理解 lowering 到 Affine 的意义。
- 能读懂 `affine.for`、`affine.load`、`affine.store` 的基本结构。
- 能区分 MemRef、Affine、Arith、Func、Toy 在 lowering 后的职责。
- 理解 `createCanonicalizerPass()` 和 `createCSEPass()` 在 lowering 后仍然有价值。
- 理解 `createLoopFusionPass()` 大致在做什么。
- 理解 `createAffineScalarReplacementPass()` 大致在做什么。
- 能对比 `affine-lowering.mlir` 中 `CHECK` 和 `OPT` 的差异。
- 能找出哪些临时 `memref.alloc` 被优化掉，哪些仍然保留。

## 本节关键文件

源码：

```text
mlir/examples/toy/Ch5/toyc.cpp
mlir/examples/toy/Ch6/toyc.cpp
mlir/examples/toy/Ch5/mlir/LowerToAffineLoops.cpp
```

测试：

```text
mlir/test/Examples/Toy/Ch5/affine-lowering.mlir
mlir/test/Examples/Toy/Ch6/affine-lowering.mlir
```

建议阅读顺序：

1. 先看 `affine-lowering.mlir` 的输入 Toy IR。
2. 再看 `CHECK` 部分，理解未开启 affine 优化时的 IR。
3. 再看 `OPT` 部分，对比开启 `-opt` 后少了哪些中间结果。
4. 最后回到 `toyc.cpp`，确认是哪几个 pass 造成差异。

## 为什么 lowering 到 Affine

Toy 是高层教学语言，表达的是：

```mlir
%2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
%3 = toy.mul %2, %2 : tensor<3x2xf64>
```

这很适合前端和高层优化，但不适合直接生成机器代码。

Affine lowering 把它变成：

```text
显式 buffer
显式循环
显式 load/store
显式标量运算
```

例如：

```mlir
affine.for %i = 0 to 3 {
  affine.for %j = 0 to 2 {
    %v = affine.load %input[%j, %i] : memref<2x3xf64>
    affine.store %v, %out[%i, %j] : memref<3x2xf64>
  }
}
```

这样后续 MLIR 就可以使用通用 affine 优化，比如 loop fusion 和 scalar replacement。

## lowering 后的 dialect 分工

Ch5 affine lowering 后，IR 中主要有这些 dialect：

| Dialect | 负责什么 |
| --- | --- |
| `func` | 函数边界，例如 `func.func`、`func.return` |
| `memref` | buffer 分配和释放，例如 `memref.alloc`、`memref.dealloc` |
| `affine` | 静态可分析循环和访存，例如 `affine.for`、`affine.load`、`affine.store` |
| `arith` | 标量常量和标量计算，例如 `arith.constant`、`arith.mulf` |
| `toy` | Ch5 仍保留 `toy.print`，但 operand 已经是 memref |

这就是 Ch5 partial lowering 的形态。

## 输入 IR

测试输入：

```mlir
toy.func @main() {
  %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
       : tensor<2x3xf64>
  %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %3 = toy.mul %2, %2 : tensor<3x2xf64>
  toy.print %3 : tensor<3x2xf64>
  toy.return
}
```

语义是：

```text
1. 创建一个 2x3 常量矩阵。
2. 转置成 3x2。
3. 和自己逐元素相乘。
4. 打印结果。
```

## 未优化 lowering 的结构

不加 `-opt`：

```bash
toyc-ch5 affine-lowering.mlir -emit=mlir-affine
```

测试期望中会看到三个 `memref.alloc`：

```mlir
[[VAL_6]] = memref.alloc() : memref<3x2xf64>
[[VAL_7]] = memref.alloc() : memref<3x2xf64>
[[VAL_8]] = memref.alloc() : memref<2x3xf64>
```

它们大致对应：

```text
VAL_8: 原始 constant 的 buffer，shape 2x3
VAL_7: transpose 的中间结果 buffer，shape 3x2
VAL_6: mul 的最终结果 buffer，shape 3x2
```

也就是说，未优化 lowering 很直接：

```text
每个产生 tensor result 的 Toy op
  -> 分配一个 memref
```

这很好理解，但不一定最高效。

## constant 初始化区域

未优化输出中，常量矩阵会展开为 scalar constants 和 stores：

```mlir
[[VAL_0]] = arith.constant 1.000000e+00 : f64
[[VAL_1]] = arith.constant 2.000000e+00 : f64
...
[[VAL_8]] = memref.alloc() : memref<2x3xf64>
affine.store [[VAL_0]], [[VAL_8]][0, 0] : memref<2x3xf64>
affine.store [[VAL_1]], [[VAL_8]][0, 1] : memref<2x3xf64>
...
```

这对应第 16 课讲的 `ConstantOpLowering`。

高层的：

```mlir
toy.constant dense<...> : tensor<2x3xf64>
```

变成了：

```text
分配 buffer
逐元素 store 常量
```

## transpose loop

未优化输出中，transpose 有自己的 loop nest：

```mlir
affine.for [[I]] = 0 to 3 {
  affine.for [[J]] = 0 to 2 {
    [[V]] = affine.load [[VAL_8]][[[J]], [[I]]] : memref<2x3xf64>
    affine.store [[V]], [[VAL_7]][[[I]], [[J]]] : memref<3x2xf64>
  }
}
```

要点：

- 输出 buffer 是 `VAL_7`，shape 是 `3x2`。
- loop bounds 是 `0..3` 和 `0..2`。
- 输出下标是 `[I, J]`。
- 输入下标是 `[J, I]`，完成转置。

这里没有标量计算，只是重新排列元素。

## mul loop

未优化输出中，mul 也有自己的 loop nest：

```mlir
affine.for [[I]] = 0 to 3 {
  affine.for [[J]] = 0 to 2 {
    [[X]] = affine.load [[VAL_7]][[[I]], [[J]]] : memref<3x2xf64>
    [[Y]] = arith.mulf [[X]], [[X]] : f64
    affine.store [[Y]], [[VAL_6]][[[I]], [[J]]] : memref<3x2xf64>
  }
}
```

它读取 transpose 的中间 buffer `VAL_7`，逐元素平方，再写入最终结果 buffer `VAL_6`。

这很直接，但有一个可优化点：

```text
transpose loop 写 VAL_7
mul loop 紧接着读 VAL_7
```

`VAL_7` 是中间 buffer，可能可以被消除。

## print 和 dealloc

最后：

```mlir
toy.print [[VAL_6]] : memref<3x2xf64>
memref.dealloc [[VAL_8]] : memref<2x3xf64>
memref.dealloc [[VAL_7]] : memref<3x2xf64>
memref.dealloc [[VAL_6]] : memref<3x2xf64>
```

`toy.print` 仍然保留，这是 partial lowering。

三个 buffer 都在函数结束前释放。

## lowering 后的 cleanup pass

Ch5 `toyc.cpp` 中，lowering 后会跑：

```cpp
OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
optPM.addPass(mlir::createCanonicalizerPass());
optPM.addPass(mlir::createCSEPass());
```

注意这里 nested 到：

```cpp
func::FuncOp
```

而不是：

```cpp
toy::FuncOp
```

因为 lowering 后函数已经变成 `func.func`。

这两个 cleanup pass 会清理一些 lowering 过程中产生的低层冗余。

## `-opt` 下的 affine 优化

如果命令带 `-opt`，Ch5/Ch6 会额外加：

```cpp
optPM.addPass(mlir::affine::createLoopFusionPass());
optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
```

这两个 pass 是 affine 层级优化：

- Loop fusion：尝试合并相关 loop，减少中间存储和遍历开销。
- Affine scalar replacement：尝试把某些 memref load/store 替换成标量值，减少临时内存访问。

本节只需要理解它们的可观察效果。

## 优化后 alloc 数量变化

开启 `-opt` 后，测试期望中只有两个主要 alloc：

```mlir
[[VAL_6]] = memref.alloc() : memref<3x2xf64>
[[VAL_7]] = memref.alloc() : memref<2x3xf64>
```

对比未优化：

```text
未优化:
  memref<2x3xf64> constant buffer
  memref<3x2xf64> transpose intermediate buffer
  memref<3x2xf64> mul result buffer

优化后:
  memref<2x3xf64> constant buffer
  memref<3x2xf64> final result buffer
```

也就是说：

```text
transpose 的中间 buffer 被消除了。
```

这就是 affine 优化的一个直接效果。

## 优化后的融合形态

优化后测试期望中，只有一个 loop nest 同时完成“转置读取”和“乘法”：

```mlir
affine.for [[I]] = 0 to 3 {
  affine.for [[J]] = 0 to 2 {
    [[X]] = affine.load [[VAL_7]][[[J]], [[I]]] : memref<2x3xf64>
    [[Y]] = arith.mulf [[X]], [[X]] : f64
    affine.store [[Y]], [[VAL_6]][[[I]], [[J]]] : memref<3x2xf64>
  }
}
```

对比未优化：

```text
loop 1:
  out_transpose[i,j] = input[j,i]

loop 2:
  out_mul[i,j] = out_transpose[i,j] * out_transpose[i,j]
```

优化后：

```text
loop:
  x = input[j,i]
  out_mul[i,j] = x * x
```

中间的 `out_transpose` 不再需要单独存储。

## loop fusion 的直观理解

Loop fusion 的目标之一是把生产者 loop 和消费者 loop 合并。

在这里：

```text
producer:
  transpose loop 写 VAL_7

consumer:
  mul loop 读 VAL_7
```

二者迭代空间兼容，而且中间 buffer 只用于传递数据。

优化后可以直接在消费者位置重新计算或转发需要的值：

```text
load input[j,i]
mul
store final
```

这样减少了：

- 一个中间 memref。
- 一组 affine.store 到中间 memref。
- 一组 affine.load 从中间 memref。
- 一个独立 loop nest。

## scalar replacement 的直观理解

Affine scalar replacement 可以把某些内存读写转成标量值传递。

在优化后 IR 中：

```mlir
[[X]] = affine.load [[VAL_7]][[[J]], [[I]]] : memref<2x3xf64>
[[Y]] = arith.mulf [[X]], [[X]] : f64
```

可以看到乘法直接使用标量 `[[X]]` 两次。

相比未优化版本通过中间 memref `VAL_7` 存取 transpose 结果，优化后更接近：

```text
let x = input[j, i]
out[i, j] = x * x
```

这就是减少临时内存访问的方向。

## 为什么仍然保留 constant buffer

优化后仍然有：

```mlir
[[VAL_7]] = memref.alloc() : memref<2x3xf64>
affine.store ... [[VAL_7]][...]
```

也就是原始常量 buffer 仍然存在。

原因是当前 pipeline 没有把整个常量矩阵完全标量化并消除 buffer。

这也提醒你：

```text
优化不是“能消掉所有东西”。
它只在满足条件、成本模型和 pass 能力范围内改写 IR。
```

本测试的重点是中间 transpose buffer 被消掉。

## 为什么仍然保留 final result buffer

最终 `toy.print` 需要一个 memref：

```mlir
toy.print [[VAL_6]] : memref<3x2xf64>
```

因此最终结果 buffer `VAL_6` 不能消掉。

只要 `toy.print` 还停留在 Ch5 的 partial lowering 形式，它就需要一个可打印的 memref operand。

后续 Ch6 会继续把 `toy.print` lowering 到 runtime call。

## Ch5 和 Ch6 的 affine 测试

Ch5 和 Ch6 都有：

```text
mlir/test/Examples/Toy/Ch5/affine-lowering.mlir
mlir/test/Examples/Toy/Ch6/affine-lowering.mlir
```

它们在 affine lowering 这一阶段的期望基本一致。

区别是 Ch6 后续还支持：

```text
-emit=mlir-llvm
-emit=llvm
-emit=jit
```

也就是说：

```text
Ch5:
  到 Affine/MemRef/Func 为止

Ch6:
  可以从 Affine/MemRef/Func 继续 lower 到 LLVM Dialect
```

本节只关注二者共有的 affine 层。

## 如何读 affine IR

建议按这几个层次读：

```text
1. 函数边界
   func.func @main

2. 内存对象
   memref.alloc / memref.dealloc

3. 常量初始化
   arith.constant + affine.store

4. 循环结构
   affine.for

5. 访存模式
   affine.load / affine.store 下标

6. 标量计算
   arith.addf / arith.mulf

7. 仍未 lowering 的 Toy op
   toy.print
```

读复杂 IR 时，不要从第一行机械读到最后一行。先分层，再看每层的职责。

## 常见误区

### 误区 1：`-emit=mlir-affine` 一定等于优化后 IR

不一定。

```text
-emit=mlir-affine
```

只表示 lowering 到 affine 层。

```text
-emit=mlir-affine -opt
```

才会额外启用 affine loop fusion 和 scalar replacement。

### 误区 2：loop fusion 一定会合并所有 loop

不会。

Loop fusion 需要满足依赖关系、迭代空间和合法性条件。

不能安全合并的 loop 不会被合并。

### 误区 3：优化后所有 memref 都会消失

不会。

输入常量 buffer 和最终 print buffer 可能仍然需要保留。

### 误区 4：`toy.print` 出现在 affine IR 里说明 lowering 失败

不是。

Ch5 是 partial lowering，`toy.print` 被允许保留，只要 operand 已经是 memref。

### 误区 5：Affine IR 只是更啰嗦的 Toy IR

不只是。

Affine IR 显式表达循环和访存，这给 loop-level 优化提供了分析基础。

## 动手观察

### 观察未优化 lowering

阅读 `affine-lowering.mlir` 的 `CHECK` 部分。

标注：

- 哪个 memref 是 constant buffer。
- 哪个 memref 是 transpose intermediate buffer。
- 哪个 memref 是 final mul result buffer。
- 哪段 loop 是 transpose。
- 哪段 loop 是 mul。

### 观察优化后 lowering

阅读 `OPT` 部分。

标注：

- 哪个中间 memref 消失了。
- transpose 和 mul 是否还分成两个 loop nest。
- `affine.load` 的下标为什么是 `[j, i]`。
- `arith.mulf` 是否直接使用从原始 constant buffer load 的值。

### 观察 pipeline

阅读：

```text
mlir/examples/toy/Ch5/toyc.cpp
mlir/examples/toy/Ch6/toyc.cpp
```

找到：

```cpp
optPM.addPass(mlir::affine::createLoopFusionPass());
optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
```

确认它们只在：

```text
isLoweringToAffine && enableOpt
```

这个组合下加入 pipeline。

## 本节练习

### 练习 1：分层注释 affine IR

选择 `affine-lowering.mlir` 的 `CHECK` 部分，把输出 IR 分成：

```text
function
alloc/dealloc
constant initialization
transpose loop
mul loop
print
```

每一层写一句说明。

### 练习 2：对比 alloc 数量

回答：

- 未优化版本有几个主要 `memref.alloc`。
- 优化版本有几个主要 `memref.alloc`。
- 消失的是哪个 shape 的 buffer。
- 它原本承担什么职责。

### 练习 3：解释融合后的 loop

阅读 `OPT` 部分：

```mlir
affine.load [[VAL_7]][[[J]], [[I]]] : memref<2x3xf64>
arith.mulf
affine.store ... [[VAL_6]][[[I]], [[J]]] : memref<3x2xf64>
```

回答：

- 为什么 load 用 `[J, I]`。
- 为什么 store 用 `[I, J]`。
- 这段 loop 同时完成了原来哪两个 Toy operation 的工作。

### 练习 4：解释 `toy.print`

回答：

- 为什么 optimized affine IR 中仍然有 `toy.print`。
- 它的 operand type 是什么。
- 这和 partial lowering 有什么关系。

### 练习 5：解释 `-opt`

回答：

- `-emit=mlir-affine` 和 `-emit=mlir-affine -opt` 的 pass pipeline 差异是什么。
- 哪两个 affine pass 只在 `-opt` 时加入。
- 它们在测试中最直观的效果是什么。

### 练习 6：预测一个不能融合的场景

思考：

如果两个 loop 之间存在复杂依赖，或者中间 buffer 被多个后续操作以不同方式使用，loop fusion 是否一定能合并？

回答：

- 为什么不能只看 loop 形状相同就合并。
- 依赖关系为什么重要。

### 练习 7：写一份低层 IR 速查表

整理：

```text
func.func
memref.alloc
memref.dealloc
affine.for
affine.load
affine.store
arith.constant
arith.mulf
toy.print
```

每个写：

- 所属 dialect。
- 在 Ch5 affine lowering 后承担什么职责。

## 本节小结

本节最重要的是学会观察 affine lowering 后的 IR：

```text
未优化:
  constant buffer
  transpose intermediate buffer
  final result buffer
  transpose loop
  mul loop

优化后:
  constant buffer
  final result buffer
  fused loop
```

需要记住：

- Affine lowering 把高层 tensor 计算显式化为 buffer、loop、load/store 和 scalar op。
- `-opt` 会在 affine 层额外启用 loop fusion 和 scalar replacement。
- Loop fusion 可以把 producer/consumer loop 合并，减少中间 memref。
- Scalar replacement 可以减少中间内存读写，更多使用标量值。
- Ch5/Ch6 的 affine 层仍然保留 `toy.print`，这是 partial lowering 的结果。

下一课会继续从 Affine/MemRef/Func 走向 LLVM Dialect，学习 Ch6 的 full lowering pipeline。

## 学习记录模板

```text
本节主题：Affine 优化与低层 IR 观察
我读过的源码：
我观察过的测试：
我标注过的 affine IR 层级：
我找到的中间 memref：
我理解的 loop fusion 效果：
我理解的 scalar replacement 效果：
我还不理解的问题：
下一步要验证的小实验：
```
