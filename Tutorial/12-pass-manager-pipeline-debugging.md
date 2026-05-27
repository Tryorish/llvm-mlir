# 第 12 课：Pass Manager、Pass Pipeline 与调试

## 本节定位

第 10 课和第 11 课分别讲了 C++ pattern rewrite 和 DRR 声明式重写。它们都回答了一个局部问题：

```text
某一条 rewrite 规则怎么写？
```

但在真实编译器里，单条规则不会孤立存在。优化、分析、lowering 都要按一定顺序串起来执行。

本节关注这个更大的问题：

```text
Toy 编译器如何组织 pass pipeline？
```

你会看到：

- Ch3 只有 canonicalizer。
- Ch4 加入 inliner、shape inference、canonicalizer、CSE。
- Ch5/Ch6 开始根据 `-emit` 目标选择 lowering pipeline。
- Ch7 因为 struct 扩展，对高层优化顺序做了细微调整。

本节不深入每个 pass 的内部算法，而是先学会读 pipeline、画 pipeline、调试 pipeline。

## 本节目标

- 理解 `PassManager` 的作用。
- 理解 module pass 和 nested function pass 的区别。
- 理解 `pm.addPass(...)` 和 `pm.nest<...>().addPass(...)` 的区别。
- 理解 `createInlinerPass()`、`createCanonicalizerPass()`、`createCSEPass()` 的位置。
- 理解 Toy 自定义 pass：`createShapeInferencePass()`、`createLowerToAffinePass()`、`createLowerToLLVMPass()`。
- 理解 `-opt` 和 `-emit=...` 如何影响 pipeline。
- 理解 `applyPassManagerCLOptions(pm)` 和 `registerPassManagerCLOptions()` 的用途。
- 能画出 Ch3、Ch4、Ch5/Ch6、Ch7 的 pass pipeline。

## 本节关键文件

源码：

```text
mlir/examples/toy/Ch3/toyc.cpp
mlir/examples/toy/Ch4/toyc.cpp
mlir/examples/toy/Ch5/toyc.cpp
mlir/examples/toy/Ch6/toyc.cpp
mlir/examples/toy/Ch7/toyc.cpp
```

相关 pass 实现：

```text
mlir/examples/toy/Ch4/mlir/ShapeInferencePass.cpp
mlir/examples/toy/Ch5/mlir/LowerToAffineLoops.cpp
mlir/examples/toy/Ch6/mlir/LowerToLLVM.cpp
```

建议阅读顺序：

1. 先看 Ch3 `toyc.cpp`，它只有 canonicalizer，最简单。
2. 再看 Ch4 `toyc.cpp`，它加入高层优化组合。
3. 再看 Ch5/Ch6 `toyc.cpp`，理解 lowering 目标如何影响 pipeline。
4. 最后看 Ch7 `toyc.cpp`，观察 struct 扩展后 pass 顺序的变化。

## Pass 是什么

Pass 是对 IR 做一次处理的单元。

它可能做：

- 分析，例如 shape inference。
- 优化，例如 canonicalization、CSE。
- 转换，例如 Toy lowering 到 Affine 或 LLVM Dialect。
- 清理，例如 lowering 后再跑 canonicalizer 和 CSE。

可以把 pass 理解成：

```text
输入 IR
  -> pass
输出 IR
```

多个 pass 串起来就是 pass pipeline：

```text
IR
  -> inliner
  -> shape inference
  -> canonicalizer
  -> CSE
  -> lowering
  -> cleanup
```

## `PassManager`

Toy 的 `toyc.cpp` 里会创建：

```cpp
mlir::PassManager pm(module.get()->getName());
```

`PassManager` 负责：

- 保存 pass 顺序。
- 管理 pass 运行层级。
- 应用 pass manager 命令行选项。
- 执行整个 pipeline。

最终运行：

```cpp
if (mlir::failed(pm.run(*module)))
  return 4;
```

这表示把整个 pipeline 跑在当前 `ModuleOp` 上。

## module pass 和 nested pass

MLIR IR 是嵌套结构：

```text
ModuleOp
  toy.func
    block
      toy.constant
      toy.reshape
      toy.return
```

有些 pass 应该跑在 module 层级，有些 pass 应该跑在函数层级。

### module pass

例如：

```cpp
pm.addPass(mlir::createInlinerPass());
```

这是加到 module pipeline 上的 pass。

Inliner 需要看到函数之间的调用关系，所以它适合在 module 层级运行。

### nested function pass

例如：

```cpp
mlir::OpPassManager &optPM = pm.nest<mlir::toy::FuncOp>();
optPM.addPass(mlir::toy::createShapeInferencePass());
optPM.addPass(mlir::createCanonicalizerPass());
optPM.addPass(mlir::createCSEPass());
```

这表示：

```text
在每个 toy.func 内部运行这些 pass
```

因为 shape inference、canonicalizer、CSE 主要处理函数体内的 operation，所以它们适合 nested 到 `toy.func`。

## `addNestedPass` 和 `nest`

Ch3 写法：

```cpp
pm.addNestedPass<mlir::toy::FuncOp>(mlir::createCanonicalizerPass());
```

Ch4 写法：

```cpp
mlir::OpPassManager &optPM = pm.nest<mlir::toy::FuncOp>();
optPM.addPass(mlir::toy::createShapeInferencePass());
optPM.addPass(mlir::createCanonicalizerPass());
optPM.addPass(mlir::createCSEPass());
```

两者都表示 nested pass。

区别是：

- `addNestedPass` 适合只添加一个 nested pass。
- `nest<...>()` 适合拿到某个层级的 `OpPassManager`，连续添加多个 pass。

## Ch3 pipeline

Ch3 的优化 pipeline 很短：

```cpp
if (enableOpt) {
  PassManager pm(module.get()->getName());
  applyPassManagerCLOptions(pm);

  pm.addNestedPass<toy::FuncOp>(createCanonicalizerPass());
  pm.run(*module);
}
```

画成图：

```text
ModuleOp
  toy.func
    -> canonicalizer
```

触发条件：

```text
只有带 -opt 时运行
```

它主要用于运行第 10、11 课讲过的 canonicalization patterns：

- `transpose(transpose(x)) -> x`
- reshape 相关 DRR patterns

## Ch4 pipeline

Ch4 中，`-opt` 后的 pipeline 变成：

```cpp
pm.addPass(mlir::createInlinerPass());

OpPassManager &optPM = pm.nest<mlir::toy::FuncOp>();
optPM.addPass(mlir::toy::createShapeInferencePass());
optPM.addPass(mlir::createCanonicalizerPass());
optPM.addPass(mlir::createCSEPass());
```

画成图：

```text
ModuleOp
  -> inliner
  toy.func
    -> shape inference
    -> canonicalizer
    -> CSE
```

这个顺序很重要。

## 为什么 inliner 在前

Toy 的函数调用是：

```mlir
toy.generic_call @multiply_transpose(...)
```

如果不内联，很多跨函数优化看不到函数体内部。

先跑 inliner 后，函数调用被展开，更多 operation 出现在同一个函数体里。

这样后面的 shape inference、canonicalizer、CSE 才有更多机会工作。

## 为什么 shape inference 在 canonicalizer 前

Shape inference 会把一些 unranked tensor：

```mlir
tensor<*xf64>
```

推成更精确的 ranked tensor：

```mlir
tensor<2x3xf64>
```

类型更精确之后，canonicalizer 和 CSE 更容易识别冗余 operation。

例如 reshape 相关 canonicalization 经常依赖输入输出类型是否相同。

## 为什么 CSE 在 canonicalizer 后

CSE 是 Common Subexpression Elimination，公共子表达式消除。

Canonicalizer 会先把 IR 变得更简单、更规范：

```text
复杂形态 -> 规范形态
```

规范化之后，两个原本看起来不同的表达式可能变得相同，这时 CSE 更容易识别和合并。

所以常见顺序是：

```text
canonicalizer -> CSE
```

## Ch5 pipeline

Ch5 开始支持：

```text
-emit=mlir-affine
```

也就是把 Toy Dialect 降低到 Affine/MemRef/Func 等更低层 Dialect。

Ch5 的核心判断：

```cpp
bool isLoweringToAffine = emitAction >= Action::DumpMLIRAffine;
```

如果：

```text
enableOpt || isLoweringToAffine
```

会先跑高层优化：

```text
ModuleOp
  -> inliner
  toy.func
    -> shape inference
    -> canonicalizer
    -> CSE
```

如果目标是 Affine：

```cpp
pm.addPass(mlir::toy::createLowerToAffinePass());
```

然后在 `func.func` 层级做清理：

```cpp
OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
optPM.addPass(mlir::createCanonicalizerPass());
optPM.addPass(mlir::createCSEPass());
```

如果同时带 `-opt`，还会加 affine 优化：

```cpp
optPM.addPass(mlir::affine::createLoopFusionPass());
optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
```

## Ch5 pipeline 图

```text
ModuleOp
  if -opt or lowering to affine:
    -> inliner
    toy.func
      -> shape inference
      -> canonicalizer
      -> CSE

  if -emit=mlir-affine:
    -> lower-to-affine
    func.func
      -> canonicalizer
      -> CSE
      if -opt:
        -> affine loop fusion
        -> affine scalar replacement
```

注意 lowering 前后函数 operation 的层级不同：

```text
lowering 前：toy.func
lowering 后：func.func
```

所以 nested pass 的类型也从：

```cpp
pm.nest<mlir::toy::FuncOp>()
```

变成：

```cpp
pm.nest<mlir::func::FuncOp>()
```

## Ch6 pipeline

Ch6 在 Ch5 的基础上继续支持：

```text
-emit=mlir-llvm
-emit=llvm
-emit=jit
```

核心判断：

```cpp
bool isLoweringToAffine = emitAction >= Action::DumpMLIRAffine;
bool isLoweringToLLVM = emitAction >= Action::DumpMLIRLLVM;
```

如果目标需要 LLVM Dialect：

```cpp
pm.addPass(mlir::toy::createLowerToLLVMPass());
pm.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());
```

所以 Ch6 的完整方向是：

```text
Toy Dialect
  -> high-level optimization
  -> Affine / MemRef / Func
  -> cleanup / affine optimizations
  -> LLVM Dialect
  -> LLVM IR / JIT
```

## Ch7 pipeline

Ch7 结构和 Ch6 很像，但高层优化顺序有一个明显差异：

```cpp
OpPassManager &optPM = pm.nest<mlir::toy::FuncOp>();
optPM.addPass(mlir::createCanonicalizerPass());
optPM.addPass(mlir::toy::createShapeInferencePass());
optPM.addPass(mlir::createCanonicalizerPass());
optPM.addPass(mlir::createCSEPass());
```

也就是：

```text
toy.func
  -> canonicalizer
  -> shape inference
  -> canonicalizer
  -> CSE
```

和 Ch6 相比，Ch7 在 shape inference 前多跑了一次 canonicalizer。

这体现了一个重要经验：

```text
pass 顺序不是随便排的。
```

某些 canonicalization 可以先简化 struct 相关 IR，让后面的 shape inference 更容易工作。shape inference 之后再跑一次 canonicalizer，又能利用更精确的类型继续清理 IR。

## Pipeline 对比表

| Chapter | 触发条件 | Module 层级 | `toy.func` 层级 | Lowering 后 `func.func` 层级 |
| --- | --- | --- | --- | --- |
| Ch3 | `-opt` | 无 | canonicalizer | 无 |
| Ch4 | `-opt` | inliner | shape inference -> canonicalizer -> CSE | 无 |
| Ch5 | `-opt` 或 `-emit=mlir-affine` | inliner；lower-to-affine | shape inference -> canonicalizer -> CSE | canonicalizer -> CSE；可选 affine opts |
| Ch6 | `-emit` 目标决定 | inliner；lower-to-affine；lower-to-LLVM | shape inference -> canonicalizer -> CSE | canonicalizer -> CSE；可选 affine opts |
| Ch7 | `-emit` 目标决定 | inliner；lower-to-affine；lower-to-LLVM | canonicalizer -> shape inference -> canonicalizer -> CSE | canonicalizer -> CSE；可选 affine opts |

## `-opt` 和 `-emit` 的关系

在 Ch3/Ch4：

```text
-opt
  -> 启用优化 pipeline
```

在 Ch5 之后，情况更复杂。

即使没有 `-opt`，如果你要求：

```text
-emit=mlir-affine
-emit=mlir-llvm
-emit=llvm
-emit=jit
```

编译器也必须跑某些前置 pass。

原因是 lowering 需要更规范、更精确的 IR。

例如 lowering 到 Affine 之前，Toy 需要尽量完成：

- inlining
- shape inference
- canonicalization
- CSE

否则后续 lowering 很难生成结构良好的低层 IR。

## `applyPassManagerCLOptions(pm)`

Toy 的 `toyc.cpp` 中都有类似代码：

```cpp
if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
  return 4;
```

同时在 `main()` 里注册：

```cpp
mlir::registerPassManagerCLOptions();
```

这两者配合，让 MLIR 的通用 pass manager 命令行选项可以作用到当前 pass pipeline 上。

这对调试非常重要。

## 常用调试思路

当你想理解某个 pass 前后 IR 如何变化时，可以尝试：

```bash
toyc-ch4 input.toy -emit=mlir -opt --mlir-print-ir-after-all
```

或：

```bash
toyc-ch4 input.toy -emit=mlir -opt --mlir-print-ir-before-all
```

这些选项来自 MLIR pass manager 的通用命令行支持。

如果输出过多，可以先用小测试文件，比如：

```text
mlir/test/Examples/Toy/Ch3/transpose_transpose.toy
mlir/test/Examples/Toy/Ch4/shape_inference.mlir
```

## 读 pass 输出时看什么

建议每次只追踪一个问题：

- 哪个 pass 删除了 `toy.transpose`？
- 哪个 pass 把 `tensor<*xf64>` 推成了 `tensor<2x3xf64>`？
- 哪个 pass 把 `toy.generic_call` 内联掉了？
- lowering 后 `toy.func` 什么时候变成 `func.func`？
- lowering 后为什么 nested pass 类型从 `toy::FuncOp` 变成 `func::FuncOp`？

不要一开始就试图读完整输出。Pass 调试输出通常很长，先抓一个现象会更有效。

## 为什么顺序重要

Pass pipeline 不是简单堆叠。

例如：

```text
canonicalizer -> shape inference
```

和：

```text
shape inference -> canonicalizer
```

可能产生不同结果。

原因是：

- canonicalizer 可能删除或简化 operation。
- shape inference 依赖 operation 结构和 operand types。
- shape inference 更新类型后，又可能暴露新的 canonicalization 机会。
- CSE 通常希望看到尽量规范化后的 IR。

所以 Ch7 采用：

```text
canonicalizer -> shape inference -> canonicalizer -> CSE
```

是很有现实意义的组合。

## 一张总体 pipeline 图

下面是从 Ch3 到 Ch6/Ch7 逐步扩展的主线：

```text
Ch3:
  Toy MLIR
    -> canonicalizer

Ch4:
  Toy MLIR
    -> inliner
    -> shape inference
    -> canonicalizer
    -> CSE

Ch5:
  Toy MLIR
    -> high-level opts
    -> lower to Affine/MemRef/Func
    -> canonicalizer
    -> CSE
    -> optional affine loop opts

Ch6:
  Toy MLIR
    -> high-level opts
    -> lower to Affine/MemRef/Func
    -> cleanup
    -> lower to LLVM Dialect
    -> LLVM IR / JIT

Ch7:
  Toy MLIR with struct
    -> canonicalizer
    -> shape inference
    -> canonicalizer
    -> CSE
    -> lowering pipeline
```

## 常见误区

### 误区 1：`-opt` 才会运行所有 pass

不一定。

在 Ch5 之后，如果目标是 lowering 到 Affine 或 LLVM，即使不加 `-opt`，也需要运行一些必要 pass。

### 误区 2：所有 pass 都跑在 module 上

不是。

有些 pass 是 module 层级，例如 inliner、lowering pass。

有些 pass nested 到函数层级，例如 canonicalizer、CSE、shape inference。

### 误区 3：lowering 前后可以用同一个 nested op 类型

不行。

Toy lowering 前是：

```cpp
pm.nest<mlir::toy::FuncOp>()
```

lowering 后变成：

```cpp
pm.nest<mlir::func::FuncOp>()
```

因为 IR 中的函数 operation 已经变了。

### 误区 4：CSE 可以替代 canonicalizer

不能。

Canonicalizer 负责把 IR 简化、规范化。

CSE 负责合并相同计算。

两者互补。

### 误区 5：pass 顺序随便调都一样

不一样。

有些 pass 会为后续 pass 创造机会，有些 pass 依赖前一个 pass 的结果。

## 动手观察

### 观察 Ch3

阅读：

```text
mlir/examples/toy/Ch3/toyc.cpp
```

找到：

```cpp
pm.addNestedPass<mlir::toy::FuncOp>(mlir::createCanonicalizerPass());
```

回答：

- 为什么它 nested 到 `toy.func`。
- 为什么只有 `-opt` 时才运行。

### 观察 Ch4

阅读：

```text
mlir/examples/toy/Ch4/toyc.cpp
```

找到：

```cpp
pm.addPass(mlir::createInlinerPass());
optPM.addPass(mlir::toy::createShapeInferencePass());
optPM.addPass(mlir::createCanonicalizerPass());
optPM.addPass(mlir::createCSEPass());
```

回答：

- 哪个是 module pass。
- 哪些是 nested function pass。
- 为什么 inliner 在前。

### 观察 Ch5/Ch6

阅读：

```text
mlir/examples/toy/Ch5/toyc.cpp
mlir/examples/toy/Ch6/toyc.cpp
```

回答：

- `isLoweringToAffine` 如何决定是否跑 lowering。
- `isLoweringToLLVM` 如何决定是否继续 lowering 到 LLVM Dialect。
- lowering 前后 nested pass 的 op 类型有什么变化。

### 观察 Ch7

阅读：

```text
mlir/examples/toy/Ch7/toyc.cpp
```

找到：

```cpp
optPM.addPass(mlir::createCanonicalizerPass());
optPM.addPass(mlir::toy::createShapeInferencePass());
optPM.addPass(mlir::createCanonicalizerPass());
optPM.addPass(mlir::createCSEPass());
```

回答：

- 为什么 canonicalizer 跑了两次。
- 这个顺序和 Ch6 有什么不同。

## 本节练习

### 练习 1：画 Ch3 pipeline

画出 `toyc-ch3 -emit=mlir -opt` 的 pass pipeline。

至少标出：

- module 层级。
- `toy.func` 层级。
- canonicalizer 的位置。

### 练习 2：画 Ch4 pipeline

画出 `toyc-ch4 -emit=mlir -opt` 的 pass pipeline。

至少标出：

- inliner。
- shape inference。
- canonicalizer。
- CSE。
- 哪些是 nested 到 `toy.func` 的 pass。

### 练习 3：解释 nested pass

回答：

- `pm.addPass(...)` 和 `pm.nest<...>().addPass(...)` 的区别是什么。
- 为什么 Ch4 里 shape inference 不直接 `pm.addPass(...)`。

### 练习 4：解释 lowering 触发条件

阅读 Ch5 或 Ch6：

```cpp
bool isLoweringToAffine = emitAction >= Action::DumpMLIRAffine;
bool isLoweringToLLVM = emitAction >= Action::DumpMLIRLLVM;
```

回答：

- `-emit=mlir` 会不会触发 Affine lowering。
- `-emit=mlir-affine` 会触发哪些阶段。
- `-emit=mlir-llvm` 会触发哪些阶段。

### 练习 5：解释 pass 顺序

回答：

- 为什么 inliner 通常在 shape inference 前。
- 为什么 canonicalizer 通常在 CSE 前。
- 为什么 shape inference 后再跑 canonicalizer 可能有意义。

### 练习 6：观察调试输出

选择一个小测试文件，尝试运行：

```bash
toyc-ch4 <file> -emit=mlir -opt --mlir-print-ir-after-all
```

记录：

- 哪个 pass 后 IR 变化最大。
- 哪个 pass 删除了冗余 operation。
- 哪个 pass 改变了 tensor type。

### 练习 7：整理 pipeline 对比表

自己整理一张表，对比：

```text
Ch3
Ch4
Ch5
Ch6
Ch7
```

每行写：

- 触发条件。
- module pass。
- nested `toy.func` pass。
- lowering pass。
- nested `func.func` cleanup pass。

## 本节小结

本节最重要的是学会读 `toyc.cpp` 中的 pass pipeline。

需要记住：

```text
PassManager
  -> 管理整个 pipeline

pm.addPass(...)
  -> 添加 module 层级 pass

pm.nest<SomeOp>().addPass(...)
  -> 添加 nested 到 SomeOp 的 pass

applyPassManagerCLOptions(pm)
  -> 让通用 pass manager 调试选项生效
```

也要记住 Toy 的演进：

- Ch3：只跑 canonicalizer。
- Ch4：inliner + shape inference + canonicalizer + CSE。
- Ch5：加入 Affine lowering。
- Ch6：加入 LLVM lowering、LLVM IR、JIT。
- Ch7：struct 扩展后调整高层优化顺序。

下一课会深入 `ShapeInferenceInterface` 和 `ShapeInferencePass.cpp`，解释 Ch4 pipeline 中最关键的 Toy 自定义分析 pass。

## 学习记录模板

```text
本节主题：Pass Manager、Pass Pipeline 与调试
我读过的源码：
我画出的 Ch3 pipeline：
我画出的 Ch4 pipeline：
我画出的 Ch5/Ch6 pipeline：
我理解的 nested pass：
我使用过的 pass 调试选项：
我还不理解的问题：
下一步要验证的小实验：
```
