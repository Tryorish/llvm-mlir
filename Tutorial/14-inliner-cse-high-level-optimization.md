# 第 14 课：Inliner、CSE 与高层 IR 优化组合

## 本节定位

第 13 课已经深入讲了 `ShapeInferencePass`。但 Ch4 的优化并不是只靠 shape inference 完成的。

Ch4 的完整高层优化 pipeline 是：

```text
inliner
  -> shape inference
  -> canonicalizer
  -> CSE
```

这节课关注这些 pass 如何协同工作。

核心问题是：

```text
为什么一个 Toy 程序经过 -opt 后，函数调用消失了，unknown shape 消失了，
冗余 transpose/reshape 消失了，重复计算也被合并了？
```

本节不会深入 lowering，也不会重复讲 shape inference 的内部算法，而是从“组合效果”角度看 Ch4 的高层优化。

## 本节目标

- 理解 inliner 在 Toy 中解决什么问题。
- 理解 `CallOpInterface`、`ToyInlinerInterface`、`toy.cast` 在 inlining 中的角色。
- 理解 shape inference 为什么依赖 inlining 提供更具体的上下文。
- 理解 canonicalizer 如何继续清理内联和 shape inference 之后的 IR。
- 理解 CSE 的基本作用：合并等价的重复计算。
- 能解释 Ch4 `shape_inference.mlir` 的最终输出为什么只剩一个 `main`。
- 能解释 Ch4 `transpose_transpose.toy` 为什么最终直接 `toy.print` 原始常量。

## 本节关键文件

源码：

```text
mlir/examples/toy/Ch4/toyc.cpp
mlir/examples/toy/Ch4/mlir/Dialect.cpp
mlir/examples/toy/Ch4/include/toy/Ops.td
mlir/examples/toy/Ch4/mlir/ShapeInferencePass.cpp
mlir/examples/toy/Ch4/mlir/ToyCombine.cpp
```

测试：

```text
mlir/test/Examples/Toy/Ch4/codegen.toy
mlir/test/Examples/Toy/Ch4/shape_inference.mlir
mlir/test/Examples/Toy/Ch4/transpose_transpose.toy
```

建议阅读顺序：

1. 先看 `toyc.cpp` 中的 pass 顺序。
2. 再看 `Dialect.cpp` 顶部的 `ToyInlinerInterface`。
3. 再看 `Ops.td` 中 `GenericCallOp` 的 `CallOpInterface` 和 `CastOp`。
4. 最后对照 `shape_inference.mlir` 和 `transpose_transpose.toy` 的 FileCheck。

## Ch4 pipeline 回顾

Ch4 `toyc.cpp` 中：

```cpp
if (enableOpt) {
  mlir::PassManager pm(module.get()->getName());
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
    return 4;

  pm.addPass(mlir::createInlinerPass());

  mlir::OpPassManager &optPM = pm.nest<mlir::toy::FuncOp>();
  optPM.addPass(mlir::toy::createShapeInferencePass());
  optPM.addPass(mlir::createCanonicalizerPass());
  optPM.addPass(mlir::createCSEPass());

  if (mlir::failed(pm.run(*module)))
    return 4;
}
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

这个顺序不是随便排的。

每个 pass 都在为后面的 pass 创造条件：

```text
inliner
  -> 把函数调用展开，让 caller 里的具体 shape 进入 callee 逻辑

shape inference
  -> 把 tensor<*xf64> 推成 ranked tensor

canonicalizer
  -> 用更具体的类型和结构消除冗余 op

CSE
  -> 合并规范化之后变得相同的重复计算
```

## 为什么需要 inliner

Toy 里用户函数调用会生成：

```mlir
%0 = toy.generic_call @multiply_transpose(%a, %b)
     : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
```

callee 函数本身通常是泛型的：

```mlir
toy.func private @multiply_transpose(
    %arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
  ...
}
```

如果不内联，callee 里只能看到：

```mlir
%arg0: tensor<*xf64>
%arg1: tensor<*xf64>
```

Shape inference 很难从这些 unranked 参数推出具体 shape。

内联后，callee 的函数体被拷贝到 caller 中，原来的 `%arg0`、`%arg1` 会对应到 caller 的实际参数：

```mlir
tensor<2x3xf64>
```

这就让 shape inference 有了足够信息。

## Toy 的 inliner 接口

Toy Dialect 在 `Dialect.cpp` 中注册了：

```cpp
addInterfaces<ToyInlinerInterface>();
```

`ToyInlinerInterface` 继承自：

```cpp
DialectInlinerInterface
```

它告诉 MLIR 通用 inliner：

- 哪些 call 可以被 inline。
- 哪些 operation 可以被 inline。
- inline 后如何处理 terminator。
- call 和 callee 类型不完全一致时如何插入转换。

## inliner 的合法性判断

`ToyInlinerInterface` 中：

```cpp
bool isLegalToInline(Operation *call, Operation *callable,
                     bool wouldBeCloned) const final {
  return true;
}

bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
  return true;
}

bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
  return true;
}
```

Toy 教程里把这些判断都简化为：

```text
都允许 inline
```

这是教学用的保守简化。真实编译器可能会检查可见性、副作用、递归、成本模型、ABI 等问题。

## `GenericCallOp` 和 `CallOpInterface`

`Ops.td` 中：

```tablegen
def GenericCallOp : Toy_Op<"generic_call",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
  ...
}
```

这让 `toy.generic_call` 对通用 inliner 表现得像一个 call operation。

`Dialect.cpp` 里实现了：

```cpp
CallInterfaceCallable GenericCallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

void GenericCallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  (*this)->setAttr("callee", cast<SymbolRefAttr>(callee));
}

Operation::operand_range GenericCallOp::getArgOperands() {
  return getInputs();
}

MutableOperandRange GenericCallOp::getArgOperandsMutable() {
  return getInputsMutable();
}
```

这些方法让 inliner 能知道：

- 这个 call 调用的是谁。
- call 的实参 operands 是什么。
- 如果需要，如何改写 callee。

## inline 后如何处理 `toy.return`

callee 函数体里有：

```mlir
toy.return %result : tensor<*xf64>
```

内联到 caller 后，不能把这个 `toy.return` 原样塞进去，因为 caller 函数不应该在这里提前 return。

所以 `ToyInlinerInterface` 实现了：

```cpp
void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
  auto returnOp = cast<ReturnOp>(op);

  assert(returnOp.getNumOperands() == valuesToRepl.size());
  for (const auto &it : llvm::enumerate(returnOp.getOperands()))
    valuesToRepl[it.index()].replaceAllUsesWith(it.value());
}
```

直观含义是：

```text
把 call result 的使用者，替换成 toy.return 的 operand。
```

所以：

```mlir
%call = toy.generic_call @foo(%x) : ...
toy.print %call : ...
```

内联后会变成：

```mlir
%inlined_result = ...
toy.print %inlined_result : ...
```

原来的 `toy.return` 只是用于把 callee 的结果接回 call result，不会留在 caller 中间。

## `toy.cast` 的作用

Ch4 新增了：

```tablegen
def CastOp : Toy_Op<"cast", [
     DeclareOpInterfaceMethods<CastOpInterface>,
     DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
     Pure,
     SameOperandsAndResultShape
  ]> {
  ...
}
```

`ToyInlinerInterface` 中：

```cpp
Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                     Type resultType,
                                     Location conversionLoc) const final {
  return builder.create<CastOp>(conversionLoc, resultType, input);
}
```

这表示：

```text
如果 inline 过程中 call/callee 的类型需要桥接，
Toy 可以插入 toy.cast。
```

例如，callee 返回的是 `tensor<*xf64>`，caller 期待更具体或不同形式的 tensor type 时，inliner 可以借助 `toy.cast` 表达这种类型转换。

`toy.cast` 不改变数据，只表示类型层面的等价转换。

## Pass 1：inliner 的效果

以 `shape_inference.mlir` 为例，输入包含：

```mlir
toy.func private @multiply_transpose(...) -> tensor<*xf64> {
  ...
}

toy.func @main() {
  ...
  %4 = toy.generic_call @multiply_transpose(%1, %3) : ...
  %5 = toy.generic_call @multiply_transpose(%3, %1) : ...
  toy.print %5 : tensor<*xf64>
  toy.return
}
```

测试要求：

```mlir
// CHECK-NOT: toy.func private @multiply_transpose
```

说明：

```text
private @multiply_transpose 被 inline 后删除了。
```

最终只剩：

```mlir
toy.func @main() {
  ...
}
```

## Pass 2：shape inference 的效果

测试还要求：

```mlir
// CHECK-NOT: tensor<*xf64>
```

说明内联后，shape inference 成功把 unknown shape 推掉了。

例如：

```mlir
toy.transpose(... : tensor<2x3xf64>) to tensor<3x2xf64>
toy.mul ... : tensor<3x2xf64>
toy.print ... : tensor<3x2xf64>
```

这个过程第 13 课已经详细讲过。

在本节中，要记住它在 pipeline 里的作用：

```text
shape inference 给 canonicalizer 和 CSE 提供更精确的类型信息。
```

## Pass 3：canonicalizer 的效果

Canonicalizer 会运行 Toy operation 自己注册的 canonicalization patterns。

Ch4 继承了 Ch3 的高层简化能力，例如：

```text
transpose(transpose(x)) -> x
reshape(reshape(x)) -> reshape(x)
reshape(x) -> x   if type(x) == type(result)
reshape(constant) -> constant
```

在 Ch4 中，shape inference 让类型更具体，canonicalizer 因此能匹配更多规则。

例如 `reshape` 的输入输出类型相同后：

```mlir
%1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>
```

可以被消掉，直接使用 `%0`。

## Pass 4：CSE 的效果

CSE 是 Common Subexpression Elimination，公共子表达式消除。

它的目标是合并重复计算。

如果 IR 中出现两段等价的 pure computation，例如：

```mlir
%0 = toy.transpose(%a : tensor<2x3xf64>) to tensor<3x2xf64>
%1 = toy.transpose(%a : tensor<2x3xf64>) to tensor<3x2xf64>
```

CSE 可以把 `%1` 的使用者改为使用 `%0`。

这依赖几个前提：

- operation 没有副作用。
- operation 及其 operands、attributes、types 足够一致。
- 前面的 canonicalizer 已经把一些不同写法整理成相同形态。

Toy 的很多计算 op 都是 `Pure`，因此更容易被 CSE 处理。

## 为什么 CSE 放最后

如果先跑 CSE，再跑 canonicalizer，可能错过机会。

例如两个计算一开始长得不完全一样：

```text
reshape(constant)
constant
```

Canonicalizer 可能先把 `reshape(constant)` 折叠成新的 `constant`。

这时 CSE 才能发现它和另一个 constant 等价。

所以常见顺序是：

```text
canonicalizer -> CSE
```

Ch4 也是这样安排的。

## 案例 1：`shape_inference.mlir`

输入中有两个函数：

```mlir
toy.func private @multiply_transpose(...)
toy.func @main()
```

`main` 调用了两次：

```mlir
toy.generic_call @multiply_transpose(%1, %3)
toy.generic_call @multiply_transpose(%3, %1)
```

优化后，测试期望：

```mlir
toy.func @main()
  %0 = toy.constant ... : tensor<2x3xf64>
  %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %2 = toy.mul %1, %1 : tensor<3x2xf64>
  toy.print %2 : tensor<3x2xf64>
  toy.return
```

这个结果体现了组合优化：

1. Inliner 删除了 private callee，把调用展开到 `main`。
2. Shape inference 把 transpose/mul/print 的类型推成 ranked tensor。
3. Canonicalizer 消除了多余 reshape，并折叠常量 reshape。
4. CSE 合并了重复常量、重复 transpose 或等价中间值。

最终 IR 比输入短很多，而且没有 `tensor<*xf64>`。

## 案例 2：`transpose_transpose.toy`

Toy 源码：

```toy
def transpose_transpose(x) {
  return transpose(transpose(x));
}

def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b = transpose_transpose(a);
  print(b);
}
```

优化后测试期望：

```mlir
toy.func @main()
  %0 = toy.constant ... : tensor<2x3xf64>
  toy.print %0 : tensor<2x3xf64>
  toy.return
```

这背后发生了：

1. Inliner 把 `transpose_transpose` 函数体展开到 `main`。
2. 函数调用消失。
3. Canonicalizer 看到了 `transpose(transpose(a))`。
4. 双重 transpose 被替换成原始输入 `a`。
5. 死掉的 pure operations 被清理。

所以最后直接打印原始常量。

## `codegen.toy`：未优化基线

Ch4 的 `codegen.toy` 运行命令是：

```text
toyc-ch4 %s -emit=mlir
```

注意没有 `-opt`。

所以它保留了：

```mlir
toy.func private @multiply_transpose(...)
toy.generic_call @multiply_transpose(...)
tensor<*xf64>
```

这很适合作为优化前基线。

对比：

```text
不加 -opt:
  保留函数调用、generic_call、unranked tensor

加 -opt:
  inline 函数、推导 shape、清理冗余、合并重复计算
```

## 为什么 `multiply_transpose` 是 private

Ch4 输出中：

```mlir
toy.func private @multiply_transpose(...)
```

表示这个函数可以被视为模块内部实现细节。

Inliner 内联并删除它之后，不会影响外部可见 API。

测试里：

```mlir
// CHECK-NOT: toy.func private @multiply_transpose
```

正是为了确认 private helper 已经被内联和清理。

## Pass 之间如何互相创造机会

可以把 Ch4 的高层优化理解成一条机会链：

```text
inliner
  暴露 callee body
  让 caller 的 concrete shapes 进入 callee 逻辑

shape inference
  消除 tensor<*xf64>
  让类型约束更具体

canonicalizer
  基于结构和类型消除冗余 reshape/transpose/cast

CSE
  在规范化后的 IR 上合并重复 pure computations
```

这说明优化 pass 通常不是互相独立的。

一个 pass 的结果，往往是另一个 pass 的输入条件。

## 和第 10、11、13 课的关系

第 10 课讲了：

```text
transpose(transpose(x)) -> x
```

第 11 课讲了：

```text
reshape 相关 DRR rewrite
```

第 13 课讲了：

```text
tensor<*xf64> -> tensor<ranked shape>
```

本节把它们串起来：

```text
Inliner 让更多 rewrite 和 shape inference 可见
Shape inference 让类型更具体
Canonicalizer 应用 C++ pattern 和 DRR pattern
CSE 清理重复计算
```

这就是 Ch4 高层优化组合的重点。

## 常见误区

### 误区 1：inliner 只是为了减少函数调用开销

不只是。

在 Toy Ch4 中，inliner 更重要的作用是暴露函数体，让 shape inference 和 canonicalizer 看到更多上下文。

### 误区 2：shape inference 做完就不需要 canonicalizer

不对。

Shape inference 只更新类型。

Canonicalizer 负责基于新类型和结构继续简化 IR。

### 误区 3：CSE 可以代替 canonicalizer

不能。

CSE 合并相同表达式。

Canonicalizer 把表达式变成更规范、更容易相同的形态。

### 误区 4：`toy.cast` 是运行时数据拷贝

不是。

在 Toy 中，`toy.cast` 表示类型层面的转换，不改变数据元素。

### 误区 5：pass 顺序无所谓

不对。

如果 shape inference 放在 inliner 前，很多 callee 里的参数仍然是 unranked。

如果 CSE 放在 canonicalizer 前，可能错过 canonicalizer 暴露出来的重复计算。

## 动手观察

### 观察 pipeline

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

写下每个 pass 的职责。

### 观察 inliner 接口

阅读：

```text
mlir/examples/toy/Ch4/mlir/Dialect.cpp
```

找到：

```cpp
ToyInlinerInterface
handleTerminator
materializeCallConversion
```

回答：

- Toy 为什么允许所有函数 inline。
- `toy.return` inline 后如何处理。
- 类型不匹配时为什么需要 `toy.cast`。

### 观察 `GenericCallOp`

阅读：

```text
mlir/examples/toy/Ch4/include/toy/Ops.td
```

找到：

```tablegen
DeclareOpInterfaceMethods<CallOpInterface>
```

再对照 `Dialect.cpp` 中：

```cpp
getCallableForCallee()
getArgOperands()
```

确认 `toy.generic_call` 如何接入通用 inliner。

### 观察优化前后

对比：

```text
mlir/test/Examples/Toy/Ch4/codegen.toy
mlir/test/Examples/Toy/Ch4/shape_inference.mlir
mlir/test/Examples/Toy/Ch4/transpose_transpose.toy
```

重点看：

- private function 是否消失。
- `toy.generic_call` 是否消失。
- `tensor<*xf64>` 是否消失。
- 双重 transpose 是否消失。
- 重复常量或重复 transpose 是否被合并。

## 本节练习

### 练习 1：解释 Ch4 pipeline

画出：

```text
inliner -> shape inference -> canonicalizer -> CSE
```

并为每个 pass 写一句职责说明。

### 练习 2：解释 inliner 为什么在前

回答：

- 如果不先 inlining，shape inference 会缺少什么信息。
- 为什么 `multiply_transpose` 的参数一开始是 `tensor<*xf64>`。
- inline 后这些参数如何获得具体 shape。

### 练习 3：解释 `handleTerminator`

阅读：

```cpp
void handleTerminator(Operation *op, ValueRange valuesToRepl) const final
```

回答：

- 这里的 terminator 是哪个 operation。
- 为什么 inline 后不能保留 callee 的 `toy.return`。
- `valuesToRepl` 最终被替换成什么。

### 练习 4：解释 `toy.cast`

阅读：

```cpp
materializeCallConversion(...)
```

回答：

- 它什么时候会被 inliner 使用。
- 它创建哪个 Toy operation。
- 为什么 `toy.cast` 不应该理解成数据拷贝。

### 练习 5：分析 `shape_inference.mlir`

阅读测试文件。

回答：

- 为什么 `private @multiply_transpose` 最终消失。
- 为什么最终不应该再出现 `tensor<*xf64>`。
- 最终 `toy.transpose` 的 result type 是什么。
- 为什么最终 `toy.mul` 的两个 operand 可以是同一个 value。

### 练习 6：分析 `transpose_transpose.toy`

回答：

- 哪个 pass 让 `transpose_transpose` 的函数体进入 `main`。
- 哪条 canonicalization 规则消除了双重 transpose。
- 为什么最终可以直接 `toy.print` 原始常量。

### 练习 7：设计一个 CSE 观察例子

写一个 Toy 程序，让同一个纯计算出现两次，例如对同一个变量做两次相同 transpose 或相同乘法。

记录：

- 不加 `-opt` 时是否有重复 operation。
- 加 `-opt` 后重复 operation 是否被合并。
- 如果没有合并，思考是 operation 不完全相同，还是 canonicalizer 没有先把它们变成相同形态。

## 本节小结

Ch4 的重点不是某一个单独 pass，而是高层优化组合：

```text
inliner
  -> 打开函数边界，暴露更多上下文

shape inference
  -> 推导 ranked tensor type

canonicalizer
  -> 消除冗余 reshape/transpose/cast，规范化 IR

CSE
  -> 合并重复 pure computations
```

需要记住：

- `GenericCallOp` 通过 `CallOpInterface` 接入 inliner。
- Toy Dialect 通过 `ToyInlinerInterface` 告诉 inliner 如何处理 Toy IR。
- `handleTerminator()` 把 callee 的 `toy.return` 结果接回 call result。
- `materializeCallConversion()` 用 `toy.cast` 桥接类型差异。
- Shape inference、canonicalizer、CSE 之间是互相创造机会的关系。

下一课会进入第四阶段：Dialect Conversion 基础，开始学习 Toy IR 如何被 lowering 到更低层的 MLIR dialect。

## 学习记录模板

```text
本节主题：Inliner、CSE 与高层 IR 优化组合
我读过的源码：
我观察过的测试：
我能解释的 inliner 机制：
我能解释的 shape inference 与 canonicalizer 配合：
我能解释的 CSE 效果：
我还不理解的问题：
下一步要验证的小实验：
```
