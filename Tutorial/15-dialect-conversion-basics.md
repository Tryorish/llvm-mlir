# 第 15 课：Dialect Conversion 基础

## 本节定位

第 14 课结束了 Toy 的高层优化阶段。到这里，Toy IR 已经经过：

```text
inliner
shape inference
canonicalizer
CSE
```

接下来课程进入 lowering 阶段。

所谓 lowering，就是把高层 Toy Dialect operation 转换成更低层、更通用的 MLIR dialect operation。

Ch5 的目标是：

```text
Toy Dialect
  -> Affine + Arith + MemRef + Func + Builtin
```

本节先不逐行讲每个 Toy op 如何生成 affine loop。那是第 16 课的内容。

本节重点讲 MLIR Dialect Conversion 的基础框架：

- `ConversionTarget`
- legal / illegal / dynamically legal
- `RewritePatternSet`
- `ConversionPattern`
- `OpConversionPattern`
- `ConversionPatternRewriter`
- `applyPartialConversion()`
- partial lowering 和 full lowering 的区别

## 本节目标

- 理解为什么 lowering 需要 Dialect Conversion。
- 理解 `ConversionTarget` 如何定义“目标 IR 允许什么”。
- 理解 legal dialect、illegal dialect、dynamically legal op。
- 理解 rewrite pattern 和 conversion pattern 的区别。
- 理解 `OpConversionPattern` 和 `ConversionPattern` 的基本使用场景。
- 理解 `applyPartialConversion()` 为什么允许保留一部分 Toy operation。
- 理解 Ch5 为什么叫 partial lowering。
- 能读懂 `LowerToAffineLoops.cpp` 的整体结构。

## 本节关键文件

源码：

```text
mlir/examples/toy/Ch5/mlir/LowerToAffineLoops.cpp
mlir/examples/toy/Ch5/include/toy/Passes.h
mlir/examples/toy/Ch5/toyc.cpp
```

测试：

```text
mlir/test/Examples/Toy/Ch5/affine-lowering.mlir
mlir/test/Examples/Toy/Ch5/codegen.toy
```

建议阅读顺序：

1. 先看 `toyc.cpp`，理解什么时候触发 lowering。
2. 再看 `Passes.h`，找到 `createLowerToAffinePass()` 入口。
3. 再看 `LowerToAffineLoops.cpp` 末尾的 `ToyToAffineLoweringPass::runOnOperation()`。
4. 最后回到文件上半部分，粗略浏览每个 lowering pattern。

## 从 Ch4 到 Ch5

Ch4 只输出高层 Toy MLIR：

```text
toyc-ch4 input.toy -emit=mlir -opt
```

Ch5 新增：

```text
toyc-ch5 input.toy -emit=mlir-affine
```

这表示输出 lowering 到 Affine/MemRef/Func 等 dialect 之后的 MLIR。

在 Ch5 `toyc.cpp` 中：

```cpp
enum Action { None, DumpAST, DumpMLIR, DumpMLIRAffine };
```

并且：

```cpp
bool isLoweringToAffine = emitAction >= Action::DumpMLIRAffine;
```

当目标是 `mlir-affine` 时，会执行：

```cpp
pm.addPass(mlir::toy::createLowerToAffinePass());
```

这就是 Ch5 lowering pass 的入口。

## lowering 前的前置条件

`LowerToAffineLoops.cpp` 文件头部注释说：

```text
This lowering expects that all calls have been inlined,
and all shapes have been resolved.
```

也就是说，lowering 前希望已经完成：

```text
inliner
shape inference
canonicalizer
CSE
```

原因很直接：

- 如果函数调用没内联，`toy.generic_call` 仍然存在，Ch5 lowering 不处理它。
- 如果 shape 还未知，无法生成固定层数和边界的 affine loops。
- 如果还有大量冗余 reshape/transpose，lowering 会生成不必要的低层 IR。

所以 Ch5 的 `toyc.cpp` 在 lowering 前会先跑高层优化：

```cpp
if (enableOpt || isLoweringToAffine) {
  pm.addPass(mlir::createInlinerPass());

  OpPassManager &optPM = pm.nest<mlir::toy::FuncOp>();
  optPM.addPass(mlir::toy::createShapeInferencePass());
  optPM.addPass(mlir::createCanonicalizerPass());
  optPM.addPass(mlir::createCSEPass());
}
```

注意这里即使没有 `-opt`，只要 `-emit=mlir-affine`，也要跑这些前置 pass。

## Dialect Conversion 是什么

Dialect Conversion 是 MLIR 提供的一套通用转换框架。

它解决的问题是：

```text
我想把一批 operation 从一些 dialect 转换到另一些 dialect，
并且希望框架帮我检查转换是否完整、是否合法。
```

它不是简单地“遍历 IR 然后手动改 op”。

Dialect Conversion 需要你提供两类东西：

```text
ConversionTarget
  -> 目标 IR 中什么是 legal，什么是 illegal。

Rewrite patterns
  -> 如何把 illegal op 转换成 legal op。
```

最后调用：

```cpp
applyPartialConversion(...)
```

让框架执行转换并检查结果。

## `ToyToAffineLoweringPass`

Ch5 lowering pass 定义：

```cpp
struct ToyToAffineLoweringPass
    : public PassWrapper<ToyToAffineLoweringPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const override { return "toy-to-affine"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, func::FuncDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() final;
};
```

几个关键点：

- 它是 `OperationPass<ModuleOp>`，跑在整个 module 上。
- 它的 pass argument 是 `toy-to-affine`。
- 它声明依赖 `affine`、`func`、`memref` dialect。
- 真正的转换逻辑在 `runOnOperation()`。

## pass 创建入口

`Passes.h` 中声明：

```cpp
std::unique_ptr<mlir::Pass> createLowerToAffinePass();
```

`LowerToAffineLoops.cpp` 末尾实现：

```cpp
std::unique_ptr<Pass> mlir::toy::createLowerToAffinePass() {
  return std::make_unique<ToyToAffineLoweringPass>();
}
```

`toyc.cpp` 中使用：

```cpp
pm.addPass(mlir::toy::createLowerToAffinePass());
```

这条链路是：

```text
toyc.cpp
  -> createLowerToAffinePass()
  -> ToyToAffineLoweringPass
  -> runOnOperation()
```

## `ConversionTarget`

`runOnOperation()` 开头：

```cpp
ConversionTarget target(getContext());
```

`ConversionTarget` 用来定义转换后的 IR 允许出现哪些 operation 或 dialect。

可以把它理解成：

```text
目标合法性规则表
```

转换结束后，Dialect Conversion 会检查 IR：

- legal 的可以留下。
- illegal 的必须被 pattern 转走。
- dynamically legal 的要根据条件判断能不能留下。

## legal dialect

Ch5 中：

```cpp
target.addLegalDialect<affine::AffineDialect, BuiltinDialect,
                       arith::ArithDialect, func::FuncDialect,
                       memref::MemRefDialect>();
```

这表示 lowering 后允许出现这些 dialect：

- `affine`
- `builtin`
- `arith`
- `func`
- `memref`

例如 lowering 后可以出现：

```mlir
func.func @main()
%0 = memref.alloc() : memref<2x3xf64>
affine.for %i = 0 to 2 { ... }
%c0 = arith.constant 0 : index
```

这些都属于 legal target。

## illegal dialect

Ch5 中：

```cpp
target.addIllegalDialect<toy::ToyDialect>();
```

这表示：

```text
默认情况下，Toy Dialect 里的 operation 都不应该留在转换后的 IR 中。
```

换句话说，转换框架会要求你把 Toy operations 转成目标 dialect operations。

如果转换结束后还残留非法 Toy op，conversion 会失败。

## dynamically legal op

但是 Ch5 是 partial lowering，并不想完全消除所有 Toy operations。

`toy.print` 暂时保留。

所以代码写了：

```cpp
target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
  return none_of(op->getOperandTypes(),
                 [](Type type) { return isa<TensorType>(type); });
});
```

这表示：

```text
toy.print 是否 legal，不只看名字，还要看 operand type。
```

如果 `toy.print` 的 operand 还是 tensor：

```mlir
toy.print %0 : tensor<2x3xf64>
```

它还不 legal。

如果 operand 已经被转换成 memref：

```mlir
toy.print %0 : memref<2x3xf64>
```

它就是 legal 的，可以留下。

这就是 dynamically legal。

## 为什么保留 `toy.print`

Ch5 的目标是 lowering 到 Affine/MemRef/Func 层级，但 `toy.print` 还没有完全降低到 runtime call。

所以 Ch5 只做 partial lowering：

```text
把计算密集的 tensor operations 降到 affine loops 和 memref。
暂时保留 toy.print 作为高层输出操作。
```

真正把 print 降到更低层，会在后续 Ch6 的 LLVM lowering 中继续处理。

## Rewrite patterns

定义完合法性目标之后，要告诉 conversion framework：

```text
遇到 illegal op 时怎么转换。
```

Ch5 中：

```cpp
RewritePatternSet patterns(&getContext());
patterns.add<AddOpLowering, ConstantOpLowering, FuncOpLowering, MulOpLowering,
             PrintOpLowering, ReturnOpLowering, TransposeOpLowering>(
    &getContext());
```

这些 pattern 负责把 Toy op 转成 lower-level IR：

| Pattern | 处理的 Toy op | 目标 |
| --- | --- | --- |
| `AddOpLowering` | `toy.add` | affine loop + `arith.addf` |
| `MulOpLowering` | `toy.mul` | affine loop + `arith.mulf` |
| `ConstantOpLowering` | `toy.constant` | `memref.alloc` + constants + stores |
| `FuncOpLowering` | `toy.func` | `func.func` |
| `PrintOpLowering` | `toy.print` | 保留 op，但更新 operands |
| `ReturnOpLowering` | `toy.return` | `func.return` |
| `TransposeOpLowering` | `toy.transpose` | affine loop + reversed indices load |

第 16 课会逐个展开这些 pattern 的生成 IR。

本节先关注它们如何接入 conversion framework。

## `ConversionPattern`

Ch5 中 `AddOpLowering` 和 `MulOpLowering` 来自模板：

```cpp
template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public ConversionPattern {
  BinaryOpLowering(MLIRContext *ctx)
      : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const final {
    ...
  }
};
```

`ConversionPattern` 和普通 rewrite pattern 类似，但它服务于 dialect conversion。

它的 `matchAndRewrite` 参数里有：

```cpp
ArrayRef<Value> operands
```

这不是原始 operands，而是 conversion framework 已经根据转换结果 remap 后的 operands。

例如，原来 operand 是 tensor value，转换过程中可能已经变成 memref value。

## `OpConversionPattern`

Ch5 中 `FuncOpLowering`、`PrintOpLowering` 使用：

```cpp
struct FuncOpLowering : public OpConversionPattern<toy::FuncOp> { ... };
struct PrintOpLowering : public OpConversionPattern<toy::PrintOp> { ... };
```

它们的签名是：

```cpp
LogicalResult matchAndRewrite(toy::FuncOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const final
```

`OpAdaptor` 提供已经转换或 remapped 之后的 operands。

这在处理 operand type conversion 时很有用。

## `OpRewritePattern`

Ch5 里也还有普通的 rewrite pattern：

```cpp
struct ConstantOpLowering : public OpRewritePattern<toy::ConstantOp> { ... };
struct ReturnOpLowering : public OpRewritePattern<toy::ReturnOp> { ... };
```

它们不需要使用 conversion adaptor，也不依赖复杂 operand remapping。

所以 Ch5 同时出现了：

- `ConversionPattern`
- `OpConversionPattern`
- `OpRewritePattern`

这不是冲突，而是根据具体转换需求选择合适的 pattern 基类。

## `ConversionPatternRewriter`

在 conversion pattern 中常见：

```cpp
ConversionPatternRewriter &rewriter
```

它和普通 `PatternRewriter` 类似，但服务于 dialect conversion。

常见用法：

```cpp
rewriter.create<...>(...)
rewriter.replaceOp(...)
rewriter.replaceOpWithNewOp<...>(...)
rewriter.inlineRegionBefore(...)
rewriter.eraseOp(...)
rewriter.modifyOpInPlace(...)
rewriter.notifyMatchFailure(...)
```

conversion rewrite 要通过它修改 IR，让 conversion driver 能追踪 legality、operand remapping 和 replacement。

## `applyPartialConversion()`

最后一步：

```cpp
if (failed(applyPartialConversion(getOperation(), target,
                                  std::move(patterns))))
  signalPassFailure();
```

这句话会：

1. 从当前 `ModuleOp` 开始执行 conversion。
2. 根据 `target` 判断哪些 op legal / illegal。
3. 用 `patterns` 尝试转换 illegal op。
4. 转换结束后检查还有没有非法 op。
5. 如果转换失败，pass 失败。

这里用的是：

```cpp
applyPartialConversion
```

不是 full conversion。

原因是 Ch5 允许保留部分 Toy op，例如 dynamically legal 的 `toy.print`。

## Partial conversion 是什么

Partial conversion 表示：

```text
不是所有原 dialect operation 都必须消失。
只要剩下的 operation 被 target 判定为 legal，就允许保留。
```

Ch5 的结果中仍可能出现：

```mlir
toy.print %0 : memref<3x2xf64>
```

这就是 partial lowering。

Toy 的计算部分已经降低到：

```text
affine / arith / memref / func
```

但输出操作仍保留为 Toy Dialect。

## Full conversion 是什么

Full conversion 通常要求：

```text
目标 IR 中不能残留任何不合法 operation。
```

例如后续 Ch6 lowering 到 LLVM Dialect 时，目标更接近：

```text
尽量消除 Toy、Affine、MemRef、Func 等高层 dialect，
最终转成 LLVM Dialect 可导出的形式。
```

Full conversion 更严格，通常用于最终 lowering 阶段。

Ch5 先做 partial conversion，是因为还没准备好一次性把所有 Toy 语义降到底。

## Type conversion 的直观理解

Ch5 有一个辅助函数：

```cpp
static MemRefType convertTensorToMemRef(RankedTensorType type) {
  return MemRefType::get(type.getShape(), type.getElementType());
}
```

这体现了一个重要转换：

```text
tensor<2x3xf64>
  -> memref<2x3xf64>
```

Toy 高层计算使用 tensor value。

Affine lowering 后，计算结果需要存储在 buffer 中，所以使用 memref。

Ch5 没有引入完整的 `TypeConverter` 类，而是手写了一个简单的 type conversion helper。

你可以先把 type conversion 理解为：

```text
operation 变了，value 的 type 往往也要跟着变。
```

## `PrintOpLowering` 为什么只更新 operands

`PrintOpLowering`：

```cpp
struct PrintOpLowering : public OpConversionPattern<toy::PrintOp> {
  LogicalResult matchAndRewrite(toy::PrintOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
    rewriter.modifyOpInPlace(op,
                             [&] { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};
```

它不删除 `toy.print`。

它只是把 print 的 operands 更新成 conversion 后的 values。

原因是：

```text
toy.print 可以暂时保留，
但它不能继续使用 tensor operand，
因为上游计算已经变成 memref。
```

所以动态合法性要求：

```text
toy.print legal iff its operands are not TensorType.
```

## `FuncOpLowering` 为什么只处理 main

`FuncOpLowering` 中：

```cpp
if (op.getName() != "main")
  return failure();
```

并且要求：

```cpp
if (op.getNumArguments() || op.getFunctionType().getNumResults())
  return notifyMatchFailure(...);
```

原因是 lowering 前预期所有函数调用已经 inline。

所以最后应该只剩：

```mlir
toy.func @main() { ... }
```

然后把它转换成：

```mlir
func.func @main() { ... }
```

如果还有别的 Toy 函数，说明前置 inlining 不完整。

## `affine-lowering.mlir` 观察

测试文件：

```text
mlir/test/Examples/Toy/Ch5/affine-lowering.mlir
```

输入：

```mlir
toy.func @main() {
  %0 = toy.constant ... : tensor<2x3xf64>
  %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %3 = toy.mul %2, %2 : tensor<3x2xf64>
  toy.print %3 : tensor<3x2xf64>
  toy.return
}
```

输出中可以看到：

```mlir
func @main()
memref.alloc
arith.constant
affine.store
affine.for
affine.load
arith.mulf
toy.print ... : memref<3x2xf64>
memref.dealloc
```

这说明：

- `toy.func` 已经变成 `func.func`。
- `toy.constant` 已经变成 memref 分配和 store。
- `toy.transpose` 已经变成 affine loops。
- `toy.mul` 已经变成 affine loops + `arith.mulf`。
- `toy.return` 已经变成 `func.return`。
- `toy.print` 仍然保留，但 operand 已经是 memref。

这就是 partial lowering 的结果。

## 常见误区

### 误区 1：lowering 就是普通 rewrite

不完全是。

普通 rewrite 关注局部替换。

Dialect Conversion 还要检查目标 IR 的 legality，确保非法 operation 被转换掉。

### 误区 2：`addIllegalDialect<ToyDialect>()` 表示所有 Toy op 都不能留下

默认是这样，但可以用 dynamically legal op 放宽。

Ch5 就允许 `toy.print` 在 operand 不再是 tensor 时留下。

### 误区 3：partial lowering 是失败的 lowering

不是。

Partial lowering 是有意保留部分合法 operation。

只要 target 认为剩下的 IR legal，conversion 就成功。

### 误区 4：type conversion 只发生在类型系统里

不只是。

当 tensor 变成 memref 后，operation 的 operands、results、loads/stores、allocation 方式都要相应改变。

### 误区 5：lowering 到 Affine 后就等于最终代码

不是。

Affine 是更低层、可优化的中间表示。

后面还要继续 lowering 到 LLVM Dialect，再导出 LLVM IR 或 JIT 执行。

## 动手观察

### 观察 pass 入口

阅读：

```text
mlir/examples/toy/Ch5/toyc.cpp
```

找到：

```cpp
bool isLoweringToAffine = emitAction >= Action::DumpMLIRAffine;
pm.addPass(mlir::toy::createLowerToAffinePass());
```

回答：

- 什么命令会触发 Affine lowering。
- 为什么 lowering 前要先跑 inliner 和 shape inference。

### 观察 conversion target

阅读：

```text
mlir/examples/toy/Ch5/mlir/LowerToAffineLoops.cpp
```

找到：

```cpp
ConversionTarget target(getContext());
target.addLegalDialect<...>();
target.addIllegalDialect<toy::ToyDialect>();
target.addDynamicallyLegalOp<toy::PrintOp>(...);
```

整理：

- 哪些 dialect legal。
- 哪个 dialect illegal。
- 哪个 op dynamically legal。

### 观察 patterns

找到：

```cpp
patterns.add<AddOpLowering, ConstantOpLowering, FuncOpLowering, MulOpLowering,
             PrintOpLowering, ReturnOpLowering, TransposeOpLowering>(
    &getContext());
```

回答：

- 每个 pattern 大概负责哪个 Toy op。
- 为什么没有 `GenericCallOpLowering`。

### 观察 partial conversion

找到：

```cpp
applyPartialConversion(getOperation(), target, std::move(patterns))
```

回答：

- 它检查什么。
- 为什么这里不是 full conversion。
- 如果转换后还有非法 Toy op，会发生什么。

### 观察测试输出

阅读：

```text
mlir/test/Examples/Toy/Ch5/affine-lowering.mlir
```

确认：

- `toy.func` 是否消失。
- `toy.constant` 是否消失。
- `toy.transpose` 是否消失。
- `toy.mul` 是否消失。
- `toy.print` 是否保留。
- `toy.print` 的 operand type 是否从 tensor 变成 memref。

## 本节练习

### 练习 1：解释 conversion target

回答：

- `ConversionTarget` 负责什么。
- legal dialect 表示什么。
- illegal dialect 表示什么。
- dynamically legal op 表示什么。

### 练习 2：解释 Ch5 为什么是 partial lowering

回答：

- 哪些 Toy op 被 lowering 掉。
- 哪个 Toy op 被保留。
- 被保留的 Toy op 满足什么条件才 legal。

### 练习 3：解释 `toy.print` 的动态合法性

阅读：

```cpp
target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
  return none_of(op->getOperandTypes(),
                 [](Type type) { return isa<TensorType>(type); });
});
```

回答：

- 如果 `toy.print` 输入是 `tensor<2x3xf64>`，是否合法。
- 如果输入是 `memref<2x3xf64>`，是否合法。
- 为什么要这样设计。

### 练习 4：比较三种 pattern

在 `LowerToAffineLoops.cpp` 中找：

```text
ConversionPattern
OpConversionPattern
OpRewritePattern
```

回答：

- 哪些 lowering 用了 `ConversionPattern`。
- 哪些用了 `OpConversionPattern`。
- 哪些用了 `OpRewritePattern`。
- 它们在参数和使用场景上有什么不同。

### 练习 5：解释 `applyPartialConversion`

回答：

- 它需要哪三个核心输入。
- 它如何判断 conversion 成功或失败。
- 如果一个 illegal op 没有匹配 pattern，会发生什么。

### 练习 6：整理 Toy op lowering 表

根据本节和 `LowerToAffineLoops.cpp`，整理表格：

```text
toy.constant -> ?
toy.add      -> ?
toy.mul      -> ?
toy.transpose -> ?
toy.func     -> ?
toy.return   -> ?
toy.print    -> ?
```

不用写细节，先写目标 dialect 和大概形式。

### 练习 7：预测转换失败

思考：

如果 lowering 前还存在：

```mlir
toy.generic_call @foo(...)
```

而 Ch5 没有提供 `GenericCallOpLowering`，会发生什么？

回答：

- 它是否属于 illegal Toy dialect。
- 它有没有 pattern 转换。
- `applyPartialConversion()` 会成功还是失败。

## 本节小结

本节最重要的是理解 Dialect Conversion 的三件套：

```text
ConversionTarget
  -> 定义目标 IR 合法性

RewritePatternSet
  -> 提供 illegal op 的转换方式

applyPartialConversion()
  -> 执行转换并检查结果
```

也要记住 Ch5 的核心特点：

- 它是 Toy 到 Affine/MemRef/Func/Arith 的 partial lowering。
- 默认 Toy Dialect illegal。
- `toy.print` 被动态标记为 legal，只要它的 operand 不再是 tensor。
- tensor type 会被转换成 memref type。
- conversion 前需要 inlining 和 shape inference。
- conversion 后还会跑 canonicalizer 和 CSE 清理低层 IR。

下一课会进入具体 lowering 细节：逐个分析 `toy.constant`、`toy.add`、`toy.mul`、`toy.transpose`、`toy.func`、`toy.return` 如何被转换成低层 dialect operation。

## 学习记录模板

```text
本节主题：Dialect Conversion 基础
我读过的源码：
我理解的 ConversionTarget：
我整理的 legal / illegal / dynamically legal：
我理解的 conversion patterns：
我理解的 partial lowering：
我还不理解的问题：
下一步要验证的小实验：
```
