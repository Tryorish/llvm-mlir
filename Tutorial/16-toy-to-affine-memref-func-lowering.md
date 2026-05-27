# 第 16 课：Toy 到 Affine/MemRef/Func 的 Partial Lowering

## 本节定位

第 15 课讲的是 Dialect Conversion 的框架：

```text
ConversionTarget
RewritePatternSet
applyPartialConversion()
```

本节进入具体 lowering 细节。

目标是回答：

```text
每一个 Toy operation 到底被改写成了什么？
```

Ch5 的 lowering 目标是：

```text
Toy Tensor IR
  -> MemRef buffer
  -> Affine loop nest
  -> Arith scalar operation
  -> Func function boundary
```

也就是：

```text
toy.constant   -> memref.alloc + arith.constant + affine.store
toy.add        -> affine.for + affine.load + arith.addf + affine.store
toy.mul        -> affine.for + affine.load + arith.mulf + affine.store
toy.transpose  -> affine.for + affine.load with reversed indices + affine.store
toy.func       -> func.func
toy.return     -> func.return
toy.print      -> 暂时保留，但 operand 从 tensor 变成 memref
```

## 本节目标

- 理解 tensor 到 memref 的转换思路。
- 理解 `convertTensorToMemRef()` 的作用。
- 理解 `insertAllocAndDealloc()` 如何管理临时 buffer。
- 理解 `lowerOpToLoops()` 如何为 tensor result 生成 affine loop nest。
- 理解 `toy.constant` 如何展开成 scalar constants 和 stores。
- 理解 `toy.add` / `toy.mul` 如何在循环内做逐元素运算。
- 理解 `toy.transpose` 如何通过反转访问下标实现。
- 理解 `toy.func` / `toy.return` 如何转到 Func dialect。
- 理解 Ch5 为什么仍然保留 `toy.print`。

## 本节关键文件

源码：

```text
mlir/examples/toy/Ch5/mlir/LowerToAffineLoops.cpp
mlir/examples/toy/Ch5/toyc.cpp
mlir/examples/toy/Ch5/include/toy/Passes.h
```

测试：

```text
mlir/test/Examples/Toy/Ch5/affine-lowering.mlir
mlir/test/Examples/Toy/Ch5/codegen.toy
```

建议阅读顺序：

1. 先看 `affine-lowering.mlir`，知道最终输出长什么样。
2. 再看 `convertTensorToMemRef()` 和 `insertAllocAndDealloc()`。
3. 再看 `lowerOpToLoops()`，理解循环生成骨架。
4. 最后逐个看 lowering pattern：constant、binary、transpose、func、print、return。

## 输入 IR

`affine-lowering.mlir` 的输入很小：

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

它已经满足 Ch5 lowering 的前置条件：

- 函数调用已经内联。
- 所有 shape 都是 ranked。
- 只有 `main`。
- `toy.return` 没有 operand。

这些前置条件很重要。Ch5 lowering 不负责处理泛型函数调用，也不负责推导未知 shape。

## 输出 IR 的整体形态

lowering 后，测试期望看到：

```mlir
func.func @main() {
  %0 = memref.alloc() : memref<3x2xf64>
  %1 = memref.alloc() : memref<3x2xf64>
  %2 = memref.alloc() : memref<2x3xf64>
  ...
  affine.store ...
  affine.for ...
  affine.load ...
  %v = arith.mulf ...
  affine.store ...
  toy.print %0 : memref<3x2xf64>
  memref.dealloc %2 : memref<2x3xf64>
  memref.dealloc %1 : memref<3x2xf64>
  memref.dealloc %0 : memref<3x2xf64>
  return
}
```

关键变化：

- `toy.func` 变成 `func.func`。
- tensor value 变成 memref buffer。
- tensor 计算变成 affine loops。
- 标量运算使用 `arith`。
- `toy.print` 仍然存在，但输入已经是 memref。

## tensor 到 memref

Ch5 的核心类型转换函数：

```cpp
static MemRefType convertTensorToMemRef(RankedTensorType type) {
  return MemRefType::get(type.getShape(), type.getElementType());
}
```

它把：

```mlir
tensor<2x3xf64>
```

转换成：

```mlir
memref<2x3xf64>
```

直观理解：

```text
tensor 是值语义，更高层。
memref 是 buffer 语义，更接近内存。
```

Affine lowering 后，计算结果需要放在可读写的 buffer 里，所以要用 memref。

## 分配和释放 buffer

每个产生 tensor result 的 Toy op，在 lowering 后通常会产生一个 memref result。

辅助函数：

```cpp
static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter) {
  auto alloc = rewriter.create<memref::AllocOp>(loc, type);

  auto *parentBlock = alloc->getBlock();
  alloc->moveBefore(&parentBlock->front());

  auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
  dealloc->moveBefore(&parentBlock->back());
  return alloc;
}
```

它做三件事：

1. 创建 `memref.alloc`。
2. 把 alloc 移到 block 开头。
3. 创建 `memref.dealloc`，并移到 block 末尾。

这适合 Toy 当前的简单控制流模型：

```text
Toy 函数没有复杂控制流，所以 block 开头分配、block 末尾释放是可行的。
```

真实编译器中，内存生命周期通常会更复杂。

## `lowerOpToLoops()`：通用循环骨架

很多 Toy operation 都是逐元素计算：

- `toy.add`
- `toy.mul`
- `toy.transpose`

它们共享一个 lowering 骨架：

```cpp
static void lowerOpToLoops(Operation *op, ValueRange operands,
                           PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {
  auto tensorType = cast<RankedTensorType>(*op->result_type_begin());
  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

  SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), 0);
  SmallVector<int64_t, 4> steps(tensorType.getRank(), 1);
  affine::buildAffineLoopNest(
      rewriter, loc, lowerBounds, tensorType.getShape(), steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        Value valueToStore = processIteration(nestedBuilder, operands, ivs);
        nestedBuilder.create<affine::AffineStoreOp>(loc, valueToStore,
                                                    alloc, ivs);
      });

  rewriter.replaceOp(op, alloc);
}
```

它的核心思想是：

```text
根据 result tensor shape 生成同样维度的 affine loop nest。
每个 iteration 计算一个元素。
把这个元素 store 到 result memref 对应下标。
最后用 result memref 替换原 Toy op。
```

## `LoopIterationFn`

`lowerOpToLoops()` 接受一个回调：

```cpp
using LoopIterationFn = function_ref<Value(
    OpBuilder &rewriter, ValueRange memRefOperands, ValueRange loopIvs)>;
```

这个回调负责回答：

```text
在当前循环下标 loopIvs 处，要存什么值？
```

不同 Toy op 的区别就在这个回调里：

- `toy.add`：load lhs/rhs，然后 `arith.addf`。
- `toy.mul`：load lhs/rhs，然后 `arith.mulf`。
- `toy.transpose`：用反转下标 load 输入。

## loop nest 的形状来自 result type

如果 result type 是：

```mlir
tensor<3x2xf64>
```

那么：

```cpp
tensorType.getRank() = 2
tensorType.getShape() = [3, 2]
```

生成循环：

```mlir
affine.for %i = 0 to 3 {
  affine.for %j = 0 to 2 {
    ...
  }
}
```

这就是为什么 lowering 前必须完成 shape inference。

如果 result 还是：

```mlir
tensor<*xf64>
```

就无法知道生成几层 loop。

## `toy.constant` lowering

`ConstantOpLowering` 做的是：

```text
toy.constant dense<...> : tensor<2x3xf64>
  -> memref.alloc() : memref<2x3xf64>
  -> arith.constant scalar values
  -> affine.store each scalar into memref
```

源码关键步骤：

```cpp
DenseElementsAttr constantValue = op.getValue();
auto tensorType = cast<RankedTensorType>(op.getType());
auto memRefType = convertTensorToMemRef(tensorType);
auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);
```

然后递归遍历常量 shape：

```cpp
std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
  if (dimension == valueShape.size()) {
    rewriter.create<affine::AffineStoreOp>(
        loc, rewriter.create<arith::ConstantOp>(loc, *valueIt++), alloc,
        ArrayRef(indices));
    return;
  }
  ...
};
```

最后：

```cpp
rewriter.replaceOp(op, alloc);
```

也就是用 memref buffer 替换原来的 tensor constant result。

## constant 输出示例

输入：

```mlir
%0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
     : tensor<2x3xf64>
```

输出会包含：

```mlir
%c1 = arith.constant 1.000000e+00 : f64
%c2 = arith.constant 2.000000e+00 : f64
...
%buf = memref.alloc() : memref<2x3xf64>
affine.store %c1, %buf[0, 0] : memref<2x3xf64>
affine.store %c2, %buf[0, 1] : memref<2x3xf64>
...
```

也就是说，常量 tensor 被展开成一个 memref 初始化过程。

## `toy.add` / `toy.mul` lowering

`AddOpLowering` 和 `MulOpLowering` 来自同一个模板：

```cpp
template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public ConversionPattern { ... };

using AddOpLowering = BinaryOpLowering<toy::AddOp, arith::AddFOp>;
using MulOpLowering = BinaryOpLowering<toy::MulOp, arith::MulFOp>;
```

这说明它们的结构一样，只是循环内部的标量运算不同：

```text
toy.add -> arith.addf
toy.mul -> arith.mulf
```

## binary op 的循环体

回调中：

```cpp
auto loadedLhs = builder.create<affine::AffineLoadOp>(
    loc, binaryAdaptor.getLhs(), loopIvs);
auto loadedRhs = builder.create<affine::AffineLoadOp>(
    loc, binaryAdaptor.getRhs(), loopIvs);

return builder.create<LoweredBinaryOp>(loc, loadedLhs, loadedRhs);
```

在每个下标处：

```text
load lhs[i, j]
load rhs[i, j]
compute lhs[i, j] + rhs[i, j] 或 lhs[i, j] * rhs[i, j]
store result[i, j]
```

例如 `toy.mul` 的输出中会看到：

```mlir
%lhs = affine.load %a[%i, %j] : memref<3x2xf64>
%rhs = affine.load %b[%i, %j] : memref<3x2xf64>
%v = arith.mulf %lhs, %rhs : f64
affine.store %v, %out[%i, %j] : memref<3x2xf64>
```

## `toy.transpose` lowering

`TransposeOpLowering` 也复用 `lowerOpToLoops()`。

区别在循环体回调：

```cpp
toy::TransposeOpAdaptor transposeAdaptor(memRefOperands);
Value input = transposeAdaptor.getInput();

SmallVector<Value, 2> reverseIvs(llvm::reverse(loopIvs));
return builder.create<affine::AffineLoadOp>(loc, input, reverseIvs);
```

如果输出下标是：

```text
[i, j]
```

读取输入时使用反转下标：

```text
[j, i]
```

然后把读取到的值 store 到输出：

```text
out[i, j] = input[j, i]
```

这就是矩阵转置。

## transpose 输出示例

输入：

```mlir
%2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
```

输出中会看到：

```mlir
%out = memref.alloc() : memref<3x2xf64>
affine.for %i = 0 to 3 {
  affine.for %j = 0 to 2 {
    %v = affine.load %input[%j, %i] : memref<2x3xf64>
    affine.store %v, %out[%i, %j] : memref<3x2xf64>
  }
}
```

对应测试里的 FileCheck：

```mlir
affine.for [[I]] = 0 to 3 {
  affine.for [[J]] = 0 to 2 {
    [[V]] = affine.load [[INPUT]][[[J]], [[I]]] : memref<2x3xf64>
    affine.store [[V]], [[OUT]][[[I]], [[J]]] : memref<3x2xf64>
  }
}
```

## `toy.func` lowering

`FuncOpLowering`：

```cpp
struct FuncOpLowering : public OpConversionPattern<toy::FuncOp> {
  LogicalResult matchAndRewrite(toy::FuncOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
    if (op.getName() != "main")
      return failure();

    if (op.getNumArguments() || op.getFunctionType().getNumResults())
      return notifyMatchFailure(...);

    auto func = rewriter.create<func::FuncOp>(op.getLoc(), op.getName(),
                                              op.getFunctionType());
    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
    rewriter.eraseOp(op);
    return success();
  }
};
```

它只处理 `main`。

原因是 lowering 前期望所有其他函数已经被 inliner 内联并删除。

转换结果：

```mlir
toy.func @main() { ... }
```

变成：

```mlir
func.func @main() { ... }
```

函数 body region 被搬到新的 `func.func` 中。

## `toy.return` lowering

`ReturnOpLowering`：

```cpp
if (op.hasOperand())
  return failure();

rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
```

Ch5 只支持无返回值的 `main`：

```mlir
toy.return
```

降低为：

```mlir
func.return
```

如果 `toy.return` 还有 operand，pattern 返回 failure。

这符合前置条件：

```text
所有用户函数调用应已内联，最终 main 没有返回值。
```

## `toy.print` lowering

Ch5 还不真正 lowering `toy.print`。

`PrintOpLowering` 做的是：

```cpp
rewriter.modifyOpInPlace(op,
                         [&] { op->setOperands(adaptor.getOperands()); });
```

也就是：

```text
保留 toy.print operation，
但把它的 operand 更新成 lowering 后的 memref value。
```

所以：

```mlir
toy.print %3 : tensor<3x2xf64>
```

变成：

```mlir
toy.print %buf : memref<3x2xf64>
```

这就是 Ch5 partial lowering 的标志之一。

真正把 print lowering 成 runtime helper call，要等 Ch6 的 LLVM lowering。

## 为什么没有 `GenericCallOpLowering`

Ch5 的 patterns 里没有：

```text
GenericCallOpLowering
```

因为 lowering 前要求：

```text
所有函数调用已经被 inline。
```

如果 IR 里还残留：

```mlir
toy.generic_call @foo(...)
```

它属于 illegal Toy dialect，又没有 lowering pattern，`applyPartialConversion()` 会失败。

这也是 pass pipeline 顺序重要的一个例子。

## lowering 后清理

Ch5 `toyc.cpp` 中，在 lower-to-affine 后还会跑：

```cpp
OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
optPM.addPass(mlir::createCanonicalizerPass());
optPM.addPass(mlir::createCSEPass());
```

注意此时 nested 类型已经从：

```cpp
toy::FuncOp
```

变成：

```cpp
func::FuncOp
```

因为 `toy.func` 已经被 lowering 成 `func.func`。

这些清理 pass 会简化 lowering 生成的低层 IR。

如果加了 `-opt`，还会进一步跑 affine 优化，第 17 课会展开。

## 加 `-opt` 和不加 `-opt` 的差异

`affine-lowering.mlir` 同时测试：

```text
toyc-ch5 %s -emit=mlir-affine
toyc-ch5 %s -emit=mlir-affine -opt
```

不加 `-opt` 时，可以看到更多临时 memref。

加 `-opt` 后，affine loop fusion 和 scalar replacement 可能减少中间 buffer。

例如测试中，未优化版本有更多 `memref.alloc`：

```mlir
memref<3x2xf64>
memref<3x2xf64>
memref<2x3xf64>
```

优化版本减少了一个中间 `memref<3x2xf64>`，并把 transpose 和 mul 的计算更紧密地组合到一起。

本节只需要知道差异来自 lowering 后的 affine 优化。第 17 课会详细观察。

## 一张 Toy op lowering 表

| Toy op | Ch5 lowering 结果 | 是否完全离开 Toy dialect |
| --- | --- | --- |
| `toy.constant` | `memref.alloc` + `arith.constant` + `affine.store` | 是 |
| `toy.add` | affine loop + `affine.load` + `arith.addf` + `affine.store` | 是 |
| `toy.mul` | affine loop + `affine.load` + `arith.mulf` + `affine.store` | 是 |
| `toy.transpose` | affine loop + reversed-index `affine.load` + `affine.store` | 是 |
| `toy.func` | `func.func` | 是 |
| `toy.return` | `func.return` | 是 |
| `toy.print` | 保留 `toy.print`，operand 改成 memref | 否 |
| `toy.generic_call` | 不支持，预期已被 inline 消除 | 不应出现 |

## 常见误区

### 误区 1：lowering 后完全没有 Toy dialect

Ch5 不是 full lowering。

`toy.print` 仍然保留，只是 operand 从 tensor 变成 memref。

### 误区 2：tensor 和 memref 只是名字不同

不是。

Tensor 更高层，偏值语义。

MemRef 表示可读写 buffer，更接近内存。

lowering 到 affine loop 时，需要 memref 来承载读写。

### 误区 3：`toy.transpose` lowering 是先真的构造一个转置 tensor

lowering 后没有高层 tensor。

它生成的是：

```text
循环 + 从输入 memref 反转下标 load + store 到输出 memref
```

### 误区 4：`toy.print` 已经 lowering 到 runtime helper

还没有。

Ch5 只是让 `toy.print` 接受 memref。

真正 lowering 到 runtime call 在后续 LLVM lowering 中处理。

### 误区 5：`memref.dealloc` 位置随便放

不随便。

当前实现把 dealloc 移到 block 末尾，是为了保证 buffer 在函数体内使用期间仍然有效。

这依赖 Toy 函数没有复杂控制流。

## 动手观察

### 观察类型转换

阅读：

```cpp
convertTensorToMemRef(RankedTensorType type)
```

回答：

- `tensor<2x3xf64>` 变成什么。
- 为什么要求输入是 `RankedTensorType`。

### 观察 alloc/dealloc

阅读：

```cpp
insertAllocAndDealloc(...)
```

回答：

- `memref.alloc` 被移动到哪里。
- `memref.dealloc` 被移动到哪里。
- 为什么 Toy 当前可以这么做。

### 观察通用循环骨架

阅读：

```cpp
lowerOpToLoops(...)
```

回答：

- loop rank 来自哪里。
- loop upper bounds 来自哪里。
- 每次 iteration 的计算由谁决定。
- 最后 `replaceOp` 用什么替换原 op。

### 观察 binary lowering

阅读：

```cpp
BinaryOpLowering
```

回答：

- `toy.add` 和 `toy.mul` 共用什么逻辑。
- 它们的区别是什么。
- 为什么需要 `affine.load`。

### 观察 transpose lowering

阅读：

```cpp
TransposeOpLowering
```

回答：

- 输出下标 `[i, j]` 时，输入下标是什么。
- 代码里哪一行完成下标反转。

### 观察 affine-lowering 测试

阅读：

```text
mlir/test/Examples/Toy/Ch5/affine-lowering.mlir
```

标注：

- constant 初始化区域。
- transpose loop。
- mul loop。
- print。
- dealloc。

## 本节练习

### 练习 1：解释 `lowerOpToLoops`

用自己的话说明：

```text
lowerOpToLoops = 分配 result memref + 生成 loop nest + 每个元素 store + replace old op
```

并指出：

- result memref type 从哪里来。
- loop 的维度从哪里来。
- 每个元素的计算由哪个回调决定。

### 练习 2：手动 lower 一个 transpose

给定：

```mlir
%0 = toy.transpose(%arg0 : tensor<2x3xf64>) to tensor<3x2xf64>
```

写出伪 lowering：

```mlir
%out = memref.alloc() : memref<3x2xf64>
affine.for %i = 0 to 3 {
  affine.for %j = 0 to 2 {
    %v = affine.load %arg0[%j, %i] : memref<2x3xf64>
    affine.store %v, %out[%i, %j] : memref<3x2xf64>
  }
}
```

解释每个下标为什么这样写。

### 练习 3：手动 lower 一个 mul

给定：

```mlir
%2 = toy.mul %0, %1 : tensor<3x2xf64>
```

写出伪 lowering：

```text
alloc result memref<3x2xf64>
for i in 0..3
  for j in 0..2
    load lhs[i,j]
    load rhs[i,j]
    arith.mulf
    store result[i,j]
```

### 练习 4：解释 constant lowering

回答：

- 为什么 constant 需要 `memref.alloc`。
- 为什么每个元素都要 `affine.store`。
- scalar constant 是用哪个 dialect 表示的。

### 练习 5：解释 `toy.print` 为什么保留

回答：

- Ch5 是否完全 lowering 掉 `toy.print`。
- 它的 operand type 发生了什么变化。
- 为什么这体现了 partial lowering。

### 练习 6：解释 `toy.func` 和 `toy.return`

回答：

- `toy.func @main` 变成什么。
- 为什么只处理 `main`。
- `toy.return` 变成什么。
- 如果 `toy.return` 有 operand，会发生什么。

### 练习 7：整理完整转换表

整理一张表，至少包含：

```text
toy.constant
toy.add
toy.mul
toy.transpose
toy.func
toy.return
toy.print
toy.generic_call
```

每行写：

- lowering pattern 名字。
- 目标 dialect。
- 是否完全消除 Toy op。

## 本节小结

本节最重要的是理解 Ch5 lowering 的基本套路：

```text
tensor value
  -> memref buffer

tensor operation
  -> affine loop nest

element computation
  -> affine.load + arith op + affine.store

function boundary
  -> toy.func / toy.return 到 func.func / func.return
```

需要记住：

- `convertTensorToMemRef()` 把 ranked tensor type 转成 memref type。
- `insertAllocAndDealloc()` 管理临时 buffer 生命周期。
- `lowerOpToLoops()` 是 add/mul/transpose 的共同 lowering 骨架。
- `toy.transpose` 的关键是反转 loop induction variables 作为输入下标。
- `toy.print` 在 Ch5 保留，所以 Ch5 是 partial lowering。
- `toy.generic_call` 不应该进入 Ch5 lowering，因为前面 inliner 应该已经消掉它。

下一课会继续观察 lowering 到 affine 后的 IR，并分析 affine loop fusion 和 scalar replacement 如何进一步优化这些低层 loop。

## 学习记录模板

```text
本节主题：Toy 到 Affine/MemRef/Func 的 Partial Lowering
我读过的源码：
我观察过的测试：
我能解释的 lowerOpToLoops：
我能解释的 Toy op lowering 表：
我手动分析过的 affine loop：
我还不理解的问题：
下一步要验证的小实验：
```
