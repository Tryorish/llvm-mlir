# 第 10 课：C++ Pattern Rewrite 与 Canonicalization

## 本节定位

第 9 课已经把 Toy 的函数、调用、作用域和符号讲清楚了。到这里为止，Toy 前端已经能生成比较完整的高层 MLIR。

从第 10 课开始，课程进入第三阶段：高层优化与分析。

本节关注 Ch3 中的第一类优化：

```text
transpose(transpose(x)) -> x
```

这是一条很典型的局部重写规则。它不需要全局数据流分析，也不需要 lowering，只需要在 IR 里找到一个局部形态：

```mlir
%0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
%1 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
```

然后把第二个 transpose 的结果替换成原始输入：

```mlir
%arg0
```

本节要回答：

- Pattern rewrite 是什么。
- `OpRewritePattern` 怎么写。
- `matchAndRewrite` 应该做什么。
- `PatternRewriter::replaceOp` 如何替换 value。
- `getCanonicalizationPatterns` 如何把 pattern 注册到 operation 上。
- `createCanonicalizerPass()` 如何触发这些 canonicalization patterns。

## 本节目标

- 理解 MLIR pattern rewrite 的基本模型。
- 能读懂 `SimplifyRedundantTranspose` 这条 C++ rewrite pattern。
- 理解 `match` 和 `rewrite` 为什么放在同一个 `matchAndRewrite()` 中。
- 理解 `getDefiningOp<TransposeOp>()` 的作用。
- 理解 `PatternRewriter::replaceOp()` 替换的是 operation result 的使用者。
- 理解 `let hasCanonicalizer = 1` 和 `getCanonicalizationPatterns()` 的关系。
- 理解 `toyc-ch3 -emit=mlir -opt` 中 `-opt` 如何启用 canonicalizer。
- 能解释 `transpose_transpose.toy` 的优化前后变化。

## 本节关键文件

源码：

```text
mlir/examples/toy/Ch3/mlir/ToyCombine.cpp
mlir/examples/toy/Ch3/include/toy/Ops.td
mlir/examples/toy/Ch3/toyc.cpp
```

测试：

```text
mlir/test/Examples/Toy/Ch3/transpose_transpose.toy
mlir/test/Examples/Toy/Ch3/trivial_reshape.toy
```

建议阅读顺序：

1. 先看 `transpose_transpose.toy`，明确要优化的 IR 形态。
2. 再看 `ToyCombine.cpp` 中的 `SimplifyRedundantTranspose`。
3. 再看 `Ops.td` 中 `TransposeOp` 的 `hasCanonicalizer`。
4. 最后看 `toyc.cpp` 中 `createCanonicalizerPass()` 的接入方式。

## 从 Ch2 到 Ch3 的变化

Ch2 的重点是生成 Toy Dialect IR。

Ch3 开始加入优化：

```text
Ch2:
  Toy AST -> Toy MLIR

Ch3:
  Toy AST -> Toy MLIR -> Canonicalization
```

在 Ch3 中，`toyc` 多了一个命令行选项：

```cpp
static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));
```

当运行：

```bash
toyc-ch3 input.toy -emit=mlir -opt
```

就会启用优化 pipeline。

## `transpose_transpose.toy`

测试文件：

```text
mlir/test/Examples/Toy/Ch3/transpose_transpose.toy
```

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

核心表达式是：

```toy
transpose(transpose(x))
```

如果不优化，它应该先生成内层 transpose，再生成外层 transpose：

```mlir
%0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
%1 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
toy.return %1 : tensor<*xf64>
```

优化后，测试期望变成：

```mlir
toy.return %arg0 : tensor<*xf64>
```

也就是两个 transpose 都被消掉。

## Pattern rewrite 是什么

Pattern rewrite 可以理解为：

```text
发现一种 IR 形态
  -> 判断它是否符合某条规则
  -> 用更简单或更规范的 IR 替换它
```

本节的规则是：

```text
输入形态：
  toy.transpose(toy.transpose(x))

输出形态：
  x
```

这种规则通常是局部的：

- 只看当前 operation。
- 可能看当前 operation 的 operand。
- 不需要扫描整个 module。
- 不需要知道完整控制流。

这正是 canonicalization 适合处理的任务。

## Canonicalization 是什么

Canonicalization 可以翻译成“规范化”。

它的目标不是做所有优化，而是把 IR 改写成更简单、更稳定、更标准的形态。

例如：

```text
transpose(transpose(x)) -> x
reshape(reshape(x)) -> reshape(x)
reshape(constant) -> new constant
```

规范化之后，后续 pass 更容易分析和处理 IR。

在 MLIR 中，很多 operation 可以自己注册 canonicalization patterns。通用 canonicalizer pass 会收集这些 patterns 并反复应用。

## `OpRewritePattern`

Ch3 的 C++ pattern 定义在：

```text
mlir/examples/toy/Ch3/mlir/ToyCombine.cpp
```

核心代码：

```cpp
struct SimplifyRedundantTranspose
    : public mlir::OpRewritePattern<TransposeOp> {
  SimplifyRedundantTranspose(mlir::MLIRContext *context)
      : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}

  llvm::LogicalResult
  matchAndRewrite(TransposeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value transposeInput = op.getOperand();
    TransposeOp transposeInputOp =
        transposeInput.getDefiningOp<TransposeOp>();

    if (!transposeInputOp)
      return failure();

    rewriter.replaceOp(op, {transposeInputOp.getOperand()});
    return success();
  }
};
```

先看继承关系：

```cpp
OpRewritePattern<TransposeOp>
```

表示这条 pattern 只尝试匹配 `toy.transpose` operation。

也就是说，canonicalizer 看到很多 operation 时，只有遇到 `TransposeOp`，才会调用这条 pattern。

## benefit 是什么

构造函数里有：

```cpp
OpRewritePattern<TransposeOp>(context, /*benefit=*/1)
```

`benefit` 表示 pattern 的收益，用于在多个 pattern 都可能匹配时帮助框架排序。

本例只有一条简单规则，所以 `benefit = 1` 就够了。

实际项目里，如果多个 pattern 都能匹配同一个 operation，benefit 可以帮助 canonicalizer 优先选择更有价值的 rewrite。

## `matchAndRewrite()` 的职责

`matchAndRewrite()` 负责两件事：

```text
match:
  判断当前 op 是否符合规则

rewrite:
  如果符合，就用 rewriter 修改 IR
```

函数签名：

```cpp
LogicalResult matchAndRewrite(TransposeOp op,
                              PatternRewriter &rewriter) const override
```

参数含义：

- `op`：当前正在尝试匹配的 `toy.transpose`。
- `rewriter`：用于修改 IR 的工具，所有替换、删除、创建 operation 都应该通过它完成。

返回值：

- `success()`：匹配并完成重写。
- `failure()`：不匹配，IR 不应该被修改。

## 匹配外层 transpose

假设 IR 是：

```mlir
%0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
%1 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
toy.return %1 : tensor<*xf64>
```

当 pattern 被应用到外层 transpose：

```mlir
%1 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
```

代码里：

```cpp
mlir::Value transposeInput = op.getOperand();
```

得到的是：

```text
transposeInput = %0
```

接着：

```cpp
TransposeOp transposeInputOp =
    transposeInput.getDefiningOp<TransposeOp>();
```

它会检查 `%0` 是否由另一个 `TransposeOp` 定义。

如果 `%0` 来自：

```mlir
%0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
```

那么 `transposeInputOp` 就是内层 transpose。

## 不匹配时返回 failure

如果当前 IR 是：

```mlir
%0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
toy.return %0 : tensor<*xf64>
```

这里当前 transpose 的输入是 `%arg0`。

但 `%arg0` 是 block argument，不是由 `toy.transpose` 定义的 operation result。

所以：

```cpp
transposeInput.getDefiningOp<TransposeOp>()
```

返回空。

pattern 应该返回：

```cpp
return failure();
```

这表示当前 op 不符合这条规则，canonicalizer 可以继续尝试其他 pattern 或其他 operation。

## 匹配成功时替换

匹配到双重 transpose 后，代码执行：

```cpp
rewriter.replaceOp(op, {transposeInputOp.getOperand()});
```

这里的 `op` 是外层 transpose。

`transposeInputOp.getOperand()` 是内层 transpose 的输入，也就是原始的 `x`。

所以这句代码的意思是：

```text
把外层 transpose 的所有结果使用者
  替换成
内层 transpose 的输入 value
```

如果原来是：

```mlir
%0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
%1 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
toy.return %1 : tensor<*xf64>
```

替换后，`toy.return` 使用的 `%1` 会改成 `%arg0`：

```mlir
%0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
toy.return %arg0 : tensor<*xf64>
```

外层 transpose 被删除。

然后内层 transpose 变成没有使用者的 pure operation。Canonicalizer 后续可以把它也删掉，所以最终输出是：

```mlir
toy.return %arg0 : tensor<*xf64>
```

## 为什么 `TransposeOp` 需要 `Pure`

Ch3 的 `Ops.td` 中，`TransposeOp` 定义为：

```tablegen
def TransposeOp : Toy_Op<"transpose", [Pure]> {
  ...
}
```

`Pure` 表示这个 operation 没有可观察副作用。

这很重要，因为如果一个 operation 有副作用，即使它的结果没人用，也不能随便删除。

例如：

```mlir
toy.print %0 : tensor<...>
```

`toy.print` 有可观察行为，不能因为结果没人用就删掉。

而 `toy.transpose` 只是计算一个新 tensor，结果没人用时可以删。

所以在这个测试里：

```text
外层 transpose 被 replaceOp 删除
内层 transpose 变成 dead op
因为它是 Pure，可以被清理掉
```

## `PatternRewriter` 的规则

在 pattern 中修改 IR 时，应该通过 `PatternRewriter`，而不是直接手动 erase 或乱改 use-list。

常见方法包括：

```cpp
rewriter.replaceOp(op, newValues);
rewriter.eraseOp(op);
rewriter.create<SomeOp>(...);
```

本节只用到：

```cpp
replaceOp
```

它同时完成：

- 替换旧 operation results 的所有 uses。
- 删除旧 operation。
- 让 rewrite driver 能正确跟踪 IR 变化。

## `getDefiningOp<T>()`

这一句很关键：

```cpp
transposeInput.getDefiningOp<TransposeOp>()
```

它的含义是：

```text
如果这个 Value 是某个 operation 的 result，
并且这个 defining operation 是 TransposeOp，
就返回这个 TransposeOp。

否则返回空。
```

常见返回空的情况：

- 这个 value 是 block argument。
- 这个 value 是其他 operation 的 result，例如 `toy.constant`。
- 这个 value 没有 defining op。

因此它天然适合检查：

```text
当前 transpose 的输入是否来自另一个 transpose
```

## 在 ODS 中启用 canonicalizer

只写 C++ pattern 还不够。operation 必须声明自己有 canonicalization patterns。

Ch3 的 `Ops.td` 中：

```tablegen
def TransposeOp : Toy_Op<"transpose", [Pure]> {
  ...
  let hasCanonicalizer = 1;
  ...
}
```

`hasCanonicalizer = 1` 会让 TableGen 生成相关声明，使你可以在 C++ 中实现：

```cpp
void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<SimplifyRedundantTranspose>(context);
}
```

这一步的作用是：

```text
把 SimplifyRedundantTranspose 注册为 TransposeOp 的 canonicalization pattern
```

## `RewritePatternSet`

`getCanonicalizationPatterns()` 的第一个参数是：

```cpp
RewritePatternSet &results
```

它是 pattern 的集合。

在 `ToyCombine.cpp` 里：

```cpp
results.add<SimplifyRedundantTranspose>(context);
```

表示往集合里添加一条 pattern。

当 canonicalizer pass 运行时，它会从 operation 上收集这些 patterns，然后尝试应用。

## 在 `toyc.cpp` 中启用 canonicalizer pass

Ch3 的 `toyc.cpp` 中：

```cpp
static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));
```

当命令行带 `-opt` 时：

```cpp
if (enableOpt) {
  mlir::PassManager pm(module.get()->getName());
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
    return 4;

  pm.addNestedPass<mlir::toy::FuncOp>(mlir::createCanonicalizerPass());
  if (mlir::failed(pm.run(*module)))
    return 4;
}
```

关键是：

```cpp
pm.addNestedPass<mlir::toy::FuncOp>(mlir::createCanonicalizerPass());
```

意思是：

```text
在每个 toy.func 内运行 canonicalizer pass
```

为什么是 nested pass？

因为 Toy 的大部分可优化 operation 都在函数 body 里：

```mlir
toy.func @foo(...) {
  %0 = toy.transpose(...)
  ...
}
```

所以 canonicalizer 被加到 `toy.func` 这个层级运行。

## `-opt` 的效果

运行：

```bash
toyc-ch3 mlir/test/Examples/Toy/Ch3/transpose_transpose.toy -emit=mlir
```

如果不加 `-opt`，你应该能看到双重 transpose。

运行：

```bash
toyc-ch3 mlir/test/Examples/Toy/Ch3/transpose_transpose.toy -emit=mlir -opt
```

加上 `-opt` 后，测试期望是：

```mlir
toy.func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64>
  toy.return %arg0 : tensor<*xf64>
}
```

这说明 canonicalizer 已经应用了 `SimplifyRedundantTranspose`。

## 优化前后的逐步变化

假设初始 IR 是：

```mlir
toy.func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
  %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
  %1 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
  toy.return %1 : tensor<*xf64>
}
```

第 1 步：pattern 作用于外层 transpose：

```text
op = %1 = toy.transpose(%0)
transposeInput = %0
transposeInputOp = %0 = toy.transpose(%arg0)
```

第 2 步：执行替换：

```cpp
rewriter.replaceOp(op, {transposeInputOp.getOperand()});
```

也就是：

```text
replace %1 with %arg0
```

第 3 步：IR 变成：

```mlir
toy.func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
  %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
  toy.return %arg0 : tensor<*xf64>
}
```

第 4 步：`%0` 没有使用者，且 `toy.transpose` 是 `Pure`，可以被删掉：

```mlir
toy.func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
  toy.return %arg0 : tensor<*xf64>
}
```

## `trivial_reshape.toy` 暂时只观察

同一个 `ToyCombine.cpp` 里还有：

```cpp
void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<ReshapeReshapeOptPattern, RedundantReshapeOptPattern,
              FoldConstantReshapeOptPattern>(context);
}
```

这些 pattern 来自：

```cpp
#include "ToyCombine.inc"
```

而 `ToyCombine.inc` 是由：

```text
mlir/examples/toy/Ch3/mlir/ToyCombine.td
```

通过 TableGen 生成的。

本节先不展开 DRR 语法，只需要知道：

- `TransposeOp` 的 canonicalization pattern 是手写 C++。
- `ReshapeOp` 的 canonicalization patterns 是声明式 DRR 生成的。

第 11 课会专门讲 `ToyCombine.td`。

## C++ pattern 适合什么场景

C++ pattern 适合：

- 需要复杂条件判断。
- 需要读取多个 operation 的属性、类型或 use-def 链。
- 需要创建多个新 operation。
- 需要调用 C++ helper 函数。
- 声明式 pattern 表达起来不直观。

本节的 `SimplifyRedundantTranspose` 虽然很简单，但它展示了 C++ pattern 的基本结构：

```cpp
struct MyPattern : OpRewritePattern<MyOp> {
  LogicalResult matchAndRewrite(MyOp op,
                                PatternRewriter &rewriter) const override {
    if (!match)
      return failure();

    rewriter.replaceOp(...);
    return success();
  }
};
```

这就是以后写复杂 rewrite 的模板。

## `success()` 和 `failure()` 的语义

在 `matchAndRewrite()` 中：

```cpp
return failure();
```

表示：

```text
这条 pattern 没匹配当前 op，也没有修改 IR。
```

而：

```cpp
return success();
```

表示：

```text
这条 pattern 匹配成功，并且已经完成 rewrite。
```

不要在已经修改 IR 后返回 `failure()`。

也不要在没有匹配时修改 IR。

这是写 pattern 时非常重要的纪律。

## 一个不会匹配的例子

下面这个函数只有一层 transpose：

```toy
def once(x) {
  return transpose(x);
}
```

大致生成：

```mlir
%0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
toy.return %0 : tensor<*xf64>
```

当 pattern 尝试匹配 `%0` 时：

```cpp
Value transposeInput = op.getOperand();
```

得到 `%arg0`。

但 `%arg0` 是 block argument，不是由 `TransposeOp` 定义。

所以：

```cpp
transposeInput.getDefiningOp<TransposeOp>()
```

返回空，pattern 返回 `failure()`。

最终 IR 不会被错误改成：

```mlir
toy.return %arg0 : tensor<*xf64>
```

因为单层 transpose 不是冗余的。

## 一张执行链路图

```text
toyc-ch3 -emit=mlir -opt
  -> load Toy / MLIR input
  -> build ModuleOp
  -> PassManager
  -> addNestedPass<FuncOp>(createCanonicalizerPass())
  -> canonicalizer visits operations in toy.func
  -> sees toy.transpose
  -> asks TransposeOp for canonicalization patterns
  -> runs SimplifyRedundantTranspose
  -> replaceOp
  -> cleanup dead Pure ops
  -> dump optimized MLIR
```

## 常见误区

### 误区 1：canonicalization 等于所有优化

不是。

Canonicalization 主要做局部、规范化、简化类变换。

它不是完整优化器，也不替代后面的 shape inference、inliner、CSE、lowering。

### 误区 2：`replaceOp` 只是删除 op

不是。

`replaceOp` 会先把旧 op 的 results 的 uses 替换成新 values，然后删除旧 op。

如果只删除 operation，不替换 uses，IR 会断。

### 误区 3：pattern 可以随便直接改 IR

不应该。

在 rewrite pattern 中，IR 修改应该通过 `PatternRewriter` 完成。

这样 rewrite driver 才能正确维护状态。

### 误区 4：一层 transpose 也能删

不能。

`transpose(x)` 和 `x` 通常不等价。

只有：

```text
transpose(transpose(x))
```

才可以简化成：

```text
x
```

### 误区 5：只要写了 C++ pattern 就会自动生效

不会。

还需要：

- operation 在 ODS 中设置 `hasCanonicalizer = 1`。
- C++ 中实现 `getCanonicalizationPatterns()`。
- pass pipeline 中运行 canonicalizer。

## 动手观察

### 观察 pattern 定义

阅读：

```text
mlir/examples/toy/Ch3/mlir/ToyCombine.cpp
```

重点标注：

- `OpRewritePattern<TransposeOp>`
- `matchAndRewrite`
- `getDefiningOp<TransposeOp>()`
- `rewriter.replaceOp`
- `getCanonicalizationPatterns`

### 观察 ODS 接入

阅读：

```text
mlir/examples/toy/Ch3/include/toy/Ops.td
```

找到：

```tablegen
def TransposeOp : Toy_Op<"transpose", [Pure]> {
  ...
  let hasCanonicalizer = 1;
}
```

确认：

- `TransposeOp` 是 pure。
- 它声明了 canonicalizer。

### 观察 pass 接入

阅读：

```text
mlir/examples/toy/Ch3/toyc.cpp
```

找到：

```cpp
pm.addNestedPass<mlir::toy::FuncOp>(mlir::createCanonicalizerPass());
```

确认：

- 只有带 `-opt` 时才运行 pass。
- canonicalizer 是跑在 `toy.func` 内部。

### 观察测试输出

阅读：

```text
mlir/test/Examples/Toy/Ch3/transpose_transpose.toy
```

重点看：

```mlir
toy.return [[VAL_0]] : tensor<*xf64>
```

确认：

- 双重 transpose 已经被消除。
- 返回的是原始函数参数。

## 本节练习

### 练习 1：解释 C++ pattern 结构

阅读：

```cpp
struct SimplifyRedundantTranspose
    : public OpRewritePattern<TransposeOp> {
  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter &rewriter) const override;
};
```

回答：

- 这条 pattern 匹配哪种 operation。
- `op` 参数代表什么。
- `rewriter` 参数负责什么。
- 为什么返回 `LogicalResult`。

### 练习 2：手动追踪双重 transpose

给定：

```mlir
%0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
%1 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
toy.return %1 : tensor<*xf64>
```

回答：

- 当 pattern 匹配外层 transpose 时，`op` 是哪个 operation。
- `op.getOperand()` 得到哪个 value。
- `getDefiningOp<TransposeOp>()` 得到哪个 operation。
- `transposeInputOp.getOperand()` 得到哪个 value。
- `replaceOp` 后 `toy.return` 使用哪个 value。

### 练习 3：分析不匹配场景

给定：

```mlir
%0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
toy.return %0 : tensor<*xf64>
```

回答：

- 为什么这不是冗余 transpose。
- `getDefiningOp<TransposeOp>()` 为什么返回空。
- pattern 为什么必须返回 `failure()`。

### 练习 4：解释 `hasCanonicalizer`

阅读：

```tablegen
let hasCanonicalizer = 1;
```

回答：

- 它让 TableGen 生成什么能力。
- 它和 `getCanonicalizationPatterns()` 有什么关系。
- 如果忘了写它，会发生什么。

### 练习 5：解释 pass pipeline

阅读：

```cpp
if (enableOpt) {
  PassManager pm(...);
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.run(*module);
}
```

回答：

- `-opt` 控制了什么。
- 为什么 canonicalizer 被加成 nested pass。
- 如果不加 `-opt`，`transpose_transpose.toy` 的输出有什么不同。

### 练习 6：比较 C++ pattern 和 DRR

本节先不展开 DRR 语法，只回答概念：

- `SimplifyRedundantTranspose` 是 C++ pattern 还是 DRR pattern。
- `ReshapeOp` 的几个 pattern 从哪里 include 进来。
- 为什么下一课要单独讲 `ToyCombine.td`。

### 练习 7：写一份 pattern 模板

根据本节内容，写一份你自己的 C++ rewrite pattern 模板，包含：

```cpp
struct MyPattern : OpRewritePattern<MyOp> {
  MyPattern(MLIRContext *context)
      : OpRewritePattern<MyOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(MyOp op,
                                PatternRewriter &rewriter) const override {
    ...
  }
};
```

并标注：

- 哪里写匹配条件。
- 哪里写 IR 替换。
- 哪里返回 `failure()`。
- 哪里返回 `success()`。

## 本节小结

本节最重要的是掌握这条链路：

```text
ODS:
  let hasCanonicalizer = 1

C++:
  OpRewritePattern<TransposeOp>
  matchAndRewrite()
  rewriter.replaceOp()
  TransposeOp::getCanonicalizationPatterns()

Pass:
  createCanonicalizerPass()
```

也要记住：

- Pattern rewrite 是局部 IR 改写。
- Canonicalization 是把 IR 变成更简单、更规范的形态。
- `transpose(transpose(x)) -> x` 通过 C++ pattern 实现。
- `replaceOp` 替换的是 old op results 的所有 uses。
- `Pure` 让无用 operation 可以被安全清理。
- 写了 pattern 以后，还必须注册并运行 canonicalizer pass。

下一课会继续看 Ch3 的 `ToyCombine.td`，学习 DRR 声明式重写，并对比它和 C++ pattern 的适用场景。

## 学习记录模板

```text
本节主题：C++ Pattern Rewrite 与 Canonicalization
我读过的源码：
我观察过的测试：
我能解释的 pattern：
我能解释的 canonicalizer 接入点：
我手动追踪过的 IR 改写：
我还不理解的问题：
下一步要验证的小实验：
```
