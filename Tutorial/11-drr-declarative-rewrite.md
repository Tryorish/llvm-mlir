# 第 11 课：DRR 声明式重写

## 本节定位

第 10 课已经讲过 C++ pattern rewrite。那种写法灵活，但当规则本身很简单、结构很固定时，手写 C++ 反而显得冗长。

本节介绍另一条路：DRR（Declarative Rewrite Rules，声明式重写规则）。

在 Ch3 里，Toy 的 canonicalization 不只有 C++ pattern：

- `transpose(transpose(x)) -> x` 用 C++ pattern 写。
- `reshape(reshape(x)) -> reshape(x)`、`reshape(constant)` 这类规则用 DRR 写。

这节课的目标是让你能读懂 `ToyCombine.td`，理解这些 rewrite 规则如何通过 TableGen 生成 `ToyCombine.inc`，再由 `ToyCombine.cpp` 引入并注册到 canonicalization framework。

## 本节目标

- 理解 DRR 的基本结构。
- 能读懂 `Pat`、`Constraint`、`NativeCodeCall`。
- 理解 source pattern 和 result pattern 的对应关系。
- 理解 `replaceWithValue` 这种 DRR result pattern 的语义。
- 理解 DRR 生成的 `ToyCombine.inc` 如何被 C++ include 进来。
- 能解释 `ReshapeReshapeOptPattern`、`FoldConstantReshapeOptPattern`、`RedundantReshapeOptPattern` 三条规则。
- 能比较 DRR 和 C++ pattern 的适用场景。

## 本节关键文件

源码：

```text
mlir/examples/toy/Ch3/mlir/ToyCombine.td
mlir/examples/toy/Ch3/mlir/ToyCombine.cpp
mlir/examples/toy/Ch3/CMakeLists.txt
```

测试：

```text
mlir/test/Examples/Toy/Ch3/trivial_reshape.toy
```

建议阅读顺序：

1. 先看 `ToyCombine.td`，理解 DRR 规则本体。
2. 再看 `CMakeLists.txt`，理解 `ToyCombine.inc` 怎么生成。
3. 再看 `ToyCombine.cpp`，理解这些规则怎么挂到 `ReshapeOp` 上。
4. 最后看 `trivial_reshape.toy`，观察规则是否生效。

## DRR 是什么

DRR 的核心思想是：

```text
把“匹配什么 IR 形态，替换成什么 IR 形态”写成声明式描述，
由 TableGen 生成实际的 rewrite code。
```

和手写 C++ pattern 相比，DRR 更像一份“规则说明书”：

- 你描述 source pattern。
- 你描述 result pattern。
- 你可以加约束。
- 你可以在必要时嵌入少量 native C++。

这适合：

- 形态清晰的局部 rewrite。
- 规则数量多但结构重复。
- 需要和生成器、表驱动基础设施配合的场景。

## `ToyCombine.td` 的作用

`ToyCombine.td` 里写的是：

```tablegen
include "mlir/IR/PatternBase.td"
include "toy/Ops.td"
```

然后定义一系列 DRR pattern。

它不是最终被编译进可执行文件的 C++ 源码，而是 TableGen 输入。

CMake 里会把它变成 `ToyCombine.inc`。

## 生成链路

在 `mlir/examples/toy/Ch3/CMakeLists.txt` 里：

```cmake
set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCh3CombineIncGen)
```

这说明：

```text
ToyCombine.td
  -> mlir_tablegen(... -gen-rewriters)
  -> ToyCombine.inc
  -> ToyCombine.cpp include "ToyCombine.inc"
```

也就是说，DRR 的“实现”实际上是 TableGen 生成的 C++ 代码。

`ToyCombine.cpp` 里：

```cpp
namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "ToyCombine.inc"
} // namespace
```

这一步把生成的 rewrite pattern 编译进最终的 `toyc-ch3`。

## DRR 的基本结构

DRR 最重要的概念是 `Pat`：

```tablegen
def SomePattern : Pat<sourcePattern, resultPattern, constraints...>;
```

它表示：

```text
如果 sourcePattern 匹配成功，
并且 constraints 都成立，
就把它替换成 resultPattern。
```

可以把它理解成：

```text
左边是“看到什么”
右边是“变成什么”
中间可加条件
```

## `ReshapeReshapeOptPattern`

第一条最简单：

```tablegen
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)),
                                   (ReshapeOp $arg)>;
```

它表示：

```text
reshape(reshape(x)) -> reshape(x)
```

注意这里结果不是直接变成 `x`，而是保留一层 `reshape`。

这是一个很典型的“去掉多余中间层”的规范化规则：

- 外层 reshape 冗余。
- 但内层 reshape 仍然保留。

## source pattern 怎么读

```tablegen
(ReshapeOp(ReshapeOp $arg))
```

可以拆成：

- 外层 operation 是 `ReshapeOp`。
- 它的 operand 是一个 `ReshapeOp`。
- 里层 `ReshapeOp` 的 operand 绑定到 `$arg`。

也就是说，这条规则只在看到“reshape 的输入本身也是 reshape”时才匹配。

## result pattern 怎么读

```tablegen
(ReshapeOp $arg)
```

表示结果只保留一层 reshape，并且它的 operand 是 `$arg`。

换成更直观的等式就是：

```text
reshape(reshape(x)) -> reshape(x)
```

## `NativeCodeCall` 是什么

有些 rewrite 不能只靠纯结构描述完成，需要一点 C++ 辅助。

DRR 提供 `NativeCodeCall`：

```tablegen
def ReshapeConstant :
  NativeCodeCall<"$0.reshape(::llvm::cast<ShapedType>($1.getType()))">;
```

它表示：在生成 result pattern 时，调用一段 C++ 代码来构造新值。

这适合：

- 需要构造复杂 type。
- 需要调用 helper method。
- 需要从现有 value 派生新值。

## `FoldConstantReshapeOptPattern`

第二条规则是：

```tablegen
def FoldConstantReshapeOptPattern : Pat<
  (ReshapeOp:$res (ConstantOp $arg)),
  (ConstantOp (ReshapeConstant $arg, $res))>;
```

这条规则表示：

```text
reshape(constant(x)) -> constant(reshape(x))
```

更准确地说：

- source pattern 是一个 `ReshapeOp`。
- 它的输入是 `ConstantOp`。
- result pattern 是一个新的 `ConstantOp`。
- 新的 constant 通过 `ReshapeConstant` 这个 native call 来构造。

### 为什么这条规则有用

它把“reshape 的代价”尽量前移到常量构造阶段，便于后续进一步消除冗余 reshape。

在 `trivial_reshape.toy` 这类例子里，常量和连续 reshape 会被逐步整理成更规整的形式。

## `RedundantReshapeOptPattern`

第三条规则是：

```tablegen
def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
def RedundantReshapeOptPattern : Pat<
  (ReshapeOp:$res $arg), (replaceWithValue $arg),
  [(TypesAreIdentical $res, $arg)]>;
```

这条规则表示：

```text
如果 reshape 的结果类型和输入类型相同，
那这个 reshape 是冗余的，
可以直接用输入 value 替换它。
```

也就是：

```text
reshape(x) -> x
```

前提是：

```text
type(result) == type(input)
```

## `Constraint` 是什么

`Constraint<CPred<"...">>` 用来给 DRR 添加额外条件。

这里的：

```tablegen
def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
```

表示：

- `$0` 和 `$1` 的类型必须相同。
- 如果不满足，pattern 不匹配。

这让 DRR 可以表达一些简单的语义条件，而不只是纯结构匹配。

## `replaceWithValue`

`RedundantReshapeOptPattern` 的 result pattern 是：

```tablegen
(replaceWithValue $arg)
```

这不是构造一个新 operation，而是直接告诉 rewrite framework：

```text
用已有 value `$arg` 替换当前 result。
```

这是 DRR 里很常用的结果模式。

它特别适合：

- “这个 op 完全多余，直接删掉。”
- “当前 operation 的结果和某个输入完全等价。”

## `ReshapeOp` 的 canonicalization 接入

`ToyCombine.cpp` 里：

```cpp
void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<ReshapeReshapeOptPattern, RedundantReshapeOptPattern,
              FoldConstantReshapeOptPattern>(context);
}
```

这说明 `ReshapeOp` 不是只有一条优化规则，而是把三条 DRR pattern 一起注册进去。

所以 canonicalizer 看到 `toy.reshape` 时，会尝试这些规则：

1. 去掉嵌套 reshape。
2. 消除类型相同的 reshape。
3. 把 reshape 常量的逻辑折叠到常量构造里。

## `hasCanonicalizer = 1`

在 `Ch3` 的 `Ops.td` 中，`ReshapeOp` 和 `TransposeOp` 都写了：

```tablegen
let hasCanonicalizer = 1;
```

这表示它们允许通过 `getCanonicalizationPatterns()` 向 canonicalization framework 注册 rewrite patterns。

在 `ReshapeOp` 上，这一点尤其重要，因为它的几个 DRR 规则都在 `ToyCombine.td` 中定义。

## `trivial_reshape.toy`

测试文件：

```text
mlir/test/Examples/Toy/Ch3/trivial_reshape.toy
```

Toy 源码：

```toy
def main() {
  var a<2,1> = [1, 2];
  var b<2,1> = a;
  var c<2,1> = b;
  print(c);
}
```

这里有一个很适合观察 DRR 的点：

```text
var b<2,1> = a;
var c<2,1> = b;
```

这些声明会生成一串看似多余的 reshape。

经过 canonicalization 后，测试期望里只剩下一个常量和一个 print：

```mlir
[[VAL_0:%.*]] = toy.constant ...
toy.print [[VAL_0]] : tensor<2x1xf64>
toy.return
```

说明冗余 reshape 已经被消除了。

## DRR 适合什么场景

DRR 很适合：

- 形态简单的重写。
- 规则固定，局部结构明显。
- 需要大量类似 rewrite 时。
- 希望逻辑保持声明式、可读性高时。

例如：

- `reshape(reshape(x)) -> reshape(x)`
- `reshape(x) -> x`，前提是类型一致
- `reshape(constant) -> constant`

这些规则写成 DRR 会很清晰。

## C++ pattern 适合什么场景

C++ pattern 更适合：

- 条件复杂。
- 需要深入看 use-def 链。
- 需要多步构造 operation。
- 需要复杂类型推导。
- 需要不容易用纯 DRR 表达的逻辑。

第 10 课里的 `transpose(transpose(x)) -> x` 就是典型 C++ pattern。

## DRR 和 C++ pattern 的边界

可以粗略这样分：

```text
DRR
  -> 规则固定、形态清晰、声明式表达更强

C++
  -> 逻辑复杂、条件多、构造过程复杂
```

Ch3 的 Toy 正好把这两类方式都展示出来了：

- `TransposeOp` 用 C++ pattern。
- `ReshapeOp` 用 DRR patterns。

这很适合学习，因为你能直接比较两者的写法和适用场景。

## `ToyCombine.inc` 是什么

`ToyCombine.inc` 不是手写文件，而是 TableGen 生成文件。

它里面包含了 DRR 生成的 rewrite 代码。

`ToyCombine.cpp` 通过：

```cpp
#include "ToyCombine.inc"
```

把这些生成规则编译进 `toyc-ch3`。

所以你可以把整个链路理解成：

```text
ToyCombine.td
  -> TableGen
  -> ToyCombine.inc
  -> ToyCombine.cpp include
  -> toyc-ch3
```

## 与 canonicalizer 的关系

DRR 规则本身不会自动执行。

它们只是被注册成 canonicalization patterns。

真正触发它们的是：

```cpp
createCanonicalizerPass()
```

所以第 10 课和第 11 课是连着的：

- 第 10 课告诉你 canonicalizer 如何跑 C++ pattern。
- 第 11 课告诉你 canonicalizer 如何跑 DRR patterns。

## 常见误区

### 误区 1：DRR 就是不需要 C++

不对。

DRR 只是把简单规则声明化。遇到复杂逻辑，还是需要 C++ pattern 或 native code call。

### 误区 2：`replaceWithValue` 会自动知道替换谁

不是。

它是 result pattern 的一种形式，表示“当前结果用某个已有 value 替换”。

### 误区 3：`NativeCodeCall` 是随便塞代码

也不是。

它主要是补足 DRR 表达能力，应该尽量小而清晰。

### 误区 4：`Constraint` 是可选装饰

它很关键。

没有约束，很多规则会过度匹配，导致错误 rewrite。

### 误区 5：TableGen 规则写了就立刻生效

不会。

还必须：

- 生成 `ToyCombine.inc`。
- 在 `ToyCombine.cpp` 里 include 它。
- 把 pattern 注册到 `getCanonicalizationPatterns()`。
- 运行 canonicalizer pass。

## 动手观察

### 观察 `ReshapeReshapeOptPattern`

阅读：

```tablegen
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)),
                                   (ReshapeOp $arg)>;
```

回答：

- source pattern 里有几层 `ReshapeOp`。
- result pattern 保留了几层 `ReshapeOp`。
- 为什么这不是完全删除 reshape。

### 观察 `RedundantReshapeOptPattern`

阅读：

```tablegen
def RedundantReshapeOptPattern : Pat<
  (ReshapeOp:$res $arg), (replaceWithValue $arg),
  [(TypesAreIdentical $res, $arg)]>;
```

回答：

- `$res` 表示什么。
- `$arg` 表示什么。
- `TypesAreIdentical` 检查什么。
- 为什么类型相同时可以直接替换 value。

### 观察 `FoldConstantReshapeOptPattern`

阅读：

```tablegen
def FoldConstantReshapeOptPattern : Pat<
  (ReshapeOp:$res (ConstantOp $arg)),
  (ConstantOp (ReshapeConstant $arg, $res))>;
```

回答：

- 它匹配哪种嵌套形态。
- 为什么要用 `NativeCodeCall`。
- 结果为什么还是 `ConstantOp`。

### 观察 CMake 生成链路

阅读：

```cmake
set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
```

回答：

- 哪个文件是输入。
- 哪个文件是输出。
- 哪个文件最终 include 生成结果。

### 观察 `trivial_reshape.toy`

阅读：

```text
mlir/test/Examples/Toy/Ch3/trivial_reshape.toy
```

回答：

- 为什么两个 `var <2,1>` 连续赋值会产生可消除的 reshape。
- canonicalizer 后为什么输出里只剩常量和 print。

## 本节练习

### 练习 1：解释 `Pat`

阅读：

```tablegen
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)),
                                   (ReshapeOp $arg)>;
```

回答：

- source pattern 是什么。
- result pattern 是什么。
- 这条规则在语言上意味着什么。

### 练习 2：解释 `Constraint`

阅读：

```tablegen
def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
```

回答：

- 它在比较什么。
- 为什么这条约束能防止错误 rewrite。
- 没有它会有什么风险。

### 练习 3：解释 `replaceWithValue`

阅读：

```tablegen
(replaceWithValue $arg)
```

回答：

- 它和“构造一个新 operation”有什么区别。
- 它通常适合哪类优化。

### 练习 4：比较 DRR 和 C++

把下面两类规则分组：

- `transpose(transpose(x)) -> x`
- `reshape(reshape(x)) -> reshape(x)`
- `reshape(x) -> x`，前提是类型相同
- 需要读取多个 value 和复杂条件判断的 rewrite

回答：

- 哪些更适合 DRR。
- 哪些更适合 C++ pattern。

### 练习 5：解释生成链路

把下面链路写成自己的话：

```text
ToyCombine.td -> mlir_tablegen -gen-rewriters -> ToyCombine.inc -> ToyCombine.cpp -> canonicalizer
```

至少说清楚：

- 哪一步负责生成。
- 哪一步负责 include。
- 哪一步负责运行。

### 练习 6：分析 `trivial_reshape.toy`

阅读测试文件和期望输出。

回答：

- 原始源码里为什么会有多余 reshape。
- canonicalizer 具体消掉了什么。
- 最终为什么还保留 `toy.constant` 和 `toy.print`。

### 练习 7：写一条你自己的 DRR

仿照本节内容，尝试写一条你自己的伪规则：

```text
opA(opA(x)) -> opA(x)
opB(constant(x)) -> constant(...)
```

并说明：

- source pattern。
- result pattern。
- 需要的约束。
- 是否需要 native code call。

## 本节小结

本节最重要的是把 DRR 看成一种“声明式 rewrite 语言”。

核心链路是：

```text
ToyCombine.td
  -> Pat / Constraint / NativeCodeCall
  -> ToyCombine.inc
  -> ToyCombine.cpp register patterns
  -> canonicalizer pass apply rewrite
```

也要记住：

- `Pat` 描述 source 和 result pattern。
- `Constraint` 控制何时匹配。
- `NativeCodeCall` 补足复杂构造能力。
- `replaceWithValue` 表示直接复用已有 value。
- DRR 适合结构清晰、规则稳定的 rewrite。
- C++ pattern 适合复杂逻辑和多步构造。

下一课会进入 Pass Manager、Pass Pipeline 与调试，看看这些 pattern 是如何被串到整个 Toy 编译流程里的。

## 学习记录模板

```text
本节主题：DRR 声明式重写
我读过的源码：
我观察过的测试：
我能解释的 Pat：
我能解释的 Constraint：
我能解释的 NativeCodeCall：
我能解释的生成链路：
我还不理解的问题：
下一步要验证的小实验：
```
