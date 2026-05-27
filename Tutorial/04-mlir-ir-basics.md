# 第 4 课：MLIR 核心概念入门

## 本节定位

前三节课都在 Ch1：Toy 源码经过 Lexer 和 Parser，最终变成 Toy 自己的 AST。

从本节开始进入 Ch2。Ch2 的关键变化是：Toy 不再只停留在 AST，而是开始生成 MLIR。

本节先不深入 `MLIRGen.cpp` 的实现细节，也不急着学习 ODS/TableGen。目标是先学会读 MLIR IR，尤其是读懂 Ch2 中 `toyc-ch2 -emit=mlir` 的输出。

本节主线是：

```text
Toy AST
  -> MLIRGen
  -> ModuleOp
  -> Toy Dialect operations
  -> SSA Value / Type / Attribute / Region / Block
```

学完本节后，你应该能拿到一段 Toy MLIR，指出里面的 operation、operand、result、type、attribute、region、block argument 和 dialect。

## 本节目标

- 理解 MLIR 为什么存在，以及它和 AST、LLVM IR 的区别。
- 理解 MLIR 的核心结构：`Operation -> Region -> Block -> Operation`。
- 理解 `Value`、operand、result、block argument、def-use。
- 理解 `Type` 和 `Attribute` 的基本作用。
- 理解 `Dialect` 是 operation/type/attribute 的命名空间。
- 能读懂 Ch2 的 Toy MLIR 输出。
- 能把一条 Toy MLIR operation 拆成：结果、名字、操作数、属性、类型、位置。

## 本节关键文件

文档：

```text
mlir/docs/Tutorials/UnderstandingTheIRStructure.md
mlir/docs/Tutorials/Toy/Ch-2.md
```

测试：

```text
mlir/test/Examples/Toy/Ch2/codegen.toy
mlir/test/Examples/Toy/Ch2/scalar.toy
mlir/test/Examples/Toy/Ch2/invalid.mlir
```

源码，先定位即可：

```text
mlir/examples/toy/Ch2/toyc.cpp
mlir/examples/toy/Ch2/mlir/MLIRGen.cpp
mlir/examples/toy/Ch2/include/toy/Ops.td
mlir/examples/toy/Ch2/mlir/Dialect.cpp
```

建议阅读顺序：

1. 先读 `mlir/test/Examples/Toy/Ch2/codegen.toy`。
2. 运行或阅读它的 `-emit=mlir` 输出。
3. 对照本节解释标注每个 MLIR 概念。
4. 最后粗略看 `Ops.td`，知道 Toy operations 是被声明出来的。

## 从 AST 到 MLIR 的变化

Ch1 输出的是 AST dump：

```text
Module:
  Function
    Proto 'main'
    Params: []
    Block {
      VarDecl a<>
        Literal: <2, 3>[ ... ]
      Print [
        var: a
      ]
    } // Block
```

Ch2 输出的是 MLIR：

```mlir
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00],
                           [4.000000e+00, 5.000000e+00, 6.000000e+00]]>
       : tensor<2x3xf64>
  toy.print %0 : tensor<2x3xf64>
  toy.return
}
```

这两个表示的差异很重要：

- AST 更接近源码语法结构。
- MLIR 是编译器中间表示，适合分析、变换、优化和 lowering。
- AST 节点是 Toy 前端自己的 C++ 类。
- MLIR 节点是统一的 operation 体系。
- AST 中的 `VarDeclExprAST` 不一定在 MLIR 中保留为一个变量声明 operation；变量通常变成 SSA value 的绑定关系。

## MLIR 的核心嵌套结构

MLIR IR 的基本嵌套结构是：

```text
Operation
  Region
    Block
      Operation
        Region
          Block
            Operation
```

也就是说，MLIR 是递归嵌套的。

最常见的顶层 operation 是：

```text
ModuleOp
```

一个 module 里面有 region，region 里面有 block，block 里面有 operations。

对于 Toy 来说，可以把它理解成：

```text
ModuleOp
  region
    block
      toy.func @multiply_transpose
        region
          block(arguments: a, b)
            toy.transpose
            toy.transpose
            toy.mul
            toy.return
      toy.func @main
        region
          block
            toy.constant
            toy.reshape
            toy.generic_call
            toy.print
            toy.return
```

这个结构和 AST 有点像，但更统一。AST 中有很多不同 C++ 节点类；MLIR 中所有东西最终都是 operation、region、block、value、type、attribute 的组合。

## Operation：MLIR 的基本单位

MLIR 中最重要的概念是 operation。

一条 operation 可以有：

- 名字。
- 输入 operands。
- 输出 results。
- attributes。
- result types。
- source location。
- nested regions。
- successor blocks。

官方 Toy 文档给过一个通用形式：

```mlir
%t_tensor = "toy.transpose"(%tensor) {inplace = true}
  : (tensor<2x3xf64>) -> tensor<3x2xf64>
  loc("example/file/path":12:1)
```

拆开看：

| 片段 | 含义 |
| --- | --- |
| `%t_tensor` | operation 产生的 result，也就是一个 SSA value |
| `"toy.transpose"` | operation 名字，属于 `toy` dialect |
| `(%tensor)` | 输入 operands |
| `{inplace = true}` | attributes，编译期常量元信息 |
| `(tensor<2x3xf64>) -> tensor<3x2xf64>` | operand types 到 result types |
| `loc(...)` | source location |

Ch2 的 Toy IR 多数使用 custom assembly format，所以看起来更友好：

```mlir
%0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
```

但它本质上仍然是一条 operation。

## Op 和 Operation 的区别

MLIR C++ 里经常看到两个层次：

- `mlir::Operation`
- 具体 op wrapper，例如 `mlir::toy::ConstantOp`、`mlir::toy::FuncOp`

可以先这样理解：

- `Operation` 是通用底层表示，任何 MLIR operation 都可以用它表示。
- `ConstantOp`、`FuncOp` 这类具体类型是强类型 wrapper，提供更方便、更安全的 API。

比如在 Toy 中：

```mlir
%0 = toy.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
```

底层它是一个 `Operation`，名字是 `toy.constant`。

在 C++ 中，它也可以被当作 `toy::ConstantOp` 使用，这样就能直接访问 `getValue()` 等专门接口。

这一节只需要知道这个区别。后面讲 ODS 和 TableGen 时，会看到这些 wrapper 是如何生成的。

## Dialect：命名空间和语义边界

Dialect 是 MLIR 的扩展机制。它为一组 operation、type、attribute 提供命名空间和语义边界。

Toy Dialect 的 operation 都以 `toy.` 开头：

```mlir
toy.func
toy.constant
toy.reshape
toy.transpose
toy.mul
toy.add
toy.generic_call
toy.print
toy.return
```

这里的 `toy` 就是 dialect namespace。

后续 lowering 后，会看到其他 dialect：

```mlir
affine.for
memref.alloc
arith.constant
func.func
llvm.call
```

不同 dialect 代表不同抽象层级：

- `toy`：Toy 语言高层语义。
- `affine`：仿射循环和访存。
- `memref`：内存引用。
- `arith`：标量算术。
- `func`：通用函数。
- `llvm`：接近 LLVM IR 的低层表示。

Ch2 主要关注 `toy` dialect。

## SSA Value：结果和操作数

MLIR 使用 SSA，Static Single Assignment。

SSA 的基本规则是：一个 value 只被定义一次，但可以被使用多次。

例如：

```mlir
%0 = toy.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
%1 = toy.reshape(%0 : tensor<2xf64>) to tensor<1x2xf64>
toy.print %1 : tensor<1x2xf64>
```

这里：

- `%0` 是 `toy.constant` 的 result。
- `%0` 又作为 `toy.reshape` 的 operand。
- `%1` 是 `toy.reshape` 的 result。
- `%1` 又作为 `toy.print` 的 operand。

从 def-use 角度看：

```text
toy.constant defines %0
toy.reshape uses %0 and defines %1
toy.print uses %1
```

这就是 MLIR 分析和变换的基础。

## Result、Operand、Block Argument

MLIR 中的 `Value` 有两种来源：

1. 某个 operation 的 result。
2. 某个 block 的 argument。

例如 Ch2 的函数：

```mlir
toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
  %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
  %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
  %2 = toy.mul %0, %1 : tensor<*xf64>
  toy.return %2 : tensor<*xf64>
}
```

这里：

- `%arg0`、`%arg1` 是 block arguments，也就是函数入口 block 的参数。
- `%0`、`%1`、`%2` 是 operation results。
- `toy.transpose` 使用 `%arg0` 或 `%arg1` 作为 operand。
- `toy.mul` 使用 `%0` 和 `%1` 作为 operands。
- `toy.return` 使用 `%2` 作为 operand。

所以 `Value` 不是“变量”。它可以来自函数参数，也可以来自 operation 结果。

## Type：值的类型

MLIR 中每个 value 都有 type。

Toy 中常见类型：

```mlir
tensor<*xf64>
tensor<2x3xf64>
tensor<6xf64>
tensor<3x2xf64>
```

含义：

| 类型 | 含义 |
| --- | --- |
| `tensor<*xf64>` | unknown ranked/unranked tensor，shape 未知，元素是 f64 |
| `tensor<2x3xf64>` | 2x3 tensor，元素是 f64 |
| `tensor<6xf64>` | 一维 6 元素 tensor，元素是 f64 |
| `tensor<3x2xf64>` | 3x2 tensor，元素是 f64 |

Toy 函数参数通常先是：

```mlir
tensor<*xf64>
```

因为 Toy 函数是 generic 的，参数 shape 要在后续 shape inference 和 specialization 中逐步确定。

## Attribute：编译期常量元信息

Attribute 是附着在 operation 上的编译期常量信息。

最典型例子是 `toy.constant`：

```mlir
%0 = toy.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
```

这里的：

```mlir
dense<[1.000000e+00, 2.000000e+00]>
```

就是常量数据。它在 `Ops.td` 中对应 `value` attribute：

```tablegen
let arguments = (ins F64ElementsAttr:$value);
```

注意：attribute 不是运行时输入 operand。它是 operation 自带的常量元信息。

可以这样区分：

- operand：来自其他 operation 或 block argument 的 SSA value。
- attribute：编译期已知的常量数据。

例如：

```mlir
%0 = toy.constant dense<[1.0, 2.0]> : tensor<2xf64>
%1 = toy.reshape(%0 : tensor<2xf64>) to tensor<1x2xf64>
```

`toy.constant` 的常量数组是 attribute。

`toy.reshape` 的输入 `%0` 是 operand。

## Region 和 Block

Operation 可以包含 region。Region 包含 block。Block 包含 operation。

最容易看到 region/block 的 Toy operation 是 `toy.func`：

```mlir
toy.func @main() {
  %0 = toy.constant dense<[1.000000e+00]> : tensor<1xf64>
  toy.print %0 : tensor<1xf64>
  toy.return
}
```

`toy.func @main()` 是一个 operation。

它的 `{ ... }` 是函数体 region 的文本形式。

这个 region 里有一个 entry block，block 里有：

```text
toy.constant
toy.print
toy.return
```

对于有参数的函数：

```mlir
toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
  ...
}
```

`%arg0` 和 `%arg1` 是 entry block 的 block arguments。

这一点很重要：函数参数在 MLIR 中不是普通 operation 结果，而是 block argument。

## ModuleOp

MLIR 顶层通常是 `ModuleOp`。

在 Toy 中，一个 `.toy` 文件对应一个 MLIR module。

`MLIRGen.cpp` 中会创建：

```cpp
theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
```

然后把每个 `toy.func` 插入 module body 中。

在打印时，顶层 module 有时会显示成：

```mlir
module {
  toy.func @main() {
    ...
  }
}
```

测试中的 `CHECK-LABEL` 通常直接匹配内部的 `toy.func`，因为这才是 Toy 教程关注的主体。

## Symbol 和 Symbol Reference

函数名在 MLIR 中通常是 symbol。

例如：

```mlir
toy.func @multiply_transpose(...)
```

这里的：

```text
@multiply_transpose
```

是函数 symbol。

调用这个函数时：

```mlir
%0 = toy.generic_call @multiply_transpose(%a, %b)
     : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
```

`@multiply_transpose` 是 callee symbol reference。

在 `Ops.td` 中，`toy.generic_call` 的 callee 是：

```tablegen
FlatSymbolRefAttr:$callee
```

这说明函数调用目标不是一个 SSA operand，而是一个 symbol attribute。

先记住这个区别：

- `%0`、`%arg0`：SSA value。
- `@main`、`@multiply_transpose`：symbol。

## Location

MLIR 中每个 operation 都有 location。

在 Toy Ch2 中，`MLIRGen.cpp` 会把 Toy AST 的 `Location` 转成 MLIR location：

```cpp
mlir::FileLineColLoc::get(...)
```

默认打印 MLIR 时，location 可能不显示。开启 debug info 打印时可以看到类似：

```mlir
loc("file.toy":11:3)
```

location 的意义是：当后续 verifier、pass 或 lowering 报错时，MLIR 能知道这个 operation 来自源码哪里。

## 读懂 Ch2 的 `codegen.toy`

测试文件：

```text
mlir/test/Examples/Toy/Ch2/codegen.toy
```

输入 Toy：

```toy
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  var c = multiply_transpose(a, b);
  var d = multiply_transpose(b, a);
  print(d);
}
```

测试命令：

```text
# RUN: toyc-ch2 %s -emit=mlir 2>&1 | FileCheck %s
```

这说明它验证的是：Toy 源码能被 Ch2 编译器转换成 Toy MLIR。

## 例 1：`toy.func`

测试中期望：

```mlir
toy.func @multiply_transpose(
  %arg0: tensor<*xf64>, %arg1: tensor<*xf64>
) -> tensor<*xf64> {
  ...
}
```

拆解：

- `toy.func`：Toy dialect 中的函数 operation。
- `@multiply_transpose`：函数 symbol。
- `%arg0`、`%arg1`：entry block arguments。
- `tensor<*xf64>`：参数 type，shape 未知。
- `-> tensor<*xf64>`：函数返回 type，shape 未知。
- `{ ... }`：函数体 region。

为什么参数是 `tensor<*xf64>`？

因为 Toy 函数是 generic 的。源码里 `def multiply_transpose(a, b)` 没有写 shape，Ch2 先用 unknown shape tensor 表示它。

## 例 2：`toy.transpose`

```mlir
%0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
```

拆解：

- `%0`：operation result。
- `toy.transpose`：operation name。
- `%arg0`：operand。
- `tensor<*xf64>`：输入 type。
- `to tensor<*xf64>`：结果 type。

这里没有具体 shape，因为输入 `%arg0` 的 shape 还未知。

## 例 3：`toy.mul`

```mlir
%2 = toy.mul %0, %1 : tensor<*xf64>
```

拆解：

- `%2`：result。
- `toy.mul`：operation name。
- `%0`、`%1`：operands。
- `tensor<*xf64>`：操作数和结果 type。

这是 element-wise multiplication。Ch2 还没有做 shape inference，所以结果仍然是 unknown shape tensor。

## 例 4：`toy.constant`

```mlir
%0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00],
                         [4.000000e+00, 5.000000e+00, 6.000000e+00]]>
     : tensor<2x3xf64>
```

拆解：

- `%0`：result。
- `toy.constant`：operation name。
- `dense<...>`：attribute，保存常量 tensor 数据。
- `tensor<2x3xf64>`：result type。
- 没有 operand。

这条 operation 把 Toy literal 变成一个 SSA value。

## 例 5：`toy.reshape`

```mlir
%1 = toy.reshape(%0 : tensor<6xf64>) to tensor<2x3xf64>
```

拆解：

- `%1`：result。
- `toy.reshape`：operation name。
- `%0`：operand。
- `tensor<6xf64>`：输入 type。
- `tensor<2x3xf64>`：结果 type。

对应 Toy 源码：

```toy
var b<2, 3> = [1, 2, 3, 4, 5, 6];
```

右侧 literal 自然 shape 是 `<6>`，变量声明要求 `<2, 3>`，所以 MLIR 中会出现 reshape。

## 例 6：`toy.generic_call`

```mlir
%2 = toy.generic_call @multiply_transpose(%a, %b)
     : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
```

拆解：

- `%2`：result。
- `toy.generic_call`：operation name。
- `@multiply_transpose`：callee symbol reference，不是 SSA operand。
- `%a`、`%b`：调用参数 operands。
- `(tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>`：函数式 type。

为什么叫 `generic_call`？

因为 Toy 函数会根据调用点 shape 进行 specialization。Ch2 先用 generic call 表示这种高层语义，后续章节再优化和 lowering。

## 例 7：`toy.print` 和 `toy.return`

```mlir
toy.print %0 : tensor<*xf64>
toy.return %0 : tensor<*xf64>
toy.return
```

`toy.print`：

- 有 operand。
- 没有 result。
- 表示打印 tensor。

`toy.return`：

- 可以有 operand，也可以没有 operand。
- 是 Toy 函数体的 terminator。

注意：并不是每个 operation 都有 result。`toy.print` 和无值 `toy.return` 都不产生 SSA value。

## FileCheck 里的 `[[VAL_0:%.*]]`

Ch2 测试中会看到：

```text
# CHECK: [[VAL_2:%.*]] = toy.transpose([[VAL_0]] : tensor<*xf64>) to tensor<*xf64>
```

这不是 MLIR 本身的语法，而是 FileCheck 的变量写法。

含义是：

- `[[VAL_2:%.*]]`：匹配一个以 `%` 开头的 SSA 名字，并保存起来。
- 后面的 `[[VAL_0]]`：引用之前匹配到的名字。

真实输出里可能是 `%0`、`%1`、`%2`，FileCheck 不依赖具体编号。

阅读测试时要区分：

- `toy.func`、`toy.constant`：MLIR 文本。
- `[[VAL_0:%.*]]`：FileCheck 匹配变量。

## Generic Form 和 Custom Assembly Format

MLIR operation 有通用形式，类似：

```mlir
%0 = "toy.transpose"(%arg0) : (tensor<*xf64>) -> tensor<*xf64>
```

也可以有自定义打印格式：

```mlir
%0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
```

后者更适合人读。

在 Ch2 的 `Ops.td` 和 `Dialect.cpp` 中，很多 Toy operation 都定义了 custom parser/printer，所以输出看起来不像完全通用形式。

这一节不用掌握 custom assembly 的实现。只需要知道：无论文本长什么样，本质上仍然是一条 MLIR operation。

## AST 和 MLIR 的对应关系

| Toy AST | Toy MLIR |
| --- | --- |
| `ModuleAST` | `ModuleOp` |
| `FunctionAST` | `toy.func` |
| `PrototypeAST` | `toy.func` 的 symbol、function type、block arguments |
| `VarDeclExprAST` | 通常不保留为 operation，而是 symbol table 中的 SSA value 绑定 |
| `LiteralExprAST` | `toy.constant` |
| 显式 shape 变量声明 | 可能生成 `toy.reshape` |
| `VariableExprAST` | 查表得到已有 SSA value |
| `BinaryExprAST('+')` | `toy.add` |
| `BinaryExprAST('*')` | `toy.mul` |
| `CallExprAST("transpose")` | `toy.transpose` |
| 普通 `CallExprAST` | `toy.generic_call` |
| `PrintExprAST` | `toy.print` |
| `ReturnExprAST` | `toy.return` |

这个表是下一节课学习 `MLIRGen.cpp` 的基础。

## 常见误区

### 误区 1：MLIR 就是 LLVM IR

不是。

MLIR 是 Multi-Level IR。它可以表示很多抽象层级。Toy Dialect 是很高层的 MLIR，LLVM Dialect 是比较低层的 MLIR，LLVM IR 则是另一个表示。

Ch2 输出的是 Toy Dialect MLIR，不是 LLVM IR。

### 误区 2：`%0` 是 Toy 源码里的变量名

不是。

`%0` 是 MLIR 打印出来的 SSA value 名字。Toy 源码里的变量名 `a`、`b` 通常在 MLIRGen 阶段被映射到 SSA value，不一定保留为同名文本。

### 误区 3：所有 operation 都有 result

不是。

例如：

```mlir
toy.print %0 : tensor<2x3xf64>
toy.return
```

它们都没有 result。

### 误区 4：Attribute 和 operand 是一回事

不是。

operand 是 SSA value，来自其他 operation result 或 block argument。

attribute 是编译期常量元信息，直接附着在 operation 上。

### 误区 5：函数参数是 operation result

不是。

MLIR 函数参数通常是 entry block 的 block arguments。

## 动手运行

如果已经构建好 `toyc-ch2`，运行：

```bash
<build-dir>/bin/toyc-ch2 mlir/test/Examples/Toy/Ch2/codegen.toy -emit=mlir
```

观察：

- 顶层是否有 module。
- 有几个 `toy.func`。
- 每个 `toy.func` 的参数是什么 type。
- 哪些 operation 有 result。
- 哪些 operation 没有 result。
- 哪些 operation 使用了 `%arg0`、`%arg1`。
- 哪些 operation 使用了 `dense<...>` attribute。
- 哪些 operation 使用了 `@multiply_transpose` symbol。

也可以运行：

```bash
<build-dir>/bin/toyc-ch2 mlir/test/Examples/Toy/Ch2/scalar.toy -emit=mlir
```

对比 scalar literal 和 tensor literal 在 MLIR 里的 type。

## 本节练习

### 练习 1：拆解一条 operation

拆解下面这条 operation：

```mlir
%0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
```

写出：

- operation name。
- result。
- operand。
- operand type。
- result type。
- 所属 dialect。

### 练习 2：区分 operand 和 attribute

拆解下面两条 operation：

```mlir
%0 = toy.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
%1 = toy.reshape(%0 : tensor<2xf64>) to tensor<1x2xf64>
```

回答：

- `toy.constant` 有没有 operand。
- `dense<[...]>` 是 operand 还是 attribute。
- `toy.reshape` 的 operand 是什么。
- `%1` 是谁定义的。

### 练习 3：标注 Ch2 `codegen.toy` 输出

运行或阅读：

```bash
<build-dir>/bin/toyc-ch2 mlir/test/Examples/Toy/Ch2/codegen.toy -emit=mlir
```

在输出中标注：

- 所有 `toy.func`。
- 所有 block arguments。
- 所有 `toy.constant`。
- 所有 `toy.reshape`。
- 所有 `toy.generic_call`。
- 所有 `toy.print` 和 `toy.return`。

### 练习 4：画出 def-use 链

针对下面 IR：

```mlir
%0 = toy.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
%1 = toy.reshape(%0 : tensor<2xf64>) to tensor<1x2xf64>
toy.print %1 : tensor<1x2xf64>
```

画出：

```text
谁定义 %0
谁使用 %0
谁定义 %1
谁使用 %1
```

### 练习 5：区分 SSA value 和 symbol

解释下面 IR 中哪些是 SSA value，哪些是 symbol：

```mlir
toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
  %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
  toy.return %0 : tensor<*xf64>
}

toy.func @main() {
  %1 = toy.generic_call @multiply_transpose(%0, %0)
       : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
  toy.return
}
```

至少说明：

- `@multiply_transpose` 是什么。
- `%arg0` 是什么。
- `%0` 是什么。
- `%1` 是什么。

### 练习 6：AST 到 MLIR 对照

对下面 Toy 源码：

```toy
def main() {
  var a<2> = [1, 2];
  print(a);
}
```

写出它大概会生成哪些 Toy MLIR operation。

不用写完全准确的 MLIR，只要列出顺序：

```text
toy.func
toy.constant
可能的 toy.reshape
toy.print
toy.return
```

并解释为什么可能有 `toy.reshape`。

## 本节小结

本节最重要的是掌握 MLIR 的核心对象模型：

```text
Operation
  Region
    Block
      Operation
```

以及这几个术语：

- dialect：operation/type/attribute 的命名空间和语义集合。
- operation：MLIR 的基本计算和结构单位。
- result：operation 定义出来的 SSA value。
- operand：operation 使用的 SSA value。
- block argument：block 的参数，也是 SSA value。
- type：value 的类型。
- attribute：operation 上的编译期常量元信息。
- symbol：函数等可被引用实体的名字，例如 `@main`。

下一节课会进入 `MLIRGen.cpp`，沿着 AST 节点逐个看它们如何被转换成这些 Toy MLIR operation。

## 学习记录模板

```text
本节主题：MLIR 核心概念入门
我读过的源码：
我观察过的测试：
我运行过的命令：
我能拆解的 operation：
我还不理解的问题：
下一步要验证的小实验：
```
