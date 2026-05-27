# 第 2 课：Toy 语言语法与 AST

## 本节定位

第一节课建立了 Toy 教程的整体地图。本节课进入第一种具体表示：AST。

Toy 编译器最开始面对的是 `.toy` 源文件。源文件是人写的文本，编译器不能直接对文本做优化或代码生成，因此第一步要把它变成结构化数据。这个结构化数据就是 AST，Abstract Syntax Tree，抽象语法树。

本节课重点回答三个问题：

1. Toy 语言本身长什么样。
2. Toy 源码中的函数、变量、字面量、调用、返回如何映射到 AST 节点。
3. `mlir/examples/toy/Ch1/include/toy/AST.h` 中的 AST 类为什么这样组织。

本节不会深入讲 Lexer 和 Parser 的递归下降实现。那是第 3 课的重点。本节只需要先理解“Parser 最终构造出来的 AST 是什么形状”。

## 本节目标

- 读懂 Ch1 的 Toy 源程序。
- 理解 Toy 的函数、变量声明、tensor literal、表达式、调用、返回。
- 理解 AST 的顶层结构：`ModuleAST -> FunctionAST -> PrototypeAST + ExprASTList -> ExprAST`。
- 掌握 Ch1 中所有 AST 节点的含义。
- 能把一小段 Toy 源码手动画成 AST。
- 能读懂 `toyc-ch1 -emit=ast` 输出的缩进结构。

## 本节关键文件

源码：

```text
mlir/examples/toy/Ch1/include/toy/AST.h
mlir/examples/toy/Ch1/parser/AST.cpp
mlir/examples/toy/Ch1/toyc.cpp
```

测试：

```text
mlir/test/Examples/Toy/Ch1/ast.toy
mlir/test/Examples/Toy/Ch1/empty.toy
```

官方文档：

```text
mlir/docs/Tutorials/Toy/Ch-1.md
```

建议阅读顺序：

1. 先读 `mlir/test/Examples/Toy/Ch1/ast.toy`。
2. 再读 `mlir/examples/toy/Ch1/include/toy/AST.h`。
3. 然后读 `mlir/examples/toy/Ch1/parser/AST.cpp`，理解 AST dump 如何打印。
4. 最后回到 `ast.toy` 的 `CHECK:`，对照输出结构。

## Toy 语言概览

Toy 是一个很小的张量语言。它支持：

- 函数定义。
- 变量声明。
- 数字字面量。
- 一维和二维 tensor literal。
- `+`、`-`、`*` 二元表达式。
- 函数调用。
- 内建 `print()`。
- 内建 `transpose()`。
- `return`。
- 使用 `<...>` 给变量声明形状。

一个典型 Toy 程序如下：

```toy
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  var a = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  var c = multiply_transpose(a, b);
  print(c);
}
```

这段程序里面包含了 Toy Ch1 的主要语法：

- `def multiply_transpose(a, b) { ... }` 是函数定义。
- `a, b` 是函数参数。
- `return transpose(a) * transpose(b);` 是返回语句，返回一个二元表达式。
- `var a = ...;` 是没有显式 shape 的变量声明。
- `var b<2, 3> = ...;` 是带显式 shape 的变量声明。
- `[[1, 2, 3], [4, 5, 6]]` 是二维 tensor literal。
- `[1, 2, 3, 4, 5, 6]` 是一维 tensor literal。
- `multiply_transpose(a, b)` 是用户函数调用。
- `print(c)` 是内建打印调用。

## Toy 的类型观念

Toy 教程为了简化，只有一种元素类型：64 位浮点数，也就是 C/C++ 中常见的 `double`。

所以在 Toy 源码里不会看到这样的类型：

```toy
var a: i32 = 1;
var b: f32 = 2;
```

Toy 的类型信息主要体现在 tensor shape 上：

```toy
var a<2, 3> = [1, 2, 3, 4, 5, 6];
```

这里 `<2, 3>` 表示变量 `a` 的形状是 2 行 3 列。元素类型仍然默认是 64 位浮点数。

没有显式 shape 时：

```toy
var a = [[1, 2, 3], [4, 5, 6]];
```

shape 可以从 literal 推导出来。这个例子里 literal 是两层嵌套，外层有 2 个元素，内层每个有 3 个数字，所以 shape 是 `<2, 3>`。

注意：Ch1 阶段主要是解析和 AST 构造，不做完整语义检查。比如变量是否声明过、函数是否存在、shape 是否兼容，这些不是 Ch1 的重点。

`Parser.h` 的注释也明确说明：Parser 只产生结构良好的 AST，不做语义检查或符号解析。因此某些“语法上能解析，但语义上不对”的程序，在 Ch1 仍然可能得到 AST。

## Toy 的基本语法

### 函数定义

Toy 函数用 `def` 定义：

```toy
def main() {
  print(1);
}
```

带参数的函数：

```toy
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}
```

函数参数只写名字，不写类型：

```toy
def f(x, y) {
  return x + y;
}
```

原因是 Toy 函数是 generic 的。参数在 Toy MLIR 中通常先表现为未知 shape 的 tensor，后续会通过调用点和 shape inference 逐步具体化。

在 AST 中，函数由两个部分组成：

- `PrototypeAST`：函数名和参数名列表。
- `ExprASTList`：函数体，也就是语句和表达式列表。

### 变量声明

Toy 用 `var` 声明变量：

```toy
var a = 1;
var b = [1, 2, 3];
var c = [[1, 2], [3, 4]];
```

也可以显式写 shape：

```toy
var b<2, 3> = [1, 2, 3, 4, 5, 6];
```

在 AST 中，变量声明对应：

```text
VarDeclExprAST
```

它保存三类信息：

- 变量名。
- 变量 shape，也就是 `VarType`。
- 初始化表达式。

例如：

```toy
var b<2, 3> = [1, 2, 3, 4, 5, 6];
```

大致对应：

```text
VarDeclExprAST
  name = "b"
  type.shape = [2, 3]
  initVal = LiteralExprAST
```

如果没有显式 shape：

```toy
var a = [[1, 2, 3], [4, 5, 6]];
```

AST dump 中会显示：

```text
VarDecl a<>
```

这里的 `<>` 表示声明处没有写显式 shape。它不代表最终 shape 一定为空，只表示这个变量声明的 shape 要靠后续推导或 literal 信息确定。

### 数字字面量

Toy 的数字字面量对应：

```text
NumberExprAST
```

例如：

```toy
1
2.5
```

在 AST 中保存为 `double`。

AST dump 中数字会以浮点格式打印，例如：

```text
1.000000e+00
```

### Tensor Literal

一维 literal：

```toy
[1, 2, 3]
```

二维 literal：

```toy
[[1, 2, 3], [4, 5, 6]]
```

在 AST 中对应：

```text
LiteralExprAST
```

`LiteralExprAST` 保存两类信息：

- `values`：literal 中的元素，可以是数字，也可以是嵌套 literal。
- `dims`：当前 literal 的 shape。

例如：

```toy
[1, 2, 3]
```

shape 是：

```text
<3>
```

例如：

```toy
[[1, 2, 3], [4, 5, 6]]
```

shape 是：

```text
<2, 3>
```

AST dump 中会打印成：

```text
Literal: <2, 3>[ <3>[ 1.000000e+00, 2.000000e+00, 3.000000e+00], <3>[ 4.000000e+00, 5.000000e+00, 6.000000e+00]]
```

这表示：

- 外层 literal shape 是 `<2, 3>`。
- 外层有 2 个元素。
- 每个元素是一个 shape 为 `<3>` 的内层 literal。

### 变量引用

变量引用对应：

```text
VariableExprAST
```

例如：

```toy
a
b
c
```

AST dump 中会显示：

```text
var: a
```

注意：Ch1 的 `VariableExprAST` 只保存变量名，不负责检查变量是否真的声明过。这个检查属于更后面的语义处理。

### 二元表达式

Toy 支持二元表达式：

```toy
a + b
a - b
a * b
```

在 AST 中对应：

```text
BinaryExprAST
```

它保存：

- 操作符，例如 `+`、`-`、`*`。
- 左操作数。
- 右操作数。

例如：

```toy
transpose(a) * transpose(b)
```

AST dump 中会显示：

```text
BinOp: *
  Call 'transpose' [
    var: a
  ]
  Call 'transpose' [
    var: b
  ]
```

这说明乘法表达式的左右两边都是函数调用。

### 函数调用

用户函数调用对应：

```text
CallExprAST
```

例如：

```toy
multiply_transpose(a, b)
```

AST 中保存：

- callee 名字：`multiply_transpose`。
- 参数表达式列表：`a`、`b`。

AST dump 中显示：

```text
Call 'multiply_transpose' [
  var: a
  var: b
]
```

### 内建 `print`

`print()` 在 Toy 中是内建函数：

```toy
print(c);
```

在 AST 中它不是普通 `CallExprAST`，而是：

```text
PrintExprAST
```

这是 Ch1 AST 设计中的一个小细节：Parser 看到函数名是 `print` 时，会构造专门的 `PrintExprAST`。

AST dump 中显示：

```text
Print [
  var: c
]
```

### 内建 `transpose`

`transpose()` 也是 Toy 的内建能力：

```toy
transpose(a)
```

但在 Ch1 的 AST 中，它暂时仍然表现为普通函数调用：

```text
CallExprAST
  callee = "transpose"
```

也就是说，Ch1 AST 并没有为 `transpose` 单独定义 `TransposeExprAST`。后续生成 MLIR 时，`transpose` 会被识别并生成对应的 Toy operation。

### Return

返回语句对应：

```text
ReturnExprAST
```

例如：

```toy
return transpose(a) * transpose(b);
```

AST dump 中显示：

```text
Return
  BinOp: *
    ...
```

`ReturnExprAST` 的返回值是可选的，所以 parser 也支持：

```toy
return;
```

如果没有返回表达式，AST dump 会显示：

```text
Return
  (void)
```

## AST 总体结构

Toy AST 的总体层级是：

```text
ModuleAST
  FunctionAST
    PrototypeAST
      VariableExprAST 参数列表
    ExprASTList 函数体
      ExprAST
      ExprAST
      ExprAST
```

用更具体的方式表示：

```text
ModuleAST
  functions: vector<FunctionAST>

FunctionAST
  proto: unique_ptr<PrototypeAST>
  body: unique_ptr<ExprASTList>

PrototypeAST
  name: string
  args: vector<unique_ptr<VariableExprAST>>

ExprASTList
  vector<unique_ptr<ExprAST>>

ExprAST
  VarDeclExprAST
  ReturnExprAST
  NumberExprAST
  LiteralExprAST
  VariableExprAST
  BinaryExprAST
  CallExprAST
  PrintExprAST
```

这就是 Ch1 的核心：所有 Toy 程序都会先被组织成这棵树。

## `AST.h` 结构详解

### `VarType`

源码位置：

```text
mlir/examples/toy/Ch1/include/toy/AST.h
```

定义：

```cpp
struct VarType {
  std::vector<int64_t> shape;
};
```

`VarType` 只保存 shape，不保存元素类型。原因是 Toy 语言只有一种元素类型：64 位浮点数。

几个例子：

| Toy 写法 | `VarType::shape` | AST dump |
| --- | --- | --- |
| `var a = ...` | `[]` | `a<>` |
| `var b<2, 3> = ...` | `[2, 3]` | `b<2, 3>` |
| `var c<6> = ...` | `[6]` | `c<6>` |

### `ExprAST`

所有表达式节点都继承自：

```cpp
class ExprAST
```

它保存两个关键字段：

```cpp
ExprASTKind kind;
Location location;
```

`kind` 用于区分具体节点类型：

```cpp
Expr_VarDecl
Expr_Return
Expr_Num
Expr_Literal
Expr_Var
Expr_BinOp
Expr_Call
Expr_Print
```

`location` 记录源码位置，用于 AST dump 和错误诊断。AST dump 里经常看到：

```text
@.../ast.toy:11:3
```

这表示节点来自 `ast.toy` 第 11 行第 3 列。

### `NumberExprAST`

表示数字字面量：

```cpp
class NumberExprAST : public ExprAST {
  double val;
};
```

Toy 中所有数字都按 `double` 存储。

### `LiteralExprAST`

表示 tensor literal：

```cpp
class LiteralExprAST : public ExprAST {
  std::vector<std::unique_ptr<ExprAST>> values;
  std::vector<int64_t> dims;
};
```

这里有一个很重要的设计：`values` 的元素类型是 `ExprAST`，不是 `double`。

原因是 literal 可以嵌套：

```toy
[[1, 2, 3], [4, 5, 6]]
```

外层 literal 的 `values` 里放的是两个内层 `LiteralExprAST`。内层 literal 的 `values` 里才是 `NumberExprAST`。

所以 literal 本身也是一棵小树。

### `VariableExprAST`

表示变量引用：

```cpp
class VariableExprAST : public ExprAST {
  std::string name;
};
```

它只保存名字，不保存“这个名字对应哪个声明”。符号解析不是 Ch1 AST 的职责。

### `VarDeclExprAST`

表示变量声明：

```cpp
class VarDeclExprAST : public ExprAST {
  std::string name;
  VarType type;
  std::unique_ptr<ExprAST> initVal;
};
```

它把变量名、显式 shape 和初始化表达式连在一起。

例如：

```toy
var c = multiply_transpose(a, b);
```

大致对应：

```text
VarDeclExprAST
  name = "c"
  type.shape = []
  initVal = CallExprAST("multiply_transpose", [a, b])
```

### `ReturnExprAST`

表示返回语句：

```cpp
class ReturnExprAST : public ExprAST {
  std::optional<std::unique_ptr<ExprAST>> expr;
};
```

返回值是 optional，所以既能表示：

```toy
return x;
```

也能表示：

```toy
return;
```

### `BinaryExprAST`

表示二元表达式：

```cpp
class BinaryExprAST : public ExprAST {
  char op;
  std::unique_ptr<ExprAST> lhs, rhs;
};
```

例如：

```toy
a * b
```

对应：

```text
BinaryExprAST
  op = '*'
  lhs = VariableExprAST("a")
  rhs = VariableExprAST("b")
```

### `CallExprAST`

表示普通函数调用：

```cpp
class CallExprAST : public ExprAST {
  std::string callee;
  std::vector<std::unique_ptr<ExprAST>> args;
};
```

例如：

```toy
multiply_transpose(a, b)
```

对应：

```text
CallExprAST
  callee = "multiply_transpose"
  args = [VariableExprAST("a"), VariableExprAST("b")]
```

### `PrintExprAST`

表示内建 `print()`：

```cpp
class PrintExprAST : public ExprAST {
  std::unique_ptr<ExprAST> arg;
};
```

Toy 把 `print` 作为内建能力单独建模，而不是普通函数调用。

### `PrototypeAST`

表示函数签名的语法部分：

```cpp
class PrototypeAST {
  Location location;
  std::string name;
  std::vector<std::unique_ptr<VariableExprAST>> args;
};
```

例如：

```toy
def multiply_transpose(a, b)
```

对应：

```text
PrototypeAST
  name = "multiply_transpose"
  args = ["a", "b"]
```

注意：这里的参数用 `VariableExprAST` 表示。它们只是参数名，还没有类型信息。

### `FunctionAST`

表示一个完整函数：

```cpp
class FunctionAST {
  std::unique_ptr<PrototypeAST> proto;
  std::unique_ptr<ExprASTList> body;
};
```

例如：

```toy
def main() {
  print(1);
}
```

对应：

```text
FunctionAST
  PrototypeAST("main", [])
  body:
    PrintExprAST(NumberExprAST(1))
```

### `ModuleAST`

表示整个源文件：

```cpp
class ModuleAST {
  std::vector<FunctionAST> functions;
};
```

Toy 的 module 就是一组函数定义。

例如：

```toy
def f(x) {
  return x;
}

def main() {
  print(f(1));
}
```

对应：

```text
ModuleAST
  FunctionAST("f")
  FunctionAST("main")
```

## AST 节点速查表

| Toy 语法 | AST 节点 | 说明 |
| --- | --- | --- |
| 整个文件 | `ModuleAST` | 一组函数 |
| `def main() { ... }` | `FunctionAST` | 一个函数定义 |
| `def f(a, b)` | `PrototypeAST` | 函数名和参数名 |
| 函数体 `{ ... }` | `ExprASTList` | 表达式列表 |
| `var a = expr;` | `VarDeclExprAST` | 变量声明 |
| `return expr;` | `ReturnExprAST` | 返回语句 |
| `1`、`2.5` | `NumberExprAST` | 数字字面量 |
| `[1, 2]`、`[[1, 2], [3, 4]]` | `LiteralExprAST` | tensor literal |
| `a` | `VariableExprAST` | 变量引用 |
| `a + b`、`a - b`、`a * b` | `BinaryExprAST` | 二元表达式 |
| `foo(a, b)` | `CallExprAST` | 普通函数调用 |
| `transpose(a)` | `CallExprAST` | Ch1 中仍是普通调用 |
| `print(a)` | `PrintExprAST` | 内建 print |

## 对照 `ast.toy` 读 AST dump

测试文件：

```text
mlir/test/Examples/Toy/Ch1/ast.toy
```

第一段函数：

```toy
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}
```

对应 AST dump 的核心结构：

```text
Function
  Proto 'multiply_transpose'
  Params: [a, b]
  Block {
    Return
      BinOp: *
        Call 'transpose' [
          var: a
        ]
        Call 'transpose' [
          var: b
        ]
  } // Block
```

逐层解释：

- `Function`：这是一个函数。
- `Proto 'multiply_transpose'`：函数名是 `multiply_transpose`。
- `Params: [a, b]`：参数名是 `a` 和 `b`。
- `Block`：函数体。
- `Return`：函数体里有一个返回语句。
- `BinOp: *`：返回值是乘法表达式。
- 两个 `Call 'transpose'`：乘法左右两边分别调用 `transpose(a)` 和 `transpose(b)`。
- `var: a`、`var: b`：调用参数是变量引用。

第二段函数：

```toy
def main() {
  var a = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  var c = multiply_transpose(a, b);
  var d = multiply_transpose(b, a);
  var e = multiply_transpose(c, d);
  var f = multiply_transpose(a, c);
}
```

核心 AST 结构：

```text
Function
  Proto 'main'
  Params: []
  Block {
    VarDecl a<>
      Literal: <2, 3>[ ... ]
    VarDecl b<2, 3>
      Literal: <6>[ ... ]
    VarDecl c<>
      Call 'multiply_transpose' [
        var: a
        var: b
      ]
    ...
  } // Block
```

几个关键观察：

- `main` 没有参数，所以 `Params: []`。
- `a<>` 没有显式 shape，但初始化 literal 本身有 `<2, 3>`。
- `b<2, 3>` 有显式 shape，但右侧 literal 是 `<6>`。这表示源码写的是“把 6 个元素 reshape 成 2x3”。
- `c<>`、`d<>`、`e<>`、`f<>` 都没有显式 shape，初始化值是函数调用。

## AST dump 是怎么打印出来的

AST dump 实现在：

```text
mlir/examples/toy/Ch1/parser/AST.cpp
```

核心类是：

```cpp
class ASTDumper
```

它做两件事：

1. 遍历 AST。
2. 按缩进打印节点。

例如 `dump(FunctionAST *node)` 会先打印函数，再打印 prototype 和 body：

```cpp
void ASTDumper::dump(FunctionAST *node) {
  INDENT();
  llvm::errs() << "Function \n";
  dump(node->getProto());
  dump(node->getBody());
}
```

`dump(ExprAST *expr)` 会用 `llvm::TypeSwitch` 根据实际节点类型分发：

```cpp
llvm::TypeSwitch<ExprAST *>(expr)
    .Case<BinaryExprAST, CallExprAST, LiteralExprAST, NumberExprAST,
          PrintExprAST, ReturnExprAST, VarDeclExprAST, VariableExprAST>(
        [&](auto *node) { this->dump(node); })
```

这说明 AST dump 的打印结构和 `AST.h` 中的节点类是一一对应的。

## Ch1 AST 的边界

理解 AST 时要特别注意 Ch1 的边界。Ch1 做的是语法结构化，不是完整编译。

Ch1 会做：

- 识别函数。
- 识别变量声明。
- 识别表达式。
- 识别 literal 的嵌套维度。
- 保存源码位置。
- 构造 AST。
- dump AST。

Ch1 不负责：

- 把 AST 转成 MLIR。
- 检查变量是否声明过。
- 检查函数是否存在。
- 检查 tensor shape 是否兼容。
- 优化表达式。
- 生成代码。

例如下面这个程序可能语法上能进入 AST：

```toy
def main() {
  print(not_declared);
}
```

但 `not_declared` 是否真的存在，不是 Ch1 AST 的职责。后续 MLIRGen 或更后面的阶段才会处理这类问题。

## 常见误区

### 误区 1：AST 是 MLIR

AST 不是 MLIR。

AST 仍然是 Toy 前端自己的 C++ 数据结构。它的节点是 `ModuleAST`、`FunctionAST`、`VarDeclExprAST` 这样的 C++ 类。

MLIR 是下一阶段才生成的 IR，里面会出现 `toy.func`、`toy.constant`、`toy.generic_call` 等 operation。

### 误区 2：`VarDecl a<>` 表示 a 没有 shape

`a<>` 表示变量声明处没有显式写 shape。

例如：

```toy
var a = [[1, 2, 3], [4, 5, 6]];
```

AST dump 中是：

```text
VarDecl a<>
  Literal: <2, 3>[ ... ]
```

声明处没有 shape，但初始化 literal 有 shape。后续可以用这些信息推导变量 shape。

### 误区 3：`transpose` 在 Ch1 有专门 AST 节点

没有。

Ch1 中的 `transpose(a)` 是：

```text
CallExprAST
  callee = "transpose"
```

后续生成 MLIR 时，才会根据 callee 名字生成 `toy.transpose`。

### 误区 4：AST 会检查 shape 错误

不会。

`ast.toy` 中最后有：

```toy
var f = multiply_transpose(a, c);
```

注释说这会触发 shape inference error，但不是 Ch1 阶段触发。Ch1 只负责把它解析成函数调用 AST。

## 动手运行

如果已经构建好 `toyc-ch1`，可以运行：

```bash
<build-dir>/bin/toyc-ch1 mlir/test/Examples/Toy/Ch1/ast.toy -emit=ast
```

观察输出中的这些关键行：

```text
Module:
Function
Proto 'multiply_transpose'
Params: [a, b]
Return
BinOp: *
Call 'transpose'
VarDecl a<>
Literal: <2, 3>
VarDecl b<2, 3>
Call 'multiply_transpose'
```

如果要验证空文件行为，可以看：

```text
mlir/test/Examples/Toy/Ch1/empty.toy
```

这个测试期望看到 parse error，而不是断言崩溃。

## 本节练习

### 练习 1：AST 节点速查表

自己整理一张表，至少包含下面这些节点：

```text
ModuleAST
FunctionAST
PrototypeAST
ExprASTList
VarDeclExprAST
ReturnExprAST
NumberExprAST
LiteralExprAST
VariableExprAST
BinaryExprAST
CallExprAST
PrintExprAST
```

每个节点写三列：

- 这个节点表示什么。
- 对应 Toy 源码长什么样。
- 在 AST dump 中大概长什么样。

### 练习 2：手动画 AST

对下面这段 Toy 程序手动画 AST：

```toy
def main() {
  var a = [1, 2, 3];
  var b = [4, 5, 6];
  print(a + b);
}
```

要求：

- 画出 `ModuleAST`。
- 画出 `FunctionAST` 和 `PrototypeAST`。
- 画出函数体中的三个表达式。
- 标明 `print(a + b)` 中 `PrintExprAST`、`BinaryExprAST`、`VariableExprAST` 的父子关系。

### 练习 3：解释 `ast.toy` 中的两个变量声明

解释下面两句在 AST dump 中为什么不同：

```toy
var a = [[1, 2, 3], [4, 5, 6]];
var b<2, 3> = [1, 2, 3, 4, 5, 6];
```

需要回答：

- 为什么 `a` 显示为 `a<>`。
- 为什么 `b` 显示为 `b<2, 3>`。
- 为什么 `a` 的 literal 是 `<2, 3>`。
- 为什么 `b` 的 literal 是 `<6>`。
- 这两句后续为什么可以表示相同 shape 的 tensor。

### 练习 4：区分普通 call、内建 print、内建 transpose

阅读 `ast.toy` 的 AST dump，回答：

- `multiply_transpose(a, b)` 对应什么 AST 节点。
- `print(c)` 对应什么 AST 节点。
- `transpose(a)` 对应什么 AST 节点。
- 为什么 `print` 和 `transpose` 在 Ch1 AST 中处理方式不同。

### 练习 5：观察一个错误边界

构造一个语法上合法但语义上有问题的程序，例如：

```toy
def main() {
  print(x);
}
```

运行：

```bash
<build-dir>/bin/toyc-ch1 your-file.toy -emit=ast
```

观察它是否能得到 AST。然后写一句话解释：为什么 Ch1 能接受它，后续阶段才会报错。

## 本节小结

本节最重要的结论是：AST 是 Toy 源码进入编译器后的第一种结构化表示。

需要记住的结构是：

```text
ModuleAST
  FunctionAST
    PrototypeAST
    ExprASTList
      VarDeclExprAST
      ReturnExprAST
      NumberExprAST
      LiteralExprAST
      VariableExprAST
      BinaryExprAST
      CallExprAST
      PrintExprAST
```

也要记住几个边界：

- AST 不是 MLIR。
- Ch1 不做完整语义检查。
- `transpose` 在 Ch1 是普通 `CallExprAST`。
- `print` 在 Ch1 是专门的 `PrintExprAST`。
- `VarDecl a<>` 只表示声明处没有显式 shape。

下一节课会继续沿着 Ch1 往前看，研究 Lexer 和 Parser 如何把字符流一步步变成这棵 AST。

## 学习记录模板

```text
本节主题：Toy 语言语法与 AST
我读过的源码：
我观察过的测试：
我运行过的命令：
我确认理解的 AST 节点：
我还不理解的问题：
下一步要验证的小实验：
```
