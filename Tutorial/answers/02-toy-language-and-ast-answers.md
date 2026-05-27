# 第 2 课练习答案：Toy 语言语法与 AST

## 填写说明

这里填写第二节课的练习答案和实验记录。写完后告诉我“检查第二课答案”，我会对照 `Tutorial/02-toy-language-and-ast.md`、Ch1 源码和测试帮你验证。

## 练习 1：AST 节点速查表

| AST 节点 | 表示什么 | 对应 Toy 源码示例 | AST dump 大概长什么样 |
| --- | --- | --- | --- |
| `ModuleAST` | 一组函数 | def f(x) { return x; } def main() { print(f(1)); } | ModuleAST    FunctionAST("f")  FunctionAST("main")|
| `FunctionAST` | 一个函数定义 | def main() { print(1)； } | FunctionAST PrototypeAST("main", []) body: printExprAST(NumberExprAST(1)) |
| `PrototypeAST` | 函数签名 | def f(a, b) | PrototypeAST name = "f" args = ["a", "b"] |
| `ExprASTList` | 表达式列表 | 函数体 {...} |  |
| `VarDeclExprAST` | 变量声明 | var c = f(a, b); | VarDeclExprAST name = 'c' type.shape = [] initVal = CallExprAST("f", [a, b]) |
| `ReturnExprAST` | 返回语句 | return x; |  |
| `NumberExprAST` | 数字字面量 |  |  |
| `LiteralExprAST` | 张量字面量 |  |  |
| `VariableExprAST` | 变量引用 |  |  |
| `BinaryExprAST` | 二元操作 | a * b | BinaryExprAST op = '*' lhs = VariableExprAST("a") rhs = VariableExprAST("b") |
| `CallExprAST` | 普通函数调用 | f(a, b) | CallExprAST callee = "f" args = [VariableExprAST("a"), VariableExprAST("b")] |
| `PrintExprAST` | 内建打印函数 |  |  |

## 练习 2：手动画 AST

题目程序：

```toy
def main() {
  var a = [1, 2, 3];
  var b = [4, 5, 6];
  print(a + b);
}
```

### 我的 AST 草图

```text
Module
  Function
    Proto 'main' @
    Params: []
    Block {
      ValDecl a<> @
        Literal: <3>[1.000000e+00, 2.000000e+00, 3.000000e+00] @
      ValDecl b<> @
        Literal: <3>[4.000000e+00, 5.000000e+00, 6.000000e+00] @
      Print 
        BinOp: + @
          var: a @
          var: b @
    }

```

### 父子关系说明

- `ModuleAST`：
- `FunctionAST`：
- `PrototypeAST`：
- 函数体中的第 1 个表达式：
- 函数体中的第 2 个表达式：
- 函数体中的第 3 个表达式：
- `print(a + b)` 的内部结构：

## 练习 3：解释 `ast.toy` 中的两个变量声明

源码：

```toy
var a = [[1, 2, 3], [4, 5, 6]];
var b<2, 3> = [1, 2, 3, 4, 5, 6];
```

### 我的解释

- 为什么 `a` 显示为 `a<>`：没有显式指定shape
- 为什么 `b` 显示为 `b<2, 3>`：显式指定了shape
- 为什么 `a` 的 literal 是 `<2, 3>`：字面量shape
- 为什么 `b` 的 literal 是 `<6>`：字面量shape
- 这两句后续为什么可以表示相同 shape 的 tensor：显式指定shape后会对字面量reshape

## 练习 4：区分普通 call、内建 print、内建 transpose

- `multiply_transpose(a, b)` 对应什么 AST 节点：FunctionAST
- `print(c)` 对应什么 AST 节点： PrintExprAST
- `transpose(a)` 对应什么 AST 节点：CallExprAST
- 为什么 `print` 和 `transpose` 在 Ch1 AST 中处理方式不同：

## 练习 5：观察一个错误边界

测试程序：

```toy
def main() {
  print(x);
}
```

### 我运行过的命令

```bash
  ./toyc-ch1 ../../mlir/test/Examples/Toy/Ch1/example.toy -emit=ast
```

### 我的观察

- 是否能得到 AST：是
- AST dump 中 `x` 如何出现：var： x
- 为什么 Ch1 能接受它：ch1只是在前端将toy语言文本转换成结构化的AST，并不做检查
- 后续阶段为什么可能报错：没有声明变量 x

## Codex 校验与修正建议

整体结论：第二课答案已经抓住了 AST 的主线。练习 3 和练习 5 基本正确；需要重点修正的是练习 1 的表格完整性、练习 2 的 AST dump 细节，以及练习 4 中普通函数调用的节点判断。

### 练习 1 修正建议

你的表格方向正确，但有几项还没填写，另外 “AST dump 大概长什么样” 建议尽量使用 `toyc-ch1 -emit=ast` 实际打印出来的形式，而不是 C++ 类名形式。

参考修正版：

| AST 节点 | 表示什么 | 对应 Toy 源码示例 | AST dump 大概长什么样 |
| --- | --- | --- | --- |
| `ModuleAST` | 整个 Toy 源文件，一组函数 | 一个 `.toy` 文件 | `Module:` |
| `FunctionAST` | 一个函数定义 | `def main() { print(1); }` | `Function` |
| `PrototypeAST` | 函数名和参数名列表 | `def f(a, b)` | `Proto 'f'`、`Params: [a, b]` |
| `ExprASTList` | 函数体中的表达式列表 | `{ var a = 1; print(a); }` | `Block { ... } // Block` |
| `VarDeclExprAST` | 变量声明 | `var c = f(a, b);` | `VarDecl c<>` |
| `ReturnExprAST` | 返回语句 | `return x;` | `Return` 后面缩进打印返回表达式 |
| `NumberExprAST` | 数字字面量 | `1`、`2.5` | `1.000000e+00 @...` |
| `LiteralExprAST` | tensor literal | `[1, 2, 3]`、`[[1, 2], [3, 4]]` | `Literal: <3>[ ... ]` 或 `Literal: <2, 2>[ ... ]` |
| `VariableExprAST` | 变量引用 | `a` | `var: a @...` |
| `BinaryExprAST` | 二元表达式 | `a * b` | `BinOp: *`，下面是 lhs 和 rhs |
| `CallExprAST` | 普通函数调用 | `f(a, b)`、`transpose(a)` | `Call 'f' [` 或 `Call 'transpose' [` |
| `PrintExprAST` | 内建 `print` 调用 | `print(a)` | `Print [`，下面是被打印的表达式 |

注意：`FunctionAST`、`PrototypeAST` 这些是 C++ 类名；AST dump 中通常打印的是 `Function`、`Proto 'main'`，不会直接打印 `FunctionAST` 字符串。

### 练习 2 修正建议

你的 AST 草图整体结构正确，但有两处细节需要改：

- `ValDecl` 应该是 `VarDecl`。
- `Print` 的 dump 形式会带 `[`，并且最后有对应的 `]`。

参考修正版：

```text
Module:
  Function
    Proto 'main' @...
    Params: []
    Block {
      VarDecl a<> @...
        Literal: <3>[ 1.000000e+00, 2.000000e+00, 3.000000e+00] @...
      VarDecl b<> @...
        Literal: <3>[ 4.000000e+00, 5.000000e+00, 6.000000e+00] @...
      Print [ @...
        BinOp: + @...
          var: a @...
          var: b @...
      ]
    } // Block
```

父子关系可以补充为：

- `ModuleAST`：包含一个 `FunctionAST`，也就是 `main`。
- `FunctionAST`：包含 `PrototypeAST("main", [])` 和一个函数体 `ExprASTList`。
- `PrototypeAST`：函数名是 `main`，参数列表为空。
- 函数体中的第 1 个表达式：`VarDeclExprAST("a")`，初始化值是 `LiteralExprAST<3>`。
- 函数体中的第 2 个表达式：`VarDeclExprAST("b")`，初始化值是 `LiteralExprAST<3>`。
- 函数体中的第 3 个表达式：`PrintExprAST`。
- `print(a + b)` 的内部结构：`PrintExprAST` 的子节点是 `BinaryExprAST('+')`，这个 `BinaryExprAST` 的左右子节点分别是 `VariableExprAST("a")` 和 `VariableExprAST("b")`。

### 练习 3 修正建议

你的答案正确，可以稍微补充得更精确：

- `a` 显示为 `a<>`：因为变量声明处没有显式写 shape。
- `b` 显示为 `b<2, 3>`：因为变量声明处显式写了 shape。
- `a` 的 literal 是 `<2, 3>`：因为右侧 literal 是二维嵌套数组，外层 2 个元素，每个内层 3 个数字。
- `b` 的 literal 是 `<6>`：因为右侧 literal 是一维数组，共 6 个数字。
- 两者后续可以表示相同 shape：`b` 的声明 shape `<2, 3>` 会要求后续把 6 个元素解释为 2x3 tensor；元素数量匹配，所以可以 reshape。

需要注意：Ch1 这里只是 AST 阶段，真正的 reshape 语义和类型处理要到后续 MLIRGen/shape inference/lowering 阶段继续展开。

### 练习 4 修正建议

这一题有一个关键错误：

- `multiply_transpose(a, b)` 对应的是 `CallExprAST`，不是 `FunctionAST`。

原因：`FunctionAST` 表示函数定义，例如：

```toy
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}
```

而：

```toy
multiply_transpose(a, b)
```

是函数调用表达式，所以是 `CallExprAST`。

参考修正版：

- `multiply_transpose(a, b)` 对应 `CallExprAST`。
- `print(c)` 对应 `PrintExprAST`。
- `transpose(a)` 对应 `CallExprAST`。
- `print` 和 `transpose` 在 Ch1 AST 中处理方式不同：Parser 对 `print` 做了特殊处理，直接构造 `PrintExprAST`；而 `transpose` 在 Ch1 仍然只是普通 callee 名字为 `"transpose"` 的 `CallExprAST`，后续 MLIRGen 阶段才会识别并生成 `toy.transpose`。

### 练习 5 修正建议

你的判断正确：

- `print(x);` 语法上合法，所以 Ch1 可以生成 AST。
- AST dump 中 `x` 会作为 `VariableExprAST` 打印成 `var: x`。
- Ch1 Parser 不做符号解析，所以不会检查 `x` 有没有声明。
- 后续阶段在把 AST 转成 MLIR 时需要查找变量定义，因此可能报“未知变量”之类的错误。

建议把 `var： x` 中的中文冒号改成英文冒号，写成：

```text
var: x
```

## 我的问题

- 
