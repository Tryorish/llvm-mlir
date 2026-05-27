# 第 3 课练习答案：Lexer、Parser 与 AST dump

## 填写说明

这里填写第三节课的练习答案和实验记录。写完后告诉我“检查第三课答案”，我会对照 `Tutorial/03-lexer-parser-ast-dump.md`、Ch1 源码和测试帮你验证。

## 练习 1：写出完整调用链

### 我的调用链

```text
main -> parseInputFile -> MemoryBuffer -> LexerBuffer -> Parser -> parseModule -> dump -> ASTDumper

```

### 每一步负责什么

- `main`：程序入口
- `parseInputFile`：解析输入文件
- `MemoryBuffer`：将输入文件写入内存buffer
- `LexerBuffer`：从buffer中一行行提供字符，识别成token
- `Parser`：把token组织成AST
- `parseModule`：Parser入口
- `dump`：dump入口
- `ASTDumper`：dump出来

## 练习 2：手写 token 流

题目程序：

```toy
def main() {
  var a = [1, 2];
  print(a);
}
```

### 我的 token 流

```text
  tok_def
  tok_identifier("main")
  '('
  ')'
  '{'
  tok_var
  tok_identifier("a")
  '='
  '['
  tok_number(1)
  ','
  tok_number(2)
  ']'
  ';'
  tok_identifier("print")
  '('
  tok_identifier("a")
  ')'
  ';'
  '}'
  tok_eof
```

### 分类说明

- 关键字 token：tok_def tok_var 
- 标识符 token：main a print 
- 数字 token：1 2
- 标点符号 token：( ) { } [ ] ; ,

## 练习 3：追踪 Parser 调用

题目语句：

```toy
var b<2, 3> = [1, 2, 3, 4, 5, 6];
```

### 我的 Parser 调用链

```text
parseBlock
  parseDeclaration
    parseType
      parseTensorLiteralExpr
```

### 最终构造的 AST 节点

- 节点类型：VarDeckExprAST
- 变量名：b
- 显式 shape：(2, 3)
- 初始化表达式：[1 ,2 , 3, 4, 5, 6]

## 练习 4：二元表达式优先级

测试程序：

```toy
def main() {
  print(1 + 2 * 3);
}
```

### 我运行过的命令

```bash
  ./toyc-ch1 ../../mlir/test/Examples/Toy/Ch1/example.toy -emit=ast
```

### 我的观察

- 顶层 `BinOp` 是 `+` 还是 `*`：+
- 为什么 `*` 会在更深一层：优先级更高
- 这和 `getTokPrecedence()` 中的优先级数字有什么关系：数字越大优先级越高

## 练习 5：构造一个 Parser 错误

测试程序：

```toy
def main() {
  print(1)
}
```

### 我运行过的命令

```bash
  ./toyc-ch1 ../../mlir/test/Examples/Toy/Ch1/example.toy -emit=ast
```

### 我的观察

- 报错信息：Parse error (3, 1): expected ';' after expression but has Token 125 '}'
Parse error (3, 1): expected 'nothing' at end of module but has Token 125 '}'
- 报错行列：3行1列
- Parser 期望什么 token：；
- 当前拿到的 token 是什么：}

## 练习 6：不规则 tensor literal

测试程序：

```toy
def main() {
  var a = [[1, 2], [3, 4, 5]];
}
```

### 我运行过的命令

```bash
 ./toyc-ch1 ../../mlir/test/Examples/Toy/Ch1/example.toy -emit=ast
```

### 我的观察

- 是否报错：是
- 报错信息：VarDecl a<> @../../mlir/test/Examples/Toy/Ch1/example.toy:3:3
Assertion failed: (detail::isPresent(Val) && "dyn_cast on a non-existent value"), function dyn_cast, file Casting.h, line 662.
[1]    32781 abort      ./toyc-ch1 ../../mlir/test/Examples/Toy/Ch1/example.toy -emit=ast
- 为什么这个错误属于 Parser 阶段能发现的结构错误：字面值嵌套shape要一致

## 我的问题

- prototype ::= def id '(' decl_list ')'
decl_list ::= identifier | identifier, decl_list 什么意思，::= 如何理解

## Codex 校验与修正建议

整体结论：第三课答案基本通过。你已经掌握了从 `main()` 到 AST dump 的大链路、token 流、二元表达式优先级和常见 parser 错误。需要修正的主要是几个职责边界和术语细节。

### 练习 1 修正建议

你的调用链：

```text
main -> parseInputFile -> MemoryBuffer -> LexerBuffer -> Parser -> parseModule -> dump -> ASTDumper
```

方向正确。建议补得更精确一些：

```text
main
  -> cl::ParseCommandLineOptions
  -> parseInputFile
    -> llvm::MemoryBuffer::getFileOrSTDIN
    -> LexerBuffer
    -> Parser
    -> Parser::parseModule
  -> dump(*moduleAST)
    -> ASTDumper::dump(ModuleAST *)
```

每一步职责建议修正为：

- `main`：程序入口，解析命令行参数，调用 `parseInputFile()`，根据 `-emit=ast` 决定是否 dump AST。
- `parseInputFile`：读取输入文件，创建 `LexerBuffer` 和 `Parser`，调用 `parseModule()` 返回 `ModuleAST`。
- `MemoryBuffer`：把输入文件或 stdin 读入内存 buffer。这里是“读入”，不是“写入”。
- `LexerBuffer`：从内存 buffer 中一行一行提供字符。
- `Lexer`：把字符识别成 token。注意这个职责属于 `Lexer`，不是 `LexerBuffer`。
- `Parser`：消费 token，并根据 Toy 语法构造 AST。
- `parseModule`：Parser 的入口，解析一组函数定义，最后构造 `ModuleAST`。
- `dump`：AST dump 的公共入口，调用 `ASTDumper`。
- `ASTDumper`：递归遍历 AST，并按缩进打印文本。

### 练习 2 修正建议

你的 token 流基本正确。分类说明里漏了一个标点符号 token：`=`。

参考分类：

- 关键字 token：`tok_def`、`tok_var`
- 标识符 token：`tok_identifier("main")`、`tok_identifier("a")`、`tok_identifier("print")`
- 数字 token：`tok_number(1)`、`tok_number(2)`
- 标点符号 token：`(`、`)`、`{`、`}`、`=`、`[`、`]`、`,`、`;`

注意：`print` 在 Lexer 阶段仍然是 `tok_identifier("print")`，不是关键字 token。

### 练习 3 修正建议

你的调用链少了一层 `parseExpression()`，并且最终节点类型有拼写错误。

参考调用链：

```text
parseBlock
  -> parseDeclaration
    -> parseType
    -> parseExpression
      -> parsePrimary
        -> parseTensorLiteralExpr
```

解释：

- `parseBlock()` 看到当前 token 是 `tok_var`，于是调用 `parseDeclaration()`。
- `parseDeclaration()` 解析 `var b`。
- 看到 `<` 后调用 `parseType()`，得到 shape `[2, 3]`。
- 消费 `=` 后调用 `parseExpression()` 解析初始化表达式。
- 初始化表达式以 `[` 开头，所以 `parsePrimary()` 分发到 `parseTensorLiteralExpr()`。

最终构造：

- 节点类型：`VarDeclExprAST`，不是 `VarDeckExprAST`。
- 变量名：`b`。
- 显式 shape：`[2, 3]`，也可以写成 `<2, 3>`。
- 初始化表达式：`LiteralExprAST`，其 literal dims 是 `[6]`，对应源码右侧的一维 `[1, 2, 3, 4, 5, 6]`。

### 练习 4 修正建议

你的结论正确：

- 顶层 `BinOp` 是 `+`。
- `*` 在更深一层，因为 `*` 优先级更高。
- `getTokPrecedence()` 中 `*` 返回 `40`，`+` 和 `-` 返回 `20`，数字越大绑定越紧。

建议运行这个练习时最好使用只包含下面代码的临时文件：

```toy
def main() {
  print(1 + 2 * 3);
}
```

如果 `example.toy` 里还包含不规则 tensor literal，会影响后面的 dump 或触发错误，干扰这个练习。

期望 AST 形状大致是：

```text
Print [
  BinOp: +
    1.000000e+00
    BinOp: *
      2.000000e+00
      3.000000e+00
]
```

### 练习 5 修正建议

你的观察正确。

缺少分号的程序：

```toy
def main() {
  print(1)
}
```

`parseBlock()` 解析完 `print(1)` 后，要求当前 token 必须是 `;`。但实际读到的是 `}`，所以报：

```text
Parse error (3, 1): expected ';' after expression but has Token 125 '}'
```

`125` 是字符 `}` 的 ASCII 值。

第二条：

```text
Parse error (3, 1): expected 'nothing' at end of module but has Token 125 '}'
```

是因为函数解析失败后，`parseModule()` 发现还没有到 EOF，于是又报了模块级错误。

### 练习 6 修正建议

你的判断“shape 要一致”是正确的，但这次输出里出现了断言崩溃，这里要单独理解。

不规则 literal：

```toy
var a = [[1, 2], [3, 4, 5]];
```

按源码意图，`parseTensorLiteralExpr()` 会检查嵌套 literal 的维度是否统一。第一行是 `<2>`，第二行是 `<3>`，不一致时应该触发：

```text
expected 'uniform well-nested dimensions'
```

但 Ch1 的 `parseDeclaration()` 有一个实现细节：它调用 `parseExpression()` 后没有检查返回值是否为 `nullptr`，就直接构造了 `VarDeclExprAST`：

```cpp
auto expr = parseExpression();
return std::make_unique<VarDeclExprAST>(..., std::move(expr));
```

所以当 literal 解析失败、初始化表达式为空时，AST 里可能留下一个空的 `initVal`。随后 `ASTDumper` 试图 dump 这个空表达式，就可能触发你看到的 `dyn_cast on a non-existent value` 断言。

因此这个练习可以这样总结：

- 这个错误确实属于 Parser 阶段能发现的结构错误，因为 literal 嵌套维度不一致。
- 你看到的 abort 是 Ch1 教程代码在错误恢复上的局限，不是正常的成功路径。
- 更理想的 parser 行为是：发现 `parseExpression()` 失败后，让 `parseDeclaration()` 返回 `nullptr`，避免后续 dump 空 AST 节点。

### 关于 `::=` 的问题

你问：

```text
prototype ::= def id '(' decl_list ')'
decl_list ::= identifier | identifier, decl_list
```

这里的 `::=` 是语法规则里的“定义为”或“可以展开为”。它常见于 BNF/EBNF 风格的语法描述。

可以读作：

```text
prototype 定义为：def id '(' decl_list ')'
decl_list 定义为：identifier，或者 identifier 后面跟逗号和另一个 decl_list
```

其中 `|` 表示“或者”。

所以：

```text
decl_list ::= identifier | identifier, decl_list
```

表示参数列表可以是：

```text
a
a, b
a, b, c
...
```

对应 Toy 代码：

```toy
def f(a) { ... }
def f(a, b) { ... }
def f(a, b, c) { ... }
```

而：

```text
prototype ::= def id '(' decl_list ')'
```

表示函数原型长这样：

```toy
def 函数名(参数列表)
```

例如：

```toy
def multiply_transpose(a, b)
```

对应：

```text
def                -> 关键字 def
id                 -> multiply_transpose
'('                -> 左括号
decl_list          -> a, b
')'                -> 右括号
```
