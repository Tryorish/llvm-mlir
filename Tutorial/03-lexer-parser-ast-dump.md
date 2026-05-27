# 第 3 课：Lexer、Parser 与 AST dump

## 本节定位

第二节课已经说明了 Toy 语言的语法和 AST 节点结构。本节课继续沿着 Ch1 往前追：编译器到底是如何从 `.toy` 源码文本得到这棵 AST 的。

这一节的主线是：

```text
.toy 源文件
  -> llvm::MemoryBuffer
  -> LexerBuffer
  -> Lexer 产生 Token
  -> Parser 消费 Token 并构造 AST
  -> dump(ModuleAST)
  -> ASTDumper 递归打印 AST
```

本节不涉及 MLIR。Ch1 的目标非常克制：能读 Toy 源码、能构造 AST、能把 AST 打印出来。

## 本节目标

- 理解 `toyc-ch1` 的入口流程。
- 理解 Lexer 如何把字符流切成 token。
- 理解 Parser 如何按 Toy 语法构造 AST。
- 理解递归下降 parser 的基本组织方式。
- 理解二元表达式优先级如何处理。
- 理解 AST dump 如何递归遍历 AST。
- 能沿着一个 Toy 程序追踪“源码文本 -> token -> AST -> dump 输出”的调用链。

## 本节关键文件

源码：

```text
mlir/examples/toy/Ch1/toyc.cpp
mlir/examples/toy/Ch1/include/toy/Lexer.h
mlir/examples/toy/Ch1/include/toy/Parser.h
mlir/examples/toy/Ch1/parser/AST.cpp
```

测试：

```text
mlir/test/Examples/Toy/Ch1/ast.toy
mlir/test/Examples/Toy/Ch1/empty.toy
```

建议阅读顺序：

1. 先读 `toyc.cpp`，看程序入口。
2. 再读 `Lexer.h`，看 token 是怎么产生的。
3. 再读 `Parser.h`，看 token 如何变成 AST。
4. 最后读 `parser/AST.cpp`，看 AST 如何被 dump。

## 总体调用链

Ch1 的完整调用链可以压缩成下面这张图：

```text
main()
  |
  v
cl::ParseCommandLineOptions(...)
  |
  v
parseInputFile(inputFilename)
  |
  v
llvm::MemoryBuffer::getFileOrSTDIN(filename)
  |
  v
LexerBuffer lexer(buffer.begin(), buffer.end(), filename)
  |
  v
Parser parser(lexer)
  |
  v
parser.parseModule()
  |
  v
std::unique_ptr<ModuleAST>
  |
  v
dump(*moduleAST)
  |
  v
ASTDumper().dump(&module)
```

这里有几个重要边界：

- `MemoryBuffer` 负责把输入文件读成内存 buffer。
- `LexerBuffer` 负责从 buffer 中一行一行提供字符。
- `Lexer` 负责把字符识别成 token。
- `Parser` 负责把 token 组织成 AST。
- `ASTDumper` 负责递归打印 AST。

## `toyc.cpp`：Ch1 编译器入口

文件：

```text
mlir/examples/toy/Ch1/toyc.cpp
```

Ch1 的 `toyc.cpp` 只支持一个输出动作：

```cpp
enum Action { None, DumpAST };
```

命令行参数是：

```cpp
static cl::opt<std::string> inputFilename(...);
static cl::opt<enum Action> emitAction("emit", ...);
```

所以运行方式是：

```bash
<build-dir>/bin/toyc-ch1 mlir/test/Examples/Toy/Ch1/ast.toy -emit=ast
```

### `parseInputFile()`

核心函数：

```cpp
std::unique_ptr<toy::ModuleAST> parseInputFile(llvm::StringRef filename)
```

它做四件事：

1. 读取输入文件。
2. 用输入 buffer 创建 `LexerBuffer`。
3. 用 lexer 创建 `Parser`。
4. 调用 `parser.parseModule()` 得到 `ModuleAST`。

对应代码结构：

```cpp
auto buffer = fileOrErr.get()->getBuffer();
LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
Parser parser(lexer);
return parser.parseModule();
```

这说明 Ch1 的真正工作不在 `toyc.cpp`，而在 `Lexer.h` 和 `Parser.h`。

### `main()`

`main()` 做的是命令行分发：

```cpp
auto moduleAST = parseInputFile(inputFilename);
if (!moduleAST)
  return 1;

switch (emitAction) {
case Action::DumpAST:
  dump(*moduleAST);
  return 0;
default:
  llvm::errs() << "No action specified ...\n";
}
```

所以 Ch1 的行为很清楚：

- 解析失败：返回 1。
- `-emit=ast`：打印 AST。
- 没写 `-emit=ast`：提示没有指定 action。

## Lexer 的职责

文件：

```text
mlir/examples/toy/Ch1/include/toy/Lexer.h
```

Lexer 的职责是把字符流转换成 token 流。

例如源码：

```toy
def main() {
  var a = [1, 2, 3];
}
```

会被切成类似这样的 token：

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
','
tok_number(3)
']'
';'
'}'
tok_eof
```

Parser 不直接看字符，它只看这些 token。

## Token 定义

`Lexer.h` 中定义了：

```cpp
enum Token : int {
  tok_semicolon = ';',
  tok_parenthese_open = '(',
  tok_parenthese_close = ')',
  tok_bracket_open = '{',
  tok_bracket_close = '}',
  tok_sbracket_open = '[',
  tok_sbracket_close = ']',

  tok_eof = -1,

  tok_return = -2,
  tok_var = -3,
  tok_def = -4,

  tok_identifier = -5,
  tok_number = -6,
};
```

这里有一个设计细节：很多标点符号直接用它们的 ASCII 值作为 token，比如：

```text
'('
')'
'{'
'}'
'['
']'
','
';'
'+'
'-'
'*'
'='
'<'
'>'
```

而关键字、标识符、数字、EOF 使用负数 token。

这样 parser 可以直接写：

```cpp
if (lexer.getCurToken() != ')')
```

也可以写：

```cpp
if (lexer.getCurToken() != tok_identifier)
```

## `Location`：源码位置

`Location` 定义在 `Lexer.h`：

```cpp
struct Location {
  std::shared_ptr<std::string> file;
  int line;
  int col;
};
```

每个 AST 节点都会带一个 `Location`。第二节课里看到的 AST dump：

```text
VarDecl a<> @.../ast.toy:11:3
```

这个 `@...:11:3` 就来自 `Location`。

它的作用有两个：

- AST dump 时告诉你节点来自源码哪里。
- 解析错误时能给出行列号。

## `Lexer` 如何读字符

`Lexer` 是抽象基类，真正读取 buffer 的是子类：

```cpp
class LexerBuffer final : public Lexer
```

`LexerBuffer::readNextLine()` 从内存 buffer 中取下一行：

```cpp
llvm::StringRef readNextLine() override
```

`Lexer::getNextChar()` 再从当前行中逐字符读取。

关系是：

```text
LexerBuffer::readNextLine()
  -> 提供一行字符
Lexer::getNextChar()
  -> 从当前行取下一个字符
Lexer::getTok()
  -> 把若干字符组成一个 token
Lexer::getNextToken()
  -> 更新 curTok
```

## `getTok()`：token 识别规则

`Lexer::getTok()` 是 Lexer 的核心。

### 1. 跳过空白

```cpp
while (isspace(lastChar))
  lastChar = Token(getNextChar());
```

空格、换行、制表符不会作为 token 暴露给 parser。

### 2. 识别标识符和关键字

规则：

```text
[a-zA-Z][a-zA-Z0-9_]*
```

代码先读出 identifier 字符串，然后判断是否是关键字：

```cpp
if (identifierStr == "return")
  return tok_return;
if (identifierStr == "def")
  return tok_def;
if (identifierStr == "var")
  return tok_var;
return tok_identifier;
```

所以：

```toy
def
return
var
```

会变成关键字 token。

而：

```toy
main
multiply_transpose
transpose
print
a
b
```

都会先变成 `tok_identifier`。

注意：`print` 和 `transpose` 在 Lexer 阶段不是关键字。它们只是普通 identifier。`print` 是 Parser 在 `parseIdentifierExpr()` 中特殊处理的。

### 3. 识别数字

规则：

```text
[0-9.]+
```

代码会把数字字符收集到 `numStr`，然后用：

```cpp
numVal = strtod(numStr.c_str(), nullptr);
```

转换成 `double`。

Parser 后续可以通过：

```cpp
lexer.getValue()
```

拿到当前数字的值。

### 4. 跳过注释

Toy 注释以 `#` 开头，持续到行尾：

```toy
# this is a comment
```

Lexer 看到 `#` 后，会持续读字符直到换行或 EOF，然后递归调用 `getTok()` 继续读下一个 token。

### 5. 识别 EOF

如果读到文件末尾：

```cpp
return tok_eof;
```

### 6. 其他字符直接返回 ASCII token

例如：

```toy
(
)
{
}
[
]
,
;
*
+
-
=
<
>
```

都会作为字符 token 返回。

## Parser 的职责

文件：

```text
mlir/examples/toy/Ch1/include/toy/Parser.h
```

Parser 的职责是消费 token，并构造 AST。

它是一个递归下降 parser。所谓递归下降，就是每一种语法结构都有一个对应的 parse 函数，例如：

| Toy 语法结构 | Parser 函数 | AST 节点 |
| --- | --- | --- |
| module | `parseModule()` | `ModuleAST` |
| function definition | `parseDefinition()` | `FunctionAST` |
| function prototype | `parsePrototype()` | `PrototypeAST` |
| block | `parseBlock()` | `ExprASTList` |
| variable declaration | `parseDeclaration()` | `VarDeclExprAST` |
| return | `parseReturn()` | `ReturnExprAST` |
| expression | `parseExpression()` | `ExprAST` |
| primary expression | `parsePrimary()` | `NumberExprAST` / `LiteralExprAST` / `VariableExprAST` / `CallExprAST` / `PrintExprAST` |
| tensor literal | `parseTensorLiteralExpr()` | `LiteralExprAST` |
| binary expression rhs | `parseBinOpRHS()` | `BinaryExprAST` |

## `parseModule()`：入口

Parser 的入口是：

```cpp
std::unique_ptr<ModuleAST> parseModule()
```

它先调用：

```cpp
lexer.getNextToken();
```

这一步叫 prime the lexer，也就是先读入第一个 token，让 `curTok` 有效。

然后循环解析函数定义：

```cpp
std::vector<FunctionAST> functions;
while (auto f = parseDefinition()) {
  functions.push_back(std::move(*f));
  if (lexer.getCurToken() == tok_eof)
    break;
}
```

最后构造：

```cpp
std::make_unique<ModuleAST>(std::move(functions))
```

这就是为什么 Toy 顶层必须是一组函数定义。

## `parsePrototype()`：函数签名

函数原型语法：

```text
prototype ::= def id '(' decl_list ')'
decl_list ::= identifier | identifier, decl_list
```

例如：

```toy
def multiply_transpose(a, b)
```

解析结果是：

```text
PrototypeAST
  name = "multiply_transpose"
  args = ["a", "b"]
```

注意：参数没有类型，只有名字。

## `parseDefinition()`：函数定义

函数定义语法：

```text
definition ::= prototype block
```

也就是：

```toy
def main() {
  ...
}
```

`parseDefinition()` 先调用 `parsePrototype()`，再调用 `parseBlock()`，最后构造：

```cpp
std::make_unique<FunctionAST>(std::move(proto), std::move(block))
```

## `parseBlock()`：函数体

block 语法：

```text
block ::= { expression_list }
expression_list ::= block_expr ; expression_list
block_expr ::= decl | "return" | expr
```

所以函数体里可以放：

- 变量声明。
- return 语句。
- 普通表达式。

`parseBlock()` 的关键分发逻辑是：

```cpp
if (lexer.getCurToken() == tok_var) {
  auto varDecl = parseDeclaration();
  exprList->push_back(std::move(varDecl));
} else if (lexer.getCurToken() == tok_return) {
  auto ret = parseReturn();
  exprList->push_back(std::move(ret));
} else {
  auto expr = parseExpression();
  exprList->push_back(std::move(expr));
}
```

每个表达式后面必须有分号：

```cpp
if (lexer.getCurToken() != ';')
  return parseError<ExprASTList>(";", "after expression");
```

因此 Toy 中下面这样少分号会报 parse error：

```toy
def main() {
  print(1)
}
```

## `parseDeclaration()`：变量声明

变量声明语法：

```text
decl ::= var identifier [ type ] = expr
```

例子：

```toy
var a = [1, 2, 3];
var b<2, 3> = [1, 2, 3, 4, 5, 6];
```

解析过程：

1. 看到 `tok_var`。
2. 读取变量名。
3. 如果看到 `<`，调用 `parseType()` 解析 shape。
4. 消费 `=`。
5. 调用 `parseExpression()` 解析初始化表达式。
6. 构造 `VarDeclExprAST`。

如果没有显式 type：

```cpp
if (!type)
  type = std::make_unique<VarType>();
```

所以 `var a = ...` 的 shape 是空 vector，AST dump 中显示为 `a<>`。

## `parseType()`：shape 声明

shape 语法：

```text
type ::= < shape_list >
shape_list ::= num | num , shape_list
```

例子：

```toy
<2, 3>
```

解析后：

```text
VarType::shape = [2, 3]
```

注意：`parseType()` 只处理 shape，不处理元素类型。Toy 元素类型默认是 `double`。

## `parseExpression()`：表达式入口

表达式入口：

```cpp
std::unique_ptr<ExprAST> parseExpression() {
  auto lhs = parsePrimary();
  if (!lhs)
    return nullptr;

  return parseBinOpRHS(0, std::move(lhs));
}
```

它先解析一个 primary expression，再尝试解析后续的二元表达式。

## `parsePrimary()`：基础表达式

`parsePrimary()` 根据当前 token 类型分发：

```cpp
case tok_identifier:
  return parseIdentifierExpr();
case tok_number:
  return parseNumberExpr();
case '(':
  return parseParenExpr();
case '[':
  return parseTensorLiteralExpr();
```

对应关系：

| 当前 token | 例子 | parse 函数 |
| --- | --- | --- |
| `tok_identifier` | `a`、`foo(a)`、`print(a)` | `parseIdentifierExpr()` |
| `tok_number` | `1` | `parseNumberExpr()` |
| `'('` | `(a + b)` | `parseParenExpr()` |
| `'['` | `[1, 2, 3]` | `parseTensorLiteralExpr()` |

## `parseIdentifierExpr()`：变量引用、函数调用、print

identifier 可能有三种含义：

```toy
a
foo(a, b)
print(a)
```

Parser 先读出名字：

```cpp
std::string name(lexer.getId());
```

如果后面不是 `(`，就是变量引用：

```cpp
return std::make_unique<VariableExprAST>(std::move(loc), name);
```

如果后面是 `(`，就是调用。Parser 会解析参数列表。

然后有一个特殊分支：

```cpp
if (name == "print") {
  if (args.size() != 1)
    return parseError<ExprAST>("<single arg>", "as argument to print()");

  return std::make_unique<PrintExprAST>(std::move(loc), std::move(args[0]));
}
```

所以：

- `print(a)` 变成 `PrintExprAST`。
- `transpose(a)` 变成 `CallExprAST`。
- `multiply_transpose(a, b)` 变成 `CallExprAST`。

这个点很重要，第二节课练习已经用到过。

## `parseTensorLiteralExpr()`：tensor literal

tensor literal 语法：

```text
tensorLiteral ::= [ literalList ] | number
literalList ::= tensorLiteral | tensorLiteral, literalList
```

例子：

```toy
[1, 2, 3]
[[1, 2, 3], [4, 5, 6]]
```

Parser 会递归处理嵌套数组。

一维 literal：

```toy
[1, 2, 3]
```

解析后：

```text
dims = [3]
values = [NumberExprAST(1), NumberExprAST(2), NumberExprAST(3)]
```

二维 literal：

```toy
[[1, 2, 3], [4, 5, 6]]
```

解析后：

```text
dims = [2, 3]
values = [
  LiteralExprAST(dims=[3], values=[1, 2, 3]),
  LiteralExprAST(dims=[3], values=[4, 5, 6])
]
```

Parser 会检查嵌套维度是否一致。比如：

```toy
[[1, 2], [3, 4, 5]]
```

左右两行长度不一致，会触发：

```text
expected 'uniform well-nested dimensions'
```

这是 Ch1 少数会做的结构检查之一。它不是完整语义检查，而是为了让 literal 自身成为一个规则的 tensor。

## `parseBinOpRHS()`：二元表达式和优先级

Toy 支持：

```toy
a + b
a - b
a * b
```

优先级定义在：

```cpp
int getTokPrecedence()
```

规则是：

```cpp
case '-':
  return 20;
case '+':
  return 20;
case '*':
  return 40;
```

所以 `*` 比 `+` 和 `-` 优先级更高。

例如：

```toy
a + b * c
```

应该解析为：

```text
a + (b * c)
```

而不是：

```text
(a + b) * c
```

`parseBinOpRHS()` 使用 precedence climbing 方式处理这个问题。

简化理解：

1. 先有一个左侧表达式 `lhs`。
2. 看当前 token 是否是二元操作符。
3. 如果是，就解析右侧 primary。
4. 如果右侧后面还有更高优先级的操作符，就先把右侧那部分解析完。
5. 最后构造 `BinaryExprAST`。

这部分是 Parser 中最绕的代码之一。第一次读不需要完全背下来，只要确认 `a + b * c` 会构造成正确优先级的 AST 即可。

## `parseReturn()`：返回语句

返回语法：

```text
return ::= return ; | return expr ;
```

所以 Toy 支持：

```toy
return;
```

也支持：

```toy
return x + y;
```

实现中返回表达式是 optional：

```cpp
std::optional<std::unique_ptr<ExprAST>> expr;
```

如果 `return` 后面立刻是 `;`，就是空返回。

## 错误处理：`parseError()`

Parser 遇到语法错误时调用：

```cpp
parseError<R>(expected, context)
```

它会打印：

```text
Parse error (line, col): expected '...' ... but has Token ...
```

例如缺少分号、括号不闭合、literal 不规则，都会走这里。

`empty.toy` 的测试就验证了错误路径：

```text
mlir/test/Examples/Toy/Ch1/empty.toy
```

测试期望看到：

```text
Parse error
```

这说明空文件不会触发断言崩溃，而是正常走 parse error。

## AST dump 的职责

文件：

```text
mlir/examples/toy/Ch1/parser/AST.cpp
```

AST dump 的公共入口是：

```cpp
void dump(ModuleAST &module) { ASTDumper().dump(&module); }
```

`ASTDumper` 会递归遍历 AST。

它为每种节点定义一个 `dump()`：

```cpp
void dump(ModuleAST *node);
void dump(FunctionAST *node);
void dump(PrototypeAST *node);
void dump(ExprASTList *exprList);
void dump(VarDeclExprAST *varDecl);
void dump(ReturnExprAST *node);
void dump(BinaryExprAST *node);
void dump(CallExprAST *node);
void dump(PrintExprAST *node);
```

通用表达式分发依赖 `llvm::TypeSwitch`：

```cpp
llvm::TypeSwitch<ExprAST *>(expr)
    .Case<BinaryExprAST, CallExprAST, LiteralExprAST, NumberExprAST,
          PrintExprAST, ReturnExprAST, VarDeclExprAST, VariableExprAST>(
        [&](auto *node) { this->dump(node); })
```

这说明 `ExprAST` 是基类，真正打印时要根据运行时具体类型分发到对应 overload。

## 缩进是怎么来的

AST dump 用 `curIndent` 记录当前缩进级别：

```cpp
int curIndent = 0;
```

`Indent` 是一个 RAII 小工具：

```cpp
struct Indent {
  Indent(int &level) : level(level) { ++level; }
  ~Indent() { --level; }
  int &level;
};
```

宏：

```cpp
#define INDENT() \
  Indent level_(curIndent); \
  indent();
```

含义是：

- 进入某个节点时缩进加一。
- 当前函数结束时对象析构，缩进自动减一。

所以 AST dump 的缩进反映了 AST 的父子关系。

## 从例子追踪完整链路

以这个程序为例：

```toy
def main() {
  var a = [1, 2, 3];
  print(a);
}
```

### 1. Lexer 产生 token

大致 token 流：

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
','
tok_number(3)
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

### 2. Parser 构造 AST

调用链大致是：

```text
parseModule()
  parseDefinition()
    parsePrototype()
    parseBlock()
      parseDeclaration()
        parseExpression()
          parseTensorLiteralExpr()
      parseExpression()
        parseIdentifierExpr()
          PrintExprAST
```

### 3. AST 结构

```text
ModuleAST
  FunctionAST
    PrototypeAST("main", [])
    ExprASTList
      VarDeclExprAST("a")
        LiteralExprAST(dims=[3])
      PrintExprAST
        VariableExprAST("a")
```

### 4. AST dump 输出

大致输出：

```text
Module:
  Function
    Proto 'main' @...
    Params: []
    Block {
      VarDecl a<> @...
        Literal: <3>[ 1.000000e+00, 2.000000e+00, 3.000000e+00] @...
      Print [ @...
        var: a @...
      ]
    } // Block
```

## 本节常见误区

### 误区 1：Lexer 知道 `print` 是内建函数

不对。

Lexer 只把 `print` 识别为：

```text
tok_identifier
```

真正把 `print(...)` 特殊处理为 `PrintExprAST` 的是 Parser 的 `parseIdentifierExpr()`。

### 误区 2：Parser 会检查变量是否声明

不对。

Parser 只检查语法结构，不做符号解析。比如：

```toy
def main() {
  print(x);
}
```

Ch1 可以生成 AST，因为语法上 `x` 是合法变量引用。

### 误区 3：所有语义错误都留给后面

也不完全对。

Parser 会做一些必要的结构检查，比如 tensor literal 必须是规则嵌套：

```toy
[[1, 2], [3, 4, 5]]
```

这个会在 Ch1 阶段报错，因为 AST 需要一个一致的 literal shape。

### 误区 4：AST dump 是 Parser 的输出文本

Parser 的输出是 C++ AST 对象，不是文本。

AST dump 是之后调用：

```cpp
dump(*moduleAST)
```

由 `ASTDumper` 把 AST 对象递归打印成文本。

## 动手运行

运行标准 AST 测试：

```bash
<build-dir>/bin/toyc-ch1 mlir/test/Examples/Toy/Ch1/ast.toy -emit=ast
```

观察：

- `def` 如何变成 `Function`。
- 参数如何出现在 `Params: [...]`。
- `return` 如何出现在 `Return` 下。
- `transpose(a) * transpose(b)` 如何变成 `BinOp: *`。
- `var b<2, 3>` 如何保留 shape。

运行空文件测试：

```bash
<build-dir>/bin/toyc-ch1 mlir/test/Examples/Toy/Ch1/empty.toy -emit=ast
```

观察：

- 是否打印 `Parse error`。
- 程序是否正常退出，而不是断言崩溃。

如果你想观察语义错误边界，可以创建：

```toy
def main() {
  print(x);
}
```

Ch1 应该能得到 AST，因为 `x` 是语法合法的变量引用。

## 本节练习

### 练习 1：写出完整调用链

用自己的话写出从 `main()` 到 AST dump 的调用链。至少包含：

```text
main
parseInputFile
MemoryBuffer
LexerBuffer
Parser
parseModule
dump
ASTDumper
```

要求说明每一步负责什么。

### 练习 2：手写 token 流

对下面程序手写 token 流：

```toy
def main() {
  var a = [1, 2];
  print(a);
}
```

要求区分：

- 关键字 token，例如 `tok_def`、`tok_var`。
- 标识符 token，例如 `main`、`a`、`print`。
- 数字 token。
- 标点符号 token。

### 练习 3：追踪 Parser 调用

对下面语句写出大致 Parser 调用链：

```toy
var b<2, 3> = [1, 2, 3, 4, 5, 6];
```

至少包含：

```text
parseBlock
parseDeclaration
parseType
parseExpression
parseTensorLiteralExpr
```

并说明最终构造哪个 AST 节点。

### 练习 4：二元表达式优先级

写一个 Toy 程序：

```toy
def main() {
  print(1 + 2 * 3);
}
```

运行 `toyc-ch1 -emit=ast`，观察 AST dump。

回答：

- 顶层 `BinOp` 是 `+` 还是 `*`。
- 为什么 `*` 会在更深一层。
- 这和 `getTokPrecedence()` 中的优先级数字有什么关系。

### 练习 5：构造一个 Parser 错误

写一个故意缺少分号的程序：

```toy
def main() {
  print(1)
}
```

运行：

```bash
<build-dir>/bin/toyc-ch1 your-file.toy -emit=ast
```

记录：

- 报错信息。
- 报错行列。
- Parser 期望什么 token。
- 当前拿到的 token 是什么。

### 练习 6：不规则 tensor literal

写一个不规则 literal：

```toy
def main() {
  var a = [[1, 2], [3, 4, 5]];
}
```

观察是否报错，并解释为什么这个错误属于 Parser 阶段能发现的结构错误。

## 本节小结

本节最重要的是建立 Ch1 的前端调用链：

```text
toyc.cpp
  parseInputFile()
    MemoryBuffer
    LexerBuffer
    Parser
      parseModule()
        parseDefinition()
          parsePrototype()
          parseBlock()
            parseDeclaration()
            parseReturn()
            parseExpression()
              parsePrimary()
              parseBinOpRHS()
    ModuleAST
  dump(*moduleAST)
    ASTDumper
```

也要记住几个职责边界：

- Lexer 负责字符到 token。
- Parser 负责 token 到 AST。
- ASTDumper 负责 AST 到文本 dump。
- Parser 不做完整语义检查。
- `print` 的特殊处理发生在 Parser，不发生在 Lexer。
- 二元表达式优先级由 `getTokPrecedence()` 和 `parseBinOpRHS()` 处理。

下一节课会离开 Ch1，进入 MLIR 的基本概念。到那时，AST 不再只是被 dump，而是会通过 Ch2 的 `MLIRGen.cpp` 生成 Toy Dialect MLIR。

## 学习记录模板

```text
本节主题：Lexer、Parser 与 AST dump
我读过的源码：
我观察过的测试：
我运行过的命令：
我确认理解的调用链：
我还不理解的问题：
下一步要验证的小实验：
```
