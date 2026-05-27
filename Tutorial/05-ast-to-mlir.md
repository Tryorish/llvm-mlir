# 第 5 课：从 AST 生成 MLIR

## 本节定位

第四节课已经学习了 MLIR 的核心概念：operation、value、type、attribute、region、block、dialect。

本节进入 Ch2 的核心实现：`MLIRGen.cpp`。目标是沿着 Toy AST 节点逐个看它们如何变成 Toy Dialect MLIR。

本节主线是：

```text
Toy 源码
  -> Lexer / Parser
  -> ModuleAST
  -> MLIRGenImpl
  -> ModuleOp
  -> toy.func / toy.constant / toy.reshape / toy.transpose / toy.mul / toy.generic_call / toy.print / toy.return
```

这一节会比第四节更贴近源码，但仍然只讲 Ch2 的 AST 到 MLIR 生成。ODS/TableGen 如何定义 `toy.constant`、`toy.func` 等 operation，会在第 6 课专门展开。

## 本节目标

- 理解 Ch2 `toyc.cpp` 如何进入 MLIR 生成流程。
- 理解 `MLIRContext`、`ToyDialect`、`ModuleOp`、`OpBuilder` 的基本职责。
- 理解 `MLIRGenImpl` 的整体结构。
- 理解 Toy AST 节点到 Toy MLIR operation 的映射。
- 理解 symbol table 如何把 Toy 变量名映射到 MLIR SSA value。
- 理解 source `Location` 如何转换为 MLIR location。
- 理解 `mlir::verify(theModule)` 在 Ch2 中的作用。
- 能从一段 Toy 源码推导大致会生成哪些 Toy MLIR operation。

## 本节关键文件

源码：

```text
mlir/examples/toy/Ch2/toyc.cpp
mlir/examples/toy/Ch2/mlir/MLIRGen.cpp
mlir/examples/toy/Ch2/include/toy/MLIRGen.h
mlir/examples/toy/Ch2/include/toy/Ops.td
mlir/examples/toy/Ch2/mlir/Dialect.cpp
```

测试：

```text
mlir/test/Examples/Toy/Ch2/codegen.toy
mlir/test/Examples/Toy/Ch2/scalar.toy
mlir/test/Examples/Toy/Ch2/invalid.mlir
```

建议阅读顺序：

1. 先读 `mlir/test/Examples/Toy/Ch2/codegen.toy`，明确输入和期望输出。
2. 再读 `mlir/examples/toy/Ch2/toyc.cpp`，看 `-emit=mlir` 如何进入 `mlirGen()`。
3. 再读 `mlir/examples/toy/Ch2/mlir/MLIRGen.cpp`，按 AST 节点逐个看生成逻辑。
4. 最后粗略看 `Ops.td`，知道每个 `builder.create<...>` 创建的是哪个 Toy operation。

## Ch2 `toyc.cpp` 的入口变化

Ch1 的 `toyc.cpp` 只支持：

```text
-emit=ast
```

Ch2 多了：

```text
-emit=mlir
```

对应枚举：

```cpp
enum Action { None, DumpAST, DumpMLIR };
```

Ch2 还增加了输入类型：

```cpp
enum InputType { Toy, MLIR };
```

这意味着 Ch2 的工具既可以：

- 读取 `.toy` 源文件，先 parse 成 AST，再生成 MLIR。
- 读取 `.mlir` 文件，直接用 MLIR parser 读入。

本节主要关注 `.toy -> AST -> MLIR` 这条路径。

## `dumpMLIR()` 调用链

`toyc.cpp` 中的 `dumpMLIR()` 是 Ch2 进入 MLIR 的入口。

核心流程：

```text
dumpMLIR()
  -> mlir::MLIRContext context
  -> context.getOrLoadDialect<mlir::toy::ToyDialect>()
  -> parseInputFile(inputFilename)
  -> mlirGen(context, *moduleAST)
  -> module->dump()
```

对应源码结构：

```cpp
mlir::MLIRContext context;
context.getOrLoadDialect<mlir::toy::ToyDialect>();

auto moduleAST = parseInputFile(inputFilename);
mlir::OwningOpRef<mlir::ModuleOp> module = mlirGen(context, *moduleAST);

module->dump();
```

几个重点：

- `MLIRContext` 是 MLIR 全局上下文，持有 dialect、type、attribute 等 uniqued 数据。
- `ToyDialect` 必须加载，否则 MLIR 不知道 `toy.*` operations 的注册信息。
- `mlirGen()` 是从 AST 到 MLIR 的核心 API。
- `module->dump()` 打印生成后的 MLIR。

## `mlirGen()` 公共 API

文件：

```text
mlir/examples/toy/Ch2/mlir/MLIRGen.cpp
```

底部有公共入口：

```cpp
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          ModuleAST &moduleAST) {
  return MLIRGenImpl(context).mlirGen(moduleAST);
}
```

这说明真正实现被封装在：

```cpp
class MLIRGenImpl
```

`MLIRGenImpl` 是一个临时对象，负责把一个 `ModuleAST` 转换成一个 `mlir::ModuleOp`。

## `MLIRGenImpl` 的关键成员

`MLIRGenImpl` 里有三个核心成员：

```cpp
mlir::ModuleOp theModule;
mlir::OpBuilder builder;
llvm::ScopedHashTable<StringRef, mlir::Value> symbolTable;
```

### `theModule`

`theModule` 是生成中的 MLIR module。

Toy 的一个 `.toy` 文件对应一个：

```text
mlir::ModuleOp
```

里面会放多个 `toy.func`。

### `builder`

`OpBuilder` 是创建 MLIR operation 的工具。

它有一个非常重要的状态：insertion point。

也就是说，builder 不只是“能创建 operation”，它还知道“下一个 operation 应该插入到哪里”。

例如：

```cpp
builder.setInsertionPointToEnd(theModule.getBody());
```

表示接下来创建的 operation 插入 module body 末尾。

又例如：

```cpp
builder.setInsertionPointToStart(&entryBlock);
```

表示接下来创建的 operation 插入函数入口 block 开头。

### `symbolTable`

Toy 源码中有变量名：

```toy
var a = [1, 2];
print(a);
```

MLIR 中使用 SSA value：

```mlir
%0 = toy.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
toy.print %0 : tensor<2xf64>
```

`symbolTable` 负责把 Toy 变量名映射到 MLIR value：

```text
"a" -> %0
```

它的类型是：

```cpp
llvm::ScopedHashTable<StringRef, mlir::Value>
```

这个结构支持作用域。进入函数体时创建一个 scope，离开时销毁，局部变量映射也随之消失。

## Location 转换

Ch1 AST 中每个节点都有 `toy::Location`：

```cpp
file / line / col
```

MLIR operation 也必须有 location。`MLIRGen.cpp` 用这个函数转换：

```cpp
mlir::Location loc(const Location &loc) {
  return mlir::FileLineColLoc::get(builder.getStringAttr(*loc.file),
                                   loc.line, loc.col);
}
```

所以 AST 节点来源位置会被保留到 MLIR operation 上。

默认 dump 可能不打印 location，但 verifier 和诊断会用到它。

## 从 `ModuleAST` 生成 `ModuleOp`

入口：

```cpp
mlir::ModuleOp mlirGen(ModuleAST &moduleAST)
```

核心逻辑：

```cpp
theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

for (FunctionAST &f : moduleAST)
  mlirGen(f);

if (failed(mlir::verify(theModule))) {
  theModule.emitError("module verification error");
  return nullptr;
}

return theModule;
```

对应关系：

```text
ModuleAST
  -> mlir::ModuleOp
```

每个 `FunctionAST` 会生成一个 `toy.func`，插入到 `ModuleOp` 中。

最后调用：

```cpp
mlir::verify(theModule)
```

这一步会检查 MLIR 结构是否合法，并调用 Toy operation 上注册的 verifier。

## 从 `PrototypeAST` 生成 `toy.func` 声明部分

函数原型：

```toy
def multiply_transpose(a, b)
```

对应 AST：

```text
PrototypeAST
  name = "multiply_transpose"
  args = ["a", "b"]
```

生成逻辑：

```cpp
mlir::toy::FuncOp mlirGen(PrototypeAST &proto)
```

关键代码：

```cpp
SmallVector<mlir::Type, 4> argTypes(proto.getArgs().size(),
                                    getType(VarType{}));
auto funcType = builder.getFunctionType(argTypes, {});
return builder.create<mlir::toy::FuncOp>(location, proto.getName(), funcType);
```

这里有几个重点：

- Toy 函数参数没有显式 type。
- `VarType{}` 的 shape 是空的。
- `getType(VarType{})` 会生成 unranked tensor type。
- 因此参数类型是 `tensor<*xf64>`。
- 初始返回类型为空，后面看到 `return expr` 时再补。

所以：

```toy
def multiply_transpose(a, b)
```

会生成：

```mlir
toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>)
```

如果函数体里有带值 return，后续会把函数类型改成返回：

```mlir
-> tensor<*xf64>
```

## 从 `FunctionAST` 生成完整 `toy.func`

函数生成入口：

```cpp
mlir::toy::FuncOp mlirGen(FunctionAST &funcAST)
```

核心流程：

```text
创建 symbolTable scope
设置插入点到 module body 末尾
根据 PrototypeAST 创建 toy.func
取得函数 entry block
把函数参数声明进 symbolTable
设置插入点到 entry block 开头
生成函数体
如果没有 return，补一个 toy.return
如果 return 有值，更新函数返回类型
```

### 函数参数如何进入 symbol table

函数参数在 MLIR 中是 entry block arguments。

源码逻辑：

```cpp
mlir::Block &entryBlock = function.front();
auto protoArgs = funcAST.getProto()->getArgs();

for (const auto nameValue : llvm::zip(protoArgs, entryBlock.getArguments())) {
  declare(std::get<0>(nameValue)->getName(), std::get<1>(nameValue));
}
```

含义是：

```text
Toy 参数名 a -> MLIR block argument %arg0
Toy 参数名 b -> MLIR block argument %arg1
```

所以在函数体里遇到 `a` 时，`VariableExprAST("a")` 能查到 `%arg0`。

## 生成函数体 `ExprASTList`

函数体是：

```cpp
ExprASTList
```

生成入口：

```cpp
llvm::LogicalResult mlirGen(ExprASTList &blockAST)
```

它遍历每个表达式：

```cpp
for (auto &expr : blockAST) {
  if (auto *vardecl = dyn_cast<VarDeclExprAST>(expr.get())) { ... }
  if (auto *ret = dyn_cast<ReturnExprAST>(expr.get())) { ... }
  if (auto *print = dyn_cast<PrintExprAST>(expr.get())) { ... }
  if (!mlirGen(*expr)) return failure();
}
```

这里有一个设计点：变量声明、return、print 是 block-level 语句，单独处理。普通表达式走通用表达式分发。

## 通用表达式分发

通用表达式入口：

```cpp
mlir::Value mlirGen(ExprAST &expr)
```

它根据 AST kind 分发：

```cpp
case Expr_BinOp:
  return mlirGen(BinaryExprAST);
case Expr_Var:
  return mlirGen(VariableExprAST);
case Expr_Literal:
  return mlirGen(LiteralExprAST);
case Expr_Call:
  return mlirGen(CallExprAST);
case Expr_Num:
  return mlirGen(NumberExprAST);
```

返回值是：

```cpp
mlir::Value
```

因为大部分表达式都会产生一个 MLIR SSA value。

例如：

- literal 产生 `toy.constant` 的 result。
- binary expression 产生 `toy.add` 或 `toy.mul` 的 result。
- call expression 产生 `toy.transpose` 或 `toy.generic_call` 的 result。
- variable expression 返回 symbol table 中已有的 value。

`print` 和 `return` 不走这个返回 `mlir::Value` 的通用路径，因为它们不一定产生 value。

## 变量引用：`VariableExprAST`

Toy 源码：

```toy
print(a);
```

`a` 是：

```text
VariableExprAST("a")
```

生成逻辑：

```cpp
mlir::Value mlirGen(VariableExprAST &expr) {
  if (auto variable = symbolTable.lookup(expr.getName()))
    return variable;

  emitError(loc(expr.loc()), "error: unknown variable '") << expr.getName() << "'";
  return nullptr;
}
```

也就是说，变量引用不创建新 operation。它只是查表，返回已有 SSA value。

例如：

```toy
var a = [1, 2];
print(a);
```

大致变成：

```mlir
%0 = toy.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
toy.print %0 : tensor<2xf64>
```

`print(a)` 中的 `a` 查表得到 `%0`。

如果查不到，就报未知变量错误。

## 变量声明：`VarDeclExprAST`

Toy 源码：

```toy
var a<2, 3> = [1, 2, 3, 4, 5, 6];
```

生成入口：

```cpp
mlir::Value mlirGen(VarDeclExprAST &vardecl)
```

流程：

```text
生成初始化表达式
如果变量声明有显式 shape，创建 toy.reshape
把变量名和最终 value 登记到 symbolTable
返回最终 value
```

关键代码：

```cpp
mlir::Value value = mlirGen(*init);

if (!vardecl.getType().shape.empty()) {
  value = builder.create<ReshapeOp>(loc(vardecl.loc()),
                                    getType(vardecl.getType()), value);
}

declare(vardecl.getName(), value);
```

例如：

```toy
var b<2, 3> = [1, 2, 3, 4, 5, 6];
```

右侧 literal 自然生成：

```mlir
%0 = toy.constant dense<[1.000000e+00, ..., 6.000000e+00]> : tensor<6xf64>
```

因为声明写了 `<2, 3>`，再生成：

```mlir
%1 = toy.reshape(%0 : tensor<6xf64>) to tensor<2x3xf64>
```

最后登记：

```text
"b" -> %1
```

注意：即使 literal 本身已经是 `<2, 3>`，只要变量声明写了显式 shape，Ch2 也会插入 `toy.reshape`。后续 canonicalization 可以把无意义 reshape 消掉。

## Literal：`LiteralExprAST`

Toy 源码：

```toy
[[1, 2, 3], [4, 5, 6]]
```

生成入口：

```cpp
mlir::Value mlirGen(LiteralExprAST &lit)
```

核心步骤：

1. 根据 literal dims 构造 tensor type。
2. 把嵌套 literal flatten 成 `std::vector<double>`。
3. 创建 `DenseElementsAttr` 保存常量数据。
4. 创建 `toy.constant`。

### type 如何生成

```cpp
auto type = getType(lit.getDims());
```

如果 dims 是：

```text
[2, 3]
```

得到：

```mlir
tensor<2x3xf64>
```

### 数据如何 flatten

辅助函数：

```cpp
void collectData(ExprAST &expr, std::vector<double> &data)
```

例如：

```toy
[[1, 2], [3, 4]]
```

会 flatten 成：

```text
[1, 2, 3, 4]
```

### attribute 如何创建

```cpp
auto dataType = mlir::RankedTensorType::get(lit.getDims(), elementType);
auto dataAttribute = mlir::DenseElementsAttr::get(dataType, llvm::ArrayRef(data));
```

这个 attribute 最后附到 `toy.constant` 上。

### 生成 operation

```cpp
return builder.create<ConstantOp>(loc(lit.loc()), type, dataAttribute);
```

输出类似：

```mlir
%0 = toy.constant dense<[[1.000000e+00, 2.000000e+00],
                         [3.000000e+00, 4.000000e+00]]>
     : tensor<2x2xf64>
```

## Number：`NumberExprAST`

Toy 源码：

```toy
5.5
```

生成入口：

```cpp
mlir::Value mlirGen(NumberExprAST &num)
```

实现：

```cpp
return builder.create<ConstantOp>(loc(num.loc()), num.getValue());
```

对应测试：

```text
mlir/test/Examples/Toy/Ch2/scalar.toy
```

源码：

```toy
def main() {
  var a<2, 2> = 5.5;
  print(a);
}
```

输出：

```mlir
%0 = toy.constant dense<5.500000e+00> : tensor<f64>
%1 = toy.reshape(%0 : tensor<f64>) to tensor<2x2xf64>
toy.print %1 : tensor<2x2xf64>
toy.return
```

这里 `5.5` 先是 scalar constant，随后因为变量声明写了 `<2, 2>`，所以插入 reshape。

## Binary：`BinaryExprAST`

Toy 源码：

```toy
transpose(a) * transpose(b)
```

生成入口：

```cpp
mlir::Value mlirGen(BinaryExprAST &binop)
```

流程：

1. 先递归生成 lhs。
2. 再递归生成 rhs。
3. 根据操作符创建 Toy operation。

关键代码：

```cpp
mlir::Value lhs = mlirGen(*binop.getLHS());
mlir::Value rhs = mlirGen(*binop.getRHS());

switch (binop.getOp()) {
case '+':
  return builder.create<AddOp>(location, lhs, rhs);
case '*':
  return builder.create<MulOp>(location, lhs, rhs);
}
```

所以：

```toy
a + b
```

生成：

```mlir
%0 = toy.add %a, %b : tensor<...>
```

而：

```toy
a * b
```

生成：

```mlir
%0 = toy.mul %a, %b : tensor<...>
```

注意：Ch2 的 MLIRGen 只支持 `+` 和 `*`。Parser 支持 `-` 的优先级，但 MLIRGen 里没有生成 `toy.sub`，遇到 `-` 会报 invalid binary operator。

## Call：`CallExprAST`

Toy 源码中有两类 call：

```toy
transpose(a)
multiply_transpose(a, b)
```

它们在 Ch1 AST 中都是 `CallExprAST`。

Ch2 的 MLIRGen 会区分 callee 名字。

### 内建 `transpose`

```cpp
if (callee == "transpose") {
  if (call.getArgs().size() != 1) { ... }
  return builder.create<TransposeOp>(location, operands[0]);
}
```

所以：

```toy
transpose(a)
```

生成：

```mlir
%0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
```

### 用户函数调用

如果不是 `transpose`，就认为是用户函数：

```cpp
return builder.create<GenericCallOp>(location, callee, operands);
```

所以：

```toy
multiply_transpose(a, b)
```

生成：

```mlir
%0 = toy.generic_call @multiply_transpose(%a, %b)
     : (...) -> tensor<*xf64>
```

这里 callee 名字变成 symbol reference attribute。

## Print：`PrintExprAST`

Toy 源码：

```toy
print(d);
```

生成入口：

```cpp
llvm::LogicalResult mlirGen(PrintExprAST &call)
```

流程：

1. 先生成或查找被打印的表达式。
2. 创建 `toy.print`。

代码：

```cpp
auto arg = mlirGen(*call.getArg());
builder.create<PrintOp>(loc(call.loc()), arg);
```

输出：

```mlir
toy.print %0 : tensor<*xf64>
```

`toy.print` 没有 result。

## Return：`ReturnExprAST`

Toy 源码：

```toy
return transpose(a) * transpose(b);
```

生成入口：

```cpp
llvm::LogicalResult mlirGen(ReturnExprAST &ret)
```

流程：

1. 如果 return 有表达式，先生成表达式。
2. 创建 `toy.return`，把表达式 value 作为 operand。

输出：

```mlir
toy.return %0 : tensor<*xf64>
```

如果源码中没有显式 return，`mlirGen(FunctionAST &funcAST)` 会补一个：

```cpp
builder.create<ReturnOp>(loc(funcAST.getProto()->loc()));
```

输出：

```mlir
toy.return
```

如果函数里有带值 return，MLIRGen 会更新 `toy.func` 的函数类型：

```cpp
function.setType(builder.getFunctionType(
    function.getFunctionType().getInputs(), getType(VarType{})));
```

因此 `multiply_transpose` 会显示：

```mlir
toy.func @multiply_transpose(...) -> tensor<*xf64>
```

## Type 生成：`getType()`

Ch2 的类型生成集中在：

```cpp
mlir::Type getType(ArrayRef<int64_t> shape)
```

逻辑：

```cpp
if (shape.empty())
  return mlir::UnrankedTensorType::get(builder.getF64Type());

return mlir::RankedTensorType::get(shape, builder.getF64Type());
```

对应：

| Toy shape | MLIR type |
| --- | --- |
| 空 shape | `tensor<*xf64>` |
| `[2]` | `tensor<2xf64>` |
| `[2, 3]` | `tensor<2x3xf64>` |
| scalar number | `tensor<f64>` |

Toy 只有 `f64` 元素类型，所以这里总是使用：

```cpp
builder.getF64Type()
```

## 验证：`mlir::verify(theModule)`

生成所有函数后，Ch2 会执行：

```cpp
mlir::verify(theModule)
```

这一步检查：

- MLIR 的结构是否合法。
- operation 的 operand/result/type 是否满足基本约束。
- Toy operation 自己的 verifier 是否通过。

例如：

```text
mlir/test/Examples/Toy/Ch2/invalid.mlir
```

里面故意写了非法 IR：

```mlir
toy.func @main() {
  %0 = "toy.print"()  : () -> tensor<2x3xf64>
}
```

问题包括：

- `toy.print` 不应该返回 value。
- `toy.print` 应该有输入 operand。
- 函数体缺少 terminator。

这类问题会在 MLIR parse/verify 阶段暴露。

## `codegen.toy` 完整映射

测试文件：

```text
mlir/test/Examples/Toy/Ch2/codegen.toy
```

源码：

```toy
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}
```

生成大致为：

```mlir
toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
  %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
  %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
  %2 = toy.mul %0, %1 : tensor<*xf64>
  toy.return %2 : tensor<*xf64>
}
```

映射关系：

| Toy 源码 | AST | MLIR |
| --- | --- | --- |
| `def multiply_transpose(a, b)` | `FunctionAST` + `PrototypeAST` | `toy.func @multiply_transpose(%arg0, %arg1)` |
| `a` | `VariableExprAST` | 查表得到 `%arg0` |
| `b` | `VariableExprAST` | 查表得到 `%arg1` |
| `transpose(a)` | `CallExprAST` | `toy.transpose(%arg0)` |
| `transpose(b)` | `CallExprAST` | `toy.transpose(%arg1)` |
| `... * ...` | `BinaryExprAST('*')` | `toy.mul` |
| `return ...` | `ReturnExprAST` | `toy.return` |

`main` 中：

```toy
var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
var b<2, 3> = [1, 2, 3, 4, 5, 6];
var c = multiply_transpose(a, b);
var d = multiply_transpose(b, a);
print(d);
```

生成大致为：

```mlir
toy.func @main() {
  %0 = toy.constant dense<[[...], [...]]> : tensor<2x3xf64>
  %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>
  %2 = toy.constant dense<[...]> : tensor<6xf64>
  %3 = toy.reshape(%2 : tensor<6xf64>) to tensor<2x3xf64>
  %4 = toy.generic_call @multiply_transpose(%1, %3)
       : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
  %5 = toy.generic_call @multiply_transpose(%3, %1)
       : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
  toy.print %5 : tensor<*xf64>
  toy.return
}
```

注意：

- `var c` 和 `var d` 本身不生成变量声明 operation，只是登记 symbol table。
- `print(d)` 中的 `d` 查表得到 `%5`。
- `main` 没有显式 return，所以 MLIRGen 补了 `toy.return`。

## 常见误区

### 误区 1：变量声明一定生成 operation

不一定。

`var a = expr` 的主要作用是把名字 `a` 绑定到 `expr` 生成的 SSA value。

只有在有显式 shape 时，MLIRGen 会额外插入 `toy.reshape`。

### 误区 2：变量引用会生成 load operation

Ch2 不会。

Toy 当前还在高层 tensor value 语义里。变量引用只是查 symbol table，返回已有 `mlir::Value`。

后续 lowering 到 memref 时，才会看到内存、load、store 相关概念。

### 误区 3：`transpose` 和普通函数调用一样

在 Ch1 AST 中它们都是 `CallExprAST`。

但在 Ch2 MLIRGen 中：

- `transpose(a)` 生成 `toy.transpose`。
- `multiply_transpose(a, b)` 生成 `toy.generic_call`。

### 误区 4：`toy.reshape` 一定代表真正运行时拷贝

在 Ch2 里先不要这样理解。

`toy.reshape` 表达的是高层 reshape 语义。后续优化可能会消除无意义 reshape，也可能在 lowering 时变成更具体的实现。

### 误区 5：MLIRGen 做完整类型检查

Ch2 的 MLIRGen 做了一部分基本检查，比如未知变量、无效二元操作、`transpose` 参数个数等。

但完整 shape inference 和更深入的语义检查是后续章节逐步加入的。

## 动手运行

运行 Ch2 标准 codegen 测试：

```bash
<build-dir>/bin/toyc-ch2 mlir/test/Examples/Toy/Ch2/codegen.toy -emit=mlir
```

观察：

- `multiply_transpose` 的参数如何变成 `%arg0`、`%arg1`。
- `transpose(a)` 如何变成 `toy.transpose(%arg0)`。
- `*` 如何变成 `toy.mul`。
- `var a<2, 3>` 为什么生成 `toy.constant` 和 `toy.reshape`。
- `var c` 为什么只生成 `toy.generic_call`，没有 `toy.var`。
- `main` 为什么最后有 `toy.return`。

运行 scalar 测试：

```bash
<build-dir>/bin/toyc-ch2 mlir/test/Examples/Toy/Ch2/scalar.toy -emit=mlir
```

观察：

- `5.5` 如何生成 `toy.constant dense<5.500000e+00> : tensor<f64>`。
- `var a<2, 2> = 5.5` 为什么又生成 `toy.reshape`。

## 本节练习

### 练习 1：写出 Ch2 `-emit=mlir` 调用链

从 `main()` 开始，写出 `.toy` 文件生成 MLIR 的调用链。至少包含：

```text
main
dumpMLIR
MLIRContext
ToyDialect
parseInputFile
mlirGen
MLIRGenImpl::mlirGen(ModuleAST)
module->dump
```

并说明每一步职责。

### 练习 2：AST 到 MLIR 映射表

整理一张表，包含：

```text
ModuleAST
FunctionAST
PrototypeAST
VarDeclExprAST
LiteralExprAST
NumberExprAST
VariableExprAST
BinaryExprAST('+')
BinaryExprAST('*')
CallExprAST("transpose")
普通 CallExprAST
PrintExprAST
ReturnExprAST
```

写出每个 AST 节点大致生成什么 Toy MLIR。

### 练习 3：追踪一个变量声明

对下面源码：

```toy
var b<2, 3> = [1, 2, 3, 4, 5, 6];
```

回答：

- literal 首先生成什么 operation。
- literal 的自然 type 是什么。
- 为什么会生成 `toy.reshape`。
- reshape 的结果 type 是什么。
- symbol table 中 `"b"` 最后绑定到哪个 value。

### 练习 4：追踪函数参数

对下面源码：

```toy
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}
```

回答：

- `a` 和 `b` 在 MLIR 中最初是什么。
- 它们什么时候被放入 symbol table。
- `transpose(a)` 中的 `a` 如何找到对应 MLIR value。
- 为什么函数参数 type 是 `tensor<*xf64>`。

### 练习 5：解释 `scalar.toy`

阅读：

```text
mlir/test/Examples/Toy/Ch2/scalar.toy
```

解释为什么：

```toy
var a<2, 2> = 5.5;
```

会生成：

```mlir
%0 = toy.constant dense<5.500000e+00> : tensor<f64>
%1 = toy.reshape(%0 : tensor<f64>) to tensor<2x2xf64>
```

### 练习 6：制造一个未知变量错误

创建一个 Toy 程序：

```toy
def main() {
  print(x);
}
```

运行：

```bash
<build-dir>/bin/toyc-ch2 your-file.toy -emit=mlir
```

回答：

- Ch1 为什么可以生成 AST。
- Ch2 为什么会报错。
- 报错来自 `MLIRGen.cpp` 中哪个逻辑。

### 练习 7：制造一个不支持的二元操作

创建一个 Toy 程序：

```toy
def main() {
  var a = 1 - 2;
  print(a);
}
```

运行：

```bash
<build-dir>/bin/toyc-ch2 your-file.toy -emit=mlir
```

回答：

- Parser 是否能解析 `-`。
- MLIRGen 是否支持 `-`。
- 报错来自哪里。
- 如果要支持 `-`，大概要增加什么 Toy operation。

## 本节小结

本节最重要的是建立 AST 到 Toy MLIR 的映射：

```text
ModuleAST
  -> ModuleOp

FunctionAST / PrototypeAST
  -> toy.func

LiteralExprAST / NumberExprAST
  -> toy.constant

显式 shape 的 VarDeclExprAST
  -> toy.reshape

VariableExprAST
  -> symbolTable lookup

BinaryExprAST('+')
  -> toy.add

BinaryExprAST('*')
  -> toy.mul

CallExprAST("transpose")
  -> toy.transpose

普通 CallExprAST
  -> toy.generic_call

PrintExprAST
  -> toy.print

ReturnExprAST
  -> toy.return
```

也要记住几个实现关键点：

- `OpBuilder` 负责创建 operation，并依赖 insertion point 决定插入位置。
- `symbolTable` 把 Toy 变量名映射到 MLIR SSA value。
- `Location` 从 AST 传递到 MLIR operation。
- `mlir::verify(theModule)` 会在生成后检查 IR 合法性。
- Ch2 仍然保留 Toy 的高层语义，不做 lowering。

下一节课会进入第 6 课：Toy Dialect 与 ODS/TableGen。到那时会解释 `toy.constant`、`toy.func`、`toy.reshape` 等 operation 是如何在 `Ops.td` 中被定义出来的。

## 学习记录模板

```text
本节主题：从 AST 生成 MLIR
我读过的源码：
我观察过的测试：
我运行过的命令：
我确认理解的 AST -> MLIR 映射：
我还不理解的问题：
下一步要验证的小实验：
```
