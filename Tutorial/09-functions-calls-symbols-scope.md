# 第 9 课：函数、调用、作用域与符号

## 本节定位

前几课已经讲过 Toy Dialect 的 operation 定义、类型、属性、builder、verifier 和 assembly format。

本节把这些知识放到一个更完整的程序结构里：

```text
Toy 函数定义
  -> toy.func

Toy 函数参数
  -> entry block arguments

Toy 变量名
  -> MLIRGen 中的 source-level symbol table

Toy 函数调用
  -> toy.generic_call + callee SymbolRefAttr

Toy return
  -> toy.return + toy.func function type
```

也就是说，本节重点不再是单个 operation 的字段，而是函数之间、函数内部、变量和 SSA value 之间是如何连接起来的。

## 本节目标

- 理解 `toy.func` 如何表示 Toy 函数定义。
- 理解 Toy 函数参数为什么会变成 block arguments。
- 理解 `FunctionType`、function body region、entry block 的关系。
- 理解 `SymbolNameAttr` 和 `FlatSymbolRefAttr` 的区别。
- 理解 `toy.generic_call` 如何表示用户自定义函数调用。
- 理解 MLIRGen 中的 `ScopedHashTable` 如何处理 Toy 变量作用域。
- 理解 `toy.return` 如何反过来更新 `toy.func` 的返回类型。
- 能区分源码级变量表和 MLIR 符号系统。

## 本节关键文件

源码：

```text
mlir/examples/toy/Ch2/include/toy/Ops.td
mlir/examples/toy/Ch2/mlir/MLIRGen.cpp
mlir/examples/toy/Ch2/mlir/Dialect.cpp
```

测试：

```text
mlir/test/Examples/Toy/Ch2/codegen.toy
mlir/test/Examples/Toy/Ch2/scalar.toy
```

建议阅读顺序：

1. 先看 `Ops.td` 中的 `FuncOp`、`GenericCallOp`、`ReturnOp`。
2. 再看 `MLIRGen.cpp` 中的 `mlirGen(PrototypeAST&)` 和 `mlirGen(FunctionAST&)`。
3. 然后看 `mlirGen(CallExprAST&)` 和 `mlirGen(ReturnExprAST&)`。
4. 最后对照 `codegen.toy` 的 FileCheck 输出。

## 从 Toy 函数到 `toy.func`

Toy 源程序：

```toy
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}
```

会生成类似：

```mlir
toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>)
    -> tensor<*xf64> {
  %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
  %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
  %2 = toy.mul %0, %1 :  tensor<*xf64>
  toy.return %2 : tensor<*xf64>
}
```

这里有几个关键点：

- `@multiply_transpose` 是函数 symbol name。
- `%arg0`、`%arg1` 是函数 body entry block 的 block arguments。
- `tensor<*xf64>` 表示参数 shape 暂时未知。
- `toy.return %2` 表示函数返回一个 SSA value。
- `-> tensor<*xf64>` 是函数类型的一部分，不是 `toy.func` 的 SSA result。

## `FuncOp` 的 ODS 定义

`Ops.td` 中的 `FuncOp`：

```tablegen
def FuncOp : Toy_Op<"func", [
    FunctionOpInterface, IsolatedFromAbove
  ]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
  >];

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}
```

拆开来看：

- `sym_name`：函数的 symbol 名字，例如 `@main`。
- `function_type`：函数类型，例如 `(tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>`。
- `body`：函数体 region。
- `FunctionOpInterface`：让 `toy.func` 具备函数类 operation 的通用接口。
- `IsolatedFromAbove`：函数体不能直接捕获外层 SSA value。

注意：`toy.func` 本身没有 SSA result。函数返回值写在 `function_type` 里，并由函数体中的 `toy.return` 对应。

## `FunctionType` 和函数体 region

一个函数 operation 同时有两部分信息：

```text
函数签名：
  function_type

函数实现：
  body region
```

例如：

```mlir
toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>)
    -> tensor<*xf64> {
  ...
}
```

它的函数类型可以理解成：

```text
(tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
```

函数体 region 中至少有一个 block。这个 block 的 arguments 就对应函数参数：

```text
%arg0: tensor<*xf64>
%arg1: tensor<*xf64>
```

这些参数不是由 `toy.constant` 或其他 operation 产生的，而是 block arguments。

## entry block arguments

在 MLIRGen 中，`FuncOp` 的 builder 会创建 entry block：

```cpp
void FuncOp::build(OpBuilder &builder, OperationState &state,
                   StringRef name, FunctionType type,
                   ArrayRef<NamedAttribute> attrs) {
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}
```

这里的关键是：

```cpp
type.getInputs()
```

函数类型的 input types 会变成 entry block arguments 的 types。

所以当 Toy 函数有两个参数：

```toy
def multiply_transpose(a, b) { ... }
```

MLIRGen 先构造一个函数类型：

```cpp
SmallVector<Type, 4> argTypes(proto.getArgs().size(), getType(VarType{}));
auto funcType = builder.getFunctionType(argTypes, {});
```

其中 `getType(VarType{})` 对应 unranked f64 tensor：

```mlir
tensor<*xf64>
```

然后 entry block 里会出现两个 block arguments。

## 为什么 Toy 函数参数先是 unranked

Ch2 还没有真正做 shape inference。

因此 Toy 函数参数统一先用：

```mlir
tensor<*xf64>
```

这表示：

- 元素类型是 `f64`。
- 但 rank 和 shape 暂时未知。

这对泛型函数很有用。例如：

```toy
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}
```

这个函数可以被不同 shape 的 tensor 调用。Ch2 先保留成高层、泛型的 `toy.func` 和 `toy.generic_call`，后面课程会继续处理 shape 推导、内联和 lowering。

## MLIRGen 如何创建函数

`MLIRGen.cpp` 中的函数原型生成：

```cpp
mlir::toy::FuncOp mlirGen(PrototypeAST &proto) {
  auto location = loc(proto.loc());

  SmallVector<Type, 4> argTypes(proto.getArgs().size(),
                                getType(VarType{}));
  auto funcType = builder.getFunctionType(argTypes, {});
  return builder.create<mlir::toy::FuncOp>(location, proto.getName(),
                                           funcType);
}
```

它做了三件事：

1. 根据参数数量创建一组 unranked tensor 参数类型。
2. 创建一个暂时没有返回值的 `FunctionType`。
3. 创建 `toy.func`。

为什么一开始没有返回值？

因为 Ch2 的 MLIRGen 是在生成函数体时看到 `return` 以后，才知道这个 Toy 函数是否返回值。

## `mlirGen(FunctionAST&)` 的流程

核心流程可以概括为：

```text
进入函数
  -> 创建变量作用域
  -> 在 module body 末尾创建 toy.func
  -> 取得 entry block
  -> 把 Toy 参数名绑定到 entry block arguments
  -> 设置 builder insertion point 到函数体开头
  -> 生成函数体 expression list
  -> 如果没有 return，补 toy.return
  -> 如果 return 有 operand，更新 function type 的 result
```

对应源码：

```cpp
ScopedHashTableScope<StringRef, Value> varScope(symbolTable);

builder.setInsertionPointToEnd(theModule.getBody());
FuncOp function = mlirGen(*funcAST.getProto());

Block &entryBlock = function.front();
auto protoArgs = funcAST.getProto()->getArgs();

for (const auto nameValue : llvm::zip(protoArgs, entryBlock.getArguments())) {
  declare(std::get<0>(nameValue)->getName(), std::get<1>(nameValue));
}

builder.setInsertionPointToStart(&entryBlock);
mlirGen(*funcAST.getBody());
```

这段代码是理解“Toy 源码变量名”和“MLIR SSA value”之间关系的关键。

## Toy 变量名如何绑定到 SSA value

MLIRGen 里有一个成员：

```cpp
llvm::ScopedHashTable<StringRef, mlir::Value> symbolTable;
```

它保存的是：

```text
Toy 源码变量名 -> MLIR SSA value
```

例如 Toy：

```toy
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}
```

进入函数后，MLIRGen 会把：

```text
a -> %arg0
b -> %arg1
```

插入 `symbolTable`。

因此后面生成 `transpose(a)` 时：

```cpp
mlir::Value mlirGen(VariableExprAST &expr) {
  if (auto variable = symbolTable.lookup(expr.getName()))
    return variable;

  emitError(..., "unknown variable");
  return nullptr;
}
```

`a` 会被查到，然后返回 `%arg0` 这个 MLIR value。

## 这里的 `symbolTable` 不是 MLIR 符号表

这一点很容易混淆。

`MLIRGen.cpp` 里的：

```cpp
llvm::ScopedHashTable<StringRef, mlir::Value> symbolTable;
```

是 Toy 前端自己维护的源码变量表。

它负责：

- `a` 对应哪个 block argument。
- `var c = ...` 里的 `c` 对应哪个 SSA value。
- 离开作用域时变量名失效。

而 MLIR 里的 symbol 是另一件事，典型例子是：

```mlir
toy.func @multiply_transpose(...)
%0 = toy.generic_call @multiply_transpose(%a, %b) : ...
```

这里的 `@multiply_transpose` 才是 MLIR symbol / symbol reference。

## 变量声明和局部作用域

Toy 变量声明：

```toy
var c = multiply_transpose(a, b);
```

会先生成右侧表达式：

```cpp
Value value = mlirGen(*init);
```

如果变量声明带 shape：

```toy
var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
```

还会插入 `toy.reshape`：

```cpp
if (!vardecl.getType().shape.empty()) {
  value = builder.create<ReshapeOp>(loc(vardecl.loc()),
                                    getType(vardecl.getType()), value);
}
```

最后把变量名绑定到这个 value：

```cpp
declare(vardecl.getName(), value);
```

所以：

```toy
var c = multiply_transpose(a, b);
print(c);
```

大致对应：

```text
c -> %call_result
print(c) -> toy.print %call_result
```

## `ScopedHashTableScope`

MLIRGen 进入函数或表达式块时，会创建一个 scope：

```cpp
ScopedHashTableScope<StringRef, Value> varScope(symbolTable);
```

它的作用是：

- 进入作用域时，后续声明进入新 scope。
- 离开作用域时，这个 scope 里声明的名字自动失效。

这符合 Toy 的源码作用域模型。

本节先重点看函数级作用域：

```cpp
mlir::toy::FuncOp mlirGen(FunctionAST &funcAST) {
  ScopedHashTableScope<StringRef, Value> varScope(symbolTable);
  ...
}
```

这样一个函数里的参数和局部变量，不会污染另一个函数。

## `declare()` 如何防止重复定义

`declare()` 的实现：

```cpp
LogicalResult declare(StringRef var, Value value) {
  if (symbolTable.count(var))
    return failure();
  symbolTable.insert(var, value);
  return success();
}
```

它检查当前可见范围内是否已有同名变量。

如果已经有同名变量，就返回 failure。

这说明 MLIRGen 不只是机械翻译 AST，它也做了一点源码级语义检查：

- 未定义变量会报错。
- 重复声明变量会失败。

但注意，Ch2 这里的语义检查仍然很有限。

## 从 Toy 调用到 `toy.generic_call`

Toy 源程序：

```toy
var c = multiply_transpose(a, b);
```

会生成：

```mlir
%0 = toy.generic_call @multiply_transpose(%a, %b)
     : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
```

`GenericCallOp` 的 ODS 定义：

```tablegen
def GenericCallOp : Toy_Op<"generic_call"> {
  let arguments = (ins FlatSymbolRefAttr:$callee,
                       Variadic<F64Tensor>:$inputs);
  let results = (outs F64Tensor);

  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

  let builders = [
    OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments)>
  ];
}
```

它有两类输入：

- `callee`：`FlatSymbolRefAttr`，例如 `@multiply_transpose`。
- `inputs`：真正的 SSA operands，例如 `%6`、`%8`。

这正好对应函数调用的两个核心部分：

```text
调用谁？
  -> callee symbol attribute

用什么参数调用？
  -> SSA operands
```

## `GenericCallOp::build()`

`Dialect.cpp` 中：

```cpp
void GenericCallOp::build(OpBuilder &builder, OperationState &state,
                          StringRef callee, ArrayRef<Value> arguments) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(arguments);
  state.addAttribute("callee",
                     SymbolRefAttr::get(builder.getContext(), callee));
}
```

它做了三件事：

1. 调用结果先设为 `tensor<*xf64>`。
2. 把实参 value 加到 operands。
3. 把 callee 字符串变成 `SymbolRefAttr`。

这说明前端在 C++ 里可以传自然的字符串：

```cpp
builder.create<GenericCallOp>(location, callee, operands);
```

builder 会把它包装成 MLIR 需要的 symbol reference attribute。

## 用户函数调用和 builtin 调用

`mlirGen(CallExprAST&)` 中有一个重要分支：

```cpp
if (callee == "transpose") {
  if (call.getArgs().size() != 1)
    ...
  return builder.create<TransposeOp>(location, operands[0]);
}

return builder.create<GenericCallOp>(location, callee, operands);
```

这说明 Toy 中的调用分为两类：

```text
transpose(x)
  -> builtin
  -> toy.transpose

foo(x, y)
  -> 用户自定义函数
  -> toy.generic_call @foo(...)
```

`print(x)` 也属于 builtin，但它在 AST 中走的是 `PrintExprAST`，生成的是：

```mlir
toy.print %x : tensor<...>
```

## `SymbolNameAttr` 和 `FlatSymbolRefAttr`

这两个 attribute 很重要。

### `SymbolNameAttr`

出现在 `FuncOp`：

```tablegen
SymbolNameAttr:$sym_name
```

表示这个 operation 定义了一个 symbol。

在文本里就是：

```mlir
toy.func @main() { ... }
toy.func @multiply_transpose(...) { ... }
```

这里的 `@main` 和 `@multiply_transpose` 是被定义的 symbol 名字。

### `FlatSymbolRefAttr`

出现在 `GenericCallOp`：

```tablegen
FlatSymbolRefAttr:$callee
```

表示这个 operation 引用了一个 symbol。

在文本里就是：

```mlir
toy.generic_call @multiply_transpose(%0, %1) : ...
```

这里的 `@multiply_transpose` 是引用，不是定义。

## symbol 和 SSA value 的区别

看这行：

```mlir
%9 = toy.generic_call @multiply_transpose(%6, %8)
     : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
```

里面有三种名字：

```text
%9
  -> operation result，是 SSA value

@multiply_transpose
  -> callee symbol reference，是 attribute

%6, %8
  -> call operands，是 SSA value
```

不要把 `@` 和 `%` 混在一起。

在 MLIR 中：

- `%` 通常表示 SSA value。
- `@` 通常表示 symbol。

## `toy.return` 和函数返回类型

`ReturnOp` 的 ODS 定义：

```tablegen
def ReturnOp : Toy_Op<"return", [Pure, HasParent<"FuncOp">,
                                 Terminator]> {
  let arguments = (ins Variadic<F64Tensor>:$input);
  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";
  let hasVerifier = 1;
}
```

它有两个关键性质：

- `HasParent<"FuncOp">`：只能放在 `toy.func` 内。
- `Terminator`：结束当前 block。

在 MLIRGen 中：

```cpp
builder.create<ReturnOp>(location,
                         expr ? ArrayRef(expr) : ArrayRef<Value>());
```

如果 Toy 源码写：

```toy
return transpose(a) * transpose(b);
```

就会生成带 operand 的 `toy.return`：

```mlir
toy.return %2 : tensor<*xf64>
```

如果函数没有显式返回值，MLIRGen 会补一个空返回：

```mlir
toy.return
```

## 返回类型是如何推导的

`mlirGen(FunctionAST&)` 中，函数一开始创建为没有返回值：

```cpp
auto funcType = builder.getFunctionType(argTypes, {});
```

生成完函数体后，MLIRGen 检查最后一个 operation：

```cpp
ReturnOp returnOp;
if (!entryBlock.empty())
  returnOp = dyn_cast<ReturnOp>(entryBlock.back());

if (!returnOp) {
  builder.create<ReturnOp>(loc(funcAST.getProto()->loc()));
} else if (returnOp.hasOperand()) {
  function.setType(builder.getFunctionType(
      function.getFunctionType().getInputs(), getType(VarType{})));
}
```

这里有两个逻辑：

1. 如果没有 `toy.return`，自动补一个空 `toy.return`。
2. 如果 `toy.return` 有 operand，就把函数类型改成返回 `tensor<*xf64>`。

因此：

```toy
def main() {
  print(d);
}
```

生成：

```mlir
toy.func @main() {
  ...
  toy.print %d : tensor<*xf64>
  toy.return
}
```

而：

```toy
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}
```

生成：

```mlir
toy.func @multiply_transpose(...) -> tensor<*xf64> {
  ...
  toy.return %2 : tensor<*xf64>
}
```

## `ReturnOp::verify()` 再回顾

第 8 课已经讲过 verifier。放到函数上下文里，它的意义更清楚。

`ReturnOp::verify()` 会检查：

- `toy.return` 最多一个 operand。
- operand 数量必须和 enclosing `toy.func` 的 result 数量一致。
- 如果有 operand，operand type 必须和函数返回类型一致，除非其中一边是 unranked tensor。

这保证了：

```text
函数签名
  和
函数体里的 return
```

不会互相矛盾。

## `IsolatedFromAbove`

`FuncOp` 有：

```tablegen
FunctionOpInterface, IsolatedFromAbove
```

`IsolatedFromAbove` 的直观含义是：

```text
函数体不能直接使用函数外部定义的 SSA value
```

这对函数非常重要。

错误直觉是：

```mlir
%outside = ...
toy.func @foo() {
  toy.print %outside : tensor<...>
  toy.return
}
```

这种跨 region 捕获外部 SSA value 的行为不符合函数的隔离模型。

如果函数需要值，应该通过函数参数传进来：

```mlir
toy.func @foo(%arg0: tensor<*xf64>) {
  toy.print %arg0 : tensor<*xf64>
  toy.return
}
```

## `FunctionOpInterface`

`FuncOp` 还实现了 `FunctionOpInterface`。

`Ops.td` 中补了几个方法：

```cpp
ArrayRef<Type> getArgumentTypes() {
  return getFunctionType().getInputs();
}

ArrayRef<Type> getResultTypes() {
  return getFunctionType().getResults();
}

Region *getCallableRegion() {
  return &getBody();
}
```

这些方法让通用 MLIR 基础设施可以把 `toy.func` 当作函数类 operation 来处理。

例如：

- 读取参数类型。
- 读取返回类型。
- 找到函数 body region。
- 使用函数解析和打印工具。

## `FuncOp` 的 parse / print

`Dialect.cpp` 中：

```cpp
ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes,
         ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) {
        return builder.getFunctionType(argTypes, results);
      };

  return function_interface_impl::parseFunctionOp(...);
}

void FuncOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(...);
}
```

也就是说，`toy.func` 的文本格式虽然是 Toy 自定义 operation，但它复用了 MLIR 函数接口的通用解析和打印工具。

这也是 `FunctionOpInterface` 的实际价值之一。

## 对照 `codegen.toy`

测试文件：

```text
mlir/test/Examples/Toy/Ch2/codegen.toy
```

Toy 源码：

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

重点观察生成的四类结构：

### 函数定义

```mlir
toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>)
    -> tensor<*xf64> {
```

这说明：

- Toy 函数名变成 MLIR symbol。
- Toy 参数变成 block arguments。
- 返回类型在看到 `toy.return` 后被补成 `tensor<*xf64>`。

### builtin 调用

```mlir
%0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
```

这说明：

- `transpose(a)` 没有变成 `toy.generic_call @transpose(...)`。
- 它走的是 builtin operation `toy.transpose`。

### 用户函数调用

```mlir
%9 = toy.generic_call @multiply_transpose(%6, %8)
     : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
```

这说明：

- `@multiply_transpose` 是 callee symbol reference。
- `%6`、`%8` 是 operands。
- result type 先是 unranked tensor。

### 空 return

```mlir
toy.return
```

`main` 没有显式返回值，所以 MLIRGen 自动补了一个空 `toy.return`。

## 一张关系图

```text
Toy source

def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

MLIRGen

FunctionAST
  -> FuncOp @multiply_transpose
  -> entry block arguments: %arg0, %arg1
  -> symbolTable:
       a -> %arg0
       b -> %arg1

CallExprAST transpose(a)
  -> lookup a
  -> %arg0
  -> toy.transpose(%arg0)

ReturnExprAST
  -> toy.return %result
  -> update FuncOp function type
```

## 常见误区

### 误区 1：Toy 变量名会直接出现在 MLIR 里

通常不会。

Toy 变量名 `a`、`b`、`c` 是前端源码层面的名字。MLIR 中更多看到的是：

```mlir
%arg0
%0
%1
```

变量名主要用于 MLIRGen 的 `symbolTable` 查找。

### 误区 2：`@multiply_transpose` 是 SSA value

不是。

`@multiply_transpose` 是 symbol reference attribute。

真正的 SSA value 是 `%arg0`、`%0`、`%9` 这种以 `%` 开头的名字。

### 误区 3：`toy.func` 的返回值是 operation result

不是。

`toy.func` 自身没有 SSA result。

函数返回类型写在 `function_type` 里，实际返回动作由函数体里的 `toy.return` 表示。

### 误区 4：所有函数调用都生成 `toy.generic_call`

不是。

`transpose` 是 builtin，生成 `toy.transpose`。

`print` 也是 builtin，生成 `toy.print`。

用户自定义函数调用才生成 `toy.generic_call`。

### 误区 5：`symbolTable` 就是 MLIR 的 SymbolTable

不是。

`MLIRGen.cpp` 的 `symbolTable` 是源码变量名到 MLIR value 的映射。

MLIR symbol 系统处理的是 `@main`、`@multiply_transpose` 这类符号。

## 动手观察

### 观察函数参数

在 `codegen.toy` 的输出里找：

```mlir
toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>)
```

然后回到 `MLIRGen.cpp`：

```cpp
for (const auto nameValue :
     llvm::zip(protoArgs, entryBlock.getArguments())) {
  declare(std::get<0>(nameValue)->getName(), std::get<1>(nameValue));
}
```

确认：

- Toy 参数 `a` 对应 `%arg0`。
- Toy 参数 `b` 对应 `%arg1`。

### 观察用户函数调用

找：

```mlir
toy.generic_call @multiply_transpose(...)
```

对照：

```cpp
return builder.create<GenericCallOp>(location, callee, operands);
```

确认：

- callee 字符串变成了 symbol reference。
- arguments 变成了 operands。

### 观察返回类型推导

找：

```mlir
toy.func @multiply_transpose(...) -> tensor<*xf64>
```

对照：

```cpp
if (returnOp.hasOperand()) {
  function.setType(builder.getFunctionType(
      function.getFunctionType().getInputs(), getType(VarType{})));
}
```

确认：

- 函数一开始没有返回值。
- 看到带 operand 的 `toy.return` 后，函数类型被更新。

## 本节练习

### 练习 1：拆解 `toy.func`

阅读：

```mlir
toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>)
    -> tensor<*xf64> {
  ...
}
```

回答：

- 哪一部分是 symbol name。
- 哪一部分是 block arguments。
- 哪一部分是 function result type。
- `toy.func` 本身有没有 SSA result。

### 练习 2：解释函数参数绑定

阅读 `MLIRGen.cpp` 中的：

```cpp
llvm::zip(protoArgs, entryBlock.getArguments())
```

回答：

- `protoArgs` 来自哪里。
- `entryBlock.getArguments()` 来自哪里。
- Toy 参数名如何绑定到 MLIR block argument。

### 练习 3：区分变量表和符号引用

解释下面两种东西的区别：

```cpp
symbolTable.insert("a", value);
```

和：

```mlir
toy.generic_call @multiply_transpose(%0, %1) : ...
```

回答：

- 哪个是源码变量表。
- 哪个是 MLIR symbol reference。
- 它们分别解决什么问题。

### 练习 4：分析 `GenericCallOp`

阅读：

```tablegen
let arguments = (ins FlatSymbolRefAttr:$callee,
                     Variadic<F64Tensor>:$inputs);
```

回答：

- `callee` 是 operand 还是 attribute。
- `inputs` 是 operands 还是 attributes。
- `@foo(%0, %1)` 中 `@foo` 和 `%0`、`%1` 分别是什么。

### 练习 5：解释 builtin 调用

阅读 `mlirGen(CallExprAST&)`。

回答：

- 为什么 `transpose(a)` 不生成 `toy.generic_call @transpose(...)`。
- 为什么 `multiply_transpose(a, b)` 会生成 `toy.generic_call`。
- `print(d)` 由哪个 AST 节点和哪个 MLIR op 处理。

### 练习 6：解释 return 类型推导

阅读：

```cpp
if (!returnOp) {
  builder.create<ReturnOp>(...);
} else if (returnOp.hasOperand()) {
  function.setType(...);
}
```

回答：

- 没有显式 return 时会发生什么。
- 有返回值时函数类型如何变化。
- 为什么返回类型先用 `tensor<*xf64>`。

### 练习 7：画出调用链

以这段 Toy 代码为例：

```toy
var d = multiply_transpose(b, a);
print(d);
```

画出：

```text
Toy variable
  -> MLIRGen symbolTable
  -> SSA value
  -> toy.generic_call
  -> toy.print
```

至少标出 `d` 对应的 SSA value 最后被谁使用。

## 本节小结

本节最重要的是分清三套名字系统：

```text
Toy source variable names
  -> a, b, c, d
  -> MLIRGen 的 ScopedHashTable 管理

MLIR SSA values
  -> %arg0, %0, %1, %9
  -> operation operands/results/block arguments

MLIR symbols
  -> @main, @multiply_transpose
  -> SymbolNameAttr / FlatSymbolRefAttr
```

也要记住：

- `toy.func` 是函数定义，函数名是 symbol。
- 函数参数是 entry block arguments。
- `toy.generic_call` 用 symbol reference 表示调用目标，用 operands 表示实参。
- `toy.return` 必须和 enclosing `toy.func` 的 function type 匹配。
- Ch2 的函数参数和返回值大多先用 `tensor<*xf64>`，因为 shape inference 还没有开始。

下一课会进入高层优化：C++ Pattern Rewrite 与 Canonicalization，开始观察 Toy IR 如何被改写和简化。

## 学习记录模板

```text
本节主题：函数、调用、作用域与符号
我读过的源码：
我观察过的测试：
我能解释的 toy.func：
我能解释的 toy.generic_call：
我能解释的变量作用域：
我能区分的名字系统：
我还不理解的问题：
下一步要验证的小实验：
```
