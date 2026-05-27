# 第 20 课：StructType、struct 操作与课程项目

## 本节定位

前 19 课已经走完了 Toy 编译器的主线：

```text
Toy 源码
  -> AST
  -> Toy Dialect MLIR
  -> 高层优化
  -> Affine/MemRef/Func/Arith
  -> LLVM Dialect
  -> LLVM IR
  -> JIT 执行
```

第 20 课进入 Toy 教程的最后一章 Ch7。

Ch7 不再只是补一个 pass 或 lowering，而是展示：

```text
如何给语言增加一个复合类型。
```

这个复合类型是：

```toy
struct Struct {
  var a;
  var b;
}
```

并且可以这样使用：

```toy
Struct value = {[[1, 2, 3], [4, 5, 6]],
                [[1, 2, 3], [4, 5, 6]]};

var c = multiply_transpose(value);
print(c);
```

这节课既是 Ch7 的学习课，也是整套课程的收束课。

本节会把前面学过的内容串起来：

```text
Parser / AST
MLIRGen
自定义 Type
ODS Operation
Verifier
Folder
Canonicalizer
Inliner
Shape Inference
Lowering
测试
```

## 本节目标

- 理解 Toy Ch7 增加了哪些语言能力。
- 理解 `struct` 定义如何进入 AST。
- 理解 `StructLiteralExprAST`、`StructAST`、`RecordAST` 的作用。
- 理解 `VarType` 如何同时表达 tensor shape 和 named struct type。
- 理解 MLIR 中自定义 `StructType` 的 storage、uniquing、parser/printer。
- 理解 ODS 中 `Toy_StructType`、`Toy_Type` 如何扩展 operation 类型约束。
- 理解 `toy.struct_constant` 和 `toy.struct_access` 两个 operation。
- 理解 struct 常量折叠与 struct access 折叠。
- 理解 Ch7 的 `struct-codegen.toy` 和 `struct-opt.mlir` 测试。
- 完成一个课程项目设计：基于 Toy 扩展一个小语言特性。

## 本节关键文件

源码：

```text
mlir/examples/toy/Ch7/include/toy/AST.h
mlir/examples/toy/Ch7/include/toy/Lexer.h
mlir/examples/toy/Ch7/include/toy/Parser.h
mlir/examples/toy/Ch7/include/toy/Dialect.h
mlir/examples/toy/Ch7/include/toy/Ops.td
mlir/examples/toy/Ch7/mlir/MLIRGen.cpp
mlir/examples/toy/Ch7/mlir/Dialect.cpp
mlir/examples/toy/Ch7/mlir/ToyCombine.cpp
mlir/examples/toy/Ch7/mlir/LowerToAffineLoops.cpp
mlir/examples/toy/Ch7/mlir/LowerToLLVM.cpp
mlir/examples/toy/Ch7/toyc.cpp
```

测试：

```text
mlir/test/Examples/Toy/Ch7/struct-ast.toy
mlir/test/Examples/Toy/Ch7/struct-codegen.toy
mlir/test/Examples/Toy/Ch7/struct-opt.mlir
mlir/test/Examples/Toy/Ch7/ast.toy
```

建议阅读顺序：

1. 先看 `struct-ast.toy`，理解源语言新增语法。
2. 再看 `AST.h` 和 `Parser.h`，理解 struct 如何进入 AST。
3. 再看 `Dialect.h` 和 `Dialect.cpp`，理解 `StructType`。
4. 再看 `Ops.td`，理解 struct 相关 operation。
5. 再看 `MLIRGen.cpp`，理解 struct 源码如何生成 MLIR。
6. 最后看 `ToyCombine.cpp` 和 `struct-opt.mlir`，理解 struct 优化。

## Ch7 新增的语言能力

Ch7 给 Toy 增加了结构体。

源语言层面可以写：

```toy
struct Struct {
  var a;
  var b;
}
```

这表示：

```text
定义一个名为 Struct 的复合类型。
它有两个字段 a 和 b。
字段类型在这个教学版本中仍然保持简化。
```

可以声明一个 struct 变量：

```toy
Struct value = {[[1, 2, 3], [4, 5, 6]],
                [[1, 2, 3], [4, 5, 6]]};
```

可以访问字段：

```toy
value.a
value.b
```

可以把 struct 传给函数：

```toy
def multiply_transpose(Struct value) {
  return transpose(value.a) * transpose(value.b);
}
```

也就是说，Ch7 把 Toy 从：

```text
只有 tensor-like value
```

扩展为：

```text
tensor value + struct value
```

这对编译器来说影响很大，因为它会穿过：

```text
词法
语法
AST
类型系统
IR 类型
operation 定义
优化
函数调用
lowering 前的清理
```

## `struct-codegen.toy` 示例

关键测试文件：

```text
mlir/test/Examples/Toy/Ch7/struct-codegen.toy
```

源程序：

```toy
struct Struct {
  var a;
  var b;
}

def multiply_transpose(Struct value) {
  return transpose(value.a) * transpose(value.b);
}

def main() {
  Struct value = {[[1, 2, 3], [4, 5, 6]],
                  [[1, 2, 3], [4, 5, 6]]};

  var c = multiply_transpose(value);
  print(c);
}
```

未开启 `-opt` 的 MLIR 里会看到：

```mlir
toy.func private @multiply_transpose(
  %arg0: !toy.struct<tensor<*xf64>, tensor<*xf64>>
) -> tensor<*xf64> {
  %0 = toy.struct_access %arg0[0]
       : !toy.struct<tensor<*xf64>, tensor<*xf64>> -> tensor<*xf64>
  %1 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
  %2 = toy.struct_access %arg0[1]
       : !toy.struct<tensor<*xf64>, tensor<*xf64>> -> tensor<*xf64>
  %3 = toy.transpose(%2 : tensor<*xf64>) to tensor<*xf64>
  %4 = toy.mul %1, %3 : tensor<*xf64>
  toy.return %4 : tensor<*xf64>
}
```

`main` 里会看到：

```mlir
%0 = toy.struct_constant [...]
     : !toy.struct<tensor<*xf64>, tensor<*xf64>>
%1 = toy.generic_call @multiply_transpose(%0)
     : (!toy.struct<tensor<*xf64>, tensor<*xf64>>) -> tensor<*xf64>
toy.print %1 : tensor<*xf64>
toy.return
```

开启 `-opt` 后，测试期望变成：

```mlir
toy.func @main() {
  %0 = toy.constant dense<...> : tensor<2x3xf64>
  %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %2 = toy.mul %1, %1 : tensor<3x2xf64>
  toy.print %2 : tensor<3x2xf64>
  toy.return
}
```

这说明：

```text
struct_constant 被折叠。
struct_access 被折叠。
函数被 inliner 内联。
shape inference 推出了具体 tensor shape。
最终 struct 这个中间抽象消失，只剩实际 tensor 计算。
```

## AST 层：RecordAST

Ch7 的 `AST.h` 里，module 不再只是函数列表。

它引入了：

```cpp
class RecordAST {
public:
  enum RecordASTKind {
    Record_Function,
    Record_Struct,
  };
};
```

然后：

```cpp
class FunctionAST : public RecordAST { ... };
class StructAST : public RecordAST { ... };
```

这意味着一个 Toy module 中可以同时有：

```text
function record
struct record
```

源码层面：

```toy
struct Struct { ... }

def main() { ... }
```

AST 层面就是：

```text
ModuleAST
  RecordAST(StructAST)
  RecordAST(FunctionAST)
```

这是一个很典型的语言扩展动作：

```text
原来 module 只有函数。
现在 module 顶层语法需要接纳更多 declaration。
```

## AST 层：VarType

Ch7 中变量类型用：

```cpp
struct VarType {
  std::string name;
  std::vector<int64_t> shape;
};
```

它同时表达两类信息：

| 字段 | 含义 |
| --- | --- |
| `name` | named type，例如 `Struct` |
| `shape` | tensor shape，例如 `<2, 3>` |

因此：

```toy
var a<2, 3> = ...
```

可以表示为：

```text
name = ""
shape = [2, 3]
```

而：

```toy
Struct value = ...
```

可以表示为：

```text
name = "Struct"
shape = []
```

这个设计很教学化，不是一个完整工业语言类型系统，但足够展示：

```text
给语言增加 named type 后，前端和 IRGen 都需要能携带这个名字。
```

## AST 层：StructLiteralExprAST

结构体字面量使用：

```cpp
class StructLiteralExprAST : public ExprAST {
  std::vector<std::unique_ptr<ExprAST>> values;
};
```

对应源码：

```toy
{[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]}
```

它和 tensor literal 的区别是：

```text
tensor literal 有 dims。
struct literal 只是字段值列表。
```

字段值本身还可以是：

```text
number
tensor literal
nested struct literal
```

这就是为什么 `MLIRGen.cpp` 里 `getConstantAttr(StructLiteralExprAST &lit)` 需要递归处理。

## Parser 层：顶层 struct

Ch7 的 parser 在解析 module 时，会根据 token 判断：

```text
def
  -> parseDefinition()

struct
  -> parseStruct()
```

`Lexer.h` 中新增关键字：

```cpp
tok_struct = -5
```

并在 identifier 识别时返回：

```cpp
if (identifierStr == "struct")
  return tok_struct;
```

`Parser.h` 中的 `parseStruct()` 处理：

```text
struct Identifier {
  var field;
  var field;
}
```

这一步只负责语法结构，不负责最终 MLIR 类型。

真正把 `Struct` 映射成 MLIR `StructType` 的地方是 `MLIRGen.cpp`。

## Parser 层：字段访问

Toy 使用 `.` 表示字段访问：

```toy
value.a
```

在 AST 中，它被表示成一个 binary expression：

```text
BinOp: .
  lhs = value
  rhs = a
```

这是一种简化实现。

更完整的语言可能会引入专门的 AST 节点：

```text
MemberAccessExprAST
```

但 Toy 为了教学，复用 `BinaryExprAST`。

因此后面的 MLIRGen 要特别判断：

```cpp
if (binop.getOp() == '.') {
  ...
  return builder.create<StructAccessOp>(location, lhs, *accessIndex);
}
```

这也说明：

```text
AST 的形状会直接影响 IRGen 的复杂度。
```

## MLIR 类型层：StructType

Ch7 在 Toy dialect 中新增了一个自定义类型：

```cpp
class StructType : public mlir::Type::TypeBase<
    StructType,
    mlir::Type,
    detail::StructTypeStorage> {
public:
  static StructType get(llvm::ArrayRef<mlir::Type> elementTypes);
  llvm::ArrayRef<mlir::Type> getElementTypes();
  size_t getNumElementTypes();
  static constexpr StringLiteral name = "toy.struct";
};
```

它表示：

```text
一个由若干 element type 组成的复合类型。
```

例如：

```mlir
!toy.struct<tensor<*xf64>, tensor<*xf64>>
```

表示：

```text
有两个字段。
字段类型都是 tensor<*xf64>。
```

嵌套 struct 也可以表示：

```mlir
!toy.struct<!toy.struct<tensor<*xf64>>, tensor<*xf64>>
```

## TypeStorage 与 uniquing

自定义 MLIR type 需要 storage。

Ch7 中定义：

```cpp
struct StructTypeStorage : public mlir::TypeStorage {
  using KeyTy = llvm::ArrayRef<mlir::Type>;
  llvm::ArrayRef<mlir::Type> elementTypes;
};
```

它的 key 是：

```text
elementTypes
```

也就是说：

```text
相同 element type 列表的 StructType 会被 uniqued 成同一个类型实例。
```

`StructType::get()` 中调用：

```cpp
return Base::get(ctx, elementTypes);
```

这会通过 MLIR 的 type uniquing 机制创建或复用类型。

学习重点不是背 storage 模板，而是理解：

```text
MLIR Type 通常是不可变、可 uniquing 的对象。
自定义 Type 需要定义它的存储内容和唯一化 key。
```

## StructType 的 parser/printer

Ch7 的 Toy dialect 启用了：

```tablegen
let useDefaultTypePrinterParser = 1;
```

并在 C++ 中实现：

```cpp
mlir::Type ToyDialect::parseType(mlir::DialectAsmParser &parser) const
void ToyDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter &printer) const
```

`parseType()` 支持：

```mlir
!toy.struct<tensor<*xf64>, tensor<*xf64>>
```

内部语法是：

```text
struct-type ::= `struct` `<` type (`,` type)* `>`
```

并检查 element type 只能是：

```text
TensorType
StructType
```

这意味着：

```text
Toy struct 可以嵌套 Toy struct。
Toy struct 里不能随意塞任意 MLIR type。
```

`printType()` 则把 `StructType` 打印回：

```mlir
struct<...>
```

最终外层 dialect 前缀会显示成：

```mlir
!toy.struct<...>
```

## ODS 层：Toy_StructType

`Ops.td` 中新增：

```tablegen
def Toy_StructType :
    DialectType<Toy_Dialect, CPred<"::llvm::isa<StructType>($_self)">,
                "Toy struct type">;
```

这让 ODS 能把 Toy 自定义 C++ type 纳入 operation 约束。

随后定义：

```tablegen
def Toy_Type : AnyTypeOf<[F64Tensor, Toy_StructType]>;
```

这表示：

```text
Toy operation 中一些位置可以接受 tensor，也可以接受 struct。
```

例如：

```tablegen
def GenericCallOp : Toy_Op<"generic_call", ...> {
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<Toy_Type>:$inputs,
    ...
  );
  let results = (outs Toy_Type);
}
```

这就是为什么 generic call 可以传递 struct：

```mlir
toy.generic_call @multiply_transpose(%value)
  : (!toy.struct<tensor<*xf64>, tensor<*xf64>>) -> tensor<*xf64>
```

## ODS 层：ReturnOp 类型扩展

Ch7 中 `ReturnOp` 的 operand 类型也改为：

```tablegen
let arguments = (ins Variadic<Toy_Type>:$input);
```

之前 Toy 主要返回 tensor。

现在理论上也可以返回 struct。

这类改动很容易漏：

```text
新增一个语言类型后，不只是新增一个 operation。
所有可能携带 value 的 operation 都要检查是否需要接纳这个类型。
```

包括：

```text
call
return
function argument
function result
variable declaration
constant materializer
inliner conversion
```

## `toy.struct_constant`

ODS 定义：

```tablegen
def StructConstantOp : Toy_Op<"struct_constant", [ConstantLike, Pure]> {
  let arguments = (ins ArrayAttr:$value);
  let results = (outs Toy_StructType:$output);
  let assemblyFormat = "$value attr-dict `:` type($output)";
  let hasVerifier = 1;
  let hasFolder = 1;
}
```

它表示一个常量 struct value。

示例：

```mlir
%0 = toy.struct_constant [
  dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf64>,
  dense<[[5.0, 6.0], [7.0, 8.0]]> : tensor<2x2xf64>
] : !toy.struct<tensor<*xf64>, tensor<*xf64>>
```

它的 value 是：

```text
ArrayAttr
```

每个字段对应一个 attribute。

字段可以是：

```text
DenseElementsAttr
ArrayAttr
```

因此可以表示嵌套 struct 常量。

## `toy.struct_access`

ODS 定义：

```tablegen
def StructAccessOp : Toy_Op<"struct_access", [Pure]> {
  let arguments = (ins Toy_StructType:$input, I64Attr:$index);
  let results = (outs Toy_Type:$output);
  let assemblyFormat = [{
    $input `[` $index `]` attr-dict `:` type($input) `->` type($output)
  }];
  let builders = [
    OpBuilder<(ins "Value":$input, "size_t":$index)>
  ];
  let hasVerifier = 1;
  let hasFolder = 1;
}
```

它表示按字段 index 访问 struct。

例如：

```mlir
%0 = toy.struct_access %arg0[0]
     : !toy.struct<tensor<*xf64>, tensor<*xf64>> -> tensor<*xf64>
```

源码中的：

```toy
value.a
```

最终会变成：

```text
toy.struct_access value[0]
```

而：

```toy
value.b
```

会变成：

```text
toy.struct_access value[1]
```

字段名只存在于 Toy 源语言和 AST 层。

MLIR 层使用数字 index。

## StructAccessOp verifier

`StructAccessOp::verify()` 检查两件事：

```cpp
StructType structTy = llvm::cast<StructType>(getInput().getType());
size_t indexValue = getIndex();
if (indexValue >= structTy.getNumElementTypes())
  return emitOpError()
         << "index should be within the range of the input struct type";

mlir::Type resultType = getResult().getType();
if (resultType != structTy.getElementTypes()[indexValue])
  return emitOpError() << "must have the same result type as the struct "
                          "element referred to by the index";
```

也就是：

```text
index 不能越界。
result type 必须等于对应字段类型。
```

这很重要。

否则下面这种 IR 可能悄悄通过：

```mlir
%0 = toy.struct_access %s[0]
     : !toy.struct<tensor<*xf64>, tensor<*xf64>> -> !toy.struct<...>
```

但字段 0 实际上是 tensor，不是 struct。

Verifier 的职责就是尽早拒绝不合法 IR。

## StructConstantOp verifier

Ch7 复用了一个递归检查函数：

```cpp
static llvm::LogicalResult verifyConstantForType(
    mlir::Type type,
    mlir::Attribute opaqueValue,
    mlir::Operation *op)
```

对于 tensor：

```text
要求 attribute 是 DenseFPElementsAttr。
如果 result 是 ranked tensor，rank 和 shape 要匹配。
```

对于 struct：

```text
要求 attribute 是 ArrayAttr。
ArrayAttr 元素数量必须等于 struct 字段数量。
每个字段递归调用 verifyConstantForType。
```

因此嵌套 struct 常量也会被递归检查。

这是一种非常常见的 verifier 写法：

```text
类型结构是递归的。
属性结构也必须递归匹配类型结构。
```

## MLIRGen：structMap

`MLIRGen.cpp` 中新增：

```cpp
llvm::StringMap<std::pair<mlir::Type, StructAST *>> structMap;
```

它把源语言 struct 名字映射到：

```text
MLIR StructType
原始 StructAST
```

为什么还要保存 `StructAST *`？

因为字段访问需要从名字找到 index。

例如：

```toy
value.b
```

MLIRGen 需要知道：

```text
Struct 的字段列表是 [a, b]
b 的 index 是 1
```

这个信息来自 `StructAST`。

所以 `structMap` 不只是类型表，还是字段元信息表。

## MLIRGen：生成 StructType

`mlirGen(StructAST &str)` 会：

1. 检查 struct 名字是否重复。
2. 遍历字段变量。
3. 禁止字段带 initializer。
4. 禁止字段声明 tensor shape。
5. 把字段类型转换为 MLIR type。
6. 创建 `StructType::get(elementTypes)`。
7. 放入 `structMap`。

简化代码：

```cpp
llvm::LogicalResult mlirGen(StructAST &str) {
  if (structMap.count(str.getName()))
    return emitError(...) << "struct type already exists";

  std::vector<mlir::Type> elementTypes;
  for (auto &variable : str.getVariables()) {
    if (variable->getInitVal())
      return emitError(...) << "variables within a struct definition "
                               "must not have initializers";
    if (!variable->getType().shape.empty())
      return emitError(...) << "variables within a struct definition "
                               "must not have initializers";

    mlir::Type type = getType(variable->getType(), variable->loc());
    elementTypes.push_back(type);
  }

  structMap.try_emplace(str.getName(),
                        StructType::get(elementTypes),
                        &str);
  return mlir::success();
}
```

这里有一个值得注意的细节：

```text
字段不能写 shape。
字段类型要么是默认 tensor<*xf64>，要么是另一个 named struct。
```

Toy 教程故意限制了功能范围，避免完整类型系统过于复杂。

## MLIRGen：生成 struct literal

结构体字面量由：

```cpp
mlir::Value mlirGen(StructLiteralExprAST &lit)
```

生成：

```mlir
toy.struct_constant
```

它先调用：

```cpp
getConstantAttr(StructLiteralExprAST &lit)
```

递归生成：

```text
ArrayAttr
StructType
```

然后：

```cpp
return builder.create<StructConstantOp>(loc(lit.loc()),
                                        dataType,
                                        dataAttr);
```

也就是说：

```toy
{[[1, 2]], [[3, 4]]}
```

会变成：

```mlir
toy.struct_constant [
  dense<...> : tensor<...>,
  dense<...> : tensor<...>
] : !toy.struct<tensor<*xf64>, tensor<*xf64>>
```

## MLIRGen：变量声明检查

对于：

```toy
Struct value = {...};
```

`mlirGen(VarDeclExprAST &vardecl)` 会检查：

```cpp
VarType varType = vardecl.getType();
if (!varType.name.empty()) {
  mlir::Type type = getType(varType, vardecl.loc());
  if (type != value.getType()) {
    emitError(...) << "struct type of initializer is different "
                      "than the variable declaration";
    return nullptr;
  }
}
```

这保证：

```text
变量声明的 named struct type
和 initializer 生成出来的 StructType
必须一致。
```

如果不一致，MLIRGen 阶段就报错。

这属于前端语义检查的一部分。

## MLIRGen：字段访问

字段访问由 `BinaryExprAST` 的 `.` 分支处理：

```cpp
if (binop.getOp() == '.') {
  std::optional<size_t> accessIndex = getMemberIndex(binop);
  if (!accessIndex) {
    emitError(location, "invalid access into struct expression");
    return nullptr;
  }
  return builder.create<StructAccessOp>(location, lhs, *accessIndex);
}
```

`getMemberIndex()` 做：

```text
找出 lhs 的 struct 类型。
确认 rhs 是字段名。
在 StructAST 字段列表中查找字段名。
返回字段 index。
```

所以：

```toy
value.a
```

生成：

```mlir
toy.struct_access %value[0]
```

而：

```toy
value.b
```

生成：

```mlir
toy.struct_access %value[1]
```

## Optimizer：struct constant fold

`ToyCombine.cpp` 中有：

```cpp
OpFoldResult StructConstantOp::fold(FoldAdaptor adaptor) {
  return getValue();
}
```

这表示：

```text
toy.struct_constant 可以折叠成它携带的 ArrayAttr。
```

类似：

```cpp
OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  return getValue();
}
```

对常量 operation 来说，这是最基础的 folder。

它允许后续 operation 在 folder 中通过 adaptor 拿到常量属性。

## Optimizer：struct access fold

`ToyCombine.cpp` 中还有：

```cpp
OpFoldResult StructAccessOp::fold(FoldAdaptor adaptor) {
  auto structAttr =
      llvm::dyn_cast_if_present<mlir::ArrayAttr>(adaptor.getInput());
  if (!structAttr)
    return nullptr;

  size_t elementIndex = getIndex();
  return structAttr[elementIndex];
}
```

这表示：

```text
如果 struct_access 的 input 是常量 struct，
那么直接返回对应字段的 attribute。
```

例如：

```mlir
%0 = toy.struct_constant [
  dense<1.0> : tensor<2x2xf64>,
  dense<2.0> : tensor<2x2xf64>
] : !toy.struct<tensor<*xf64>, tensor<*xf64>>

%1 = toy.struct_access %0[0]
     : !toy.struct<tensor<*xf64>, tensor<*xf64>> -> tensor<*xf64>
```

可以被折叠成：

```mlir
%1 = toy.constant dense<1.0> : tensor<2x2xf64>
```

这正是 `struct-opt.mlir` 测试在验证的行为。

## `struct-opt.mlir` 测试

测试输入：

```mlir
toy.func @main() {
  %0 = toy.struct_constant [
    [dense<4.000000e+00> : tensor<2x2xf64>],
    dense<4.000000e+00> : tensor<2x2xf64>
  ] : !toy.struct<!toy.struct<tensor<*xf64>>, tensor<*xf64>>
  %1 = toy.struct_access %0[0]
       : !toy.struct<!toy.struct<tensor<*xf64>>, tensor<*xf64>>
         -> !toy.struct<tensor<*xf64>>
  %2 = toy.struct_access %1[0]
       : !toy.struct<tensor<*xf64>> -> tensor<*xf64>
  toy.print %2 : tensor<*xf64>
  toy.return
}
```

这是嵌套 struct：

```text
outer struct
  field 0: inner struct
    field 0: tensor
  field 1: tensor
```

优化后检查：

```mlir
// CHECK-LABEL: toy.func @main
// CHECK-NEXT: %[[CST:.*]] = toy.constant dense<4.0
// CHECK-NEXT: toy.print %[[CST]]
```

这说明两级 `struct_access` 都被折叠掉：

```text
struct_constant
  -> access field 0
  -> access nested field 0
  -> tensor constant
```

最终只剩：

```text
toy.constant
toy.print
```

这个测试很适合作为 Ch7 优化的最小模型。

## Inliner 与 struct

`struct-codegen.toy` 开启 `-opt` 后，`multiply_transpose` 被内联进 `main`。

未优化时：

```mlir
%0 = toy.struct_constant [...]
%1 = toy.generic_call @multiply_transpose(%0)
toy.print %1
```

优化后：

```mlir
%0 = toy.constant dense<...> : tensor<2x3xf64>
%1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
%2 = toy.mul %1, %1 : tensor<3x2xf64>
toy.print %2 : tensor<3x2xf64>
```

这里发生了多件事：

```text
generic_call 被 inline。
函数参数 value 被替换成具体 struct_constant。
value.a / value.b 变成 struct_access。
struct_access 从 struct_constant 中取出字段。
两个字段常量相同，CSE 可以合并。
shape inference 推出具体 tensor shape。
```

这说明 Ch7 的 struct 支持并不是孤立的。

它和前面学过的：

```text
inliner
canonicalizer
CSE
shape inference
fold
```

全部发生了互动。

## Shape Inference 与 struct

Shape inference 的目标仍然主要是 tensor operation。

例如：

```mlir
toy.transpose : tensor<*xf64> -> tensor<*xf64>
toy.mul       : tensor<*xf64> -> tensor<*xf64>
```

在 struct 被折叠和访问后，实际 tensor value 变得更具体：

```mlir
tensor<2x3xf64>
```

于是 shape inference 可以推出：

```mlir
transpose: tensor<2x3xf64> -> tensor<3x2xf64>
mul:       tensor<3x2xf64> -> tensor<3x2xf64>
```

这说明：

```text
优化和分析往往互相促进。
```

常量折叠让 IR 更具体。

更具体的 IR 又让 shape inference 能推导更多信息。

推导出的具体 shape 又让后续 lowering 更容易。

## Lowering 与 struct

一个很关键的问题：

```text
Toy struct 最终如何 lowering 到 Affine/LLVM？
```

Ch7 的设计答案是：

```text
尽量在高层优化中消掉 struct。
```

`LowerToAffineLoops.cpp` 主要处理：

```text
toy.constant
toy.add
toy.mul
toy.transpose
toy.print
toy.return
toy.func
```

它并没有把 `toy.struct_constant` 和 `toy.struct_access` lowering 成某种低层 runtime struct。

这意味着：

```text
进入 affine lowering 前，struct 相关 operation 应该已经被 inliner/canonicalizer/folder 消掉。
```

这是一种很实用的教学策略：

```text
struct 只是高层编译期抽象。
最终生成代码时，只保留实际 tensor 计算。
```

如果你以后想支持真正运行时 struct，就需要设计新的 lowering：

```text
StructType -> LLVM struct type / memref descriptor tuple / runtime object
struct_constant -> aggregate construction
struct_access -> extractvalue / pointer field load
```

Toy Ch7 没有走这条路。

## Ch7 pipeline 的细节

Ch7 的 `toyc.cpp` 和 Ch6 很像，但高层优化部分有一个小差异：

```cpp
optPM.addPass(mlir::createCanonicalizerPass());
optPM.addPass(mlir::toy::createShapeInferencePass());
optPM.addPass(mlir::createCanonicalizerPass());
optPM.addPass(mlir::createCSEPass());
```

也就是说，Ch7 在 shape inference 前后都跑了 canonicalizer。

这对 struct 很有用：

```text
前一个 canonicalizer 可以先折叠 struct_constant / struct_access。
shape inference 接着利用更具体的 tensor。
后一个 canonicalizer 再清理 shape inference 后暴露出的机会。
```

这是一个值得学习的 pass pipeline 设计点：

```text
pass 顺序不是随便排的。
某个 pass 的结果可能为后续 pass 暴露机会。
后续 pass 也可能让前面的 cleanup 再跑一次更有效。
```

## 常见误区

### 误区 1：struct 字段名会进入 MLIR 类型

不会。

Toy 的 `StructType` 只保存字段类型列表：

```mlir
!toy.struct<tensor<*xf64>, tensor<*xf64>>
```

它不保存：

```text
a
b
```

字段名只在源语言和 AST/MLIRGen 的 `StructAST` 中用于计算 index。

MLIR operation 使用数字 index：

```mlir
toy.struct_access %value[0]
toy.struct_access %value[1]
```

### 误区 2：struct 一定要 lowering 到 LLVM struct

不一定。

Toy Ch7 中 struct 主要是高层抽象。

在示例程序中，它会通过内联、fold、canonicalization 消失。

这比设计完整运行时 struct 简单很多，也足以展示 MLIR 的高层优化能力。

### 误区 3：新增类型只要写 `Dialect.h`

不够。

新增 `StructType` 至少涉及：

```text
Dialect.h
Dialect.cpp
Ops.td
MLIRGen.cpp
Parser.h
AST.h
ToyCombine.cpp
tests
```

如果还要 lowering，则还要修改：

```text
LowerToAffineLoops.cpp
LowerToLLVM.cpp
```

### 误区 4：fold 和 rewrite 是一回事

不完全一样。

`fold()` 通常用于：

```text
根据已有 attribute 或 operand 快速计算更简单的结果。
```

Rewrite pattern 更通用，可以：

```text
匹配多个 operation。
创建新 operation。
替换复杂 IR 子图。
```

Ch7 中：

```text
StructConstantOp::fold
StructAccessOp::fold
```

是 folder。

`SimplifyRedundantTranspose` 是 C++ rewrite pattern。

`ToyCombine.td` 中的 reshape 优化是 DRR pattern。

### 误区 5：`-opt` 只是让代码更快

在 Toy 教程中，`-opt` 还可能让 IR 更容易继续 lowering。

对于 Ch7 struct 示例，如果 struct 没有在高层消掉，后面的 affine lowering 并不知道如何处理 struct operation。

因此这里的优化不只是性能问题，也是分阶段 lowering 设计的一部分。

## 动手观察

### 观察 1：AST 中的 struct

运行：

```bash
toyc-ch7 mlir/test/Examples/Toy/Ch7/struct-ast.toy -emit=ast
```

观察：

- `Struct: Struct` 是否出现在 module 顶层。
- `VarDecl value<Struct>` 如何显示。
- `value.a` 是否显示成 `BinOp: .`。
- struct literal 如何显示。

### 观察 2：未优化 MLIR

运行：

```bash
toyc-ch7 mlir/test/Examples/Toy/Ch7/struct-codegen.toy -emit=mlir
```

观察：

- `!toy.struct<...>` 类型。
- `toy.struct_constant`。
- `toy.struct_access`。
- `toy.generic_call @multiply_transpose`。
- `multiply_transpose` 函数是否仍然存在。

### 观察 3：优化后 MLIR

运行：

```bash
toyc-ch7 mlir/test/Examples/Toy/Ch7/struct-codegen.toy -emit=mlir -opt
```

观察：

- `toy.struct_constant` 是否消失。
- `toy.struct_access` 是否消失。
- `toy.generic_call` 是否消失。
- `multiply_transpose` 是否被内联。
- tensor shape 是否变成具体的 `tensor<2x3xf64>`、`tensor<3x2xf64>`。

### 观察 4：嵌套 struct fold

运行：

```bash
toyc-ch7 mlir/test/Examples/Toy/Ch7/struct-opt.mlir -emit=mlir -opt
```

观察：

- 两级 `toy.struct_access` 是否被折叠。
- 最终是否只剩 `toy.constant` 和 `toy.print`。

### 观察 5：尝试进入 lowering

运行：

```bash
toyc-ch7 mlir/test/Examples/Toy/Ch7/struct-codegen.toy -emit=mlir-affine -opt
```

观察：

- struct 是否已经在进入 affine lowering 前被消掉。
- 输出是否主要由 `func`、`memref`、`affine`、`arith`、`toy.print` 构成。

再思考：

```text
如果不加 -opt，struct 相关 operation 还存在，lowering 会发生什么？
```

## 课程项目：给 Toy 增加一个小特性

第 20 课之后，建议做一个小项目，不要只停留在读代码。

推荐项目规模：

```text
能改动前端、IR、优化或 lowering 中至少两个层级。
不要求很大，但要形成闭环。
```

下面给出三个可选项目。

## 项目 A：新增 `toy.neg`

目标：

```toy
var b = -a;
```

或者：

```toy
var b = neg(a);
```

建议实现路径：

1. Parser 支持一元负号或 builtin `neg`。
2. AST 增加节点或复用 call。
3. ODS 增加 `NegOp`。
4. MLIRGen 生成 `toy.neg`。
5. Shape inference 中让输出 shape 等于输入 shape。
6. LowerToAffineLoops 中 lowering 到 `arith.negf` 或 `0 - x`。
7. 添加测试：

```text
neg-ast.toy
neg-codegen.toy
neg-affine-lowering.mlir
```

这个项目适合练习：

```text
前端 + ODS + shape inference + lowering
```

## 项目 B：新增 struct 字段名打印

目标：

让 MLIR `StructType` 或某个辅助属性保留字段名。

例如源语言：

```toy
struct Pair {
  var lhs;
  var rhs;
}
```

你可以尝试在 IR 中保留：

```text
字段名到 index 的映射
```

这个项目比项目 A 更难，因为当前 `StructType` 只保存 element types。

你需要思考：

```text
字段名应该是类型的一部分吗？
字段名不同但字段类型相同的 struct 是否应该是同一种 MLIR type？
字段名是否只应该作为 symbol table / frontend metadata 存在？
```

这个项目适合练习：

```text
类型系统设计 + parser/printer + verifier
```

## 项目 C：让 struct 真正 lowering 到 LLVM

目标：

不要依赖优化把 struct 消掉，而是支持：

```text
toy.struct_constant
toy.struct_access
```

一路 lowering 到 LLVM Dialect。

这非常有挑战。

你需要设计：

```text
StructType 如何映射到 LLVM 类型。
struct_constant 如何构造 aggregate。
struct_access 如何提取字段。
嵌套 struct 如何处理。
tensor/memref 字段如何表示。
```

这个项目适合练习：

```text
Dialect Conversion + TypeConverter + LLVM lowering
```

建议在完成项目 A 后再做。

## 项目验收标准

不管选择哪个项目，至少准备这些材料：

```text
1. 一个源语言示例。
2. 一个 -emit=ast 测试。
3. 一个 -emit=mlir 测试。
4. 如果涉及优化，添加 -opt 测试。
5. 如果涉及 lowering，添加 -emit=mlir-affine 或 -emit=mlir-llvm 测试。
6. 一段说明：这个特性经过了哪些编译阶段。
```

写测试时优先使用已有风格：

```text
RUN: toyc-ch7 ...
FileCheck
CHECK-LABEL
CHECK-NEXT
CHECK-SAME
```

不要只靠手动观察。

## 本节练习

### 练习 1：解释 Ch7 新增语法

回答：

- `struct Struct { var a; var b; }` 在 AST 中如何表示？
- `Struct value = {...}` 和 `var a<2, 3> = ...` 的类型信息有什么不同？
- `value.a` 为什么在 AST 中是 `BinOp: .`？

### 练习 2：解释 StructType

回答：

- `StructTypeStorage` 存储什么？
- `StructType::get()` 为什么需要 element types？
- `!toy.struct<tensor<*xf64>, tensor<*xf64>>` 表示什么？
- Ch7 的 `StructType` 是否保存字段名？

### 练习 3：解释 struct operation

回答：

- `toy.struct_constant` 的 attribute 是什么类型？
- `toy.struct_access` 的 `index` 是什么？
- `StructAccessOp::verify()` 检查哪些条件？
- 为什么 `StructConstantOp::verify()` 需要递归？

### 练习 4：解释 struct 优化

回答：

- `StructConstantOp::fold()` 返回什么？
- `StructAccessOp::fold()` 在什么情况下能成功？
- `struct-opt.mlir` 中两级 access 为什么能折叠成一个 `toy.constant`？

### 练习 5：解释 `struct-codegen.toy`

回答：

- 未优化 IR 中为什么有 `toy.generic_call`？
- `-opt` 后为什么 `multiply_transpose` 消失？
- `value.a` 和 `value.b` 最终为什么能变成同一个 tensor constant？
- shape inference 在这里起了什么作用？

### 练习 6：解释 lowering 设计

回答：

- Ch7 是否实现了 struct 到 LLVM struct 的 lowering？
- 为什么 Toy 可以不实现它？
- 如果 struct 在 affine lowering 前没有被消掉，可能会发生什么？
- 如果要支持运行时 struct，你会从哪里开始改？

### 练习 7：设计一个扩展

从项目 A/B/C 中选一个，写设计说明：

```text
语法：
AST 改动：
ODS 改动：
MLIRGen 改动：
优化改动：
Lowering 改动：
测试计划：
风险点：
```

## 本节小结

第 20 课完成了 Toy 教程的最后一块拼图：

```text
给语言增加复合类型，并让它进入 MLIR 类型系统和优化流程。
```

需要记住：

- Ch7 把 module 顶层从只有 function 扩展为 function + struct。
- `VarType` 同时承载 named type 和 tensor shape。
- `StructLiteralExprAST` 表示 `{...}` 形式的 struct 字面量。
- `StructType` 是 Toy dialect 的自定义 MLIR type。
- `StructTypeStorage` 用 element type 列表作为 uniquing key。
- `Toy_StructType` 让 ODS 可以识别 Toy 自定义 struct type。
- `toy.struct_constant` 表示常量 struct value。
- `toy.struct_access` 表示按 index 访问 struct 字段。
- struct 字段名不会进入 MLIR type，MLIR 层使用 index。
- struct 相关 operation 主要通过 fold/canonicalizer 在高层被消掉。
- Ch7 没有实现真正运行时 struct lowering。
- 最后的课程项目应该至少贯穿两个编译阶段，并配套 FileCheck 测试。

到这里，20 节课的主线已经完整覆盖：

```text
前端
AST
MLIR 基础
Dialect / ODS / Operation / Type
Verifier / Builder / Assembly Format
Pattern Rewrite / DRR / Canonicalization
Pass Manager / Shape Inference / Inliner / CSE
Dialect Conversion
Affine lowering
LLVM Dialect lowering
LLVM IR 导出
JIT 执行
语言扩展
```

## 学习记录模板

```text
本节主题：StructType、struct 操作与课程项目
我读过的源码：
我观察过的测试：
我理解的 struct AST：
我理解的 StructType：
我理解的 struct_constant：
我理解的 struct_access：
我理解的 struct fold：
我选择的课程项目：
我的实现计划：
我还不理解的问题：
下一步要验证的小实验：
```
