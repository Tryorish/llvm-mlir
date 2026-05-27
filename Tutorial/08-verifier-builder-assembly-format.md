# 第 8 课：Verifier、Builder 与自定义 Assembly Format

## 本节定位

第 7 课已经把 Toy operations 的输入、输出、属性、类型和 trait 讲清楚了。

但一个 operation 还远不止“长什么样”。它通常还要回答三类问题：

```text
1. 这个 operation 合不合法？      -> verifier
2. 这个 operation 怎么更方便地创建？ -> builder
3. 这个 operation 文本怎么读写？    -> parser / printer / assembly format
```

本节就围绕这三件事展开，直接看 Ch2 的 `Ops.td` 和 `Dialect.cpp`：

- `let hasVerifier = 1`
- `OpBuilder<...>`
- `hasCustomAssemblyFormat`
- `assemblyFormat`
- `parse` / `print`

如果说第 7 课是在看 operation 的“数据模型”，那第 8 课就是在看 operation 的“行为模型”。

## 本节目标

- 理解 `hasVerifier = 1` 的作用。
- 能看懂 `ConstantOp::verify()`、`ReturnOp::verify()`、`TransposeOp::verify()`。
- 理解 ODS builder 的用途，以及它如何服务于 `builder.create<...>()`。
- 能区分 ODS 自动生成的 builder 和自定义 C++ builder。
- 理解 `hasCustomAssemblyFormat`、`assemblyFormat`、`parse`、`print` 的关系。
- 能读懂 `toy.constant`、`toy.add`、`toy.mul`、`toy.generic_call`、`toy.return`、`toy.reshape`、`toy.transpose` 的文本格式。
- 能解释 `invalid.mlir` 为什么不合法。

## 本节关键文件

源码：

```text
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

1. 先看 `Ops.td` 里每个 op 的 `builders`、`hasVerifier`、`assemblyFormat`。
2. 再看 `Dialect.cpp` 里对应的 `build`、`verify`、`parse`、`print`。
3. 最后对照测试文件里的实际 IR 输出。

## verifier 是什么

Verifier 是 operation 的合法性检查函数。

ODS 里写：

```tablegen
let hasVerifier = 1;
```

表示这个 operation 需要额外的 C++ 校验逻辑。生成的 wrapper 类里会声明 `verify()`，实现通常写在 `Dialect.cpp`。

### 为什么需要 verifier

`arguments` 和 `results` 只能描述“结构”：

- 有几个 operands。
- 有几个 results。
- 每个 operand/result 的类型约束是什么。

但很多语义约束无法只靠 ODS 表达，例如：

- `toy.constant` 的结果类型必须和常量属性的 shape 一致。
- `toy.return` 的返回值数量必须和 enclosing function 的返回类型一致。
- `toy.transpose` 的结果 shape 必须是输入 shape 的转置。

这些就要交给 verifier。

## `ConstantOp::verify()`

在 `Ops.td` 中，`ConstantOp` 写了：

```tablegen
def ConstantOp : Toy_Op<"constant", [Pure]> {
  let arguments = (ins F64ElementsAttr:$value);
  let results = (outs F64Tensor);
  let hasCustomAssemblyFormat = 1;
  let builders = [
    OpBuilder<(ins "DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,
    OpBuilder<(ins "double":$value)>
  ];
  let hasVerifier = 1;
}
```

对应 `Dialect.cpp`：

```cpp
llvm::LogicalResult ConstantOp::verify() {
  auto resultType = llvm::dyn_cast<RankedTensorType>(getResult().getType());
  if (!resultType)
    return success();

  auto attrType = llvm::cast<RankedTensorType>(getValue().getType());
  if (attrType.getRank() != resultType.getRank())
    ...

  for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim)
    if (attrType.getShape()[dim] != resultType.getShape()[dim])
      ...
}
```

它检查的是：

- 如果 result 不是 ranked tensor，就直接放行。
- 如果 result 是 ranked tensor，那么 attribute 的 ranked tensor 类型必须和 result 的 rank 一致。
- 每个维度也必须一致。

这意味着：

```mlir
%0 = toy.constant dense<[1.0, 2.0]> : tensor<2xf64>
```

是合法的。

但如果写成：

```mlir
%0 = toy.constant dense<[1.0, 2.0]> : tensor<3xf64>
```

就会被 verifier 拒绝，因为 attribute 里只有 2 个元素。

### 这个 verifier 的特点

它不是在检查“值是否正确”，而是在检查“属性类型和 result 类型是否一致”。

## `ReturnOp::verify()`

`ReturnOp` 在 `Ops.td` 里写得很短：

```tablegen
def ReturnOp : Toy_Op<"return", [Pure, HasParent<"FuncOp">, Terminator]> {
  let arguments = (ins Variadic<F64Tensor>:$input);
  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, {}); }]>
  ];
  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];
  let hasVerifier = 1;
}
```

对应 `Dialect.cpp`：

```cpp
llvm::LogicalResult ReturnOp::verify() {
  auto function = cast<FuncOp>((*this)->getParentOp());
  if (getNumOperands() > 1)
    return emitOpError() << "expects at most 1 return operand";

  const auto &results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError() << "does not return the same number of values ...";

  if (!hasOperand())
    return success();

  auto inputType = *operand_type_begin();
  auto resultType = results.front();
  if (inputType == resultType ||
      llvm::isa<UnrankedTensorType>(inputType) ||
      llvm::isa<UnrankedTensorType>(resultType))
    return success();

  return emitError() << "type of return operand ... doesn't match ...";
}
```

它检查三件事：

1. 最多只能有 1 个返回值。
2. 返回值个数必须和 enclosing `toy.func` 的函数结果个数一致。
3. 如果有返回值，operand type 必须和函数结果类型匹配，或者其中一边是 unranked tensor。

### 为什么这里要看 parent function

`toy.return` 不是独立存在的。它的合法性依赖所在函数的签名，所以 `HasParent<"FuncOp">` 很重要。

### 一个直观例子

```mlir
toy.func @foo() -> tensor<2xf64> {
  toy.return
}
```

不合法，因为函数声明了 1 个返回值，但 `toy.return` 没有返回 operand。

再看：

```mlir
toy.func @foo() -> tensor<2xf64> {
  toy.return %0 : tensor<3xf64>
}
```

也不合法，因为类型不一致。

## `TransposeOp::verify()`

`TransposeOp` 在 `Ops.td` 里是：

```tablegen
def TransposeOp : Toy_Op<"transpose"> {
  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor);
  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)
  }];
  let builders = [
    OpBuilder<(ins "Value":$input)>
  ];
  let hasVerifier = 1;
}
```

它的 verifier 在 `Dialect.cpp` 里：

```cpp
llvm::LogicalResult TransposeOp::verify() {
  auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
  auto resultType = llvm::dyn_cast<RankedTensorType>(getType());
  if (!inputType || !resultType)
    return success();

  auto inputShape = inputType.getShape();
  if (!std::equal(inputShape.begin(), inputShape.end(),
                  resultType.getShape().rbegin())) {
    return emitError()
           << "expected result shape to be a transpose of the input";
  }
  return success();
}
```

它只在 input 和 result 都是 ranked tensor 时做严格检查。

意思是：

- 输入 `tensor<2x3xf64>`，输出应该是 `tensor<3x2xf64>`。
- 输入或输出如果是 `tensor<*xf64>`，先不强制检查。

这是很典型的 verifier 风格：只检查它能确认的部分，不把 unranked 情况错误地判死。

## builder 是什么

Builder 是“构造 operation 的便利入口”。

它的目标不是表达语义，而是减少调用者手动填 `OperationState` 的麻烦。

### 为什么需要 builder

如果没有 builder，创建 op 时往往要自己填：

- result types
- operands
- attributes
- location

这很啰嗦，也容易漏。

Builder 让调用代码变成：

```cpp
builder.create<toy::AddOp>(loc, lhs, rhs);
```

而不是手写一大段 `OperationState` 组装逻辑。

## `ConstantOp` 的 builder

`ConstantOp` 在 `Ops.td` 里定义了两个 builder：

```tablegen
OpBuilder<(ins "DenseElementsAttr":$value), [{
  build($_builder, $_state, value.getType(), value);
}]>,
OpBuilder<(ins "double":$value)>
```

对应 `Dialect.cpp` 里还有一个自定义实现：

```cpp
void ConstantOp::build(OpBuilder &builder, OperationState &state,
                       double value) {
  auto dataType = RankedTensorType::get({}, builder.getF64Type());
  auto dataAttribute = DenseElementsAttr::get(dataType, value);
  ConstantOp::build(builder, state, dataType, dataAttribute);
}
```

这表示：

- 当你传入 `DenseElementsAttr` 时，ODS 生成的 builder 可以直接用。
- 当你传入 `double` 时，先把它包装成 `tensor<f64>` 的常量，再复用另一个 builder。

### 为什么 `double` builder 很实用

因为在前端里，很多 literal 常量最初就是一个普通浮点数，比如 `5.5`。

Ch2 的 `scalar.toy`：

```toy
var a<2, 2> = 5.5;
```

最终会先变成：

```mlir
%0 = toy.constant dense<5.500000e+00> : tensor<f64>
```

这就是 `double` builder 服务的场景。

## `AddOp` 和 `MulOp` 的 builder

`AddOp` 和 `MulOp` 在 `Ops.td` 中都写了：

```tablegen
let builders = [
  OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
];
```

`Dialect.cpp` 中的实现类似：

```cpp
void AddOp::build(OpBuilder &builder, OperationState &state,
                  Value lhs, Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}
```

`MulOp` 同理。

这说明这类 binary op 的 builder 做了两件事：

- 添加两个 operands。
- 先把 result type 设成 unranked `tensor<*xf64>`。

### 为什么先用 unranked result type

因为在 Ch2 里，`add` 和 `mul` 还不会做 shape 推导。它们先产生一个类型宽松的结果，后面的分析或 lowering 再处理更细的 shape。

## `GenericCallOp` 的 builder

`GenericCallOp` 在 `Ops.td` 中写了：

```tablegen
let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<F64Tensor>:$inputs);
let results = (outs F64Tensor);
let builders = [
  OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments)>
];
```

`Dialect.cpp` 里对应：

```cpp
void GenericCallOp::build(OpBuilder &builder, OperationState &state,
                          StringRef callee, ArrayRef<Value> arguments) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(arguments);
  state.addAttribute("callee",
                     SymbolRefAttr::get(builder.getContext(), callee));
}
```

这里 builder 做了三件事：

- 添加 unranked result type。
- 添加所有输入 operands。
- 把 callee 名字转成 `SymbolRefAttr`。

这也是一个常见模式：builder 接受更自然的 C++ 参数，内部把它们转成 MLIR 需要的属性和值。

## `FuncOp` 的 builder

`FuncOp` 的 builder 不是简单地把 operands 和 results 塞进 state，而是调用 `buildWithEntryBlock`：

```cpp
void FuncOp::build(OpBuilder &builder, OperationState &state,
                   StringRef name, FunctionType type,
                   ArrayRef<NamedAttribute> attrs) {
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}
```

这说明 `toy.func` 不是普通 operation：

- 它是一个带 region 的符号 operation。
- 它还需要自动创建 entry block。

所以它的 builder 直接帮你把函数框架搭好。

## Assembly format 是什么

Assembly format 决定 operation 的文本长相。

它解决的问题是：

- 这个 op 在 `.mlir` 文本里怎么写。
- 哪些部分是固定符号。
- 哪些部分是 operand、attribute、type。

Ch2 里有两种写法：

1. 自定义 parser/printer
2. 声明式 `assemblyFormat`

## `hasCustomAssemblyFormat`

像 `ConstantOp`、`AddOp`、`MulOp`、`FuncOp` 都用了：

```tablegen
let hasCustomAssemblyFormat = 1;
```

这表示它们不用纯 declarative assembly format，而是自己写 `parse()` / `print()`。

原因通常是：

- 文本格式比较特殊。
- 需要复用通用解析逻辑。
- 需要比 declarative format 更灵活的控制。

## `ConstantOp` 的 parse / print

`Dialect.cpp` 中：

```cpp
mlir::ParseResult ConstantOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  DenseElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(value, "value", result.attributes))
    return failure();

  result.addTypes(value.getType());
  return success();
}

void ConstantOp::print(OpAsmPrinter &printer) {
  printer << " ";
  printer.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  printer << getValue();
}
```

这说明 `toy.constant` 的文本格式是：

```mlir
%0 = toy.constant dense<...> : tensor<...>
```

其中 `dense<...>` 是 attribute，result type 会从 attribute 中推出来。

## `parseBinaryOp` 和 `printBinaryOp`

`AddOp` 和 `MulOp` 共享一套 parser/printer：

```cpp
static ParseResult parseBinaryOp(OpAsmParser &parser, OperationState &result)
static void printBinaryOp(OpAsmPrinter &printer, Operation *op)
```

它们做的是：

- 先解析两个 operands。
- 再解析属性字典。
- 再解析一个类型。

如果类型是 function type，就表示 input/result 类型不完全相同；
如果不是，就把这个 type 同时当作 operands 和 results 的类型。

这让 binary op 的文本形式更紧凑，也更统一。

### 对应的打印效果

如果所有 operand 类型和 result 类型相同，就可能打印成：

```mlir
%0 = toy.add %a, %b : tensor<*xf64>
```

如果类型更复杂，就会打印成 functional type。

## Declarative `assemblyFormat`

相比 `hasCustomAssemblyFormat = 1`，`assemblyFormat` 是更声明式的写法。

Ch2 中典型的例子有：

### `GenericCallOp`

```tablegen
let assemblyFormat = [{
  $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
}];
```

它会打印成：

```mlir
%0 = toy.generic_call @multiply_transpose(%a, %b)
     : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
```

### `PrintOp`

```tablegen
let assemblyFormat = "$input attr-dict `:` type($input)";
```

打印成：

```mlir
toy.print %1 : tensor<2x2xf64>
```

### `ReshapeOp`

```tablegen
let assemblyFormat = [{
  `(` $input `:` type($input) `)` attr-dict `to` type(results)
}];
```

打印成：

```mlir
%1 = toy.reshape(%0 : tensor<6xf64>) to tensor<2x3xf64>
```

### `ReturnOp`

```tablegen
let assemblyFormat = "($input^ `:` type($input))? attr-dict ";
```

表示返回值是可选的：

```mlir
toy.return
toy.return %0 : tensor<*xf64>
```

### `TransposeOp`

```tablegen
let assemblyFormat = [{
  `(` $input `:` type($input) `)` attr-dict `to` type(results)
}];
```

和 `reshape` 类似，但语义不同。

## `invalid.mlir` 在检查什么

测试文件：

```text
mlir/test/Examples/Toy/Ch2/invalid.mlir
```

内容：

```mlir
toy.func @main() {
  %0 = "toy.print"()  : () -> tensor<2x3xf64>
}
```

它故意同时触发多个错误：

- `toy.print` 不应该有 result。
- `toy.print` 应该有一个 input operand。
- `toy.func` body 里缺少 terminator。

这个文件的价值在于：它说明 verifier、parser、以及 region 结构检查是一起工作的。

## 一张总表

| Operation | verifier | builder | assembly format |
| --- | --- | --- | --- |
| `toy.constant` | 检查 result shape 和 attribute 一致 | `DenseElementsAttr` / `double` builder | custom parse/print |
| `toy.add` | 无额外 verifier | 2 个 `Value` builder | custom parse/print |
| `toy.mul` | 无额外 verifier | 2 个 `Value` builder | custom parse/print |
| `toy.print` | 无额外 verifier | 无特殊 builder | declarative `assemblyFormat` |
| `toy.reshape` | 无额外 verifier | 默认 builder / 生成 builder | declarative `assemblyFormat` |
| `toy.generic_call` | 无额外 verifier | `StringRef` + `ArrayRef<Value>` builder | declarative `assemblyFormat` |
| `toy.func` | 由 function interface 支持解析/打印 | `buildWithEntryBlock` | custom parse/print |
| `toy.return` | 检查返回值个数和类型 | no-operand builder | declarative `assemblyFormat` |
| `toy.transpose` | 检查转置后的 shape | 1 个 `Value` builder | declarative `assemblyFormat` |

## 常见误区

### 误区 1：builder 只是语法糖

不完全是。

builder 不只是省代码，它还承担了默认类型推导、属性包装、entry block 创建等职责。

### 误区 2：verifier 会替代类型约束

不会。

类型约束负责“先天结构合法”，verifier 负责“语义进一步合法”。

### 误区 3：`hasCustomAssemblyFormat` 和 `assemblyFormat` 是同一件事

不是。

- `hasCustomAssemblyFormat = 1` 说明你自己写 parse/print。
- `assemblyFormat` 说明你让 TableGen 生成 parser/printer。

## 动手观察

### 观察 `toy.constant`

在 `scalar.toy` 里找：

```mlir
%0 = toy.constant dense<5.500000e+00> : tensor<f64>
```

对照 `ConstantOp::build(double)` 和 `ConstantOp::verify()`。

你要能解释：

- 为什么 `5.5` 会变成 `tensor<f64>`。
- 为什么这里不需要手写 result type。
- verifier 会检查什么。

### 观察 `toy.return`

在 `codegen.toy` 里找：

```mlir
toy.return
toy.return %4 : tensor<*xf64>
```

对照 `ReturnOp::verify()`。

你要能解释：

- 为什么有的函数没返回值。
- 为什么 `toy.return` 的 operand 数量必须和函数签名一致。

### 观察 `toy.generic_call`

在 `codegen.toy` 里找：

```mlir
%9 = toy.generic_call @multiply_transpose(%6, %8)
     : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
```

对照 `GenericCallOp::build()` 和 `assemblyFormat`。

你要能解释：

- `@multiply_transpose` 是什么。
- `%6`、`%8` 是什么。
- result type 为什么先是 `tensor<*xf64>`。

## 本节练习

### 练习 1：解释 `hasVerifier`

阅读：

```tablegen
let hasVerifier = 1;
```

回答：

- 这个字段的作用是什么。
- 为什么 `toy.constant`、`toy.return`、`toy.transpose` 需要它。
- 为什么 `toy.add` 和 `toy.mul` 不一定需要它。

### 练习 2：拆解 `ConstantOp::verify()`

阅读 `ConstantOp::verify()`。

回答：

- 它先检查什么类型。
- 它为什么只在 result 是 ranked tensor 时继续检查。
- 它是如何比较 rank 和 shape 的。
- 给出一个会失败的例子。

### 练习 3：拆解 `ReturnOp::verify()`

阅读 `ReturnOp::verify()`。

回答：

- 为什么它需要先拿到 parent `FuncOp`。
- 它怎样检查返回值个数。
- 它怎样检查返回值类型。
- 为什么 unranked tensor 可以放宽检查。

### 练习 4：对比 builder 形式

阅读：

```tablegen
OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments)>
OpBuilder<(ins "double":$value)>
```

回答：

- 这三种 builder 各自服务哪种创建场景。
- 哪些需要在 C++ 里额外写实现。
- 哪些只是 ODS 声明就够了。

### 练习 5：区分 custom parser/printer 和 declarative assembly format

阅读：

```tablegen
let hasCustomAssemblyFormat = 1;
let assemblyFormat = "...";
```

回答：

- 两者的区别是什么。
- `toy.constant` 为什么更适合 custom parser/printer。
- `toy.print` 为什么更适合 declarative format。

### 练习 6：分析 `invalid.mlir`

阅读：

```mlir
toy.func @main() {
  %0 = "toy.print"()  : () -> tensor<2x3xf64>
}
```

回答：

- 它违反了 `PrintOp` 的哪个输入约束。
- 它违反了 `PrintOp` 的哪个输出约束。
- 它违反了 `ReturnOp` / block terminator 的什么要求。

### 练习 7：补写一张“生成路径图”

把下面这条链路写成你自己的话：

```text
builder.create<...>()
  -> ODS builder
  -> OperationState
  -> verifier
  -> parse/print
```

至少说清楚：

- 哪一步负责构造。
- 哪一步负责校验。
- 哪一步负责文本序列化。

## 本节小结

本节要记住的核心是：

```text
ODS 负责声明
builder 负责构造
verifier 负责合法性
assembly format 负责文本形式
```

在 Ch2 中：

- `toy.constant` 用 verifier 校验常量属性和 result type。
- `toy.return` 用 verifier 校验函数返回语义。
- `toy.transpose` 用 verifier 校验转置 shape。
- `toy.constant`、`toy.add`、`toy.mul`、`toy.func` 用 custom parse/print。
- `toy.print`、`toy.reshape`、`toy.generic_call`、`toy.return`、`toy.transpose` 可以用 declarative assembly format。

学完这一课后，你应该已经能从一个 `Ops.td` 定义，推断出一个 op 在创建、验证、打印、解析时分别会做什么。

下一课会继续沿着 `FuncOp` 和 `GenericCallOp` 往下走，重点看函数、调用、符号与作用域。

## 学习记录模板

```text
本节主题：Verifier、Builder 与自定义 Assembly Format
我读过的源码：
我观察过的测试：
我能解释的 verifier：
我能解释的 builder：
我能解释的 assembly format：
我还不理解的问题：
下一步要验证的小实验：
```
