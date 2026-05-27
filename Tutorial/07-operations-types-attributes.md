# 第 7 课：Operation 的参数、结果、类型和属性

## 本节定位

第六课已经说明 Toy operations 来自 `Ops.td`，并通过 TableGen 生成 C++ wrapper 和注册代码。

本节继续深入 `Ops.td`，重点学习每个 operation 的数据模型：

```text
arguments
results
attributes
operands
types
traits
constraints
```

换句话说，本节要回答：

- `toy.constant` 为什么没有 operand，但有一个 `dense<...>` 常量？
- `toy.add` 为什么有两个输入和一个输出？
- `toy.print` 为什么没有 result？
- `toy.reshape` 为什么要求结果是静态 shape tensor？
- `toy.generic_call` 里的 `@callee` 和 `%arg` 为什么不是同一种东西？

本节仍然以 Ch2 为范围。第 8 课会继续讲 verifier、builder 和 custom assembly format。

## 本节目标

- 理解 ODS 中 `let arguments = (ins ...)` 的含义。
- 理解 ODS 中 `let results = (outs ...)` 的含义。
- 区分 operation operand、attribute、result。
- 理解 `F64Tensor`、`F64ElementsAttr`、`StaticShapeTensorOf<[F64]>` 等约束。
- 理解 ranked tensor、unranked tensor、scalar tensor 在 Toy 中的表现。
- 理解常见 trait：`Pure`、`Terminator`、`HasParent<"FuncOp">`、`IsolatedFromAbove`。
- 能为 Ch2 中的 Toy operations 整理一张语义表。

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

MLIR 公共约束定义：

```text
mlir/include/mlir/IR/CommonTypeConstraints.td
mlir/include/mlir/IR/CommonAttrConstraints.td
```

建议阅读顺序：

1. 先读 `Ops.td` 中每个 operation 的 `arguments` 和 `results`。
2. 再对照 Ch2 的 `codegen.toy` 输出。
3. 再看 `scalar.toy` 中 scalar constant 和 reshape。
4. 最后看 `invalid.mlir`，理解不合法 operation 大概违反了哪些约束。

## `arguments` 和 `results`

在 ODS 中，operation 的输入和输出主要由两个字段描述：

```tablegen
let arguments = (ins ...);
let results = (outs ...);
```

### `arguments`

`arguments` 表示 operation 的输入。

注意，ODS 里的 argument 可能是两类东西：

- SSA operand，也就是运行时使用的 value。
- attribute，也就是编译期常量元信息。

例如：

```tablegen
let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
```

这里 `lhs`、`rhs` 是 SSA operands。

而：

```tablegen
let arguments = (ins F64ElementsAttr:$value);
```

这里 `value` 是 attribute，不是 SSA operand。

所以 ODS 中的 `arguments` 不要直接等同于 “operands”。它是 operation 输入声明，里面可以包含 operand，也可以包含 attribute。

### `results`

`results` 表示 operation 产生的 SSA result。

例如：

```tablegen
let results = (outs F64Tensor);
```

表示这个 operation 产生一个 f64 tensor 类型的 SSA value。

如果没有 `results` 字段，或者 `results` 为空，这个 operation 就不产生 SSA value。

例如 `toy.print` 没有 `results`，所以：

```mlir
toy.print %0 : tensor<2x3xf64>
```

前面没有 `%1 =`。

## `ins` 和 `outs`

`ins` 和 `outs` 是 ODS 用来描述输入和输出列表的语法。

例子：

```tablegen
let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
let results = (outs F64Tensor);
```

可以读作：

```text
inputs:
  lhs: F64Tensor
  rhs: F64Tensor

outputs:
  one F64Tensor result
```

带名字的输入：

```tablegen
F64Tensor:$lhs
```

会生成访问器，方便 C++ 代码使用。

结果也可以命名，不过 Ch2 的很多 result 没有显式名字。

## Type constraint 和 attribute constraint

`F64Tensor`、`F64ElementsAttr`、`StaticShapeTensorOf<[F64]>` 都是约束。

它们告诉 MLIR：这个 operand、attribute 或 result 必须符合什么类型条件。

这些约束不是 Toy 自己手写的类型系统，而是来自 MLIR 的公共 ODS 定义：

```text
mlir/include/mlir/IR/CommonTypeConstraints.td
mlir/include/mlir/IR/CommonAttrConstraints.td
```

本节关注 Toy 中出现的几个。

## `F64Tensor`

定义位置：

```text
mlir/include/mlir/IR/CommonTypeConstraints.td
```

含义：

```text
TensorOf<[F64]>
```

也就是元素类型为 `f64` 的 tensor。

它可以匹配：

```mlir
tensor<*xf64>
tensor<f64>
tensor<2xf64>
tensor<2x3xf64>
```

它不应该匹配：

```mlir
tensor<2xi32>
memref<2xf64>
i32
f64
```

注意：`tensor<f64>` 是 0-rank tensor，也就是 scalar tensor。它仍然是 tensor，不是裸 `f64`。

## `F64ElementsAttr`

定义位置：

```text
mlir/include/mlir/IR/CommonAttrConstraints.td
```

含义：元素类型为 f64 的 elements attribute。

它常出现在 `toy.constant`：

```tablegen
let arguments = (ins F64ElementsAttr:$value);
```

对应 MLIR：

```mlir
%0 = toy.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
```

这里的：

```mlir
dense<[1.000000e+00, 2.000000e+00]>
```

就是 `value` attribute。

它是编译期常量数据，不是 operand。

## `StaticShapeTensorOf<[F64]>`

定义位置：

```text
mlir/include/mlir/IR/CommonTypeConstraints.td
```

在 Toy 中用于 `ReshapeOp` 的 result：

```tablegen
let results = (outs StaticShapeTensorOf<[F64]>);
```

含义：结果必须是静态 shape 的 f64 tensor。

例如：

```mlir
tensor<2x3xf64>
tensor<6xf64>
tensor<f64>
```

这些都是静态 shape。

而：

```mlir
tensor<*xf64>
```

是 unranked tensor，不是静态 shape。

这和 `toy.reshape` 的语义一致：reshape 目标形状必须明确。

## Ranked、Unranked、Scalar Tensor

Toy 中经常看到三类 tensor：

### Unranked tensor

```mlir
tensor<*xf64>
```

含义：元素是 f64，但 rank/shape 未知。

常见于泛型函数参数：

```mlir
toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>)
```

### Ranked tensor

```mlir
tensor<2x3xf64>
tensor<6xf64>
```

含义：rank 和每个维度都已知。

常见于 literal 和 reshape 结果。

### Scalar tensor

```mlir
tensor<f64>
```

含义：0-rank tensor，元素是 f64。

对应 `scalar.toy`：

```toy
var a<2, 2> = 5.5;
```

先生成：

```mlir
%0 = toy.constant dense<5.500000e+00> : tensor<f64>
```

再 reshape 到：

```mlir
%1 = toy.reshape(%0 : tensor<f64>) to tensor<2x2xf64>
```

## Operation 语义表

下面逐个看 Ch2 中的 Toy operations。

## `ConstantOp`

ODS：

```tablegen
def ConstantOp : Toy_Op<"constant", [Pure]> {
  let arguments = (ins F64ElementsAttr:$value);
  let results = (outs F64Tensor);
  ...
}
```

语义：

- full name：`toy.constant`
- 输入 attribute：`value`
- SSA operands：无
- results：一个 `F64Tensor`
- trait：`Pure`

对应 MLIR：

```mlir
%0 = toy.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
```

拆解：

- `%0` 是 result。
- `dense<[...]>` 是 attribute。
- `tensor<2xf64>` 是 result type。
- 没有 operand。

为什么 `value` 在 `arguments` 里却不是 operand？

因为 ODS 的 `arguments` 包含 attribute 和 operand 两类输入。`F64ElementsAttr` 是 attribute constraint，所以它不是 SSA operand。

## `AddOp`

ODS：

```tablegen
def AddOp : Toy_Op<"add"> {
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);
}
```

语义：

- full name：`toy.add`
- operands：`lhs`、`rhs`
- attributes：无固定 attribute
- results：一个 `F64Tensor`

对应 MLIR：

```mlir
%0 = toy.add %a, %b : tensor<2x3xf64>
```

两个输入都是 SSA value。

## `MulOp`

ODS：

```tablegen
def MulOp : Toy_Op<"mul"> {
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);
}
```

语义和 `AddOp` 类似：

- full name：`toy.mul`
- operands：两个 f64 tensor。
- results：一个 f64 tensor。

对应 Ch2：

```mlir
%2 = toy.mul %0, %1 : tensor<*xf64>
```

## `PrintOp`

ODS：

```tablegen
def PrintOp : Toy_Op<"print"> {
  let arguments = (ins F64Tensor:$input);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}
```

语义：

- full name：`toy.print`
- operands：一个 `input`
- results：无

对应 MLIR：

```mlir
toy.print %1 : tensor<2x2xf64>
```

注意：`toy.print` 不应该写成：

```mlir
%0 = toy.print ...
```

因为它没有 result。

`invalid.mlir` 正是故意写了类似非法形式。

## `ReshapeOp`

ODS：

```tablegen
def ReshapeOp : Toy_Op<"reshape"> {
  let arguments = (ins F64Tensor:$input);
  let results = (outs StaticShapeTensorOf<[F64]>);
}
```

语义：

- full name：`toy.reshape`
- operands：一个 f64 tensor input。
- results：一个静态 shape f64 tensor。

对应 MLIR：

```mlir
%1 = toy.reshape(%0 : tensor<6xf64>) to tensor<2x3xf64>
```

这里：

- `%0` 是 operand。
- `%1` 是 result。
- 输入 type 是 `tensor<6xf64>`。
- result type 是 `tensor<2x3xf64>`，是静态 shape tensor。

如果 result 是 `tensor<*xf64>`，就不符合 `StaticShapeTensorOf<[F64]>` 的语义。

## `TransposeOp`

ODS：

```tablegen
def TransposeOp : Toy_Op<"transpose"> {
  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor);
  let hasVerifier = 1;
}
```

语义：

- full name：`toy.transpose`
- operands：一个 f64 tensor。
- results：一个 f64 tensor。
- 有额外 verifier。

对应 MLIR：

```mlir
%0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
```

`F64Tensor` 本身只说明输入和输出都是 f64 tensor。转置的 shape 是否正确，需要 `TransposeOp::verify()` 检查。

例如如果输入是 `tensor<2x3xf64>`，结果应该是 `tensor<3x2xf64>`。

## `GenericCallOp`

ODS：

```tablegen
def GenericCallOp : Toy_Op<"generic_call"> {
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<F64Tensor>:$inputs);
  let results = (outs F64Tensor);
}
```

语义：

- full name：`toy.generic_call`
- attribute：`callee`
- operands：可变数量的 `inputs`
- results：一个 f64 tensor

对应 MLIR：

```mlir
%0 = toy.generic_call @multiply_transpose(%a, %b)
     : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
```

拆解：

- `@multiply_transpose` 是 `FlatSymbolRefAttr`。
- `%a`、`%b` 是 SSA operands。
- `%0` 是 result。

这体现了一个关键区别：

```text
@multiply_transpose 是 symbol attribute
%a / %b 是 SSA value operands
```

## `FuncOp`

ODS：

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
}
```

语义：

- full name：`toy.func`
- attributes：函数名、函数类型、参数属性、结果属性。
- regions：函数体 body。
- results：无 operation result。
- traits/interfaces：`FunctionOpInterface`、`IsolatedFromAbove`。

对应 MLIR：

```mlir
toy.func @main() {
  ...
}
```

或者：

```mlir
toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>)
  -> tensor<*xf64> {
  ...
}
```

注意：`toy.func` 不是通过 SSA result 表示函数。它是一个 symbol operation，函数名是 symbol attribute。

## `ReturnOp`

ODS：

```tablegen
def ReturnOp : Toy_Op<"return", [Pure, HasParent<"FuncOp">,
                                 Terminator]> {
  let arguments = (ins Variadic<F64Tensor>:$input);
}
```

语义：

- full name：`toy.return`
- operands：0 个或多个 f64 tensor。Ch2 verifier 进一步限制最多 1 个。
- results：无。
- traits：`Pure`、`HasParent<"FuncOp">`、`Terminator`。

对应 MLIR：

```mlir
toy.return
toy.return %0 : tensor<*xf64>
```

`HasParent<"FuncOp">` 表示 `toy.return` 必须出现在 `toy.func` 内。

`Terminator` 表示它是 block 结束 operation。

## Trait：给 operation 附加通用性质

Trait 是 MLIR 给 operation 附加行为或约束的一种机制。

Ch2 中常见 trait：

### `Pure`

出现在：

```tablegen
ConstantOp : Toy_Op<"constant", [Pure]>
ReturnOp : Toy_Op<"return", [Pure, ...]>
```

含义：operation 没有可观察副作用，便于优化。

例如 dead code elimination 可以更放心地删除没有使用结果的 pure operation。

### `Terminator`

出现在：

```tablegen
ReturnOp : Toy_Op<"return", [..., Terminator]>
```

含义：这个 operation 结束一个 block。

函数 body 的 block 需要 terminator，所以 `toy.return` 承担这个角色。

### `HasParent<"FuncOp">`

出现在：

```tablegen
ReturnOp : Toy_Op<"return", [HasParent<"FuncOp">, ...]>
```

含义：`toy.return` 必须直接位于 `toy.func` 内。

### `IsolatedFromAbove`

出现在：

```tablegen
FuncOp : Toy_Op<"func", [FunctionOpInterface, IsolatedFromAbove]>
```

含义：函数 body 不能随意捕获外层 SSA value。这对函数、模块这类 symbol/region operation 很重要。

### `FunctionOpInterface`

它是 interface，不是普通 trait，但在 ODS 里和 trait 一起列出。

它让 `toy.func` 具备函数类 operation 的通用能力，例如读取 function type、参数类型、结果类型等。

## `invalid.mlir` 为什么非法

测试：

```text
mlir/test/Examples/Toy/Ch2/invalid.mlir
```

内容：

```mlir
toy.func @main() {
  %0 = "toy.print"()  : () -> tensor<2x3xf64>
}
```

注释说明它不合法：

```text
toy.print should not return a value.
toy.print should take an argument.
There should be a block terminator.
```

对照 `PrintOp` 定义：

```tablegen
let arguments = (ins F64Tensor:$input);
```

说明 `toy.print` 应该有一个 input operand。

它没有 `results` 字段，说明不应该返回 value。

对照 `ReturnOp` 的 `Terminator`，函数体应该以 terminator 结束。这里没有 `toy.return`。

所以这个 IR 同时违反了多个 operation/region 结构约束。

## Operation 语义总表

| Operation | ODS definition | operands | attributes | results | traits / interfaces |
| --- | --- | --- | --- | --- | --- |
| `toy.constant` | `ConstantOp : Toy_Op<"constant", [Pure]>` | 无 | `value: F64ElementsAttr` | 1 个 `F64Tensor` | `Pure` |
| `toy.add` | `AddOp : Toy_Op<"add">` | `lhs: F64Tensor`, `rhs: F64Tensor` | 无固定 attribute | 1 个 `F64Tensor` | 无显式 trait |
| `toy.mul` | `MulOp : Toy_Op<"mul">` | `lhs: F64Tensor`, `rhs: F64Tensor` | 无固定 attribute | 1 个 `F64Tensor` | 无显式 trait |
| `toy.print` | `PrintOp : Toy_Op<"print">` | `input: F64Tensor` | 无固定 attribute | 无 | 无显式 trait |
| `toy.reshape` | `ReshapeOp : Toy_Op<"reshape">` | `input: F64Tensor` | 无固定 attribute | 1 个 `StaticShapeTensorOf<[F64]>` | 无显式 trait |
| `toy.transpose` | `TransposeOp : Toy_Op<"transpose">` | `input: F64Tensor` | 无固定 attribute | 1 个 `F64Tensor` | 有 verifier |
| `toy.generic_call` | `GenericCallOp : Toy_Op<"generic_call">` | variadic `inputs: F64Tensor` | `callee: FlatSymbolRefAttr` | 1 个 `F64Tensor` | 无显式 trait |
| `toy.func` | `FuncOp : Toy_Op<"func", [...]>` | 无普通 SSA operand | `sym_name`, `function_type`, optional attrs | 无 | `FunctionOpInterface`, `IsolatedFromAbove` |
| `toy.return` | `ReturnOp : Toy_Op<"return", [...]>` | variadic `input: F64Tensor` | 无固定 attribute | 无 | `Pure`, `HasParent<"FuncOp">`, `Terminator` |

## 常见误区

### 误区 1：ODS 的 `arguments` 都是 SSA operands

不对。

`arguments` 可以包含 operand，也可以包含 attribute。

例如：

```tablegen
F64ElementsAttr:$value
```

是 attribute。

```tablegen
F64Tensor:$input
```

是 operand。

### 误区 2：`toy.constant` 的 dense 数据是 operand

不是。

`dense<...>` 是 attribute，常量数据直接附在 operation 上。

### 误区 3：`toy.func` 有 result

`toy.func` 自身没有 SSA result。

函数返回值是函数类型的一部分，并通过函数体内的 `toy.return` 表示控制流返回。

### 误区 4：`tensor<f64>` 是裸 `f64`

不是。

`tensor<f64>` 是 0-rank tensor。它仍然满足 `F64Tensor`。

### 误区 5：`F64Tensor` 一定是静态 shape

不是。

`F64Tensor` 可以是：

```mlir
tensor<*xf64>
tensor<2x3xf64>
tensor<f64>
```

如果要求静态 shape，需要像 `ReshapeOp` 那样使用：

```tablegen
StaticShapeTensorOf<[F64]>
```

## 动手观察

### 观察 `toy.constant`

在 Ch2 输出中找：

```mlir
%0 = toy.constant dense<...> : tensor<...>
```

然后回到 `Ops.td`：

```tablegen
let arguments = (ins F64ElementsAttr:$value);
let results = (outs F64Tensor);
```

确认：

- `dense<...>` 对应 `$value`。
- `%0` 对应 result。
- 没有 operand。

### 观察 `toy.reshape`

在 `scalar.toy` 输出中找：

```mlir
%1 = toy.reshape(%0 : tensor<f64>) to tensor<2x2xf64>
```

对照：

```tablegen
let arguments = (ins F64Tensor:$input);
let results = (outs StaticShapeTensorOf<[F64]>);
```

确认：

- `%0` 是 input operand。
- `%1` 是 result。
- result type 是静态 shape tensor。

### 观察 `toy.generic_call`

在 `codegen.toy` 输出中找：

```mlir
%0 = toy.generic_call @multiply_transpose(%a, %b) : ...
```

对照：

```tablegen
let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<F64Tensor>:$inputs);
```

确认：

- `@multiply_transpose` 是 callee attribute。
- `%a`、`%b` 是 input operands。

## 本节练习

### 练习 1：标注 `ConstantOp`

阅读：

```tablegen
def ConstantOp : Toy_Op<"constant", [Pure]> {
  let arguments = (ins F64ElementsAttr:$value);
  let results = (outs F64Tensor);
}
```

回答：

- full operation name 是什么。
- 它有几个 SSA operands。
- 它有哪些 attributes。
- 它有几个 results。
- result type 受什么约束。
- `Pure` 表示什么。

### 练习 2：区分 `AddOp` 和 `PrintOp`

对比：

```tablegen
def AddOp : Toy_Op<"add"> {
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);
}

def PrintOp : Toy_Op<"print"> {
  let arguments = (ins F64Tensor:$input);
}
```

回答：

- 两者各有几个 operands。
- 两者各有几个 results。
- 为什么 `toy.add` 前面有 `%0 =`，而 `toy.print` 没有。

### 练习 3：解释 `scalar.toy`

阅读：

```text
mlir/test/Examples/Toy/Ch2/scalar.toy
```

回答：

- `5.5` 生成的 `toy.constant` result type 是什么。
- `tensor<f64>` 是什么含义。
- `toy.reshape` 的 input type 是什么。
- `toy.reshape` 的 result type 是什么。
- 为什么 result type 满足 `StaticShapeTensorOf<[F64]>`。

### 练习 4：解释 `GenericCallOp`

阅读：

```tablegen
let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<F64Tensor>:$inputs);
let results = (outs F64Tensor);
```

回答：

- `callee` 是 operand 还是 attribute。
- `inputs` 是 operand 还是 attribute。
- `Variadic<F64Tensor>` 表示什么。
- `@multiply_transpose(%a, %b)` 中哪些是 symbol，哪些是 SSA value。

### 练习 5：解释 `ReturnOp` traits

阅读：

```tablegen
def ReturnOp : Toy_Op<"return", [Pure, HasParent<"FuncOp">,
                                 Terminator]> {
  let arguments = (ins Variadic<F64Tensor>:$input);
}
```

回答：

- `Pure` 表示什么。
- `HasParent<"FuncOp">` 限制了什么。
- `Terminator` 表示什么。
- 为什么 `toy.return` 没有 result。

### 练习 6：分析 `invalid.mlir`

阅读：

```mlir
toy.func @main() {
  %0 = "toy.print"()  : () -> tensor<2x3xf64>
}
```

回答：

- 它为什么违反 `PrintOp` 的输入要求。
- 它为什么违反 `PrintOp` 的 result 要求。
- 它为什么缺少 terminator。

### 练习 7：整理 Toy operation 语义表

自己整理一张表，至少包含：

```text
toy.constant
toy.add
toy.mul
toy.print
toy.reshape
toy.transpose
toy.generic_call
toy.func
toy.return
```

每行写：

- operands。
- attributes。
- results。
- type constraints。
- traits/interfaces。

## 本节小结

本节最重要的是能读懂 ODS operation 的数据模型。

需要记住：

```text
let arguments = (ins ...)
  -> operation 输入，可以是 operands，也可以是 attributes

let results = (outs ...)
  -> operation 产生的 SSA results

F64Tensor
  -> 元素类型为 f64 的 tensor，rank/shape 可已知也可未知

F64ElementsAttr
  -> f64 elements attribute，常用于 dense constant

StaticShapeTensorOf<[F64]>
  -> 静态 shape 的 f64 tensor
```

也要记住几个关键区分：

- `dense<...>` 是 attribute，不是 operand。
- `%0`、`%arg0` 是 SSA value。
- `@main`、`@multiply_transpose` 是 symbol。
- `toy.print` 和 `toy.return` 没有 result。
- `toy.func` 自身没有 SSA result，但有 region 和 function type。

下一节课会继续看 `Ops.td` 和 `Dialect.cpp`，重点是 verifier、builder、自定义 parser/printer 和 assembly format。

## 学习记录模板

```text
本节主题：Operation 的参数、结果、类型和属性
我读过的源码：
我观察过的测试：
我能解释的 operation：
我整理的语义表：
我还不理解的问题：
下一步要验证的小实验：
```
