# 第 6 课：Toy Dialect 与 ODS/TableGen

## 本节定位

第四课学习了 MLIR 的基本 IR 模型，第五课学习了 Ch2 如何从 AST 生成 Toy MLIR。

现在要回答一个关键问题：`toy.constant`、`toy.func`、`toy.reshape` 这些 operation 是从哪里来的？

答案是：它们在 `Ops.td` 中用 ODS 定义，再由 TableGen 生成 C++ 声明和实现片段，最后接入 `Dialect.h` 和 `Dialect.cpp`。

本节主线是：

```text
Ops.td
  -> mlir_tablegen
  -> Dialect.h.inc / Dialect.cpp.inc
  -> Ops.h.inc / Ops.cpp.inc
  -> Dialect.h / Dialect.cpp
  -> ToyDialect 注册 Toy operations
  -> MLIRGen 可以 builder.create<ConstantOp>(...)
```

本节重点是 ODS 文件结构、TableGen 生成链路和 C++ 接入方式。`arguments`、`results`、trait、verifier、builder、custom assembly format 的细节会在第 7、8 课继续深入。

## 本节目标

- 理解 Dialect 在 MLIR 中的作用。
- 理解 Toy Dialect 如何在 `Ops.td` 中声明。
- 理解 `Toy_Op` 基类模板的作用。
- 理解 mnemonic 和 operation name 的关系。
- 理解 ODS 如何定义 operation。
- 理解 TableGen 如何生成 `.inc` 文件。
- 理解 `Dialect.h` 和 `Dialect.cpp` 如何包含生成文件。
- 理解 `GET_OP_CLASSES`、`GET_OP_LIST`、`Dialect.cpp.inc` 的接入点。
- 能从 `Ops.td` 找到某个 `toy.*` operation 的定义。

## 本节关键文件

源码：

```text
mlir/examples/toy/Ch2/include/toy/Ops.td
mlir/examples/toy/Ch2/include/toy/Dialect.h
mlir/examples/toy/Ch2/mlir/Dialect.cpp
mlir/examples/toy/Ch2/include/CMakeLists.txt
mlir/examples/toy/Ch2/include/toy/CMakeLists.txt
```

辅助观察：

```text
mlir/examples/toy/Ch2/mlir/MLIRGen.cpp
mlir/test/Examples/Toy/Ch2/codegen.toy
```

建议阅读顺序：

1. 先看 `Ops.td` 顶部的 include、`Toy_Dialect`、`Toy_Op`。
2. 再看 `ConstantOp`、`AddOp`、`MulOp`、`TransposeOp` 的定义。
3. 再看 `include/toy/CMakeLists.txt` 中的 `mlir_tablegen(...)`。
4. 再看 `Dialect.h` 如何 include 生成的 `.inc`。
5. 最后看 `Dialect.cpp` 如何注册 operations。

## Dialect 是什么

Dialect 是 MLIR 的扩展单元。它为一组 operation、type、attribute 提供命名空间和语义集合。

Toy Dialect 的命名空间是：

```text
toy
```

所以 Toy operations 叫：

```text
toy.constant
toy.add
toy.mul
toy.func
toy.generic_call
toy.print
toy.reshape
toy.return
toy.transpose
```

这里的：

```text
toy.constant
```

可以拆成：

```text
dialect namespace: toy
mnemonic: constant
full operation name: toy.constant
```

Dialect 的意义是：

- 把 Toy 语言的高层语义组织在同一个命名空间里。
- 让 MLIR 知道这些 operation 如何解析、打印、验证和构造。
- 让后续 pass 可以识别和变换 Toy operations。

## ODS 和 TableGen 是什么

ODS 是 Operation Definition Specification，操作定义规范。

它基于 TableGen，用声明式方式定义 MLIR dialect 和 operation。

不用 ODS 时，你需要手写大量 C++：

- operation class。
- operation name。
- operand/result accessor。
- parser/printer 声明。
- verifier 声明。
- builder 声明。
- trait 和 interface 接入。
- dialect registration。

使用 ODS 后，这些大部分可以从 `.td` 文件生成。

Toy 的 ODS 文件是：

```text
mlir/examples/toy/Ch2/include/toy/Ops.td
```

它是第六课的主角。

## `Ops.td` 顶部 include

文件开头：

```tablegen
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
```

这些 include 提供 ODS 中会用到的基础定义：

- `OpBase.td`：`Dialect`、`Op`、type constraint、attribute constraint、trait 等基础设施。
- `FunctionInterfaces.td`：函数类 operation 相关 interface，例如 `FunctionOpInterface`。
- `SymbolInterfaces.td`：symbol 相关支持。
- `SideEffectInterfaces.td`：side effect / pure operation 相关 trait 或 interface。

现阶段不需要深入这些文件，只要知道 `Ops.td` 依赖它们提供的 ODS 基础类型。

## 定义 Toy Dialect

`Ops.td` 中 Toy Dialect 的定义：

```tablegen
def Toy_Dialect : Dialect {
  let name = "toy";
  let cppNamespace = "::mlir::toy";
}
```

含义：

- `def Toy_Dialect`：定义一个 TableGen record，名字叫 `Toy_Dialect`。
- `: Dialect`：它继承 ODS 的 `Dialect` 基类。
- `let name = "toy"`：dialect namespace 是 `toy`。
- `let cppNamespace = "::mlir::toy"`：生成的 C++ 类放在 `mlir::toy` 命名空间下。

这个定义最终会生成 Toy dialect 的 C++ 声明和实现片段。

在源码中，生成文件会被接入：

```cpp
#include "toy/Dialect.h.inc"
#include "toy/Dialect.cpp.inc"
```

## `Toy_Op` 基类模板

`Ops.td` 中定义了一个 Toy operation 的基类模板：

```tablegen
class Toy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;
```

这是为了避免每个 Toy operation 都重复写：

```tablegen
Op<Toy_Dialect, "...", [...]>
```

它的参数：

- `mnemonic`：operation 在 dialect 内部的短名字。
- `traits`：operation 的 trait 列表，默认空。

例如：

```tablegen
def ConstantOp : Toy_Op<"constant", [Pure]> {
  ...
}
```

这里：

- mnemonic 是 `constant`。
- dialect 是 `Toy_Dialect`。
- full operation name 是 `toy.constant`。
- trait 是 `Pure`。

再例如：

```tablegen
def MulOp : Toy_Op<"mul"> {
  ...
}
```

这里 full operation name 是：

```text
toy.mul
```

## mnemonic 与 operation name

这个关系非常重要：

```text
full operation name = dialect name + "." + mnemonic
```

例子：

| ODS 定义 | dialect name | mnemonic | full operation name |
| --- | --- | --- | --- |
| `def ConstantOp : Toy_Op<"constant">` | `toy` | `constant` | `toy.constant` |
| `def AddOp : Toy_Op<"add">` | `toy` | `add` | `toy.add` |
| `def MulOp : Toy_Op<"mul">` | `toy` | `mul` | `toy.mul` |
| `def FuncOp : Toy_Op<"func">` | `toy` | `func` | `toy.func` |
| `def TransposeOp : Toy_Op<"transpose">` | `toy` | `transpose` | `toy.transpose` |

在 `MLIRGen.cpp` 里写：

```cpp
builder.create<ConstantOp>(...)
```

最终打印出来就是：

```mlir
toy.constant
```

因为 `ConstantOp` 的 ODS mnemonic 是 `constant`，所属 dialect 是 `toy`。

## 一个最小 operation 定义长什么样

先看 `PrintOp`：

```tablegen
def PrintOp : Toy_Op<"print"> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input tensor, and produces
    no results.
  }];

  let arguments = (ins F64Tensor:$input);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}
```

从这个定义可以读出：

- C++ wrapper 类名是 `PrintOp`。
- MLIR operation 名字是 `toy.print`。
- 它有一个输入，名字是 `$input`。
- 输入类型约束是 `F64Tensor`。
- 它没有 `results` 字段，所以不产生 SSA result。
- 它有 declarative assembly format。

对应 MLIR：

```mlir
toy.print %0 : tensor<2x3xf64>
```

## `ConstantOp` 定义概览

`ConstantOp`：

```tablegen
def ConstantOp : Toy_Op<"constant", [Pure]> {
  let summary = "constant";
  let description = [{ ... }];

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

本节只抓结构：

- C++ wrapper：`ConstantOp`
- operation name：`toy.constant`
- trait：`Pure`
- argument：一个 attribute，`value`
- result：一个 tensor value
- 有自定义 assembly format
- 有 builder
- 有 verifier

对应 MLIR：

```mlir
%0 = toy.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
```

更细的 `arguments/results/builders/verifier/assemblyFormat` 会在第 7、8 课展开。

## `AddOp` 与 `MulOp`

`AddOp`：

```tablegen
def AddOp : Toy_Op<"add"> {
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);
  let hasCustomAssemblyFormat = 1;
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}
```

`MulOp` 和它结构类似，mnemonic 是 `"mul"`。

对应关系：

```cpp
builder.create<AddOp>(location, lhs, rhs)
  -> toy.add

builder.create<MulOp>(location, lhs, rhs)
  -> toy.mul
```

这里 ODS 说明了：

- 两个输入 operand：`lhs`、`rhs`。
- 一个输出 result。
- 输入和输出都受 `F64Tensor` 约束。

## `FuncOp`

`FuncOp` 定义：

```tablegen
def FuncOp : Toy_Op<"func", [
    FunctionOpInterface, IsolatedFromAbove
  ]> {
  ...
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);
  ...
}
```

本节先抓几个重点：

- C++ wrapper：`FuncOp`
- operation name：`toy.func`
- 它是一个函数类 operation。
- 它有 symbol name，例如 `@main`。
- 它有 function type。
- 它有一个 body region。
- 它实现 `FunctionOpInterface`。
- 它有 `IsolatedFromAbove` trait。

对应 MLIR：

```mlir
toy.func @main() {
  ...
}
```

下一节会继续解释 `arguments` 和 `regions` 的具体含义。

## `GenericCallOp`

`GenericCallOp`：

```tablegen
def GenericCallOp : Toy_Op<"generic_call"> {
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<F64Tensor>:$inputs);
  let results = (outs F64Tensor);

  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

  let builders = [
    OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments)>
  ];
}
```

对应 MLIR：

```mlir
%0 = toy.generic_call @multiply_transpose(%a, %b)
     : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
```

重点：

- `callee` 是 `FlatSymbolRefAttr`，不是 SSA operand。
- `inputs` 是 variadic tensor operands。
- 它返回一个 tensor。

## `ReshapeOp`

`ReshapeOp`：

```tablegen
def ReshapeOp : Toy_Op<"reshape"> {
  let arguments = (ins F64Tensor:$input);
  let results = (outs StaticShapeTensorOf<[F64]>);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)
  }];
}
```

对应 MLIR：

```mlir
%1 = toy.reshape(%0 : tensor<6xf64>) to tensor<2x3xf64>
```

重点：

- 一个输入 tensor。
- 一个静态 shape 的结果 tensor。
- assembly format 解释了为什么文本中有 `(...) to ...`。

## `ReturnOp`

`ReturnOp`：

```tablegen
def ReturnOp : Toy_Op<"return", [Pure, HasParent<"FuncOp">,
                                 Terminator]> {
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

本节重点：

- operation name：`toy.return`
- 它必须在 `FuncOp` 里面。
- 它是 terminator。
- 可以没有 operand，也可以有返回 operand。
- 有无参 builder，所以可以写 `builder.create<ReturnOp>(loc)`。

对应 MLIR：

```mlir
toy.return
toy.return %0 : tensor<*xf64>
```

## `TransposeOp`

`TransposeOp`：

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

对应 MLIR：

```mlir
%0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
```

重点：

- 一个输入 tensor。
- 一个输出 tensor。
- 有 builder，所以 `MLIRGen.cpp` 可以写：

```cpp
builder.create<TransposeOp>(location, operands[0])
```

- 有 verifier，`Dialect.cpp` 中实现 shape 检查。

## TableGen 生成哪些文件

文件：

```text
mlir/examples/toy/Ch2/include/toy/CMakeLists.txt
```

内容：

```cmake
set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(ToyCh2OpsIncGen)
```

含义：

| 生成文件 | 生成动作 | 大致内容 |
| --- | --- | --- |
| `Ops.h.inc` | `-gen-op-decls` | operation C++ class 声明 |
| `Ops.cpp.inc` | `-gen-op-defs` | operation C++ class 方法定义、注册列表等 |
| `Dialect.h.inc` | `-gen-dialect-decls` | ToyDialect 声明 |
| `Dialect.cpp.inc` | `-gen-dialect-defs` | ToyDialect 定义片段 |

这些 `.inc` 文件通常出现在构建目录中，不在源码目录里直接维护。

也就是说，你修改的是：

```text
Ops.td
```

构建系统根据它生成：

```text
Ops.h.inc
Ops.cpp.inc
Dialect.h.inc
Dialect.cpp.inc
```

## `Dialect.h` 如何接入生成文件

文件：

```text
mlir/examples/toy/Ch2/include/toy/Dialect.h
```

关键内容：

```cpp
#include "toy/Dialect.h.inc"

#define GET_OP_CLASSES
#include "toy/Ops.h.inc"
```

含义：

- `Dialect.h.inc` 提供 `ToyDialect` 的 C++ 声明。
- `Ops.h.inc` 在 `GET_OP_CLASSES` 宏控制下展开 operation class 声明。

所以当其他 C++ 文件 include：

```cpp
#include "toy/Dialect.h"
```

它就能看到：

```cpp
mlir::toy::ToyDialect
mlir::toy::ConstantOp
mlir::toy::AddOp
mlir::toy::MulOp
mlir::toy::FuncOp
...
```

这就是为什么 `MLIRGen.cpp` 可以写：

```cpp
builder.create<ConstantOp>(...)
builder.create<MulOp>(...)
builder.create<GenericCallOp>(...)
```

## `Dialect.cpp` 如何接入生成文件

文件：

```text
mlir/examples/toy/Ch2/mlir/Dialect.cpp
```

首先 include dialect 定义片段：

```cpp
#include "toy/Dialect.cpp.inc"
```

然后实现 dialect 初始化：

```cpp
void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "toy/Ops.cpp.inc"
      >();
}
```

这里的 `GET_OP_LIST` 会从 `Ops.cpp.inc` 中展开 operation 列表，类似概念上：

```cpp
addOperations<
  ConstantOp,
  AddOp,
  FuncOp,
  GenericCallOp,
  MulOp,
  PrintOp,
  ReshapeOp,
  ReturnOp,
  TransposeOp
>();
```

文件末尾还有：

```cpp
#define GET_OP_CLASSES
#include "toy/Ops.cpp.inc"
```

这里展开 operation 方法定义。

所以 `Ops.cpp.inc` 被包含两次，但在不同宏下展开不同内容：

- `GET_OP_LIST`：展开注册用 operation 列表。
- `GET_OP_CLASSES`：展开 operation C++ 方法定义。

这是 MLIR TableGen `.inc` 常见用法。

## 从 `Ops.td` 到 `builder.create<ConstantOp>`

把链路串起来：

```text
Ops.td:
  def ConstantOp : Toy_Op<"constant", [Pure]> { ... }

CMake:
  mlir_tablegen(Ops.h.inc -gen-op-decls)
  mlir_tablegen(Ops.cpp.inc -gen-op-defs)

Dialect.h:
  #define GET_OP_CLASSES
  #include "toy/Ops.h.inc"

Dialect.cpp:
  #define GET_OP_LIST
  #include "toy/Ops.cpp.inc"
  #define GET_OP_CLASSES
  #include "toy/Ops.cpp.inc"

MLIRGen.cpp:
  builder.create<ConstantOp>(...)

Printed MLIR:
  toy.constant
```

这条链路是本节最重要的内容。

## 为什么不手写所有 C++ operation 类

ODS/TableGen 的价值在于减少样板代码，并让 operation 的结构信息集中在一个地方。

例如从这一段：

```tablegen
def AddOp : Toy_Op<"add"> {
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);
}
```

TableGen 可以生成：

- `AddOp` C++ wrapper。
- operation name：`toy.add`。
- operand accessor。
- result/type 相关接口。
- 基础 verifier。
- builder 声明或辅助逻辑。
- parser/printer 接入点。

这样后续 pass 可以写更强类型的 C++：

```cpp
if (auto add = dyn_cast<AddOp>(op)) {
  ...
}
```

而不需要到处手动解析 operation name 和 operands。

## 与第五课的连接

第五课看到：

```cpp
return builder.create<MulOp>(location, lhs, rhs);
```

第六课解释了 `MulOp` 从哪里来：

```tablegen
def MulOp : Toy_Op<"mul"> {
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}
```

第五课看到：

```cpp
context.getOrLoadDialect<mlir::toy::ToyDialect>();
```

第六课解释了 `ToyDialect` 从哪里来：

```tablegen
def Toy_Dialect : Dialect {
  let name = "toy";
  let cppNamespace = "::mlir::toy";
}
```

并且通过：

```cpp
#include "toy/Dialect.h.inc"
```

接入 C++。

## 常见误区

### 误区 1：`ConstantOp` 这个 C++ 类是手写完整实现的

不是。

它的声明和大量基础方法来自 TableGen 生成的 `Ops.h.inc` / `Ops.cpp.inc`。

但某些自定义逻辑仍然手写在 `Dialect.cpp`，例如：

- `ConstantOp::build(double)`
- `ConstantOp::parse`
- `ConstantOp::print`
- `ConstantOp::verify`

### 误区 2：`Toy_Op<"constant">` 的名字就是 `constant`

不完整。

`constant` 是 mnemonic。完整 operation name 是：

```text
toy.constant
```

因为它属于 `Toy_Dialect`，而 `Toy_Dialect` 的 name 是 `toy`。

### 误区 3：`.inc` 文件是普通手写头文件

不是。

这些 `.inc` 是 TableGen 根据 `Ops.td` 生成的构建产物。源码通过 `#include` 把它们拼进 C++ 编译单元。

### 误区 4：`GET_OP_CLASSES` 和 `GET_OP_LIST` 是随便定义的宏

不是。

它们是 TableGen 生成 `.inc` 文件时约定好的展开开关。不同宏会让同一个 `.inc` 文件展开不同内容。

### 误区 5：ODS 只影响打印

不是。

ODS 同时影响：

- operation 名字。
- C++ wrapper。
- operands/results/attributes。
- traits/interfaces。
- verifier。
- builders。
- parser/printer。
- 文档。

## 动手观察

### 观察 `toy.constant`

在 `Ops.td` 中找到：

```tablegen
def ConstantOp : Toy_Op<"constant", [Pure]> {
  ...
}
```

然后在 `MLIRGen.cpp` 中找到：

```cpp
builder.create<ConstantOp>(...)
```

再在 Ch2 输出中找到：

```mlir
%0 = toy.constant ...
```

这就是：

```text
ODS 定义 -> C++ wrapper -> builder.create -> MLIR 文本
```

### 观察 `toy.mul`

在 `Ops.td` 中找到：

```tablegen
def MulOp : Toy_Op<"mul"> {
  ...
}
```

在 `MLIRGen.cpp` 中找到：

```cpp
builder.create<MulOp>(location, lhs, rhs)
```

在 Ch2 `codegen.toy` 输出中找到：

```mlir
%0 = toy.mul ...
```

### 观察 `toy.transpose`

在 `Ops.td` 中找到：

```tablegen
def TransposeOp : Toy_Op<"transpose"> {
  ...
}
```

在 `MLIRGen.cpp` 中找到：

```cpp
builder.create<TransposeOp>(location, operands[0])
```

在 Ch2 输出中找到：

```mlir
%0 = toy.transpose(...)
```

## 本节练习

### 练习 1：画出 TableGen 生成链路

画出从 `Ops.td` 到 C++ 可用 operation 类的链路。至少包含：

```text
Ops.td
mlir_tablegen
Ops.h.inc
Ops.cpp.inc
Dialect.h.inc
Dialect.cpp.inc
Dialect.h
Dialect.cpp
MLIRGen.cpp
```

并说明每个文件或步骤负责什么。

### 练习 2：解释 `Toy_Dialect`

阅读：

```tablegen
def Toy_Dialect : Dialect {
  let name = "toy";
  let cppNamespace = "::mlir::toy";
}
```

回答：

- `name = "toy"` 影响什么。
- `cppNamespace = "::mlir::toy"` 影响什么。
- `toy.constant` 里的 `toy` 来自哪里。

### 练习 3：解释 `Toy_Op`

阅读：

```tablegen
class Toy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;
```

回答：

- `mnemonic` 是什么。
- `traits` 是什么。
- 为什么所有 Toy op 都继承 `Toy_Op`。
- `Toy_Op<"mul">` 最终 operation name 是什么。

### 练习 4：找出四个 operation 的 ODS 定义

在 `Ops.td` 中找到：

```text
ConstantOp
AddOp
MulOp
TransposeOp
```

为每个 operation 写出：

- C++ wrapper 类名。
- mnemonic。
- full operation name。
- 有几个 inputs。
- 有几个 results。
- 有没有 builder。
- 有没有 verifier。

### 练习 5：解释 `GET_OP_CLASSES`

阅读：

```cpp
#define GET_OP_CLASSES
#include "toy/Ops.h.inc"
```

和：

```cpp
#define GET_OP_CLASSES
#include "toy/Ops.cpp.inc"
```

回答：

- 它们分别出现在什么文件。
- `Ops.h.inc` 中展开的主要是什么。
- `Ops.cpp.inc` 中展开的主要是什么。

### 练习 6：解释 `GET_OP_LIST`

阅读：

```cpp
void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "toy/Ops.cpp.inc"
      >();
}
```

回答：

- `GET_OP_LIST` 大概展开成什么。
- 为什么 `ToyDialect::initialize()` 需要调用 `addOperations<...>()`。
- 如果某个 op 没注册，会有什么后果。

### 练习 7：从 ODS 追踪到 MLIR 输出

选择一个 operation，例如 `TransposeOp`。

写出完整链路：

```text
Ops.td 中的定义
生成的 C++ wrapper
MLIRGen.cpp 中 builder.create
最终 MLIR 输出
```

要求至少写出源码路径和关键代码片段。

## 本节小结

本节最重要的是理解 Toy operation 的来源：

```text
Ops.td
  -> TableGen
  -> .inc generated files
  -> Dialect.h / Dialect.cpp
  -> ToyDialect registration
  -> C++ Op wrappers
  -> MLIRGen builder.create<...>
  -> toy.* MLIR text
```

需要记住的关键词：

- `Dialect`：MLIR 扩展命名空间。
- `ODS`：声明式 operation 定义。
- `TableGen`：根据 `.td` 生成 C++ 片段。
- `Toy_Dialect`：Toy dialect 的 ODS record。
- `Toy_Op`：Toy operation 的 ODS 基类模板。
- `mnemonic`：dialect 内部短名字。
- `Ops.h.inc`：operation class 声明。
- `Ops.cpp.inc`：operation class 定义和注册列表。
- `Dialect.h.inc` / `Dialect.cpp.inc`：dialect 声明和定义片段。
- `GET_OP_CLASSES` / `GET_OP_LIST`：控制 `.inc` 展开的宏。

下一节课会继续深入 `Ops.td`，重点看 operation 的参数、结果、类型和属性：`arguments`、`results`、`F64Tensor`、`F64ElementsAttr`、ranked/unranked tensor 等。

## 学习记录模板

```text
本节主题：Toy Dialect 与 ODS/TableGen
我读过的源码：
我观察过的测试：
我确认理解的生成链路：
我能解释的 ODS 定义：
我还不理解的问题：
下一步要验证的小实验：
```
