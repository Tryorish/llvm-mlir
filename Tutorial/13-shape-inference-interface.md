# 第 13 课：Shape Inference Interface

## 本节定位

第 12 课已经讲过 Toy 的 pass pipeline。Ch4 的高层优化组合是：

```text
inliner
  -> shape inference
  -> canonicalizer
  -> CSE
```

本节深入其中最关键的 Toy 自定义 pass：

```text
Shape Inference
```

在 Ch2/Ch3 中，很多 operation 的结果类型先写成：

```mlir
tensor<*xf64>
```

这表示元素类型已知，但 rank 和 shape 未知。

Shape inference 的目标是尽量把它们推成：

```mlir
tensor<2x3xf64>
tensor<3x2xf64>
```

这种更精确的 ranked tensor type。

本节要回答：

- 为什么 Toy 需要 shape inference。
- 为什么用 interface，而不是 pass 里写一堆 `if isa<AddOp>`。
- 哪些 op 实现了 `inferShapes()`。
- `ShapeInferencePass` 的工作列表算法怎么运行。
- 为什么 shape inference 要放在 inliner 之后、canonicalizer 之前。

## 本节目标

- 理解 unranked tensor 和 ranked tensor 对后续优化/lowering 的影响。
- 理解 `ShapeInferenceOpInterface` 的定义。
- 理解 `DeclareOpInterfaceMethods<ShapeInferenceOpInterface>` 的作用。
- 理解 `AddOp::inferShapes()`、`MulOp::inferShapes()`、`CastOp::inferShapes()`、`TransposeOp::inferShapes()`。
- 理解 `ShapeInferencePass` 为什么是 `OperationPass<toy::FuncOp>`。
- 理解 worklist、`returnsDynamicShape()`、`allOperandsInferred()`。
- 能解释 `shape_inference.mlir` 中 `tensor<*xf64>` 如何消失。

## 本节关键文件

源码：

```text
mlir/examples/toy/Ch4/include/toy/ShapeInferenceInterface.td
mlir/examples/toy/Ch4/include/toy/Ops.td
mlir/examples/toy/Ch4/mlir/Dialect.cpp
mlir/examples/toy/Ch4/mlir/ShapeInferencePass.cpp
mlir/examples/toy/Ch4/include/toy/Passes.h
mlir/examples/toy/Ch4/toyc.cpp
```

测试：

```text
mlir/test/Examples/Toy/Ch4/shape_inference.mlir
```

建议阅读顺序：

1. 先看 `shape_inference.mlir`，明确 shape inference 的最终效果。
2. 再看 `ShapeInferenceInterface.td`，理解接口只要求一个方法。
3. 再看 `Ops.td`，找哪些 op 声明实现这个接口。
4. 再看 `Dialect.cpp`，读每个 `inferShapes()`。
5. 最后看 `ShapeInferencePass.cpp`，理解工作列表算法。

## 为什么需要 shape inference

Toy 前端为了保持简单，很多 operation 一开始会产生 unranked tensor：

```mlir
tensor<*xf64>
```

例如：

```mlir
%0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
%1 = toy.mul %0, %0 : tensor<*xf64>
```

这种 IR 在高层表达上是合法的，但对后续 pass 不够友好。

Lowering 到 Affine/MemRef 时，编译器经常需要知道：

- tensor 有几维。
- 每一维大小是多少。
- 循环 nest 应该生成几层。
- memref type 应该是什么 shape。

所以在 lowering 前，需要尽量把：

```mlir
tensor<*xf64>
```

变成：

```mlir
tensor<3x2xf64>
```

这就是 shape inference 的任务。

## `shape_inference.mlir` 的目标

测试文件：

```text
mlir/test/Examples/Toy/Ch4/shape_inference.mlir
```

输入里有很多 unranked tensor：

```mlir
toy.func private @multiply_transpose(%arg0: tensor<*xf64>,
                                     %arg1: tensor<*xf64>) -> tensor<*xf64> {
  %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
  %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
  %2 = toy.mul %0, %1 : tensor<*xf64>
  toy.return %2 : tensor<*xf64>
}
```

经过：

```bash
toyc-ch4 shape_inference.mlir -emit=mlir -opt
```

测试要求：

```mlir
// CHECK-NOT: tensor<*xf64>
```

并期望关键结果变成：

```mlir
%0 = toy.transpose(... : tensor<2x3xf64>) to tensor<3x2xf64>
%1 = toy.mul %0, %0 : tensor<3x2xf64>
toy.print %1 : tensor<3x2xf64>
```

也就是说，shape inference 之后，未知 shape 基本被推掉。

## 为什么先 inliner

`shape_inference.mlir` 的输入里有：

```mlir
toy.generic_call @multiply_transpose(...)
```

Ch4 pipeline 先跑：

```cpp
pm.addPass(mlir::createInlinerPass());
```

把被调用函数内联到 `main`，这样 shape inference 就能在同一个 `toy.func` 内看到：

- 实参的 ranked tensor type。
- transpose 的输入类型。
- mul 的输入类型。

如果不内联，callee 函数参数仍是：

```mlir
tensor<*xf64>
```

shape inference 很难仅凭函数体本身推到具体 shape。

所以 Ch4 的顺序是：

```text
inliner -> shape inference
```

## ShapeInference interface

接口定义在：

```text
mlir/examples/toy/Ch4/include/toy/ShapeInferenceInterface.td
```

内容很短：

```tablegen
def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  let description = [{
    Interface to access a registered method to infer the return types for an
    operation that can be used during type inference.
  }];

  let methods = [
    InterfaceMethod<"Infer and set the output shape for the current operation.",
                    "void", "inferShapes">
  ];
}
```

这个 interface 要求实现一个方法：

```cpp
void inferShapes();
```

它的职责是：

```text
根据当前 operation 的 operand types，
设置当前 operation 的 result type。
```

## 为什么用 interface

Shape inference pass 不想知道每个具体 operation 的细节。

不要写成：

```cpp
if (auto add = dyn_cast<AddOp>(op)) ...
else if (auto mul = dyn_cast<MulOp>(op)) ...
else if (auto transpose = dyn_cast<TransposeOp>(op)) ...
```

这样 pass 会和具体 operation 强耦合。

更好的方式是：

```cpp
if (auto shapeOp = dyn_cast<ShapeInference>(op))
  shapeOp.inferShapes();
```

也就是说：

```text
pass 面向能力编程，而不是面向具体 op 名字编程。
```

只要某个 operation 实现了 `ShapeInference` interface，pass 就能调用它的 `inferShapes()`。

## 在 ODS 中声明接口

Ch4 的 `Ops.td` 里引入：

```tablegen
include "toy/ShapeInferenceInterface.td"
```

然后在 operation traits 中声明：

```tablegen
def AddOp : Toy_Op<"add",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  ...
}
```

同样的接口还出现在：

```text
AddOp
MulOp
CastOp
TransposeOp
```

其中：

```tablegen
DeclareOpInterfaceMethods<ShapeInferenceOpInterface>
```

表示 operation 类会声明接口方法，C++ 里需要实现相应的 `inferShapes()`。

## 哪些 op 实现了 shape inference

Ch4 中实现 `ShapeInferenceOpInterface` 的主要 operation：

| Operation | inferShapes 逻辑 |
| --- | --- |
| `toy.add` | result type 设成 lhs type |
| `toy.mul` | result type 设成 lhs type |
| `toy.cast` | result type 设成 input type |
| `toy.transpose` | result shape 设成 input shape 的反转 |

注意：

- `toy.constant` 通常已经有 ranked result type，不需要 shape inference。
- `toy.reshape` 的 result type 在 IR 中已经显式写出静态 shape。
- `toy.print` 没有 result。
- `toy.return` 没有 result。

## `AddOp::inferShapes()`

`Dialect.cpp` 中：

```cpp
void AddOp::inferShapes() {
  getResult().setType(getLhs().getType());
}
```

含义：

```text
add(lhs, rhs) 的输出 shape 和 lhs 相同。
```

前提是 pass 已经确认所有 operands 都是 ranked tensor。

如果：

```mlir
%0 = toy.add %a, %b : tensor<*xf64>
```

并且 `%a` 的类型是：

```mlir
tensor<2x3xf64>
```

那么推导后 result type 可以设成：

```mlir
tensor<2x3xf64>
```

## `MulOp::inferShapes()`

`MulOp` 和 `AddOp` 类似：

```cpp
void MulOp::inferShapes() {
  getResult().setType(getLhs().getType());
}
```

Toy 的 `mul` 是逐元素乘法，所以输出 shape 与输入 shape 相同。

在 `shape_inference.mlir` 中：

```mlir
%2 = toy.mul %0, %1 : tensor<*xf64>
```

如果 `%0` 和 `%1` 都已经推成：

```mlir
tensor<3x2xf64>
```

那么 `%2` 也会变成：

```mlir
tensor<3x2xf64>
```

## `CastOp::inferShapes()`

Ch4 新增了 `toy.cast`：

```tablegen
def CastOp : Toy_Op<"cast", [
     DeclareOpInterfaceMethods<CastOpInterface>,
     DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
     Pure,
     SameOperandsAndResultShape
  ]> {
  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}
```

对应：

```cpp
void CastOp::inferShapes() {
  getResult().setType(getInput().getType());
}
```

`toy.cast` 不改变数据，只是在类型层面做等价转换。

因此 shape inference 可以把 output type 设成 input type。

## `TransposeOp::inferShapes()`

`TransposeOp` 的推导最有代表性：

```cpp
void TransposeOp::inferShapes() {
  auto arrayTy = cast<RankedTensorType>(getOperand().getType());
  SmallVector<int64_t, 2> dims(reverse(arrayTy.getShape()));
  getResult().setType(RankedTensorType::get(dims, arrayTy.getElementType()));
}
```

如果输入是：

```mlir
tensor<2x3xf64>
```

那么：

```text
input shape = [2, 3]
reverse     = [3, 2]
```

输出变成：

```mlir
tensor<3x2xf64>
```

这正好符合转置语义。

## `ShapeInferencePass`

pass 实现在：

```text
mlir/examples/toy/Ch4/mlir/ShapeInferencePass.cpp
```

类定义：

```cpp
struct ShapeInferencePass
    : public PassWrapper<ShapeInferencePass, OperationPass<toy::FuncOp>> {
  StringRef getArgument() const override {
    return "toy-shape-inference";
  }

  void runOnOperation() override {
    ...
  }
};
```

它是：

```text
OperationPass<toy::FuncOp>
```

也就是说，这个 pass 跑在每个 `toy.func` 上。

这和 Ch4 `toyc.cpp` 里的 nested pipeline 对应：

```cpp
OpPassManager &optPM = pm.nest<mlir::toy::FuncOp>();
optPM.addPass(mlir::toy::createShapeInferencePass());
```

## 工作列表算法

`ShapeInferencePass` 的核心是 worklist。

算法可以概括为：

```text
1. 遍历当前 toy.func 中所有 operation。
2. 把 result type 仍然是 dynamic shape 的 operation 放进 worklist。
3. 反复从 worklist 中找一个“ready”的 operation。
4. ready 的条件是：所有 operand type 都已经是 ranked tensor。
5. 对 ready operation 调用 inferShapes()。
6. 如果 worklist 清空，成功。
7. 如果无法继续但 worklist 还没清空，失败。
```

## `returnsDynamicShape()`

源码：

```cpp
static bool returnsDynamicShape(Operation *op) {
  return any_of(op->getResultTypes(), [](Type resultType) {
    return !isa<RankedTensorType>(resultType);
  });
}
```

它判断：

```text
这个 operation 是否还有非 ranked result type。
```

例如：

```mlir
%0 = toy.transpose(...) to tensor<*xf64>
```

结果是 unranked tensor，所以进入 worklist。

而：

```mlir
%0 = toy.constant ... : tensor<2x3xf64>
```

结果已经是 ranked tensor，一般不需要进入 worklist。

## `allOperandsInferred()`

源码：

```cpp
static bool allOperandsInferred(Operation *op) {
  return all_of(op->getOperandTypes(), [](Type operandType) {
    return isa<RankedTensorType>(operandType);
  });
}
```

它判断：

```text
这个 operation 的所有输入 shape 是否已经已知。
```

只有所有输入都是 ranked tensor，才可以安全调用 `inferShapes()`。

例如：

```mlir
%0 = toy.transpose(%arg0 : tensor<2x3xf64>) to tensor<*xf64>
```

可以推导。

但：

```mlir
%0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
```

还不能推导，因为输入 shape 本身未知。

## 为什么可能失败

如果 worklist 中所有 operation 都不 ready，pass 会停止。

最后如果 worklist 不为空：

```cpp
f.emitError("Shape inference failed, ")
    << opWorklist.size() << " operations couldn't be inferred\n";
signalPassFailure();
```

这说明：

```text
还有 operation 的 result shape 无法推导。
```

常见原因：

- 输入 shape 本身一直未知。
- 某个产生 dynamic shape 的 op 没实现 `ShapeInference` interface。
- IR 中存在无法从 operands 推断 output shape 的 operation。

## 面向接口的调用

核心代码：

```cpp
if (auto shapeOp = dyn_cast<ShapeInference>(op)) {
  shapeOp.inferShapes();
} else {
  op->emitError("unable to infer shape of operation without shape "
                "inference interface");
  return signalPassFailure();
}
```

这体现了 MLIR interface 的价值：

```text
ShapeInferencePass 不关心具体 op 是 AddOp、MulOp 还是 TransposeOp。
它只关心这个 op 是否支持 ShapeInference interface。
```

这让 pass 更容易扩展。

如果未来新增一个 Toy operation，只要它也实现 `ShapeInferenceOpInterface`，这个 pass 就可以直接调用它。

## 和 canonicalizer 的关系

Ch4 pipeline 是：

```text
shape inference -> canonicalizer -> CSE
```

shape inference 会更新类型：

```text
tensor<*xf64> -> tensor<3x2xf64>
```

类型变精确后，canonicalizer 可能发现新的简化机会。

例如：

- 输入输出类型相同的 reshape 可以被消掉。
- 某些 operation 的打印形式变得更具体。
- CSE 更容易判断两个表达式是否等价。

这也是为什么 shape inference 在 pipeline 中位置很关键。

## 和 inliner 的关系

`shape_inference.mlir` 的测试注释说：

```text
Check the result of inlining+shape inference on an input module.
```

这说明这个测试不是只看 shape inference。

它还依赖 inliner 把：

```mlir
toy.generic_call @multiply_transpose(...)
```

展开到 `main` 里。

内联之后，原本 callee 里的 `%arg0: tensor<*xf64>` 可以对应到 caller 中的具体 ranked tensor。

这样 shape inference 才能继续推进。

## 测试逐步观察

输入里：

```mlir
%0 = toy.constant ... : tensor<2x3xf64>
%1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>
%4 = toy.generic_call @multiply_transpose(%1, %3)
     : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
```

经过 inliner 后，函数体中的：

```mlir
toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
```

会对应到具体输入：

```mlir
toy.transpose(%0 : tensor<2x3xf64>) to tensor<*xf64>
```

然后 shape inference 把它推成：

```mlir
toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
```

再把后续：

```mlir
toy.mul ... : tensor<*xf64>
```

推成：

```mlir
toy.mul ... : tensor<3x2xf64>
```

最终 `toy.print` 也使用：

```mlir
tensor<3x2xf64>
```

## 常见误区

### 误区 1：shape inference 会改变数据

不会。

Shape inference 只修改类型信息，不改变 tensor 中的值。

### 误区 2：所有 operation 都需要实现 ShapeInference

不需要。

只有 result shape 可能是 dynamic，并且可以从 operands 推导的 operation 才需要。

### 误区 3：unranked tensor 是错误

不是。

`tensor<*xf64>` 是合法类型，只是信息不够精确。

Shape inference 的目标是尽量减少它，而不是说它一出现就是错误。

### 误区 4：pass 直接知道所有 Toy op 的推导规则

不是。

pass 只调用 interface：

```cpp
shapeOp.inferShapes();
```

具体规则在每个 operation 自己的 C++ 方法里。

### 误区 5：shape inference 可以替代 verifier

不能。

Verifier 检查 IR 是否符合约束。

Shape inference 传播和补全类型信息。

两者职责不同。

## 动手观察

### 观察接口定义

阅读：

```text
mlir/examples/toy/Ch4/include/toy/ShapeInferenceInterface.td
```

确认：

- interface 名字是 `ShapeInference`。
- 它要求的方法是 `inferShapes()`。
- 返回类型是 `void`。

### 观察 ODS 声明

阅读：

```text
mlir/examples/toy/Ch4/include/toy/Ops.td
```

找到：

```tablegen
DeclareOpInterfaceMethods<ShapeInferenceOpInterface>
```

记录它出现在哪些 operation 上。

### 观察 C++ 实现

阅读：

```text
mlir/examples/toy/Ch4/mlir/Dialect.cpp
```

找到：

```cpp
AddOp::inferShapes()
MulOp::inferShapes()
CastOp::inferShapes()
TransposeOp::inferShapes()
```

写出每个方法如何设置 result type。

### 观察 pass 算法

阅读：

```text
mlir/examples/toy/Ch4/mlir/ShapeInferencePass.cpp
```

标注：

- worklist 初始化。
- `returnsDynamicShape()`。
- `allOperandsInferred()`。
- `dyn_cast<ShapeInference>(op)`。
- `shapeOp.inferShapes()`。

### 观察测试输出

阅读：

```text
mlir/test/Examples/Toy/Ch4/shape_inference.mlir
```

重点看：

```mlir
// CHECK-NOT: tensor<*xf64>
```

以及：

```mlir
toy.transpose(... : tensor<2x3xf64>) to tensor<3x2xf64>
toy.mul ... : tensor<3x2xf64>
toy.print ... : tensor<3x2xf64>
```

## 本节练习

### 练习 1：解释 ShapeInference interface

回答：

- `ShapeInferenceOpInterface` 定义在哪里。
- 它要求 operation 实现哪个方法。
- 为什么 pass 要通过 interface 调用，而不是直接判断具体 op 类型。

### 练习 2：列出实现接口的 op

阅读 Ch4 `Ops.td`。

回答：

- 哪些 operation 声明了 `ShapeInferenceOpInterface`。
- 哪些 operation 没有声明。
- 没声明的 operation 为什么可能不需要 shape inference。

### 练习 3：手动推导 transpose

给定：

```mlir
%0 = toy.transpose(%arg0 : tensor<4x5xf64>) to tensor<*xf64>
```

回答：

- 输入 shape 是什么。
- 输出 shape 应该是什么。
- `TransposeOp::inferShapes()` 会把 result type 设置成什么。

### 练习 4：手动推导 mul

给定：

```mlir
%0 = toy.mul %a, %b : tensor<*xf64>
```

其中 `%a`、`%b` 都是：

```mlir
tensor<3x2xf64>
```

回答：

- `MulOp::inferShapes()` 会设置什么 result type。
- 为什么它直接使用 lhs type。

### 练习 5：解释 worklist

回答：

- 哪些 operation 会进入 worklist。
- 什么样的 operation 是 ready 的。
- 如果一直找不到 ready operation，会发生什么。

### 练习 6：解释 shape inference 和 inliner

回答：

- 为什么 `shape_inference.mlir` 的测试依赖 inlining。
- 如果不内联，callee 的参数为什么仍可能是 `tensor<*xf64>`。
- inlining 后 shape inference 获得了什么额外信息。

### 练习 7：画出 shape inference 流程

画出下面流程：

```text
toy.func
  -> walk operations
  -> collect dynamic result ops
  -> find ready op
  -> dyn_cast<ShapeInference>
  -> inferShapes()
  -> update result type
  -> repeat
```

并标注每一步对应 `ShapeInferencePass.cpp` 中的哪段代码。

## 本节小结

本节最重要的是理解：

```text
ShapeInferencePass 面向 ShapeInference interface 编程。
每个 operation 自己实现 inferShapes()。
pass 用 worklist 按依赖顺序调用这些方法。
```

需要记住：

- `tensor<*xf64>` 是 shape 未知，不是错误。
- shape inference 的目标是把 unranked tensor 推成 ranked tensor。
- `ShapeInferenceOpInterface` 只要求 `inferShapes()`。
- `AddOp` / `MulOp` 输出 shape 跟 lhs 一致。
- `CastOp` 输出 shape 跟 input 一致。
- `TransposeOp` 输出 shape 是 input shape 的反转。
- pass 只处理 operands 已经 ranked 的 operation。
- shape inference 通常需要 inliner 提供更具体的上下文。

下一课会继续看 Ch4 的高层优化组合，重点讲 inliner、CSE 和 canonicalization 如何协同工作。

## 学习记录模板

```text
本节主题：Shape Inference Interface
我读过的源码：
我观察过的测试：
我能解释的 interface：
我能解释的 inferShapes 方法：
我能解释的 worklist 算法：
我还不理解的问题：
下一步要验证的小实验：
```
