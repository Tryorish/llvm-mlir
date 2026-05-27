# 第 18 课：Toy/MLIR 到 LLVM Dialect 的 full lowering

## 本节定位

前两节课已经完成了这条路径：

```text
Toy Dialect
  -> Affine / MemRef / Func / Arith
```

第 16 课关注 Toy operation 如何被 partial lowering 成低层 MLIR operation。

第 17 课关注 lowering 后的 Affine IR，以及 `-opt` 对 loop 和中间 buffer 的影响。

本节继续向下走一步：

```text
Affine / MemRef / Func / Arith / Toy.print
  -> LLVM Dialect
```

这一节的重点不是 LLVM IR 文本本身，而是 MLIR 内部的 LLVM Dialect。也就是说，本节关注的是：

```text
还在 MLIR 世界里，
但 operation 已经基本变成 llvm.*。
```

下一课再讲：

```text
LLVM Dialect
  -> LLVM IR
  -> JIT 执行
```

## 本节目标

- 理解 full lowering 和 partial lowering 的区别。
- 理解 Ch6 为什么需要一个新的 `LowerToLLVM.cpp`。
- 理解 `createLowerToLLVMPass()` 在 pipeline 中的位置。
- 理解 `LLVMConversionTarget` 的作用。
- 理解 `LLVMTypeConverter` 为什么必须出现。
- 理解 conversion pattern 如何把不同 dialect 降到 LLVM Dialect。
- 理解 `toy.print` 为什么需要自定义 lowering。
- 理解 `toy.print` lowering 到 `printf` 调用的大致过程。
- 能区分 `-emit=mlir-llvm` 和 `-emit=llvm`。

## 本节关键文件

源码：

```text
mlir/examples/toy/Ch6/toyc.cpp
mlir/examples/toy/Ch6/mlir/LowerToLLVM.cpp
mlir/examples/toy/Ch6/include/toy/Passes.h
mlir/examples/toy/Ch6/CMakeLists.txt
```

测试：

```text
mlir/test/Examples/Toy/Ch6/llvm-lowering.mlir
mlir/test/Examples/Toy/Ch6/jit.toy
```

建议阅读顺序：

1. 先看 `toyc.cpp` 中 `-emit=mlir-llvm`、`-emit=llvm`、`-emit=jit` 三个 action。
2. 再看 `loadAndProcessMLIR()` 中 LLVM lowering pipeline 的触发条件。
3. 然后看 `LowerToLLVM.cpp` 的文件顶部注释，理解整体转换图。
4. 最后阅读 `PrintOpLowering` 和 `ToyToLLVMLoweringPass::runOnOperation()`。

## 从第 17 课接上来

第 17 课看到，Ch5/Ch6 经过 affine lowering 后，IR 大致变成：

```mlir
func.func @main() {
  %buffer = memref.alloc() : memref<3x2xf64>
  affine.for %i = 0 to 3 {
    affine.for %j = 0 to 2 {
      %v = affine.load %input[%j, %i] : memref<2x3xf64>
      %r = arith.mulf %v, %v : f64
      affine.store %r, %buffer[%i, %j] : memref<3x2xf64>
    }
  }
  toy.print %buffer : memref<3x2xf64>
  func.return
}
```

注意这里仍然混合着多个 dialect：

| Dialect | 还在做什么 |
| --- | --- |
| `func` | 表示函数边界和返回 |
| `memref` | 表示内存 buffer |
| `affine` | 表示循环和访存 |
| `arith` | 表示标量计算 |
| `toy` | 只剩下 `toy.print` |

所以第 17 课结束时，还不能直接导出到 LLVM IR 或 JIT 执行。

Ch6 新增的 full lowering 目标是：

```text
把这些剩余 dialect 都转成 LLVM Dialect 能表达的形式。
```

## partial lowering 与 full lowering

第 16 课的 `LowerToAffineLoops.cpp` 属于 partial lowering。

它的特点是：

```text
只降低一部分 operation。
允许 IR 里继续保留其他 dialect。
```

例如：

```text
toy.constant   -> memref.alloc + affine.store
toy.transpose  -> affine.for + affine.load/store
toy.mul        -> affine.for + arith.mulf + affine.load/store
toy.print      -> 保留
```

这就是为什么 affine lowering 后仍然能看到：

```mlir
toy.print %arg : memref<3x2xf64>
```

第 18 课的 `LowerToLLVM.cpp` 属于 full lowering。

它的特点是：

```text
转换完成后，不应该再剩下 Toy、Affine、SCF、Func、MemRef、Arith 等非法 operation。
最终 IR 只允许目标认为合法的 operation。
```

Toy 示例中使用的是：

```cpp
applyFullConversion(module, target, std::move(patterns))
```

这意味着：

```text
如果有任何非法 operation 没有被成功转换，
整个 pass 就失败。
```

## Ch6 的命令行动作

在 `mlir/examples/toy/Ch6/toyc.cpp` 中，`Action` 多了几个和代码生成相关的动作：

```cpp
enum Action {
  None,
  DumpAST,
  DumpMLIR,
  DumpMLIRAffine,
  DumpMLIRLLVM,
  DumpLLVMIR,
  RunJIT
};
```

对应命令行：

```text
-emit=mlir-llvm
-emit=llvm
-emit=jit
```

它们的区别是：

| 命令 | 输出层级 | 说明 |
| --- | --- | --- |
| `-emit=mlir-llvm` | MLIR LLVM Dialect | 仍然是 MLIR 文本 |
| `-emit=llvm` | LLVM IR | 已经导出到 LLVM IR |
| `-emit=jit` | 执行结果 | 通过 ExecutionEngine JIT 执行 |

本节重点是第一项：

```bash
toyc-ch6 input.toy -emit=mlir-llvm
```

也可以从 `.mlir` 输入开始：

```bash
toyc-ch6 input.mlir -x=mlir -emit=mlir-llvm
```

## lowering pipeline 的触发条件

`toyc.cpp` 中有两个关键布尔变量：

```cpp
bool isLoweringToAffine = emitAction >= Action::DumpMLIRAffine;
bool isLoweringToLLVM = emitAction >= Action::DumpMLIRLLVM;
```

这表示：

```text
如果要输出 mlir-affine、mlir-llvm、llvm、jit，
都要至少走到 affine lowering。

如果要输出 mlir-llvm、llvm、jit，
还要继续走 LLVM lowering。
```

因此当你执行：

```bash
toyc-ch6 test.toy -emit=mlir-llvm
```

大致 pipeline 是：

```text
Toy 源码
  -> AST
  -> Toy Dialect MLIR
  -> Inliner
  -> Shape Inference
  -> Canonicalizer
  -> CSE
  -> LowerToAffine
  -> Canonicalizer
  -> CSE
  -> LowerToLLVM
  -> DIScopeForLLVMFuncOp
  -> dump MLIR
```

如果加上 `-opt`：

```bash
toyc-ch6 test.toy -emit=mlir-llvm -opt
```

在 Affine lowering 之后还会多两个 affine 优化：

```text
LoopFusion
AffineScalarReplacement
```

所以 `-emit=mlir-llvm -opt` 看到的 LLVM Dialect IR，已经受到 affine 层优化影响。

## `createLowerToLLVMPass()` 在哪里加入

`toyc.cpp` 中的关键代码是：

```cpp
if (isLoweringToLLVM) {
  pm.addPass(mlir::toy::createLowerToLLVMPass());
  pm.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());
}
```

这里有两个 pass：

```text
createLowerToLLVMPass()
  真正执行 Toy/Affine/Func/MemRef/Arith 到 LLVM Dialect 的转换。

createDIScopeForLLVMFuncOpPass()
  给 LLVM function 补充调试相关 scope。
```

本节主要关注第一个。

## `LowerToLLVM.cpp` 的整体转换图

`LowerToLLVM.cpp` 文件顶部给了一个非常重要的图：

```text
                         Affine --
                                  |
                                  v
                       Arithmetic + Func --> LLVM (Dialect)
                                  ^
                                  |
     'toy.print' --> Loop (SCF) --
```

这张图可以拆开理解：

```text
Affine
  -> 先降到更普通的 loop/control-flow 形式
  -> 再继续降到 LLVM Dialect

Arith
  -> 降到 LLVM arithmetic / LLVM instruction 形式

Func
  -> 降到 llvm.func 等 LLVM Dialect 函数形式

MemRef
  -> 降到 LLVM 能表达的 memref descriptor / pointer 相关形式

toy.print
  -> 先变成循环 + memref.load + printf call
  -> 再由通用 conversion patterns 继续降到 LLVM Dialect
```

注意这里的 `toy.print` 不是直接一步变成完整 LLVM IR。

它先被改写成 MLIR 中更低层的 operation：

```text
SCF loop
memref.load
LLVM.call printf
LLVM.global string
```

然后 full conversion 会继续处理其中还不合法的部分。

## 为什么 `toy.print` 需要单独处理

前面大多数 operation 都有 MLIR 自带的转换 pattern：

| 来源 | 可用通用转换 |
| --- | --- |
| `affine` | `populateAffineToStdConversionPatterns` |
| `scf` | `populateSCFToControlFlowConversionPatterns` |
| `arith` | `populateArithToLLVMConversionPatterns` |
| `memref` | `populateFinalizeMemRefToLLVMConversionPatterns` |
| `cf` | `populateControlFlowToLLVMConversionPatterns` |
| `func` | `populateFuncToLLVMConversionPatterns` |

但 `toy.print` 是 Toy 自己定义的 operation。

MLIR 不知道：

```text
toy.print 应该打印到哪里？
按什么格式打印？
多维 tensor/memref 应该按什么顺序打印？
每行在哪里换行？
要调用哪个运行时函数？
```

所以 Ch6 需要自己写：

```cpp
class PrintOpLowering : public ConversionPattern
```

它负责把：

```mlir
toy.print %buffer : memref<3x2xf64>
```

改写成：

```text
嵌套循环遍历 buffer
每个元素 memref.load
调用 printf("%f ", value)
适当位置调用 printf("\n")
```

## `PrintOpLowering` 的入口

`PrintOpLowering` 继承自：

```cpp
ConversionPattern
```

构造函数中指定它匹配的 operation 名称：

```cpp
ConversionPattern(toy::PrintOp::getOperationName(), 1, context)
```

也就是说，它只处理：

```mlir
toy.print
```

核心方法是：

```cpp
LogicalResult matchAndRewrite(
    Operation *op,
    ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const override
```

普通 rewrite pattern 通常只关心“匹配后改写”。

conversion pattern 还要参与 dialect conversion，因此它有两个额外重点：

```text
operands 可能已经经过 type conversion 映射。
rewriter 负责在 conversion 过程中替换、删除、创建 operation。
```

Toy 这里为了教学简洁，主要使用原始 `toy.print` operand 的 memref type 来生成循环。

## 取得 memref 形状

`toy.print` 在进入 LLVM lowering 前，operand 已经从 tensor 变成 memref：

```mlir
toy.print %0 : memref<3x2xf64>
```

所以 `PrintOpLowering` 一开始取出：

```cpp
auto memRefType = llvm::cast<MemRefType>((*op->operand_type_begin()));
auto memRefShape = memRefType.getShape();
```

这里依赖一个前提：

```text
toy.print 的 operand 已经是 statically shaped memref。
```

这来自前面的 Toy 到 Affine lowering。

如果 `toy.print` 的 operand 还是：

```mlir
tensor<3x2xf64>
```

那这里就不能按 memref shape 生成 `memref.load`。

这也说明 pipeline 顺序很重要：

```text
必须先 LowerToAffine，
再 LowerToLLVM。
```

## 插入 `printf` 声明

`toy.print` 最终通过 C 标准库的 `printf` 打印。

所以 lowering 需要在 module 里准备一个函数声明：

```cpp
static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                           ModuleOp module)
```

它做两件事：

```text
如果 module 里已经有 llvm.func @printf，就复用。
如果没有，就插入一个声明。
```

`printf` 类型由这个函数创建：

```cpp
static LLVM::LLVMFunctionType getPrintfType(MLIRContext *context)
```

对应 C 里的大致签名：

```c
int printf(char *, ...);
```

在 LLVM Dialect 中表达为：

```text
i32 (ptr, ...)
```

这里用的是 vararg 函数类型，因为 `printf` 的参数数量和类型会随格式字符串变化。

## 插入全局字符串

`printf` 需要格式字符串。

Toy 这里创建两个字符串：

```cpp
Value formatSpecifierCst = getOrCreateGlobalString(
    loc, rewriter, "frmt_spec", StringRef("%f \0", 4), parentModule);

Value newLineCst = getOrCreateGlobalString(
    loc, rewriter, "nl", StringRef("\n\0", 2), parentModule);
```

分别表示：

```text
"%f "
"\n"
```

`getOrCreateGlobalString()` 会做三件事：

1. 在 module 顶部插入 `llvm.mlir.global`。
2. 用 `llvm.mlir.addressof` 取得全局变量地址。
3. 用 `llvm.getelementptr` 得到字符串首字符指针。

概念上就是：

```c
static const char frmt_spec[] = "%f ";
static const char nl[] = "\n";

printf(frmt_spec, value);
printf(nl);
```

但在 MLIR LLVM Dialect 中，它们会以 LLVM operation 的形式出现。

## 为每个维度生成循环

`PrintOpLowering` 根据 memref rank 生成嵌套循环：

```cpp
SmallVector<Value, 4> loopIvs;
for (unsigned i = 0, e = memRefShape.size(); i != e; ++i) {
  auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto upperBound =
      rewriter.create<arith::ConstantIndexOp>(loc, memRefShape[i]);
  auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  auto loop =
      rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
  ...
  loopIvs.push_back(loop.getInductionVar());
}
```

如果 memref 是：

```mlir
memref<3x2xf64>
```

那概念上生成：

```c
for (int i = 0; i < 3; ++i) {
  for (int j = 0; j < 2; ++j) {
    printf("%f ", buffer[i][j]);
  }
  printf("\n");
}
```

这里生成的是 `scf.for`，不是 `affine.for`。

原因是：

```text
toy.print 的任务是生成通用循环打印元素，
后面已有 SCF 到 ControlFlow、ControlFlow 到 LLVM 的通用 conversion pattern。
```

## 删除 loop body 中的默认 terminator

创建 `scf::ForOp` 时，body 里会有一些默认结构。

Toy 示例中有这段代码：

```cpp
for (Operation &nested : make_early_inc_range(*loop.getBody()))
  rewriter.eraseOp(&nested);
```

它把默认 body 内容删掉，然后自己控制插入位置：

```cpp
rewriter.setInsertionPointToEnd(loop.getBody());
...
rewriter.create<scf::YieldOp>(loc);
rewriter.setInsertionPointToStart(loop.getBody());
```

这段代码的目标是：

```text
构造出干净的嵌套循环结构，
并确保每层 scf.for body 末尾有 scf.yield。
```

对于刚接触 MLIR region/block 的读者，这里要关注两个点：

```text
scf.for 有 region/body。
rewriter 的 insertion point 决定新 operation 插到哪里。
```

## 打印元素

嵌套循环建好后，代码在最内层插入元素访问：

```cpp
auto printOp = cast<toy::PrintOp>(op);
auto elementLoad =
    rewriter.create<memref::LoadOp>(loc, printOp.getInput(), loopIvs);
```

这相当于：

```mlir
%v = memref.load %buffer[%i, %j] : memref<3x2xf64>
```

然后插入：

```cpp
rewriter.create<LLVM::CallOp>(
    loc, getPrintfType(context), printfRef,
    ArrayRef<Value>({formatSpecifierCst, elementLoad}));
```

也就是：

```text
printf("%f ", value)
```

因为 `printf` 本身已经是 LLVM Dialect operation，所以这里直接生成 `LLVM::CallOp`。

但 `elementLoad` 仍然是 `memref.load` 的结果。

后续 full conversion 会继续把 `memref.load` 降到 LLVM Dialect。

## 删除原始 `toy.print`

完成替换后，最后调用：

```cpp
rewriter.eraseOp(op);
```

也就是说：

```text
原来的 toy.print 不再存在。
```

如果这一步没做，full conversion 会失败，因为 `toy.print` 不是合法目标 operation。

这是 full conversion 中非常重要的规则：

```text
所有非法 operation 都必须被替换或删除。
```

## `ToyToLLVMLoweringPass`

真正的 pass 定义是：

```cpp
struct ToyToLLVMLoweringPass
    : public PassWrapper<ToyToLLVMLoweringPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const override { return "toy-to-llvm"; }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
  }
  void runOnOperation() final;
};
```

几个重点：

```text
OperationPass<ModuleOp>
  说明这个 pass 跑在 module 层。

getArgument() = "toy-to-llvm"
  说明 pass 命令行名字是 toy-to-llvm。

getDependentDialects()
  声明 pass 运行时可能创建 LLVM 和 SCF dialect 的 operation。
```

为什么是 module-level pass？

因为 lowering 需要：

```text
插入 module-level 的 printf 声明。
插入 module-level 的 global string。
整体检查 module 中是否还有非法 operation。
```

所以它不能只是一个 function-level pass。

## Conversion Target

`runOnOperation()` 一开始创建：

```cpp
LLVMConversionTarget target(getContext());
target.addLegalOp<ModuleOp>();
```

`ConversionTarget` 回答的问题是：

```text
转换完成后，哪些 operation 是合法的？
哪些 operation 必须被转换掉？
```

这里使用的是：

```cpp
LLVMConversionTarget
```

它表示目标主要是 LLVM Dialect。

同时额外声明：

```cpp
target.addLegalOp<ModuleOp>();
```

因为 MLIR module 本身作为容器可以保留。

最终目标可以理解为：

```text
module 可以存在。
LLVM Dialect operation 可以存在。
其他不合法 operation 必须被 conversion patterns 消掉。
```

这就是 full lowering 的约束。

## Type Converter

接着创建：

```cpp
LLVMTypeConverter typeConverter(&getContext());
```

它回答的问题是：

```text
源类型应该如何变成 LLVM 能表达的类型？
```

例如：

| 源类型 | LLVM lowering 后的大致形态 |
| --- | --- |
| `index` | 目标相关整数类型 |
| `f64` | LLVM double |
| `memref<3x2xf64>` | memref descriptor / pointer 相关结构 |
| `func.func` 类型 | LLVM function type |

为什么需要 TypeConverter？

因为 operation lowering 不只是改名字。

例如：

```mlir
func.func @main()
```

降到 LLVM Dialect 后，不只是变成：

```mlir
llvm.func @main()
```

函数参数、返回值、memref 参数、index 类型、region block argument 等都可能需要类型转换。

所以 full conversion 必须同时处理：

```text
operation 转换
type 转换
region/block argument 转换
```

这就是 `LLVMTypeConverter` 的职责。

## Conversion Patterns

接下来构造 pattern 集合：

```cpp
RewritePatternSet patterns(&getContext());
```

然后加入一组现成转换：

```cpp
populateAffineToStdConversionPatterns(patterns);
populateSCFToControlFlowConversionPatterns(patterns);
mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
populateFuncToLLVMConversionPatterns(typeConverter, patterns);
```

再加入 Toy 自己的转换：

```cpp
patterns.add<PrintOpLowering>(&getContext());
```

这说明 Toy 到 LLVM lowering 并不是所有东西都自己写。

更准确的理解是：

```text
Toy 只处理 Toy 自己的剩余 operation。
MLIR 已经提供的标准 dialect lowering 直接复用。
```

这也是 MLIR 的核心价值之一：

```text
自定义 dialect 可以只实现自己特殊的那部分 lowering，
然后接入已有 lowering 基础设施。
```

## 转换链不是一步完成的

`LowerToLLVM.cpp` 中有一段注释提到 transitive lowering。

意思是：

```text
A 不一定直接降到 C。
A 可以先降到 B，再由另一组 pattern 把 B 降到 C。
```

本节中的例子：

```text
toy.print
  -> scf.for + memref.load + llvm.call
  -> cf.br / cf.cond_br + LLVM ops
  -> LLVM Dialect
```

再比如：

```text
affine.for
  -> 更普通的循环或控制流形式
  -> LLVM Dialect
```

因此不要把 full conversion 理解成：

```text
每个源 operation 都必须有一个直接到 LLVM 的 pattern。
```

更准确的是：

```text
pattern 集合整体必须能把所有非法 operation 递归转换到合法目标。
```

## applyFullConversion

最后调用：

```cpp
auto module = getOperation();
if (failed(applyFullConversion(module, target, std::move(patterns))))
  signalPassFailure();
```

这一步做三件事：

1. 遍历 module 中的 operation。
2. 对非法 operation 应用 conversion pattern。
3. 检查转换后是否只剩合法 operation。

如果失败：

```cpp
signalPassFailure();
```

pass manager 会认为这个 pass 失败，`toyc` 最终返回错误。

常见失败原因包括：

```text
有某个 operation 没有对应 lowering pattern。
某个 type 无法转换。
pattern 生成了目标不允许的 operation。
IR 不满足 lowering pattern 的前提。
```

比如如果你新增一个 Toy operation，但忘记给它写 lowering，那么 full conversion 很可能失败。

## `-emit=mlir-llvm` 会看到什么

执行：

```bash
toyc-ch6 mlir/test/Examples/Toy/Ch6/affine-lowering.mlir -x=mlir -emit=mlir-llvm -opt
```

应该会看到 MLIR LLVM Dialect 层级的输出。

典型特征包括：

```text
llvm.func
llvm.call
llvm.mlir.global
llvm.mlir.addressof
llvm.getelementptr
llvm.load / llvm.store
llvm.br / llvm.cond_br
llvm.return
```

具体文本会因为 LLVM/MLIR 版本和打印格式不同而变化。

本节不要求背输出全文，只需要能识别：

```text
toy.print 已经消失。
affine.for 已经消失。
func.func 已经变成 llvm.func。
高层 memref/affine/func/arith operation 已经被 LLVM Dialect operation 替代。
```

## `-emit=llvm` 和本节的关系

`-emit=llvm` 不只是运行 `createLowerToLLVMPass()`。

它还会继续调用：

```cpp
dumpLLVMIR(*module)
```

里面关键步骤是：

```cpp
mlir::registerBuiltinDialectTranslation(*module->getContext());
mlir::registerLLVMDialectTranslation(*module->getContext());
auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
```

也就是说：

```text
LLVM Dialect MLIR
  -> LLVM IR
```

这是下一课的重点。

本节只需要记住：

```text
-emit=mlir-llvm 输出 MLIR 中的 LLVM Dialect。
-emit=llvm 输出真正的 LLVM IR。
```

## `llvm-lowering.mlir` 测试在检查什么

`mlir/test/Examples/Toy/Ch6/llvm-lowering.mlir` 的 RUN 行是：

```mlir
// RUN: toyc-ch6 %s -emit=llvm -opt
```

注意它检查的是：

```text
导出后的 LLVM IR。
```

不是 `-emit=mlir-llvm` 的输出。

输入 Toy IR 是：

```mlir
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00],
                          [4.000000e+00, 5.000000e+00, 6.000000e+00]]>
       : tensor<2x3xf64>
  %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %3 = toy.mul %2, %2 : tensor<3x2xf64>
  toy.print %3 : tensor<3x2xf64>
  toy.return
}
```

文件里保留了以下 `CHECK` 注释：

```mlir
// CHECK-LABEL: define void @main()
// CHECK: @printf
// CHECK-SAME: 1.000000e+00
// CHECK: @printf
// CHECK-SAME: 1.600000e+01
// CHECK: @printf
// CHECK-SAME: 4.000000e+00
// CHECK: @printf
// CHECK-SAME: 2.500000e+01
// CHECK: @printf
// CHECK-SAME: 9.000000e+00
// CHECK: @printf
// CHECK-SAME: 3.000000e+01
```

注意：当前 `llvm-lowering.mlir` 的 `RUN` 行没有把输出管给 `FileCheck`，所以下面的 `CHECK` 注释更适合作为阅读线索，而不是本文件实际执行的严格断言。学习时应该先按 Toy 语义算出结果，再用实际命令输出确认 lowering 没有改变语义。

从 Toy 语义看，这段程序最终打印的是：

```text
transpose 后再逐元素相乘的结果
```

原始矩阵：

```text
[[1, 2, 3],
 [4, 5, 6]]
```

transpose 后：

```text
[[1, 4],
 [2, 5],
 [3, 6]]
```

逐元素相乘后：

```text
[[1, 16],
 [4, 25],
 [9, 36]]
```

重点是：

```text
LLVM lowering 之后，Toy 程序语义仍然保持。
printf 调用确实被生成出来。
```

## 为什么 `-opt` 会影响 LLVM lowering 输出

`llvm-lowering.mlir` 使用：

```text
-emit=llvm -opt
```

这意味着在进入 LLVM lowering 前，已经做过 affine 层优化。

所以最终 LLVM IR 中可能不会保留明显的中间 transpose buffer 行为。

从语义上看：

```text
先 transpose，再 mul
```

但优化后可能变成：

```text
直接从原始 buffer 以转置后的 index 读取，
立刻平方，
存入最终结果或直接用于打印链路。
```

这和第 17 课看到的优化现象一致。

因此阅读 LLVM Dialect 或 LLVM IR 时要记住：

```text
低层输出不一定一一对应源程序里的每个 Toy operation。
优化会改变 IR 形状，但不能改变语义。
```

## Full Conversion 的合法性思维

学习 `LowerToLLVM.cpp` 时，建议用这张表检查：

| 转换前 | 谁负责转换 | 转换后 |
| --- | --- | --- |
| `toy.print` | `PrintOpLowering` | `scf.for`、`memref.load`、`llvm.call` |
| `affine.for/load/store` | Affine conversion patterns | 更低层循环和访存 |
| `scf.for` | SCF conversion patterns | control flow |
| `cf.*` | ControlFlow conversion patterns | LLVM branch |
| `func.func` | Func conversion patterns | `llvm.func` |
| `memref.*` | MemRef conversion patterns | LLVM descriptor/pointer/load/store |
| `arith.*` | Arith conversion patterns | LLVM arithmetic |

当你以后新增一个 dialect 或 operation 时，也可以按这个方式思考：

```text
这个 operation 最终要消失吗？
如果要消失，由哪个 pattern 负责？
它生成的新 operation 是否也能继续被转换？
目标是否合法？
类型是否能转换？
```

## `getDependentDialects()` 的意义

`ToyToLLVMLoweringPass` 中有：

```cpp
void getDependentDialects(DialectRegistry &registry) const override {
  registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
}
```

这不是 lowering 逻辑本身，但很重要。

它告诉 pass manager：

```text
这个 pass 运行时可能会创建 LLVM Dialect 和 SCF Dialect 的 operation。
```

如果 dialect 没有注册，创建对应 operation 可能失败。

Toy 的 lowering 过程中确实会创建：

```text
scf.for
scf.yield
llvm.func
llvm.call
llvm.mlir.global
llvm.getelementptr
```

所以 dependent dialect 必须声明。

## `toyc.cpp` 中的 LLVM Dialect 注册

Ch6 `main()` 里还有：

```cpp
mlir::DialectRegistry registry;
mlir::func::registerAllExtensions(registry);
mlir::LLVM::registerInlinerInterface(registry);

mlir::MLIRContext context(registry);
context.getOrLoadDialect<mlir::toy::ToyDialect>();
```

这里有两个层面的注册要区分：

```text
DialectRegistry
  给 context 准备可用的 dialect、extension、interface。

getDependentDialects()
  给 pass 声明运行时会依赖或创建哪些 dialect。
```

另外，在导出 LLVM IR 或 JIT 前，还需要注册 translation：

```cpp
mlir::registerBuiltinDialectTranslation(*module->getContext());
mlir::registerLLVMDialectTranslation(*module->getContext());
```

这是下一课重点。

本节只要先分清：

```text
创建 LLVM Dialect operation
  需要 dialect 可用。

把 LLVM Dialect 翻译成 LLVM IR
  需要 translation 注册。
```

## 常见误区

### 误区 1：LLVM Dialect 就是 LLVM IR

不是。

LLVM Dialect 仍然是 MLIR dialect。

它用 MLIR operation 表达接近 LLVM IR 的结构，例如：

```text
llvm.func
llvm.call
llvm.load
llvm.store
llvm.return
```

LLVM IR 是另一个层级，需要通过 translation 导出。

### 误区 2：`toy.print` 会直接变成一个 `printf` 调用

不准确。

`toy.print` 打印的是整个 memref。

所以它需要：

```text
嵌套循环
逐元素 load
每个元素 printf
每行换行
```

对于二维 memref，不是一个 `printf` 调用，而是一组循环中的 `printf` 调用。

### 误区 3：full lowering 只要写一个 Toy pattern 就够了

不够。

Toy 自己只写了 `PrintOpLowering`，但完整 lowering 还依赖很多 MLIR 内置 pattern：

```text
Affine -> Standard/SCF/CF
SCF -> CF
CF -> LLVM
Arith -> LLVM
MemRef -> LLVM
Func -> LLVM
```

### 误区 4：`-emit=llvm` 可以跳过 `-emit=mlir-llvm`

从用户命令看可以直接执行：

```bash
toyc-ch6 input.toy -emit=llvm
```

但内部 pipeline 仍然会先走到 LLVM Dialect，再导出 LLVM IR。

所以概念路径不能跳过：

```text
Toy MLIR -> Affine/MemRef/Func -> LLVM Dialect -> LLVM IR
```

### 误区 5：低层 IR 没有出现 transpose 就说明语义丢了

不一定。

如果开启 `-opt`，transpose 和 mul 可能已经被 loop fusion 和 scalar replacement 融合。

应该检查最终访存、计算和打印结果，而不是期待每个高层 operation 都在低层 IR 中有明显对应块。

## 动手观察

### 观察 1：对比 affine 和 llvm dialect 输出

运行：

```bash
toyc-ch6 mlir/test/Examples/Toy/Ch6/affine-lowering.mlir -x=mlir -emit=mlir-affine -opt
```

再运行：

```bash
toyc-ch6 mlir/test/Examples/Toy/Ch6/affine-lowering.mlir -x=mlir -emit=mlir-llvm -opt
```

观察：

- `affine.for` 是否消失。
- `toy.print` 是否消失。
- `func.func` 是否变成 LLVM Dialect 中的函数形式。
- 是否出现 `printf`、global string、LLVM call。

### 观察 2：不加 `-opt`

运行：

```bash
toyc-ch6 mlir/test/Examples/Toy/Ch6/affine-lowering.mlir -x=mlir -emit=mlir-llvm
```

和：

```bash
toyc-ch6 mlir/test/Examples/Toy/Ch6/affine-lowering.mlir -x=mlir -emit=mlir-llvm -opt
```

对比：

- 临时 buffer 是否更多。
- load/store 是否更多。
- 最终打印结果语义是否一致。

### 观察 3：看 pass pipeline

尝试给命令加 pass manager 打印选项。

不同构建配置支持的选项可能略有差异，常见形式包括：

```bash
toyc-ch6 input.toy -emit=mlir-llvm -mlir-print-ir-after-all
```

观察每个 pass 之后 IR 如何变化。

重点看：

```text
LowerToAffine 前后
LowerToLLVM 前后
```

### 观察 4：改动 `toy.print` 输入 shape

写一个一维或二维 Toy 程序，观察 `PrintOpLowering` 生成的循环层数。

例如二维：

```toy
def main() {
  var a<2, 2> = [[1, 2], [3, 4]];
  print(a);
}
```

观察 lowering 后：

```text
是否生成两层循环。
是否每行有 newline。
```

## 本节练习

### 练习 1：解释 full lowering

回答：

- 什么是 partial lowering？
- 什么是 full lowering？
- Ch5 affine lowering 为什么是 partial lowering？
- Ch6 LLVM lowering 为什么要用 `applyFullConversion()`？

### 练习 2：画出 Ch6 pipeline

根据 `toyc.cpp`，画出：

```text
Toy source
  -> AST
  -> Toy MLIR
  -> optimized Toy MLIR
  -> Affine/MemRef/Func/Arith
  -> LLVM Dialect
  -> LLVM IR
  -> JIT
```

标注每一步对应的 pass 或函数。

### 练习 3：解释 `toy.print` lowering

回答：

- `toy.print` 为什么不能由 MLIR 内置 conversion patterns 自动处理？
- `PrintOpLowering` 为 `toy.print` 创建了哪些 operation？
- 为什么它需要知道 memref shape？
- 为什么要插入 `printf` 声明和 global string？

### 练习 4：解释 TypeConverter

回答：

- `LLVMTypeConverter` 负责什么？
- 为什么 memref 类型不能简单地原样保留？
- 为什么 function type 和 block argument 也可能需要转换？

### 练习 5：解释 ConversionTarget

回答：

- `LLVMConversionTarget` 定义了什么？
- 为什么还要 `target.addLegalOp<ModuleOp>()`？
- 如果 conversion 后还剩 `toy.print`，会发生什么？

### 练习 6：跟踪一个 operation

从这几个 operation 中任选一个：

```text
affine.for
memref.load
arith.mulf
func.return
toy.print
```

回答：

- 它在 LLVM lowering 前长什么样？
- 由哪组 pattern 负责转换？
- 转换后大致变成什么？

### 练习 7：新增 Toy operation 的 lowering 思考

假设你给 Toy 新增：

```text
toy.neg
```

回答：

- 它应该在哪个阶段 lowering？
- 可以先 lowering 到 `arith.negf` 或等价 operation 吗？
- 如果进入 full conversion 前还保留 `toy.neg`，会发生什么？
- 需要添加什么 pattern？

## 本节小结

本节的核心是：

```text
Ch6 把 Toy lowering 从 partial lowering 推进到 full lowering。
```

需要记住：

- `LowerToAffineLoops.cpp` 负责把大多数 Toy 计算降到 Affine/MemRef/Func/Arith。
- `LowerToLLVM.cpp` 负责把剩余 IR 降到 LLVM Dialect。
- `toy.print` 是 Toy 中最后需要自定义处理的 operation。
- `PrintOpLowering` 把 `toy.print` 改写成循环、load、`printf` 调用和字符串全局变量。
- `LLVMConversionTarget` 定义最终合法 IR。
- `LLVMTypeConverter` 负责类型转换。
- `applyFullConversion()` 要求所有非法 operation 都被转换掉。
- `-emit=mlir-llvm` 输出的是 LLVM Dialect MLIR，不是 LLVM IR。
- `-emit=llvm` 会在 LLVM Dialect 之后继续导出真正的 LLVM IR。

下一课会继续讲 LLVM IR 导出和 JIT 执行，重点阅读 `dumpLLVMIR()`、`runJit()` 和 `ExecutionEngine`。

## 学习记录模板

```text
本节主题：Toy/MLIR 到 LLVM Dialect 的 full lowering
我读过的源码：
我观察过的测试：
我理解的 full lowering：
我理解的 ConversionTarget：
我理解的 TypeConverter：
我理解的 PrintOpLowering：
我能解释的 conversion pattern：
我观察到的 mlir-llvm 输出特征：
我还不理解的问题：
下一步要验证的小实验：
```
