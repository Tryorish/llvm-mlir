# 第 19 课：LLVM IR 导出与 JIT 执行

## 本节定位

第 18 课已经完成了：

```text
Toy / Affine / MemRef / Func / Arith
  -> LLVM Dialect
```

本节继续走完 Ch6 的最后一段：

```text
LLVM Dialect
  -> LLVM IR
  -> JIT 执行
```

到这一课为止，Toy 编译链路已经从源语言走到了可执行代码：

```text
.toy 源码
  -> AST
  -> Toy Dialect MLIR
  -> Affine/MemRef/Func/Arith
  -> LLVM Dialect MLIR
  -> LLVM IR
  -> ORC JIT 执行
```

本节重点是 `mlir/examples/toy/Ch6/toyc.cpp` 中的两个函数：

```cpp
int dumpLLVMIR(mlir::ModuleOp module)
int runJit(mlir::ModuleOp module)
```

它们分别对应：

```text
-emit=llvm
-emit=jit
```

## 本节目标

- 理解 `-emit=mlir-llvm` 和 `-emit=llvm` 的边界。
- 理解 LLVM Dialect 为什么还需要 translation 才能变成 LLVM IR。
- 理解 `registerBuiltinDialectTranslation()` 和 `registerLLVMDialectTranslation()` 的作用。
- 理解 `translateModuleToLLVMIR()` 做了什么。
- 理解 target triple 和 data layout 为什么需要设置。
- 理解 `makeOptimizingTransformer()` 在 LLVM IR/JIT 阶段的作用。
- 理解 MLIR `ExecutionEngine` 如何创建并执行 JIT 代码。
- 理解 `invokePacked("main")` 的调用约定。
- 能读懂 Ch6 的 `jit.toy` 测试和 `llvm-lowering.mlir` 测试。

## 本节关键文件

源码：

```text
mlir/examples/toy/Ch6/toyc.cpp
mlir/examples/toy/Ch6/mlir/LowerToLLVM.cpp
mlir/examples/toy/Ch6/CMakeLists.txt
```

测试：

```text
mlir/test/Examples/Toy/Ch6/llvm-lowering.mlir
mlir/test/Examples/Toy/Ch6/jit.toy
```

建议阅读顺序：

1. 先看 `toyc.cpp` 中 `main()` 如何分发 `-emit=llvm` 和 `-emit=jit`。
2. 再看 `dumpLLVMIR()`，理解 MLIR LLVM Dialect 如何导出 LLVM IR。
3. 再看 `runJit()`，理解 ExecutionEngine 如何 JIT 执行。
4. 最后看 `CMakeLists.txt`，理解 Ch6 为什么依赖 ExecutionEngine 和 ORC JIT。

## 从第 18 课接上来

第 18 课的终点是：

```text
module {
  llvm.func @main() {
    ...
    llvm.call @printf(...)
    ...
    llvm.return
  }
}
```

这是 LLVM Dialect MLIR。

它看起来很像 LLVM IR，但仍然是 MLIR：

```text
有 MLIR 的 Operation。
有 MLIR 的 Attribute。
有 MLIR 的 ModuleOp。
由 MLIRContext 管理。
可以继续跑 MLIR pass。
```

真正的 LLVM IR 是 LLVM 项目中另一套 IR 结构：

```text
llvm::Module
llvm::Function
llvm::BasicBlock
llvm::Instruction
llvm::LLVMContext
```

所以从 `-emit=mlir-llvm` 到 `-emit=llvm`，本质是：

```text
MLIR ModuleOp
  -> llvm::Module
```

这个过程在 MLIR 中叫 translation。

## Ch6 的输出分支

`toyc.cpp` 中 `main()` 的最后几段非常关键：

```cpp
bool isOutputingMLIR = emitAction <= Action::DumpMLIRLLVM;
if (isOutputingMLIR) {
  module->dump();
  return 0;
}

if (emitAction == Action::DumpLLVMIR)
  return dumpLLVMIR(*module);

if (emitAction == Action::RunJIT)
  return runJit(*module);
```

也就是说：

| action | 行为 |
| --- | --- |
| `-emit=ast` | 只 dump AST |
| `-emit=mlir` | dump Toy Dialect MLIR |
| `-emit=mlir-affine` | dump Affine/MemRef/Func/Arith 层 MLIR |
| `-emit=mlir-llvm` | dump LLVM Dialect MLIR |
| `-emit=llvm` | 调用 `dumpLLVMIR()` |
| `-emit=jit` | 调用 `runJit()` |

注意：

```text
-emit=llvm 和 -emit=jit 在进入 dumpLLVMIR/runJit 前，
已经先经过 loadAndProcessMLIR()。
```

也就是说，它们不是从 Toy 直接去 LLVM IR/JIT，而是先完整跑完：

```text
Toy lowering
Affine lowering
LLVM Dialect lowering
```

## `-emit=mlir-llvm` 与 `-emit=llvm`

这两个命令很容易混淆。

```bash
toyc-ch6 input.toy -emit=mlir-llvm
```

输出的是 MLIR：

```text
llvm.func
llvm.call
llvm.return
llvm.mlir.global
```

而：

```bash
toyc-ch6 input.toy -emit=llvm
```

输出的是 LLVM IR：

```llvm
define void @main() {
  ...
  call i32 (ptr, ...) @printf(...)
  ret void
}
```

可以这样记：

```text
mlir-llvm
  重点是前面的 mlir：仍然是 MLIR，只是使用 LLVM Dialect。

llvm
  已经离开 MLIR operation 层，进入 llvm::Module / LLVM IR。
```

## `dumpLLVMIR()` 总览

`dumpLLVMIR()` 的结构很清晰：

```cpp
int dumpLLVMIR(mlir::ModuleOp module) {
  register translations;
  translate MLIR module to llvm::Module;
  initialize native target;
  detect host target machine;
  set target triple and data layout;
  optionally optimize LLVM IR;
  print LLVM IR;
}
```

对应源码：

```cpp
int dumpLLVMIR(mlir::ModuleOp module) {
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  ...
  mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(),
                                                        tmOrError.get().get());

  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  ...
  llvm::errs() << *llvmModule << "\n";
  return 0;
}
```

这段代码的作用不是执行程序，而是：

```text
把 LLVM Dialect MLIR 转成 LLVM IR 并打印出来。
```

## 注册 Dialect Translation

`dumpLLVMIR()` 一开始做：

```cpp
mlir::registerBuiltinDialectTranslation(*module->getContext());
mlir::registerLLVMDialectTranslation(*module->getContext());
```

这里要和第 18 课的 dialect 注册区分开。

第 18 课里，为了创建和解析 LLVM Dialect operation，需要 dialect 注册。

本节这里，为了把 LLVM Dialect 翻译成 LLVM IR，需要 translation 注册。

可以这样理解：

| 注册内容 | 解决什么问题 |
| --- | --- |
| Dialect 注册 | MLIRContext 是否认识某个 dialect |
| Dialect translation 注册 | 是否知道如何把某个 dialect 翻译成 LLVM IR |

如果没有 translation 注册，MLIR 可能能保存 `llvm.func`、`llvm.call`，但不知道如何把它们转成 `llvm::Function` 和 `llvm::CallInst`。

Toy 这里注册了两个 translation：

```text
Builtin Dialect Translation
LLVM Dialect Translation
```

因为 module 容器、一些 builtin 属性，以及 LLVM Dialect operation 都要参与导出。

## `translateModuleToLLVMIR()`

真正导出的核心调用是：

```cpp
llvm::LLVMContext llvmContext;
auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
```

这里出现了一个新的 context：

```cpp
llvm::LLVMContext llvmContext;
```

注意它不是：

```cpp
mlir::MLIRContext
```

两者区别：

| Context | 管理什么 |
| --- | --- |
| `mlir::MLIRContext` | MLIR dialect、operation、attribute、type 等 |
| `llvm::LLVMContext` | LLVM IR 的 type、constant、metadata 等 |

`translateModuleToLLVMIR()` 会创建：

```text
std::unique_ptr<llvm::Module>
```

如果失败，Toy 会输出：

```text
Failed to emit LLVM IR
```

失败通常意味着：

```text
LLVM Dialect IR 不满足导出要求。
有不能翻译的 operation。
缺少 translation 注册。
LLVM Dialect 的结构不合法。
```

## 初始化 Native Target

导出 LLVM IR 后，Toy 调用：

```cpp
llvm::InitializeNativeTarget();
llvm::InitializeNativeTargetAsmPrinter();
```

这两行的作用是初始化当前机器的 LLVM target 支持。

即使只是 `-emit=llvm` 打印 IR，Toy 也会继续配置 target triple 和 data layout，所以需要 target machine。

这也是为什么 Ch6 的 CMake 需要：

```cmake
nativecodegen
OrcJIT
```

相关 LLVM 组件。

概念上：

```text
LLVM IR 不是完全脱离目标机器的。
如果要生成、优化或 JIT 当前机器可执行的代码，需要知道目标平台。
```

## Target Triple 与 Data Layout

`dumpLLVMIR()` 里接着做：

```cpp
auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
auto tmOrError = tmBuilderOrError->createTargetMachine();
mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(),
                                                      tmOrError.get().get());
```

这段代码做三件事：

1. 检测当前 host 平台。
2. 创建适合当前平台的 target machine。
3. 给 LLVM module 设置 target triple 和 data layout。

target triple 描述的是：

```text
架构、厂商、操作系统、ABI 等目标信息。
```

例如概念上可能类似：

```text
x86_64-apple-darwin
arm64-apple-darwin
x86_64-unknown-linux-gnu
```

data layout 描述的是：

```text
指针大小
整数/浮点对齐
大小端
类型布局规则
```

为什么重要？

因为同一份 LLVM IR 中的指针、结构体、memref descriptor、GEP 计算等，最终都要落到具体平台的内存布局。

如果没有正确的 data layout，后续优化、代码生成或 JIT 都可能不可靠。

## LLVM IR 优化管线

Toy 使用：

```cpp
auto optPipeline = mlir::makeOptimizingTransformer(
    /*optLevel=*/enableOpt ? 3 : 0,
    /*sizeLevel=*/0,
    /*targetMachine=*/nullptr);
```

这里的 `enableOpt` 对应命令行：

```text
-opt
```

如果不加 `-opt`：

```text
optLevel = 0
```

如果加 `-opt`：

```text
optLevel = 3
```

这和前面 MLIR 层的 `-opt` 是同一个 flag，但影响了多个阶段：

```text
MLIR 高层优化
Affine 层优化
LLVM IR 优化
JIT 内部优化
```

所以：

```bash
toyc-ch6 input.toy -emit=llvm
```

和：

```bash
toyc-ch6 input.toy -emit=llvm -opt
```

输出可能明显不同。

不要把 `-opt` 理解成只影响某一个 pass。

## 打印 LLVM IR

最后：

```cpp
llvm::errs() << *llvmModule << "\n";
```

这会把 LLVM IR 打印到标准错误流。

学习时可以把它重定向到文件：

```bash
toyc-ch6 input.toy -emit=llvm > out.ll 2>&1
```

或者直接在终端观察。

由于 Toy 使用 `llvm::errs()`，具体重定向方式要注意 stdout/stderr 的区别。

输出中常见内容包括：

```llvm
target datalayout = "..."
target triple = "..."

@frmt_spec = internal constant ...
@nl = internal constant ...

declare i32 @printf(ptr, ...)

define void @main() {
  ...
  call i32 (ptr, ...) @printf(...)
  ret void
}
```

不同 LLVM 版本的文本格式可能不完全一样。

学习时重点观察：

```text
是否有 target datalayout。
是否有 target triple。
是否有 printf 声明。
是否有 main 函数。
是否有 call @printf。
是否没有 toy/affine/mlir 这些高层痕迹。
```

## `runJit()` 总览

`runJit()` 的结构是：

```cpp
int runJit(mlir::ModuleOp module) {
  initialize native target;
  register translations;
  create optimization pipeline;
  create ExecutionEngine;
  invoke packed main;
}
```

对应源码：

```cpp
int runJit(mlir::ModuleOp module) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0,
      /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  mlir::ExecutionEngineOptions engineOptions;
  engineOptions.transformer = optPipeline;
  auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  auto invocationResult = engine->invokePacked("main");
  ...
}
```

它和 `dumpLLVMIR()` 的共同点：

```text
都要初始化 native target。
都要注册 LLVM IR translation。
都使用 makeOptimizingTransformer()。
```

区别是：

```text
dumpLLVMIR()
  显式 translateModuleToLLVMIR()，然后打印 llvm::Module。

runJit()
  把 MLIR module 交给 ExecutionEngine，由 ExecutionEngine 完成翻译、编译和执行。
```

## ExecutionEngine 是什么

MLIR `ExecutionEngine` 是一个帮助类，用于：

```text
把可翻译到 LLVM IR 的 MLIR module JIT 编译并执行。
```

Toy 中创建方式是：

```cpp
mlir::ExecutionEngineOptions engineOptions;
engineOptions.transformer = optPipeline;
auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
```

这里传入的是：

```text
已经 lowering 到 LLVM Dialect 的 MLIR module。
```

`ExecutionEngine::create()` 内部会做类似：

```text
LLVM Dialect MLIR
  -> LLVM IR
  -> LLVM 优化
  -> ORC JIT 编译
```

所以 `runJit()` 没有手动调用：

```cpp
translateModuleToLLVMIR()
```

这一步被 ExecutionEngine 包装掉了。

## `engineOptions.transformer`

这行代码很关键：

```cpp
engineOptions.transformer = optPipeline;
```

它表示：

```text
ExecutionEngine 在拿到 LLVM module 后，可以先对 LLVM IR 应用一个 transformer。
```

Toy 这里使用的是：

```cpp
makeOptimizingTransformer(...)
```

也就是 LLVM IR 优化 pipeline。

所以 `-emit=jit -opt` 不只是前面的 MLIR pass 更积极，JIT 前的 LLVM IR 也会使用更高优化级别。

## `invokePacked("main")`

JIT 创建成功后，Toy 调用：

```cpp
auto invocationResult = engine->invokePacked("main");
```

这里的 `"main"` 对应 Toy 程序里的：

```toy
def main() {
  ...
}
```

经过 lowering 后会成为 LLVM 层的：

```text
main
```

`invokePacked` 使用 MLIR ExecutionEngine 的 packed calling convention。

可以先简单理解为：

```text
按 ExecutionEngine 约定调用一个已经 JIT 编译好的函数。
```

Toy 的 `main` 没有参数、没有返回值，所以调用非常简单：

```cpp
invokePacked("main")
```

如果未来要 JIT 调用带参数或返回值的函数，就需要理解 packed ABI 如何传递参数。

本课程目前只需要掌握：

```text
Toy Ch6 的 JIT 入口固定调用 main。
main 必须存在。
main 的签名要符合 invokePacked 的调用方式。
```

## JIT 失败处理

`invokePacked()` 返回一个错误对象：

```cpp
auto invocationResult = engine->invokePacked("main");
if (invocationResult) {
  llvm::errs() << "JIT invocation failed\n";
  return -1;
}
```

如果失败，常见原因包括：

```text
找不到 main 符号。
module 没有正确 lowering 到可执行形式。
LLVM IR 翻译失败。
JIT 编译失败。
外部符号无法解析。
运行时调用约定不匹配。
```

Toy 中 `printf` 是外部符号。

在一般 Unix-like 环境下，JIT 可以解析当前进程中的 C runtime 符号。

测试文件中有：

```text
# UNSUPPORTED: target={{.*windows.*}}
```

这说明这个 JIT 测试在 Windows 上被禁用。

## Ch6 为什么需要 ExecutionEngine 支持

`mlir/examples/toy/Ch6/CMakeLists.txt` 开头写着：

```cmake
if(NOT MLIR_ENABLE_EXECUTION_ENGINE)
  return()
endif()
```

这表示：

```text
如果构建 MLIR 时没有启用 ExecutionEngine，
Ch6 的 toyc-ch6 target 不会继续构建。
```

这和前面章节不同。

Ch6 需要 JIT，所以依赖：

```cmake
MLIRExecutionEngine
MLIRBuiltinToLLVMIRTranslation
MLIRLLVMToLLVMIRTranslation
MLIRTargetLLVMIRExport
```

还需要 LLVM 组件：

```cmake
nativecodegen
OrcJIT
```

如果你本地构建时找不到 `toyc-ch6`，要检查：

```text
MLIR_ENABLE_EXECUTION_ENGINE 是否开启。
LLVM 是否构建了 native target 和 ORC JIT 支持。
```

## `jit.toy` 测试

测试文件：

```text
mlir/test/Examples/Toy/Ch6/jit.toy
```

内容很短：

```toy
# RUN: toyc-ch6 -emit=jit %s
# UNSUPPORTED: target={{.*windows.*}}

def main() {
 print([[1, 2], [3, 4]]);
}
```

这个测试主要验证：

```text
Toy 源码可以一路 lowering 到 JIT 执行。
JIT 运行不会崩溃。
printf 调用链路能工作。
```

它没有用 `FileCheck` 检查输出文本。

所以学习时要区分：

```text
测试是否只要求命令成功。
测试是否还检查具体输出。
```

运行后你应该能看到类似：

```text
1.000000 2.000000
3.000000 4.000000
```

具体空格和换行由 `PrintOpLowering` 中的格式字符串和 newline 逻辑决定。

## `llvm-lowering.mlir` 测试

测试文件：

```text
mlir/test/Examples/Toy/Ch6/llvm-lowering.mlir
```

RUN 行是：

```mlir
// RUN: toyc-ch6 %s -emit=llvm -opt
```

输入已经是 Toy Dialect MLIR，不是 `.toy` 源码：

```mlir
toy.func @main() {
  %0 = toy.constant dense<...> : tensor<2x3xf64>
  %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %3 = toy.mul %2, %2 : tensor<3x2xf64>
  toy.print %3 : tensor<3x2xf64>
  toy.return
}
```

这条命令会走：

```text
Toy Dialect MLIR
  -> shape inference / canonicalizer / CSE
  -> affine lowering
  -> LLVM Dialect lowering
  -> LLVM IR 导出
```

和上一课一样要注意：

```text
当前文件里的 CHECK 注释没有接 FileCheck，
所以它们是阅读线索，而不是实际执行的检查。
```

学习时重点看：

```text
LLVM IR 中是否有 define void @main()
是否有 @printf
是否能看到常量计算后的结果或 printf 参数
```

## 源码到 JIT 的完整路径

以：

```toy
def main() {
  print([[1, 2], [3, 4]]);
}
```

为例，完整路径是：

```text
1. parseInputFile()
   .toy 文本 -> AST

2. mlirGen()
   AST -> Toy Dialect MLIR

3. loadAndProcessMLIR()
   根据 emitAction 决定跑到哪个 lowering 层级

4. Inliner + ShapeInference + Canonicalizer + CSE
   整理 Toy 高层 IR

5. createLowerToAffinePass()
   Toy tensor operation -> Affine/MemRef/Func/Arith

6. createLowerToLLVMPass()
   Affine/MemRef/Func/Arith/Toy.print -> LLVM Dialect

7. runJit()
   LLVM Dialect MLIR -> LLVM IR -> JIT 编译 -> invoke main
```

这条路径贯穿了前面大部分课程。

到这里，你应该能把 Toy 教程看成一个完整编译器，而不是零散 feature。

## `-opt` 的多阶段影响

同一个 `-opt` 在 Ch6 中至少影响三处：

### 1. Toy 高层优化

只要：

```cpp
enableOpt || isLoweringToAffine
```

就会加入：

```text
Inliner
ShapeInference
Canonicalizer
CSE
```

对于 `-emit=llvm` 和 `-emit=jit`，这部分必然会跑。

### 2. Affine 层优化

只有 `enableOpt` 为真时，affine lowering 后才会加入：

```text
LoopFusion
AffineScalarReplacement
```

这会影响低层 loop 和 buffer 形状。

### 3. LLVM IR/JIT 优化

`dumpLLVMIR()` 和 `runJit()` 都使用：

```cpp
makeOptimizingTransformer(enableOpt ? 3 : 0, 0, nullptr)
```

所以 `-opt` 会决定 LLVM IR 优化级别：

```text
不加 -opt：O0
加 -opt：O3
```

这也是为什么观察 LLVM IR 时，建议同时看：

```bash
toyc-ch6 input.toy -emit=llvm
toyc-ch6 input.toy -emit=llvm -opt
```

它们可能差异很大。

## 为什么 JIT 入口是 `main`

Toy 中约定用户程序入口是：

```toy
def main() {
  ...
}
```

`runJit()` 中直接写死：

```cpp
engine->invokePacked("main")
```

所以如果 Toy 程序没有 `main`，JIT 阶段很可能找不到入口。

这和普通 C/C++ 程序类似：

```text
可执行程序需要入口函数。
```

但这里不是操作系统调用 `main`，而是 Toy 编译器自己通过 ExecutionEngine 查找并调用 JIT 出来的 `main` 符号。

## 为什么 `printf` 能被调用

第 18 课讲过，`toy.print` 会 lowering 成：

```text
printf("%f ", value)
printf("\n")
```

在 LLVM IR 中会看到：

```llvm
declare i32 @printf(ptr, ...)
```

这只是声明，不是定义。

JIT 执行时，`printf` 通常由当前进程或系统 C runtime 提供。

所以这条链路依赖：

```text
JIT 能解析外部符号 printf。
目标平台支持对应调用约定。
```

这也是 JIT 教程中常见的运行时依赖问题。

Toy 示例为了教学简单，没有自己实现完整 runtime library，而是直接调用 C runtime 的 `printf`。

## 常见误区

### 误区 1：LLVM Dialect 已经等于 LLVM IR

不对。

LLVM Dialect 是 MLIR 中的 dialect。

LLVM IR 是 LLVM 的 IR 对象模型和文本格式。

`translateModuleToLLVMIR()` 才是两者之间的桥。

### 误区 2：`-emit=llvm` 会执行程序

不会。

`-emit=llvm` 只导出并打印 LLVM IR。

真正执行程序的是：

```bash
toyc-ch6 input.toy -emit=jit
```

### 误区 3：JIT 不需要 LLVM IR translation

不对。

Toy 的 `runJit()` 虽然没有显式调用 `translateModuleToLLVMIR()`，但 `ExecutionEngine` 内部仍然需要把 LLVM Dialect MLIR 翻译成 LLVM IR。

所以它同样需要：

```cpp
registerBuiltinDialectTranslation(...)
registerLLVMDialectTranslation(...)
```

### 误区 4：`-opt` 只影响 LLVM 优化

不对。

在 Ch6 中，`-opt` 会影响 Affine 层优化，也会影响 LLVM IR 优化级别。

另外，对 `-emit=llvm` 和 `-emit=jit` 来说，一些高层 cleanup pass 即使不加 `-opt` 也会跑，因为 lowering 需要它们提供更规整的 IR。

### 误区 5：JIT 测试一定会检查输出

不一定。

`jit.toy` 只运行命令，不用 `FileCheck` 检查打印结果。

这类测试能验证：

```text
编译和执行没有失败。
```

但不能严格验证 stdout 内容。

## 动手观察

### 观察 1：导出 LLVM IR

运行：

```bash
toyc-ch6 mlir/test/Examples/Toy/Ch6/jit.toy -emit=llvm
```

观察：

- 是否有 `target datalayout`。
- 是否有 `target triple`。
- 是否有 `declare ... @printf`。
- 是否有 `define void @main()`。
- 是否有 `call ... @printf`。

### 观察 2：比较 `-opt`

运行：

```bash
toyc-ch6 mlir/test/Examples/Toy/Ch6/jit.toy -emit=llvm
```

再运行：

```bash
toyc-ch6 mlir/test/Examples/Toy/Ch6/jit.toy -emit=llvm -opt
```

对比：

- LLVM IR 行数是否变化。
- alloc/load/store 是否变化。
- 常量是否被折叠。
- 控制流是否更简洁。

### 观察 3：运行 JIT

运行：

```bash
toyc-ch6 mlir/test/Examples/Toy/Ch6/jit.toy -emit=jit
```

观察输出。

如果命令失败，优先检查：

- 是否构建了 `toyc-ch6`。
- 是否启用了 `MLIR_ENABLE_EXECUTION_ENGINE`。
- 当前平台是否支持该测试。
- 是否能解析 `printf`。

### 观察 4：对比三个层级

对同一个输入依次运行：

```bash
toyc-ch6 input.toy -emit=mlir-llvm
toyc-ch6 input.toy -emit=llvm
toyc-ch6 input.toy -emit=jit
```

分别回答：

- 哪一个输出 MLIR？
- 哪一个输出 LLVM IR？
- 哪一个直接执行？
- 哪一个还能看到 MLIR operation？
- 哪一个还能看到 LLVM IR 文本？

### 观察 5：改掉 `main` 名字

把 Toy 程序改成：

```toy
def not_main() {
  print([[1, 2]]);
}
```

然后运行：

```bash
toyc-ch6 input.toy -emit=jit
```

观察是否能找到入口。

思考：

```text
为什么 dump MLIR 或 dump LLVM IR 可能成功，
但 JIT 执行会失败？
```

## 本节练习

### 练习 1：解释 translation

回答：

- LLVM Dialect 和 LLVM IR 有什么区别？
- 为什么需要 `translateModuleToLLVMIR()`？
- `mlir::MLIRContext` 和 `llvm::LLVMContext` 分别管理什么？

### 练习 2：解释 translation 注册

回答：

- `registerBuiltinDialectTranslation()` 的作用是什么？
- `registerLLVMDialectTranslation()` 的作用是什么？
- dialect 注册和 translation 注册有什么区别？

### 练习 3：解释 target 信息

回答：

- `InitializeNativeTarget()` 做什么？
- `InitializeNativeTargetAsmPrinter()` 做什么？
- target triple 描述什么？
- data layout 描述什么？
- 为什么导出/JIT 前需要设置这些信息？

### 练习 4：解释 LLVM IR 优化

回答：

- `makeOptimizingTransformer()` 返回的 transformer 用在哪里？
- `-opt` 如何影响 optLevel？
- 为什么 `-emit=llvm -opt` 和 `-emit=llvm` 输出可能不同？

### 练习 5：解释 JIT 执行路径

回答：

- `ExecutionEngine::create()` 大致做什么？
- 为什么 `runJit()` 不显式调用 `translateModuleToLLVMIR()`？
- `engineOptions.transformer` 的作用是什么？
- `invokePacked("main")` 调用的是什么？

### 练习 6：解释 `printf`

回答：

- `printf` 声明在哪里生成？
- `printf` 的定义来自哪里？
- 为什么 JIT 执行时可能遇到外部符号解析问题？
- 为什么 `jit.toy` 在 Windows 上被标为 unsupported？

### 练习 7：画完整链路图

画出从 `.toy` 到 JIT 的完整流程，并标注这些函数或 pass：

```text
parseInputFile
mlirGen
createInlinerPass
createShapeInferencePass
createLowerToAffinePass
createLowerToLLVMPass
translateModuleToLLVMIR
ExecutionEngine::create
invokePacked
```

## 本节小结

本节完成了 Toy Ch6 的代码生成闭环：

```text
LLVM Dialect
  -> LLVM IR
  -> JIT 执行
```

需要记住：

- `-emit=mlir-llvm` 输出 LLVM Dialect MLIR。
- `-emit=llvm` 调用 `dumpLLVMIR()`，输出真正的 LLVM IR。
- `-emit=jit` 调用 `runJit()`，直接 JIT 编译并执行 `main`。
- LLVM Dialect 导出 LLVM IR 前必须注册 dialect translation。
- `translateModuleToLLVMIR()` 把 `mlir::ModuleOp` 转成 `llvm::Module`。
- target triple 和 data layout 让 LLVM module 具备目标平台信息。
- `makeOptimizingTransformer()` 在 LLVM IR/JIT 阶段应用优化。
- `ExecutionEngine` 封装了 LLVM IR translation、优化、ORC JIT 编译和符号调用。
- `invokePacked("main")` 是 Toy JIT 的执行入口。

下一课会进入 Ch7，学习 Toy 如何扩展 `struct` 复合类型，并把前面学过的 Dialect、Type、Operation、Lowering 串起来。

## 学习记录模板

```text
本节主题：LLVM IR 导出与 JIT 执行
我读过的源码：
我观察过的测试：
我理解的 LLVM Dialect 与 LLVM IR 区别：
我理解的 translation 注册：
我理解的 target triple / data layout：
我理解的 dumpLLVMIR 路径：
我理解的 runJit 路径：
我观察到的 JIT 输出：
我还不理解的问题：
下一步要验证的小实验：
```
