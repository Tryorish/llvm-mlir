# 第 19 课练习答案：LLVM IR 导出与 JIT 执行

## 填写说明

这里填写第十九节课的练习答案和实验记录。写完后告诉我“检查第十九课答案”，我会对照 `Tutorial/19-llvm-ir-export-and-jit.md` 和 Ch6 `toyc.cpp` 帮你验证。

## 练习 1：解释 translation

### 我的答案

- LLVM Dialect 和 LLVM IR 的区别：
- 为什么需要 `translateModuleToLLVMIR()`：
- `mlir::MLIRContext` 管理：
- `llvm::LLVMContext` 管理：

## 练习 2：解释 translation 注册

### 我的答案

- `registerBuiltinDialectTranslation()`：
- `registerLLVMDialectTranslation()`：
- dialect 注册和 translation 注册区别：

## 练习 3：解释 target 信息

### 我的答案

- `InitializeNativeTarget()`：
- `InitializeNativeTargetAsmPrinter()`：
- target triple：
- data layout：
- 为什么需要这些信息：

## 练习 4：解释 LLVM IR 优化

### 我的答案

- `makeOptimizingTransformer()` 用在哪里：
- `-opt` 如何影响 optLevel：
- 为什么输出可能不同：

## 练习 5：解释 JIT 执行路径

### 我的答案

- `ExecutionEngine::create()`：
- 为什么 `runJit()` 不显式调用 `translateModuleToLLVMIR()`：
- `engineOptions.transformer`：
- `invokePacked("main")`：

## 练习 6：解释 `printf`

### 我的答案

- `printf` 声明在哪里生成：
- `printf` 定义来自哪里：
- 外部符号解析风险：
- Windows 上 unsupported 的原因：

## 练习 7：画完整链路图

### 我的答案

```text

```

## 我的问题

- 
