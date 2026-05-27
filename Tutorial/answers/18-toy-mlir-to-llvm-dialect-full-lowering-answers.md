# 第 18 课练习答案：Toy/MLIR 到 LLVM Dialect 的 full lowering

## 填写说明

这里填写第十八节课的练习答案和实验记录。写完后告诉我“检查第十八课答案”，我会对照 `Tutorial/18-toy-mlir-to-llvm-dialect-full-lowering.md` 和 Ch6 `LowerToLLVM.cpp` 帮你验证。

## 练习 1：解释 full lowering

### 我的答案

- partial lowering：
- full lowering：
- Ch5 affine lowering 为什么是 partial：
- Ch6 为什么用 `applyFullConversion()`：

## 练习 2：画出 Ch6 pipeline

### 我的答案

```text

```

## 练习 3：解释 `toy.print` lowering

### 我的答案

- 为什么需要自定义 lowering：
- 创建了哪些 operation：
- 为什么需要 memref shape：
- 为什么要插入 `printf` 和 global string：

## 练习 4：解释 TypeConverter

### 我的答案

- `LLVMTypeConverter` 负责：
- memref 为什么不能原样保留：
- function type / block argument 为什么也要转换：

## 练习 5：解释 ConversionTarget

### 我的答案

- `LLVMConversionTarget` 定义：
- `target.addLegalOp<ModuleOp>()` 的原因：
- 如果还剩 `toy.print` 会发生什么：

## 练习 6：跟踪一个 operation

### 我的答案

- 我选择的 operation：
- lowering 前：
- 负责的 pattern：
- lowering 后：

## 练习 7：新增 Toy operation 的 lowering 思考

### 我的答案

- `toy.neg` 应该在哪个阶段 lowering：
- 是否可以先 lowering 到 `arith.negf`：
- 如果 full conversion 前还保留会发生什么：
- 需要添加的 pattern：

## 我的问题

- 
