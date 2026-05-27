# 第 9 课练习答案：函数、调用、作用域与符号

## 填写说明

这里填写第九节课的练习答案和实验记录。写完后告诉我“检查第九课答案”，我会对照 `Tutorial/09-functions-calls-symbols-scope.md`、`MLIRGen.cpp` 和 `Ops.td` 帮你验证。

## 练习 1：拆解 `toy.func`

### 我的答案

- operation name：
- symbol name：
- function type：
- region：
- block arguments：
- traits/interfaces：

## 练习 2：解释函数参数绑定

### 我的答案

- AST 参数在哪里：
- MLIR block argument 在哪里创建：
- symbol table 如何绑定：
- 函数体中如何引用参数：

## 练习 3：区分变量表和符号引用

### 我的答案

| 概念 | 例子 | 作用域 | 用途 |
| --- | --- | --- | --- |
| SSA value / 变量表 |  |  |  |
| symbol / 符号引用 |  |  |  |

## 练习 4：分析 `GenericCallOp`

### 我的答案

- callee attribute：
- operands：
- result type：
- CallOpInterface 方法：
- 和 `toy.func` 的关联：

## 练习 5：解释 builtin 调用

### 我的答案

- `transpose` 为什么不是 `toy.generic_call`：
- `print` 为什么不是普通表达式 call：
- MLIRGen 中对应分支：

## 练习 6：解释 return 类型推导

### 我的答案

- 函数创建时 result type：
- 遇到 `toy.return` 后如何更新：
- `main` 为什么通常没有返回值：

## 练习 7：画出调用链

### 我的答案

```text

```

## 我的问题

- 
