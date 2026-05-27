# 第 16 课练习答案：Toy 到 Affine/MemRef/Func 的 Partial Lowering

## 填写说明

这里填写第十六节课的练习答案和实验记录。写完后告诉我“检查第十六课答案”，我会对照 `Tutorial/16-toy-to-affine-memref-func-lowering.md` 和 `LowerToAffineLoops.cpp` 帮你验证。

## 练习 1：解释 `lowerOpToLoops`

### 我的答案

- 输入 op：
- 分配 result memref：
- loop nest 如何生成：
- `processIteration` 的作用：
- 如何替换原 op：

## 练习 2：手动 lower 一个 transpose

### 我的答案

```mlir

```

### 我的解释

- load index：
- store index：
- shape 变化：

## 练习 3：手动 lower 一个 mul

### 我的答案

```mlir

```

### 我的解释

- 两个 operand 如何 load：
- 标量运算：
- result 如何 store：

## 练习 4：解释 constant lowering

### 我的答案

- dense attr 如何展开：
- memref 如何分配：
- 每个元素如何 store：

## 练习 5：解释 `toy.print` 为什么保留

### 我的答案

- 保留后的 operand type：
- 为什么这叫 partial lowering：
- 后续由哪里处理：

## 练习 6：解释 `toy.func` 和 `toy.return`

### 我的答案

- `toy.func` lowering 到：
- `toy.return` lowering 到：
- 函数类型如何转换：

## 练习 7：整理完整转换表

### 我的答案

| Toy op | Lowering pattern | 输出 dialect/op |
| --- | --- | --- |
|  |  |  |

## 我的问题

- 
