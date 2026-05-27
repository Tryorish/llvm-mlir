# 第 14 课练习答案：Inliner、CSE 与高层 IR 优化组合

## 填写说明

这里填写第十四节课的练习答案和实验记录。写完后告诉我“检查第十四课答案”，我会对照 `Tutorial/14-inliner-cse-high-level-optimization.md`、Ch4 `toyc.cpp` 和优化测试帮你验证。

## 练习 1：解释 Ch4 pipeline

### 我的答案

```text

```

## 练习 2：解释 inliner 为什么在前

### 我的答案

- inliner 前 IR：
- inliner 后 IR：
- 为什么有利于 shape inference：

## 练习 3：解释 `handleTerminator`

### 我的答案

- 它处理哪个 op：
- return operand 如何替换调用结果：
- 为什么 inliner 需要这个 hook：

## 练习 4：解释 `toy.cast`

### 我的答案

- `toy.cast` 什么时候产生：
- 它是否改变数据：
- canonicalizer 如何清理：

## 练习 5：分析 `shape_inference.mlir`

### 我的答案

- 输入中的未知 shape：
- 推导后的具体 shape：
- 哪些 pass 参与：

## 练习 6：分析 `transpose_transpose.toy`

### 我的答案

- 原始计算：
- 优化前 IR：
- 优化后 IR：
- 应用的 pattern：

## 练习 7：设计一个 CSE 观察例子

### 我的答案

```toy

```

### 我的观察

- 重复表达式：
- CSE 前：
- CSE 后：

## 我的问题

- 
