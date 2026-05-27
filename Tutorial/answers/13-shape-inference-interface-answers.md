# 第 13 课练习答案：Shape Inference Interface

## 填写说明

这里填写第十三节课的练习答案和实验记录。写完后告诉我“检查第十三课答案”，我会对照 `Tutorial/13-shape-inference-interface.md`、`ShapeInferenceInterface.td` 和 `ShapeInferencePass.cpp` 帮你验证。

## 练习 1：解释 ShapeInference interface

### 我的答案

- interface 定义位置：
- operation 如何声明实现：
- C++ 需要实现什么方法：
- pass 如何调用：

## 练习 2：列出实现接口的 op

### 我的答案

| Operation | 是否实现 | inferShapes 行为 |
| --- | --- | --- |
|  |  |  |

## 练习 3：手动推导 transpose

### 我的答案

- 输入类型：
- 输出 shape：
- 推导依据：

## 练习 4：手动推导 mul

### 我的答案

- lhs 类型：
- rhs 类型：
- 输出类型：
- 是否需要检查 shape 兼容：

## 练习 5：解释 worklist

### 我的答案

- worklist 中放什么：
- 何时处理一个 op：
- 何时无法继续：

## 练习 6：解释 shape inference 和 inliner

### 我的答案

- 为什么 inliner 在前：
- inline 后 shape 信息如何变多：
- shape inference 如何利用这些信息：

## 练习 7：画出 shape inference 流程

### 我的答案

```text

```

## 我的问题

- 
