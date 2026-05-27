# 第 17 课练习答案：Affine 优化与低层 IR 观察

## 填写说明

这里填写第十七节课的练习答案和实验记录。写完后告诉我“检查第十七课答案”，我会对照 `Tutorial/17-affine-optimization-and-low-level-ir.md` 和 Ch5/Ch6 affine lowering 测试帮你验证。

## 练习 1：分层注释 affine IR

### 我的答案

```mlir

```

## 练习 2：对比 alloc 数量

### 我的答案

| 模式 | alloc 数量 | 每个 alloc 的用途 |
| --- | --- | --- |
| 未开启 `-opt` |  |  |
| 开启 `-opt` |  |  |

## 练习 3：解释融合后的 loop

### 我的答案

- load 使用 `[J, I]` 的原因：
- store 使用 `[I, J]` 的原因：
- 同时完成的 Toy operation：

## 练习 4：解释 `toy.print`

### 我的答案

- 为什么还存在：
- operand type：
- 和 partial lowering 的关系：

## 练习 5：解释 `-opt`

### 我的答案

- pipeline 差异：
- 额外 affine pass：
- 最直观效果：

## 练习 6：预测一个不能融合的场景

### 我的答案

- 示例场景：
- 为什么不能融合：
- 依赖关系如何影响：

## 练习 7：写一份低层 IR 速查表

### 我的答案

| op | dialect | 职责 |
| --- | --- | --- |
|  |  |  |

## 我的问题

- 
