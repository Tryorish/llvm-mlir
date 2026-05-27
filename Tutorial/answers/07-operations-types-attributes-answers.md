# 第 7 课练习答案：Operation 的参数、结果、类型和属性

## 填写说明

这里填写第七节课的练习答案和实验记录。写完后告诉我“检查第七课答案”，我会对照 `Tutorial/07-operations-types-attributes.md` 和 Toy ODS 定义帮你验证。

## 练习 1：标注 `ConstantOp`

### 我的答案

- operation name：
- operands：
- attributes：
- results：
- result type：
- traits：
- verifier/folder：

## 练习 2：区分 `AddOp` 和 `PrintOp`

### 我的答案

| 项目 | `toy.add` | `toy.print` |
| --- | --- | --- |
| operands |  |  |
| results |  |  |
| side effect / purity |  |  |
| 是否参与 shape inference |  |  |

## 练习 3：解释 `scalar.toy`

### 我的答案

- scalar 在 Toy 源码中的写法：
- scalar 在 AST 中的表达：
- scalar 在 MLIR 中的 tensor 类型：
- 为什么仍然使用 tensor：

## 练习 4：解释 `GenericCallOp`

### 我的答案

- callee 是 operand 还是 attribute：
- inputs 是什么：
- result type 如何设置：
- 为什么它需要 CallOpInterface：

## 练习 5：解释 `ReturnOp` traits

### 我的答案

- `Pure`：
- `HasParent<"FuncOp">`：
- `Terminator`：
- verifier 还要检查什么：

## 练习 6：分析 `invalid.mlir`

### 我的答案

- 我观察到的非法 IR：
- 报错信息：
- 触发哪个 verifier：
- 为什么非法：

## 练习 7：整理 Toy operation 语义表

### 我的答案

| Operation | 语义 | operands | results | attributes | 备注 |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

## 我的问题

- 
