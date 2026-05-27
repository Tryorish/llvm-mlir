# 第 6 课练习答案：Toy Dialect 与 ODS/TableGen

## 填写说明

这里填写第六节课的练习答案和实验记录。写完后告诉我“检查第六课答案”，我会对照 `Tutorial/06-dialect-and-ods.md`、Ch2/Ch3 的 ODS 和生成文件帮你验证。

## 练习 1：画出 TableGen 生成链路

### 我的答案

```text

```

## 练习 2：解释 `Toy_Dialect`

### 我的答案

- `name`：
- `cppNamespace`：
- `hasConstantMaterializer`：
- 对应生成文件：

## 练习 3：解释 `Toy_Op`

### 我的答案

- `Toy_Op` 的作用：
- 它继承自：
- mnemonic 的作用：
- traits 的作用：

## 练习 4：找出四个 operation 的 ODS 定义

### 我的答案

| Operation | ODS 定义位置 | operands | results | traits/interfaces |
| --- | --- | --- | --- | --- |
| `toy.constant` |  |  |  |  |
| `toy.reshape` |  |  |  |  |
| `toy.transpose` |  |  |  |  |
| `toy.return` |  |  |  |  |

## 练习 5：解释 `GET_OP_CLASSES`

### 我的答案

- 出现位置：
- 展开后提供什么：
- 为什么 `Dialect.h` 需要它：

## 练习 6：解释 `GET_OP_LIST`

### 我的答案

- 出现位置：
- 展开后提供什么：
- 为什么 `Dialect.cpp` 注册 operation 时需要它：

## 练习 7：从 ODS 追踪到 MLIR 输出

### 我的答案

- 我选择的 operation：
- ODS 定义：
- C++ 类/方法：
- 输出 MLIR 文本：
- 对应测试：

## 我的问题

- 
