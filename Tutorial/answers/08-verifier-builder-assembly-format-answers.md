# 第 8 课练习答案：Verifier、Builder 与自定义 Assembly Format

## 填写说明

这里填写第八节课的练习答案和实验记录。写完后告诉我“检查第八课答案”，我会对照 `Tutorial/08-verifier-builder-assembly-format.md` 和 Ch2/Ch3 的 `Dialect.cpp`、`Ops.td` 帮你验证。

## 练习 1：解释 `hasVerifier`

### 我的答案

- ODS 中如何开启：
- 生成了什么声明：
- C++ 中在哪里实现：
- verifier 何时运行：

## 练习 2：拆解 `ConstantOp::verify()`

### 我的答案

- 检查的 attribute：
- 检查的 result type：
- rank 检查：
- shape 检查：
- 报错场景：

## 练习 3：拆解 `ReturnOp::verify()`

### 我的答案

- parent 检查依赖什么 trait：
- operand 数量检查：
- function result 数量检查：
- operand type 和 result type 如何比较：

## 练习 4：对比 builder 形式

### 我的答案

| Builder | 输入参数 | 设置 operands | 设置 attributes | 设置 result type |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

## 练习 5：区分 custom parser/printer 和 declarative assembly format

### 我的答案

- custom parser/printer 适合什么场景：
- declarative assembly format 适合什么场景：
- Toy 中哪些 operation 使用 custom：
- Toy 中哪些 operation 使用 declarative：

## 练习 6：分析 `invalid.mlir`

### 我的答案

- 我运行的命令：
- 报错信息：
- 对应 verifier：
- 修正方式：

## 练习 7：补写一张“生成路径图”

### 我的答案

```text

```

## 我的问题

- 
