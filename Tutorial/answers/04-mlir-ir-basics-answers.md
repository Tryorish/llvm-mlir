# 第 4 课练习答案：MLIR 核心概念入门

## 填写说明

这里填写第四节课的练习答案和实验记录。写完后告诉我“检查第四课答案”，我会对照 `Tutorial/04-mlir-ir-basics.md`、Ch2 源码和测试帮你验证。

## 练习 1：拆解一条 operation

题目 operation：

```mlir
%0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
```

### 我的拆解

- operation name：
- result：
- operand：
- operand type：
- result type：
- 所属 dialect：

## 练习 2：区分 operand 和 attribute

题目 operation：

```mlir
%0 = toy.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
%1 = toy.reshape(%0 : tensor<2xf64>) to tensor<1x2xf64>
```

### 我的答案

- `toy.constant` 有没有 operand：
- `dense<[...]>` 是 operand 还是 attribute：
- `toy.reshape` 的 operand 是什么：
- `%1` 是谁定义的：

## 练习 3：标注 Ch2 `codegen.toy` 输出

### 我运行过的命令

```bash

```

### 我的标注

- 所有 `toy.func`：
- 所有 block arguments：
- 所有 `toy.constant`：
- 所有 `toy.reshape`：
- 所有 `toy.generic_call`：
- 所有 `toy.print`：
- 所有 `toy.return`：

## 练习 4：画出 def-use 链

题目 IR：

```mlir
%0 = toy.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
%1 = toy.reshape(%0 : tensor<2xf64>) to tensor<1x2xf64>
toy.print %1 : tensor<1x2xf64>
```

### 我的 def-use 链

- 谁定义 `%0`：
- 谁使用 `%0`：
- 谁定义 `%1`：
- 谁使用 `%1`：

## 练习 5：区分 SSA value 和 symbol

题目 IR：

```mlir
toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
  %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
  toy.return %0 : tensor<*xf64>
}

toy.func @main() {
  %1 = toy.generic_call @multiply_transpose(%0, %0)
       : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
  toy.return
}
```

### 我的答案

- `@multiply_transpose` 是什么：
- `%arg0` 是什么：
- `%0` 是什么：
- `%1` 是什么：
- 哪些是 SSA value：
- 哪些是 symbol：

## 练习 6：AST 到 MLIR 对照

题目 Toy 源码：

```toy
def main() {
  var a<2> = [1, 2];
  print(a);
}
```

### 我认为会生成的 Toy MLIR operation 顺序

```text

```

### 我的解释

- 为什么会有 `toy.func`：
- 为什么会有 `toy.constant`：
- 是否会有 `toy.reshape`：
- 为什么会有 `toy.print`：
- 为什么会有 `toy.return`：

## 我的问题

- 
