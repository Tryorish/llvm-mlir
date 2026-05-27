# 第 15 课练习答案：Dialect Conversion 基础

## 填写说明

这里填写第十五节课的练习答案和实验记录。写完后告诉我“检查第十五课答案”，我会对照 `Tutorial/15-dialect-conversion-basics.md` 和 Ch5 lowering 代码帮你验证。

## 练习 1：解释 conversion target

### 我的答案

- ConversionTarget 定义什么：
- legal op：
- illegal op：
- dynamically legal op：

## 练习 2：解释 Ch5 为什么是 partial lowering

### 我的答案

- 被 lowering 的 op：
- 保留的 op：
- 为什么不是 full lowering：

## 练习 3：解释 `toy.print` 的动态合法性

### 我的答案

- 哪种 `toy.print` 合法：
- 哪种 `toy.print` 不合法：
- 为什么需要动态判断：

## 练习 4：比较三种 pattern

### 我的答案

| Pattern 类型 | 适合场景 | Toy 中例子 |
| --- | --- | --- |
| `OpRewritePattern` |  |  |
| `ConversionPattern` |  |  |
| `OpConversionPattern` |  |  |

## 练习 5：解释 `applyPartialConversion`

### 我的答案

- 输入：
- target：
- patterns：
- 成功条件：
- 失败条件：

## 练习 6：整理 Toy op lowering 表

### 我的答案

| Toy op | lowering 后 | 是否保留 |
| --- | --- | --- |
|  |  |  |

## 练习 7：预测转换失败

### 我的答案

- 失败 IR：
- 失败原因：
- 缺少的 pattern 或合法性配置：

## 我的问题

- 
