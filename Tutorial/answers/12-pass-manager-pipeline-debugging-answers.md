# 第 12 课练习答案：Pass Manager、Pass Pipeline 与调试

## 填写说明

这里填写第十二节课的练习答案和实验记录。写完后告诉我“检查第十二课答案”，我会对照 `Tutorial/12-pass-manager-pipeline-debugging.md` 和各章 `toyc.cpp` 帮你验证。

## 练习 1：画 Ch3 pipeline

### 我的答案

```text

```

## 练习 2：画 Ch4 pipeline

### 我的答案

```text

```

## 练习 3：解释 nested pass

### 我的答案

- `pm.nest<...>()` 的作用：
- 为什么要嵌套到函数：
- 和 module pass 的区别：

## 练习 4：解释 lowering 触发条件

### 我的答案

- `isLoweringToAffine`：
- `isLoweringToLLVM`：
- `emitAction` 如何影响 pipeline：

## 练习 5：解释 pass 顺序

### 我的答案

- pass 顺序：
- 为什么这样排序：
- 如果调换可能发生什么：

## 练习 6：观察调试输出

### 我的实验记录

```bash

```

### 我的观察

- pass 前后变化：
- 哪个 pass 影响最大：

## 练习 7：整理 pipeline 对比表

### 我的答案

| 章节/命令 | Pipeline | 输出层级 |
| --- | --- | --- |
|  |  |  |

## 我的问题

- 
