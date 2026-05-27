# 第 20 课练习答案：StructType、struct 操作与课程项目

## 填写说明

这里填写第二十节课的练习答案、实验记录和课程项目设计。写完后告诉我“检查第二十课答案”，我会对照 `Tutorial/20-struct-type-and-course-project.md` 和 Ch7 源码/测试帮你验证。

## 练习 1：解释 Ch7 新增语法

### 我的答案

- `struct Struct { var a; var b; }` 在 AST 中如何表示：
- `Struct value = {...}` 和 `var a<2, 3> = ...` 的类型信息差异：
- `value.a` 为什么在 AST 中是 `BinOp: .`：

## 练习 2：解释 StructType

### 我的答案

- `StructTypeStorage` 存储：
- `StructType::get()` 为什么需要 element types：
- `!toy.struct<tensor<*xf64>, tensor<*xf64>>` 表示：
- Ch7 的 `StructType` 是否保存字段名：

## 练习 3：解释 struct operation

### 我的答案

- `toy.struct_constant` 的 attribute 类型：
- `toy.struct_access` 的 `index`：
- `StructAccessOp::verify()` 检查：
- `StructConstantOp::verify()` 为什么需要递归：

## 练习 4：解释 struct 优化

### 我的答案

- `StructConstantOp::fold()` 返回：
- `StructAccessOp::fold()` 成功条件：
- `struct-opt.mlir` 中两级 access 如何折叠：

## 练习 5：解释 `struct-codegen.toy`

### 我的答案

- 未优化 IR 中为什么有 `toy.generic_call`：
- `-opt` 后为什么 `multiply_transpose` 消失：
- `value.a` 和 `value.b` 最终为什么能变成同一个 tensor constant：
- shape inference 的作用：

## 练习 6：解释 lowering 设计

### 我的答案

- Ch7 是否实现 struct 到 LLVM struct 的 lowering：
- 为什么 Toy 可以不实现：
- struct 未在 affine lowering 前消掉会怎样：
- 如果要支持运行时 struct，从哪里改：

## 练习 7：设计一个扩展

### 我的项目设计

```text
语法：
新增 builtin 调用形式 `neg(x)`，先不扩展一元 `-x` 语法。

AST 改动：
不新增 AST 节点，直接复用现有 `CallExprAST`。
原因是 `neg(x)` 可以先走和 `transpose(x)` 一样的 builtin 路线，改动最小。

ODS 改动：
在 `mlir/examples/toy/Ch7/include/toy/Ops.td` 中新增 `NegOp`。
输入为一个 `F64Tensor`，输出为一个 `F64Tensor`。
声明 `ShapeInferenceOpInterface`，并开启 `hasCanonicalizer = 1`。

MLIRGen 改动：
在 `mlir/examples/toy/Ch7/mlir/MLIRGen.cpp` 的 `mlirGen(CallExprAST &call)` 中识别 `callee == "neg"`。
当参数个数为 1 时，生成 `builder.create<NegOp>(location, operands[0])`。

优化改动：
在 `mlir/examples/toy/Ch7/mlir/ToyCombine.cpp` 中新增 C++ canonicalization pattern：
`neg(neg(x)) -> x`。
并在 `NegOp::getCanonicalizationPatterns(...)` 中注册。

Lowering 改动：
在 `mlir/examples/toy/Ch7/mlir/LowerToAffineLoops.cpp` 中新增一元 lowering helper，
将 `toy.neg` lowering 为：
`affine.load -> arith.negf -> affine.store`。

测试计划：
1. `neg-codegen.toy`
   验证 Toy 源码 `neg(a)` 能生成 `toy.neg`。
2. `neg-affine-lowering.mlir`
   验证 `toy.neg` 能 lowering 到 `arith.negf`。
3. `neg-neg.toy`
   验证 `-opt` 下 `neg(neg(x)) -> x` 的 canonicalization 生效。

风险点：
1. ODS assembly format 中 `type(results)` 和命名 result 的写法容易出错。
2. C++ 中 helper、op 名、`NegFOp` 大小写容易拼错。
3. `.toy` 和 `.mlir` 测试文件的注释风格不同，前者用 `#`，后者用 `//`。
4. FileCheck 变量绑定顺序必须和实际 lowering 输出一致。
```

## 课程项目记录

- 我选择的项目：
  项目 A：新增 `toy.neg`

- 当前进度：
  已完成最小闭环实现，并通过相关测试。

- 已修改文件：
  `mlir/examples/toy/Ch7/include/toy/Ops.td`
  `mlir/examples/toy/Ch7/mlir/Dialect.cpp`
  `mlir/examples/toy/Ch7/mlir/MLIRGen.cpp`
  `mlir/examples/toy/Ch7/mlir/LowerToAffineLoops.cpp`
  `mlir/examples/toy/Ch7/mlir/ToyCombine.cpp`

- 已添加测试：
  `mlir/test/Examples/Toy/Ch7/neg-codegen.toy`
  `mlir/test/Examples/Toy/Ch7/neg-affine-lowering.mlir`
  `mlir/test/Examples/Toy/Ch7/neg-neg.toy`

- 实现结果：
  `neg(a)` 可以在 Toy 源码层生成 `toy.neg`。
  `toy.neg` 可以经过 shape inference，并在 affine lowering 中变成 `arith.negf`。
  在 `-opt` 下，`neg(neg(x)) -> x` 会被 canonicalizer 消掉。

- 踩坑记录：
  1. `LowerToAffineLoops.cpp` 中一度把 `lowerOpToLoops` 拼成了 `lowerOPToLoops`。
  2. `arith::NegFOp` 一度写成了错误的大小写形式。
  3. `.toy` 测试文件首行必须使用 `# RUN:`，不能写成 `// RUN:`。
  4. `neg-affine-lowering.mlir` 的 FileCheck 里，输入/输出 `memref.alloc` 绑定顺序一开始写反了。
  5. `neg(neg(a))` 的优化结果比预期更强，连冗余 `reshape` 一起被 canonicalizer 清掉了，因此测试期望需要按真实优化结果收紧。

- 待解决问题：
  1. 是否继续支持一元语法 `-x`，而不只是 builtin `neg(x)`。
  2. 是否给 `NegOp` 增加更多 canonicalization，例如常量折叠。
  3. 是否需要补充 `-emit=mlir -opt` 下 `toy.neg` shape 推导的专门测试。

## 我的问题

- 
