# MLIR Toy 20 节课程大纲

## 课程定位

本课程面向正在学习 `mlir/examples/toy` 的读者，目标是用约 20 节课循序渐进地覆盖 Toy 教程涉及的主要知识点：Toy 语言前端、AST、MLIR 基础 IR、Dialect、ODS/TableGen、Operation 定义、Verifier、Builder、自定义 assembly format、MLIRGen、Pattern Rewrite、Canonicalization、DRR、Pass、Shape Inference、Dialect Conversion、Affine lowering、LLVM lowering、JIT、测试与调试，以及 Ch7 的 `struct` 复合类型扩展。

课程主线是：

```text
Toy 源码
  -> Lexer / Parser / AST
  -> Toy Dialect MLIR
  -> Canonicalization / Shape Inference
  -> Affine / Arith / MemRef / Func 等低层 Dialect
  -> LLVM Dialect
  -> LLVM IR / JIT 执行
```

每节课建议包含四个固定环节：

- 概念：这节课要解决哪个编译器或 MLIR 问题。
- 源码：阅读 `mlir/examples/toy/Ch*` 中的关键实现。
- IR：观察 `mlir/test/Examples/Toy/Ch*` 中输入、输出和 FileCheck 期望。
- 实验：做一个小改动，验证自己真的理解了这节课。

## 推荐前置知识

学习前不要求完全掌握 LLVM/MLIR，但建议具备：

- C++ 基础：类、继承、模板、智能指针、`llvm::StringRef`、`llvm::ArrayRef`。
- 编译器基础：词法分析、语法分析、AST、IR、优化、代码生成。
- CMake/Ninja 基础：能构建 target，能运行测试。
- LLVM 项目基本结构：知道 `llvm/`、`mlir/`、`mlir/examples/`、`mlir/test/` 的职责。

如果本地已经配置好构建目录，可以优先构建这些目标：

```bash
ninja toyc-ch1
ninja toyc-ch2
ninja toyc-ch3
ninja toyc-ch4
ninja toyc-ch5
ninja toyc-ch6
ninja toyc-ch7
```

如果启用了 lit 测试，可以按章节运行 Toy 相关测试：

```bash
llvm-lit mlir/test/Examples/Toy/Ch1
llvm-lit mlir/test/Examples/Toy/Ch2
llvm-lit mlir/test/Examples/Toy/Ch3
llvm-lit mlir/test/Examples/Toy/Ch4
llvm-lit mlir/test/Examples/Toy/Ch5
llvm-lit mlir/test/Examples/Toy/Ch6
llvm-lit mlir/test/Examples/Toy/Ch7
```

## 课程总目标

完成 20 节课后，应该能够：

- 读懂 Toy 源语言、AST dump、Toy Dialect IR、lowering 后的 MLIR 和 LLVM Dialect IR。
- 理解 MLIR 中 Operation、Value、Type、Attribute、Region、Block、Module、Dialect 的关系。
- 使用 ODS/TableGen 定义自定义 Dialect 和 Operation。
- 理解 verifier、builder、自定义 parser/printer、trait、interface 的用途。
- 编写简单的 C++ rewrite pattern 和 DRR rewrite pattern。
- 理解 pass pipeline 如何串联 inliner、canonicalizer、CSE、shape inference、lowering pass。
- 理解 partial lowering 和 full lowering 的区别。
- 看懂 Toy 到 Affine/MemRef/Func/LLVM Dialect 的转换过程。
- 能基于 Toy 增加一个简单语言特性、一个 operation、一个 canonicalization pattern 或一个 lowering pattern。

## 阶段划分

### 第一阶段：前端与 MLIR 入门

目标是从 Toy 源码进入 AST，再进入最基本的 MLIR。

- 第 1 课：课程导览与环境确认
- 第 2 课：Toy 语言语法与 AST
- 第 3 课：Lexer、Parser 与 AST dump
- 第 4 课：MLIR 核心概念入门
- 第 5 课：从 AST 生成 MLIR

### 第二阶段：自定义 Dialect 与 Operation

目标是掌握 Toy Dialect 如何被定义、验证、打印和解析。

- 第 6 课：Toy Dialect 与 ODS/TableGen
- 第 7 课：Operation 的参数、结果、类型和属性
- 第 8 课：Verifier、Builder 与自定义 assembly format
- 第 9 课：函数、调用、作用域与符号

### 第三阶段：高层优化与分析

目标是掌握 MLIR 的 rewrite、canonicalization、interface 和 shape inference。

- 第 10 课：C++ Pattern Rewrite 与 Canonicalization
- 第 11 课：DRR 声明式重写
- 第 12 课：Pass Manager、Pass Pipeline 与调试
- 第 13 课：Shape Inference Interface
- 第 14 课：Inliner、CSE 与高层 IR 优化组合

### 第四阶段：逐步 Lowering 与代码生成

目标是理解 Toy IR 如何逐步降低到可执行代码。

- 第 15 课：Dialect Conversion 基础
- 第 16 课：Toy 到 Affine/MemRef/Func 的 partial lowering
- 第 17 课：Affine 优化与低层 IR 观察
- 第 18 课：Toy/MLIR 到 LLVM Dialect 的 full lowering
- 第 19 课：LLVM IR 导出与 JIT 执行

### 第五阶段：扩展 Toy 语言

目标是学习 Ch7 的复合类型扩展，并完成综合练习。

- 第 20 课：StructType、struct 操作与课程项目

## 详细课程安排

### 第 1 课：课程导览与环境确认

目标：

- 建立 Toy 教程的整体地图。
- 明确每个章节在编译链路中的位置。
- 确认本地源码、测试和构建目标的位置。

关键源码：

- `mlir/examples/toy/README.md`
- `mlir/docs/Tutorials/Toy/_index.md`
- `mlir/examples/toy/CMakeLists.txt`
- `mlir/test/Examples/Toy/`

核心知识点：

- Toy 教程 Ch1 到 Ch7 的演进关系。
- 源码目录和测试目录的对应关系。
- `toyc-ch1` 到 `toyc-ch7` 的角色。
- `-emit=ast`、`-emit=mlir`、`-emit=mlir-affine`、`-emit=mlir-llvm`、`-emit=llvm` 的观察方式。

实验：

- 找到每章的 `toyc.cpp`、`Ops.td`、`MLIRGen.cpp`、`Dialect.cpp`。
- 对比 Ch1、Ch2、Ch6、Ch7 的文件数量变化，理解功能是如何逐步加上的。

产出：

- 一张 Toy 编译链路笔记：从 `.toy` 到 AST、MLIR、Affine、LLVM Dialect、LLVM IR/JIT。

### 第 2 课：Toy 语言语法与 AST

目标：

- 理解 Toy 语言支持的基本语法。
- 理解 AST 如何表达函数、变量、字面量、调用和返回。

关键源码：

- `mlir/examples/toy/Ch1/include/toy/AST.h`
- `mlir/examples/toy/Ch1/parser/AST.cpp`
- `mlir/test/Examples/Toy/Ch1/ast.toy`
- `mlir/test/Examples/Toy/Ch1/empty.toy`

核心知识点：

- Module、Function、Prototype、Expression 的层次结构。
- `NumberExprAST`、`LiteralExprAST`、`VariableExprAST`、`VarDeclExprAST`。
- `BinaryExprAST`、`CallExprAST`、`PrintExprAST`、`ReturnExprAST`。
- AST dump 的缩进结构和语义。

实验：

- 在 `ast.toy` 中加入一个变量声明和一个二元表达式，观察 AST dump。
- 手动画出一个 Toy 函数对应的 AST 树。

产出：

- 一份 AST 节点速查表：节点名、含义、对应 Toy 语法。

### 第 3 课：Lexer、Parser 与 AST dump

目标：

- 理解 Toy 前端如何从字符流构造 AST。
- 掌握递归下降 parser 的基本结构。

关键源码：

- `mlir/examples/toy/Ch1/include/toy/Lexer.h`
- `mlir/examples/toy/Ch1/include/toy/Parser.h`
- `mlir/examples/toy/Ch1/toyc.cpp`
- `mlir/examples/toy/Ch1/parser/AST.cpp`

核心知识点：

- Token 类型和关键字识别。
- 表达式解析、函数解析、变量声明解析。
- Parser 错误处理方式。
- `toyc` 如何读取输入并触发 AST dump。

实验：

- 增加一个故意写错的 Toy 程序，观察 parser 报错。
- 找到 `print`、`return`、`def`、`var` 分别在哪里被识别和解析。

产出：

- 一份从源码文本到 AST dump 的调用链笔记。

### 第 4 课：MLIR 核心概念入门

目标：

- 在进入 Toy Dialect 之前，先理解 MLIR 的基本 IR 模型。
- 能读懂最基础的 MLIR 文本结构。

关键源码与文档：

- `mlir/docs/Tutorials/UnderstandingTheIRStructure.md`
- `mlir/docs/Tutorials/Toy/Ch-2.md`
- `mlir/test/Examples/Toy/Ch2/codegen.toy`

核心知识点：

- `Operation` 与 `Op` wrapper 的区别。
- `Value`、`Type`、`Attribute`。
- `Region`、`Block`、block argument。
- `ModuleOp`、symbol、location。
- Dialect 是 operation/type/attribute 的命名空间。

实验：

- 观察 Ch2 的 `-emit=mlir` 输出，标注每个 operation、operand、result、type、attribute。
- 找出 Toy IR 中哪些是 Toy Dialect，哪些是 builtin 或 func 相关概念。

产出：

- 一份 Toy MLIR 文本格式注释版。

### 第 5 课：从 AST 生成 MLIR

目标：

- 理解 MLIRGen 如何把 AST 转换为 Toy Dialect IR。
- 掌握 `OpBuilder`、location、symbol table 的基本使用。

关键源码：

- `mlir/examples/toy/Ch2/mlir/MLIRGen.cpp`
- `mlir/examples/toy/Ch2/include/toy/MLIRGen.h`
- `mlir/examples/toy/Ch2/toyc.cpp`
- `mlir/test/Examples/Toy/Ch2/codegen.toy`

核心知识点：

- `MLIRContext` 与 Dialect 注册。
- `OpBuilder` 创建 operation。
- AST location 到 MLIR location 的映射。
- 变量作用域与 symbol table。
- Toy 函数、变量、literal、binary、call、print 的 MLIRGen 路径。

实验：

- 选择一个 Toy 表达式，从 AST 节点一路追踪到生成的 MLIR operation。
- 给一个未知变量制造错误，观察 `emitError` 的位置和信息。

产出：

- 一份 AST 节点到 Toy operation 的映射表。

### 第 6 课：Toy Dialect 与 ODS/TableGen

目标：

- 理解 Toy Dialect 如何用 ODS 定义。
- 理解 TableGen 生成的 C++ 声明和定义如何接入源码。

关键源码：

- `mlir/examples/toy/Ch2/include/toy/Ops.td`
- `mlir/examples/toy/Ch2/include/toy/Dialect.h`
- `mlir/examples/toy/Ch2/mlir/Dialect.cpp`
- `mlir/examples/toy/Ch2/include/CMakeLists.txt`

核心知识点：

- `Dialect` 定义。
- `Toy_Op` 基类模板。
- mnemonic 与 operation name。
- ODS 生成 `.inc` 文件。
- `GET_OP_CLASSES`、`GET_OP_LIST` 等生成入口。

实验：

- 找到 `toy.constant`、`toy.add`、`toy.mul`、`toy.transpose` 在 ODS 中的定义。
- 对照生成后的 operation 在 C++ 中的使用方式。

产出：

- 一份 ODS 文件结构说明：Dialect、Op class、arguments、results、traits。

### 第 7 课：Operation 的参数、结果、类型和属性

目标：

- 深入理解 Toy operation 的数据建模。
- 能解释每个 Toy operation 的 operands、results、attributes 和 types。

关键源码：

- `mlir/examples/toy/Ch2/include/toy/Ops.td`
- `mlir/examples/toy/Ch2/mlir/Dialect.cpp`
- `mlir/test/Examples/Toy/Ch2/scalar.toy`
- `mlir/test/Examples/Toy/Ch2/invalid.mlir`

核心知识点：

- `ins`、`outs`。
- `F64Tensor`、`AnyTypeOf`、`StaticShapeTensorOf`。
- `DenseElementsAttr` / elements attribute。
- ranked tensor 与 unranked tensor。
- operation trait：`Pure`、`ConstantLike`、terminator、parent 限制等。

实验：

- 修改一个 Toy MLIR 测试，让 operation operand 类型不匹配，观察 verifier 报错。
- 手动标注 `toy.constant` 的 attribute 和 result type。

产出：

- 一张 Toy operation 语义表：operation、输入、输出、属性、约束。

### 第 8 课：Verifier、Builder 与自定义 Assembly Format

目标：

- 理解 operation 定义不只是声明字段，还要定义合法性、构造方式和文本格式。

关键源码：

- `mlir/examples/toy/Ch2/include/toy/Ops.td`
- `mlir/examples/toy/Ch2/mlir/Dialect.cpp`
- `mlir/test/Examples/Toy/Ch2/invalid.mlir`

核心知识点：

- `let hasVerifier = 1`。
- `verify()` 如何报告错误。
- ODS builder 与 C++ builder。
- custom parser/printer。
- generic assembly format 与 custom assembly format 的区别。

实验：

- 找到 `ConstantOp::verify()` 和 `TransposeOp::verify()`，总结各自检查了什么。
- 用 generic form 和 custom form 对照理解同一个 operation。

产出：

- 一份 verifier 编写原则笔记：检查什么、在哪里报错、如何定位。

### 第 9 课：函数、调用、作用域与符号

目标：

- 理解 Toy 中函数定义、函数调用和 return 如何映射到 MLIR。
- 理解符号引用和函数类型。

关键源码：

- `mlir/examples/toy/Ch2/include/toy/Ops.td`
- `mlir/examples/toy/Ch2/mlir/MLIRGen.cpp`
- `mlir/examples/toy/Ch2/mlir/Dialect.cpp`
- `mlir/test/Examples/Toy/Ch2/codegen.toy`

核心知识点：

- `toy.func`、`toy.generic_call`、`toy.return`。
- function type 与 function body region。
- block argument 与 Toy 函数参数。
- symbol name、callee attribute。
- return type 推导和函数签名。

实验：

- 写一个 Toy 函数调用另一个 Toy 函数，观察 MLIR。
- 制造返回类型不匹配的 MLIR，观察 verifier。

产出：

- 一份 Toy 函数从源码到 MLIR 的逐行对应说明。

### 第 10 课：C++ Pattern Rewrite 与 Canonicalization

目标：

- 理解 MLIR 如何通过 pattern rewrite 做局部优化。
- 掌握 canonicalization pattern 的注册方式。

关键源码：

- `mlir/examples/toy/Ch3/mlir/ToyCombine.cpp`
- `mlir/examples/toy/Ch3/include/toy/Ops.td`
- `mlir/test/Examples/Toy/Ch3/transpose_transpose.toy`

核心知识点：

- `OpRewritePattern`。
- `PatternRewriter`。
- `matchAndRewrite`。
- `replaceOp` 与 value 替换。
- `getCanonicalizationPatterns`。
- `createCanonicalizerPass()`。

实验：

- 追踪 `transpose(transpose(x)) -> x` 的优化过程。
- 增加一个不会匹配的例子，确认 canonicalizer 不会错误修改 IR。

产出：

- 一份 C++ rewrite pattern 模板笔记。

### 第 11 课：DRR 声明式重写

目标：

- 理解 TableGen DRR 如何表达简单 rewrite。
- 比较 C++ pattern 和 DRR 的适用场景。

关键源码：

- `mlir/examples/toy/Ch3/mlir/ToyCombine.td`
- `mlir/examples/toy/Ch3/mlir/ToyCombine.cpp`
- `mlir/examples/toy/Ch3/CMakeLists.txt`
- `mlir/test/Examples/Toy/Ch3/trivial_reshape.toy`

核心知识点：

- `Pat`。
- source pattern 与 result pattern。
- DRR 生成的 rewriter include 文件。
- DRR 适合结构简单、声明式清晰的变换。
- C++ pattern 适合复杂逻辑、类型推导或多步构造。

实验：

- 分析 trivial reshape 消除规则。
- 尝试写一个只匹配特定 operation 形态的 DRR 规则。

产出：

- 一份 C++ rewrite 与 DRR 对比表。

### 第 12 课：Pass Manager、Pass Pipeline 与调试

目标：

- 理解 Toy 编译器如何组织多个 pass。
- 学会观察 pass 前后的 IR 变化。

关键源码：

- `mlir/examples/toy/Ch3/toyc.cpp`
- `mlir/examples/toy/Ch4/toyc.cpp`
- `mlir/examples/toy/Ch6/toyc.cpp`
- `mlir/examples/toy/Ch7/toyc.cpp`

核心知识点：

- `PassManager`。
- `OpPassManager` 与 nested pass。
- module pass 与 function pass。
- inliner、canonicalizer、CSE、shape inference 的顺序。
- `applyPassManagerCLOptions()`。
- pass pipeline 调试选项。

实验：

- 对比 Ch3、Ch4、Ch6、Ch7 的 pass pipeline 差异。
- 调整 canonicalizer 与 shape inference 的顺序，思考为什么顺序重要。

产出：

- 一张 Toy pass pipeline 图。

### 第 13 课：Shape Inference Interface

目标：

- 理解为什么 Toy 需要 shape inference。
- 掌握 MLIR interface 如何让 pass 面向能力而不是面向具体 op。

关键源码：

- `mlir/examples/toy/Ch4/include/toy/ShapeInferenceInterface.td`
- `mlir/examples/toy/Ch4/include/toy/Ops.td`
- `mlir/examples/toy/Ch4/mlir/ShapeInferencePass.cpp`
- `mlir/examples/toy/Ch4/mlir/Dialect.cpp`
- `mlir/test/Examples/Toy/Ch4/shape_inference.mlir`

核心知识点：

- unranked tensor 与 ranked tensor。
- operation interface。
- `inferShapes()`。
- shape inference pass 的工作列表算法。
- 为什么 lowering 前需要尽量确定 shape。

实验：

- 观察 `shape_inference.mlir` 中类型如何从未知 shape 变成已知 shape。
- 找到 `AddOp`、`MulOp`、`TransposeOp` 的 shape 推导逻辑。

产出：

- 一份 Toy shape inference 流程说明。

### 第 14 课：Inliner、CSE 与高层 IR 优化组合

目标：

- 理解 Toy 高层 IR 优化不只依赖一个 pass，而是多个 pass 配合。
- 理解函数内联、公共子表达式消除和 canonicalization 的互补关系。

关键源码：

- `mlir/examples/toy/Ch4/toyc.cpp`
- `mlir/examples/toy/Ch4/mlir/Dialect.cpp`
- `mlir/test/Examples/Toy/Ch4/codegen.toy`
- `mlir/test/Examples/Toy/Ch4/transpose_transpose.toy`

核心知识点：

- inliner 的作用。
- CSE 的基本效果。
- canonicalization 如何暴露更多 CSE 机会。
- shape inference 如何为后续 lowering 提供静态信息。
- pass 顺序如何影响最终 IR。

实验：

- 写一个重复计算的 Toy 程序，观察 CSE 是否消除重复 operation。
- 使用函数调用构造一个 inlining 可见的例子。

产出：

- 一份高层优化组合案例笔记。

### 第 15 课：Dialect Conversion 基础

目标：

- 理解 MLIR lowering 的通用机制。
- 区分 rewrite pattern、conversion pattern、partial conversion、full conversion。

关键源码：

- `mlir/docs/Tutorials/Toy/Ch-5.md`
- `mlir/examples/toy/Ch5/mlir/LowerToAffineLoops.cpp`
- `mlir/examples/toy/Ch5/include/toy/Passes.h`

核心知识点：

- `ConversionTarget`。
- legal op、illegal op、dynamically legal op。
- `ConversionPattern`。
- `OpConversionPattern`。
- `ConversionPatternRewriter`。
- `applyPartialConversion()`。
- Type conversion 的基本概念。

实验：

- 找出 Ch5 中哪些 Toy op 被标为 illegal，哪些 dialect 被标为 legal。
- 修改一个 conversion target 的合法性，预测并观察转换失败点。

产出：

- 一份 Dialect Conversion 术语表。

### 第 16 课：Toy 到 Affine/MemRef/Func 的 Partial Lowering

目标：

- 理解 Toy operation 如何被 lowering 到更低层的 MLIR dialect。
- 掌握张量到 memref、表达式到 loop nest 的基本转换思路。

关键源码：

- `mlir/examples/toy/Ch5/mlir/LowerToAffineLoops.cpp`
- `mlir/test/Examples/Toy/Ch5/affine-lowering.mlir`
- `mlir/test/Examples/Toy/Ch5/codegen.toy`

核心知识点：

- `toy.constant` 到 `memref.alloc` / `affine.store`。
- `toy.add` / `toy.mul` 到循环内 `arith` operation。
- `toy.transpose` 到访问下标交换。
- `toy.print` 到低层 print helper 语义。
- `toy.func` 到 `func.func`。
- 为什么 Ch5 是 partial lowering。

实验：

- 选择 `toy.transpose`，逐行分析 lowering 后的 affine loop。
- 写一个矩阵加法 Toy 程序，观察产生的 loop nest。

产出：

- 一份 Toy op 到低层 dialect op 的转换表。

### 第 17 课：Affine 优化与低层 IR 观察

目标：

- 理解 lowering 到 affine 的意义：不是直接生成 LLVM，而是进入可优化的中间层。
- 理解 affine loop fusion 和 scalar replacement 的效果。

关键源码：

- `mlir/examples/toy/Ch5/toyc.cpp`
- `mlir/examples/toy/Ch6/toyc.cpp`
- `mlir/test/Examples/Toy/Ch5/affine-lowering.mlir`
- `mlir/test/Examples/Toy/Ch6/affine-lowering.mlir`

核心知识点：

- Affine dialect 的循环与访存表达。
- MemRef dialect 的 buffer 语义。
- Arith dialect 的标量计算。
- Func dialect 的函数边界。
- loop fusion。
- affine scalar replacement。

实验：

- 对比开启和不开启 affine 优化时 IR 的差异。
- 找出哪些临时 memref 被优化掉，哪些仍然保留。

产出：

- 一份 affine lowering 后 IR 的层级注释。

### 第 18 课：Toy/MLIR 到 LLVM Dialect 的 Full Lowering

目标：

- 理解最终 lowering 到 LLVM Dialect 的转换路径。
- 掌握 Ch6 中 LowerToLLVM pass 的结构。

关键源码：

- `mlir/examples/toy/Ch6/mlir/LowerToLLVM.cpp`
- `mlir/examples/toy/Ch6/toyc.cpp`
- `mlir/test/Examples/Toy/Ch6/llvm-lowering.mlir`

核心知识点：

- full lowering 的目标：消除高层 dialect。
- `LLVMConversionTarget`。
- `LLVMTypeConverter`。
- `populateAffineToStdConversionPatterns()`。
- `populateSCFToControlFlowConversionPatterns()`。
- `populateArithToLLVMConversionPatterns()`。
- `populateFinalizeMemRefToLLVMConversionPatterns()`。
- `populateFuncToLLVMConversionPatterns()`。
- Toy `print` 如何 lowering 到 runtime 函数调用。

实验：

- 跟踪一个 `memref` 参数如何变成 LLVM Dialect 中的 descriptor。
- 对照 `llvm-lowering.mlir`，标注 remaining dialect。

产出：

- 一份 Ch6 lowering pipeline 说明。

### 第 19 课：LLVM IR 导出与 JIT 执行

目标：

- 理解 MLIR 如何走出 MLIR 世界，进入 LLVM IR 和执行。
- 理解 Toy JIT 的入口、runtime hook 和执行流程。

关键源码：

- `mlir/examples/toy/Ch6/toyc.cpp`
- `mlir/test/Examples/Toy/Ch6/jit.toy`
- `mlir/examples/toy/Ch7/toyc.cpp`
- `mlir/test/Examples/Toy/Ch7/jit.toy`

核心知识点：

- LLVM Dialect 到 LLVM IR translation。
- `ExecutionEngine`。
- JIT invocation。
- external function / runtime print helper。
- `MLIR_ENABLE_EXECUTION_ENGINE` 对 Ch6/Ch7 target 的影响。

实验：

- 运行 `jit.toy`，观察输出。
- 比较 `-emit=mlir-llvm` 和 `-emit=llvm` 的差异。

产出：

- 一份从 LLVM Dialect 到 JIT 执行的调用链笔记。

### 第 20 课：StructType、struct 操作与课程项目

目标：

- 理解 Ch7 如何为 Toy 增加复合类型。
- 综合前 19 节课的内容，完成一个小型扩展项目。

关键源码：

- `mlir/docs/Tutorials/Toy/Ch-7.md`
- `mlir/examples/toy/Ch7/include/toy/AST.h`
- `mlir/examples/toy/Ch7/include/toy/Ops.td`
- `mlir/examples/toy/Ch7/mlir/Dialect.cpp`
- `mlir/examples/toy/Ch7/mlir/MLIRGen.cpp`
- `mlir/examples/toy/Ch7/mlir/ToyCombine.cpp`
- `mlir/test/Examples/Toy/Ch7/struct-ast.toy`
- `mlir/test/Examples/Toy/Ch7/struct-codegen.toy`
- `mlir/test/Examples/Toy/Ch7/struct-opt.mlir`

核心知识点：

- Toy 语言中 `struct` 的语法扩展。
- 自定义 MLIR Type。
- type storage。
- type parser/printer。
- ODS 中暴露自定义 type。
- `toy.struct_constant`。
- `toy.struct_access`。
- struct verifier。
- struct 相关 canonicalization。

实验：

- 追踪一个 Toy struct 从源码到 AST，再到 MLIR type 和 operation。
- 修改 struct access 的错误用例，观察 verifier。
- 分析 `struct-opt.mlir` 中 struct 优化规则。

课程项目建议：

- 选项 A：给 Toy 增加一个简单 unary operation，例如 `neg`，并完成 AST、MLIRGen、ODS、verifier、canonicalization、测试。
- 选项 B：给 Toy 增加一个简单内建函数，例如 `sqrt` 或 `exp`，并设计 lowering 策略。
- 选项 C：扩展 shape inference，让更多 operation 能在未知 shape 输入下推导结果。
- 选项 D：为已有 lowering pattern 增加一个错误诊断或测试覆盖。

产出：

- 一份 Ch7 struct 扩展笔记。
- 一个能通过测试的小型 Toy 扩展补丁。

## 推荐学习节奏

如果每周学习 3 到 4 节课，建议这样安排：

- 第 1 周：第 1 到 4 课，建立 Toy 前端和 MLIR 基础。
- 第 2 周：第 5 到 8 课，集中学习 Dialect 和 Operation 定义。
- 第 3 周：第 9 到 12 课，进入函数语义、rewrite 和 pass pipeline。
- 第 4 周：第 13 到 16 课，学习 shape inference 和 affine lowering。
- 第 5 周：第 17 到 20 课，完成 LLVM lowering、JIT 和 Ch7 扩展。

每节课不建议只读文档。更有效的节奏是：

1. 先运行或阅读测试输入。
2. 看输出 IR。
3. 回到源码找生成或转换它的位置。
4. 做一个小实验。
5. 记录一条“源码 -> IR -> 行为”的链路。

## 课程知识点覆盖清单

前端：

- Toy 语言语法
- Lexer
- Parser
- AST
- AST dump
- 错误诊断

MLIR 基础：

- `MLIRContext`
- `ModuleOp`
- `Operation`
- `Op`
- `Value`
- `Type`
- `Attribute`
- `Location`
- `Region`
- `Block`
- Symbol

Dialect 与 ODS：

- Dialect 定义
- Operation 定义
- Operand/result/attribute
- Type constraint
- Trait
- Interface
- Builder
- Verifier
- Parser/printer
- Custom assembly format
- TableGen 生成文件

MLIRGen：

- AST 到 IR
- `OpBuilder`
- Symbol table
- Function generation
- Expression generation
- Error handling

优化与分析：

- C++ rewrite pattern
- DRR rewrite pattern
- Canonicalization
- CSE
- Inliner
- Shape inference
- Operation interface
- Pass manager
- Nested pass

Lowering：

- Dialect conversion
- Conversion target
- Legal/illegal operation
- Conversion pattern
- Partial conversion
- Full conversion
- Type conversion
- Affine lowering
- MemRef lowering
- Func lowering
- LLVM lowering

代码生成与运行：

- LLVM Dialect
- LLVM IR export
- ExecutionEngine
- JIT
- Runtime helper function

测试与调试：

- Toy test input
- MLIR test input
- FileCheck 思路
- lit 测试
- pass pipeline 调试
- IR dump 对比

扩展能力：

- 新语法
- 新 AST 节点
- 新 operation
- 新 type
- 新 verifier
- 新 rewrite pattern
- 新 lowering pattern
- 新测试用例

## 后续教程文件建议

后续可以在 `Tutorial/` 目录下按课程拆分详细教程：

- `Tutorial/01-course-navigation-and-environment.md`
- `Tutorial/02-toy-language-and-ast.md`
- `Tutorial/03-lexer-parser-ast-dump.md`
- `Tutorial/04-mlir-ir-basics.md`
- `Tutorial/05-ast-to-mlir.md`
- `Tutorial/06-dialect-and-ods.md`
- `Tutorial/07-operations-types-attributes.md`
- `Tutorial/08-verifier-builder-assembly-format.md`
- `Tutorial/09-functions-calls-symbols.md`
- `Tutorial/10-cpp-pattern-rewrite.md`
- `Tutorial/11-drr-rewrite.md`
- `Tutorial/12-pass-manager-pipeline.md`
- `Tutorial/13-shape-inference-interface.md`
- `Tutorial/14-high-level-optimization.md`
- `Tutorial/15-dialect-conversion.md`
- `Tutorial/16-lower-to-affine.md`
- `Tutorial/17-affine-optimization.md`
- `Tutorial/18-lower-to-llvm.md`
- `Tutorial/19-llvm-ir-and-jit.md`
- `Tutorial/20-struct-type-and-final-project.md`

每个详细教程建议固定包含：

- 本节目标
- 需要阅读的源码
- 需要运行或观察的测试
- 核心概念解释
- 源码调用链
- IR 变化说明
- 动手实验
- 常见问题
- 本节小结
