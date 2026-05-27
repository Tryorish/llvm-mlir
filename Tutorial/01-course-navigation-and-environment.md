# 第 1 课：课程导览与环境确认

## 本节定位

这一节课不急着深入某个具体源码实现，而是先建立一张完整地图：Toy 教程为什么这样分章、每一章增加了什么能力、源码和测试应该怎么看、构建产物应该怎么运行、后续学习时怎样观察 AST、MLIR、Affine、LLVM Dialect 和 LLVM IR。

学完本节后，你应该能够回答三个问题：

1. Toy 编译器从 `.toy` 文件到可执行代码，中间经历了哪些表示。
2. `mlir/examples/toy/Ch1` 到 `Ch7` 分别在这个过程中增加了什么。
3. 如果要观察某个阶段的结果，应该运行哪个 `toyc-ch*`，使用哪个 `-emit=` 参数，看哪个测试文件。

## 本节目标

- 建立 Toy 教程 Ch1 到 Ch7 的整体学习路线。
- 熟悉本地源码、文档、测试和构建目标的位置。
- 理解 `toyc-ch1` 到 `toyc-ch7` 的区别。
- 掌握常用观察命令：AST dump、Toy MLIR、Affine lowering、LLVM lowering、LLVM IR、JIT。
- 形成后续每节课都能复用的学习方法：先看输入，再看输出，再回源码找生成逻辑。

## 先建立一张总图

Toy 教程的主线是从一个极小的语言开始，逐步把它变成一个能生成 LLVM IR 并 JIT 执行的小编译器。

```text
Toy 源文件
  |
  v
Lexer / Parser
  |
  v
AST
  |
  v
MLIRGen
  |
  v
Toy Dialect IR
  |
  v
高层优化：canonicalization / CSE / inlining / shape inference
  |
  v
Partial lowering：Affine / Arith / MemRef / Func
  |
  v
Full lowering：LLVM Dialect
  |
  v
LLVM IR
  |
  v
JIT 执行
```

这张图里最重要的不是记住每个 pass 的名字，而是理解“表示层级”的变化：

- AST：仍然是 Toy 语言自己的语法树。
- Toy Dialect IR：已经进入 MLIR，但仍保留 Toy 语言的高层语义。
- Affine/MemRef/Func 等 Dialect：开始把 Toy 的张量计算变成循环、内存和函数。
- LLVM Dialect：接近 LLVM IR 的 MLIR 表示。
- LLVM IR/JIT：离开 MLIR，进入 LLVM 代码生成和执行链路。

## 目录地图

本课程主要围绕四类文件学习。

### 1. 教程源码

源码入口：

```text
mlir/examples/toy/
```

每一章都有自己的目录：

```text
mlir/examples/toy/Ch1
mlir/examples/toy/Ch2
mlir/examples/toy/Ch3
mlir/examples/toy/Ch4
mlir/examples/toy/Ch5
mlir/examples/toy/Ch6
mlir/examples/toy/Ch7
```

常见子目录和文件：

```text
toyc.cpp                  # 当前章节 Toy 编译器入口
include/toy/Lexer.h       # 词法分析
include/toy/Parser.h      # 语法分析
include/toy/AST.h         # AST 节点定义
parser/AST.cpp            # AST dump 实现
include/toy/Ops.td        # Toy Dialect 和 Operation 的 ODS 定义
include/toy/Dialect.h     # Toy Dialect C++ 声明
mlir/MLIRGen.cpp          # AST 到 MLIR 的生成逻辑
mlir/Dialect.cpp          # Dialect、op parser/printer、verifier 等实现
mlir/ToyCombine.cpp       # C++ rewrite / canonicalization
mlir/ToyCombine.td        # DRR 声明式 rewrite
mlir/ShapeInferencePass.cpp
mlir/LowerToAffineLoops.cpp
mlir/LowerToLLVM.cpp
```

不是每一章都有所有文件。文件数量本身就是学习线索：Ch1 只有前端和 AST，Ch2 开始有 MLIR，Ch5 开始有 affine lowering，Ch6 开始有 LLVM lowering，Ch7 开始有复合类型。

### 2. 官方 Toy 文档

文档入口：

```text
mlir/docs/Tutorials/Toy/
```

对应章节：

```text
mlir/docs/Tutorials/Toy/Ch-1.md
mlir/docs/Tutorials/Toy/Ch-2.md
mlir/docs/Tutorials/Toy/Ch-3.md
mlir/docs/Tutorials/Toy/Ch-4.md
mlir/docs/Tutorials/Toy/Ch-5.md
mlir/docs/Tutorials/Toy/Ch-6.md
mlir/docs/Tutorials/Toy/Ch-7.md
```

建议阅读顺序是：先看本课程教程，再看官方文档，再回到源码。官方文档讲概念比较系统，源码能帮助你确认每个概念到底落在哪里。

### 3. 测试输入和期望输出

测试入口：

```text
mlir/test/Examples/Toy/
```

这里的测试非常适合学习，因为每个文件通常同时包含：

- Toy 或 MLIR 输入。
- `RUN:` 命令，说明应该用哪个 `toyc-ch*` 执行。
- `CHECK:` 断言，说明期望输出 IR 长什么样。

例如：

```text
mlir/test/Examples/Toy/Ch1/ast.toy
mlir/test/Examples/Toy/Ch2/codegen.toy
mlir/test/Examples/Toy/Ch3/transpose_transpose.toy
mlir/test/Examples/Toy/Ch4/shape_inference.mlir
mlir/test/Examples/Toy/Ch5/affine-lowering.mlir
mlir/test/Examples/Toy/Ch6/llvm-lowering.mlir
mlir/test/Examples/Toy/Ch7/struct-codegen.toy
```

学习时不要只看源码。更有效的方法是先读测试，因为测试会告诉你“输入是什么，输出应该是什么”。

### 4. 本课程教程

本课程生成的教程都放在：

```text
Tutorial/
```

课程总纲：

```text
Tutorial/00-mlir-toy-course-outline.md
```

当前文件：

```text
Tutorial/01-course-navigation-and-environment.md
```

后续每节课会继续放在这个目录下。

## Ch1 到 Ch7 学习地图

| 章节 | 核心主题 | 新增能力 | 重点源码 | 重点测试 |
| --- | --- | --- | --- | --- |
| Ch1 | Toy 语言和 AST | 读取 Toy 源码，生成 AST 并 dump | `Ch1/include/toy/AST.h`、`Ch1/include/toy/Parser.h`、`Ch1/toyc.cpp` | `Ch1/ast.toy` |
| Ch2 | 生成基础 MLIR | 定义 Toy Dialect，把 AST 转成 Toy MLIR | `Ch2/include/toy/Ops.td`、`Ch2/mlir/MLIRGen.cpp`、`Ch2/mlir/Dialect.cpp` | `Ch2/codegen.toy`、`Ch2/invalid.mlir` |
| Ch3 | 高层 rewrite | 用 pattern rewrite 做 Toy 层优化 | `Ch3/mlir/ToyCombine.cpp`、`Ch3/mlir/ToyCombine.td` | `Ch3/transpose_transpose.toy`、`Ch3/trivial_reshape.toy` |
| Ch4 | Interface 和 shape inference | 使用接口做通用分析和转换 | `Ch4/include/toy/ShapeInferenceInterface.td`、`Ch4/mlir/ShapeInferencePass.cpp` | `Ch4/shape_inference.mlir` |
| Ch5 | Partial lowering | 把 Toy 部分降低到 Affine/MemRef/Func | `Ch5/mlir/LowerToAffineLoops.cpp` | `Ch5/affine-lowering.mlir` |
| Ch6 | LLVM lowering 和 codegen | 降到 LLVM Dialect，导出 LLVM IR，JIT 执行 | `Ch6/mlir/LowerToLLVM.cpp`、`Ch6/toyc.cpp` | `Ch6/llvm-lowering.mlir`、`Ch6/jit.toy` |
| Ch7 | 复合类型扩展 | 增加 `struct` 类型和相关 operation | `Ch7/include/toy/Ops.td`、`Ch7/mlir/Dialect.cpp`、`Ch7/mlir/MLIRGen.cpp` | `Ch7/struct-ast.toy`、`Ch7/struct-codegen.toy`、`Ch7/struct-opt.mlir` |

这张表后续会反复使用。遇到概念卡住时，先判断它属于哪一层：

- 前端问题：通常看 Ch1。
- Dialect/op 定义问题：通常看 Ch2。
- rewrite 问题：通常看 Ch3。
- interface/pass 问题：通常看 Ch4。
- affine lowering 问题：通常看 Ch5。
- LLVM/codegen 问题：通常看 Ch6。
- 自定义 type/语言扩展问题：通常看 Ch7。

## `toyc-ch*` 是什么

`toyc-ch1` 到 `toyc-ch7` 是 Toy 教程每一章构建出的编译器可执行文件。

它们不是同一个程序的七种运行模式，而是七个逐章演进的程序：

- `toyc-ch1`：只支持解析 Toy 并 dump AST。
- `toyc-ch2`：支持 AST 和基础 Toy MLIR。
- `toyc-ch3`：增加高层优化。
- `toyc-ch4`：增加 interface、shape inference、inlining 相关能力。
- `toyc-ch5`：增加到 affine/memref/func 的 lowering。
- `toyc-ch6`：增加到 LLVM Dialect、LLVM IR 和 JIT 的路径。
- `toyc-ch7`：在 Ch6 基础上增加 struct 复合类型。

顶层 CMake 通过 `mlir/examples/toy/CMakeLists.txt` 把这些章节加入构建：

```text
add_subdirectory(Ch1)
add_subdirectory(Ch2)
add_subdirectory(Ch3)
add_subdirectory(Ch4)
add_subdirectory(Ch5)
add_subdirectory(Ch6)
add_subdirectory(Ch7)
```

Ch6 和 Ch7 依赖 JIT 支持。如果构建 LLVM/MLIR 时没有启用 `MLIR_ENABLE_EXECUTION_ENGINE`，这两章的 target 可能不会生成。

## 输出模式速查

不同章节支持的 `-emit=` 选项不同。完整链路主要看 Ch6 或 Ch7。

| 输出模式 | 典型命令 | 含义 | 适合观察什么 |
| --- | --- | --- | --- |
| `-emit=ast` | `toyc-ch1 input.toy -emit=ast` | dump Toy AST | 前端解析结果 |
| `-emit=mlir` | `toyc-ch2 input.toy -emit=mlir` | dump Toy Dialect MLIR | AST 到 MLIR 的转换 |
| `-emit=mlir-affine` | `toyc-ch5 input.toy -emit=mlir-affine` | lowering 到 Affine/MemRef/Func 后 dump | partial lowering |
| `-emit=mlir-llvm` | `toyc-ch6 input.toy -emit=mlir-llvm` | lowering 到 LLVM Dialect 后 dump | full lowering |
| `-emit=llvm` | `toyc-ch6 input.toy -emit=llvm` | 导出 LLVM IR | MLIR 到 LLVM IR |
| `-emit=jit` | `toyc-ch6 input.toy -emit=jit` | JIT 编译并运行 | 端到端执行 |

Ch2 到 Ch7 还支持 `-x` 指定输入类型：

```text
-x toy     # 按 Toy 源文件读取
-x mlir    # 按 MLIR 文件读取
```

如果输入文件后缀是 `.mlir`，较新的章节也会按 MLIR 输入处理。后续课程会在需要时明确说明。

## 构建确认

本课程不假设构建目录一定在源码树内。下面用 `<build-dir>` 表示你的 LLVM/MLIR 构建目录。

如果使用 Ninja，可以构建单章：

```bash
ninja -C <build-dir> toyc-ch1
ninja -C <build-dir> toyc-ch2
ninja -C <build-dir> toyc-ch3
ninja -C <build-dir> toyc-ch4
ninja -C <build-dir> toyc-ch5
ninja -C <build-dir> toyc-ch6
ninja -C <build-dir> toyc-ch7
```

也可以构建 Toy 总 target：

```bash
ninja -C <build-dir> Toy
```

构建成功后，通常可执行文件在：

```text
<build-dir>/bin/toyc-ch1
<build-dir>/bin/toyc-ch2
<build-dir>/bin/toyc-ch3
<build-dir>/bin/toyc-ch4
<build-dir>/bin/toyc-ch5
<build-dir>/bin/toyc-ch6
<build-dir>/bin/toyc-ch7
```

如果 `toyc-ch6` 或 `toyc-ch7` 不存在，优先检查构建配置是否启用了：

```text
MLIR_ENABLE_EXECUTION_ENGINE
```

Ch6/Ch7 的 `CMakeLists.txt` 中有条件判断：没有 execution engine 时会直接跳过对应章节。

## 第一组观察命令

下面的命令用于建立直觉。它们不要求你理解每一行输出，只要求你观察输出层级在变低。

把 `<build-dir>` 换成自己的构建目录。

### 1. 观察 Ch1 AST

```bash
<build-dir>/bin/toyc-ch1 mlir/test/Examples/Toy/Ch1/ast.toy -emit=ast
```

你应该重点观察：

- 顶层是不是一个 Module。
- Module 里有哪些函数。
- 函数参数、变量声明、字面量、return、print 如何出现在 AST dump 中。

### 2. 观察 Ch2 Toy MLIR

```bash
<build-dir>/bin/toyc-ch2 mlir/test/Examples/Toy/Ch2/codegen.toy -emit=mlir
```

你应该重点观察：

- 输出已经不是 AST，而是 MLIR 文本。
- Toy 自定义 operation 以 `toy.` 开头。
- 典型 operation 包括 `toy.func`、`toy.constant`、`toy.reshape`、`toy.transpose`、`toy.mul`、`toy.generic_call`、`toy.print`、`toy.return`。

### 3. 观察 Ch5 Affine lowering

```bash
<build-dir>/bin/toyc-ch5 mlir/test/Examples/Toy/Ch5/codegen.toy -emit=mlir-affine
```

你应该重点观察：

- 一部分 Toy operation 消失了。
- 出现 `func.func`、`memref.alloc`、`affine.for`、`affine.load`、`affine.store`、`arith.*`。
- `toy.print` 可能仍然存在，所以这是 partial lowering。

### 4. 观察 Ch6 LLVM Dialect

```bash
<build-dir>/bin/toyc-ch6 mlir/test/Examples/Toy/Ch6/codegen.toy -emit=mlir-llvm
```

你应该重点观察：

- 高层 Toy/Affine/MemRef/Func 语义进一步消失。
- 输出接近 LLVM IR，但仍然是 MLIR 的 LLVM Dialect 文本。
- 会看到 `llvm.func`、`llvm.call`、`llvm.return` 等 operation。

### 5. 观察 Ch6 LLVM IR

```bash
<build-dir>/bin/toyc-ch6 mlir/test/Examples/Toy/Ch6/codegen.toy -emit=llvm
```

你应该重点观察：

- 这一步输出已经是 LLVM IR，不再是 MLIR。
- 语法会接近 `.ll` 文件中的形式。
- 这是从 MLIR 世界导出到 LLVM 世界的关键边界。

### 6. 观察 JIT 执行

```bash
<build-dir>/bin/toyc-ch6 mlir/test/Examples/Toy/Ch6/jit.toy -emit=jit
```

你应该重点观察：

- 这一步不是打印 IR，而是运行 Toy 程序。
- 如果命令失败，先检查 Ch6 是否构建成功，以及是否启用了 execution engine。

## 测试系统入口

Toy 测试位于：

```text
mlir/test/Examples/Toy/
```

如果构建目录中有 `llvm-lit`，可以运行单章测试：

```bash
<build-dir>/bin/llvm-lit -sv mlir/test/Examples/Toy/Ch1
<build-dir>/bin/llvm-lit -sv mlir/test/Examples/Toy/Ch2
<build-dir>/bin/llvm-lit -sv mlir/test/Examples/Toy/Ch3
<build-dir>/bin/llvm-lit -sv mlir/test/Examples/Toy/Ch4
<build-dir>/bin/llvm-lit -sv mlir/test/Examples/Toy/Ch5
<build-dir>/bin/llvm-lit -sv mlir/test/Examples/Toy/Ch6
<build-dir>/bin/llvm-lit -sv mlir/test/Examples/Toy/Ch7
```

一个 Toy 测试通常长这样：

```text
# RUN: toyc-ch6 %s -emit=mlir 2>&1 | FileCheck %s

def main() {
  ...
}

# CHECK-LABEL: toy.func @main()
# CHECK:       ...
```

阅读测试时按这个顺序：

1. 先看 `RUN:`，确定使用哪个编译器和哪个输出模式。
2. 再看输入程序，理解测试场景。
3. 最后看 `CHECK:`，理解期望 IR。

后续课程会反复使用这个方法。它比直接读源码更稳，因为测试先给出了行为边界。

## 本节重点源码入口

这一节建议只做“定位阅读”，不要深挖每个函数。

### `mlir/examples/toy/CMakeLists.txt`

看点：

- `Toy` 总 target。
- `add_toy_chapter` 宏。
- Ch1 到 Ch7 的 `add_subdirectory`。

你要理解：Toy 教程是按章节构建多个可执行文件，而不是只构建一个最终程序。

### `mlir/examples/toy/Ch1/toyc.cpp`

看点：

- `inputFilename` 命令行参数。
- `-emit=ast`。
- `parseInputFile()`。
- `LexerBuffer`、`Parser`、`parseModule()`。
- `dump(*moduleAST)`。

你要理解：Ch1 的编译器只做到 AST，不进入 MLIR。

### `mlir/examples/toy/Ch6/toyc.cpp`

看点：

- `-emit=ast`。
- `-emit=mlir`。
- `-emit=mlir-affine`。
- `-emit=mlir-llvm`。
- `-emit=llvm`。
- `-emit=jit`。
- `PassManager`。
- lowering 到 affine 和 LLVM 的判断逻辑。

你要理解：Ch6 是一条相对完整的端到端链路，后续很多课程都可以用它观察最终效果。

## 推荐学习方法

后续每节课都建议按这五步学习：

1. 先读测试输入。
2. 运行对应 `toyc-ch*` 或阅读测试中的 `CHECK:`。
3. 找到生成这段 IR 的源码。
4. 改一个很小的例子。
5. 记录“输入 -> 中间表示 -> 关键源码”的对应关系。

不要一开始就试图把所有 C++ 模板、TableGen 和 MLIR API 都弄懂。Toy 教程最有效的学习方式是沿着一个具体程序走完整条链路。

## 本节练习

### 练习 1：画出你的 Toy 学习地图

用自己的话写出下面这条链路中每一层的作用：

```text
Toy source -> AST -> Toy MLIR -> Affine/MemRef/Func -> LLVM Dialect -> LLVM IR -> JIT
```

要求：

- 每一层写 1 到 2 句话。
- 不要求准确到 API 名称。
- 重点写“这一层比上一层更适合做什么”。

### 练习 2：定位每章新增文件

在源码根目录运行：

```bash
find mlir/examples/toy/Ch1 -maxdepth 3 -type f | sort
find mlir/examples/toy/Ch2 -maxdepth 3 -type f | sort
find mlir/examples/toy/Ch6 -maxdepth 3 -type f | sort
find mlir/examples/toy/Ch7 -maxdepth 3 -type f | sort
```

观察：

- Ch1 为什么没有 `mlir/MLIRGen.cpp`。
- Ch2 为什么开始有 `Ops.td` 和 `Dialect.cpp`。
- Ch6 为什么多了 `LowerToLLVM.cpp`。
- Ch7 为什么 `MLIRGen.cpp` 和 `Dialect.cpp` 明显更复杂。

### 练习 3：从测试反推功能

阅读这些文件的 `RUN:` 行和文件名：

```text
mlir/test/Examples/Toy/Ch1/ast.toy
mlir/test/Examples/Toy/Ch2/codegen.toy
mlir/test/Examples/Toy/Ch3/transpose_transpose.toy
mlir/test/Examples/Toy/Ch4/shape_inference.mlir
mlir/test/Examples/Toy/Ch5/affine-lowering.mlir
mlir/test/Examples/Toy/Ch6/llvm-lowering.mlir
mlir/test/Examples/Toy/Ch7/struct-codegen.toy
```

为每个文件写一句话：

- 这个测试验证了什么能力。
- 它大概属于编译链路的哪一层。

### 练习 4：运行一条最短链路

如果本地已经构建成功，运行：

```bash
<build-dir>/bin/toyc-ch1 mlir/test/Examples/Toy/Ch1/ast.toy -emit=ast
<build-dir>/bin/toyc-ch2 mlir/test/Examples/Toy/Ch2/codegen.toy -emit=mlir
<build-dir>/bin/toyc-ch6 mlir/test/Examples/Toy/Ch6/codegen.toy -emit=mlir-affine
<build-dir>/bin/toyc-ch6 mlir/test/Examples/Toy/Ch6/codegen.toy -emit=mlir-llvm
```

不用完全读懂输出，只需要记录：

- 哪些输出还带有 `toy.`。
- 哪些输出开始出现 `affine.`、`memref.`、`arith.`。
- 哪些输出开始出现 `llvm.`。

## 常见问题

### 找不到 `toyc-ch*`

常见原因：

- 还没有构建对应 target。
- 构建目录不是你以为的位置。
- Ch6/Ch7 因为没有启用 execution engine 被跳过。

先尝试：

```bash
ninja -C <build-dir> toyc-ch1
find <build-dir> -name 'toyc-ch1' -type f
```

### `toyc-ch6` 或 `toyc-ch7` 不存在

优先检查构建配置是否启用了：

```text
MLIR_ENABLE_EXECUTION_ENGINE
```

Ch6 和 Ch7 的 `CMakeLists.txt` 里有条件判断，没有 JIT 支持时会直接 `return()`。

### 对 `.mlir` 文件使用 `-emit=ast` 失败

这是正常的。AST 是 Toy 源码前端的结果，而 `.mlir` 文件已经绕过了 Toy parser。

如果要读取 `.mlir` 文件，使用支持 MLIR 输入的章节，并观察 MLIR 相关输出。

### 输出太长，看不懂

第一次看只抓三类信息：

- operation 名字，例如 `toy.transpose`、`affine.for`、`llvm.call`。
- type，例如 `tensor<2x3xf64>`、`memref<2x3xf64>`。
- 层级结构，例如函数体、循环体、block。

细节会在后续课程逐步展开。

## 本节小结

这一节最重要的结论是：Toy 教程不是七个孤立例子，而是一条逐步降低抽象层级的编译链路。

你现在需要记住的核心路径是：

```text
Ch1：Toy 源码 -> AST
Ch2：AST -> Toy Dialect MLIR
Ch3：Toy 层 rewrite
Ch4：interface + shape inference + inlining
Ch5：Toy -> Affine/MemRef/Func
Ch6：Affine/MemRef/Func -> LLVM Dialect -> LLVM IR/JIT
Ch7：增加 struct 类型，验证扩展一个语言特性的完整流程
```

下一节课将进入 Ch1，详细学习 Toy 语言本身和 AST 结构。重点会从“地图”转向“第一个具体表示”：Toy AST。

## 学习记录模板

建议在每节课后保留一段记录，格式如下：

```text
本节主题：
我读过的源码：
我观察过的测试：
我运行过的命令：
我确认理解的链路：
我还不理解的问题：
下一步要验证的小实验：
```

本节可以填写为：

```text
本节主题：课程导览与环境确认
我读过的源码：mlir/examples/toy/CMakeLists.txt、Ch1/toyc.cpp、Ch6/toyc.cpp
我观察过的测试：Ch1/ast.toy、Ch2/codegen.toy、Ch6/codegen.toy
我运行过的命令：
我确认理解的链路：Toy source -> AST -> Toy MLIR -> Affine/MemRef/Func -> LLVM Dialect -> LLVM IR/JIT
我还不理解的问题：
下一步要验证的小实验：运行 Ch1 AST dump，并手动标注 AST 层级
```
