# 第 1 课练习答案：课程导览与环境确认

## 填写说明

这里先写你的答案和实验记录。写完后告诉我检查这个文件，我会按第一课教程内容帮你验证是否正确，并指出需要补充或修正的地方。

## 练习 1：画出你的 Toy 学习地图

```text
Toy source -> toy源程序

AST -> 将toy源程序转换成抽象语法树

Toy MLIR -> 将抽象语法树降级成toy方言下的操作

Affine/MemRef/Func -> 进一步降级到Affine/MemRef/Func方言下的操作

LLVM Dialect -> 降级到LLVM方言

LLVM IR -> 降级到LLVM 中间表示

JIT -> 可以运行
```

## 练习 2：定位每章新增文件

### Ch1

```text
    mlir/examples/toy/Ch1/CMakeLists.txt
    mlir/examples/toy/Ch1/include/toy/AST.h
    mlir/examples/toy/Ch1/include/toy/Lexer.h
    mlir/examples/toy/Ch1/include/toy/Parser.h
    mlir/examples/toy/Ch1/parser/AST.cpp
    mlir/examples/toy/Ch1/toyc.cpp
```

### Ch2

```text
    mlir/examples/toy/Ch2/CMakeLists.txt
    mlir/examples/toy/Ch2/include/CMakeLists.txt
    mlir/examples/toy/Ch2/include/toy/AST.h
    mlir/examples/toy/Ch2/include/toy/CMakeLists.txt
    mlir/examples/toy/Ch2/include/toy/Dialect.h
    mlir/examples/toy/Ch2/include/toy/Lexer.h
    mlir/examples/toy/Ch2/include/toy/MLIRGen.h
    mlir/examples/toy/Ch2/include/toy/Ops.td
    mlir/examples/toy/Ch2/include/toy/Parser.h
    mlir/examples/toy/Ch2/mlir/Dialect.cpp
    mlir/examples/toy/Ch2/mlir/MLIRGen.cpp
    mlir/examples/toy/Ch2/parser/AST.cpp
    mlir/examples/toy/Ch2/toyc.cpp
```

### Ch6

```text
    mlir/examples/toy/Ch6/CMakeLists.txt
    mlir/examples/toy/Ch6/include/CMakeLists.txt
    mlir/examples/toy/Ch6/include/toy/AST.h
    mlir/examples/toy/Ch6/include/toy/CMakeLists.txt
    mlir/examples/toy/Ch6/include/toy/Dialect.h
    mlir/examples/toy/Ch6/include/toy/Lexer.h
    mlir/examples/toy/Ch6/include/toy/MLIRGen.h
    mlir/examples/toy/Ch6/include/toy/Ops.td
    mlir/examples/toy/Ch6/include/toy/Parser.h
    mlir/examples/toy/Ch6/include/toy/Passes.h
    mlir/examples/toy/Ch6/include/toy/ShapeInferenceInterface.h
    mlir/examples/toy/Ch6/include/toy/ShapeInferenceInterface.td
    mlir/examples/toy/Ch6/mlir/Dialect.cpp
    mlir/examples/toy/Ch6/mlir/LowerToAffineLoops.cpp
    mlir/examples/toy/Ch6/mlir/LowerToLLVM.cpp
    mlir/examples/toy/Ch6/mlir/MLIRGen.cpp
    mlir/examples/toy/Ch6/mlir/ShapeInferencePass.cpp
    mlir/examples/toy/Ch6/mlir/ToyCombine.cpp
    mlir/examples/toy/Ch6/mlir/ToyCombine.td
    mlir/examples/toy/Ch6/parser/AST.cpp
    mlir/examples/toy/Ch6/toyc.cpp
```

### Ch7

```text
    mlir/examples/toy/Ch7/CMakeLists.txt
    mlir/examples/toy/Ch7/include/CMakeLists.txt
    mlir/examples/toy/Ch7/include/toy/AST.h
    mlir/examples/toy/Ch7/include/toy/CMakeLists.txt
    mlir/examples/toy/Ch7/include/toy/Dialect.h
    mlir/examples/toy/Ch7/include/toy/Lexer.h
    mlir/examples/toy/Ch7/include/toy/MLIRGen.h
    mlir/examples/toy/Ch7/include/toy/Ops.td
    mlir/examples/toy/Ch7/include/toy/Parser.h
    mlir/examples/toy/Ch7/include/toy/Passes.h
    mlir/examples/toy/Ch7/include/toy/ShapeInferenceInterface.h
    mlir/examples/toy/Ch7/include/toy/ShapeInferenceInterface.td
    mlir/examples/toy/Ch7/mlir/Dialect.cpp
    mlir/examples/toy/Ch7/mlir/LowerToAffineLoops.cpp
    mlir/examples/toy/Ch7/mlir/LowerToLLVM.cpp
    mlir/examples/toy/Ch7/mlir/MLIRGen.cpp
    mlir/examples/toy/Ch7/mlir/ShapeInferencePass.cpp
    mlir/examples/toy/Ch7/mlir/ToyCombine.cpp
    mlir/examples/toy/Ch7/mlir/ToyCombine.td
    mlir/examples/toy/Ch7/parser/AST.cpp
    mlir/examples/toy/Ch7/toyc.cpp
```

### 我的观察

- Ch1 为什么没有 `mlir/MLIRGen.cpp`：Ch1只是toy source到ast这层逻辑，MLIRGen.cpp是AST到MLIR的生成逻辑
- Ch2 为什么开始有 `Ops.td` 和 `Dialect.cpp`：Ch2涉及到了从ast到toy mlir这层，Ops.td是Toy Dialect和Operation的ODS定义，Dialect.cpp是dialect,op parser/printer,verifier的实现
- Ch6 为什么多了 `LowerToLLVM.cpp`：LowerToLLVM.cpp是最终降级到LLVM IR的实现
- Ch7 为什么 `MLIRGen.cpp` 和 `Dialect.cpp` 明显更复杂: Ch7增加了struct类型的实现：

## 练习 3：从测试反推功能

| 文件 | 验证的能力 | 所属编译链路层级 |
| --- | --- | --- |
| `mlir/test/Examples/Toy/Ch1/ast.toy` | 是否生成ast | toy source -> ast |
| `mlir/test/Examples/Toy/Ch2/codegen.toy` | 是否生成tir | ast -> tir |
| `mlir/test/Examples/Toy/Ch3/transpose_transpose.toy` | 是否生成tir | ast -> tir |
| `mlir/test/Examples/Toy/Ch4/shape_inference.mlir` | 检测内联和shape推导 | tir |
| `mlir/test/Examples/Toy/Ch5/affine-lowering.mlir` | 是否降级到affine并进行优化 | affine dialect |
| `mlir/test/Examples/Toy/Ch6/llvm-lowering.mlir` | 是否可运行 | jit |
| `mlir/test/Examples/Toy/Ch7/struct-codegen.toy` | 验证结构体 | tir |

## 练习 4：运行一条最短链路

### 我运行过的命令

```bash
    ./toyc-ch1 ../../mlir/test/Examples/Toy/Ch1/ast.toy -emit=ast
    ./toyc-ch2 ../../mlir/test/Examples/Toy/Ch2/codegen.toy -emit=mlir
    ./toyc-ch5 ../../mlir/test/Examples/Toy/Ch5/codegen.toy -emit=mlir-affine
    ./toyc-ch6 ../../mlir/test/Examples/Toy/Ch6/codegen.toy -emit=mlir-llvm
    ./toyc-ch6 ../../mlir/test/Examples/Toy/Ch6/codegen.toy -emit=llvm
    ./toyc-ch6 ../../mlir/test/Examples/Toy/Ch6/jit.toy -emit=jit
```

### 我的观察

- 哪些输出还带有 `toy.`：-emit=mlir
- 哪些输出开始出现 `affine.`、`memref.`、`arith.`：-emit=mlir-affine
- 哪些输出开始出现 `llvm.`：-emit=mlir-llvm

## Codex 校验与修正建议

整体结论：第一课主线已经掌握，练习 1 和练习 2 基本正确。需要重点修正的是术语使用，以及练习 3 中几个测试文件验证能力的判断。

### 术语修正

- `Toy MLIR -> 将抽象语法树降级成toy方言下的操作`

  建议改成：

  ```text
  Toy MLIR -> 将 AST 生成为 Toy Dialect IR，也就是 MLIR 中的 toy.* 操作
  ```

  原因：AST 到 Toy MLIR 通常叫 `MLIRGen` 或 IR 生成；“lowering/降级”更适合 Ch5/Ch6 中从高层 dialect 到低层 dialect 的转换。

- `LowerToLLVM.cpp是最终降级到LLVM IR的实现`

  建议改成：

  ```text
  LowerToLLVM.cpp 负责把 MLIR 中的较高层 dialect lowering 到 LLVM Dialect；真正导出 LLVM IR 是后续 LLVM IR translation 流程完成的。
  ```

  原因：LLVM Dialect 仍然是 MLIR 里的 dialect，不等同于最终 LLVM IR 文本。

- `Ch7 为什么 MLIRGen.cpp 和 Dialect.cpp 明显更复杂`

  可以补充为：

  ```text
  Ch7 增加了 struct 复合类型，所以 MLIRGen.cpp 要处理 struct 定义、struct 字面量、字段访问；Dialect.cpp 要处理自定义 StructType、type parser/printer、toy.struct_constant、toy.struct_access 和相关 verifier。
  ```

### 练习 3 修订版

| 文件 | 更准确的验证能力 | 所属编译链路层级 |
| --- | --- | --- |
| `mlir/test/Examples/Toy/Ch1/ast.toy` | 验证 Toy 源码能被解析并生成 AST dump | Toy source -> AST |
| `mlir/test/Examples/Toy/Ch2/codegen.toy` | 验证 AST 能生成 Toy Dialect MLIR | AST -> Toy MLIR |
| `mlir/test/Examples/Toy/Ch3/transpose_transpose.toy` | 验证 `-opt` 下 canonicalization 能消除 `transpose(transpose(x))` | Toy MLIR 高层优化 |
| `mlir/test/Examples/Toy/Ch4/shape_inference.mlir` | 验证 Toy MLIR 上的 inlining 和 shape inference | Toy MLIR 高层分析与优化 |
| `mlir/test/Examples/Toy/Ch5/affine-lowering.mlir` | 验证 Toy 部分 lowering 到 affine/memref/func/arith，并对比 `-opt` 优化结果 | Partial lowering |
| `mlir/test/Examples/Toy/Ch6/llvm-lowering.mlir` | 验证 lowering 后可以导出 LLVM IR，不是 JIT 测试 | LLVM lowering / LLVM IR export |
| `mlir/test/Examples/Toy/Ch7/struct-codegen.toy` | 验证 struct 语法生成 Toy MLIR，包括 `StructType`、`toy.struct_constant`、`toy.struct_access` | Toy 语言扩展 / Toy MLIR |

### 关于 execution engine

你的 CMake 命令没有显式传入：

```bash
-DMLIR_ENABLE_EXECUTION_ENGINE=ON
```

但 Ch6 和 Ch7 依然能编译成功，是因为 MLIR 会根据 native target 自动设置 `MLIR_ENABLE_EXECUTION_ENGINE`。

在 `mlir/CMakeLists.txt` 中有如下逻辑：

```cmake
if(${LLVM_NATIVE_ARCH} IN_LIST LLVM_TARGETS_TO_BUILD)
  set(MLIR_ENABLE_EXECUTION_ENGINE 1)
else()
  set(MLIR_ENABLE_EXECUTION_ENGINE 0)
endif()
```

你的配置里包含：

```bash
-DLLVM_TARGETS_TO_BUILD="AArch64"
```

如果当前机器的 native arch 正好是 `AArch64`，那么 `LLVM_NATIVE_ARCH` 在 `LLVM_TARGETS_TO_BUILD` 中，`MLIR_ENABLE_EXECUTION_ENGINE` 就会自动变成 `1`。因此 Ch6/Ch7 没有被 `CMakeLists.txt` 里的条件跳过。

## 我的问题

- 这是我编译时的命令，为什么没有启用execution engine，ch6，ch7依然编译成功
```
mkdir build && cd build

cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_BUILD_EXAMPLES=ON \
    -DLLVM_TARGETS_TO_BUILD="AArch64" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_ENABLE_LLD=OFF \
    -DLLVM_CCACHE_BUILD=ON
    
cmake --build . --target check-mlir
```
