# Stage 7: Split Device Memory Management

目标：把 device buffer 管理从 matmul kernel 生成中拆出来。

当前 `MatMulOpLowering` 同时负责：

```text
gpu.alloc
gpu.memcpy host -> device
gpu.launch
gpu.memcpy device -> host
gpu.dealloc
gpu.wait
```

更清楚的分层是：

```text
Pass A:
  只生成使用 host memref 的 gpu.launch

Pass B:
  为 gpu.launch 插入 device alloc/copy/dealloc
  并把 launch 内部引用替换成 device memref
```

## 已新增 Pass

```text
toy-gpu-insert-device-memory
```

实现文件：

```text
mlir/examples/toy/Ch7/mlir/GPUDeviceMemory.cpp
mlir/examples/toy/Ch7/include/toy/Passes.h
mlir/examples/toy/Ch7/toyc.cpp
mlir/examples/toy/Ch7/CMakeLists.txt
```

新增 emit action：

```text
-emit=mlir-gpu-device-memory-matmul
```

pipeline 形状：

```text
Toy/MLIR input for -emit=mlir-gpu-device-memory-matmul
  -> inliner
  -> canonicalizer
  -> toy shape inference
  -> canonicalizer
  -> CSE
  -> toy-matmul-to-scf
  -> toy-matmul-tile-loops
  -> toy-matmul-reorder-tiled-loops
  -> toy-matmul-map-to-gpu
  -> toy-matmul-promote-workgroup-memory
  -> toy-gpu-insert-device-memory
  -> func-level canonicalizer
  -> func-level CSE
  -> dump MLIR
```

完整 GPU pipeline 也已统一复用 Stage 2-7：

```text
-emit=mlir-gpu
-emit=mlir-gpu-outlined
-emit=mlir-gpu-nvvm
-emit=mlir-gpu-binary
-emit=mlir-gpu-host
-emit=llvm-gpu
-emit=gpu-jit
```

这些 action 不再先走旧的 monolithic `createLowerToGPUPass()` matmul lowering，
而是先经过：

```text
toy-matmul-to-scf
  -> toy-matmul-tile-loops
  -> toy-matmul-reorder-tiled-loops
  -> toy-matmul-map-to-gpu
  -> toy-matmul-promote-workgroup-memory
  -> toy-gpu-insert-device-memory
```

之后再继续进入标准 GPU 后端阶段：

```text
gpu.launch outlining
  -> GPU to NVVM
  -> gpu.binary
  -> GPU host runtime lowering
  -> LLVM IR / JIT
```

## Pass A 输出

Pass A 是阶段 5/6 的 kernel 生成 pass。当前已经拆成：

```text
Stage 5:
  toy-matmul-map-to-gpu

Stage 6:
  toy-matmul-promote-workgroup-memory
```

它们只负责：

```text
toy.matmul -> host memref alloc + gpu.launch
```

暂时允许 `gpu.launch` body 内引用 host memref。这个 IR 不是最终可运行形态，
只作为中间分层。

## Pass B 输出

Pass B 是本阶段新增的 `toy-gpu-insert-device-memory`。它插入：

```text
gpu.wait async
gpu.alloc lhs/rhs/out
gpu.memcpy lhs/rhs host -> device
gpu.launch async
gpu.memcpy out device -> host
gpu.dealloc lhs/rhs/out
gpu.wait
```

同时要把 launch body 内部引用从 host memref 替换成对应 device memref。

## 实施步骤

```text
1. 已先只支持当前 matmul 场景，不泛化所有 gpu.launch。
2. 已在 launch body 内识别被 memref.load/store 捕获的 host memref。
3. 已在 launch 前插入 device alloc。
4. 已对只读 buffer 插入 host -> device memcpy。
5. 已维护 async token 链。
6. 已克隆 gpu.launch，并用 device memref 替换 launch region 内部引用。
7. 已在 launch 后对写 buffer 插入 device -> host memcpy。
8. 已插入 device dealloc 和最终 wait。
```

## 风险点

```text
需要替换 gpu.launch region 内部使用的 memref。
需要同步更新 gpu.launch_func 的 args。
async token 链必须保持顺序。
不能留下 builtin.unrealized_conversion_cast。
```

当前实现范围：

```text
只匹配：
  - 无 async token / async dependency 的 gpu.launch。
  - 无 cluster size 的 gpu.launch。
  - launch body 内通过 memref.load/store 捕获外层 host memref。
  - 静态 shape memref。

不处理：
  - 已经 outlined 的 gpu.launch_func。
  - 动态 shape memref。
  - 通用 GPU 程序的 alias/escape 分析。
```

替换策略：

```text
1. 收集 launch body 内 load/store 使用的外层 memref。
2. 为每个 host memref 分配同 shape 的 device memref。
3. 读 buffer 在 launch 前 copy host -> device。
4. 克隆 launch body，IRMapping 中把 host memref 映射到 device memref。
5. 写 buffer 在 launch 后 copy device -> host。
6. 最后按 token 链 dealloc device memref 并 gpu.wait。
```

## 验证

```bash
./build/bin/toyc-ch7 mlir/test/Examples/Toy/Ch7/matmul-codegen.toy -emit=mlir-gpu-host
./build/bin/toyc-ch7 mlir/test/Examples/Toy/Ch7/matmul-codegen.toy -emit=llvm-gpu
```

不能出现：

```text
builtin.unrealized_conversion_cast
```

建议新增测试：

```text
mlir/test/Examples/Toy/Ch7/matmul64-gpu-device-memory.mlir
```

## 当前执行状态

已完成：

```text
- 新增 GPUDeviceMemory.cpp。
- 新增 createGPUInsertDeviceMemoryPass() 声明。
- 新增 -emit=mlir-gpu-device-memory-matmul。
- CMake 已加入新源文件。
- 新增 matmul64-gpu-device-memory.mlir。
```

未在本地执行：

```text
- ninja -C build toyc-ch7
- toyc-ch7 -emit=mlir-gpu-device-memory-matmul
- llvm-lit matmul64-gpu-device-memory.mlir
```

未执行原因：

```text
延续前置要求：本地不要编译。
```
