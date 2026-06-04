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

## 建议新增 Pass

```text
toy-gpu-insert-device-memory
```

建议实现文件：

```text
mlir/examples/toy/Ch7/mlir/GPUDeviceMemory.cpp
mlir/examples/toy/Ch7/include/toy/Passes.h
mlir/examples/toy/Ch7/toyc.cpp
mlir/examples/toy/Ch7/CMakeLists.txt
```

## Pass A 输出

Pass A 是阶段 5/6 的 kernel 生成 pass。它只负责：

```text
toy.matmul -> host memref alloc + gpu.launch
```

暂时允许 `gpu.launch` body 内引用 host memref。这个 IR 不是最终可运行形态，只作为中间分层。

## Pass B 输出

Pass B 插入：

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
1. 先只支持当前 matmul 场景，不泛化所有 gpu.launch。
2. 在 launch 前识别 lhs/rhs/out host memref。
3. 插入 device alloc 和 host -> device memcpy。
4. 维护 async token 链。
5. 替换 gpu.launch region 内部使用的 memref。
6. 如果后续使用 gpu.launch_func，需要同步更新 launch_func args。
7. 在 launch 后插入 device -> host memcpy。
8. 插入 device dealloc 和最终 wait。
```

## 风险点

```text
需要替换 gpu.launch region 内部使用的 memref。
需要同步更新 gpu.launch_func 的 args。
async token 链必须保持顺序。
不能留下 builtin.unrealized_conversion_cast。
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
