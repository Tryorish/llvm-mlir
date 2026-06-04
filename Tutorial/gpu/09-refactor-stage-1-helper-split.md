# Stage 1: MatMulOpLowering Helper Split

目标：保持 IR 完全不变，只把 `MatMulOpLowering` 拆成 helper。

执行文件：

```text
mlir/examples/toy/Ch7/mlir/LowerToGPU.cpp
```

这一阶段只允许做局部函数级拆分：

```text
允许：
  - 新增局部 struct 承载已有 Value。
  - 把原来 matchAndRewrite 内部连续代码块搬到 static helper。
  - 保持 helper 仍在 LowerToGPU.cpp 匿名 namespace 内。

不允许：
  - 新增 pass。
  - 改变 pipeline。
  - 改变 block/thread mapping。
  - 改变 shared memory tile 形状。
  - 改变 async token 依赖顺序。
  - 改变最终 IR 行为。
```

## 数据结构

已经拆出：

```cpp
struct MatMulConfig {
  Value c0;
  Value c1;
  Value mVal;
  Value nVal;
  Value kVal;
  Value blockX;
  Value blockY;
  Value gridX;
  Value gridY;
  Value zero;
};

struct DeviceBuffers {
  Value lhs;
  Value rhs;
  Value out;
  Value token;
};
```

`MatMulConfig` 只保存已有常量 Value，不新增语义。

`DeviceBuffers` 只保存 device memref 和当前 async token。

## Helper 划分

```cpp
static MatMulConfig createMatMulConfig(...);
```

负责从 `lhs/rhs` memref 静态 shape 读取 `m/k/n`，并创建原来直接写在
`MatMulOpLowering` 中的常量：

```text
c0 / c1
mVal / nVal / kVal
blockX / blockY
gridX / gridY
zero
```

注意：

```text
blockSize 仍固定为 16。
gridX = ceil(n / 16)
gridY = ceil(m / 16)
zero 仍是 f64 0.0。
```

```cpp
static DeviceBuffers createDeviceBuffersAndCopies(...);
```

负责创建 device buffer 和 host -> device copy：

```text
gpu.wait async
gpu.alloc lhs
gpu.alloc rhs
gpu.alloc out
gpu.memcpy lhs host -> device
gpu.memcpy rhs host -> device
```

注意：

```text
这个 helper 必须保留原 token 串行链。
不能把三个 gpu.alloc 或两个 gpu.memcpy 改成并行依赖。
```

```cpp
static gpu::LaunchOp createTiledMatMulLaunch(...);
```

负责创建 `gpu.launch` 和完整 launch body：

```text
创建两个 workgroup tile：
  memref<16x16xf64, #gpu.address_space<workgroup>>

在 launch body 内生成：
  block/thread 到 i/j 的映射
  M/N 边界判断
  k tile 外层 scf.for step 16
  tile load
  gpu.barrier
  k inner reduction scf.for step 1
  gpu.barrier
  out device store
  gpu.terminator
```

mapping 不变：

```text
blockIdx.x * 16 + threadIdx.x -> j
blockIdx.y * 16 + threadIdx.y -> i
```

```cpp
static Value createPaddedTileLoad(...);
```

负责一个带边界保护的 global memory load：

```text
if inBounds:
  memref.load
else:
  yield zero
```

越界线程必须 yield zero，不能跳过 store。所有线程仍必须写 workgroup tile，保证后续 barrier 和 shared load 行为一致。

```cpp
static Value createInnerTileReduction(...);
```

负责内层 k reduction：

```text
scf.for k = 0 to 16 step 1 iter_args(acc)
  lhs = load lhsTile[threadIdx.y, k]
  rhs = load rhsTile[k, threadIdx.x]
  acc = acc + lhs * rhs
```

这个 helper 只负责 shared memory 读取和 f64 multiply-add。barrier 仍由 `createTiledMatMulLaunch` 在调用点前后显式创建。

```cpp
static Value createCopyBackAndCleanup(...);
```

负责 device -> host copy 和资源释放：

```text
gpu.memcpy out host <- out device
gpu.dealloc lhs
gpu.dealloc rhs
gpu.dealloc out
gpu.wait
```

这个 helper 继续串接 launch 返回的 async token。最终 `gpu.wait` 仍是无返回值同步 wait。

## matchAndRewrite 形状

`MatMulOpLowering::matchAndRewrite` 第一阶段完成后只保留编排逻辑：

```cpp
auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
auto memRefType = convertTensorToMemRef(resultType);
auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

auto lhsType = cast<MemRefType>(operands[0].getType());
auto rhsType = cast<MemRefType>(operands[1].getType());

DeviceBuffers buffers = createDeviceBuffersAndCopies(...);
MatMulConfig config = createMatMulConfig(...);
gpu::LaunchOp launch = createTiledMatMulLaunch(...);
buffers.token = launch.getAsyncToken();

rewriter.setInsertionPointAfter(launch);
createCopyBackAndCleanup(...);
rewriter.replaceOp(op, alloc);
```

调用顺序必须保持为：

```text
1. host output memref.alloc
2. gpu.wait async
3. device alloc lhs/rhs/out
4. memcpy lhs/rhs host -> device
5. 常量和 grid/block 配置
6. gpu.launch
7. launch body
8. memcpy out device -> host
9. device dealloc lhs/rhs/out
10. gpu.wait
11. replace toy.matmul with host output
```

第一阶段验证目标是 IR 完全不变。即使只是把常量提前创建，也可能改变 `mlir-gpu` 文本输出顺序，所以 helper 的调用顺序也按原 `MatMulOpLowering` 的代码顺序保留。

## Review Checklist

```text
- MatMulConfig 只保存已有常量 Value，不新增语义。
- DeviceBuffers 只保存 device memref 和当前 token。
- createPaddedTileLoad 的 then/else region 都必须 scf.yield。
- createInnerTileReduction 仍从 lhsTile/rhsTile 读取，不直接读 global memory。
- createTiledMatMulLaunch 仍创建两个 workgroup tile。
- launch token 必须在 createCopyBackAndCleanup 前写回 buffers.token。
- gpu.barrier 仍在 shared tile store 后、inner reduction 后各出现一次。
- MatMulOpLowering 不再直接展开整段 kernel 生成逻辑。
```

## 验证

```bash
ninja -C build toyc-ch7
./build/bin/toyc-ch7 mlir/test/Examples/Toy/Ch7/matmul64.toy -emit=mlir-gpu
```

输出应和重构前一致，仍然包含：

```text
workgroup(... memref<16x16xf64, #gpu.address_space<workgroup>> ...)
gpu.barrier
scf.for ... step %c16
scf.for ... step %c1
```

推荐再跑现有 FileCheck：

```bash
./build/bin/llvm-lit mlir/test/Examples/Toy/Ch7/matmul64-gpu-lowering.mlir
```

当前执行状态：

```text
已完成：
  - 在 LowerToGPU.cpp 中新增 MatMulConfig。
  - 在 LowerToGPU.cpp 中新增 DeviceBuffers。
  - 拆出 createMatMulConfig。
  - 拆出 createDeviceBuffersAndCopies。
  - 拆出 createTiledMatMulLaunch。
  - 拆出 createPaddedTileLoad。
  - 拆出 createInnerTileReduction。
  - 拆出 createCopyBackAndCleanup。
  - MatMulOpLowering::matchAndRewrite 已缩减为 orchestration。

未在本地执行：
  - ninja -C build toyc-ch7
  - toyc-ch7 -emit=mlir-gpu
  - llvm-lit matmul64-gpu-lowering.mlir

未执行原因：
  - 本轮要求本地不要编译。
```
