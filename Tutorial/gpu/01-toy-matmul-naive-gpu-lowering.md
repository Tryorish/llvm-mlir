# Toy MatMul Naive GPU Lowering 代码改动说明

本文解释当前为了把 `toy.matmul` lower 到 GPU 所做的代码改动。当前阶段目标不是直接运行 CUDA kernel，而是先让 Toy Ch7 能输出包含 `gpu.launch` 的 MLIR：

```text
toy.matmul -> memref.alloc + gpu.launch + scf.for + memref.load/store
```

也就是说，第一步验证目标是：

```bash
toyc-ch7 input.toy -emit=mlir-gpu
```

输出 IR 中应该看到 `gpu.launch`，并且不再看到 `toy.matmul`。

## 改动文件总览

这次涉及四类改动：

```text
mlir/examples/toy/Ch7/mlir/LowerToGPU.cpp
  新增 Toy -> GPU 的 partial lowering pass。

mlir/examples/toy/Ch7/include/toy/Passes.h
  声明 createLowerToGPUPass()。

mlir/examples/toy/Ch7/toyc.cpp
  新增 -emit=mlir-gpu，并把 GPU lowering 接入 pass pipeline。
  第二阶段又新增 -emit=mlir-gpu-outlined，用于 kernel outlining。

mlir/examples/toy/Ch7/CMakeLists.txt
  把 LowerToGPU.cpp 加进 toyc-ch7 的编译源文件。
  第二阶段又显式链接 MLIRGPUTransforms。
```

## LowerToGPU.cpp 的整体结构

新增文件：

```text
mlir/examples/toy/Ch7/mlir/LowerToGPU.cpp
```

这个文件的结构基本仿照 `LowerToAffineLoops.cpp`：

```text
helper functions
  convertTensorToMemRef
  insertAllocAndDealloc
  lowerOpToLoops

rewrite patterns
  AddOpLowering
  MulOpLowering
  NegOpLowering
  ConstantOpLowering
  FuncOpLowering
  PrintOpLowering
  ReturnOpLowering
  MatMulOpLowering
  TransposeOpLowering

pass
  ToyToGPULoweringPass
  createLowerToGPUPass()
```

这个 pass 是 partial lowering：它把大部分 Toy op 转成 `func` / `memref` / `arith` / `affine` / `scf` / `gpu`，但保留 `toy.print`，因为 Toy Ch7 原本就是在后续 `LowerToLLVM.cpp` 里专门 lowering `toy.print`。

## Tensor 到 MemRef

Toy 前端生成的是 tensor 类型，例如：

```mlir
%0 = toy.matmul %a, %b : tensor<2x3xf64>, tensor<3x2xf64> to tensor<2x2xf64>
```

GPU kernel 和已有 Toy lowering 更适合操作 memref，所以新增 pass 中继续使用 helper：

```cpp
static MemRefType convertTensorToMemRef(RankedTensorType type) {
  return MemRefType::get(type.getShape(), type.getElementType());
}
```

它只做类型转换：

```text
tensor<MxNxf64> -> memref<MxNxf64>
```

实际 buffer 由 `insertAllocAndDealloc` 创建：

```cpp
auto alloc = rewriter.create<memref::AllocOp>(loc, type);
alloc->moveBefore(&parentBlock->front());

auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
dealloc->moveBefore(&parentBlock->back());
```

这里沿用 Toy 教程风格：把 `alloc` 移到函数 block 开头，把 `dealloc` 移到 block 末尾。因为 Toy 示例没有复杂控制流，这种做法足够教学使用。

## 普通逐元素 op 仍走 affine loop

`toy.add`、`toy.mul`、`toy.neg`、`toy.transpose` 暂时仍复用 affine loop 的 lowering 思路：

```text
toy.add/tensor -> memref.alloc + affine.for + affine.load/store + arith.addf
```

这不是最终 GPU 性能路线，只是为了让本阶段集中解决 `toy.matmul -> gpu.launch`。如果后面要完整走 GPU 后端，需要进一步把 host 侧残留 affine lowering 到 scf/cf/llvm，或者把这些 op 也改成更统一的 GPU/Linalg 路线。

## MatMulOpLowering 的核心目标

当前 naive GPU matmul 的并行策略是：

```text
一个 GPU thread 计算一个 C[i, j]
每个 thread 内部串行循环 k
```

矩阵乘法：

```text
A: M x K
B: K x N
C: M x N

C[i, j] = sum(A[i, k] * B[k, j]), k = 0..K-1
```

对应 GPU 映射：

```text
threadIdx.x / blockIdx.x -> j，也就是列
threadIdx.y / blockIdx.y -> i，也就是行
z 维不用，固定为 1
```

## shape 信息的来源

`MatMulOpLowering` 里先取 result/lhs/rhs 类型：

```cpp
auto resultType = llvm::cast<RankedTensorType>(op->getResult(0).getType());
auto memRefType = convertTensorToMemRef(resultType);
auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

auto lhsType = llvm::cast<MemRefType>(operands[0].getType());
auto rhsType = llvm::cast<MemRefType>(operands[1].getType());

int64_t m = lhsType.getShape()[0];
int64_t k = lhsType.getShape()[1];
int64_t n = rhsType.getShape()[1];
```

这里要求 shape inference 已经跑过。否则 `toy.matmul` 的结果还是 unranked tensor，就无法静态取到 M/N/K。

所以 `toyc.cpp` 中 GPU pipeline 必须先运行：

```text
canonicalizer
shape-inference
canonicalizer
cse
```

再运行：

```text
createLowerToGPUPass()
```

## 常量的作用

代码中有几类常量：

```cpp
Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
Value mVal = rewriter.create<arith::ConstantIndexOp>(loc, m);
Value nVal = rewriter.create<arith::ConstantIndexOp>(loc, n);
Value kVal = rewriter.create<arith::ConstantIndexOp>(loc, k);
Value blockX = rewriter.create<arith::ConstantIndexOp>(loc, blockSize);
Value blockY = rewriter.create<arith::ConstantIndexOp>(loc, blockSize);
Value zero = rewriter.create<arith::ConstantOp>(
    loc, rewriter.getF64FloatAttr(0.0));
```

含义：

```text
c0    index 类型的 0，用作 k 循环下界
c1    index 类型的 1，用作循环步长、gridZ、blockZ
mVal  M，边界检查用
nVal  N，边界检查用
kVal  K，k 循环上界
blockX/blockY  block 维度，当前固定 16 x 16
zero  f64 类型的 0.0，用作累加器初始值
```

注意 `c0` 和 `zero` 不一样：

```text
c0   : index 0，给循环索引用
zero : f64 0.0，给浮点累加用
```

## grid 和 block 的计算

当前 block 大小固定：

```cpp
constexpr int64_t blockSize = 16;
```

因此：

```text
block = (16, 16, 1)
```

grid 需要覆盖整个输出矩阵：

```cpp
Value gridX = rewriter.create<arith::ConstantIndexOp>(
    loc, (n + blockSize - 1) / blockSize);
Value gridY = rewriter.create<arith::ConstantIndexOp>(
    loc, (m + blockSize - 1) / blockSize);
```

这是静态 ceildiv：

```text
gridX = ceil(N / 16)
gridY = ceil(M / 16)
gridZ = 1
```

如果 `N` 或 `M` 不是 16 的整数倍，多出来的 thread 会靠边界检查跳过。

## 创建 gpu.launch

关键代码：

```cpp
auto launch = rewriter.create<gpu::LaunchOp>(
    loc, gridX, gridY, c1, blockX, blockY, c1);
```

`gpu::LaunchOp` 的参数顺序是：

```text
gridSizeX, gridSizeY, gridSizeZ,
blockSizeX, blockSizeY, blockSizeZ
```

所以这里等价于：

```text
grid  = (gridX, gridY, 1)
block = (16, 16, 1)
```

两个 `c1` 的作用不同：

```text
第一个 c1 -> gridSizeZ = 1
第二个 c1 -> blockSizeZ = 1
```

因为矩阵乘只用二维并行，z 维不用。

## 从 block/thread id 计算 i 和 j

`gpu.launch` region 中有 block id 和 thread id。代码中取出来：

```cpp
gpu::KernelDim3 blockIds = launch.getBlockIds();
gpu::KernelDim3 threadIds = launch.getThreadIds();
```

然后计算输出坐标：

```cpp
Value jBlock = rewriter.create<arith::MulIOp>(loc, blockIds.x, blockX);
Value j = rewriter.create<arith::AddIOp>(loc, jBlock, threadIds.x);

Value iBlock = rewriter.create<arith::MulIOp>(loc, blockIds.y, blockY);
Value i = rewriter.create<arith::AddIOp>(loc, iBlock, threadIds.y);
```

等价于 CUDA 写法：

```cpp
int j = blockIdx.x * blockDim.x + threadIdx.x;
int i = blockIdx.y * blockDim.y + threadIdx.y;
```

映射关系：

```text
i -> C 的行
j -> C 的列
```

## 边界检查

当 `M` 或 `N` 不是 16 的倍数时，最后一个 block 会产生越界 thread。代码用 `scf.if` 包住实际计算：

```cpp
Value inM =
    rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, i, mVal);
Value inN =
    rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, j, nVal);
Value inBounds = rewriter.create<arith::AndIOp>(loc, inM, inN);

auto ifOp = rewriter.create<scf::IfOp>(
    loc, inBounds, /*withElseRegion=*/false);
```

语义是：

```cpp
if (i < M && j < N) {
  compute C[i, j]
}
```

这里用 `ult` 是 unsigned less-than。`i` 和 `j` 都从 block/thread id 算出来，不会是负数，因此用 unsigned predicate 可以接受。

## k 维累加循环

每个 thread 负责一个 `C[i, j]`，但 `k` 维仍然串行：

```cpp
auto forK = rewriter.create<scf::ForOp>(
    loc, c0, kVal, c1, ValueRange{zero},
    [&](OpBuilder &nestedBuilder, Location loc, Value ivK,
        ValueRange iterArgs) {
      Value acc = iterArgs[0];
      Value lhs = nestedBuilder.create<memref::LoadOp>(
          loc, operands[0], ValueRange{i, ivK});
      Value rhs = nestedBuilder.create<memref::LoadOp>(
          loc, operands[1], ValueRange{ivK, j});
      Value prod = nestedBuilder.create<arith::MulFOp>(loc, lhs, rhs);
      Value sum = nestedBuilder.create<arith::AddFOp>(loc, acc, prod);
      nestedBuilder.create<scf::YieldOp>(loc, sum);
    });
```

对应伪代码：

```cpp
double acc = 0.0;
for (int kk = 0; kk < K; ++kk) {
  acc += A[i][kk] * B[kk][j];
}
```

`ValueRange{zero}` 是 `scf.for` 的 iter_arg 初始值：

```text
acc 初始为 0.0
每轮 yield sum
循环结果 forK.getResult(0) 是最终 acc
```

## 写回 C[i, j]

循环结束后，把累加结果写入输出 buffer：

```cpp
rewriter.create<memref::StoreOp>(
    loc, forK.getResult(0), alloc, ValueRange{i, j});
```

对应：

```cpp
C[i][j] = acc;
```

然后为 `gpu.launch` region 添加 terminator：

```cpp
rewriter.setInsertionPointToEnd(&launch.getBody().front());
rewriter.create<gpu::TerminatorOp>(loc);
```

最后把原来的 `toy.matmul` 替换成输出 memref：

```cpp
rewriter.replaceOp(op, alloc);
```

所以原始 IR：

```mlir
%c = toy.matmul %a, %b : tensor<MxKxf64>, tensor<KxNxf64> to tensor<MxNxf64>
```

会变成概念上的：

```mlir
%c = memref.alloc() : memref<MxNxf64>
gpu.launch ... {
  ...
  scf.if %in_bounds {
    %sum = scf.for ... iter_args(%acc = %zero) -> f64 {
      ...
    }
    memref.store %sum, %c[%i, %j] : memref<MxNxf64>
  }
  gpu.terminator
}
```

## ConversionTarget 的合法性设置

pass 中设置了合法 dialect：

```cpp
target.addLegalDialect<affine::AffineDialect, BuiltinDialect,
                       arith::ArithDialect, func::FuncDialect,
                       gpu::GPUDialect, memref::MemRefDialect,
                       scf::SCFDialect>();
```

并把 Toy dialect 设为非法：

```cpp
target.addIllegalDialect<toy::ToyDialect>();
```

但是 `toy.print` 被动态标成合法：

```cpp
target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
  return llvm::none_of(op->getOperandTypes(),
                       [](Type type) { return llvm::isa<TensorType>(type); });
});
```

意思是：

```text
toy.print 可以保留，但它的输入不能还是 tensor。
```

这样 `toy.print` 会从：

```mlir
toy.print %c : tensor<MxNxf64>
```

变成：

```mlir
toy.print %c : memref<MxNxf64>
```

后续如果走 LLVM lowering，再由 `LowerToLLVM.cpp` 处理 `toy.print`。

## Passes.h 的改动

新增声明：

```cpp
std::unique_ptr<mlir::Pass> createLowerToGPUPass();
```

这让 `toyc.cpp` 可以调用 GPU lowering pass。

## toyc.cpp 的改动

### 新增 emit action

`Action` enum 增加：

```cpp
DumpMLIRGPU,
```

命令行增加：

```cpp
clEnumValN(DumpMLIRGPU, "mlir-gpu",
           "output the MLIR dump after gpu lowering")
```

因此可以使用：

```bash
toyc-ch7 input.toy -emit=mlir-gpu
```

### 拆开 affine 和 GPU lowering 条件

原本逻辑大致是：

```cpp
bool isLoweringToAffine = emitAction >= Action::DumpMLIRAffine;
```

这对线性 pipeline 有效，但加入 `DumpMLIRGPU` 后有风险：如果 `DumpMLIRGPU` 排在 `DumpMLIRAffine` 后面，会误触发 CPU affine lowering。

所以现在改成：

```cpp
bool isLoweringToGPU = emitAction == Action::DumpMLIRGPU ||
                       emitAction == Action::DumpMLIRGPUOutlined;
bool isOutliningGPU = emitAction == Action::DumpMLIRGPUOutlined;
bool isLoweringToAffine = emitAction == Action::DumpMLIRAffine ||
                          emitAction >= Action::DumpMLIRLLVM;
bool isLoweringToLLVM = emitAction >= Action::DumpMLIRLLVM;
```

含义：

```text
-emit=mlir-gpu           只走 Toy -> gpu.launch
-emit=mlir-gpu-outlined  先走 Toy -> gpu.launch，再做 gpu kernel outlining
-emit=mlir-affine        只走 affine lowering
-emit=mlir-llvm          仍走原来的 affine -> llvm 路线
```

### GPU lowering pipeline

新增：

```cpp
if (isLoweringToGPU) {
  pm.addPass(mlir::toy::createLowerToGPUPass());

  mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
  optPM.addPass(mlir::createCanonicalizerPass());
  optPM.addPass(mlir::createCSEPass());
}
```

也就是说完整 `-emit=mlir-gpu` pipeline 是：

```text
load Toy/MLIR
  -> inline
  -> canonicalize
  -> shape inference
  -> canonicalize
  -> cse
  -> toy-to-gpu
  -> func.func(canonicalize)
  -> func.func(cse)
  -> dump MLIR
```

### 修正 MLIR 输出判断

`main` 中输出 MLIR 的判断也改成显式枚举：

```cpp
bool isOutputingMLIR = emitAction == Action::DumpMLIR ||
                       emitAction == Action::DumpMLIRAffine ||
                       emitAction == Action::DumpMLIRGPU ||
                       emitAction == Action::DumpMLIRGPUOutlined ||
                       emitAction == Action::DumpMLIRLLVM;
```

否则 `-emit=mlir-gpu` 可能不会停在 dump 阶段。

## CMakeLists.txt 的改动

把新文件加入 `toyc-ch7`：

```cmake
mlir/LowerToGPU.cpp
```

当前 Ch7 已经链接：

```cmake
${dialect_libs}
${conversion_libs}
${extension_libs}
```

第二阶段使用 `createGpuKernelOutliningPass()`，因此还需要显式链接：

```cmake
MLIRGPUTransforms
```

## 预期输出形态

输入：

```toy
def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b<3, 2> = [[7, 8], [9, 10], [11, 12]];
  var c = matmul(a, b);
  print(c);
}
```

运行：

```bash
toyc-ch7 /tmp/matmul.toy -emit=mlir-gpu
```

期望看到类似结构：

```mlir
%alloc = memref.alloc() : memref<2x2xf64>
gpu.launch blocks(...) in (...) threads(...) in (...) {
  %j = ...
  %i = ...
  %in_bounds = ...
  scf.if %in_bounds {
    %sum = scf.for %k = ... iter_args(%acc = ...) -> (f64) {
      %a = memref.load ...
      %b = memref.load ...
      %p = arith.mulf ...
      %s = arith.addf ...
      scf.yield %s : f64
    }
    memref.store %sum, %alloc[%i, %j] : memref<2x2xf64>
  }
  gpu.terminator
}
toy.print %alloc : memref<2x2xf64>
```

并且不应该再出现：

```mlir
toy.matmul
```

## 当前阶段的限制

当前实现是 naive baseline，有几个明确限制：

```text
1. 还没有真正处理 host/device 内存搬运。
   也就是还没有显式生成 gpu.alloc / gpu.memcpy。

2. 还不是完整可运行 GPU pipeline。
   当前主要目标是生成 gpu.launch IR。

3. 每个 thread 串行算一个 C[i,j] 的 k 循环。
   没有 shared memory tiling、vectorization、Tensor Core。

4. Toy 当前元素类型是 f64。
   GPU 上 f64 通常可用但性能差，后续性能实验建议支持 f32。

5. 普通 add/mul/neg/transpose 暂时仍走 affine loop。
   后续如果要完整 GPU 后端，需要继续 lower 或转 Linalg 路线。
```

## 下一步建议

按阶段推进：

```text
第一阶段：验证 toy.matmul -> gpu.launch
  -emit=mlir-gpu

第二阶段：验证 gpu.launch 能 outline
  gpu-kernel-outlining

第三阶段：接 NVVM lowering
  gpu-lower-to-nvvm-pipeline

第四阶段：补显式 device memory
  gpu.alloc / gpu.memcpy / gpu.dealloc

第五阶段：真正运行
  CUDA/ROCm runtime wrapper + binary serialization + runner/JIT

第六阶段：优化
  linalg.matmul + tiling + vector/nvgpu/tensor core
```

目前 `-emit=mlir-gpu` 完成第一阶段。第二阶段的 kernel outlining 见：

```text
Tutorial/gpu/02-toy-matmul-gpu-kernel-outlining.md
```
