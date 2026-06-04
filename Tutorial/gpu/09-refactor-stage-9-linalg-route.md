# Stage 9: Consider A Linalg-Based Route

目标：在手写 pipeline 稳定后，评估是否引入更正规的 MLIR linalg 路线。

这不是当前重构的第一步。它应该放在手写 GPU lowering 的教学链路清楚、测试矩阵稳定之后。

## 候选路线

```text
toy.matmul
  -> linalg.matmul
  -> bufferization
  -> linalg tiling
  -> mapping to gpu blocks/threads
  -> promote to workgroup memory
  -> vectorize / unroll
  -> lower to NVVM
```

## 为什么不先做

这条路线更接近真实 compiler，但会同时引入：

```text
linalg
bufferization
tiling interface
mapping attribute
transform dialect 或 linalg transform utilities
vector/nvgpu 后续优化空间
```

如果一开始就引入这些概念，容易模糊当前文档的主线：

```text
toy.matmul
  -> gpu.launch
  -> gpu.module / gpu.func
  -> NVVM
  -> gpu.binary
  -> host runtime calls
  -> gpu-jit
  -> shared memory tiling
```

## 评估前置条件

```text
- Stage 1 到 Stage 8 已经完成。
- 手写 lowering 的每一层都有测试。
- 非 16 倍数矩阵边界行为已有覆盖。
- device memory 管理已经从 kernel 生成中拆出。
- 当前教学路线可以独立解释和验证。
```

## 调研任务

```text
1. 设计 toy.matmul -> linalg.matmul lowering。
2. 确认 Toy tensor/memref lowering 与 bufferization 的衔接方式。
3. 调研 linalg tiling 到 scf.forall 或 gpu.launch 的路径。
4. 调研 workgroup memory promotion 的标准 pass/transform。
5. 明确哪些部分替代手写 pass，哪些部分继续保留教学版本。
```

## 产出

建议单独形成新文档，而不是继续扩展 `09` 系列：

```text
Tutorial/gpu/10-toy-matmul-linalg-gpu-route.md
```

该文档应该回答：

```text
- toy.matmul 到 linalg.matmul 的 IR 形状。
- bufferization 前后的 IR 形状。
- tiling/mapping/promotion 使用哪些 MLIR 标准能力。
- 与手写 pipeline 的优缺点对比。
- 哪条路径作为教学主线，哪条路径作为工程化路线。
```
