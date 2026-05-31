# Toy MatMul GPU Shared Memory Tiling

本文记录第八阶段：把 naive matmul kernel 改成 16x16 shared memory tiling。

第七阶段已经能运行 GPU matmul，但 kernel 还是：

```text
one thread computes one C[i, j]
for k in 0..K:
  load A[i, k] from global memory
  load B[k, j] from global memory
```

这样的问题是 global memory 访问重复很多。一个 block 内的 16x16 个线程会反复读取相邻的 A/B 元素。

第八阶段改成：

```text
每个 block 计算一个 16x16 的 C tile
每轮从 global memory 搬一个 A tile 和一个 B tile 到 workgroup/shared memory
block 内线程复用 shared memory 里的 tile
```

## 线程和 tile 映射

仍然使用：

```text
blockDim = 16 x 16 x 1
gridDim.x = ceil(N / 16)
gridDim.y = ceil(M / 16)
```

每个线程计算一个输出元素：

```text
row = blockIdx.y * 16 + threadIdx.y
col = blockIdx.x * 16 + threadIdx.x
```

每个 block 有两个 shared tile：

```mlir
workgroup(%tileA: memref<16x16xf64, #gpu.address_space<workgroup>>,
          %tileB: memref<16x16xf64, #gpu.address_space<workgroup>>)
```

## 核心算法

伪代码：

```cpp
acc = 0.0;

for (ktile = 0; ktile < K; ktile += 16) {
  if (row < M && ktile + threadIdx.x < K)
    tileA[threadIdx.y][threadIdx.x] =
        A[row][ktile + threadIdx.x];
  else
    tileA[threadIdx.y][threadIdx.x] = 0;

  if (ktile + threadIdx.y < K && col < N)
    tileB[threadIdx.y][threadIdx.x] =
        B[ktile + threadIdx.y][col];
  else
    tileB[threadIdx.y][threadIdx.x] = 0;

  gpu.barrier();

  for (kk = 0; kk < 16; ++kk) {
    acc += tileA[threadIdx.y][kk] * tileB[kk][threadIdx.x];
  }

  gpu.barrier();
}

if (row < M && col < N)
  C[row][col] = acc;
```

第一个 `gpu.barrier` 保证整个 block 都把 A/B tile 搬进 shared memory 后再计算。

第二个 `gpu.barrier` 保证所有线程都用完当前 tile 后，下一轮 `ktile` 才能覆盖 shared memory。

## LowerToGPU.cpp 的改动

创建 `gpu.launch` 后，新增两个 workgroup attribution：

```cpp
auto workgroupAddrSpace =
    gpu::AddressSpaceAttr::get(rewriter.getContext(),
                               gpu::AddressSpace::Workgroup);
auto tileType = MemRefType::get({blockSize, blockSize},
                                memRefType.getElementType(),
                                MemRefLayoutAttrInterface{},
                                Attribute(workgroupAddrSpace));
Value lhsTile = launch.addWorkgroupAttribution(tileType, loc);
Value rhsTile = launch.addWorkgroupAttribution(tileType, loc);
```

原来的 K 循环是：

```cpp
for k = 0; k < K; k += 1
```

现在改成两层：

```cpp
for ktile = 0; ktile < K; ktile += 16
  load global A/B -> lhsTile/rhsTile
  gpu.barrier
  for kk = 0; kk < 16; kk += 1
    acc += lhsTile[thread_y, kk] * rhsTile[kk, thread_x]
  gpu.barrier
```

## 生成的 MLIR 形状

`-emit=mlir-gpu` 应该能看到：

```mlir
gpu.launch ... workgroup(
  %arg... : memref<16x16xf64, #gpu.address_space<workgroup>>,
  %arg... : memref<16x16xf64, #gpu.address_space<workgroup>>
) {
  ...
  scf.for %ktile = %c0 to %c64 step %c16 iter_args(%acc = %zero) -> (f64) {
    scf.if %lhs_in_bounds {
      memref.load %lhsDevice[%row, %ktile_plus_thread_x]
    } else {
      yield %zero
    }
    scf.if %rhs_in_bounds {
      memref.load %rhsDevice[%ktile_plus_thread_y, %col]
    } else {
      yield %zero
    }
    memref.store ... %tileA[%thread_y, %thread_x]
    memref.store ... %tileB[%thread_y, %thread_x]
    gpu.barrier

    scf.for %kk = %c0 to %c16 step %c1 iter_args(%inner = %acc) -> (f64) {
      memref.load %tileA[%thread_y, %kk]
      memref.load %tileB[%kk, %thread_x]
      ...
    }

    gpu.barrier
    scf.yield ...
  }
}
```

## Barrier 放置

`gpu.barrier` 不能放在只由部分线程进入的 `scf.if` 里。否则有些线程到达
barrier，有些线程没有到达，kernel 会挂住。

所以本阶段的结构是：

```text
所有线程都执行：
  tile load 或写 0
  gpu.barrier
  inner compute
  gpu.barrier

只有最后 store C 时才用 row < M && col < N 保护
```

## 当前限制

这一阶段已经给 M/N/K 的边界 load 做了补 0 处理，也保留了输出 store 的
`row < M && col < N` 保护。因此 2x2 和 64x64 都能走同一套结构。

后续还可以优化的是减少边界判断开销：对已知 64x64 这种整除 16 的情况，
可以生成无边界判断的 fast path。

## 云端验证

重新编译：

```bash
ninja -C build toyc-ch7
```

看 GPU dialect：

```bash
./build/bin/toyc-ch7 mlir/test/Examples/Toy/Ch7/matmul64.toy \
  -emit=mlir-gpu
```

应能看到：

```text
workgroup(... memref<16x16xf64, #gpu.address_space<workgroup>> ...)
gpu.barrier
scf.for ... step %c16
scf.for ... step %c1
```

如果第七阶段 `gpu-jit` 已经跑通，也可以直接运行 2x2 示例。
