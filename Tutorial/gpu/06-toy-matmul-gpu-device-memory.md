# Toy MatMul GPU Device Memory

本文记录第六阶段改动：不再把 host `memref` 指针直接传给 GPU kernel，而是在 host 侧显式插入 device buffer 分配、拷贝和释放。

第五阶段的 LLVM IR 已经能看到：

```llvm
call void @mgpuLaunchKernel(...)
```

但 kernel 参数仍然来自 host `malloc` 的指针。真实 CUDA 程序通常需要：

```text
host input -> device input
kernel 写 device output
device output -> host output
```

所以第六阶段在 `toy.matmul` lowering 中插入：

```mlir
%t0 = gpu.wait async
%lhs_dev, %t1 = gpu.alloc async [%t0] () : memref<MxKxf64>
%rhs_dev, %t2 = gpu.alloc async [%t1] () : memref<KxNxf64>
%out_dev, %t3 = gpu.alloc async [%t2] () : memref<MxNxf64>

%t4 = gpu.memcpy async [%t3] %lhs_dev, %lhs : memref<MxKxf64>, memref<MxKxf64>
%t5 = gpu.memcpy async [%t4] %rhs_dev, %rhs : memref<KxNxf64>, memref<KxNxf64>

%t6 = gpu.launch async [%t5] ... {
  memref.load %lhs_dev[...]
  memref.load %rhs_dev[...]
  memref.store ... %out_dev[...]
}

%t7 = gpu.memcpy async [%t6] %host_out, %out_dev : memref<MxNxf64>, memref<MxNxf64>
%t8 = gpu.dealloc async [%t7] %lhs_dev : memref<MxKxf64>
%t9 = gpu.dealloc async [%t8] %rhs_dev : memref<KxNxf64>
%t10 = gpu.dealloc async [%t9] %out_dev : memref<MxNxf64>
gpu.wait [%t10]
```

## LowerToGPU.cpp 的改动

在 `MatMulOpLowering` 中，原来只创建 host 输出：

```cpp
auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);
```

现在额外创建 device buffer：

```cpp
Type asyncTokenType = gpu::AsyncTokenType::get(rewriter.getContext());
Value token =
    rewriter.create<gpu::WaitOp>(loc, asyncTokenType, ValueRange{})
        .getAsyncToken();

auto lhsDeviceAlloc = rewriter.create<gpu::AllocOp>(
    loc, lhsType, asyncTokenType, ValueRange{token}, ValueRange{},
    ValueRange{}, false);
token = lhsDeviceAlloc.getAsyncToken();
auto rhsDeviceAlloc = rewriter.create<gpu::AllocOp>(
    loc, rhsType, asyncTokenType, ValueRange{token}, ValueRange{},
    ValueRange{}, false);
token = rhsDeviceAlloc.getAsyncToken();
auto outDeviceAlloc = rewriter.create<gpu::AllocOp>(
    loc, memRefType, asyncTokenType, ValueRange{token}, ValueRange{},
    ValueRange{}, false);
token = outDeviceAlloc.getAsyncToken();
```

取出 memref result：

```cpp
Value lhsDevice = lhsDeviceAlloc.getMemref();
Value rhsDevice = rhsDeviceAlloc.getMemref();
Value outDevice = outDeviceAlloc.getMemref();
```

然后在 launch 前拷贝输入：

```cpp
token = rewriter
            .create<gpu::MemcpyOp>(loc, asyncTokenType, ValueRange{token},
                                   lhsDevice, operands[0])
            .getAsyncToken();
token = rewriter
            .create<gpu::MemcpyOp>(loc, asyncTokenType, ValueRange{token},
                                   rhsDevice, operands[1])
            .getAsyncToken();
```

kernel 内部改成读写 device buffer：

```cpp
nestedBuilder.create<memref::LoadOp>(loc, lhsDevice, ValueRange{i, ivK});
nestedBuilder.create<memref::LoadOp>(loc, rhsDevice, ValueRange{ivK, j});
rewriter.create<memref::StoreOp>(loc, forK.getResult(0), outDevice,
                                 ValueRange{i, j});
```

launch 后拷回 host 输出并释放 device buffer：

```cpp
rewriter.setInsertionPointAfter(launch);
token = rewriter
            .create<gpu::MemcpyOp>(loc, asyncTokenType, ValueRange{token},
                                   alloc, outDevice)
            .getAsyncToken();
token = rewriter
            .create<gpu::DeallocOp>(loc, asyncTokenType, ValueRange{token},
                                    lhsDevice)
            .getAsyncToken();
token = rewriter
            .create<gpu::DeallocOp>(loc, asyncTokenType, ValueRange{token},
                                    rhsDevice)
            .getAsyncToken();
token = rewriter
            .create<gpu::DeallocOp>(loc, asyncTokenType, ValueRange{token},
                                    outDevice)
            .getAsyncToken();
rewriter.create<gpu::WaitOp>(loc, Type(), ValueRange{token});
```

最后仍然：

```cpp
rewriter.replaceOp(op, alloc);
```

因为 `toy.print` 要打印 host 侧输出 `alloc`。

## 为什么还保留 host output alloc

`toy.print` 是 host 侧打印逻辑。它不能直接打印 device pointer，所以第六阶段保留 host output：

```text
device output -> gpu.memcpy -> host output -> toy.print/printf
```

这也是为什么 kernel 写的是 `outDevice`，但 `replaceOp` 返回的是 `alloc`。

## 与第五阶段的关系

第五阶段 `gpu-to-llvm` 会把新增的 GPU memory op lower 成 runtime wrapper：

```text
gpu.alloc   -> mgpuMemAlloc
gpu.memcpy  -> mgpuMemcpy
gpu.dealloc -> mgpuMemFree
```

这里必须使用 async 形式。当前 MLIR 的 `gpu-to-llvm` runtime lowering 对
`gpu.alloc`、`gpu.memcpy`、`gpu.dealloc` 的要求是：

```text
1. op 有 async result token。
2. op 恰好有一个 async dependency。
```

否则这些 op 会留在 `-emit=mlir-gpu-host` 输出里，并产生无法消掉的
`builtin.unrealized_conversion_cast`，最终 `-emit=llvm-gpu` 会报：

```text
LLVM Translation failed for operation: builtin.unrealized_conversion_cast
```

所以 `-emit=llvm-gpu` 应该能看到：

```llvm
call ptr @mgpuMemAlloc(...)
call void @mgpuMemcpy(...)
call void @mgpuLaunchKernel(...)
call void @mgpuMemcpy(...)
call void @mgpuMemFree(...)
```

## 当前限制

这一阶段仍然是教学版：

```text
1. 使用一条串行 async token 链，没有把输入拷贝并行化。
2. 没有做错误检查。
3. 没有做 tiling/shared memory 优化。
4. kernel 仍是 naive one-thread-one-output。
```

但内存语义已经从“直接传 host 指针”推进到“显式 device allocation/copy”。

## 云平台验证命令

重新编译：

```bash
ninja -C build toyc-ch7
```

看 GPU dialect 阶段：

```bash
./build/bin/toyc-ch7 mlir/test/Examples/Toy/Ch7/matmul-codegen.toy \
  -emit=mlir-gpu
```

应能看到：

```text
gpu.wait async
gpu.alloc async
gpu.memcpy async
gpu.launch async
gpu.dealloc async
gpu.wait
```

看 host runtime lowering 后的 MLIR：

```bash
./build/bin/toyc-ch7 mlir/test/Examples/Toy/Ch7/matmul-codegen.toy \
  -emit=mlir-gpu-host
```

应能看到：

```text
llvm.call @mgpuMemAlloc
llvm.call @mgpuMemcpy
llvm.call @mgpuMemFree
gpu.launch_func
```

看最终 LLVM IR：

```bash
./build/bin/toyc-ch7 mlir/test/Examples/Toy/Ch7/matmul-codegen.toy \
  -emit=llvm-gpu
```

应能看到：

```text
call ptr @mgpuMemAlloc
call void @mgpuMemcpy
call void @mgpuLaunchKernel
call void @mgpuMemFree
```
