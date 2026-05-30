// RUN: toyc-ch7 %s -emit=llvm-gpu -x=mlir 2>&1 | FileCheck %s

module {
  toy.func @main() {
    %0 = toy.constant dense<1.000000e+00> : tensor<64x64xf64>
    %1 = toy.constant dense<1.000000e+00> : tensor<64x64xf64>
    %2 = toy.matmul %0, %1 : tensor<64x64xf64>, tensor<64x64xf64> to tensor<64x64xf64>
    toy.print %2 : tensor<64x64xf64>
    toy.return
  }
}

// CHECK: @main_kernel_binary = internal constant
// CHECK: @main_kernel_module = internal global ptr null
// CHECK: @llvm.global_ctors
// CHECK: @llvm.global_dtors
// CHECK: call ptr @mgpuModuleLoad
// CHECK: call ptr @mgpuModuleGetFunction
// CHECK: call ptr @mgpuStreamCreate()
// CHECK: call void @mgpuLaunchKernel
// CHECK: call void @mgpuStreamSynchronize
// CHECK: call void @mgpuStreamDestroy
// CHECK: call void @mgpuModuleUnload
// CHECK-NOT: gpu.launch_func
// CHECK-NOT: gpu.binary
