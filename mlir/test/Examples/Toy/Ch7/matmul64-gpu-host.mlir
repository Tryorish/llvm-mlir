// RUN: toyc-ch7 %s -emit=mlir-gpu-host -x=mlir 2>&1 | FileCheck %s

module {
  toy.func @main() {
    %0 = toy.constant dense<1.000000e+00> : tensor<64x64xf64>
    %1 = toy.constant dense<1.000000e+00> : tensor<64x64xf64>
    %2 = toy.matmul %0, %1 : tensor<64x64xf64>, tensor<64x64xf64> to tensor<64x64xf64>
    toy.print %2 : tensor<64x64xf64>
    toy.return
  }
}

// CHECK-LABEL: llvm.func @main()
// CHECK:         gpu.launch_func
// CHECK:         llvm.call @printf
// CHECK:       gpu.binary
// CHECK:         #gpu.object<#nvvm.target
// CHECK-NOT:     toy.print
// CHECK-NOT:     toy.matmul
// CHECK-NOT:     affine.
// CHECK-NOT:     builtin.unrealized_conversion_cast
// CHECK-NOT:     func.func
// CHECK-NOT:     gpu.module
