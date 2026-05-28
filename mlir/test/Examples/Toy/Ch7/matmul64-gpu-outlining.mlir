// RUN: toyc-ch7 %s -emit=mlir-gpu-outlined -x=mlir 2>&1 | FileCheck %s

module {
  toy.func @main() {
    %0 = toy.constant dense<1.000000e+00> : tensor<64x64xf64>
    %1 = toy.constant dense<1.000000e+00> : tensor<64x64xf64>
    %2 = toy.matmul %0, %1 : tensor<64x64xf64>, tensor<64x64xf64> to tensor<64x64xf64>
    toy.print %2 : tensor<64x64xf64>
    toy.return
  }
}

// CHECK-LABEL: func.func @main()
// CHECK:         gpu.launch_func
// CHECK-SAME:      blocks in
// CHECK-SAME:      threads in
// CHECK-SAME:      args(
// CHECK:         toy.print {{.*}} : memref<64x64xf64>
// CHECK:       gpu.module
// CHECK:         gpu.func
// CHECK-SAME:      kernel
// CHECK:           scf.if
// CHECK:             scf.for
// CHECK:               memref.load
// CHECK:               arith.mulf
// CHECK:               arith.addf
// CHECK:             memref.store
// CHECK:           gpu.return
// CHECK-NOT:     toy.matmul
// CHECK-NOT:     gpu.launch blocks
