// RUN: toyc-ch7 %s -emit=mlir-gpu-nvvm -x=mlir 2>&1 | FileCheck %s

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
// CHECK:         toy.print {{.*}} : memref<64x64xf64>
// CHECK:       gpu.module
// CHECK:         llvm.func
// CHECK:           nvvm.kernel
// CHECK:           nvvm.read.ptx.sreg.ctaid.x
// CHECK:           nvvm.read.ptx.sreg.tid.x
// CHECK:           llvm.load
// CHECK:           llvm.fmul
// CHECK:           llvm.fadd
// CHECK:           llvm.store
// CHECK-NOT:     toy.matmul
// CHECK-NOT:     gpu.func
// CHECK-NOT:     gpu.block_id
// CHECK-NOT:     gpu.thread_id
