// RUN: toyc-ch7 %s -emit=mlir-gpu-device-memory-matmul -x=mlir 2>&1 | FileCheck %s

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
// CHECK-NOT:     toy.matmul
// CHECK-DAG:     [[C1:%.*]] = arith.constant 1 : index
// CHECK-DAG:     [[C16:%.*]] = arith.constant 16 : index
// CHECK-DAG:     [[C64:%.*]] = arith.constant 64 : index
// CHECK:         [[LHS_HOST:%.*]] = memref.alloc() : memref<64x64xf64>
// CHECK:         [[RHS_HOST:%.*]] = memref.alloc() : memref<64x64xf64>
// CHECK:         [[OUT_HOST:%.*]] = memref.alloc() : memref<64x64xf64>
// CHECK:         [[T0:%.*]] = gpu.wait async
// CHECK:         [[LHS_DEV:%.*]], [[T1:%.*]] = gpu.alloc async {{.*}} : memref<64x64xf64>
// CHECK:         [[RHS_DEV:%.*]], [[T2:%.*]] = gpu.alloc async {{.*}} : memref<64x64xf64>
// CHECK:         [[OUT_DEV:%.*]], [[T3:%.*]] = gpu.alloc async {{.*}} : memref<64x64xf64>
// CHECK:         [[T4:%.*]] = gpu.memcpy async {{.*}} [[LHS_DEV]], [[LHS_HOST]]
// CHECK:         [[T5:%.*]] = gpu.memcpy async {{.*}} [[RHS_DEV]], [[RHS_HOST]]
// CHECK:         [[T6:%.*]] = gpu.launch async
// CHECK-SAME:      blocks
// CHECK-SAME:      threads
// CHECK-SAME:      workgroup(
// CHECK-SAME:        memref<16x16xf64, #gpu.address_space<workgroup>>
// CHECK-SAME:        memref<16x16xf64, #gpu.address_space<workgroup>>
// CHECK:           memref.load [[LHS_DEV]]
// CHECK:           memref.load [[RHS_DEV]]
// CHECK:           gpu.barrier
// CHECK:           memref.store {{.*}}, [[OUT_DEV]]
// CHECK:           gpu.terminator
// CHECK:         [[T7:%.*]] = gpu.memcpy async {{.*}} [[OUT_HOST]], [[OUT_DEV]]
// CHECK:         [[T8:%.*]] = gpu.dealloc async {{.*}} [[LHS_DEV]]
// CHECK:         [[T9:%.*]] = gpu.dealloc async {{.*}} [[RHS_DEV]]
// CHECK:         [[T10:%.*]] = gpu.dealloc async {{.*}} [[OUT_DEV]]
// CHECK:         gpu.wait
// CHECK-SAME:      [[T10]]
// CHECK:         toy.print [[OUT_HOST]] : memref<64x64xf64>
// CHECK-NOT:     builtin.unrealized_conversion_cast
