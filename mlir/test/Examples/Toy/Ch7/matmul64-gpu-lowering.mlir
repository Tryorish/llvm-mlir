// RUN: toyc-ch7 %s -emit=mlir-gpu -x=mlir 2>&1 | FileCheck %s

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
// CHECK-DAG:     [[C0:%.*]] = arith.constant 0 : index
// CHECK-DAG:     [[C1:%.*]] = arith.constant 1 : index
// CHECK-DAG:     [[C4:%.*]] = arith.constant 4 : index
// CHECK-DAG:     [[C16:%.*]] = arith.constant 16 : index
// CHECK-DAG:     [[C64:%.*]] = arith.constant 64 : index
// CHECK-DAG:     [[ZERO:%.*]] = arith.constant 0.000000e+00 : f64
// CHECK:         [[OUT:%.*]] = memref.alloc() : memref<64x64xf64>
// CHECK:         [[T0:%.*]] = gpu.wait async
// CHECK:         [[LHS_DEV:%.*]], [[T1:%.*]] = gpu.alloc async {{.*}} : memref<64x64xf64>
// CHECK:         [[RHS_DEV:%.*]], [[T2:%.*]] = gpu.alloc async {{.*}} : memref<64x64xf64>
// CHECK:         [[OUT_DEV:%.*]], [[T3:%.*]] = gpu.alloc async {{.*}} : memref<64x64xf64>
// CHECK:         [[T4:%.*]] = gpu.memcpy async {{.*}} [[LHS_DEV]]
// CHECK:         [[T5:%.*]] = gpu.memcpy async {{.*}} [[RHS_DEV]]
// CHECK:         [[T6:%.*]] = gpu.launch async
// CHECK-SAME:      blocks({{.*}}) in ({{.*}} = [[C4]], {{.*}} = [[C4]], {{.*}} = [[C1]])
// CHECK-SAME:      threads({{.*}}) in ({{.*}} = [[C16]], {{.*}} = [[C16]], {{.*}} = [[C1]])
// CHECK:           [[IN_M:%.*]] = arith.cmpi ult, {{.*}}, [[C64]] : index
// CHECK:           [[IN_N:%.*]] = arith.cmpi ult, {{.*}}, [[C64]] : index
// CHECK:           [[IN_BOUNDS:%.*]] = arith.andi [[IN_M]], [[IN_N]] : i1
// CHECK:           scf.if [[IN_BOUNDS]] {
// CHECK:             [[SUM:%.*]] = scf.for {{.*}} = [[C0]] to [[C64]] step [[C1]] iter_args({{.*}} = [[ZERO]]) -> (f64) {
// CHECK:               [[LHS:%.*]] = memref.load {{.*}} : memref<64x64xf64>
// CHECK:               [[RHS:%.*]] = memref.load {{.*}} : memref<64x64xf64>
// CHECK:               [[MUL:%.*]] = arith.mulf [[LHS]], [[RHS]] : f64
// CHECK:               [[ADD:%.*]] = arith.addf {{.*}}, [[MUL]] : f64
// CHECK:               scf.yield [[ADD]] : f64
// CHECK:             }
// CHECK:             memref.store [[SUM]], [[OUT_DEV]]
// CHECK:           }
// CHECK:           gpu.terminator
// CHECK:         [[T7:%.*]] = gpu.memcpy async {{.*}} [[OUT]], [[OUT_DEV]]
// CHECK:         [[T8:%.*]] = gpu.dealloc async {{.*}} [[LHS_DEV]]
// CHECK:         [[T9:%.*]] = gpu.dealloc async {{.*}} [[RHS_DEV]]
// CHECK:         [[T10:%.*]] = gpu.dealloc async {{.*}} [[OUT_DEV]]
// CHECK:         gpu.wait
// CHECK-SAME:    [[T10]]
// CHECK:         toy.print [[OUT]] : memref<64x64xf64>
// CHECK-NOT:     toy.matmul
