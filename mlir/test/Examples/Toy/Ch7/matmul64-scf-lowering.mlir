// RUN: toyc-ch7 %s -emit=mlir-scf-matmul -x=mlir 2>&1 | FileCheck %s

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
// CHECK-NOT:     gpu.launch
// CHECK-NOT:     workgroup
// CHECK-DAG:     [[C0:%.*]] = arith.constant 0 : index
// CHECK-DAG:     [[C1:%.*]] = arith.constant 1 : index
// CHECK-DAG:     [[C64:%.*]] = arith.constant 64 : index
// CHECK-DAG:     [[ZERO:%.*]] = arith.constant 0.000000e+00 : f64
// CHECK:         [[OUT:%.*]] = memref.alloc() : memref<64x64xf64>
// CHECK:         scf.for [[I:%.*]] = [[C0]] to [[C64]] step [[C1]] {
// CHECK:           scf.for [[J:%.*]] = [[C0]] to [[C64]] step [[C1]] {
// CHECK:             [[SUM:%.*]] = scf.for [[K:%.*]] = [[C0]] to [[C64]] step [[C1]] iter_args([[ACC:%.*]] = [[ZERO]]) -> (f64) {
// CHECK:               [[LHS:%.*]] = memref.load {{.*}}[[I]], [[K]]{{.*}} : memref<64x64xf64>
// CHECK:               [[RHS:%.*]] = memref.load {{.*}}[[K]], [[J]]{{.*}} : memref<64x64xf64>
// CHECK:               [[MUL:%.*]] = arith.mulf [[LHS]], [[RHS]] : f64
// CHECK:               [[ADD:%.*]] = arith.addf [[ACC]], [[MUL]] : f64
// CHECK:               scf.yield [[ADD]] : f64
// CHECK:             }
// CHECK:             memref.store [[SUM]], [[OUT]][[[I]], [[J]]] : memref<64x64xf64>
// CHECK:           }
// CHECK:         }
// CHECK:         toy.print [[OUT]] : memref<64x64xf64>
