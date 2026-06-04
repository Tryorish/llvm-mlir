// RUN: toyc-ch7 %s -emit=mlir-gpu-naive-matmul -x=mlir 2>&1 | FileCheck %s

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
// CHECK-NOT:     workgroup
// CHECK-NOT:     gpu.barrier
// CHECK-DAG:     [[C0:%.*]] = arith.constant 0 : index
// CHECK-DAG:     [[C1:%.*]] = arith.constant 1 : index
// CHECK-DAG:     [[C16:%.*]] = arith.constant 16 : index
// CHECK-DAG:     [[C64:%.*]] = arith.constant 64 : index
// CHECK:         [[OUT:%.*]] = memref.alloc() : memref<64x64xf64>
// CHECK:         gpu.launch blocks([[BX:%.*]], [[BY:%.*]], {{%.*}}) in
// CHECK-SAME:               ({{%.*}} = {{.*}}, {{%.*}} = {{.*}}, {{%.*}} = [[C1]])
// CHECK-SAME:               threads([[TX:%.*]], [[TY:%.*]], {{%.*}}) in
// CHECK-SAME:               ({{%.*}} = [[C16]], {{%.*}} = [[C16]], {{%.*}} = [[C1]])
// CHECK:           [[JBLOCK:%.*]] = arith.muli [[BX]], [[C16]] : index
// CHECK:           [[J:%.*]] = arith.addi [[JBLOCK]], [[TX]] : index
// CHECK:           [[IBLOCK:%.*]] = arith.muli [[BY]], [[C16]] : index
// CHECK:           [[I:%.*]] = arith.addi [[IBLOCK]], [[TY]] : index
// CHECK:           [[IN_M:%.*]] = arith.cmpi ult, [[I]], [[C64]] : index
// CHECK:           [[IN_N:%.*]] = arith.cmpi ult, [[J]], [[C64]] : index
// CHECK:           [[IN_MN:%.*]] = arith.andi [[IN_M]], [[IN_N]] : i1
// CHECK:           scf.if [[IN_MN]] {
// CHECK:             [[SUM:%.*]] = scf.for [[KO:%.*]] = [[C0]] to [[C64]] step [[C16]]
// CHECK-SAME:            iter_args
// CHECK-SAME:            -> (f64) {
// CHECK:               scf.for [[KI:%.*]] = [[C0]] to [[C16]] step [[C1]]
// CHECK-SAME:              iter_args
// CHECK-SAME:              -> (f64) {
// CHECK:                 [[K:%.*]] = arith.addi [[KO]], [[KI]] : index
// CHECK:                 memref.load {{.*}}[[I]], [[K]]{{.*}} : memref<64x64xf64>
// CHECK:                 memref.load {{.*}}[[K]], [[J]]{{.*}} : memref<64x64xf64>
// CHECK:                 arith.mulf
// CHECK:                 arith.addf
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               scf.yield
// CHECK:             }
// CHECK:             memref.store [[SUM]], [[OUT]][[[I]], [[J]]] : memref<64x64xf64>
// CHECK:           }
// CHECK-NOT:       workgroup
// CHECK-NOT:       gpu.barrier
// CHECK:           gpu.terminator
// CHECK:         toy.print [[OUT]] : memref<64x64xf64>
