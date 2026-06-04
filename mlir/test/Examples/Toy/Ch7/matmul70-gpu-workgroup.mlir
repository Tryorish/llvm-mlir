// RUN: toyc-ch7 %s -emit=mlir-gpu-workgroup-matmul -x=mlir 2>&1 | FileCheck %s

module {
  toy.func @main() {
    %0 = toy.constant dense<1.000000e+00> : tensor<70x70xf64>
    %1 = toy.constant dense<1.000000e+00> : tensor<70x70xf64>
    %2 = toy.matmul %0, %1 : tensor<70x70xf64>, tensor<70x70xf64> to tensor<70x70xf64>
    toy.print %2 : tensor<70x70xf64>
    toy.return
  }
}

// CHECK-LABEL: func.func @main()
// CHECK-NOT:     toy.matmul
// CHECK-DAG:     [[C0:%.*]] = arith.constant 0 : index
// CHECK-DAG:     [[C1:%.*]] = arith.constant 1 : index
// CHECK-DAG:     [[C16:%.*]] = arith.constant 16 : index
// CHECK-DAG:     [[C70:%.*]] = arith.constant 70 : index
// CHECK:         gpu.launch
// CHECK:               workgroup({{%.*}} : memref<16x16xf64, #gpu.address_space<workgroup>>, {{%.*}} : memref<16x16xf64, #gpu.address_space<workgroup>>)
// CHECK:           [[SUM:%.*]] = scf.for [[KO:%.*]] = [[C0]] to [[C70]] step [[C16]]
// CHECK-SAME:          iter_args
// CHECK-SAME:          -> (f64) {
// CHECK:             [[LHS_K:%.*]] = arith.addi [[KO]], {{%.*}} : index
// CHECK:             [[RHS_K:%.*]] = arith.addi [[KO]], {{%.*}} : index
// CHECK:             arith.cmpi ult, [[LHS_K]], [[C70]] : index
// CHECK:             arith.cmpi ult, [[RHS_K]], [[C70]] : index
// CHECK:             scf.if {{.*}} -> (f64) {
// CHECK:               memref.load
// CHECK:               scf.yield
// CHECK:             } else {
// CHECK:               scf.yield
// CHECK:             }
// CHECK:             scf.if {{.*}} -> (f64) {
// CHECK:               memref.load
// CHECK:               scf.yield
// CHECK:             } else {
// CHECK:               scf.yield
// CHECK:             }
// CHECK:             gpu.barrier
// CHECK:             scf.for {{%.*}} = [[C0]] to [[C16]] step [[C1]]
// CHECK:             gpu.barrier
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           memref.store [[SUM]]
