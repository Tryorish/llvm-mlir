// RUN: toyc-ch7 %s -emit=mlir-tiled-matmul -x=mlir 2>&1 | FileCheck %s

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
// CHECK-NOT:     gpu.launch
// CHECK-NOT:     workgroup
// CHECK-DAG:     [[C0:%.*]] = arith.constant 0 : index
// CHECK-DAG:     [[C1:%.*]] = arith.constant 1 : index
// CHECK-DAG:     [[C16:%.*]] = arith.constant 16 : index
// CHECK-DAG:     [[C70:%.*]] = arith.constant 70 : index
// CHECK:         [[OUT:%.*]] = memref.alloc() : memref<70x70xf64>
// CHECK:         scf.for [[IO:%.*]] = [[C0]] to [[C70]] step [[C16]] {
// CHECK:           scf.for [[JO:%.*]] = [[C0]] to [[C70]] step [[C16]] {
// CHECK:             scf.for [[KO:%.*]] = [[C0]] to [[C70]] step [[C16]] {
// CHECK:               scf.for [[II:%.*]] = [[C0]] to [[C16]] step [[C1]] {
// CHECK:                 [[I:%.*]] = arith.addi [[IO]], [[II]] : index
// CHECK:                 [[IN_M:%.*]] = arith.cmpi ult, [[I]], [[C70]] : index
// CHECK:                 scf.for [[JI:%.*]] = [[C0]] to [[C16]] step [[C1]] {
// CHECK:                   [[J:%.*]] = arith.addi [[JO]], [[JI]] : index
// CHECK:                   [[IN_N:%.*]] = arith.cmpi ult, [[J]], [[C70]] : index
// CHECK:                   [[IN_MN:%.*]] = arith.andi [[IN_M]], [[IN_N]] : i1
// CHECK:                   scf.if [[IN_MN]] {
// CHECK:                     scf.for [[KI:%.*]] = [[C0]] to [[C16]] step [[C1]]
// CHECK-SAME:                  iter_args
// CHECK-SAME:                  -> (f64) {
// CHECK:                         [[K:%.*]] = arith.addi [[KO]], [[KI]] : index
// CHECK:                         [[IN_K:%.*]] = arith.cmpi ult, [[K]], [[C70]] : index
// CHECK:                         arith.andi {{.*}}, [[IN_K]] : i1
// CHECK:                         arith.andi [[IN_K]], {{.*}} : i1
// CHECK:                         scf.yield
// CHECK:                       }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:         }
// CHECK:         toy.print [[OUT]] : memref<70x70xf64>
