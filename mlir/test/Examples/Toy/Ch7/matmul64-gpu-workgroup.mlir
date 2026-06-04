// RUN: toyc-ch7 %s -emit=mlir-gpu-workgroup-matmul -x=mlir 2>&1 | FileCheck %s

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
// CHECK-DAG:     [[C0:%.*]] = arith.constant 0 : index
// CHECK-DAG:     [[C1:%.*]] = arith.constant 1 : index
// CHECK-DAG:     [[C16:%.*]] = arith.constant 16 : index
// CHECK-DAG:     [[C64:%.*]] = arith.constant 64 : index
// CHECK:         [[OUT:%.*]] = memref.alloc() : memref<64x64xf64>
// CHECK:         gpu.launch blocks([[BX:%.*]], [[BY:%.*]], {{%.*}}) in
// CHECK-SAME:               threads([[TX:%.*]], [[TY:%.*]], {{%.*}}) in
// CHECK:               workgroup([[TILE_A:%.*]] : memref<16x16xf64, #gpu.address_space<workgroup>>, [[TILE_B:%.*]] : memref<16x16xf64, #gpu.address_space<workgroup>>)
// CHECK:           [[J_BLOCK:%.*]] = arith.muli [[BX]], [[C16]] : index
// CHECK:           [[J:%.*]] = arith.addi [[J_BLOCK]], [[TX]] : index
// CHECK:           [[I_BLOCK:%.*]] = arith.muli [[BY]], [[C16]] : index
// CHECK:           [[I:%.*]] = arith.addi [[I_BLOCK]], [[TY]] : index
// CHECK:           [[SUM:%.*]] = scf.for [[KO:%.*]] = [[C0]] to [[C64]] step [[C16]]
// CHECK-SAME:          iter_args
// CHECK-SAME:          -> (f64) {
// CHECK:             [[LHS_K:%.*]] = arith.addi [[KO]], [[TX]] : index
// CHECK:             [[RHS_K:%.*]] = arith.addi [[KO]], [[TY]] : index
// CHECK:             memref.load {{.*}}[[I]], [[LHS_K]]{{.*}} : memref<64x64xf64>
// CHECK:             memref.load {{.*}}[[RHS_K]], [[J]]{{.*}} : memref<64x64xf64>
// CHECK:             memref.store {{.*}}, [[TILE_A]][[[TY]], [[TX]]] : memref<16x16xf64, #gpu.address_space<workgroup>>
// CHECK:             memref.store {{.*}}, [[TILE_B]][[[TY]], [[TX]]] : memref<16x16xf64, #gpu.address_space<workgroup>>
// CHECK:             gpu.barrier
// CHECK:             scf.for [[KI:%.*]] = [[C0]] to [[C16]] step [[C1]]
// CHECK-SAME:            iter_args
// CHECK-SAME:            -> (f64) {
// CHECK:               memref.load [[TILE_A]][[[TY]], [[KI]]] : memref<16x16xf64, #gpu.address_space<workgroup>>
// CHECK:               memref.load [[TILE_B]][[[KI]], [[TX]]] : memref<16x16xf64, #gpu.address_space<workgroup>>
// CHECK:               arith.mulf
// CHECK:               arith.addf
// CHECK:               scf.yield
// CHECK:             }
// CHECK:             gpu.barrier
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           memref.store [[SUM]], [[OUT]][[[I]], [[J]]] : memref<64x64xf64>
// CHECK:           gpu.terminator
// CHECK:         toy.print [[OUT]] : memref<64x64xf64>
