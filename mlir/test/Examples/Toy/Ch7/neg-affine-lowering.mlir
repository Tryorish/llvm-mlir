// RUN: toyc-ch7 %s -emit=mlir-affine 2>&1 | FileCheck %s

toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>
  %1 = toy.neg %0 : tensor<2x2xf64> to tensor<2x2xf64>
  toy.print %1 : tensor<2x2xf64>
  toy.return
}

// CHECK-LABEL: func @main()
// CHECK:         [[OUT:%.*]] = memref.alloc() : memref<2x2xf64>
// CHECK:         [[IN:%.*]] = memref.alloc() : memref<2x2xf64>
// CHECK:         affine.for [[I:%.*]] = 0 to 2 {
// CHECK:           affine.for [[J:%.*]] = 0 to 2 {
// CHECK:             [[VAL:%.*]] = affine.load [[IN]]{{\[}}[[I]], [[J]]] : memref<2x2xf64>
// CHECK:             [[NEG:%.*]] = arith.negf [[VAL]] : f64
// CHECK:             affine.store [[NEG]], [[OUT]]{{\[}}[[I]], [[J]]] : memref<2x2xf64>
// CHECK:         toy.print [[OUT]] : memref<2x2xf64>