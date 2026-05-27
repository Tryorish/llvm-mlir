// RUN: toyc-ch7 %s -emit=mlir-affine -x=mlir 2>&1 | FileCheck %s

module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>
    %1 = toy.reshape(%0 : tensor<2x2xf64>) to tensor<2x2xf64>
    %2 = toy.constant dense<[[5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00]]> : tensor<2x2xf64>
    %3 = toy.reshape(%2 : tensor<2x2xf64>) to tensor<2x2xf64>
    %4 = toy.matmul %1, %3 : tensor<2x2xf64>, tensor<2x2xf64> to tensor<2x2xf64>
    toy.print %4 : tensor<2x2xf64>
    toy.return
  }
}

// CHECK-LABEL: func.func @main()
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f64
// CHECK: %[[OUT:.*]] = memref.alloc() : memref<2x2xf64>
// CHECK: affine.for %[[I:.*]] = 0 to 2 {
// CHECK: affine.for %[[J:.*]] = 0 to 2 {
// CHECK: %[[SUM:.*]] = affine.for %[[K:.*]] = 0 to 2 iter_args(%[[ACC:.*]] = %[[ZERO]]) -> (f64) {
// CHECK: %[[A:.*]] = affine.load
// CHECK: %[[B:.*]] = affine.load
// CHECK: %[[MUL:.*]] = arith.mulf %[[A]], %[[B]] : f64
// CHECK: %[[ADD:.*]] = arith.addf %[[ACC]], %[[MUL]] : f64
// CHECK: affine.yield %[[ADD]] : f64
// CHECK: affine.store %[[SUM]], %[[OUT]]