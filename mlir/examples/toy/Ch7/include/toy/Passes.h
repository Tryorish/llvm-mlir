//===- Passes.h - Toy Passes Definition -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file exposes the entry points to create compiler passes for Toy.
//
//===----------------------------------------------------------------------===//

#ifndef TOY_PASSES_H
#define TOY_PASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace toy {
std::unique_ptr<Pass> createShapeInferencePass();

/// Create a pass for lowering to operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<mlir::Pass> createLowerToAffinePass();

/// Create a pass for lowering `toy.matmul` to a naive `scf.for` loop nest.
std::unique_ptr<mlir::Pass> createMatMulToSCFPass();

/// Create a pass for splitting the naive matmul loop nest into tile and point
/// loops.
std::unique_ptr<mlir::Pass> createMatMulTileLoopsPass();

/// Create a pass for lowering to operations in the `GPU` dialect,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<mlir::Pass> createLowerToGPUPass();

/// Create a pass for lowering operations the remaining `Toy` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

/// Create a pass for lowering only `toy.print` to loops and LLVM `printf`.
std::unique_ptr<mlir::Pass> createLowerPrintToLLVMPass();

} // namespace toy
} // namespace mlir

#endif // TOY_PASSES_H
