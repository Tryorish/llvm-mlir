//====- MatMulToSCF.cpp - Lower Toy matmul to naive SCF loops -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a partial lowering of Toy operations to memref, scf,
// arith, and func operations. The matmul lowering intentionally produces a
// straightforward i/j/k loop nest so later refactor stages have a simple shape
// to tile, reorder, and map to GPU.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "toy/Dialect.h"
#include "toy/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>

using namespace mlir;

//===----------------------------------------------------------------------===//
// ToyToSCF RewritePatterns
//===----------------------------------------------------------------------===//

static MemRefType convertTensorToMemRef(RankedTensorType type) {
  return MemRefType::get(type.getShape(), type.getElementType());
}

static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter) {
  auto alloc = rewriter.create<memref::AllocOp>(loc, type);

  auto *parentBlock = alloc->getBlock();
  alloc->moveBefore(&parentBlock->front());

  auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
  dealloc->moveBefore(&parentBlock->back());
  return alloc;
}

using LoopIterationFn = function_ref<Value(
    OpBuilder &rewriter, ValueRange memRefOperands, ValueRange loopIvs)>;

static void buildSCFLoopNest(OpBuilder &builder, Location loc,
                             ArrayRef<int64_t> upperBounds,
                             SmallVectorImpl<Value> &ivs,
                             function_ref<void(OpBuilder &)> buildBody) {
  if (ivs.size() == upperBounds.size()) {
    buildBody(builder);
    return;
  }

  Value lower = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value upper =
      builder.create<arith::ConstantIndexOp>(loc, upperBounds[ivs.size()]);
  Value step = builder.create<arith::ConstantIndexOp>(loc, 1);
  builder.create<scf::ForOp>(
      loc, lower, upper, step, ValueRange{},
      [&](OpBuilder &nestedBuilder, Location loc, Value iv,
          ValueRange iterArgs) {
        ivs.push_back(iv);
        buildSCFLoopNest(nestedBuilder, loc, upperBounds, ivs, buildBody);
        ivs.pop_back();
        nestedBuilder.create<scf::YieldOp>(loc);
      });
}

static void lowerOpToLoops(Operation *op, ValueRange operands,
                           PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {
  auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
  auto loc = op->getLoc();

  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

  SmallVector<Value, 4> ivs;
  buildSCFLoopNest(
      rewriter, loc, tensorType.getShape(), ivs, [&](OpBuilder &nestedBuilder) {
        Value valueToStore =
            processIteration(nestedBuilder, operands, ValueRange(ivs));
        nestedBuilder.create<memref::StoreOp>(loc, valueToStore, alloc,
                                              ValueRange(ivs));
      });

  rewriter.replaceOp(op, alloc);
}

namespace {

template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public ConversionPattern {
  BinaryOpLowering(MLIRContext *ctx)
      : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(op, operands, rewriter,
                   [loc](OpBuilder &builder, ValueRange memRefOperands,
                         ValueRange loopIvs) {
                     typename BinaryOp::Adaptor binaryAdaptor(memRefOperands);
                     auto loadedLhs = builder.create<memref::LoadOp>(
                         loc, binaryAdaptor.getLhs(), loopIvs);
                     auto loadedRhs = builder.create<memref::LoadOp>(
                         loc, binaryAdaptor.getRhs(), loopIvs);
                     return builder.create<LoweredBinaryOp>(loc, loadedLhs,
                                                            loadedRhs);
                   });
    return success();
  }
};
using AddOpLowering = BinaryOpLowering<toy::AddOp, arith::AddFOp>;
using MulOpLowering = BinaryOpLowering<toy::MulOp, arith::MulFOp>;

template <typename UnaryOp, typename LoweredUnaryOp>
struct UnaryOpLowering : public ConversionPattern {
  UnaryOpLowering(MLIRContext *ctx)
      : ConversionPattern(UnaryOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(op, operands, rewriter,
                   [loc](OpBuilder &builder, ValueRange memRefOperands,
                         ValueRange loopIvs) {
                     typename UnaryOp::Adaptor unaryAdaptor(memRefOperands);
                     auto loadedInput = builder.create<memref::LoadOp>(
                         loc, unaryAdaptor.getInput(), loopIvs);
                     return builder.create<LoweredUnaryOp>(loc, loadedInput);
                   });
    return success();
  }
};
using NegOpLowering = UnaryOpLowering<toy::NegOp, arith::NegFOp>;

struct ConstantOpLowering : public OpRewritePattern<toy::ConstantOp> {
  using OpRewritePattern<toy::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(toy::ConstantOp op,
                                PatternRewriter &rewriter) const final {
    DenseElementsAttr constantValue = op.getValue();
    Location loc = op.getLoc();

    auto tensorType = llvm::cast<RankedTensorType>(op.getType());
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    auto valueShape = memRefType.getShape();
    SmallVector<Value, 8> constantIndices;
    if (!valueShape.empty()) {
      for (auto i : llvm::seq<int64_t>(0, *llvm::max_element(valueShape)))
        constantIndices.push_back(
            rewriter.create<arith::ConstantIndexOp>(loc, i));
    } else {
      constantIndices.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, 0));
    }

    SmallVector<Value, 2> indices;
    auto valueIt = constantValue.value_begin<FloatAttr>();
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      if (dimension == valueShape.size()) {
        rewriter.create<memref::StoreOp>(
            loc, rewriter.create<arith::ConstantOp>(loc, *valueIt++), alloc,
            llvm::ArrayRef(indices));
        return;
      }

      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };

    storeElements(/*dimension=*/0);

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

struct FuncOpLowering : public OpConversionPattern<toy::FuncOp> {
  using OpConversionPattern<toy::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (op.getName() != "main")
      return failure();

    if (op.getNumArguments() || op.getFunctionType().getNumResults()) {
      return rewriter.notifyMatchFailure(op, [](Diagnostic &diag) {
        diag << "expected 'main' to have 0 inputs and 0 results";
      });
    }

    auto func = rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getName(),
                                                    op.getFunctionType());
    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
    rewriter.eraseOp(op);
    return success();
  }
};

struct PrintOpLowering : public OpConversionPattern<toy::PrintOp> {
  using OpConversionPattern<toy::PrintOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.modifyOpInPlace(op,
                             [&] { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

struct ReturnOpLowering : public OpRewritePattern<toy::ReturnOp> {
  using OpRewritePattern<toy::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(toy::ReturnOp op,
                                PatternRewriter &rewriter) const final {
    if (op.hasOperand())
      return failure();

    rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
    return success();
  }
};

struct MatMulOpLowering : public ConversionPattern {
  MatMulOpLowering(MLIRContext *ctx)
      : ConversionPattern(toy::MatMulOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    auto resultType = llvm::cast<RankedTensorType>(op->getResult(0).getType());
    auto memRefType = convertTensorToMemRef(resultType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    auto lhsType = llvm::cast<MemRefType>(operands[0].getType());
    auto rhsType = llvm::cast<MemRefType>(operands[1].getType());
    int64_t m = lhsType.getShape()[0];
    int64_t k = lhsType.getShape()[1];
    int64_t n = rhsType.getShape()[1];

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value mVal = rewriter.create<arith::ConstantIndexOp>(loc, m);
    Value nVal = rewriter.create<arith::ConstantIndexOp>(loc, n);
    Value kVal = rewriter.create<arith::ConstantIndexOp>(loc, k);
    Value zero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getF64FloatAttr(0.0));

    rewriter.create<scf::ForOp>(
        loc, c0, mVal, c1, ValueRange{},
        [&](OpBuilder &iBuilder, Location loc, Value ivI,
            ValueRange iterArgs) {
          iBuilder.create<scf::ForOp>(
              loc, c0, nVal, c1, ValueRange{},
              [&](OpBuilder &jBuilder, Location loc, Value ivJ,
                  ValueRange iterArgs) {
                auto forK = jBuilder.create<scf::ForOp>(
                    loc, c0, kVal, c1, ValueRange{zero},
                    [&](OpBuilder &kBuilder, Location loc, Value ivK,
                        ValueRange iterArgs) {
                      Value acc = iterArgs[0];
                      Value lhs = kBuilder.create<memref::LoadOp>(
                          loc, operands[0], ValueRange{ivI, ivK});
                      Value rhs = kBuilder.create<memref::LoadOp>(
                          loc, operands[1], ValueRange{ivK, ivJ});
                      Value prod =
                          kBuilder.create<arith::MulFOp>(loc, lhs, rhs);
                      Value sum =
                          kBuilder.create<arith::AddFOp>(loc, acc, prod);
                      kBuilder.create<scf::YieldOp>(loc, sum);
                    });
                jBuilder.create<memref::StoreOp>(
                    loc, forK.getResult(0), alloc, ValueRange{ivI, ivJ});
                jBuilder.create<scf::YieldOp>(loc);
              });
          iBuilder.create<scf::YieldOp>(loc);
        });

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

struct TransposeOpLowering : public ConversionPattern {
  TransposeOpLowering(MLIRContext *ctx)
      : ConversionPattern(toy::TransposeOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(op, operands, rewriter,
                   [loc](OpBuilder &builder, ValueRange memRefOperands,
                         ValueRange loopIvs) {
                     toy::TransposeOpAdaptor transposeAdaptor(memRefOperands);
                     Value input = transposeAdaptor.getInput();
                     SmallVector<Value, 2> reverseIvs(llvm::reverse(loopIvs));
                     return builder.create<memref::LoadOp>(loc, input,
                                                           reverseIvs);
                   });
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// ToyMatMulToSCFPass
//===----------------------------------------------------------------------===//

namespace {
struct ToyMatMulToSCFPass
    : public PassWrapper<ToyMatMulToSCFPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyMatMulToSCFPass)
  StringRef getArgument() const override { return "toy-matmul-to-scf"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void ToyMatMulToSCFPass::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<BuiltinDialect, arith::ArithDialect,
                         func::FuncDialect, memref::MemRefDialect,
                         scf::SCFDialect>();

  target.addIllegalDialect<toy::ToyDialect>();
  target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
    return llvm::none_of(op->getOperandTypes(),
                         [](Type type) { return llvm::isa<TensorType>(type); });
  });

  RewritePatternSet patterns(&getContext());
  patterns.add<AddOpLowering, ConstantOpLowering, FuncOpLowering, MulOpLowering,
               MatMulOpLowering, NegOpLowering, PrintOpLowering,
               ReturnOpLowering, TransposeOpLowering>(&getContext());

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::toy::createMatMulToSCFPass() {
  return std::make_unique<ToyMatMulToSCFPass>();
}
