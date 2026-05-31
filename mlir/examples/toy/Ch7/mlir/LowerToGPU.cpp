//====- LowerToGPU.cpp - Partial lowering from Toy to GPU -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a partial lowering of Toy operations to a combination of
// gpu.launch, memref operations, scf control flow and standard operations. This
// lowering expects that all calls have been inlined, and all shapes have been
// resolved.
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

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
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
#include <utility>

using namespace mlir;

//===----------------------------------------------------------------------===//
// ToyToGPU RewritePatterns
//===----------------------------------------------------------------------===//

/// Convert the given RankedTensorType into the corresponding MemRefType.
static MemRefType convertTensorToMemRef(RankedTensorType type) {
  return MemRefType::get(type.getShape(), type.getElementType());
}

/// Insert an allocation and deallocation for the given MemRefType.
static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter) {
  auto alloc = rewriter.create<memref::AllocOp>(loc, type);

  // Make sure to allocate at the beginning of the block.
  auto *parentBlock = alloc->getBlock();
  alloc->moveBefore(&parentBlock->front());

  // Make sure to deallocate this alloc at the end of the block. This is fine
  // as toy functions have no control flow.
  auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
  dealloc->moveBefore(&parentBlock->back());
  return alloc;
}

/// This defines the function type used to process an iteration of a lowered
/// loop.
using LoopIterationFn = function_ref<Value(
    OpBuilder &rewriter, ValueRange memRefOperands, ValueRange loopIvs)>;

static void lowerOpToLoops(Operation *op, ValueRange operands,
                           PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {
  auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
  auto loc = op->getLoc();

  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

  SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
  SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);
  affine::buildAffineLoopNest(
      rewriter, loc, lowerBounds, tensorType.getShape(), steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        Value valueToStore = processIteration(nestedBuilder, operands, ivs);
        nestedBuilder.create<affine::AffineStoreOp>(loc, valueToStore, alloc,
                                                    ivs);
      });

  rewriter.replaceOp(op, alloc);
}

namespace {
//===----------------------------------------------------------------------===//
// ToyToGPU RewritePatterns: Binary operations
//===----------------------------------------------------------------------===//

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

                     auto loadedLhs = builder.create<affine::AffineLoadOp>(
                         loc, binaryAdaptor.getLhs(), loopIvs);
                     auto loadedRhs = builder.create<affine::AffineLoadOp>(
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
                     auto loadedInput = builder.create<affine::AffineLoadOp>(
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
        rewriter.create<affine::AffineStoreOp>(
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

//===----------------------------------------------------------------------===//
// ToyToGPU RewritePatterns: Func operations
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// ToyToGPU RewritePatterns: Print operations
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// ToyToGPU RewritePatterns: Return operations
//===----------------------------------------------------------------------===//

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
    Type asyncTokenType = gpu::AsyncTokenType::get(rewriter.getContext());
    Value token =
        rewriter.create<gpu::WaitOp>(loc, asyncTokenType, ValueRange{})
            .getAsyncToken();

    auto lhsDeviceAlloc = rewriter.create<gpu::AllocOp>(
        loc, lhsType, asyncTokenType, ValueRange{token}, ValueRange{},
        ValueRange{}, false);
    token = lhsDeviceAlloc.getAsyncToken();
    auto rhsDeviceAlloc = rewriter.create<gpu::AllocOp>(
        loc, rhsType, asyncTokenType, ValueRange{token}, ValueRange{},
        ValueRange{}, false);
    token = rhsDeviceAlloc.getAsyncToken();
    auto outDeviceAlloc = rewriter.create<gpu::AllocOp>(
        loc, memRefType, asyncTokenType, ValueRange{token}, ValueRange{},
        ValueRange{}, false);
    token = outDeviceAlloc.getAsyncToken();
    Value lhsDevice = lhsDeviceAlloc.getMemref();
    Value rhsDevice = rhsDeviceAlloc.getMemref();
    Value outDevice = outDeviceAlloc.getMemref();

    token = rewriter
                .create<gpu::MemcpyOp>(loc, asyncTokenType, ValueRange{token},
                                       lhsDevice, operands[0])
                .getAsyncToken();
    token = rewriter
                .create<gpu::MemcpyOp>(loc, asyncTokenType, ValueRange{token},
                                       rhsDevice, operands[1])
                .getAsyncToken();

    int64_t m = lhsType.getShape()[0];
    int64_t k = lhsType.getShape()[1];
    int64_t n = rhsType.getShape()[1];
    constexpr int64_t blockSize = 16;

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value mVal = rewriter.create<arith::ConstantIndexOp>(loc, m);
    Value nVal = rewriter.create<arith::ConstantIndexOp>(loc, n);
    Value kVal = rewriter.create<arith::ConstantIndexOp>(loc, k);
    Value blockX = rewriter.create<arith::ConstantIndexOp>(loc, blockSize);
    Value blockY = rewriter.create<arith::ConstantIndexOp>(loc, blockSize);
    Value gridX = rewriter.create<arith::ConstantIndexOp>(
        loc, (n + blockSize - 1) / blockSize);
    Value gridY = rewriter.create<arith::ConstantIndexOp>(
        loc, (m + blockSize - 1) / blockSize);
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64FloatAttr(0.0));

    auto workgroupAddrSpace =
        gpu::AddressSpaceAttr::get(rewriter.getContext(),
                                   gpu::AddressSpace::Workgroup);
    auto tileType = MemRefType::get({blockSize, blockSize},
                                    memRefType.getElementType(),
                                    MemRefLayoutAttrInterface{},
                                    Attribute(workgroupAddrSpace));
    SmallVector<Type, 2> tileTypes{tileType, tileType};
    auto launch = rewriter.create<gpu::LaunchOp>(
        loc, gridX, gridY, c1, blockX, blockY, c1,
        /*dynamicSharedMemorySize=*/Value(), asyncTokenType, ValueRange{token},
        tileTypes);
    token = launch.getAsyncToken();
    unsigned firstWorkgroupArg = launch.getNumConfigRegionAttributes();
    Value lhsTile = launch.getBody().getArgument(firstWorkgroupArg);
    Value rhsTile = launch.getBody().getArgument(firstWorkgroupArg + 1);
    gpu::KernelDim3 blockIds = launch.getBlockIds();
    gpu::KernelDim3 threadIds = launch.getThreadIds();

    rewriter.setInsertionPointToStart(&launch.getBody().front());
    Value jBlock = rewriter.create<arith::MulIOp>(loc, blockIds.x, blockX);
    Value j = rewriter.create<arith::AddIOp>(loc, jBlock, threadIds.x);
    Value iBlock = rewriter.create<arith::MulIOp>(loc, blockIds.y, blockY);
    Value i = rewriter.create<arith::AddIOp>(loc, iBlock, threadIds.y);

    Value inM =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, i, mVal);
    Value inN =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, j, nVal);
    Value inBounds = rewriter.create<arith::AndIOp>(loc, inM, inN);

    auto forKTile = rewriter.create<scf::ForOp>(
        loc, c0, kVal, blockX, ValueRange{zero},
        [&](OpBuilder &nestedBuilder, Location loc, Value ivK,
            ValueRange iterArgs) {
          Value acc = iterArgs[0];
          Value lhsK =
              nestedBuilder.create<arith::AddIOp>(loc, ivK, threadIds.x);
          Value rhsK =
              nestedBuilder.create<arith::AddIOp>(loc, ivK, threadIds.y);
          Value lhsInK = nestedBuilder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::ult, lhsK, kVal);
          Value lhsInBounds =
              nestedBuilder.create<arith::AndIOp>(loc, inM, lhsInK);
          Value rhsInK = nestedBuilder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::ult, rhsK, kVal);
          Value rhsInBounds =
              nestedBuilder.create<arith::AndIOp>(loc, rhsInK, inN);
          auto lhsIf = nestedBuilder.create<scf::IfOp>(
              loc, TypeRange{memRefType.getElementType()}, lhsInBounds,
              /*withElseRegion=*/true);
          OpBuilder lhsThenBuilder =
              lhsIf.getThenBodyBuilder(nestedBuilder.getListener());
          Value lhs = lhsThenBuilder.create<memref::LoadOp>(
              loc, lhsDevice, ValueRange{i, lhsK});
          lhsThenBuilder.create<scf::YieldOp>(loc, lhs);
          OpBuilder lhsElseBuilder =
              lhsIf.getElseBodyBuilder(nestedBuilder.getListener());
          lhsElseBuilder.create<scf::YieldOp>(loc, zero);

          auto rhsIf = nestedBuilder.create<scf::IfOp>(
              loc, TypeRange{memRefType.getElementType()}, rhsInBounds,
              /*withElseRegion=*/true);
          OpBuilder rhsThenBuilder =
              rhsIf.getThenBodyBuilder(nestedBuilder.getListener());
          Value rhs = rhsThenBuilder.create<memref::LoadOp>(
              loc, rhsDevice, ValueRange{rhsK, j});
          rhsThenBuilder.create<scf::YieldOp>(loc, rhs);
          OpBuilder rhsElseBuilder =
              rhsIf.getElseBodyBuilder(nestedBuilder.getListener());
          rhsElseBuilder.create<scf::YieldOp>(loc, zero);
          nestedBuilder.create<memref::StoreOp>(
              loc, lhsIf.getResult(0), lhsTile,
              ValueRange{threadIds.y, threadIds.x});
          nestedBuilder.create<memref::StoreOp>(
              loc, rhsIf.getResult(0), rhsTile,
              ValueRange{threadIds.y, threadIds.x});
          nestedBuilder.create<gpu::BarrierOp>(loc);

          auto forK = nestedBuilder.create<scf::ForOp>(
              loc, c0, blockX, c1, ValueRange{acc},
              [&](OpBuilder &innerBuilder, Location loc, Value ivInnerK,
                  ValueRange innerIterArgs) {
                Value innerAcc = innerIterArgs[0];
                Value tiledLhs = innerBuilder.create<memref::LoadOp>(
                    loc, lhsTile, ValueRange{threadIds.y, ivInnerK});
                Value tiledRhs = innerBuilder.create<memref::LoadOp>(
                    loc, rhsTile, ValueRange{ivInnerK, threadIds.x});
                Value prod =
                    innerBuilder.create<arith::MulFOp>(loc, tiledLhs, tiledRhs);
                Value sum =
                    innerBuilder.create<arith::AddFOp>(loc, innerAcc, prod);
                innerBuilder.create<scf::YieldOp>(loc, sum);
              });
          nestedBuilder.create<gpu::BarrierOp>(loc);
          nestedBuilder.create<scf::YieldOp>(loc, forK.getResult(0));
        });

    auto storeIf = rewriter.create<scf::IfOp>(loc, inBounds,
                                              /*withElseRegion=*/false);
    rewriter.setInsertionPointToStart(storeIf.thenBlock());
    rewriter.create<memref::StoreOp>(loc, forKTile.getResult(0), outDevice,
                                     ValueRange{i, j});
    rewriter.setInsertionPointToEnd(&launch.getBody().front());
    rewriter.create<gpu::TerminatorOp>(loc);

    rewriter.setInsertionPointAfter(launch);
    token = rewriter
                .create<gpu::MemcpyOp>(loc, asyncTokenType, ValueRange{token},
                                       alloc, outDevice)
                .getAsyncToken();
    token = rewriter
                .create<gpu::DeallocOp>(loc, asyncTokenType, ValueRange{token},
                                        lhsDevice)
                .getAsyncToken();
    token = rewriter
                .create<gpu::DeallocOp>(loc, asyncTokenType, ValueRange{token},
                                        rhsDevice)
                .getAsyncToken();
    token = rewriter
                .create<gpu::DeallocOp>(loc, asyncTokenType, ValueRange{token},
                                        outDevice)
                .getAsyncToken();
    rewriter.create<gpu::WaitOp>(loc, Type(), ValueRange{token});

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToGPU RewritePatterns: Transpose operations
//===----------------------------------------------------------------------===//

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
                     return builder.create<affine::AffineLoadOp>(loc, input,
                                                                 reverseIvs);
                   });
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// ToyToGPULoweringPass
//===----------------------------------------------------------------------===//

namespace {
struct ToyToGPULoweringPass
    : public PassWrapper<ToyToGPULoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyToGPULoweringPass)
  StringRef getArgument() const override { return "toy-to-gpu"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    func::FuncDialect, gpu::GPUDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void ToyToGPULoweringPass::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<affine::AffineDialect, BuiltinDialect,
                         arith::ArithDialect, func::FuncDialect,
                         gpu::GPUDialect, memref::MemRefDialect,
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

/// Create a pass for lowering operations in the Toy dialect to gpu.launch.
std::unique_ptr<Pass> mlir::toy::createLowerToGPUPass() {
  return std::make_unique<ToyToGPULoweringPass>();
}
