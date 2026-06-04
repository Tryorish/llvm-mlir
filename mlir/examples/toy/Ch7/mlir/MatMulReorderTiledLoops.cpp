//====- MatMulReorderTiledLoops.cpp - Reorder tiled matmul loops ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the fourth Toy GPU refactor stage. It rewrites the
// split tiled matmul loop nest from io/jo/ko/ii/ji/ki order into the
// output-point accumulation order io/jo/ii/ji/ko/ki. The output is still
// host-side SCF/memref IR and intentionally does not introduce GPU operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "toy/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Casting.h"
#include <memory>

using namespace mlir;

namespace {

static bool hasConstantIndex(Value value, int64_t expected) {
  APInt intValue;
  return matchPattern(value, m_ConstantInt(&intValue)) &&
         intValue.getSExtValue() == expected;
}

static Operation *firstNonTerminator(Block *block) {
  for (Operation &op : block->without_terminator())
    return &op;
  return nullptr;
}

static Operation *lastNonTerminator(Block *block) {
  Operation *last = nullptr;
  for (Operation &op : block->without_terminator())
    last = &op;
  return last;
}

static scf::ForOp getOnlyNestedFor(scf::ForOp parent) {
  Operation *onlyOp = nullptr;
  for (Operation &op : parent.getBody()->without_terminator()) {
    if (onlyOp)
      return nullptr;
    onlyOp = &op;
  }
  if (!onlyOp)
    return nullptr;
  return dyn_cast<scf::ForOp>(onlyOp);
}

static scf::ForOp getLastNestedFor(scf::ForOp parent) {
  return dyn_cast_or_null<scf::ForOp>(lastNonTerminator(parent.getBody()));
}

static scf::IfOp getLastNestedIf(scf::ForOp parent) {
  return dyn_cast_or_null<scf::IfOp>(lastNonTerminator(parent.getBody()));
}

static scf::IfOp firstIfInBlock(Block *block) {
  for (Operation &op : block->without_terminator())
    if (auto ifOp = dyn_cast<scf::IfOp>(&op))
      return ifOp;
  return nullptr;
}

static scf::IfOp nextIfAfter(Operation *op) {
  for (Operation *next = op ? op->getNextNode() : nullptr; next;
       next = next->getNextNode())
    if (auto ifOp = dyn_cast<scf::IfOp>(next))
      return ifOp;
  return nullptr;
}

static Value createInBoundsLoadOrZero(OpBuilder &builder, Location loc,
                                      Value memref, ValueRange indices,
                                      Value inBounds, Value zero,
                                      Type elementType) {
  auto loadIf = builder.create<scf::IfOp>(
      loc, TypeRange{elementType}, inBounds,
      /*withElseRegion=*/true);

  OpBuilder thenBuilder = loadIf.getThenBodyBuilder(builder.getListener());
  Value loaded = thenBuilder.create<memref::LoadOp>(loc, memref, indices);
  thenBuilder.create<scf::YieldOp>(loc, loaded);

  OpBuilder elseBuilder = loadIf.getElseBodyBuilder(builder.getListener());
  elseBuilder.create<scf::YieldOp>(loc, zero);
  return loadIf.getResult(0);
}

static Value createTiledReductionForPoint(OpBuilder &builder, Location loc,
                                          Value lhs, Value rhs, Value i,
                                          Value j, Value inM, Value inN,
                                          Value kVal, Value c0, Value c1,
                                          Value c16, Value zero,
                                          Type elementType) {
  auto forKO = builder.create<scf::ForOp>(
      loc, c0, kVal, c16, ValueRange{zero},
      [&](OpBuilder &koBuilder, Location loc, Value ko, ValueRange iterArgs) {
        Value tileAcc = iterArgs[0];
        auto forKI = koBuilder.create<scf::ForOp>(
            loc, c0, c16, c1, ValueRange{tileAcc},
            [&](OpBuilder &kiBuilder, Location loc, Value ki,
                ValueRange iterArgs) {
              Value acc = iterArgs[0];
              Value k = kiBuilder.create<arith::AddIOp>(loc, ko, ki);
              Value inK = kiBuilder.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::ult, k, kVal);
              Value lhsInBounds =
                  kiBuilder.create<arith::AndIOp>(loc, inM, inK);
              Value rhsInBounds =
                  kiBuilder.create<arith::AndIOp>(loc, inK, inN);
              Value lhsValue =
                  createInBoundsLoadOrZero(kiBuilder, loc, lhs,
                                           ValueRange{i, k}, lhsInBounds, zero,
                                           elementType);
              Value rhsValue =
                  createInBoundsLoadOrZero(kiBuilder, loc, rhs,
                                           ValueRange{k, j}, rhsInBounds, zero,
                                           elementType);
              Value prod =
                  kiBuilder.create<arith::MulFOp>(loc, lhsValue, rhsValue);
              Value sum = kiBuilder.create<arith::AddFOp>(loc, acc, prod);
              kiBuilder.create<scf::YieldOp>(loc, sum);
            });
        koBuilder.create<scf::YieldOp>(loc, forKI.getResult(0));
      });

  return forKO.getResult(0);
}

struct SplitTiledMatMulLoopNest {
  scf::ForOp forIO;
  scf::ForOp forJO;
  scf::ForOp forKO;
  scf::ForOp forII;
  scf::ForOp forJI;
  scf::ForOp forKI;
  memref::LoadOp lhsLoad;
  memref::LoadOp rhsLoad;
  memref::StoreOp store;
  Value zero;
};

static FailureOr<SplitTiledMatMulLoopNest>
matchSplitTiledMatMul(scf::ForOp forIO) {
  if (!hasConstantIndex(forIO.getLowerBound(), 0) ||
      !hasConstantIndex(forIO.getStep(), 16) || forIO.getNumResults() != 0)
    return failure();

  scf::ForOp forJO = getOnlyNestedFor(forIO);
  if (!forJO || !hasConstantIndex(forJO.getLowerBound(), 0) ||
      !hasConstantIndex(forJO.getStep(), 16) || forJO.getNumResults() != 0)
    return failure();

  scf::ForOp forKO = getOnlyNestedFor(forJO);
  if (!forKO || !hasConstantIndex(forKO.getLowerBound(), 0) ||
      !hasConstantIndex(forKO.getStep(), 16) || forKO.getNumResults() != 0)
    return failure();

  scf::ForOp forII = getOnlyNestedFor(forKO);
  if (!forII || !hasConstantIndex(forII.getLowerBound(), 0) ||
      !hasConstantIndex(forII.getUpperBound(), 16) ||
      !hasConstantIndex(forII.getStep(), 1) || forII.getNumResults() != 0)
    return failure();

  scf::ForOp forJI = getLastNestedFor(forII);
  if (!forJI || !hasConstantIndex(forJI.getLowerBound(), 0) ||
      !hasConstantIndex(forJI.getUpperBound(), 16) ||
      !hasConstantIndex(forJI.getStep(), 1) || forJI.getNumResults() != 0)
    return failure();

  scf::IfOp storeGuard = getLastNestedIf(forJI);
  if (!storeGuard || storeGuard.getNumResults() != 0)
    return failure();

  scf::IfOp accInitIf = firstIfInBlock(storeGuard.thenBlock());
  if (!accInitIf || accInitIf.getNumResults() != 1)
    return failure();

  Value zero = nullptr;
  if (Operation *thenYield = accInitIf.thenBlock()->getTerminator()) {
    auto yield = dyn_cast<scf::YieldOp>(thenYield);
    if (!yield || yield.getNumOperands() != 1)
      return failure();
    zero = yield.getOperand(0);
  }
  if (!zero)
    return failure();

  Operation *forKIOp = accInitIf->getNextNode();
  auto forKI = dyn_cast_or_null<scf::ForOp>(forKIOp);
  if (!forKI || !hasConstantIndex(forKI.getLowerBound(), 0) ||
      !hasConstantIndex(forKI.getUpperBound(), 16) ||
      !hasConstantIndex(forKI.getStep(), 1) || forKI.getNumResults() != 1 ||
      forKI.getInitArgs().size() != 1)
    return failure();

  Operation *lastThenOp = lastNonTerminator(storeGuard.thenBlock());
  auto store = dyn_cast_or_null<memref::StoreOp>(lastThenOp);
  if (!store || store.getValueToStore() != forKI.getResult(0))
    return failure();

  scf::IfOp lhsIf = firstIfInBlock(forKI.getBody());
  if (!lhsIf || lhsIf.getNumResults() != 1)
    return failure();
  auto lhsLoad = dyn_cast_or_null<memref::LoadOp>(
      firstNonTerminator(lhsIf.thenBlock()));
  if (!lhsLoad || lhsLoad.getIndices().size() != 2)
    return failure();

  scf::IfOp rhsIf = nextIfAfter(lhsIf);
  if (!rhsIf || rhsIf.getNumResults() != 1)
    return failure();
  auto rhsLoad = dyn_cast_or_null<memref::LoadOp>(
      firstNonTerminator(rhsIf.thenBlock()));
  if (!rhsLoad || rhsLoad.getIndices().size() != 2)
    return failure();

  if (store.getMemRef() == lhsLoad.getMemRef() ||
      store.getMemRef() == rhsLoad.getMemRef())
    return failure();

  return SplitTiledMatMulLoopNest{forIO,  forJO,  forKO, forII, forJI,
                                  forKI, lhsLoad, rhsLoad, store, zero};
}

struct ReorderTiledMatMulLoopPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forIO,
                                PatternRewriter &rewriter) const final {
    FailureOr<SplitTiledMatMulLoopNest> maybeNest =
        matchSplitTiledMatMul(forIO);
    if (failed(maybeNest))
      return failure();

    SplitTiledMatMulLoopNest nest = *maybeNest;
    Location loc = forIO.getLoc();
    Type elementType = nest.zero.getType();

    rewriter.setInsertionPoint(forIO);
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value c16 = rewriter.create<arith::ConstantIndexOp>(loc, 16);

    Value mVal = nest.forIO.getUpperBound();
    Value nVal = nest.forJO.getUpperBound();
    Value kVal = nest.forKO.getUpperBound();
    Value lhs = nest.lhsLoad.getMemRef();
    Value rhs = nest.rhsLoad.getMemRef();
    Value out = nest.store.getMemRef();

    rewriter.create<scf::ForOp>(
        loc, c0, mVal, c16, ValueRange{},
        [&](OpBuilder &ioBuilder, Location loc, Value io,
            ValueRange iterArgs) {
          ioBuilder.create<scf::ForOp>(
              loc, c0, nVal, c16, ValueRange{},
              [&](OpBuilder &joBuilder, Location loc, Value jo,
                  ValueRange iterArgs) {
                joBuilder.create<scf::ForOp>(
                    loc, c0, c16, c1, ValueRange{},
                    [&](OpBuilder &iiBuilder, Location loc, Value ii,
                        ValueRange iterArgs) {
                      Value i =
                          iiBuilder.create<arith::AddIOp>(loc, io, ii);
                      Value inM = iiBuilder.create<arith::CmpIOp>(
                          loc, arith::CmpIPredicate::ult, i, mVal);

                      iiBuilder.create<scf::ForOp>(
                          loc, c0, c16, c1, ValueRange{},
                          [&](OpBuilder &jiBuilder, Location loc, Value ji,
                              ValueRange iterArgs) {
                            Value j =
                                jiBuilder.create<arith::AddIOp>(loc, jo, ji);
                            Value inN = jiBuilder.create<arith::CmpIOp>(
                                loc, arith::CmpIPredicate::ult, j, nVal);
                            Value inMN =
                                jiBuilder.create<arith::AndIOp>(loc, inM, inN);

                            jiBuilder.create<scf::IfOp>(
                                loc, inMN,
                                [&](OpBuilder &thenBuilder, Location loc) {
                                  Value sum = createTiledReductionForPoint(
                                      thenBuilder, loc, lhs, rhs, i, j, inM,
                                      inN, kVal, c0, c1, c16, nest.zero,
                                      elementType);
                                  thenBuilder.create<memref::StoreOp>(
                                      loc, sum, out, ValueRange{i, j});
                                  thenBuilder.create<scf::YieldOp>(loc,
                                                                   ValueRange{});
                                });
                            jiBuilder.create<scf::YieldOp>(loc);
                          });
                      iiBuilder.create<scf::YieldOp>(loc);
                    });
                joBuilder.create<scf::YieldOp>(loc);
              });
          ioBuilder.create<scf::YieldOp>(loc);
        });

    rewriter.eraseOp(forIO);
    return success();
  }
};

struct MatMulReorderTiledLoopsPass
    : public PassWrapper<MatMulReorderTiledLoopsPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatMulReorderTiledLoopsPass)
  StringRef getArgument() const override {
    return "toy-matmul-reorder-tiled-loops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<ReorderTiledMatMulLoopPattern>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::toy::createMatMulReorderTiledLoopsPass() {
  return std::make_unique<MatMulReorderTiledLoopsPass>();
}
