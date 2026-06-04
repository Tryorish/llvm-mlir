//====- MatMulTileLoops.cpp - Tile naive SCF matmul loops -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the third Toy GPU refactor stage. It rewrites the
// canonical naive SCF matmul loop nest produced by MatMulToSCF.cpp into a
// six-level tile/point loop nest. The output is still host-side SCF/memref IR
// and intentionally does not introduce GPU operations.
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
#include "llvm/ADT/STLExtras.h"
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

struct MatMulLoopNest {
  scf::ForOp forI;
  scf::ForOp forJ;
  scf::ForOp forK;
  memref::LoadOp lhsLoad;
  memref::LoadOp rhsLoad;
  memref::StoreOp store;
  Value output;
};

static FailureOr<MatMulLoopNest> matchNaiveMatMul(scf::ForOp forI) {
  if (!hasConstantIndex(forI.getLowerBound(), 0) ||
      !hasConstantIndex(forI.getStep(), 1) || forI.getNumResults() != 0)
    return failure();

  scf::ForOp forJ = getOnlyNestedFor(forI);
  if (!forJ || !hasConstantIndex(forJ.getLowerBound(), 0) ||
      !hasConstantIndex(forJ.getStep(), 1) || forJ.getNumResults() != 0)
    return failure();

  Operation *firstJBodyOp = firstNonTerminator(forJ.getBody());
  auto forK = dyn_cast_or_null<scf::ForOp>(firstJBodyOp);
  if (!forK || !hasConstantIndex(forK.getLowerBound(), 0) ||
      !hasConstantIndex(forK.getStep(), 1) || forK.getNumResults() != 1 ||
      forK.getInitArgs().size() != 1)
    return failure();

  Operation *lastJBodyOp = lastNonTerminator(forJ.getBody());
  auto store = dyn_cast_or_null<memref::StoreOp>(lastJBodyOp);
  if (!store || store.getValueToStore() != forK.getResult(0))
    return failure();
  if (store.getIndices().size() != 2 ||
      store.getIndices()[0] != forI.getInductionVar() ||
      store.getIndices()[1] != forJ.getInductionVar())
    return failure();

  auto lhsLoad = dyn_cast_or_null<memref::LoadOp>(
      firstNonTerminator(forK.getBody()));
  if (!lhsLoad || lhsLoad.getIndices().size() != 2 ||
      lhsLoad.getIndices()[0] != forI.getInductionVar() ||
      lhsLoad.getIndices()[1] != forK.getInductionVar())
    return failure();

  auto rhsLoad = dyn_cast_or_null<memref::LoadOp>(
      lhsLoad->getNextNode());
  if (!rhsLoad || rhsLoad.getIndices().size() != 2 ||
      rhsLoad.getIndices()[0] != forK.getInductionVar() ||
      rhsLoad.getIndices()[1] != forJ.getInductionVar())
    return failure();

  if (store.getMemRef() == lhsLoad.getMemRef() ||
      store.getMemRef() == rhsLoad.getMemRef())
    return failure();

  return MatMulLoopNest{forI, forJ, forK, lhsLoad, rhsLoad, store,
                        store.getMemRef()};
}

struct TileMatMulLoopPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forI,
                                PatternRewriter &rewriter) const final {
    FailureOr<MatMulLoopNest> maybeNest = matchNaiveMatMul(forI);
    if (failed(maybeNest))
      return failure();

    MatMulLoopNest nest = *maybeNest;
    Location loc = forI.getLoc();
    constexpr int64_t tileSize = 16;

    rewriter.setInsertionPoint(forI);
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value c16 = rewriter.create<arith::ConstantIndexOp>(loc, tileSize);
    Value zero = nest.forK.getInitArgs()[0];
    Type elementType = zero.getType();

    Value mVal = nest.forI.getUpperBound();
    Value nVal = nest.forJ.getUpperBound();
    Value kVal = nest.forK.getUpperBound();
    Value lhs = nest.lhsLoad.getMemRef();
    Value rhs = nest.rhsLoad.getMemRef();
    Value out = nest.output;

    rewriter.create<scf::ForOp>(
        loc, c0, mVal, c16, ValueRange{},
        [&](OpBuilder &ioBuilder, Location loc, Value io,
            ValueRange iterArgs) {
          ioBuilder.create<scf::ForOp>(
              loc, c0, nVal, c16, ValueRange{},
              [&](OpBuilder &joBuilder, Location loc, Value jo,
                  ValueRange iterArgs) {
                joBuilder.create<scf::ForOp>(
                    loc, c0, kVal, c16, ValueRange{},
                    [&](OpBuilder &koBuilder, Location loc, Value ko,
                        ValueRange iterArgs) {
                      koBuilder.create<scf::ForOp>(
                          loc, c0, c16, c1, ValueRange{},
                          [&](OpBuilder &iiBuilder, Location loc, Value ii,
                              ValueRange iterArgs) {
                            Value i =
                                iiBuilder.create<arith::AddIOp>(loc, io, ii);
                            Value inM = iiBuilder.create<arith::CmpIOp>(
                                loc, arith::CmpIPredicate::ult, i, mVal);

                            iiBuilder.create<scf::ForOp>(
                                loc, c0, c16, c1, ValueRange{},
                                [&](OpBuilder &jiBuilder, Location loc,
                                    Value ji, ValueRange iterArgs) {
                                  Value j = jiBuilder.create<arith::AddIOp>(
                                      loc, jo, ji);
                                  Value inN = jiBuilder.create<arith::CmpIOp>(
                                      loc, arith::CmpIPredicate::ult, j, nVal);
                                  Value inMN =
                                      jiBuilder.create<arith::AndIOp>(loc, inM,
                                                                      inN);

                                  auto storeIf = jiBuilder.create<scf::IfOp>(
                                      loc, inMN, /*withElseRegion=*/true);
                                  OpBuilder thenBuilder =
                                      storeIf.getThenBodyBuilder(
                                          jiBuilder.getListener());

                                  Value isFirstKTile =
                                      thenBuilder.create<arith::CmpIOp>(
                                          loc, arith::CmpIPredicate::eq, ko,
                                          c0);
                                  auto accInitIf =
                                      thenBuilder.create<scf::IfOp>(
                                          loc, TypeRange{elementType},
                                          isFirstKTile,
                                          /*withElseRegion=*/true);
                                  OpBuilder initThenBuilder =
                                      accInitIf.getThenBodyBuilder(
                                          thenBuilder.getListener());
                                  initThenBuilder.create<scf::YieldOp>(loc,
                                                                       zero);
                                  OpBuilder initElseBuilder =
                                      accInitIf.getElseBodyBuilder(
                                          thenBuilder.getListener());
                                  Value old = initElseBuilder
                                                  .create<memref::LoadOp>(
                                                      loc, out,
                                                      ValueRange{i, j});
                                  initElseBuilder.create<scf::YieldOp>(loc,
                                                                       old);

                                  auto forKI =
                                      thenBuilder.create<scf::ForOp>(
                                          loc, c0, c16, c1,
                                          ValueRange{accInitIf.getResult(0)},
                                          [&](OpBuilder &kiBuilder,
                                              Location loc, Value ki,
                                              ValueRange iterArgs) {
                                            Value acc = iterArgs[0];
                                            Value k = kiBuilder
                                                          .create<arith::AddIOp>(
                                                              loc, ko, ki);
                                            Value inK =
                                                kiBuilder.create<arith::CmpIOp>(
                                                    loc,
                                                    arith::CmpIPredicate::ult,
                                                    k, kVal);
                                            Value lhsInBounds =
                                                kiBuilder.create<arith::AndIOp>(
                                                    loc, inM, inK);
                                            Value rhsInBounds =
                                                kiBuilder.create<arith::AndIOp>(
                                                    loc, inK, inN);
                                            Value lhsValue =
                                                createInBoundsLoadOrZero(
                                                    kiBuilder, loc, lhs,
                                                    ValueRange{i, k},
                                                    lhsInBounds, zero,
                                                    elementType);
                                            Value rhsValue =
                                                createInBoundsLoadOrZero(
                                                    kiBuilder, loc, rhs,
                                                    ValueRange{k, j},
                                                    rhsInBounds, zero,
                                                    elementType);
                                            Value prod =
                                                kiBuilder
                                                    .create<arith::MulFOp>(
                                                        loc, lhsValue,
                                                        rhsValue);
                                            Value sum =
                                                kiBuilder
                                                    .create<arith::AddFOp>(
                                                        loc, acc, prod);
                                            kiBuilder.create<scf::YieldOp>(
                                                loc, sum);
                                          });
                                  thenBuilder.create<memref::StoreOp>(
                                      loc, forKI.getResult(0), out,
                                      ValueRange{i, j});
                                  thenBuilder.create<scf::YieldOp>(loc);
                                  OpBuilder elseBuilder =
                                      storeIf.getElseBodyBuilder(
                                          jiBuilder.getListener());
                                  elseBuilder.create<scf::YieldOp>(loc);
                                });
                          });
                    });
              });
        });

    rewriter.eraseOp(forI);
    return success();
  }
};

struct MatMulTileLoopsPass
    : public PassWrapper<MatMulTileLoopsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatMulTileLoopsPass)
  StringRef getArgument() const override { return "toy-matmul-tile-loops"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TileMatMulLoopPattern>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::toy::createMatMulTileLoopsPass() {
  return std::make_unique<MatMulTileLoopsPass>();
}
