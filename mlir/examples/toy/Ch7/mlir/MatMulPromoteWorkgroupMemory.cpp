//====- MatMulPromoteWorkgroupMemory.cpp - Promote matmul tiles ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the sixth Toy GPU refactor stage. It consumes the naive
// gpu.launch produced by stage 5 and explicitly promotes the A/B tiles to
// workgroup memory. Device memory management and outlining are intentionally
// left to later stages.
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
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
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

struct NaiveGPUMatMulLaunch {
  scf::ForOp forKO;
  memref::LoadOp lhsLoad;
  memref::LoadOp rhsLoad;
  memref::StoreOp store;
  Value mVal;
  Value nVal;
  Value kVal;
  Value zero;
};

static FailureOr<Value> matchUpperBound(Value cmpValue, Value index) {
  auto cmp = cmpValue.getDefiningOp<arith::CmpIOp>();
  if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::ult ||
      cmp.getLhs() != index)
    return failure();
  return cmp.getRhs();
}

static FailureOr<NaiveGPUMatMulLaunch>
matchNaiveGPUMatMulLaunch(gpu::LaunchOp launch) {
  if (launch.getNumWorkgroupAttributions() != 0 ||
      launch.getNumPrivateAttributions() != 0 || launch.hasClusterSize() ||
      !launch.getAsyncDependencies().empty())
    return failure();

  gpu::KernelDim3 blockSize = launch.getBlockSizeOperandValues();
  if (!hasConstantIndex(blockSize.x, 16) || !hasConstantIndex(blockSize.y, 16) ||
      !hasConstantIndex(blockSize.z, 1))
    return failure();

  scf::IfOp storeGuard =
      dyn_cast_or_null<scf::IfOp>(lastNonTerminator(&launch.getBody().front()));
  if (!storeGuard || storeGuard.getNumResults() != 0)
    return failure();

  Operation *firstThenOp = firstNonTerminator(storeGuard.thenBlock());
  auto forKO = dyn_cast_or_null<scf::ForOp>(firstThenOp);
  if (!forKO || !hasConstantIndex(forKO.getLowerBound(), 0) ||
      !hasConstantIndex(forKO.getStep(), 16) || forKO.getNumResults() != 1 ||
      forKO.getInitArgs().size() != 1)
    return failure();

  Value zero = forKO.getInitArgs()[0];

  Operation *lastThenOp = lastNonTerminator(storeGuard.thenBlock());
  auto store = dyn_cast_or_null<memref::StoreOp>(lastThenOp);
  if (!store || store.getValueToStore() != forKO.getResult(0))
    return failure();

  Operation *firstKOBodyOp = firstNonTerminator(forKO.getBody());
  auto forKI = dyn_cast_or_null<scf::ForOp>(firstKOBodyOp);
  if (!forKI || !hasConstantIndex(forKI.getLowerBound(), 0) ||
      !hasConstantIndex(forKI.getUpperBound(), 16) ||
      !hasConstantIndex(forKI.getStep(), 1) || forKI.getNumResults() != 1 ||
      forKI.getInitArgs().size() != 1)
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

  if (lhsLoad.getIndices().size() != 2 || rhsLoad.getIndices().size() != 2 ||
      store.getIndices().size() != 2)
    return failure();

  Value i = lhsLoad.getIndices()[0];
  Value j = rhsLoad.getIndices()[1];
  if (store.getIndices()[0] != i || store.getIndices()[1] != j)
    return failure();

  auto inMN = storeGuard.getCondition().getDefiningOp<arith::AndIOp>();
  if (!inMN)
    return failure();
  Value inM = inMN.getLhs();
  Value inN = inMN.getRhs();

  FailureOr<Value> maybeMVal = matchUpperBound(inM, i);
  FailureOr<Value> maybeNVal = matchUpperBound(inN, j);
  if (failed(maybeMVal) || failed(maybeNVal))
    return failure();

  return NaiveGPUMatMulLaunch{forKO, lhsLoad, rhsLoad, store,
                              *maybeMVal, *maybeNVal, forKO.getUpperBound(),
                              zero};
}

struct PromoteMatMulWorkgroupPattern : public OpRewritePattern<gpu::LaunchOp> {
  using OpRewritePattern<gpu::LaunchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::LaunchOp oldLaunch,
                                PatternRewriter &rewriter) const final {
    FailureOr<NaiveGPUMatMulLaunch> maybeNest =
        matchNaiveGPUMatMulLaunch(oldLaunch);
    if (failed(maybeNest))
      return failure();

    NaiveGPUMatMulLaunch nest = *maybeNest;
    Location loc = oldLaunch.getLoc();
    Type elementType = nest.zero.getType();

    rewriter.setInsertionPoint(oldLaunch);
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value c16 = rewriter.create<arith::ConstantIndexOp>(loc, 16);

    Value mVal = nest.mVal;
    Value nVal = nest.nVal;
    Value kVal = nest.kVal;
    Value lhs = nest.lhsLoad.getMemRef();
    Value rhs = nest.rhsLoad.getMemRef();
    Value out = nest.store.getMemRef();

    gpu::KernelDim3 gridSize = oldLaunch.getGridSizeOperandValues();
    gpu::KernelDim3 blockSize = oldLaunch.getBlockSizeOperandValues();

    auto workgroupAddrSpace =
        gpu::AddressSpaceAttr::get(rewriter.getContext(),
                                   gpu::AddressSpace::Workgroup);
    auto tileType =
        MemRefType::get({16, 16}, elementType, MemRefLayoutAttrInterface{},
                        Attribute(workgroupAddrSpace));
    SmallVector<Type, 2> tileTypes{tileType, tileType};
    auto launch = rewriter.create<gpu::LaunchOp>(
        loc, gridSize.x, gridSize.y, gridSize.z, blockSize.x, blockSize.y,
        blockSize.z,
        /*dynamicSharedMemorySize=*/Value(),
        /*asyncTokenType=*/Type(), /*asyncDependencies=*/ValueRange{},
        /*workgroupAttributions=*/tileTypes,
        /*privateAttributions=*/TypeRange{});

    unsigned firstWorkgroupArg = launch.getNumConfigRegionAttributes();
    Value lhsTile = launch.getBody().getArgument(firstWorkgroupArg);
    Value rhsTile = launch.getBody().getArgument(firstWorkgroupArg + 1);
    gpu::KernelDim3 blockIds = launch.getBlockIds();
    gpu::KernelDim3 threadIds = launch.getThreadIds();

    rewriter.setInsertionPointToStart(&launch.getBody().front());
    Value jBlock = rewriter.create<arith::MulIOp>(loc, blockIds.x, c16);
    Value j = rewriter.create<arith::AddIOp>(loc, jBlock, threadIds.x);
    Value iBlock = rewriter.create<arith::MulIOp>(loc, blockIds.y, c16);
    Value i = rewriter.create<arith::AddIOp>(loc, iBlock, threadIds.y);

    Value inM =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, i, mVal);
    Value inN =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, j, nVal);
    Value inMN = rewriter.create<arith::AndIOp>(loc, inM, inN);

    auto forKTile = rewriter.create<scf::ForOp>(
        loc, c0, kVal, c16, ValueRange{nest.zero},
        [&](OpBuilder &tileBuilder, Location loc, Value ko,
            ValueRange iterArgs) {
          Value acc = iterArgs[0];
          Value lhsK =
              tileBuilder.create<arith::AddIOp>(loc, ko, threadIds.x);
          Value rhsK =
              tileBuilder.create<arith::AddIOp>(loc, ko, threadIds.y);
          Value lhsInK = tileBuilder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::ult, lhsK, kVal);
          Value rhsInK = tileBuilder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::ult, rhsK, kVal);
          Value lhsInBounds =
              tileBuilder.create<arith::AndIOp>(loc, inM, lhsInK);
          Value rhsInBounds =
              tileBuilder.create<arith::AndIOp>(loc, rhsInK, inN);

          Value lhsValue = createInBoundsLoadOrZero(
              tileBuilder, loc, lhs, ValueRange{i, lhsK}, lhsInBounds,
              nest.zero, elementType);
          Value rhsValue = createInBoundsLoadOrZero(
              tileBuilder, loc, rhs, ValueRange{rhsK, j}, rhsInBounds,
              nest.zero, elementType);
          tileBuilder.create<memref::StoreOp>(
              loc, lhsValue, lhsTile, ValueRange{threadIds.y, threadIds.x});
          tileBuilder.create<memref::StoreOp>(
              loc, rhsValue, rhsTile, ValueRange{threadIds.y, threadIds.x});
          tileBuilder.create<gpu::BarrierOp>(loc);

          auto forKI = tileBuilder.create<scf::ForOp>(
              loc, c0, c16, c1, ValueRange{acc},
              [&](OpBuilder &kiBuilder, Location loc, Value ki,
                  ValueRange iterArgs) {
                Value innerAcc = iterArgs[0];
                Value lhsTileValue = kiBuilder.create<memref::LoadOp>(
                    loc, lhsTile, ValueRange{threadIds.y, ki});
                Value rhsTileValue = kiBuilder.create<memref::LoadOp>(
                    loc, rhsTile, ValueRange{ki, threadIds.x});
                Value prod = kiBuilder.create<arith::MulFOp>(
                    loc, lhsTileValue, rhsTileValue);
                Value sum =
                    kiBuilder.create<arith::AddFOp>(loc, innerAcc, prod);
                kiBuilder.create<scf::YieldOp>(loc, sum);
              });

          tileBuilder.create<gpu::BarrierOp>(loc);
          tileBuilder.create<scf::YieldOp>(loc, forKI.getResult(0));
        });

    rewriter.create<scf::IfOp>(
        loc, inMN, [&](OpBuilder &thenBuilder, Location loc) {
          thenBuilder.create<memref::StoreOp>(
              loc, forKTile.getResult(0), out, ValueRange{i, j});
          thenBuilder.create<scf::YieldOp>(loc, ValueRange{});
        });

    rewriter.create<gpu::TerminatorOp>(loc);
    rewriter.eraseOp(oldLaunch);
    return success();
  }
};

struct MatMulPromoteWorkgroupMemoryPass
    : public PassWrapper<MatMulPromoteWorkgroupMemoryPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      MatMulPromoteWorkgroupMemoryPass)
  StringRef getArgument() const override {
    return "toy-matmul-promote-workgroup-memory";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, gpu::GPUDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<PromoteMatMulWorkgroupPattern>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::toy::createMatMulPromoteWorkgroupMemoryPass() {
  return std::make_unique<MatMulPromoteWorkgroupMemoryPass>();
}
