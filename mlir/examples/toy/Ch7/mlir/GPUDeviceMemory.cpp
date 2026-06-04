//====- GPUDeviceMemory.cpp - Insert GPU device buffers ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the seventh Toy GPU refactor stage. It consumes the
// host-memref gpu.launch produced by the earlier matmul stages and inserts
// explicit gpu.alloc/gpu.memcpy/gpu.dealloc operations around it.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
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
#include "llvm/ADT/DenseMap.h"
#include <memory>

using namespace mlir;

namespace {

enum BufferRole : unsigned {
  Read = 1,
  Write = 2,
};

struct CapturedBuffer {
  Value host;
  unsigned role;
};

static bool isCapturedHostMemref(Value value, gpu::LaunchOp launch) {
  auto type = dyn_cast<MemRefType>(value.getType());
  return type && type.hasStaticShape() &&
         value.getParentRegion()->isProperAncestor(&launch.getBody());
}

static SmallVector<CapturedBuffer> collectCapturedBuffers(gpu::LaunchOp launch) {
  SmallVector<Value> orderedBuffers;
  DenseMap<Value, unsigned> roles;

  auto mark = [&](Value memref, BufferRole role) {
    if (!isCapturedHostMemref(memref, launch))
      return;
    if (!roles.contains(memref))
      orderedBuffers.push_back(memref);
    roles[memref] |= role;
  };

  launch.walk([&](Operation *op) {
    if (isa<gpu::LaunchOp>(op) && op != launch.getOperation())
      return WalkResult::skip();
    if (auto load = dyn_cast<memref::LoadOp>(op))
      mark(load.getMemRef(), BufferRole::Read);
    if (auto store = dyn_cast<memref::StoreOp>(op))
      mark(store.getMemRef(), BufferRole::Write);
    return WalkResult::advance();
  });

  SmallVector<CapturedBuffer> buffers;
  for (Value buffer : orderedBuffers)
    buffers.push_back({buffer, roles.lookup(buffer)});
  return buffers;
}

struct DeviceBuffer {
  Value host;
  Value device;
  unsigned role;
};

static SmallVector<Type> getAttributionTypes(ArrayRef<BlockArgument> args) {
  SmallVector<Type> types;
  types.reserve(args.size());
  for (BlockArgument arg : args)
    types.push_back(arg.getType());
  return types;
}

struct InsertDeviceMemoryPattern : public OpRewritePattern<gpu::LaunchOp> {
  using OpRewritePattern<gpu::LaunchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::LaunchOp oldLaunch,
                                PatternRewriter &rewriter) const final {
    if (oldLaunch.getAsyncToken() || !oldLaunch.getAsyncDependencies().empty() ||
        oldLaunch.hasClusterSize())
      return failure();

    SmallVector<CapturedBuffer> capturedBuffers =
        collectCapturedBuffers(oldLaunch);
    if (capturedBuffers.empty())
      return failure();

    Location loc = oldLaunch.getLoc();
    Type asyncTokenType = gpu::AsyncTokenType::get(rewriter.getContext());

    rewriter.setInsertionPoint(oldLaunch);
    Value token =
        rewriter.create<gpu::WaitOp>(loc, asyncTokenType, ValueRange{})
            .getAsyncToken();

    SmallVector<DeviceBuffer> deviceBuffers;
    IRMapping mapper;
    for (CapturedBuffer buffer : capturedBuffers) {
      auto memrefType = cast<MemRefType>(buffer.host.getType());
      auto alloc = rewriter.create<gpu::AllocOp>(
          loc, memrefType, asyncTokenType, ValueRange{token}, ValueRange{},
          ValueRange{}, false);
      token = alloc.getAsyncToken();
      deviceBuffers.push_back({buffer.host, alloc.getMemref(), buffer.role});
      mapper.map(buffer.host, alloc.getMemref());
    }

    for (const DeviceBuffer &buffer : deviceBuffers) {
      if (!(buffer.role & BufferRole::Read))
        continue;
      token = rewriter
                  .create<gpu::MemcpyOp>(loc, asyncTokenType,
                                         ValueRange{token}, buffer.device,
                                         buffer.host)
                  .getAsyncToken();
    }

    gpu::KernelDim3 gridSize = oldLaunch.getGridSizeOperandValues();
    gpu::KernelDim3 blockSize = oldLaunch.getBlockSizeOperandValues();
    SmallVector<Type> workgroupTypes =
        getAttributionTypes(oldLaunch.getWorkgroupAttributions());
    SmallVector<Type> privateTypes =
        getAttributionTypes(oldLaunch.getPrivateAttributions());

    auto newLaunch = rewriter.create<gpu::LaunchOp>(
        loc, gridSize.x, gridSize.y, gridSize.z, blockSize.x, blockSize.y,
        blockSize.z,
        /*dynamicSharedMemorySize=*/Value(), asyncTokenType,
        ValueRange{token}, workgroupTypes, privateTypes);

    for (auto [oldArg, newArg] :
         llvm::zip(oldLaunch.getBody().getArguments(),
                   newLaunch.getBody().getArguments()))
      mapper.map(oldArg, newArg);

    rewriter.setInsertionPointToStart(&newLaunch.getBody().front());
    for (Operation &op : oldLaunch.getBody().front().without_terminator())
      rewriter.clone(op, mapper);
    rewriter.create<gpu::TerminatorOp>(loc);

    token = newLaunch.getAsyncToken();
    rewriter.setInsertionPointAfter(newLaunch);
    for (const DeviceBuffer &buffer : deviceBuffers) {
      if (!(buffer.role & BufferRole::Write))
        continue;
      token = rewriter
                  .create<gpu::MemcpyOp>(loc, asyncTokenType,
                                         ValueRange{token}, buffer.host,
                                         buffer.device)
                  .getAsyncToken();
    }

    for (const DeviceBuffer &buffer : deviceBuffers)
      token = rewriter
                  .create<gpu::DeallocOp>(loc, asyncTokenType,
                                          ValueRange{token}, buffer.device)
                  .getAsyncToken();

    rewriter.create<gpu::WaitOp>(loc, Type(), ValueRange{token});
    rewriter.eraseOp(oldLaunch);
    return success();
  }
};

struct GPUInsertDeviceMemoryPass
    : public PassWrapper<GPUInsertDeviceMemoryPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPUInsertDeviceMemoryPass)
  StringRef getArgument() const override {
    return "toy-gpu-insert-device-memory";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, gpu::GPUDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<InsertDeviceMemoryPattern>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::toy::createGPUInsertDeviceMemoryPass() {
  return std::make_unique<GPUInsertDeviceMemoryPass>();
}
