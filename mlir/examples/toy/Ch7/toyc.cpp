//===- toyc.cpp - The Toy Compiler ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the entry point for the Toy compiler.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "toy/AST.h"
#include "toy/Dialect.h"
#include "toy/Lexer.h"
#include "toy/MLIRGen.h"
#include "toy/Parser.h"
#include "toy/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUCommon/GPUToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <memory>
#include <string>
#include <system_error>
#include <utility>

using namespace toy;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace {
enum InputType { Toy, MLIR };
} // namespace
static cl::opt<enum InputType> inputType(
    "x", cl::init(Toy), cl::desc("Decided the kind of output desired"),
    cl::values(clEnumValN(Toy, "toy", "load the input file as a Toy source.")),
    cl::values(clEnumValN(MLIR, "mlir",
                          "load the input file as an MLIR file")));

namespace {
enum Action {
  None,
  DumpAST,
  DumpMLIR,
  DumpMLIRSCFMatMul,
  DumpMLIRTiledMatMul,
  DumpMLIRReorderedTiledMatMul,
  DumpMLIRAffine,
  DumpMLIRGPU,
  DumpMLIRGPUOutlined,
  DumpMLIRGPUNVVM,
  DumpMLIRGPUBinary,
  DumpMLIRGPUHost,
  DumpLLVMGPU,
  DumpMLIRLLVM,
  DumpLLVMIR,
  RunJIT,
  RunGPUJIT
};
} // namespace
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(
        DumpMLIRSCFMatMul, "mlir-scf-matmul",
        "output the MLIR dump after lowering toy.matmul to scf.for loops")),
    cl::values(clEnumValN(DumpMLIRTiledMatMul, "mlir-tiled-matmul",
                          "output the MLIR dump after tiling matmul loops")),
    cl::values(clEnumValN(
        DumpMLIRReorderedTiledMatMul, "mlir-reordered-tiled-matmul",
        "output the MLIR dump after reordering tiled matmul loops")),
    cl::values(clEnumValN(DumpMLIRAffine, "mlir-affine",
                          "output the MLIR dump after affine lowering")),
    cl::values(clEnumValN(DumpMLIRGPU, "mlir-gpu",
                          "output the MLIR dump after gpu lowering")),
    cl::values(clEnumValN(DumpMLIRGPUOutlined, "mlir-gpu-outlined",
                          "output the MLIR dump after gpu kernel outlining")),
    cl::values(clEnumValN(DumpMLIRGPUNVVM, "mlir-gpu-nvvm",
                          "output the MLIR dump after gpu to nvvm lowering")),
    cl::values(clEnumValN(DumpMLIRGPUBinary, "mlir-gpu-binary",
                          "output the MLIR dump after gpu binary lowering")),
    cl::values(clEnumValN(DumpMLIRGPUHost, "mlir-gpu-host",
                          "output the MLIR dump after gpu host lowering")),
    cl::values(clEnumValN(DumpLLVMGPU, "llvm-gpu",
                          "output the LLVM IR dump after gpu host lowering")),
    cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm",
                          "output the MLIR dump after llvm lowering")),
    cl::values(clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR dump")),
    cl::values(
        clEnumValN(RunJIT, "jit",
                   "JIT the code and run it by invoking the main function")),
    cl::values(clEnumValN(
        RunGPUJIT, "gpu-jit",
        "JIT the GPU-lowered code and run it by invoking the main function")));

static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));

static cl::opt<std::string>
    gpuBinaryFormat("gpu-binary-format", cl::init("llvm"),
                    cl::desc("GPU binary format for -emit=mlir-gpu-binary, "
                             "-emit=mlir-gpu-host, and -emit=llvm-gpu "
                             "(llvm, isa, bin, or fatbin)."));

static cl::list<std::string>
    jitSharedLibs("shared-libs", cl::ZeroOrMore, cl::CommaSeparated,
                  cl::desc("Shared libraries to load when JIT-running."));

/// Returns a Toy AST resulting from parsing the file or a nullptr on error.
std::unique_ptr<toy::ModuleAST> parseInputFile(llvm::StringRef filename) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }
  auto buffer = fileOrErr.get()->getBuffer();
  LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
  Parser parser(lexer);
  return parser.parseModule();
}

int loadMLIR(mlir::MLIRContext &context,
             mlir::OwningOpRef<mlir::ModuleOp> &module) {
  // Handle '.toy' input to the compiler.
  if (inputType != InputType::MLIR &&
      !llvm::StringRef(inputFilename).ends_with(".mlir")) {
    auto moduleAST = parseInputFile(inputFilename);
    if (!moduleAST)
      return 6;
    module = mlirGen(context, *moduleAST);
    return !module ? 1 : 0;
  }

  // Otherwise, the input is '.mlir'.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }

  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }
  return 0;
}

int loadAndProcessMLIR(mlir::MLIRContext &context,
                       mlir::OwningOpRef<mlir::ModuleOp> &module) {
  if (int error = loadMLIR(context, module))
    return error;

  mlir::PassManager pm(module.get()->getName());
  // Apply any generic pass manager command line options and run the pipeline.
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
    return 4;

  // Check to see what granularity of MLIR we are compiling to.
  bool isLoweringToGPU = emitAction == Action::DumpMLIRGPU ||
                         emitAction == Action::DumpMLIRGPUOutlined ||
                         emitAction == Action::DumpMLIRGPUNVVM ||
                         emitAction == Action::DumpMLIRGPUBinary ||
                         emitAction == Action::DumpMLIRGPUHost ||
                         emitAction == Action::DumpLLVMGPU ||
                         emitAction == Action::RunGPUJIT;
  bool isOutliningGPU = emitAction == Action::DumpMLIRGPUOutlined ||
                        emitAction == Action::DumpMLIRGPUNVVM ||
                        emitAction == Action::DumpMLIRGPUBinary ||
                        emitAction == Action::DumpMLIRGPUHost ||
                        emitAction == Action::DumpLLVMGPU ||
                        emitAction == Action::RunGPUJIT;
  bool isLoweringGPUToNVVM = emitAction == Action::DumpMLIRGPUNVVM ||
                             emitAction == Action::DumpMLIRGPUBinary ||
                             emitAction == Action::DumpMLIRGPUHost ||
                             emitAction == Action::DumpLLVMGPU ||
                             emitAction == Action::RunGPUJIT;
  bool isLoweringGPUToBinary = emitAction == Action::DumpMLIRGPUBinary ||
                               emitAction == Action::DumpMLIRGPUHost ||
                               emitAction == Action::DumpLLVMGPU ||
                               emitAction == Action::RunGPUJIT;
  bool isLoweringGPUHost = emitAction == Action::DumpMLIRGPUHost ||
                           emitAction == Action::DumpLLVMGPU ||
                           emitAction == Action::RunGPUJIT;
  bool isLoweringToSCFMatMul =
      emitAction == Action::DumpMLIRSCFMatMul ||
      emitAction == Action::DumpMLIRTiledMatMul ||
      emitAction == Action::DumpMLIRReorderedTiledMatMul;
  bool isTilingMatMul = emitAction == Action::DumpMLIRTiledMatMul ||
                        emitAction == Action::DumpMLIRReorderedTiledMatMul;
  bool isReorderingTiledMatMul =
      emitAction == Action::DumpMLIRReorderedTiledMatMul;
  bool isLoweringToAffine = emitAction == Action::DumpMLIRAffine ||
                            emitAction == Action::DumpMLIRLLVM ||
                            emitAction == Action::DumpLLVMIR ||
                            emitAction == Action::RunJIT;
  bool isLoweringToLLVM = emitAction == Action::DumpMLIRLLVM ||
                          emitAction == Action::DumpLLVMIR ||
                          emitAction == Action::RunJIT;

  if (enableOpt || isLoweringToSCFMatMul || isLoweringToAffine ||
      isLoweringToGPU) {
    // Inline all functions into main and then delete them.
    pm.addPass(mlir::createInlinerPass());

    // Now that there is only one function, we can infer the shapes of each of
    // the operations.
    mlir::OpPassManager &optPM = pm.nest<mlir::toy::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::toy::createShapeInferencePass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
  }

  if (isLoweringToSCFMatMul) {
    pm.addPass(mlir::toy::createMatMulToSCFPass());

    if (isTilingMatMul)
      pm.addPass(mlir::toy::createMatMulTileLoopsPass());

    if (isReorderingTiledMatMul)
      pm.addPass(mlir::toy::createMatMulReorderTiledLoopsPass());

    mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
  }

  if (isLoweringToAffine) {
    // Partially lower the toy dialect.
    pm.addPass(mlir::toy::createLowerToAffinePass());

    // Add a few cleanups post lowering.
    mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());

    // Add optimizations if enabled.
    if (enableOpt) {
      optPM.addPass(mlir::affine::createLoopFusionPass());
      optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
    }
  }

  if (isLoweringToGPU) {
    // Partially lower the toy dialect to GPU operations.
    pm.addPass(mlir::toy::createLowerToGPUPass());

    // Add a few cleanups post lowering.
    mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
  }

  if (isOutliningGPU) {
    pm.addPass(mlir::createGpuKernelOutliningPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
  }

  if (isLoweringGPUToNVVM) {
    pm.addPass(mlir::createSCFToControlFlowPass());

    mlir::ConvertGpuOpsToNVVMOpsOptions gpuToNVVMOptions;
    gpuToNVVMOptions.indexBitwidth = 64;
    gpuToNVVMOptions.useBarePtrCallConv = true;
    pm.addNestedPass<mlir::gpu::GPUModuleOp>(
        mlir::createConvertGpuOpsToNVVMOps(gpuToNVVMOptions));
    pm.addNestedPass<mlir::gpu::GPUModuleOp>(
        mlir::createCanonicalizerPass());
    pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createCSEPass());
  }

  if (isLoweringGPUToBinary) {
    mlir::GpuNVVMAttachTargetOptions nvvmTargetOptions;
    nvvmTargetOptions.triple = "nvptx64-nvidia-cuda";
    nvvmTargetOptions.chip = "sm_86";
    nvvmTargetOptions.features = "+ptx60";
    pm.addPass(mlir::createGpuNVVMAttachTarget(nvvmTargetOptions));

    mlir::GpuModuleToBinaryPassOptions binaryOptions;
    binaryOptions.compilationTarget =
        emitAction == Action::RunGPUJIT && !gpuBinaryFormat.getNumOccurrences()
            ? std::string("fatbin")
            : gpuBinaryFormat.getValue();
    pm.addPass(mlir::createGpuModuleToBinaryPass(binaryOptions));
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
  }

  if (isLoweringGPUHost) {
    pm.addPass(mlir::toy::createLowerPrintToLLVMPass());
    pm.addPass(mlir::createLowerAffinePass());
    pm.addPass(mlir::createSCFToControlFlowPass());

    mlir::GpuToLLVMConversionPassOptions gpuToLLVMOptions;
    gpuToLLVMOptions.kernelBarePtrCallConv = true;
    pm.addPass(mlir::createGpuToLLVMConversionPass(gpuToLLVMOptions));
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
  }

  if (isLoweringToLLVM) {
    // Finish lowering the toy IR to the LLVM dialect.
    pm.addPass(mlir::toy::createLowerToLLVMPass());
    // This is necessary to have line tables emitted and basic
    // debugger working. In the future we will add proper debug information
    // emission directly from our frontend.
    pm.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());
  }

  if (mlir::failed(pm.run(*module)))
    return 4;
  return 0;
}

int dumpAST() {
  if (inputType == InputType::MLIR) {
    llvm::errs() << "Can't dump a Toy AST when the input is MLIR\n";
    return 5;
  }

  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST)
    return 1;

  dump(*moduleAST);
  return 0;
}

int dumpLLVMIR(mlir::ModuleOp module) {
  // Register the translation to LLVM IR with the MLIR context.
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerGPUDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());
  mlir::registerNVVMDialectTranslation(*module->getContext());

  // Convert the module to LLVM IR in a new LLVM IR context.
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Create target machine and configure the LLVM Module
  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError) {
    llvm::errs() << "Could not create JITTargetMachineBuilder\n";
    return -1;
  }

  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) {
    llvm::errs() << "Could not create TargetMachine\n";
    return -1;
  }
  mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(),
                                                        tmOrError.get().get());

  /// Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  llvm::errs() << *llvmModule << "\n";
  return 0;
}

int runJit(mlir::ModuleOp module, bool isGpuJit = false) {
  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Register the translation from MLIR to LLVM IR, which must happen before we
  // can JIT-compile.
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  if (isGpuJit)
    mlir::registerGPUDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());
  if (isGpuJit)
    mlir::registerNVVMDialectTranslation(*module->getContext());

  // An optimization pipeline to use within the execution engine.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  mlir::ExecutionEngineOptions engineOptions;
  engineOptions.transformer = optPipeline;
  llvm::SmallVector<llvm::StringRef, 4> sharedLibRefs;
  for (const std::string &lib : jitSharedLibs)
    sharedLibRefs.push_back(lib);
  engineOptions.sharedLibPaths = sharedLibRefs;
  auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
  if (!maybeEngine) {
    llvm::errs() << "Failed to construct an execution engine\n";
    llvm::logAllUnhandledErrors(maybeEngine.takeError(), llvm::errs());
    return -1;
  }
  auto &engine = maybeEngine.get();

  // Invoke the JIT-compiled function.
  auto invocationResult = engine->invokePacked("main");
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    llvm::logAllUnhandledErrors(std::move(invocationResult), llvm::errs());
    return -1;
  }

  return 0;
}

int main(int argc, char **argv) {
  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");

  if (emitAction == Action::DumpAST)
    return dumpAST();

  // If we aren't dumping the AST, then we are compiling with/to MLIR.
  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);
  mlir::arith::registerConvertArithToLLVMInterface(registry);
  mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::registerConvertNVVMToLLVMInterface(registry);
  mlir::ub::registerConvertUBToLLVMInterface(registry);
  mlir::vector::registerConvertVectorToLLVMInterface(registry);
  mlir::gpu::registerConvertGpuToLLVMInterface(registry);
  mlir::LLVM::registerInlinerInterface(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerGPUDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  mlir::gpu::registerOffloadingLLVMTranslationInterfaceExternalModels(registry);
  mlir::NVVM::registerNVVMTargetInterfaceExternalModels(registry);

  mlir::MLIRContext context(registry);
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::toy::ToyDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (int error = loadAndProcessMLIR(context, module))
    return error;

  // If we aren't exporting to non-mlir, then we are done.
  bool isOutputingMLIR = emitAction == Action::DumpMLIR ||
                         emitAction == Action::DumpMLIRSCFMatMul ||
                         emitAction == Action::DumpMLIRTiledMatMul ||
                         emitAction ==
                             Action::DumpMLIRReorderedTiledMatMul ||
                         emitAction == Action::DumpMLIRAffine ||
                         emitAction == Action::DumpMLIRGPU ||
                         emitAction == Action::DumpMLIRGPUOutlined ||
                         emitAction == Action::DumpMLIRGPUNVVM ||
                         emitAction == Action::DumpMLIRGPUBinary ||
                         emitAction == Action::DumpMLIRGPUHost ||
                         emitAction == Action::DumpMLIRLLVM;
  if (isOutputingMLIR) {
    module->dump();
    return 0;
  }

  // Check to see if we are compiling to LLVM IR.
  if (emitAction == Action::DumpLLVMIR)
    return dumpLLVMIR(*module);

  if (emitAction == Action::DumpLLVMGPU)
    return dumpLLVMIR(*module);

  // Otherwise, we must be running the jit.
  if (emitAction == Action::RunJIT)
    return runJit(*module);
  if (emitAction == Action::RunGPUJIT)
    return runJit(*module, /*isGpuJit=*/true);

  llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  return -1;
}
