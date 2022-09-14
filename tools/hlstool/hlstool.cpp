//===- hlstool.cpp - The hlstool utility for working with .fir files ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements 'hlstool', which composes together a variety of
// CIRCT libraries that can be used to realise HLS (High Level Synthesis)
// flows.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Chrono.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "circt/Conversion/ExportVerilog.h"
#include "circt/Conversion/Passes.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Support/LoweringOptions.h"
#include "circt/Transforms/Passes.h"

#include "circt/InitAllDialects.h"
#include "circt/InitAllPasses.h"

#include <iostream>

using namespace llvm;
using namespace mlir;
using namespace circt;

// --------------------------------------------------------------------------
// Tool options
// --------------------------------------------------------------------------

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<std::string>
    outputFilename("o",
                   cl::desc("Output filename, or directory for split output"),
                   cl::value_desc("filename"), cl::init("-"));

static cl::opt<bool>
    splitInputFile("split-input-file",
                   cl::desc("Split the input file into pieces and process each "
                            "chunk independently"),
                   cl::init(false), cl::Hidden);

static cl::opt<bool>
    verifyDiagnostics("verify-diagnostics",
                      cl::desc("Check that emitted diagnostics match "
                               "expected-* lines on the corresponding line"),
                      cl::init(false), cl::Hidden);

static cl::opt<bool>
    verbosePassExecutions("verbose-pass-executions",
                          cl::desc("Log executions of toplevel module passes"),
                          cl::init(false));

static cl::opt<bool>
    verifyPasses("verify-each",
                 cl::desc("Run the verifier after each transformation pass"),
                 cl::init(true));

static cl::opt<bool>
    allowUnregisteredDialects("allow-unregistered-dialects",
                              cl::desc("Allow unknown dialects in the input"),
                              cl::init(false), cl::Hidden);

enum HLSFlow { HLSFlowDynamic };

static cl::opt<HLSFlow> hlsFlow(
    cl::desc("HLS flow"),
    cl::values(clEnumValN(HLSFlowDynamic, "dynamic", "Dynamically scheduled")));

enum OutputFormatKind { OutputIR, OutputVerilog };

static cl::opt<int>
    irInputLevel("ir-input-level",
                 cl::desc("Level at which to input IR at. It is flow-defined "
                          "which value corersponds to which IR level."),
                 cl::init(-1));

static cl::opt<int>
    irOutputLevel("ir-output-level",
                  cl::desc("Level at which to output IR at. It is flow-defined "
                           "which value corersponds to which IR level."),
                  cl::init(-1));

static cl::opt<OutputFormatKind> outputFormat(
    cl::desc("Specify output format:"),
    cl::values(clEnumValN(OutputIR, "ir", "Emit post-HLS IR"),
               clEnumValN(OutputVerilog, "verilog", "Emit Verilog")),
    cl::init(OutputVerilog));

// --------------------------------------------------------------------------
// (Configurable) pass pipelines
// --------------------------------------------------------------------------

/// Create a simple canonicalizer pass.
static std::unique_ptr<Pass> createSimpleCanonicalizerPass() {
  mlir::GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  config.enableRegionSimplification = false;
  return mlir::createCanonicalizerPass(config);
}

static void loadDHLSPipeline(OpPassManager &pm) {
  // Software lowering
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(circt::createFlattenMemRefPass());

  // DHLS conversion
  pm.addPass(circt::createStandardToHandshakePass());
}

static void loadHandshakeTransformsPipeline(OpPassManager &pm) {
  pm.nest<handshake::FuncOp>().addPass(createSimpleCanonicalizerPass());
  pm.nest<handshake::FuncOp>().addPass(
      handshake::createHandshakeMaterializeForksSinksPass());
  pm.nest<handshake::FuncOp>().addPass(createSimpleCanonicalizerPass());
  // Todo: arguments
  pm.nest<handshake::FuncOp>().addPass(
      handshake::createHandshakeInsertBuffersPass());
  pm.nest<handshake::FuncOp>().addPass(createSimpleCanonicalizerPass());
}

static void loadFIRRTLLoweringPipeline(OpPassManager &pm) {
  // FIRRTL lowering; inspired by firtool but without the parameters.
  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createDedupPass());
  pm.addNestedPass<firrtl::CircuitOp>(firrtl::createLowerFIRRTLTypesPass(
      /*preserveAggregate=*/firrtl::PreserveAggregate::PreserveMode::None,
      /*preservePublicTypes=*/false));
  auto &modulePM = pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>();
  modulePM.addPass(firrtl::createExpandWhensPass());
  // a bit of cleanup.
  pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
      createSimpleCanonicalizerPass());
  pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
      firrtl::createInferReadWritePass());
  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createLowerMemoryPass());
  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createInlinerPass());
  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createIMConstPropPass());
  // The above passes, IMConstProp in particular, introduce additional
  // canonicalization opportunities that we should pick up here before we
  // proceed to output-specific pipelines.
  pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
      createSimpleCanonicalizerPass());
  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createRemoveUnusedPortsPass());
  pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
      firrtl::createMergeConnectionsPass(
          /*mergeConnectionsAgggresively=*/false));
}

static void loadHWLoweringPipeline(OpPassManager &pm) {
  pm.nest<hw::HWModuleOp>().addPass(seq::createSeqFIRRTLLowerToSVPass());
  pm.addPass(sv::createHWMemSimImplPass(false, false));
  pm.addPass(seq::createSeqLowerToSVPass());
  pm.nest<hw::HWModuleOp>().addPass(sv::createHWCleanupPass());

  // Legalize unsupported operations within the modules.
  pm.nest<hw::HWModuleOp>().addPass(sv::createHWLegalizeModulesPass());

  // Tidy up the IR to improve verilog emission quality.
  auto &modulePM = pm.nest<hw::HWModuleOp>();
  modulePM.addPass(sv::createPrettifyVerilogPass());
}

// --------------------------------------------------------------------------
// Tool driver code
// --------------------------------------------------------------------------

enum HLSFlowDynamicIRLevel {
  High = 0,
  Handshake = 1,
  Firrtl = 2,
  Rtl = 3,
};

static void printHLSFlowDynamic() {
  llvm::errs() << "Valid levels are:\n";
  llvm::errs() << HLSFlowDynamicIRLevel::High << ": 'cf/scf/affine' level IR\n";
  llvm::errs() << HLSFlowDynamicIRLevel::Handshake << ": 'handshake' IR\n";
  llvm::errs() << HLSFlowDynamicIRLevel::Firrtl << ": 'firrtl'\n";
  llvm::errs() << HLSFlowDynamicIRLevel::Rtl << ": 'hw/comb' IR\n";
}

static LogicalResult
doHLSFlowDynamic(PassManager &pm, ModuleOp module,
                 Optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {

  if (irInputLevel < 0)
    irInputLevel = HLSFlowDynamicIRLevel::High; // Default to highest level

  if (irOutputLevel < 0)
    irOutputLevel = HLSFlowDynamicIRLevel::Rtl; // Default to lowest level

  if (irInputLevel > HLSFlowDynamicIRLevel::Rtl) {
    llvm::errs() << "Invalid IR input level: " << irInputLevel << "\n";
    printHLSFlowDynamic();
    return failure();
  }

  if (outputFormat == OutputIR &&
      (irOutputLevel > HLSFlowDynamicIRLevel::Rtl)) {
    llvm::errs() << "Invalid IR output level: " << irOutputLevel << "\n";
    printHLSFlowDynamic();
    return failure();
  }

  bool suppressLaterPasses = false;
  auto notSuppressed = [&]() { return !suppressLaterPasses; };
  auto addIfNeeded = [&](llvm::function_ref<bool()> predicate,
                         llvm::function_ref<void()> passAdder) {
    if (predicate())
      passAdder();
  };

  auto addIRLevel = [&](int level, llvm::function_ref<void()> passAdder) {
    addIfNeeded(notSuppressed, [&]() {
      // Add the pass if the input IR level is at least the current
      // abstraction.
      if (irInputLevel <= level)
        passAdder();
      // Suppresses later passes if we're emitting IR and the output IR level is
      // the current level.
      if (outputFormat == OutputIR && irOutputLevel == level)
        suppressLaterPasses = true;
    });
  };

  addIRLevel(HLSFlowDynamicIRLevel::High, [&]() { loadDHLSPipeline(pm); });
  addIRLevel(HLSFlowDynamicIRLevel::Handshake,
             [&]() { loadHandshakeTransformsPipeline(pm); });
  addIRLevel(HLSFlowDynamicIRLevel::Firrtl, [&]() {
    pm.addPass(circt::createHandshakeToFIRRTLPass());
    loadFIRRTLLoweringPipeline(pm);
  });
  addIRLevel(HLSFlowDynamicIRLevel::Rtl, [&]() {
    pm.addPass(createLowerFIRRTLToHWPass(/*enableAnnotationWarning=*/false,
                                         /*emitChiselAssertsAsSVA=*/false));
    loadHWLoweringPipeline(pm);
  });

  if (outputFormat == OutputVerilog) {
    applyLoweringCLOptions(module);
    pm.addPass(createExportVerilogPass(outputFile.value()->os()));
  }

  // Go execute!
  if (failed(pm.run(module)))
    return failure();

  if (outputFormat == OutputIR)
    module->print(outputFile.value()->os());

  return success();
}

/// Process a single buffer of the input.
static LogicalResult
processBuffer(MLIRContext &context, TimingScope &ts, llvm::SourceMgr &sourceMgr,
              Optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
  // Parse the input.
  mlir::OwningOpRef<mlir::ModuleOp> module;
  llvm::sys::TimePoint<> parseStartTime;
  if (verbosePassExecutions) {
    llvm::errs() << "[hlstool] Running MLIR parser\n";
    parseStartTime = llvm::sys::TimePoint<>::clock::now();
  }
  auto parserTimer = ts.nest("MLIR Parser");
  module = parseSourceFile<ModuleOp>(sourceMgr, &context);

  if (!module)
    return failure();

  if (verbosePassExecutions) {
    auto elpased = std::chrono::duration<double>(
                       llvm::sys::TimePoint<>::clock::now() - parseStartTime) /
                   std::chrono::seconds(1);
    llvm::errs() << "[hlstool] -- Done in " << llvm::format("%.3f", elpased)
                 << " sec\n";
  }

  // Apply any pass manager command line options.
  PassManager pm(&context);
  pm.enableVerifier(verifyPasses);
  pm.enableTiming(ts);
  applyPassManagerCLOptions(pm);

  if (hlsFlow == HLSFlow::HLSFlowDynamic)
    if (failed(doHLSFlowDynamic(pm, module.get(), outputFile)))
      return failure();

  // We intentionally "leak" the Module into the MLIRContext instead of
  // deallocating it.  There is no need to deallocate it right before process
  // exit.
  (void)module.release();
  return success();
}

/// Process a single split of the input. This allocates a source manager and
/// creates a regular or verifying diagnostic handler, depending on whether
/// the user set the verifyDiagnostics option.
static LogicalResult
processInputSplit(MLIRContext &context, TimingScope &ts,
                  std::unique_ptr<llvm::MemoryBuffer> buffer,
                  Optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
  if (!verifyDiagnostics) {
    SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
    return processBuffer(context, ts, sourceMgr, outputFile);
  }

  SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr, &context);
  context.printOpOnDiagnostic(false);
  (void)processBuffer(context, ts, sourceMgr, outputFile);
  return sourceMgrHandler.verify();
}

/// Process the entire input provided by the user, splitting it up if the
/// corresponding option was specified.
static LogicalResult
processInput(MLIRContext &context, TimingScope &ts,
             std::unique_ptr<llvm::MemoryBuffer> input,
             Optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
  if (!splitInputFile)
    return processInputSplit(context, ts, std::move(input), outputFile);

  return splitAndProcessBuffer(
      std::move(input),
      [&](std::unique_ptr<MemoryBuffer> buffer, raw_ostream &) {
        return processInputSplit(context, ts, std::move(buffer), outputFile);
      },
      llvm::outs());
}

static LogicalResult executeHlstool(MLIRContext &context) {
  if (allowUnregisteredDialects)
    context.allowUnregisteredDialects();

  // Create the timing manager we use to sample execution times.
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  auto ts = tm.getRootScope();

  // Set up the input file.
  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  Optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
  outputFile.emplace(openOutputFile(outputFilename, &errorMessage));
  if (!outputFile.value()) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  // Process the input.
  if (failed(processInput(context, ts, std::move(input), outputFile)))
    return failure();

  // If the result succeeded and we're emitting a file, close it.
  if (outputFile.has_value())
    outputFile.value()->keep();

  return success();
}

/// Main driver for hlstool command.  This sets up LLVM and MLIR, and parses
/// command line options before passing off to 'executeHlstool'.  This is set
/// up so we can `exit(0)` at the end of the program to avoid teardown of the
/// MLIRContext and modules inside of it (reducing compile time).
int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerAsmPrinterCLOptions();
  registerLoweringCLOptions();

  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, "CIRCT HLS tool\n");

  DialectRegistry registry;
  // Register MLIR dialects.
  registry.insert<mlir::AffineDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithmeticDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::scf::SCFDialect>();

  // Register MLIR passes.
  mlir::registerCSEPass();
  mlir::registerSCCPPass();
  mlir::registerInlinerPass();
  mlir::registerCanonicalizerPass();

  // Register CIRCT dialects.
  registry
      .insert<firrtl::FIRRTLDialect, hw::HWDialect, comb::CombDialect,
              seq::SeqDialect, sv::SVDialect, handshake::HandshakeDialect>();

  // Do the guts of the hlstool process.
  MLIRContext context(registry);
  auto result = executeHlstool(context);

  // Use "exit" instead of return'ing to signal completion.  This avoids
  // invoking the MLIRContext destructor, which spends a bunch of time
  // deallocating memory etc which process exit will do for us.
  exit(failed(result));
}