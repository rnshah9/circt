//===- MSFTPasses.cpp - Implement MSFT passes -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/MSFT/MSFTPasses.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/MSFT/ExportTcl.h"
#include "circt/Dialect/MSFT/MSFTDialect.h"
#include "circt/Dialect/MSFT/MSFTOpInterfaces.h"
#include "circt/Dialect/MSFT/MSFTOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/Namespace.h"

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"

using namespace circt;
using namespace msft;

namespace circt {
namespace msft {
#define GEN_PASS_CLASSES
#include "circt/Dialect/MSFT/MSFTPasses.h.inc"
} // namespace msft
} // namespace circt

/// TODO: Migrate these to some sort of OpInterface shared with hw.
static bool isAnyModule(Operation *module) {
  return isa<MSFTModuleOp, MSFTModuleExternOp>(module) ||
         hw::isAnyModule(module);
}
hw::ModulePortInfo getModulePortInfo(Operation *op) {
  if (auto mod = dyn_cast<MSFTModuleOp>(op))
    return mod.getPorts();
  if (auto mod = dyn_cast<MSFTModuleExternOp>(op))
    return mod.getPorts();
  return hw::getModulePortInfo(op);
}

static SymbolRefAttr getPart(Operation *op) {
  return op->getAttrOfType<SymbolRefAttr>("targetDesignPartition");
}

//===----------------------------------------------------------------------===//
// Lower dynamic instances to global refs.
//===----------------------------------------------------------------------===//

namespace {
struct LowerInstancesPass : public LowerInstancesBase<LowerInstancesPass> {
  void runOnOperation() override;

  LogicalResult lower(DynamicInstanceOp inst, InstanceHierarchyOp hier,
                      OpBuilder &b);

  // Aggregation of the global ref attributes populated as a side-effect of the
  // conversion.
  DenseMap<Operation *, SmallVector<hw::GlobalRefAttr, 0>> globalRefsToApply;

  // Cache the top-level symbols. Insert the new ones we're creating for new
  // global ref ops.
  SymbolCache topSyms;

  // In order to be efficient, cache the "symbols" in each module.
  DenseMap<MSFTModuleOp, SymbolCache> perModSyms;
  // Accessor for `perModSyms` which lazily constructs each cache.
  const SymbolCache &getSyms(MSFTModuleOp mod);
};
} // anonymous namespace

const SymbolCache &LowerInstancesPass::getSyms(MSFTModuleOp mod) {
  auto symsFound = perModSyms.find(mod);
  if (symsFound != perModSyms.end())
    return symsFound->getSecond();

  // Build the cache.
  SymbolCache &syms = perModSyms[mod];
  mod.walk([&syms, mod](Operation *op) {
    if (op == mod)
      return;
    if (auto name =
            op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
      syms.addDefinition(name, op);
  });
  return syms;
}

LogicalResult LowerInstancesPass::lower(DynamicInstanceOp inst,
                                        InstanceHierarchyOp hier,
                                        OpBuilder &b) {

  hw::GlobalRefOp ref = nullptr;

  // If 'inst' doesn't contain any ops which use a global ref op, don't create
  // one.
  if (llvm::any_of(inst.getOps(), [](Operation &op) {
        return isa<DynInstDataOpInterface>(op);
      })) {

    // Come up with a unique symbol name.
    auto refSym = StringAttr::get(&getContext(), "instref");
    auto origRefSym = refSym;
    unsigned ctr = 0;
    while (topSyms.getDefinition(refSym))
      refSym = StringAttr::get(&getContext(),
                               origRefSym.getValue() + "_" + Twine(++ctr));

    // Create a global ref to replace us.
    ArrayAttr globalRefPath = inst.globalRefPath();
    ref = b.create<hw::GlobalRefOp>(inst.getLoc(), refSym, globalRefPath);
    auto refAttr = hw::GlobalRefAttr::get(ref);

    // Add the new symbol to the symbol cache.
    topSyms.addDefinition(refSym, ref);

    // For each level of `globalRef`, find the static operation which needs a
    // back reference to the global ref which is replacing us.
    bool symNotFound = false;
    for (auto innerRef : globalRefPath.getAsRange<hw::InnerRefAttr>()) {
      MSFTModuleOp mod =
          cast<MSFTModuleOp>(topSyms.getDefinition(innerRef.getModule()));
      const SymbolCache &modSyms = getSyms(mod);
      Operation *tgtOp = modSyms.getDefinition(innerRef.getName());
      if (!tgtOp) {
        symNotFound = true;
        inst.emitOpError("Could not find ")
            << innerRef.getName() << " in module " << innerRef.getModule();
        continue;
      }
      // Add the backref to the list of attributes to apply.
      globalRefsToApply[tgtOp].push_back(refAttr);

      // Since GlobalRefOp uses the `inner_sym` attribute, assign the
      // 'inner_sym' attribute if it's not already assigned.
      if (!tgtOp->hasAttr("inner_sym")) {
        tgtOp->setAttr("inner_sym", innerRef.getName());
      }
    }
    if (symNotFound)
      return inst.emitOpError(
          "Could not find operation corresponding to instance reference");
  }

  // Relocate all my children.
  OpBuilder hierBlock(&hier.body().getBlocks().front().front());
  for (Operation &op : llvm::make_early_inc_range(inst.getOps())) {
    // Child instances should have been lowered already.
    assert(!isa<DynamicInstanceOp>(op));
    op.remove();
    hierBlock.insert(&op);

    // Assign a ref for ops which need it.
    if (auto specOp = dyn_cast<DynInstDataOpInterface>(op)) {
      assert(ref);
      specOp.setGlobalRef(ref);
    }
  }

  inst.erase();
  return success();
}
void LowerInstancesPass::runOnOperation() {
  auto top = getOperation();
  auto *ctxt = &getContext();

  // Populate the top level symbol cache.
  topSyms.addDefinitions(top);

  size_t numFailed = 0;
  OpBuilder builder(ctxt);

  // Find all of the InstanceHierarchyOps.
  for (Operation &op : llvm::make_early_inc_range(top.getOps())) {
    auto instHierOp = dyn_cast<InstanceHierarchyOp>(op);
    if (!instHierOp)
      continue;
    builder.setInsertionPoint(&op);
    // Walk the child dynamic instances in _post-order_ so we lower and delete
    // the children first.
    instHierOp->walk<mlir::WalkOrder::PostOrder>([&](DynamicInstanceOp inst) {
      if (failed(lower(inst, instHierOp, builder)))
        ++numFailed;
    });
  }
  if (numFailed)
    signalPassFailure();

  // Since applying a large number of attributes is very expensive in MLIR (both
  // in terms of time and memory), bulk-apply the attributes necessary for
  // `hw.globalref`s.
  for (auto opRefPair : globalRefsToApply) {
    ArrayRef<hw::GlobalRefAttr> refArr = opRefPair.getSecond();
    SmallVector<Attribute> newGlobalRefs(
        llvm::map_range(refArr, [](hw::GlobalRefAttr ref) { return ref; }));
    Operation *op = opRefPair.getFirst();
    if (auto refArr =
            op->getAttrOfType<ArrayAttr>(hw::GlobalRefAttr::DialectAttrName))
      newGlobalRefs.append(refArr.getValue().begin(), refArr.getValue().end());
    op->setAttr(hw::GlobalRefAttr::DialectAttrName,
                ArrayAttr::get(ctxt, newGlobalRefs));
  }
}

namespace circt {
namespace msft {
std::unique_ptr<Pass> createLowerInstancesPass() {
  return std::make_unique<LowerInstancesPass>();
}
} // namespace msft
} // namespace circt

//===----------------------------------------------------------------------===//
// Lower MSFT to HW.
//===----------------------------------------------------------------------===//

namespace {
/// Lower MSFT's InstanceOp to HW's. Currently trivial since `msft.instance` is
/// currently a subset of `hw.instance`.
struct InstanceOpLowering : public OpConversionPattern<InstanceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(InstanceOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};
} // anonymous namespace

LogicalResult
InstanceOpLowering::matchAndRewrite(InstanceOp msftInst, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  Operation *referencedModule = msftInst.getReferencedModule();
  if (!referencedModule)
    return rewriter.notifyMatchFailure(msftInst,
                                       "Could not find referenced module");
  if (!hw::isAnyModule(referencedModule))
    return rewriter.notifyMatchFailure(
        msftInst, "Referenced module was not an HW module");

  ArrayAttr paramValues;
  if (isa<hw::HWModuleExternOp>(referencedModule)) {
    paramValues = msftInst.parametersAttr();
    if (!paramValues)
      paramValues = rewriter.getArrayAttr({});
  } else {
    auto instAppendParam = hw::ParamExprAttr::get(
        hw::PEO::StrConcat,
        {hw::ParamDeclRefAttr::get(rewriter.getStringAttr("__INST_HIER"),
                                   rewriter.getNoneType()),
         rewriter.getStringAttr("."), msftInst.sym_nameAttr()});
    paramValues = rewriter.getArrayAttr(
        {hw::ParamDeclAttr::get("__INST_HIER", instAppendParam)});
  }

  auto hwInst = rewriter.create<hw::InstanceOp>(
      msftInst.getLoc(), referencedModule, msftInst.instanceNameAttr(),
      SmallVector<Value>(adaptor.getOperands().begin(),
                         adaptor.getOperands().end()),
      paramValues, msftInst.sym_nameAttr());
  hwInst->setDialectAttrs(msftInst->getDialectAttrs());
  rewriter.replaceOp(msftInst, hwInst.getResults());
  return success();
}

namespace {
/// Lower MSFT's ModuleOp to HW's.
struct ModuleOpLowering : public OpConversionPattern<MSFTModuleOp> {
public:
  ModuleOpLowering(MLIRContext *context, StringRef outputFile)
      : OpConversionPattern::OpConversionPattern(context),
        outputFile(outputFile) {}

  LogicalResult
  matchAndRewrite(MSFTModuleOp mod, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;

private:
  StringRef outputFile;
};
} // anonymous namespace

LogicalResult
ModuleOpLowering::matchAndRewrite(MSFTModuleOp mod, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  if (mod.body().empty()) {
    std::string comment;
    llvm::raw_string_ostream(comment)
        << "// Module not generated: \"" << mod.getName() << "\" params "
        << mod.parameters();
    // TODO: replace this with proper comment op when it's created.
    rewriter.replaceOpWithNewOp<sv::VerbatimOp>(mod, comment);
    return success();
  }

  ArrayAttr params = rewriter.getArrayAttr({hw::ParamDeclAttr::get(
      rewriter.getStringAttr("__INST_HIER"),
      rewriter.getStringAttr("INSTANTIATE_WITH_INSTANCE_PATH"))});
  auto hwmod = rewriter.replaceOpWithNewOp<hw::HWModuleOp>(
      mod, mod.getNameAttr(), mod.getPorts(), params);
  rewriter.eraseBlock(hwmod.getBodyBlock());
  rewriter.inlineRegionBefore(mod.getBody(), hwmod.getBody(),
                              hwmod.getBody().end());

  auto opOutputFile = mod.fileName();
  if (opOutputFile) {
    auto outputFileAttr = hw::OutputFileAttr::getFromFilename(
        rewriter.getContext(), *opOutputFile, false, true);
    hwmod->setAttr("output_file", outputFileAttr);
  } else if (!outputFile.empty()) {
    auto outputFileAttr = hw::OutputFileAttr::getFromFilename(
        rewriter.getContext(), outputFile, false, true);
    hwmod->setAttr("output_file", outputFileAttr);
  }

  return success();
}
namespace {

/// Lower MSFT's ModuleExternOp to HW's.
struct ModuleExternOpLowering : public OpConversionPattern<MSFTModuleExternOp> {
public:
  ModuleExternOpLowering(MLIRContext *context, StringRef outputFile)
      : OpConversionPattern::OpConversionPattern(context),
        outputFile(outputFile) {}

  LogicalResult
  matchAndRewrite(MSFTModuleExternOp mod, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;

private:
  StringRef outputFile;
};
} // anonymous namespace

LogicalResult ModuleExternOpLowering::matchAndRewrite(
    MSFTModuleExternOp mod, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto hwMod = rewriter.replaceOpWithNewOp<hw::HWModuleExternOp>(
      mod, mod.getNameAttr(), mod.getPorts(), mod.verilogName().value_or(""),
      mod.parameters());

  if (!outputFile.empty()) {
    auto outputFileAttr = hw::OutputFileAttr::getFromFilename(
        rewriter.getContext(), outputFile, false, true);
    hwMod->setAttr("output_file", outputFileAttr);
  }

  return success();
}

namespace {
/// Lower MSFT's OutputOp to HW's.
struct OutputOpLowering : public OpConversionPattern<OutputOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OutputOp out, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<hw::OutputOp>(out, out.getOperands());
    return success();
  }
};
} // anonymous namespace

namespace {
/// Simply remove the OpTy op when done.
template <typename OpTy>
struct RemoveOpLowering : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<OpTy>::OpAdaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
    return success();
  }
};
} // anonymous namespace

namespace {
struct LowerToHWPass : public LowerToHWBase<LowerToHWPass> {
  void runOnOperation() override;
};
} // anonymous namespace

void LowerToHWPass::runOnOperation() {
  auto top = getOperation();
  auto *ctxt = &getContext();

  // The `hw::InstanceOp` (which `msft::InstanceOp` lowers to) convenience
  // builder gets its argNames and resultNames from the `hw::HWModuleOp`. So we
  // have to lower `msft::MSFTModuleOp` before we lower `msft::InstanceOp`.

  // Convert everything except instance ops first.

  ConversionTarget target(*ctxt);
  target.addIllegalOp<MSFTModuleOp, MSFTModuleExternOp, OutputOp>();
  target.addLegalDialect<hw::HWDialect>();
  target.addLegalDialect<sv::SVDialect>();

  RewritePatternSet patterns(ctxt);
  patterns.insert<ModuleOpLowering>(ctxt, verilogFile);
  patterns.insert<ModuleExternOpLowering>(ctxt, verilogFile);
  patterns.insert<OutputOpLowering>(ctxt);
  patterns.insert<RemoveOpLowering<EntityExternOp>>(ctxt);
  patterns.insert<RemoveOpLowering<DesignPartitionOp>>(ctxt);

  if (failed(applyPartialConversion(top, target, std::move(patterns))))
    signalPassFailure();

  // Then, convert the InstanceOps
  target.addDynamicallyLegalDialect<MSFTDialect>([](Operation *op) {
    return isa<DynInstDataOpInterface, DeclPhysicalRegionOp,
               InstanceHierarchyOp>(op);
  });
  RewritePatternSet instancePatterns(ctxt);
  instancePatterns.insert<InstanceOpLowering>(ctxt);
  if (failed(applyPartialConversion(top, target, std::move(instancePatterns))))
    signalPassFailure();
}

namespace circt {
namespace msft {
std::unique_ptr<Pass> createLowerToHWPass() {
  return std::make_unique<LowerToHWPass>();
}
} // namespace msft
} // namespace circt

//===----------------------------------------------------------------------===//
// Export tcl -- create tcl verbatim ops
//===----------------------------------------------------------------------===//

namespace {
template <typename PhysOpTy>
struct RemovePhysOpLowering : public OpConversionPattern<PhysOpTy> {
  using OpConversionPattern<PhysOpTy>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<PhysOpTy>::OpAdaptor;

  LogicalResult
  matchAndRewrite(PhysOpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
    return success();
  }
};
} // anonymous namespace

namespace {
struct ExportTclPass : public ExportTclBase<ExportTclPass> {
  void runOnOperation() override;
};
} // anonymous namespace

void ExportTclPass::runOnOperation() {
  auto top = getOperation();
  auto *ctxt = &getContext();
  TclEmitter emitter(top);

  // Traverse MSFT location attributes and export the required Tcl into
  // templated `sv::VerbatimOp`s with symbolic references to the instance paths.
  for (std::string moduleName : tops) {
    Operation *hwmod =
        emitter.getDefinition(FlatSymbolRefAttr::get(ctxt, moduleName));
    if (!hwmod) {
      top.emitError("Failed to find module '") << moduleName << "'";
      signalPassFailure();
      return;
    }
    if (failed(emitter.emit(hwmod, tclFile))) {
      hwmod->emitError("failed to emit tcl");
      signalPassFailure();
      return;
    }
  }

  ConversionTarget target(*ctxt);
  target.addIllegalDialect<msft::MSFTDialect>();
  target.addLegalDialect<hw::HWDialect>();
  target.addLegalDialect<sv::SVDialect>();

  RewritePatternSet patterns(ctxt);
  patterns.insert<RemovePhysOpLowering<PDPhysLocationOp>>(ctxt);
  patterns.insert<RemovePhysOpLowering<PDRegPhysLocationOp>>(ctxt);
  patterns.insert<RemovePhysOpLowering<PDPhysRegionOp>>(ctxt);
  patterns.insert<RemovePhysOpLowering<InstanceHierarchyOp>>(ctxt);
  patterns.insert<RemovePhysOpLowering<DynamicInstanceVerbatimAttrOp>>(ctxt);
  patterns.insert<RemoveOpLowering<DeclPhysicalRegionOp>>(ctxt);
  if (failed(applyPartialConversion(top, target, std::move(patterns))))
    signalPassFailure();

  target.addDynamicallyLegalOp<hw::GlobalRefOp>([&](hw::GlobalRefOp ref) {
    return !emitter.getRefsUsed().contains(ref);
  });
  patterns.clear();
  patterns.insert<RemoveOpLowering<hw::GlobalRefOp>>(ctxt);
  if (failed(applyPartialConversion(top, target, std::move(patterns))))
    signalPassFailure();
}

namespace circt {
namespace msft {
std::unique_ptr<Pass> createExportTclPass() {
  return std::make_unique<ExportTclPass>();
}
} // namespace msft
} // namespace circt

//===----------------------------------------------------------------------===//
// Wire and partitioning passes
//===----------------------------------------------------------------------===//

namespace {
struct MSFTPassCommon : PassCommon {
protected:
  /// Update all the instantiations of 'mod' to match the port list. For any
  /// output ports which survived, automatically map the result according to
  /// `newToOldResultMap`. Calls 'getOperandsFunc' with the new instance op, the
  /// old instance op, and expects the operand vector to return filled.
  /// `getOperandsFunc` can (and often does) modify other operations. The update
  /// call deletes the original instance op, so all references are invalidated
  /// after this call.
  SmallVector<InstanceOp, 1> updateInstances(
      MSFTModuleOp mod, ArrayRef<unsigned> newToOldResultMap,
      llvm::function_ref<void(InstanceOp, InstanceOp, SmallVectorImpl<Value> &)>
          getOperandsFunc);

  void getAndSortModules(ModuleOp topMod, SmallVectorImpl<MSFTModuleOp> &mods);

  void bubbleWiresUp(MSFTModuleOp mod);
  void dedupOutputs(MSFTModuleOp mod);
  void sinkWiresDown(MSFTModuleOp mod);
  void dedupInputs(MSFTModuleOp mod);
};
} // anonymous namespace

/// Is this operation "free" and copy-able?
static bool isWireManipulationOp(Operation *op) {
  return isa<hw::ArrayConcatOp, hw::ArrayCreateOp, hw::ArrayGetOp,
             hw::ArraySliceOp, hw::StructCreateOp, hw::StructExplodeOp,
             hw::StructExtractOp, hw::StructInjectOp, hw::StructCreateOp,
             hw::ConstantOp>(op);
}

SmallVector<InstanceOp, 1> MSFTPassCommon::updateInstances(
    MSFTModuleOp mod, ArrayRef<unsigned> newToOldResultMap,
    llvm::function_ref<void(InstanceOp, InstanceOp, SmallVectorImpl<Value> &)>
        getOperandsFunc) {

  SmallVector<hw::HWInstanceLike, 1> newInstances;
  SmallVector<InstanceOp, 1> newMsftInstances;
  for (hw::HWInstanceLike instLike : moduleInstantiations[mod]) {
    assert(instLike->getParentOp());
    auto inst = dyn_cast<InstanceOp>(instLike.getOperation());
    if (!inst) {
      instLike.emitWarning("Can not update hw.instance ops");
      continue;
    }

    OpBuilder b(inst);
    auto newInst = b.create<InstanceOp>(inst.getLoc(), mod.getResultTypes(),
                                        inst.getOperands(), inst->getAttrs());

    SmallVector<Value> newOperands;
    getOperandsFunc(newInst, inst, newOperands);
    newInst->setOperands(newOperands);

    for (auto oldResult : llvm::enumerate(newToOldResultMap))
      if (oldResult.value() < inst.getNumResults())
        inst.getResult(oldResult.value())
            .replaceAllUsesWith(newInst.getResult(oldResult.index()));

    newInstances.push_back(newInst);
    newMsftInstances.push_back(newInst);
    inst->dropAllUses();
    inst->erase();
  }
  moduleInstantiations[mod].swap(newInstances);
  return newMsftInstances;
}

// Run a post-order DFS.
void PassCommon::getAndSortModulesVisitor(
    hw::HWModuleLike mod, SmallVectorImpl<hw::HWModuleLike> &mods,
    DenseSet<Operation *> &modsSeen) {
  if (modsSeen.contains(mod))
    return;
  modsSeen.insert(mod);

  mod.walk([&](hw::HWInstanceLike inst) {
    Operation *modOp =
        topLevelSyms.getDefinition(inst.referencedModuleNameAttr());
    assert(modOp);
    moduleInstantiations[modOp].push_back(inst);
    if (auto modLike = dyn_cast<hw::HWModuleLike>(modOp))
      getAndSortModulesVisitor(modLike, mods, modsSeen);
  });

  mods.push_back(mod);
}

void MSFTPassCommon::getAndSortModules(ModuleOp topMod,
                                       SmallVectorImpl<MSFTModuleOp> &mods) {
  SmallVector<hw::HWModuleLike, 16> moduleLikes;
  PassCommon::getAndSortModules(topMod, moduleLikes);
  mods.clear();
  for (auto modLike : moduleLikes) {
    auto mod = dyn_cast<MSFTModuleOp>(modLike.getOperation());
    if (mod)
      mods.push_back(mod);
  }
}

void PassCommon::getAndSortModules(ModuleOp topMod,
                                   SmallVectorImpl<hw::HWModuleLike> &mods) {
  // Add here _before_ we go deeper to prevent infinite recursion.
  DenseSet<Operation *> modsSeen;
  mods.clear();
  moduleInstantiations.clear();
  topMod.walk([&](hw::HWModuleLike mod) {
    getAndSortModulesVisitor(mod, mods, modsSeen);
  });
}

LogicalResult PassCommon::verifyInstances(mlir::ModuleOp mod) {
  WalkResult r = mod.walk([&](InstanceOp inst) {
    Operation *modOp = topLevelSyms.getDefinition(inst.moduleNameAttr());
    if (!isAnyModule(modOp))
      return WalkResult::interrupt();

    hw::ModulePortInfo ports = getModulePortInfo(modOp);
    return succeeded(inst.verifySignatureMatch(ports))
               ? WalkResult::advance()
               : WalkResult::interrupt();
  });
  return failure(r.wasInterrupted());
}

namespace {
struct PartitionPass : public PartitionBase<PartitionPass>, MSFTPassCommon {
  void runOnOperation() override;

private:
  void partition(MSFTModuleOp mod);
  MSFTModuleOp partition(DesignPartitionOp part, Block *partBlock);

  void bubbleUp(MSFTModuleOp mod, Block *ops);
  void bubbleUpGlobalRefs(Operation *op, StringAttr parentMod,
                          StringAttr parentName,
                          llvm::DenseSet<hw::GlobalRefAttr> &refsMoved);
  void pushDownGlobalRefs(Operation *op, DesignPartitionOp partOp,
                          llvm::SetVector<Attribute> &newGlobalRefs);

  // Tag wire manipulation ops in this module.
  static void
  copyWireOps(MSFTModuleOp,
              DenseMap<SymbolRefAttr, DenseSet<Operation *>> &perPartOpsToMove);

  MLIRContext *ctxt;
};
} // anonymous namespace

void PartitionPass::runOnOperation() {
  ModuleOp outerMod = getOperation();
  ctxt = outerMod.getContext();
  topLevelSyms.addDefinitions(outerMod);
  if (failed(verifyInstances(outerMod))) {
    signalPassFailure();
    return;
  }

  // Get a properly sorted list, then partition the mods in order.
  SmallVector<MSFTModuleOp, 64> sortedMods;
  getAndSortModules(outerMod, sortedMods);

  for (auto mod : sortedMods) {
    // Make partition's job easier by cleaning up first.
    (void)mlir::applyPatternsAndFoldGreedily(mod,
                                             mlir::FrozenRewritePatternSet());
    // Do the partitioning.
    partition(mod);
    // Cleanup whatever mess we made.
    (void)mlir::applyPatternsAndFoldGreedily(mod,
                                             mlir::FrozenRewritePatternSet());
  }
}

/// Determine if 'op' is driven exclusively by other tagged ops or wires which
/// are themselves exclusively driven by tagged ops. Recursive but memoized via
/// `seen`.
static bool isDrivenByPartOpsOnly(Operation *op,
                                  const BlockAndValueMapping &partOps,
                                  DenseMap<Operation *, bool> &seen) {
  auto prevResult = seen.find(op);
  if (prevResult != seen.end())
    return prevResult->second;
  bool &result = seen[op];
  // Default to true.
  result = true;

  for (Value oper : op->getOperands()) {
    if (partOps.contains(oper))
      continue;
    if (oper.isa<BlockArgument>())
      continue;
    Operation *defOp = oper.getDefiningOp();
    if (!isWireManipulationOp(defOp) ||
        !isDrivenByPartOpsOnly(defOp, partOps, seen))
      result = false;
  }
  return result;
}

/// Move the list of tagged operations in to 'partBlock' and copy/move any free
/// (wire) ops connecting them in also. If 'extendMaximalUp` is specified,
/// attempt to copy all the way up to the block args.
void copyIntoPart(ArrayRef<Operation *> taggedOps, Block *partBlock,
                  bool extendMaximalUp) {
  BlockAndValueMapping map;
  if (taggedOps.empty())
    return;
  OpBuilder b(taggedOps[0]->getContext());
  // Copy all of the ops listed.
  for (Operation *op : taggedOps) {
    op->moveBefore(partBlock, partBlock->end());
    for (Value result : op->getResults())
      map.map(result, result);
  }

  // Memoization space.
  DenseMap<Operation *, bool> seen;

  // Treat the 'partBlock' as a queue, iterating through and appending as
  // necessary.
  for (Operation &op : *partBlock) {
    // Make sure we are always appending.
    b.setInsertionPointToEnd(partBlock);

    // Go through the operands and copy any which we can.
    for (auto &opOper : op.getOpOperands()) {
      Value operValue = opOper.get();
      assert(operValue);

      // Check if there's already a copied op for this value.
      Value existingValue = map.lookupOrNull(operValue);
      if (existingValue) {
        opOper.set(existingValue);
        continue;
      }

      // Determine if we can copy the op into our partition.
      Operation *defOp = operValue.getDefiningOp();
      if (!defOp)
        continue;

      // We don't copy anything which isn't "free".
      if (!isWireManipulationOp(defOp))
        continue;

      // Copy operand wire ops into the partition.
      //   If `extendMaximalUp` is set, we want to copy unconditionally.
      //   Otherwise, we only want to copy wire ops which connect this operation
      //   to another in the partition.
      if (extendMaximalUp || isDrivenByPartOpsOnly(defOp, map, seen)) {
        // Optimization: if all the consumers of this wire op are in the
        // partition, move instead of clone.
        if (llvm::all_of(defOp->getUsers(), [&](Operation *user) {
              return user->getBlock() == partBlock;
            })) {
          defOp->moveBefore(partBlock, b.getInsertionPoint());
        } else {
          b.insert(defOp->clone(map));
          opOper.set(map.lookup(opOper.get()));
        }
      }
    }

    // Move any "free" consumers which we can.
    for (auto *user : llvm::make_early_inc_range(op.getUsers())) {
      // Stop if it's not "free" or already in a partition.
      if (!isWireManipulationOp(user) || getPart(user) ||
          user->getBlock() == partBlock)
        continue;
      // Op must also only have its operands driven (or indirectly driven) by
      // ops in the partition.
      if (!isDrivenByPartOpsOnly(user, map, seen))
        continue;

      // All the conditions are met, move it!
      user->moveBefore(partBlock, partBlock->end());
      // Mark it as being in the block by putting it into the map.
      for (Value result : user->getResults())
        map.map(result, result);
      // Re-map the inputs to results in the block, if they exist.
      for (OpOperand &oper : user->getOpOperands())
        oper.set(map.lookupOrDefault(oper.get()));
    }
  }
}

/// Move tagged ops into separate blocks. Copy any wire ops connecting them as
/// well.
void copyInto(MSFTModuleOp mod, DenseMap<SymbolRefAttr, Block *> &perPartBlocks,
              Block *nonLocalBlock) {
  DenseMap<SymbolRefAttr, SmallVector<Operation *, 8>> perPartTaggedOps;
  SmallVector<Operation *, 16> nonLocalTaggedOps;

  // Bucket the ops by partition tag.
  mod.walk([&](Operation *op) {
    auto partRef = getPart(op);
    if (!partRef)
      return;
    auto partBlockF = perPartBlocks.find(partRef);
    if (partBlockF != perPartBlocks.end())
      perPartTaggedOps[partRef].push_back(op);
    else
      nonLocalTaggedOps.push_back(op);
  });

  // Copy into the appropriate partition block.
  for (auto &partOpQueuePair : perPartTaggedOps) {
    copyIntoPart(partOpQueuePair.second, perPartBlocks[partOpQueuePair.first],
                 false);
  }
  copyIntoPart(nonLocalTaggedOps, nonLocalBlock, true);
}

void PartitionPass::partition(MSFTModuleOp mod) {
  auto modSymbol = SymbolTable::getSymbolName(mod);

  // Construct all the blocks we're going to need.
  Block *nonLocal = mod.addBlock();
  DenseMap<SymbolRefAttr, Block *> perPartBlocks;
  mod.walk([&](DesignPartitionOp part) {
    SymbolRefAttr partRef =
        SymbolRefAttr::get(modSymbol, {SymbolRefAttr::get(part)});
    perPartBlocks[partRef] = mod.addBlock();
  });

  // Sort the tagged ops into ops to hoist (bubble up) and per-partition blocks.
  copyInto(mod, perPartBlocks, nonLocal);

  // Hoist the appropriate ops and erase the partition block.
  if (!nonLocal->empty())
    bubbleUp(mod, nonLocal);
  nonLocal->dropAllReferences();
  nonLocal->dropAllDefinedValueUses();
  mod.getBlocks().remove(nonLocal);

  // Sink all of the "locally-tagged" ops into new partition modules.
  for (auto part :
       llvm::make_early_inc_range(mod.getOps<DesignPartitionOp>())) {
    SymbolRefAttr partRef =
        SymbolRefAttr::get(modSymbol, {SymbolRefAttr::get(part)});
    Block *partBlock = perPartBlocks[partRef];
    partition(part, partBlock);
    part.erase();
  }
}

/// Heuristics to get the entity name.
static StringRef getOpName(Operation *op) {
  StringAttr name;
  if ((name = op->getAttrOfType<StringAttr>("name")) && name.size())
    return name.getValue();
  if ((name = op->getAttrOfType<StringAttr>("sym_name")) && name.size())
    return name.getValue();
  return op->getName().getStringRef();
}
/// Try to set the entity name.
/// TODO: this needs to be more complex to deal with renaming symbols.
static void setEntityName(Operation *op, Twine name) {
  StringAttr nameAttr = StringAttr::get(op->getContext(), name);
  if (op->hasAttrOfType<StringAttr>("name"))
    op->setAttr("name", nameAttr);
  if (op->hasAttrOfType<StringAttr>("sym_name"))
    op->setAttr("sym_name", nameAttr);
}

/// Try to get a "good" name for the given Value.
static StringRef getValueName(Value v, const SymbolCache &syms,
                              std::string &buff) {
  Operation *defOp = v.getDefiningOp();
  if (auto inst = dyn_cast_or_null<InstanceOp>(defOp)) {
    Operation *modOp = syms.getDefinition(inst.moduleNameAttr());
    if (modOp) { // If modOp isn't in the cache, it's probably a new module;
      assert(isAnyModule(modOp) && "Instance must point to a module");
      OpResult instResult = v.cast<OpResult>();
      hw::ModulePortInfo ports = getModulePortInfo(modOp);
      buff.clear();
      llvm::raw_string_ostream os(buff);
      os << inst.sym_name() << ".";
      StringAttr name = ports.outputs[instResult.getResultNumber()].name;
      if (name)
        os << name.getValue();
      return buff;
    }
  }
  if (auto blockArg = v.dyn_cast<BlockArgument>()) {
    auto portInfo =
        getModulePortInfo(blockArg.getOwner()->getParent()->getParentOp());
    return portInfo.inputs[blockArg.getArgNumber()].getName();
  }
  if (auto constOp = dyn_cast<hw::ConstantOp>(defOp)) {
    buff.clear();
    llvm::raw_string_ostream(buff) << "c" << constOp.getValue();
    return buff;
  }

  return "";
}

/// Heuristics to get the output name.
static StringRef getResultName(OpResult res, const SymbolCache &syms,
                               std::string &buff) {

  StringRef valName = getValueName(res, syms, buff);
  if (!valName.empty())
    return valName;
  if (res.getOwner()->getNumResults() == 1)
    return {};

  // Fallback. Not ideal.
  buff.clear();
  llvm::raw_string_ostream(buff) << "out" << res.getResultNumber();
  return buff;
}

/// Heuristics to get the input name.
static StringRef getOperandName(OpOperand &oper, const SymbolCache &syms,
                                std::string &buff) {
  Operation *op = oper.getOwner();
  if (auto inst = dyn_cast<InstanceOp>(op)) {
    Operation *modOp = syms.getDefinition(inst.moduleNameAttr());
    if (modOp) { // If modOp isn't in the cache, it's probably a new module;
      assert(isAnyModule(modOp) && "Instance must point to a module");
      hw::ModulePortInfo ports = getModulePortInfo(modOp);
      return ports.inputs[oper.getOperandNumber()].name;
    }
  }
  if (auto blockArg = oper.get().dyn_cast<BlockArgument>()) {
    auto portInfo =
        getModulePortInfo(blockArg.getOwner()->getParent()->getParentOp());
    return portInfo.inputs[blockArg.getArgNumber()].getName();
  }

  if (oper.getOwner()->getNumOperands() == 1)
    return "in";

  // Fallback. Not ideal.
  buff.clear();
  llvm::raw_string_ostream(buff) << "in" << oper.getOperandNumber();
  return buff;
}

/// Helper to get the circt.globalRef attribute.
static ArrayAttr getGlobalRefs(Operation *op) {
  return op->getAttrOfType<ArrayAttr>(hw::GlobalRefAttr::DialectAttrName);
}

/// Helper to update GlobalRefOps after referenced ops bubble up.
void PartitionPass::bubbleUpGlobalRefs(
    Operation *op, StringAttr parentMod, StringAttr parentName,
    llvm::DenseSet<hw::GlobalRefAttr> &refsMoved) {
  auto globalRefs = getGlobalRefs(op);
  if (!globalRefs)
    return;

  // GlobalRefs use the inner_sym attribute, so keep it up to date.
  auto oldInnerSym = op->getAttrOfType<StringAttr>("inner_sym");
  auto newInnerSym = StringAttr::get(op->getContext(), ::getOpName(op));
  op->setAttr("inner_sym", newInnerSym);

  for (auto globalRef : globalRefs.getAsRange<hw::GlobalRefAttr>()) {
    // Resolve the GlobalRefOp and get its path.
    auto refSymbol = globalRef.getGlblSym();
    auto globalRefOp = dyn_cast_or_null<hw::GlobalRefOp>(
        topLevelSyms.getDefinition(refSymbol));
    assert(globalRefOp && "symbol must reference a GlobalRefOp");
    auto oldPath = globalRefOp.getNamepath().getValue();
    assert(!oldPath.empty());

    // If the path already points to the target design partition, we are done.
    auto leafModule = oldPath.back().cast<hw::InnerRefAttr>().getModule();
    auto partAttr = op->getAttrOfType<SymbolRefAttr>("targetDesignPartition");
    if (partAttr.getRootReference() == leafModule)
      return;
    assert(oldPath.size() > 1);

    // Find the index of the node in the path that points to the opName. The
    // previous node in the path must point to parentName.
    size_t opIndex = 0;
    bool found = false;
    (void)found;
    for (; opIndex < oldPath.size(); ++opIndex) {
      auto oldNode = oldPath[opIndex].cast<hw::InnerRefAttr>();
      if (oldNode.getModule() == parentMod &&
          oldNode.getName() == oldInnerSym) {
        found = true;
        break;
      }
    }

    assert(found && opIndex > 0);
    auto parentIndex = opIndex - 1;
    auto parentNode = oldPath[parentIndex].cast<hw::InnerRefAttr>();
    assert(parentNode.getName() == parentName);

    // Split the old path into two chunks: the parent chunk is everything before
    // the node pointing to parentName, and the child chunk is everything after
    // the node pointing to opName.
    auto parentChunk = oldPath.take_front(parentIndex);
    auto childChunk = oldPath.take_back((oldPath.size() - 1) - opIndex);

    // Splice together the nodes that parentName and opName point to.
    auto splicedNode =
        hw::InnerRefAttr::get(parentNode.getModule(), newInnerSym);

    // Construct a new path from the parentChunk, splicedNode, and childChunk.
    SmallVector<Attribute> newPath(parentChunk.begin(), parentChunk.end());
    newPath.push_back(splicedNode);
    newPath.append(childChunk.begin(), childChunk.end());

    // Update the path on the GlobalRefOp.
    auto newPathAttr = ArrayAttr::get(op->getContext(), newPath);
    globalRefOp.setNamepathAttr(newPathAttr);

    refsMoved.insert(globalRef);
  }
}

/// Helper to update GlobalRefops after referenced ops are pushed down.
void PartitionPass::pushDownGlobalRefs(
    Operation *op, DesignPartitionOp partOp,
    llvm::SetVector<Attribute> &newGlobalRefs) {
  auto globalRefs = getGlobalRefs(op);
  if (!globalRefs)
    return;

  for (auto globalRef : globalRefs.getAsRange<hw::GlobalRefAttr>()) {
    // Resolve the GlobalRefOp and get its path.
    auto refSymbol = globalRef.getGlblSym();
    auto globalRefOp = dyn_cast_or_null<hw::GlobalRefOp>(
        topLevelSyms.getDefinition(refSymbol));
    assert(globalRefOp && "symbol must reference a GlobalRefOp");
    auto oldPath = globalRefOp.getNamepath().getValue();
    assert(!oldPath.empty());

    // Get the module containing the partition and the partition's name.
    auto partAttr = op->getAttrOfType<SymbolRefAttr>("targetDesignPartition");
    auto partMod = partAttr.getRootReference();
    auto partName = partAttr.getLeafReference();
    auto partModName = partOp.verilogNameAttr();
    assert(partModName);

    // Find the index of the node in the path that points to the innerSym.
    auto innerSym = op->getAttrOfType<StringAttr>("inner_sym");
    size_t opIndex = 0;
    bool found = false;
    for (; opIndex < oldPath.size(); ++opIndex) {
      auto oldNode = oldPath[opIndex].cast<hw::InnerRefAttr>();
      if (oldNode.getModule() == partMod && oldNode.getName() == innerSym) {
        found = true;
        break;
      }
    }

    (void)found;
    assert(found);

    // If this path already points to the design partition, we are done.
    if (oldPath[opIndex].cast<hw::InnerRefAttr>().getModule() == partModName)
      return;

    // Split the old path into two chunks: the parent chunk is everything before
    // the node pointing to innerSym, and the child chunk is everything after
    // the node pointing to innerSym.
    auto parentChunk = oldPath.take_front(opIndex);
    auto childChunk = oldPath.take_back((oldPath.size() - 1) - opIndex);

    // Create a new node for the partition within the partition's parent module,
    // and a new node for the op within the partition module.
    auto partRef = hw::InnerRefAttr::get(partMod, partName);
    auto leafRef = hw::InnerRefAttr::get(partModName, innerSym);

    // Construct a new path from the parentChunk, partRef, leafRef, and
    // childChunk.
    SmallVector<Attribute> newPath(parentChunk.begin(), parentChunk.end());
    newPath.push_back(partRef);
    newPath.push_back(leafRef);
    newPath.append(childChunk.begin(), childChunk.end());

    // Update the path on the GlobalRefOp.
    auto newPathAttr = ArrayAttr::get(op->getContext(), newPath);
    globalRefOp.setNamepathAttr(newPathAttr);

    // Ensure the part instance will have this GlobalRefAttr.
    // global refs if not.
    newGlobalRefs.insert(globalRef);
  }
}

/// Utility for creating {0, 1, 2, ..., size}.
static SmallVector<unsigned> makeSequentialRange(unsigned size) {
  SmallVector<unsigned> seq;
  for (size_t i = 0; i < size; ++i)
    seq.push_back(i);
  return seq;
}

void PartitionPass::bubbleUp(MSFTModuleOp mod, Block *partBlock) {
  auto *ctxt = mod.getContext();
  FunctionType origType = mod.getFunctionType();
  std::string nameBuffer;

  //*************
  //   Figure out all the new ports 'mod' is going to need. The outputs need to
  //   know where they're being driven from, which'll be some of the outputs of
  //   'ops'. Also determine which of the existing ports are no longer used.
  //
  //   Don't do any mutation here, just assemble bookkeeping info.

  // The new input ports for operands not defined in 'partBlock'.
  SmallVector<std::pair<StringAttr, Type>, 64> newInputs;
  // Map the operand value to new input port.
  DenseMap<Value, size_t> oldValueNewResultNum;

  // The new output ports.
  SmallVector<std::pair<StringAttr, Value>, 64> newOutputs;
  // Store the original result value in new port order. Used later on to remap
  // the moved operations to the new block arguments.
  SmallVector<Value, 64> newInputOldValue;

  for (Operation &op : *partBlock) {
    StringRef opName = ::getOpName(&op);
    if (opName.empty())
      opName = op.getName().getIdentifier().getValue();

    // Tagged operation might need new inputs ports to drive its consumers.
    for (OpResult res : op.getOpResults()) {
      // If all the operations will get moved, no new port is necessary.
      if (llvm::all_of(res.getUsers(), [partBlock](Operation *op) {
            return op->getBlock() == partBlock || isa<OutputOp>(op);
          }))
        continue;

      // Create a new inpurt port.
      StringRef name = getResultName(res, topLevelSyms, nameBuffer);
      newInputs.push_back(std::make_pair(
          StringAttr::get(ctxt, opName + (name.empty() ? "" : "." + name)),
          res.getType()));
      newInputOldValue.push_back(res);
    }

    // Tagged operations may need new output ports to drive their operands.
    for (OpOperand &oper : op.getOpOperands()) {
      Value operVal = oper.get();

      // If the value was coming from outside the module, unnecessary.
      if (auto operArg = operVal.dyn_cast<BlockArgument>())
        continue;

      Operation *defOp = operVal.getDefiningOp();
      assert(defOp && "Value must be operation if not block arg");
      // New port unnecessary if source will be moved or there's already a port
      // for that value.
      if (defOp->getBlock() == partBlock || oldValueNewResultNum.count(operVal))
        continue;

      // Create a new output port.
      oldValueNewResultNum[oper.get()] = newOutputs.size();
      StringRef name = getOperandName(oper, topLevelSyms, nameBuffer);
      newOutputs.push_back(std::make_pair(
          StringAttr::get(ctxt, opName + (name.empty() ? "" : "." + name)),
          operVal));
    }
  }

  // Figure out which of the original output ports can be removed.
  llvm::BitVector outputsToRemove(origType.getNumResults() + newOutputs.size());
  DenseMap<size_t, Value> oldResultOldValues;
  Operation *term = mod.getBodyBlock()->getTerminator();
  assert(term && "Invalid IR");
  for (auto outputValIdx : llvm::enumerate(term->getOperands())) {
    Operation *defOp = outputValIdx.value().getDefiningOp();
    if (!defOp || defOp->getBlock() != partBlock)
      continue;
    outputsToRemove.set(outputValIdx.index());
    oldResultOldValues[outputValIdx.index()] = outputValIdx.value();
  }

  // Figure out which of the original input ports will no longer be used and can
  // be removed.
  llvm::BitVector inputsToRemove(origType.getNumInputs() + newInputs.size());
  for (auto blockArg : mod.getBodyBlock()->getArguments()) {
    if (llvm::all_of(blockArg.getUsers(), [&](Operation *op) {
          return op->getBlock() == partBlock;
        }))
      inputsToRemove.set(blockArg.getArgNumber());
  }

  //*************
  //   Add the new ports and re-wire the operands using the new ports. The
  //   `addPorts` method handles adding the correct values to the terminator op.
  SmallVector<BlockArgument> newBlockArgs = mod.addPorts(newInputs, newOutputs);
  for (size_t inputNum = 0, e = newBlockArgs.size(); inputNum < e; ++inputNum)
    for (OpOperand &use : newInputOldValue[inputNum].getUses())
      if (use.getOwner()->getBlock() != partBlock)
        use.set(newBlockArgs[inputNum]);

  //*************
  //   For all of the instantiation sites (for 'mod'):
  //     - Create a new instance with the correct result types.
  //     - Clone in 'ops'.
  //     - Fix up the new operations' operands.
  auto cloneOpsGetOperands = [&](InstanceOp newInst, InstanceOp oldInst,
                                 SmallVectorImpl<Value> &newOperands) {
    OpBuilder b(newInst);
    BlockAndValueMapping map;

    // Add all of 'mod''s block args to the map in case one of the tagged ops
    // was driven by a block arg. Map to the oldInst operand Value.
    unsigned oldInstNumInputs = oldInst.getNumOperands();
    for (BlockArgument arg : mod.getBodyBlock()->getArguments())
      if (arg.getArgNumber() < oldInstNumInputs)
        map.map(arg, oldInst.getOperand(arg.getArgNumber()));

    // Add all the old values which got moved to output ports to the map.
    size_t origNumResults = origType.getNumResults();
    for (auto valueResultNum : oldValueNewResultNum)
      map.map(valueResultNum.first,
              newInst->getResult(origNumResults + valueResultNum.second));

    // Clone the ops, rename appropriately, and update the global refs.
    llvm::SmallVector<Operation *, 32> newOps;
    llvm::DenseSet<hw::GlobalRefAttr> movedRefs;
    for (Operation &op : *partBlock) {
      Operation *newOp = b.insert(op.clone(map));
      newOps.push_back(newOp);
      setEntityName(newOp, oldInst.getName() + "." + ::getOpName(&op));
      auto *oldInstMod = oldInst.getReferencedModule();
      assert(oldInstMod);
      auto oldModName = oldInstMod->getAttrOfType<StringAttr>("sym_name");
      bubbleUpGlobalRefs(newOp, oldModName, oldInst.getNameAttr(), movedRefs);
    }

    // Remove the hoisted global refs from new instance.
    if (ArrayAttr oldInstRefs = oldInst->getAttrOfType<ArrayAttr>(
            hw::GlobalRefAttr::DialectAttrName)) {
      llvm::SmallVector<Attribute> newInstRefs;
      for (Attribute oldRef : oldInstRefs.getValue()) {
        if (hw::GlobalRefAttr ref = oldRef.dyn_cast<hw::GlobalRefAttr>())
          if (movedRefs.contains(ref))
            continue;
        newInstRefs.push_back(oldRef);
      }
      if (newInstRefs.empty())
        newInst->removeAttr(hw::GlobalRefAttr::DialectAttrName);
      else
        newInst->setAttr(hw::GlobalRefAttr::DialectAttrName,
                         ArrayAttr::get(ctxt, newInstRefs));
    }

    // Fix up operands of cloned ops (backedges didn't exist in the map so they
    // didn't get mapped during the initial clone).
    for (Operation *newOp : newOps)
      for (OpOperand &oper : newOp->getOpOperands())
        oper.set(map.lookupOrDefault(oper.get()));

    // Since we're not removing any ports, start with the old operands.
    newOperands.append(oldInst.getOperands().begin(),
                       oldInst.getOperands().end());
    // Gather new operands for the new instance.
    for (Value oldValue : newInputOldValue)
      newOperands.push_back(map.lookup(oldValue));

    // Fix up existing ops which used the old instance's results.
    for (auto oldResultOldValue : oldResultOldValues)
      oldInst.getResult(oldResultOldValue.first)
          .replaceAllUsesWith(map.lookup(oldResultOldValue.second));
  };
  updateInstances(mod, makeSequentialRange(origType.getNumResults()),
                  cloneOpsGetOperands);

  //*************
  //   Lastly, remove the unnecessary ports. Doing this as a separate mutation
  //   makes the previous steps simpler without any practical degradation.
  SmallVector<unsigned> resValues =
      mod.removePorts(inputsToRemove, outputsToRemove);
  updateInstances(mod, resValues,
                  [&](InstanceOp newInst, InstanceOp oldInst,
                      SmallVectorImpl<Value> &newOperands) {
                    for (auto oldOperand :
                         llvm::enumerate(oldInst->getOperands()))
                      if (!inputsToRemove.test(oldOperand.index()))
                        newOperands.push_back(oldOperand.value());
                  });
}

MSFTModuleOp PartitionPass::partition(DesignPartitionOp partOp,
                                      Block *partBlock) {

  auto *ctxt = partOp.getContext();
  auto loc = partOp.getLoc();
  std::string nameBuffer;

  //*************
  //   Determine the partition module's interface. Keep some bookkeeping around.
  SmallVector<hw::PortInfo> inputPorts;
  SmallVector<hw::PortInfo> outputPorts;
  DenseMap<Value, size_t> newInputMap;
  SmallVector<Value, 32> instInputs;
  SmallVector<Value, 32> newOutputs;

  for (Operation &op : *partBlock) {
    StringRef opName = ::getOpName(&op);
    if (opName.empty())
      opName = op.getName().getIdentifier().getValue();

    for (OpOperand &oper : op.getOpOperands()) {
      Value v = oper.get();
      // Don't need a new input if we're consuming a value in the same block.
      if (v.getParentBlock() == partBlock)
        continue;
      auto existingF = newInputMap.find(v);
      if (existingF == newInputMap.end()) {
        // If there's not an existing input, create one.
        auto arg = partBlock->addArgument(v.getType(), loc);
        oper.set(arg);

        newInputMap[v] = inputPorts.size();
        StringRef portName = getValueName(v, topLevelSyms, nameBuffer);

        instInputs.push_back(v);
        inputPorts.push_back(hw::PortInfo{
            /*name*/ StringAttr::get(
                ctxt, opName + (portName.empty() ? "" : "." + portName)),
            /*direction*/ hw::PortDirection::INPUT,
            /*type*/ v.getType(),
            /*argNum*/ inputPorts.size()});
      } else {
        // There's already an existing port. Just set it.
        oper.set(partBlock->getArgument(existingF->second));
      }
    }

    for (OpResult res : op.getResults()) {
      // If all the consumers of this result are in the same partition, we don't
      // need a new output port.
      if (llvm::all_of(res.getUsers(), [partBlock](Operation *op) {
            return op->getBlock() == partBlock;
          }))
        continue;

      // If not, add one.
      newOutputs.push_back(res);
      StringRef portName = getResultName(res, topLevelSyms, nameBuffer);
      outputPorts.push_back(hw::PortInfo{
          /*name*/ StringAttr::get(
              ctxt, opName + (portName.empty() ? "" : "." + portName)),
          /*direction*/ hw::PortDirection::OUTPUT,
          /*type*/ res.getType(),
          /*argNum*/ outputPorts.size()});
    }
  }

  //*************
  //   Construct the partition module and replace the design partition op.

  // Build the module.
  hw::ModulePortInfo modPortInfo(inputPorts, outputPorts);
  auto partMod =
      OpBuilder(partOp->getParentOfType<MSFTModuleOp>())
          .create<MSFTModuleOp>(loc, partOp.verilogNameAttr(), modPortInfo,
                                ArrayRef<NamedAttribute>{});
  partBlock->moveBefore(partMod.getBodyBlock());
  partMod.getBlocks().back().erase();

  OpBuilder::atBlockEnd(partBlock).create<OutputOp>(partOp.getLoc(),
                                                    newOutputs);

  // Replace partOp with an instantion of the partition.
  SmallVector<Type> instRetTypes(
      llvm::map_range(newOutputs, [](Value v) { return v.getType(); }));
  auto partInst = OpBuilder(partOp).create<InstanceOp>(
      loc, instRetTypes, partOp.getNameAttr(), FlatSymbolRefAttr::get(partMod),
      instInputs);
  moduleInstantiations[partMod].push_back(partInst);

  // And set the outputs properly.
  for (size_t outputNum = 0, e = newOutputs.size(); outputNum < e; ++outputNum)
    for (OpOperand &oper :
         llvm::make_early_inc_range(newOutputs[outputNum].getUses()))
      if (oper.getOwner()->getBlock() != partBlock)
        oper.set(partInst.getResult(outputNum));

  // Push down any global refs to include the partition. Update the
  // partition to include the new set of global refs, and set its inner_sym.
  llvm::SetVector<Attribute> newGlobalRefs;
  for (Operation &op : *partBlock)
    pushDownGlobalRefs(&op, partOp, newGlobalRefs);
  SmallVector<Attribute> newGlobalRefVec(newGlobalRefs.begin(),
                                         newGlobalRefs.end());
  auto newRefsAttr = ArrayAttr::get(partInst->getContext(), newGlobalRefVec);
  partInst->setAttr(hw::GlobalRefAttr::DialectAttrName, newRefsAttr);
  partInst->setAttr("inner_sym", partInst.sym_nameAttr());

  return partMod;
}

namespace circt {
namespace msft {
std::unique_ptr<Pass> createPartitionPass() {
  return std::make_unique<PartitionPass>();
}
} // namespace msft
} // namespace circt

namespace {
struct WireCleanupPass : public WireCleanupBase<WireCleanupPass>,
                         MSFTPassCommon {
  void runOnOperation() override;
};
} // anonymous namespace

void WireCleanupPass::runOnOperation() {
  ModuleOp topMod = getOperation();
  topLevelSyms.addDefinitions(topMod);
  if (failed(verifyInstances(topMod))) {
    signalPassFailure();
    return;
  }

  SmallVector<MSFTModuleOp> sortedMods;
  getAndSortModules(topMod, sortedMods);

  for (auto mod : sortedMods) {
    bubbleWiresUp(mod);
    dedupOutputs(mod);
  }

  for (auto mod : llvm::reverse(sortedMods)) {
    sinkWiresDown(mod);
    dedupInputs(mod);
  }
}

/// Remove outputs driven by the same value.
void MSFTPassCommon::dedupOutputs(MSFTModuleOp mod) {
  Block *body = mod.getBodyBlock();
  Operation *terminator = body->getTerminator();

  DenseMap<Value, unsigned> valueToOutputIdx;
  SmallVector<unsigned> outputMap;
  llvm::BitVector outputPortsToRemove(terminator->getNumOperands());
  for (OpOperand &outputVal : terminator->getOpOperands()) {
    auto existing = valueToOutputIdx.find(outputVal.get());
    if (existing != valueToOutputIdx.end()) {
      outputMap.push_back(existing->second);
      outputPortsToRemove.set(outputVal.getOperandNumber());
    } else {
      outputMap.push_back(valueToOutputIdx.size());
      valueToOutputIdx[outputVal.get()] = valueToOutputIdx.size();
    }
  }

  mod.removePorts(llvm::BitVector(mod.getNumArguments()), outputPortsToRemove);
  updateInstances(mod, makeSequentialRange(mod.getNumResults()),
                  [&](InstanceOp newInst, InstanceOp oldInst,
                      SmallVectorImpl<Value> &newOperands) {
                    // Operands don't change.
                    llvm::append_range(newOperands, oldInst.getOperands());
                    // The results have to be remapped.
                    for (OpResult res : oldInst.getResults())
                      res.replaceAllUsesWith(
                          newInst.getResult(outputMap[res.getResultNumber()]));
                  });
}

/// Push up any wires which are simply passed-through.
void MSFTPassCommon::bubbleWiresUp(MSFTModuleOp mod) {
  Block *body = mod.getBodyBlock();
  Operation *terminator = body->getTerminator();
  hw::ModulePortInfo ports = mod.getPorts();

  // Find all "passthough" internal wires, filling 'inputPortsToRemove' as a
  // side-effect.
  DenseMap<Value, hw::PortInfo> passThroughs;
  llvm::BitVector inputPortsToRemove(ports.inputs.size());
  for (hw::PortInfo inputPort : ports.inputs) {
    BlockArgument portArg = body->getArgument(inputPort.argNum);
    bool removePort = true;
    for (OpOperand user : portArg.getUsers()) {
      if (user.getOwner() == terminator)
        passThroughs[portArg] = inputPort;
      else
        removePort = false;
    }
    if (removePort)
      inputPortsToRemove.set(inputPort.argNum);
  }

  // Find all output ports which we can remove. Fill in 'outputToInputIdx' to
  // help rewire instantiations later on.
  DenseMap<unsigned, unsigned> outputToInputIdx;
  llvm::BitVector outputPortsToRemove(ports.outputs.size());
  for (hw::PortInfo outputPort : ports.outputs) {
    assert(outputPort.argNum < terminator->getNumOperands() && "Invalid IR");
    Value outputValue = terminator->getOperand(outputPort.argNum);
    auto inputNumF = passThroughs.find(outputValue);
    if (inputNumF == passThroughs.end())
      continue;
    hw::PortInfo inputPort = inputNumF->second;
    outputToInputIdx[outputPort.argNum] = inputPort.argNum;
    outputPortsToRemove.set(outputPort.argNum);
  }

  // Use MSFTModuleOp's `removePorts` method to remove the ports. It returns a
  // mapping of the new output port to old output port indices to assist in
  // updating the instantiations later on.
  auto newToOldResult =
      mod.removePorts(inputPortsToRemove, outputPortsToRemove);

  // Update the instantiations.
  auto setPassthroughsGetOperands = [&](InstanceOp newInst, InstanceOp oldInst,
                                        SmallVectorImpl<Value> &newOperands) {
    // Re-map the passthrough values around the instance.
    for (auto idxPair : outputToInputIdx) {
      size_t outputPortNum = idxPair.first;
      assert(outputPortNum <= oldInst.getNumResults());
      size_t inputPortNum = idxPair.second;
      assert(inputPortNum <= oldInst.getNumOperands());
      oldInst.getResult(outputPortNum)
          .replaceAllUsesWith(oldInst.getOperand(inputPortNum));
    }
    // Use a sort-merge-join approach to figure out the operand mapping on the
    // fly.
    for (size_t operNum = 0, e = oldInst.getNumOperands(); operNum < e;
         ++operNum)
      if (!inputPortsToRemove.test(operNum))
        newOperands.push_back(oldInst.getOperand(operNum));
  };
  updateInstances(mod, newToOldResult, setPassthroughsGetOperands);
}

void MSFTPassCommon::dedupInputs(MSFTModuleOp mod) {
  const auto &instantiations = moduleInstantiations[mod];
  // TODO: remove this limitation. This would involve looking at the common
  // loopbacks for all the instances.
  if (instantiations.size() != 1)
    return;
  InstanceOp inst =
      dyn_cast<InstanceOp>(static_cast<Operation *>(instantiations[0]));
  if (!inst)
    return;

  // Find all the arguments which are driven by the same signal. Remap them
  // appropriately within the module, and mark that input port for deletion.
  Block *body = mod.getBodyBlock();
  DenseMap<Value, unsigned> valueToInput;
  llvm::BitVector argsToErase(body->getNumArguments());
  for (OpOperand &oper : inst->getOpOperands()) {
    auto existingValue = valueToInput.find(oper.get());
    if (existingValue != valueToInput.end()) {
      unsigned operNum = oper.getOperandNumber();
      unsigned duplicateInputNum = existingValue->second;
      body->getArgument(operNum).replaceAllUsesWith(
          body->getArgument(duplicateInputNum));
      argsToErase.set(operNum);
    } else {
      valueToInput[oper.get()] = oper.getOperandNumber();
    }
  }

  // Remove the ports.
  auto remappedResults =
      mod.removePorts(argsToErase, llvm::BitVector(inst.getNumResults()));
  // and update the instantiations.
  auto getOperands = [&](InstanceOp newInst, InstanceOp oldInst,
                         SmallVectorImpl<Value> &newOperands) {
    for (unsigned argNum = 0, e = oldInst.getNumOperands(); argNum < e;
         ++argNum)
      if (!argsToErase.test(argNum))
        newOperands.push_back(oldInst.getOperand(argNum));
  };
  inst = updateInstances(mod, remappedResults, getOperands)[0];

  SmallVector<Attribute, 32> newArgNames;
  std::string buff;
  for (Value oper : inst->getOperands()) {
    newArgNames.push_back(StringAttr::get(
        mod.getContext(), getValueName(oper, topLevelSyms, buff)));
  }
  mod.argNamesAttr(ArrayAttr::get(mod.getContext(), newArgNames));
}

/// Sink all the instance connections which are loops.
void MSFTPassCommon::sinkWiresDown(MSFTModuleOp mod) {
  const auto &instantiations = moduleInstantiations[mod];
  // TODO: remove this limitation. This would involve looking at the common
  // loopbacks for all the instances.
  if (instantiations.size() != 1)
    return;
  InstanceOp inst =
      dyn_cast<InstanceOp>(static_cast<Operation *>(instantiations[0]));
  if (!inst)
    return;

  // Find all the "loopback" connections in the instantiation. Populate
  // 'inputToOutputLoopback' with a mapping of input port to output port which
  // drives it. Populate 'resultsToErase' with output ports which only drive
  // input ports.
  DenseMap<unsigned, unsigned> inputToOutputLoopback;
  llvm::BitVector resultsToErase(inst.getNumResults());
  for (unsigned resNum = 0, e = inst.getNumResults(); resNum < e; ++resNum) {
    bool allLoops = true;
    for (auto &use : inst.getResult(resNum).getUses()) {
      if (use.getOwner() != inst.getOperation())
        allLoops = false;
      else
        inputToOutputLoopback[use.getOperandNumber()] = resNum;
    }
    if (allLoops)
      resultsToErase.set(resNum);
  }

  // Add internal connections to replace the instantiation's loop back
  // connections.
  Block *body = mod.getBodyBlock();
  Operation *terminator = body->getTerminator();
  llvm::BitVector argsToErase(body->getNumArguments());
  for (auto resOper : inputToOutputLoopback) {
    body->getArgument(resOper.first)
        .replaceAllUsesWith(terminator->getOperand(resOper.second));
    argsToErase.set(resOper.first);
  }

  // Remove the ports.
  SmallVector<unsigned> newToOldResultMap =
      mod.removePorts(argsToErase, resultsToErase);
  // and update the instantiations.
  auto getOperands = [&](InstanceOp newInst, InstanceOp oldInst,
                         SmallVectorImpl<Value> &newOperands) {
    // Use sort-merge-join to compute the new operands;
    for (unsigned argNum = 0, e = oldInst.getNumOperands(); argNum < e;
         ++argNum)
      if (!argsToErase.test(argNum))
        newOperands.push_back(oldInst.getOperand(argNum));
  };
  updateInstances(mod, newToOldResultMap, getOperands);
}

namespace circt {
namespace msft {
std::unique_ptr<Pass> createWireCleanupPass() {
  return std::make_unique<WireCleanupPass>();
}
} // namespace msft
} // namespace circt

//===----------------------------------------------------------------------===//
// Lower MSFT constructs
//===----------------------------------------------------------------------===//

namespace {

struct LowerConstructsPass : public LowerConstructsBase<LowerConstructsPass>,
                             PassCommon {
  void runOnOperation() override;

  /// For naming purposes, get the inner Namespace for a module, building it
  /// lazily.
  Namespace &getNamespaceFor(Operation *mod) {
    auto ns = moduleNamespaces.find(mod);
    if (ns != moduleNamespaces.end())
      return ns->getSecond();
    Namespace &nsNew = moduleNamespaces[mod];
    SymbolCache syms;
    syms.addDefinitions(mod);
    nsNew.add(syms);
    return nsNew;
  }

private:
  DenseMap<Operation *, circt::Namespace> moduleNamespaces;
};
} // anonymous namespace

namespace {
/// Lower MSFT's OutputOp to HW's.
struct SystolicArrayOpLowering : public OpConversionPattern<SystolicArrayOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SystolicArrayOp array, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    MLIRContext *ctxt = getContext();
    Location loc = array.getLoc();
    Block &peBlock = array.pe().front();
    rewriter.setInsertionPointAfter(array);

    // For the row broadcasts, break out the row values which must be broadcast
    // to each PE.
    hw::ArrayType rowInputs =
        hw::type_cast<hw::ArrayType>(array.rowInputs().getType());
    IntegerType rowIdxType = rewriter.getIntegerType(
        std::max(1u, llvm::Log2_64_Ceil(rowInputs.getSize())));
    SmallVector<Value> rowValues;
    for (size_t rowNum = 0, numRows = rowInputs.getSize(); rowNum < numRows;
         ++rowNum) {
      Value rowNumVal =
          rewriter.create<hw::ConstantOp>(loc, rowIdxType, rowNum);
      auto rowValue =
          rewriter.create<hw::ArrayGetOp>(loc, array.rowInputs(), rowNumVal);
      rowValue->setAttr("sv.namehint",
                        StringAttr::get(ctxt, "row_" + Twine(rowNum)));
      rowValues.push_back(rowValue);
    }

    // For the column broadcasts, break out the column values which must be
    // broadcast to each PE.
    hw::ArrayType colInputs =
        hw::type_cast<hw::ArrayType>(array.colInputs().getType());
    IntegerType colIdxType = rewriter.getIntegerType(
        std::max(1u, llvm::Log2_64_Ceil(colInputs.getSize())));
    SmallVector<Value> colValues;
    for (size_t colNum = 0, numCols = colInputs.getSize(); colNum < numCols;
         ++colNum) {
      Value colNumVal =
          rewriter.create<hw::ConstantOp>(loc, colIdxType, colNum);
      auto colValue =
          rewriter.create<hw::ArrayGetOp>(loc, array.colInputs(), colNumVal);
      colValue->setAttr("sv.namehint",
                        StringAttr::get(ctxt, "col_" + Twine(colNum)));
      colValues.push_back(colValue);
    }

    // Build the PE matrix.
    SmallVector<Value> peOutputs;
    for (size_t rowNum = 0, numRows = rowInputs.getSize(); rowNum < numRows;
         ++rowNum) {
      Value rowValue = rowValues[rowNum];
      SmallVector<Value> colPEOutputs;
      for (size_t colNum = 0, numCols = colInputs.getSize(); colNum < numCols;
           ++colNum) {
        Value colValue = colValues[colNum];
        // Clone the PE block, substituting %row (arg 0) and %col (arg 1) for
        // the corresponding row/column broadcast value.
        // NOTE: the PE region is NOT a graph region so we don't have to deal
        // with backedges.
        BlockAndValueMapping mapper;
        mapper.map(peBlock.getArgument(0), rowValue);
        mapper.map(peBlock.getArgument(1), colValue);
        for (Operation &peOperation : peBlock)
          // If we see the output op (which should be the block terminator), add
          // its operand to the output matrix.
          if (auto outputOp = dyn_cast<PEOutputOp>(peOperation)) {
            colPEOutputs.push_back(mapper.lookup(outputOp.output()));
          } else {
            Operation *clone = rewriter.clone(peOperation, mapper);

            StringRef nameSource = "name";
            auto name = clone->getAttrOfType<StringAttr>(nameSource);
            if (!name) {
              nameSource = "sv.namehint";
              name = clone->getAttrOfType<StringAttr>(nameSource);
            }
            if (name)
              clone->setAttr(nameSource,
                             StringAttr::get(ctxt, name.getValue() + "_" +
                                                       Twine(rowNum) + "_" +
                                                       Twine(colNum)));
          }
      }
      // Reverse the vector since ArrayCreateOp has the opposite ordering to C
      // vectors.
      std::reverse(colPEOutputs.begin(), colPEOutputs.end());
      peOutputs.push_back(
          rewriter.create<hw::ArrayCreateOp>(loc, colPEOutputs));
    }

    std::reverse(peOutputs.begin(), peOutputs.end());
    rewriter.replaceOp(array,
                       {rewriter.create<hw::ArrayCreateOp>(loc, peOutputs)});
    return success();
  }
};
} // anonymous namespace

namespace {
/// Lower MSFT's ChannelOp to a set of registers.
struct ChannelOpLowering : public OpConversionPattern<ChannelOp> {
public:
  ChannelOpLowering(MLIRContext *ctxt, LowerConstructsPass &pass)
      : OpConversionPattern(ctxt), pass(pass) {}

  LogicalResult
  matchAndRewrite(ChannelOp chan, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = chan.getLoc();
    Operation *mod = chan->getParentOfType<MSFTModuleOp>();
    assert(mod && "ChannelOp must be contained by module");
    Namespace &ns = pass.getNamespaceFor(mod);
    Value clk = chan.clk();
    Value v = chan.input();
    for (uint64_t stageNum = 0, e = chan.defaultStages(); stageNum < e;
         ++stageNum)
      v = rewriter.create<seq::CompRegOp>(loc, v, clk,
                                          ns.newName(chan.sym_name()));
    rewriter.replaceOp(chan, {v});
    return success();
  }

protected:
  LowerConstructsPass &pass;
};
} // namespace

void LowerConstructsPass::runOnOperation() {
  auto top = getOperation();
  auto *ctxt = &getContext();

  ConversionTarget target(*ctxt);
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

  RewritePatternSet patterns(ctxt);
  patterns.insert<SystolicArrayOpLowering>(ctxt);
  target.addIllegalOp<SystolicArrayOp>();
  patterns.insert<ChannelOpLowering>(ctxt, *this);
  target.addIllegalOp<ChannelOp>();

  if (failed(mlir::applyPartialConversion(top, target, std::move(patterns))))
    signalPassFailure();
}

namespace circt {
namespace msft {
std::unique_ptr<Pass> createLowerConstructsPass() {
  return std::make_unique<LowerConstructsPass>();
}
} // namespace msft
} // namespace circt

//===----------------------------------------------------------------------===//
// Discover AppIDs pass
//===----------------------------------------------------------------------===//

namespace {
struct DiscoverAppIDsPass : public DiscoverAppIDsBase<DiscoverAppIDsPass>,
                            MSFTPassCommon {
  void runOnOperation() override;
  void processMod(MSFTModuleOp);
};
} // anonymous namespace

void DiscoverAppIDsPass::runOnOperation() {
  ModuleOp topMod = getOperation();
  topLevelSyms.addDefinitions(topMod);
  if (failed(verifyInstances(topMod))) {
    signalPassFailure();
    return;
  }

  // Sort modules in partial order be use. Enables single-pass processing.
  SmallVector<MSFTModuleOp> sortedMods;
  getAndSortModules(topMod, sortedMods);

  for (MSFTModuleOp mod : sortedMods)
    processMod(mod);
}

/// Find the AppIDs in a given module.
void DiscoverAppIDsPass::processMod(MSFTModuleOp mod) {
  SmallDenseMap<StringAttr, uint64_t> appBaseCounts;
  SmallPtrSet<StringAttr, 32> localAppIDBases;
  SmallDenseMap<AppIDAttr, Operation *> localAppIDs;

  mod.walk([&](Operation *op) {
    // If an operation has an "appid" dialect attribute, it is considered a
    // "local" appid.
    if (auto appid = op->getAttrOfType<AppIDAttr>("msft.appid")) {
      if (localAppIDs.find(appid) != localAppIDs.end()) {
        op->emitOpError("Found multiple identical AppIDs in same module")
                .attachNote(localAppIDs[appid]->getLoc())
            << "first AppID located here";
        signalPassFailure();
      } else {
        localAppIDs[appid] = op;
      }
      localAppIDBases.insert(appid.getName());
    }

    // Instance ops should expose their module's AppIDs recursively. Track the
    // number of instances which contain a base name.
    if (auto inst = dyn_cast<InstanceOp>(op)) {
      auto targetMod = dyn_cast<MSFTModuleOp>(
          topLevelSyms.getDefinition(inst.moduleNameAttr()));
      if (targetMod && targetMod.childAppIDBases())
        for (auto base :
             targetMod.childAppIDBasesAttr().getAsRange<StringAttr>())
          appBaseCounts[base] += 1;
    }
  });

  // Collect the list of AppID base names with which to annotate 'mod'.
  SmallVector<Attribute, 32> finalModBases;
  for (auto baseCount : appBaseCounts) {
    // If multiple instances expose the same base name, don't expose them
    // through this module. If any of the instances expose basenames which are
    // exposed locally, also don't expose them up.
    if (baseCount.getSecond() == 1 &&
        !localAppIDBases.contains(baseCount.getFirst()))
      finalModBases.push_back(baseCount.getFirst());
  }

  // Add all of the local base names.
  for (StringAttr lclBase : localAppIDBases)
    finalModBases.push_back(lclBase);

  if (finalModBases.empty())
    return;
  ArrayAttr childrenBases = ArrayAttr::get(mod.getContext(), finalModBases);
  mod.childAppIDBasesAttr(childrenBases);
}

namespace circt {
namespace msft {
std::unique_ptr<Pass> createDiscoverAppIDsPass() {
  return std::make_unique<DiscoverAppIDsPass>();
}
} // namespace msft
} // namespace circt
namespace {
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/MSFT/MSFTPasses.h.inc"
} // namespace

void circt::msft::registerMSFTPasses() { registerPasses(); }
