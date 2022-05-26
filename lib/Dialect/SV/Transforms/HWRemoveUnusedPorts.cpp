#include "PassDetail.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hw-remove-unused-ports"

using namespace llvm;
using namespace mlir;
using namespace circt;
using namespace hw;

static void eraseDeadOperations(SmallDenseSet<Operation *> &ops) {
  for (auto op : ops) {
    if (op && isOpTriviallyDead(op))
      op->erase();
  }
}

/// Return true if the port is deletable.
static bool isDeletablePort(PortInfo port) {
  return !port.sym || port.sym.getValue().empty();
}

/// Return true if the port is deletable.
static bool isOutputPortUnusedInAllInstances(InstanceGraphNode *node,
                                             unsigned index) {
  return llvm::all_of(node->uses(), [&](InstanceRecord *record) {
    auto port = record->getInstance()->getResult(index);
    return port.use_empty();
  });
}

static bool isOutputPortUnused(InstanceRecord *record, unsigned index) {
  auto port = record->getInstance()->getResult(index);
  return port.use_empty();
}

/// Return true if the output port is deletable.
static bool isDeletableOutputPort(HWModuleOp module, InstanceGraphNode *node,
                                  unsigned index) {
  if (!isDeletablePort(module.getOutputPort(index)))
    return false;

  return llvm::all_of(node->uses(), [&](InstanceRecord *record) {
    auto port = record->getInstance()->getResult(index);
    return port.use_empty();
  });
}

/// Return true if the input port is deletable.
static bool isDeletableInputPort(HWModuleOp module, unsigned index) {
  if (!module.getArgument(index).use_empty())
    return false;

  if (!isDeletablePort(module.getInOrInoutPort(index)))
    return false;

  return true;
}

namespace {
struct HWRemoveUnusedPortsPass
    : public sv::HWRemoveUnusedPortsBase<HWRemoveUnusedPortsPass> {
  void visitModule(HWModuleOp module, InstanceGraphNode *instanceGraphNode);
  void finalize(HWModuleOp module, InstanceGraphNode *instanceGraphNode);
  void runOnOperation() override;
  void visitPrivateModules(bool doFinalize);
  void visitOutputPort(StringAttr moduleName, unsigned index);
  void visitInputPort(StringAttr moduleName, unsigned index);
  void visitInputPort(HWModuleOp module, InstanceGraphNode *node,
                      unsigned index);
  void visitValue(Value value);

  /// A worklist of values that might be dead. We have to use a set to avoid
  /// double free. SetVector is used to make the pass deterministic.
  llvm::SetVector<Value> worklist;

  InstanceGraph *instanceGraph;
  OpBuilder *builder;

  void addToWorklist(Value value) { worklist.insert(value); }

  // Return a place holder for the given value. Values created here must be
  // deleted at post-processing.
  Value getDummyValue(Value value) {
    builder->setInsertionPointAfterValue(value);
    return builder
        ->create<mlir::UnrealizedConversionCastOp>(
            value.getLoc(), TypeRange{value.getType()}, ValueRange{})
        .getResult(0);
  }

  std::pair<HWModuleOp, InstanceGraphNode *>
  getModuleIfPrivate(StringAttr moduleName) {
    auto node = instanceGraph->lookup(moduleName);
    auto module = dyn_cast<HWModuleOp>(node->getModule());
    if (!module || !module.isPrivate())
      return {};
    return {module, node};
  }
};
} // namespace

void HWRemoveUnusedPortsPass::visitOutputPort(StringAttr moduleName,
                                              unsigned index) {
  auto [module, node] = getModuleIfPrivate(moduleName);
  if (!module)
    return;

  if (!isDeletableOutputPort(module, node, index))
    return;
  auto output = cast<OutputOp>(module.getBodyBlock()->getTerminator());
  auto operand = output->getOperand(index);
  output->setOperand(index, getDummyValue(operand));
  addToWorklist(operand);
}

void HWRemoveUnusedPortsPass::visitInputPort(HWModuleOp module,
                                             InstanceGraphNode *node,
                                             unsigned index) {
  if (!isDeletableInputPort(module, index))
    return;

  for (auto *use : node->uses()) {
    if (!use->getInstance())
      continue;
    auto instance = dyn_cast<InstanceOp>(*use->getInstance());

    if (!instance)
      return;
    auto operand = instance.getOperand(index);
    instance->setOperand(index, getDummyValue(operand));
    addToWorklist(operand);
  }
}

void HWRemoveUnusedPortsPass::visitInputPort(StringAttr moduleName,
                                             unsigned index) {
  auto [module, node] = getModuleIfPrivate(moduleName);
  if (!module)
    return;
  visitInputPort(module, node, index);
}

void HWRemoveUnusedPortsPass::visitValue(Value value) {
  // If the value has an use, we cannot remove.
  if (!value.use_empty())
    return;
  LLVM_DEBUG({ llvm::dbgs() << "Now looking at " << value << "\n"; });

  if (auto instance = value.getDefiningOp<HWInstanceLike>())
    visitOutputPort(instance.referencedModuleNameAttr(),
                    value.cast<mlir::OpResult>().getResultNumber());

  if (auto inputPort = value.dyn_cast<BlockArgument>()) {
    auto hwmodule =
        dyn_cast<HWModuleLike>(inputPort.getParentBlock()->getParentOp());
    if (!hwmodule)
      return;
    visitInputPort(hwmodule.moduleNameAttr(), inputPort.getArgNumber());
  }

  if (auto op = value.getDefiningOp()) {
    // Check that op can be erased. `isOpTriviallyDead` checks uses or
    // symbols appropriately. If the op is dead, add its operands to the
    // worklist.
    if (isOpTriviallyDead(op)) {
      for (auto operand : op->getOperands())
        addToWorklist(operand);
      op->erase();
    }
  }
}

void HWRemoveUnusedPortsPass::visitPrivateModules(bool doFinalize) {
  for (auto *node : llvm::post_order(instanceGraph)) {
    if (!node->getModule())
      continue;
    auto module = dyn_cast<HWModuleOp>(node->getModule());
    if (!module || module.isPublic())
      continue;
    if (doFinalize)
      finalize(module, node);
    else
      visitModule(module, node);
  }
}

void HWRemoveUnusedPortsPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "===----- Remove unused ports -----===\n");
  instanceGraph = &getAnalysis<InstanceGraph>();
  OpBuilder theBuilder(&getContext());
  builder = &theBuilder;

  visitPrivateModules(/*doFinalize=*/false);

  while (!worklist.empty()) {
    auto value = worklist.pop_back_val();
    visitValue(value);
  }

  visitPrivateModules(/*doFinalize=*/true);
}

/// Remove elements at the specified indices from the input array, returning
/// the elements not mentioned.  The indices array is expected to be sorted
/// and unique.
template <typename T, typename F>
static SmallVector<T>
removeElementsAtIndices(F input, ArrayRef<unsigned> indicesToDrop) {
  // Copy over the live chunks.
  size_t lastCopied = 0;
  SmallVector<T> result;
  result.reserve(input.size() - indicesToDrop.size());

  for (unsigned indexToDrop : indicesToDrop) {
    // If we skipped over some valid elements, copy them over.
    if (indexToDrop > lastCopied) {
      result.append(input.begin() + lastCopied, input.begin() + indexToDrop);
      lastCopied = indexToDrop;
    }
    // Ignore this value so we don't copy it in the next iteration.
    ++lastCopied;
  }

  // If there are live elements at the end, copy them over.
  if (lastCopied < input.size())
    result.append(input.begin() + lastCopied, input.end());

  return result;
}

void HWRemoveUnusedPortsPass::finalize(HWModuleOp module,
                                       InstanceGraphNode *instanceGraphNode) {
  LLVM_DEBUG(llvm::dbgs() << "Prune ports of module: " << module.getName()
                          << "\n");
  SmallVector<unsigned> removalInputPortIndexes;
  SmallVector<unsigned> removalOutputPortIndexes;

  SmallDenseSet<Operation *, 4> mightDeadOperations;

  auto ports = module.getPorts();

  for (auto e : llvm::enumerate(ports.outputs)) {
    unsigned index = e.index();
    auto port = e.value();
    if (isDeletableOutputPort(module, instanceGraphNode, index))
      removalOutputPortIndexes.push_back(index);
  }

  // Traverse output ports.
  auto output = cast<OutputOp>(module.getBodyBlock()->getTerminator());
  ImplicitLocOpBuilder builder(output.getLoc(), output);
  if (!removalOutputPortIndexes.empty()) {
    auto newOutput = removeElementsAtIndices<Value>(output.operands(),
                                                    removalOutputPortIndexes);
    builder.create<hw::OutputOp>(newOutput);
    for (auto operand : output.operands())
      mightDeadOperations.insert(operand.getDefiningOp());
    output.erase();
  }

  // Traverse input ports.
  for (auto index : llvm::seq(0u, module.getNumArguments()))
    if (isDeletableInputPort(module, index))
      removalInputPortIndexes.push_back(index);

  // If there is nothing to remove, abort.
  if (removalInputPortIndexes.empty() && removalOutputPortIndexes.empty())
    return;

  // Delete ports from the module.
  module.erasePorts(removalInputPortIndexes, removalOutputPortIndexes);

  // Delete arguments. It is necessary to remove the argument in the reverse
  // order of `removalInputPortIndexes`.
  for (auto arg : llvm::reverse(removalInputPortIndexes))
    module.getBody().eraseArgument(arg);

  // Rewrite all uses.
  for (auto *use : instanceGraphNode->uses()) {
    auto instance = dyn_cast<InstanceOp>(*use->getInstance());
    if (!instance)
      continue;
    for (auto c : removalInputPortIndexes)
      mightDeadOperations.insert(instance.getOperand(c).getDefiningOp());

    builder.setInsertionPoint(instance);
    // Create a new instance op without unused ports.
    auto newInstance = instance.erasePorts(
        builder, module, removalInputPortIndexes, removalOutputPortIndexes);

    instanceGraph->replaceInstance(instance, newInstance);
    // Remove old one.
    instance.erase();
  }

  eraseDeadOperations(mightDeadOperations);
  numRemovedPorts += removalInputPortIndexes.size();
  numRemovedPorts += removalOutputPortIndexes.size();
}

void HWRemoveUnusedPortsPass::visitModule(
    HWModuleOp module, InstanceGraphNode *instanceGraphNode) {
  LLVM_DEBUG(llvm::dbgs() << "Preprocess module: " << module.getName() << "\n");
  // These track port indexes that can be erased.
  SmallVector<unsigned> removalInputPortIndexes;
  SmallVector<unsigned> removalOutputPortIndexes;

  // This tracks constant values of output ports. None indicates either
  // ConstantX or dummy values.
  SmallVector<llvm::Optional<APInt>> outputPortConstants;
  auto ports = module.getPorts();

  // Traverse output ports.
  auto output = cast<OutputOp>(module.getBodyBlock()->getTerminator());
  for (auto e : llvm::enumerate(ports.outputs)) {
    unsigned index = e.index();
    auto port = e.value();
    if (!isDeletablePort(port))
      continue;

    if (llvm::all_of(instanceGraphNode->uses(), [&](auto record) {
          return isOutputPortUnused(record, index);
        })) {
      // Replace the port with a wire if it is unused.
      outputPortConstants.push_back(None);
      removalOutputPortIndexes.push_back(index);
      addToWorklist(output.getOperand(index));
      continue;
    }

    auto src = output.getOperand(index).getDefiningOp();
    if (!isa_and_nonnull<hw::ConstantOp, sv::ConstantXOp>(src))
      continue;

    if (auto constant = dyn_cast<hw::ConstantOp>(src))
      outputPortConstants.push_back(constant.value());
    else
      outputPortConstants.push_back(None);

    removalOutputPortIndexes.push_back(index);
    addToWorklist(output->getOperand(index));
  }

  ImplicitLocOpBuilder builder(output.getLoc(), output);

  for (auto index : llvm::seq(0u, module.getNumArguments()))
    if (isDeletableInputPort(module, index))
      removalInputPortIndexes.push_back(index);

  // If there is nothing to remove, abort.
  if (removalInputPortIndexes.empty() && removalOutputPortIndexes.empty())
    return;

  // Rewrite all uses.
  for (auto *use : instanceGraphNode->uses()) {
    auto instance = dyn_cast<InstanceOp>(*use->getInstance());
    if (!instance)
      continue;
    ImplicitLocOpBuilder builder(instance.getLoc(), instance);
    for (auto [index, constant] :
         llvm::zip(removalOutputPortIndexes, outputPortConstants)) {
      auto result = instance.getResult(index);
      if (result.use_empty())
        continue;

      Value value;
      if (constant) {
        value = builder.create<hw::ConstantOp>(*constant);
      } else {
        value = builder.create<sv::ConstantXOp>(result.getType());
      }

      result.replaceAllUsesWith(value);
    }

    for (auto inputPort : removalInputPortIndexes) {
      auto operand = instance.getOperand(inputPort);
      instance.setOperand(inputPort, getDummyValue(operand));
      addToWorklist(operand);
    }
  }
}

std::unique_ptr<mlir::Pass> circt::sv::createHWRemoveUnusedPortsPass() {
  return std::make_unique<HWRemoveUnusedPortsPass>();
}