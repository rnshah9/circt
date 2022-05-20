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
namespace {
struct HWRemoveUnusedPortsPass
    : public sv::HWRemoveUnusedPortsBase<HWRemoveUnusedPortsPass> {
  void removeUnusedModulePorts(HWModuleOp module,
                               InstanceGraphNode *instanceGraphNode);

  void runOnOperation() override {
    auto &instanceGraph = getAnalysis<InstanceGraph>();
    LLVM_DEBUG(llvm::dbgs() << "===----- Remove unused ports -----==="
                            << "\n");
    // Iterate in the reverse order of instance graph iterator, i.e. from leaves
    // to top.
    for (auto *node : llvm::post_order(&instanceGraph)) {
      assert(node && "bar");
      if (!node->getModule())
        continue;
      if (auto module = dyn_cast_or_null<HWModuleOp>(node->getModule()))
        // Don't prune the main module.
        if (!module.isPublic())
          removeUnusedModulePorts(module, node);
    }
  }
};
} // namespace

/// Remove elements at the specified indices from the input array, returning the
/// elements not mentioned.  The indices array is expected to be sorted and
/// unique.
template <typename T>
static SmallVector<T>
removeElementsAtIndices(ArrayRef<T> input, ArrayRef<unsigned> indicesToDrop) {
#ifndef NDEBUG // Check sortedness.
  if (!input.empty()) {
    for (size_t i = 1, e = indicesToDrop.size(); i != e; ++i)
      assert(indicesToDrop[i - 1] < indicesToDrop[i] &&
             "indicesToDrop isn't sorted and unique");
    assert(indicesToDrop.back() < input.size() && "index out of range");
  }
#endif

  // If the input is empty (which is an optimization we do for certain array
  // attributes), simply return an empty vector.
  if (input.empty())
    return {};

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
void HWRemoveUnusedPortsPass::removeUnusedModulePorts(
    HWModuleOp module, InstanceGraphNode *instanceGraphNode) {
  LLVM_DEBUG(llvm::dbgs() << "Prune ports of module: " << module.getName()
                          << "\n");
  // These track port indexes that can be erased.
  SmallVector<unsigned> removalInputPortIndexes;
  SmallVector<unsigned> removalOutputPortIndexes;
  // llvm::dbgs() << "bar!";

  // This tracks constant values of output ports. None indicates an
  // uninitialized value.
  SmallVector<llvm::Optional<APInt>> outputPortConstants;
  auto ports = module.getPorts();

  // Traverse output ports.
  auto output = cast<OutputOp>(module.getBodyBlock()->getTerminator());
  for (auto e : llvm::enumerate(ports.outputs)) {
    unsigned index = e.index();
    auto port = e.value();
    // llvm::dbgs() << "output: " << index << " " << port.argNum << "\n";
    if (port.sym && !port.sym.getValue().empty())
      continue;

    auto portIsUnused = [&](InstanceRecord *a) -> bool {
      auto port = a->getInstance()->getResult(index);
      return port.getUses().empty();
    };

    if (llvm::all_of(instanceGraphNode->uses(), portIsUnused)) {
      // Replace the port with a wire if it is unused.
      outputPortConstants.push_back(None);
      removalOutputPortIndexes.push_back(index);
      continue;
    }

    auto src = output.getOperand(index).getDefiningOp();
    if (!isa_and_nonnull<hw::ConstantOp, sv::ConstantXOp>(src))
      continue;

    if (auto constant = dyn_cast<hw::ConstantOp>(src))
      outputPortConstants.push_back(constant.value());
    else {
      outputPortConstants.push_back(None);
    }

    removalOutputPortIndexes.push_back(index);
  }

  ImplicitLocOpBuilder builder(output.getLoc(), output);
  if (!removalOutputPortIndexes.empty()) {
    SmallVector<Value> oldOperand(output.operands());
    auto newOutput =
        removeElementsAtIndices<Value>(oldOperand, removalOutputPortIndexes);
    builder.create<hw::OutputOp>(newOutput);
    output.erase();
    // while (!oldOperand.empty()) {
    //   auto op = oldOperand.pop_back_val().getDefiningOp();
    //   if (op && op->use_empty()) {
    //     auto operand = op->getOperands();
    //     while (op->getNumOperands()) {
    //       auto operand = op->getOperand(0);
    //       op->eraseOperand(0);
    //       if (operand.use_empty()) {
    //         oldOperand.push_back(operand);
    //       }
    //     }
    //     op->erase();
    //   }
    // }
  }

  // Traverse input ports.
  for (auto e : llvm::enumerate(ports.inputs)) {
    unsigned index = e.index();
    auto port = e.value();
    auto arg = module.getArgument(port.argNum);
    // llvm::dbgs() << "input: " << index << " " << port.argNum << "\n";

    if (port.isInOut() || (port.sym && !port.sym.getValue().empty()))
      continue;
    // If the port is not dead, skip.
    if (!arg.use_empty())
      continue;
    removalInputPortIndexes.push_back(index);
  }

  // If there is nothing to remove, abort.
  if (removalInputPortIndexes.empty() && removalOutputPortIndexes.empty())
    return;
  LLVM_DEBUG({
    for (auto c : removalInputPortIndexes) {
      llvm::dbgs() << "remove input: " << c << "\n";
    }
    for (auto c : removalOutputPortIndexes) {
      llvm::dbgs() << "remove output: " << c << "\n";
    }
  });

  // Delete ports from the module.
  module.erasePorts(removalInputPortIndexes, removalOutputPortIndexes);

  // module.dump();
  for (auto c : llvm::reverse(removalInputPortIndexes)) {
    module.getBody().eraseArgument(c);
  }

  // module.dump();

  // auto newOutput = removeElementsAtIndices<Value>(
  //     SmallVector<Value>(output.operands()), removalOutputPortIndexes);
  // builder.create<hw::OutputOp>(newOutput);
  // output.erase();

  // Rewrite all uses.
  for (auto *use : instanceGraphNode->uses()) {
    auto instance = ::cast<InstanceOp>(*use->getInstance());
    ImplicitLocOpBuilder builder(instance.getLoc(), instance);
    for (auto [index, constant] :
         llvm::zip(removalOutputPortIndexes, outputPortConstants)) {
      auto result = instance.getResult(index);
      if (result.use_empty())
        continue;
      Value value;
      if (constant)
        value = builder.create<hw::ConstantOp>(*constant);
      else
        value = builder.create<sv::ConstantXOp>(result.getType());
      result.replaceAllUsesWith(value);
    }
    // Create a new instance op without unused ports.
    instance.erasePorts(builder, module, removalInputPortIndexes,
                        removalOutputPortIndexes);
    // Remove old one.
    instance.erase();
  }
  numRemovedPorts += removalInputPortIndexes.size();
  numRemovedPorts += removalOutputPortIndexes.size();
}

std::unique_ptr<mlir::Pass> circt::sv::createHWRemoveUnusedPortsPass() {
  return std::make_unique<HWRemoveUnusedPortsPass>();
}