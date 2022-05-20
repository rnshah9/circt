#include "PassDetails.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/SV/SVOps.h"
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

namespace circt {
namespace hw {
struct HWRemoveUnusedPortsPass
    : public hw::HWRemoveUnusedPortsBase<HWRemoveUnusedPortsPass> {
  void removeUnusedModulePorts(HWModuleOp module,
                               InstanceGraphNode *instanceGraphNode);

  void runOnOperation() override {
    auto &instanceGraph = getAnalysis<InstanceGraph>();
    LLVM_DEBUG(llvm::dbgs() << "===----- Remove unused ports -----==="
                            << "\n");
    // Iterate in the reverse order of instance graph iterator, i.e. from leaves
    // to top.
    for (auto *node : llvm::post_order(&instanceGraph))
      if (auto module = dyn_cast<HWModuleOp>(*node->getModule()))
        // Don't prune the main module.
        if (!module.isPublic())
          removeUnusedModulePorts(module, node);
  }
};
void HWRemoveUnusedPortsPass::removeUnusedModulePorts(
    HWModuleOp module, InstanceGraphNode *instanceGraphNode) {
  LLVM_DEBUG(llvm::dbgs() << "Prune ports of module: " << module.getName()
                          << "\n");
  // These track port indexes that can be erased.
  SmallVector<unsigned> removalInputPortIndexes;
  SmallVector<unsigned> removalOutputPortIndexes;

  // This tracks constant values of output ports. None indicates an
  // uninitialized value.
  SmallVector<llvm::Optional<APInt>> outputPortConstants;
  auto ports = module.getPorts();

  // Traverse input ports.
  for (auto e : llvm::enumerate(ports.inputs)) {
    unsigned index = e.index();
    auto port = e.value();
    auto arg = module.getArgument(port.argNum);

    if (port.isInOut() || (port.sym && !port.sym.getValue().empty()))
      continue;
    // If the port is not dead, skip.
    if (!arg.use_empty())
      continue;
    removalInputPortIndexes.push_back(index);
  }

  // Traverse output ports.
  auto output = cast<OutputOp>(module.getBodyBlock()->getTerminator());
  for (auto e : llvm::enumerate(ports.outputs)) {
    unsigned index = e.index();
    auto port = e.value();
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

  // If there is nothing to remove, abort.
  if (removalInputPortIndexes.empty() && removalOutputPortIndexes.empty())
    return;

  // Delete ports from the module.
  module.erasePorts(removalInputPortIndexes, removalOutputPortIndexes);

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
std::unique_ptr<mlir::Pass> createHWRemoveUnusedPortsPass() {
  return std::make_unique<HWRemoveUnusedPortsPass>();
}
} // namespace hw
} // namespace circt