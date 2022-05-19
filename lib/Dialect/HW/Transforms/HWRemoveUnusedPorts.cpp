#include "PassDetails.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
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
  SmallVector<llvm::Optional<APSInt>> outputPortConstants;
  auto ports = module.getPorts();
  for (auto e : llvm::enumerate(ports.inputs)) {
    unsigned index = e.index();
    auto port = e.value();
    // If the port is don't touch or has unprocessed annotations, we cannot
    // remove the port. Maybe we can allow annotations though.
    if (port.sym && !port.sym.getValue().empty())
      continue;
  }

  for (auto e : llvm::enumerate(ports)) {
    unsigned index = e.index();
    auto port = e.value();
    auto arg = module.getArgument(index);

    // If the port is don't touch or has unprocessed annotations, we cannot
    // remove the port. Maybe we can allow annotations though.
    if (port.sym && !port.sym.getValue().empty())
      continue;

    // TODO: Handle inout ports.
    if (port.isInOut())
      continue;

    // If the port is input and has an user, we cannot remove the
    // port.
    if (port.isInput() && !arg.use_empty())
      continue;

    auto portIsUnused = [&](InstanceRecord *a) -> bool {
      auto port = a->getInstance()->getResult(arg.getArgNumber());
      return port.getUses().empty();
    };

    // Output port.
    if (port.isOutput()) {
      if (arg.use_empty()) {
        // Sometimes the connection is already removed possibly by IMCP.
        // In that case, regard the port value as an invalid value.
        outputPortConstants.push_back(None);
      } else if (llvm::all_of(instanceGraphNode->uses(), portIsUnused)) {
        // Replace the port with a wire if it is unused.
        auto builder =
            ImplicitLocOpBuilder::atBlockBegin(arg.getLoc(), module.getBody());
        auto wire = builder.create<sv::WireOp>(arg.getType());
        arg.replaceAllUsesWith(wire);
        outputPortConstants.push_back(None);
      } else if (arg.hasOneUse()) {
        // If the port has a single use, check the port is only connected to
        // invalid or constant
        Operation *op = arg.use_begin().getUser();
        auto connectLike = dyn_cast<FConnectLike>(op);
        if (!connectLike)
          continue;
        auto *srcOp = connectLike.src().getDefiningOp();
        if (!isa_and_nonnull<InvalidValueOp, ConstantOp>(srcOp))
          continue;

        if (auto constant = dyn_cast<ConstantOp>(srcOp))
          outputPortConstants.push_back(constant.value());
        else {
          assert(isa<InvalidValueOp>(srcOp) && "only expect invalid");
          outputPortConstants.push_back(None);
        }

        // Erase connect op because we are going to remove this output ports.
        op->erase();

        if (srcOp->use_empty())
          srcOp->erase();
      } else {
        // Otherwise, we cannot remove the port.
        continue;
      }
    }

    removalPortIndexes.push_back(index);
  }

  // If there is nothing to remove, abort.
  if (removalPortIndexes.empty())
    return;

  // Delete ports from the module.
  module.erasePorts(removalPortIndexes);
  LLVM_DEBUG(llvm::for_each(removalPortIndexes, [&](unsigned index) {
               llvm::dbgs() << "Delete port: " << ports[index].name << "\n";
             }););

  // Rewrite all uses.
  for (auto *use : instanceGraphNode->uses()) {
    auto instance = ::cast<InstanceOp>(*use->getInstance());
    ImplicitLocOpBuilder builder(instance.getLoc(), instance);
    unsigned outputPortIndex = 0;
    for (auto index : removalPortIndexes) {
      auto result = instance.getResult(index);
      assert(!ports[index].isInOut() && "don't expect inout ports");

      // If the port is input, replace the port with an unwritten wire
      // so that we can remove use-chains in SV dialect canonicalization.
      if (ports[index].isInput()) {
        WireOp wire = builder.create<WireOp>(result.getType());

        // Check that the input port is only written. Sometimes input ports are
        // used as temporary wires. In that case, we cannot erase connections.
        bool onlyWritten = llvm::all_of(result.getUsers(), [&](Operation *op) {
          if (auto connect = dyn_cast<FConnectLike>(op))
            return connect.dest() == result;
          return false;
        });

        result.replaceUsesWithIf(wire, [&](OpOperand &op) -> bool {
          // Connects can be deleted directly.
          if (onlyWritten && isa<FConnectLike>(op.getOwner())) {
            op.getOwner()->erase();
            return false;
          }
          return true;
        });

        // If the wire doesn't have an user, just erase it.
        if (wire.use_empty())
          wire.erase();

        continue;
      }

      // Output port. Replace with the output port with an invalid or constant
      // value.
      auto portConstant = outputPortConstants[outputPortIndex++];
      Value value;
      if (portConstant)
        value = builder.create<ConstantOp>(*portConstant);
      else
        value = builder.create<InvalidValueOp>(result.getType());

      result.replaceAllUsesWith(value);
    }

    // Create a new instance op without unused ports.
    instance.erasePorts(builder, removalPortIndexes);
    // Remove old one.
    instance.erase();
  }

  numRemovedPorts += removalPortIndexes.size();
}
std::unique_ptr<mlir::Pass> createHWRemoveUnusedPortsPass() {
  return std::make_unique<HWRemoveUnusedPortsPass>();
}
} // namespace hw
} // namespace circt