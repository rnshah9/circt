//===- SFCCompat.cpp - SFC Compatible Pass ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This pass makes a number of updates to the circuit that are required to match
// the behavior of the Scala FIRRTL Compiler (SFC).  This pass removes invalid
// values from the circuit.  This is a combination of the Scala FIRRTL
// Compiler's RemoveRests pass and RemoveValidIf.  This is done to remove two
// "interpretations" of invalid.  Namely: (1) registers that are initialized to
// an invalid value (module scoped and looking through wires and connects only)
// are converted to an unitialized register and (2) invalid values are converted
// to zero (after rule 1 is applied).  Additionally, this pass checks and
// disallows async reset registers that are not driven with a constant when
// looking through wires, connects, and nodes.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-remove-resets"

using namespace circt;
using namespace firrtl;

struct SFCCompatPass : public SFCCompatBase<SFCCompatPass> {
  void runOnOperation() override;
};

void SFCCompatPass::runOnOperation() {
  LLVM_DEBUG(
      llvm::dbgs() << "==----- Running SFCCompat "
                      "---------------------------------------------------===\n"
                   << "Module: '" << getOperation().getName() << "'\n";);

  bool madeModifications = false;
  SmallVector<InvalidValueOp> invalidOps;
  for (auto &op : llvm::make_early_inc_range(getOperation().getOps())) {
    // Populate invalidOps for later handling.
    if (auto inv = dyn_cast<InvalidValueOp>(op)) {
      invalidOps.push_back(inv);
      continue;
    }
    auto reg = dyn_cast<RegResetOp>(op);
    if (!reg)
      continue;

    // Skip if the reg has an aggregate type.
    // TODO: Support aggregate types.
    if (reg.getType().isa<FVectorType, BundleType>()) {
      reg.emitWarning() << "Aggregate values are not supported by SFCCompat "
                           "pass. This may result in incorrect results";
      continue;
    }

    // If the `RegResetOp` has an invalidated initialization, then replace it
    // with a `RegOp`.
    if (isModuleScopedDrivenBy<InvalidValueOp>(reg.getResetValue(), true, false,
                                               false)) {
      LLVM_DEBUG(llvm::dbgs() << "  - RegResetOp '" << reg.getName()
                              << "' will be replaced with a RegOp\n");
      ImplicitLocOpBuilder builder(reg.getLoc(), reg);
      RegOp newReg = builder.create<RegOp>(
          reg.getType(), reg.getClockVal(), reg.getName(), reg.getNameKind(),
          reg.getAnnotations(), reg.getInnerSymAttr());
      reg.replaceAllUsesWith(newReg.getResult());
      reg.erase();
      madeModifications = true;
      continue;
    }

    // If the `RegResetOp` has an asynchronous reset and the reset value is not
    // a module-scoped constant when looking through wires and nodes, then
    // generate an error.  This implements the SFC's CheckResets pass.
    if (!reg.getResetSignal().getType().isa<AsyncResetType>())
      continue;
    if (isModuleScopedDrivenBy<ConstantOp, InvalidValueOp, SpecialConstantOp>(
            reg.getResetValue(), true, true, true))
      continue;
    auto resetDriver =
        getModuleScopedDriver(reg.getResetValue(), true, true, true);
    auto diag = reg.emitOpError()
                << "has an async reset, but its reset value is not driven with "
                   "a constant value through wires, nodes, or connects";
    diag.attachNote(resetDriver.getLoc()) << "reset driver is here";
    return signalPassFailure();
  }

  // Convert all invalid values to zero.
  for (auto inv : invalidOps) {
    // Delete invalids which have no uses.
    if (inv->getUses().empty()) {
      inv->erase();
      continue;
    }
    ImplicitLocOpBuilder builder(inv.getLoc(), inv);
    Value replacement =
        TypeSwitch<FIRRTLType, Value>(inv.getType())
            .Case<ClockType, AsyncResetType, ResetType>(
                [&](auto type) -> Value {
                  return builder.create<SpecialConstantOp>(
                      type, builder.getBoolAttr(false));
                })
            .Case<IntType>([&](IntType type) -> Value {
              return builder.create<ConstantOp>(type, getIntZerosAttr(type));
            })
            .Case<BundleType, FVectorType>([&](auto type) -> Value {
              auto width = circt::firrtl::getBitWidth(type);
              assert(width && "width must be inferred");
              auto zero = builder.create<ConstantOp>(APSInt(*width));
              return builder.create<BitCastOp>(type, zero);
            })
            .Default([&](auto) {
              llvm_unreachable("all types are supported");
              return Value();
            });
    inv.replaceAllUsesWith(replacement);
    inv.erase();
    madeModifications = true;
  }

  if (!madeModifications)
    return markAllAnalysesPreserved();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createSFCCompatPass() {
  return std::make_unique<SFCCompatPass>();
}
