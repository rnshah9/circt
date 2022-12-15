//===- Passes.h - ESI pass entry points -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_ESIPASSES_H
#define CIRCT_DIALECT_ESI_ESIPASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/Optional.h"
#include <memory>

namespace circt {
namespace esi {

std::unique_ptr<OperationPass<ModuleOp>> createESIEmitCollateralPass();
std::unique_ptr<OperationPass<ModuleOp>> createESIPhysicalLoweringPass();
std::unique_ptr<OperationPass<ModuleOp>> createESIPortLoweringPass();
std::unique_ptr<OperationPass<ModuleOp>> createESItoHWPass();
std::unique_ptr<OperationPass<ModuleOp>> createESIConnectServicesPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/ESI/ESIPasses.h.inc"

} // namespace esi
} // namespace circt

#endif // CIRCT_DIALECT_ESI_ESIPASSES_H
