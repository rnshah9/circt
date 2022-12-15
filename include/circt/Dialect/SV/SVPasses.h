//===- SVPasses.h - SV pass entry points ------------------------*- C++ -*-===//
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

#ifndef CIRCT_DIALECT_SV_SVPASSES_H
#define CIRCT_DIALECT_SV_SVPASSES_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"

namespace circt {
namespace sv {

std::unique_ptr<mlir::Pass> createPrettifyVerilogPass();
std::unique_ptr<mlir::Pass> createHWCleanupPass();
std::unique_ptr<mlir::Pass> createHWStubExternalModulesPass();
std::unique_ptr<mlir::Pass> createHWLegalizeModulesPass();
std::unique_ptr<mlir::Pass> createSVTraceIVerilogPass();
std::unique_ptr<mlir::Pass> createHWGeneratorCalloutPass();
std::unique_ptr<mlir::Pass> createHWMemSimImplPass(
    bool replSeqMem = false, bool ignoreReadEnableMem = false,
    bool stripMuxPragmas = false, bool disableMemRandomization = false,
    bool disableRegRandomization = false,
    bool addVivadoRAMAddressConflictSynthesisBugWorkaround = false);
std::unique_ptr<mlir::Pass>
createSVExtractTestCodePass(bool disableInstanceExtraction = false,
                            bool disableModuleInlining = false);
std::unique_ptr<mlir::Pass>
createHWExportModuleHierarchyPass(llvm::Optional<std::string> directory = {});
/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/SV/SVPasses.h.inc"

} // namespace sv
} // namespace circt

#endif // CIRCT_DIALECT_SV_SVPASSES_H
