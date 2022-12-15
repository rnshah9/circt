//===- Passes.h - Conversion Pass Construction and Registration -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This fle contains the declarations to register conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_PASSES_H
#define CIRCT_CONVERSION_PASSES_H

#include "circt/Conversion/AffineToPipeline.h"
#include "circt/Conversion/CalyxToFSM.h"
#include "circt/Conversion/CalyxToHW.h"
#include "circt/Conversion/CombToLLVM.h"
#include "circt/Conversion/ExportChiselInterface.h"
#include "circt/Conversion/ExportVerilog.h"
#include "circt/Conversion/FIRRTLToHW.h"
#include "circt/Conversion/FSMToSV.h"
#include "circt/Conversion/HWArithToHW.h"
#include "circt/Conversion/HWToLLHD.h"
#include "circt/Conversion/HWToLLVM.h"
#include "circt/Conversion/HWToSystemC.h"
#include "circt/Conversion/HandshakeToFIRRTL.h"
#include "circt/Conversion/HandshakeToHW.h"
#include "circt/Conversion/LLHDToLLVM.h"
#include "circt/Conversion/MooreToCore.h"
#include "circt/Conversion/PipelineToCalyx.h"
#include "circt/Conversion/PipelineToHW.h"
#include "circt/Conversion/SCFToCalyx.h"
#include "circt/Conversion/StandardToHandshake.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace circt {

// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "circt/Conversion/Passes.h.inc"

} // namespace circt

#endif // CIRCT_CONVERSION_PASSES_H
