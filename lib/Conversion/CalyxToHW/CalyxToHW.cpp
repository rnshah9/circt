//===- CalyxToHW.cpp - Translate Calyx into HW ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This is the main Calyx to HW Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/CalyxToHW.h"
#include "../PassDetail.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;
using namespace circt::calyx;
using namespace circt::comb;
using namespace circt::hw;
using namespace circt::seq;
using namespace circt::sv;

/// ConversionPatterns.

struct ConvertComponentOp : public OpConversionPattern<ComponentOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ComponentOp component, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<hw::PortInfo> hwInputInfo;
    auto portInfo = component.getPortInfo();
    for (auto [name, type, direction, _] : portInfo)
      hwInputInfo.push_back({name, hwDirection(direction), type});
    ModulePortInfo hwPortInfo(hwInputInfo);

    SmallVector<Value> argValues;
    auto hwMod = rewriter.create<HWModuleOp>(
        component.getLoc(), component.getNameAttr(), hwPortInfo,
        [&](OpBuilder &b, HWModulePortAccessor &ports) {
          for (auto [name, type, direction, _] : portInfo) {
            switch (direction) {
            case calyx::Direction::Input:
              assert(ports.getInput(name).getType() == type);
              argValues.push_back(ports.getInput(name));
              break;
            case calyx::Direction::Output:
              auto wire = b.create<sv::WireOp>(component.getLoc(), type, name);
              auto wireRead =
                  b.create<sv::ReadInOutOp>(component.getLoc(), wire);
              argValues.push_back(wireRead);
              ports.setOutput(name, wireRead);
              break;
            }
          }
        });

    auto *outputOp = hwMod.getBodyBlock()->getTerminator();
    rewriter.mergeBlocks(component.getBodyBlock(), hwMod.getBodyBlock(),
                         argValues);
    outputOp->moveAfter(&hwMod.getBodyBlock()->back());
    rewriter.eraseOp(component);
    return success();
  }

private:
  hw::PortDirection hwDirection(calyx::Direction dir) const {
    switch (dir) {
    case calyx::Direction::Input:
      return hw::PortDirection::INPUT;
    case calyx::Direction::Output:
      return hw::PortDirection::OUTPUT;
    }
    llvm_unreachable("unknown direction");
  }
};

struct ConvertWiresOp : public OpConversionPattern<WiresOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(WiresOp wires, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    HWModuleOp hwMod = wires->getParentOfType<HWModuleOp>();
    rewriter.inlineRegionBefore(wires.getBody(), hwMod.getBodyRegion(),
                                hwMod.getBodyRegion().end());
    rewriter.eraseOp(wires);
    rewriter.mergeBlockBefore(&hwMod.getBodyRegion().getBlocks().back(),
                              &hwMod.getBodyBlock()->back());
    return success();
  }
};

struct ConvertControlOp : public OpConversionPattern<ControlOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ControlOp control, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!control.getBodyBlock()->empty())
      return control.emitOpError("calyx control must be structural");
    rewriter.eraseOp(control);
    return success();
  }
};

struct ConvertAssignOp : public OpConversionPattern<calyx::AssignOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(calyx::AssignOp assign, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value dest = adaptor.getDest();

    // To make life easy in ConvertComponentOp, we read from the output wires so
    // the dialect conversion block argument mapping would work without a type
    // converter. This means assigns to ComponentOp outputs will try to assign
    // to a read from a wire, so we need to map to the wire.
    if (auto readInOut =
            dyn_cast<ReadInOutOp>(adaptor.getDest().getDefiningOp()))
      dest = readInOut.getInput();

    Value src = adaptor.getSrc();
    if (auto guard = adaptor.getGuard()) {
      auto zero =
          rewriter.create<hw::ConstantOp>(assign.getLoc(), src.getType(), 0);
      src = rewriter.create<MuxOp>(assign.getLoc(), guard, src, zero, false);
    }

    rewriter.replaceOpWithNewOp<sv::AssignOp>(assign, dest, src);

    return success();
  }
};

struct ConvertCellOp : public OpInterfaceConversionPattern<CellInterface> {
  using OpInterfaceConversionPattern::OpInterfaceConversionPattern;

  LogicalResult
  matchAndRewrite(CellInterface cell, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    assert(operands.empty() && "calyx cells do not have operands");

    SmallVector<Value> wires;
    ImplicitLocOpBuilder builder(cell.getLoc(), rewriter);
    convertPrimitiveOp(cell, wires, builder);
    if (wires.size() != cell.getPortInfo().size()) {
      auto diag = cell.emitOpError("couldn't convert to core primitive");
      for (Value wire : wires)
        diag.attachNote() << "with wire: " << wire;
      return diag;
    }

    rewriter.replaceOp(cell, wires);

    return success();
  }

private:
  void convertPrimitiveOp(Operation *op, SmallVectorImpl<Value> &wires,
                          ImplicitLocOpBuilder &b) const {
    TypeSwitch<Operation *>(op)
        // Comparison operations.
        .Case([&](EqLibOp op) {
          convertCompareBinaryOp(op, ICmpPredicate::eq, wires, b);
        })
        .Case([&](NeqLibOp op) {
          convertCompareBinaryOp(op, ICmpPredicate::ne, wires, b);
        })
        .Case([&](LtLibOp op) {
          convertCompareBinaryOp(op, ICmpPredicate::ult, wires, b);
        })
        .Case([&](LeLibOp op) {
          convertCompareBinaryOp(op, ICmpPredicate::ule, wires, b);
        })
        .Case([&](GtLibOp op) {
          convertCompareBinaryOp(op, ICmpPredicate::ugt, wires, b);
        })
        .Case([&](GeLibOp op) {
          convertCompareBinaryOp(op, ICmpPredicate::uge, wires, b);
        })
        .Case([&](SltLibOp op) {
          convertCompareBinaryOp(op, ICmpPredicate::slt, wires, b);
        })
        .Case([&](SleLibOp op) {
          convertCompareBinaryOp(op, ICmpPredicate::sle, wires, b);
        })
        .Case([&](SgtLibOp op) {
          convertCompareBinaryOp(op, ICmpPredicate::sgt, wires, b);
        })
        .Case([&](SgeLibOp op) {
          convertCompareBinaryOp(op, ICmpPredicate::sge, wires, b);
        })
        // Combinational arithmetic and logical operations.
        .Case([&](AddLibOp op) {
          convertArithBinaryOp<AddLibOp, AddOp>(op, wires, b);
        })
        .Case([&](SubLibOp op) {
          convertArithBinaryOp<SubLibOp, SubOp>(op, wires, b);
        })
        .Case([&](RshLibOp op) {
          convertArithBinaryOp<RshLibOp, ShrUOp>(op, wires, b);
        })
        .Case([&](SrshLibOp op) {
          convertArithBinaryOp<SrshLibOp, ShrSOp>(op, wires, b);
        })
        .Case([&](LshLibOp op) {
          convertArithBinaryOp<LshLibOp, ShlOp>(op, wires, b);
        })
        .Case([&](AndLibOp op) {
          convertArithBinaryOp<AndLibOp, AndOp>(op, wires, b);
        })
        .Case([&](OrLibOp op) {
          convertArithBinaryOp<OrLibOp, OrOp>(op, wires, b);
        })
        .Case([&](XorLibOp op) {
          convertArithBinaryOp<XorLibOp, XorOp>(op, wires, b);
        })
        // Pipelined arithmetic operations.
        .Case([&](MultPipeLibOp op) {
          convertPipelineOp<MultPipeLibOp, comb::MulOp>(op, wires, b);
        })
        .Case([&](DivUPipeLibOp op) {
          convertPipelineOp<DivUPipeLibOp, comb::DivUOp>(op, wires, b);
        })
        .Case([&](DivSPipeLibOp op) {
          convertPipelineOp<DivSPipeLibOp, comb::DivSOp>(op, wires, b);
        })
        .Case([&](RemSPipeLibOp op) {
          convertPipelineOp<RemSPipeLibOp, comb::ModSOp>(op, wires, b);
        })
        .Case([&](RemUPipeLibOp op) {
          convertPipelineOp<RemUPipeLibOp, comb::ModUOp>(op, wires, b);
        })
        // Sequential operations.
        .Case([&](RegisterOp op) {
          auto in =
              wireIn(op.getIn(), op.instanceName(), op.portName(op.getIn()), b);
          auto writeEn = wireIn(op.getWriteEn(), op.instanceName(),
                                op.portName(op.getWriteEn()), b);
          auto clk = wireIn(op.getClk(), op.instanceName(),
                            op.portName(op.getClk()), b);
          auto reset = wireIn(op.getReset(), op.instanceName(),
                              op.portName(op.getReset()), b);

          auto outReg = reg(in, clk, reset, op.instanceName() + "_reg", b);
          auto doneReg =
              reg(writeEn, clk, reset, op.instanceName() + "_done_reg", b);

          auto out = wireOut(outReg, op.instanceName(), "", b);
          auto done =
              wireOut(doneReg, op.instanceName(), op.portName(op.getDone()), b);
          wires.append({in.getInput(), writeEn.getInput(), clk.getInput(),
                        reset.getInput(), out, done});
        })
        // Unary operqations.
        .Case([&](SliceLibOp op) {
          auto in =
              wireIn(op.getIn(), op.instanceName(), op.portName(op.getIn()), b);
          auto outWidth = op.getOut().getType().getIntOrFloatBitWidth();

          auto extract = b.create<ExtractOp>(in, 0, outWidth);

          auto out =
              wireOut(extract, op.instanceName(), op.portName(op.getOut()), b);
          wires.append({in.getInput(), out});
        })
        .Case([&](NotLibOp op) {
          auto in =
              wireIn(op.getIn(), op.instanceName(), op.portName(op.getIn()), b);
          auto one = b.create<hw::ConstantOp>(op.getIn().getType(), 0);

          auto xorOp = b.create<XorOp>(in, one, false);

          auto out =
              wireOut(xorOp, op.instanceName(), op.portName(op.getOut()), b);
          wires.append({in.getInput(), out});
        })
        .Case([&](WireLibOp op) {
          auto wire = wireIn(op.getIn(), op.instanceName(), "", b);
          wires.append({wire.getInput(), wire});
        })
        .Case([&](PadLibOp op) {
          auto in =
              wireIn(op.getIn(), op.instanceName(), op.portName(op.getIn()), b);
          auto srcWidth = in.getType().getIntOrFloatBitWidth();
          auto destWidth = op.getOut().getType().getIntOrFloatBitWidth();
          auto zero = b.create<hw::ConstantOp>(op.getLoc(),
                                               APInt(destWidth - srcWidth, 0));
          auto padded = wireOut(b.createOrFold<comb::ConcatOp>(zero, in),
                                op.instanceName(), op.portName(op.getOut()), b);
          wires.append({in.getInput(), padded});
        })
        .Case([&](ExtSILibOp op) {
          auto in =
              wireIn(op.getIn(), op.instanceName(), op.portName(op.getIn()), b);
          auto extsi = wireOut(
              createOrFoldSExt(op.getLoc(), in, op.getOut().getType(), b),
              op.instanceName(), op.portName(op.getOut()), b);
          wires.append({in.getInput(), extsi});
        })
        .Default([](Operation *) { return SmallVector<Value>(); });
  }

  template <typename OpTy, typename ResultTy>
  void convertArithBinaryOp(OpTy op, SmallVectorImpl<Value> &wires,
                            ImplicitLocOpBuilder &b) const {
    auto left =
        wireIn(op.getLeft(), op.instanceName(), op.portName(op.getLeft()), b);
    auto right =
        wireIn(op.getRight(), op.instanceName(), op.portName(op.getRight()), b);

    auto add = b.create<ResultTy>(left, right, false);

    auto out = wireOut(add, op.instanceName(), op.portName(op.getOut()), b);
    wires.append({left.getInput(), right.getInput(), out});
  }

  template <typename OpTy>
  void convertCompareBinaryOp(OpTy op, ICmpPredicate pred,
                              SmallVectorImpl<Value> &wires,
                              ImplicitLocOpBuilder &b) const {
    auto left =
        wireIn(op.getLeft(), op.instanceName(), op.portName(op.getLeft()), b);
    auto right =
        wireIn(op.getRight(), op.instanceName(), op.portName(op.getRight()), b);

    auto add = b.create<ICmpOp>(pred, left, right, false);

    auto out = wireOut(add, op.instanceName(), op.portName(op.getOut()), b);
    wires.append({left.getInput(), right.getInput(), out});
  }

  template <typename SrcOpTy, typename TargetOpTy>
  void convertPipelineOp(SrcOpTy op, SmallVectorImpl<Value> &wires,
                         ImplicitLocOpBuilder &b) const {
    auto clk =
        wireIn(op.getClk(), op.instanceName(), op.portName(op.getClk()), b);
    auto reset =
        wireIn(op.getReset(), op.instanceName(), op.portName(op.getReset()), b);
    auto go = wireIn(op.getGo(), op.instanceName(), op.portName(op.getGo()), b);
    auto left =
        wireIn(op.getLeft(), op.instanceName(), op.portName(op.getLeft()), b);
    auto right =
        wireIn(op.getRight(), op.instanceName(), op.portName(op.getRight()), b);
    wires.append({clk.getInput(), reset.getInput(), go.getInput(),
                  left.getInput(), right.getInput()});

    auto targetOp = b.create<TargetOpTy>(left, right, false);
    for (auto &&[targetRes, sourceRes] :
         llvm::zip(targetOp->getResults(), op.getOutputPorts())) {
      auto portName = op.portName(sourceRes);
      auto resReg = reg(targetRes, clk, reset,
                        createName(op.instanceName(), portName), b);
      wires.push_back(wireOut(resReg, op.instanceName(), portName, b));
    }

    auto doneReg = reg(go, clk, reset,
                       op.instanceName() + "_" + op.portName(op.getDone()), b);
    auto done =
        wireOut(doneReg, op.instanceName(), op.portName(op.getDone()), b);
    wires.push_back(done);
  }

  ReadInOutOp wireIn(Value source, StringRef instanceName, StringRef portName,
                     ImplicitLocOpBuilder &b) const {
    auto wire =
        b.create<WireOp>(source.getType(), createName(instanceName, portName));
    return b.create<ReadInOutOp>(wire);
  }

  ReadInOutOp wireOut(Value source, StringRef instanceName, StringRef portName,
                      ImplicitLocOpBuilder &b) const {
    auto wire =
        b.create<WireOp>(source.getType(), createName(instanceName, portName));
    b.create<sv::AssignOp>(wire, source);
    return b.create<ReadInOutOp>(wire);
  }

  CompRegOp reg(Value source, Value clock, Value reset, Twine name,
                ImplicitLocOpBuilder &b) const {
    auto resetValue = b.create<hw::ConstantOp>(source.getType(), 0);
    auto regName = b.getStringAttr(name);
    return b.create<CompRegOp>(source.getType(), source, clock, regName, reset,
                               resetValue, regName);
  }

  std::string createName(StringRef instanceName, StringRef portName) const {
    std::string name = instanceName.str();
    if (!portName.empty())
      name += ("_" + portName).str();
    return name;
  }
};

/// Pass entrypoint.

namespace {
class CalyxToHWPass : public CalyxToHWBase<CalyxToHWPass> {
public:
  void runOnOperation() override;

private:
  LogicalResult runOnModule(ModuleOp module);
};
} // end anonymous namespace

void CalyxToHWPass::runOnOperation() {
  ModuleOp mod = getOperation();
  if (failed(runOnModule(mod)))
    return signalPassFailure();
}

LogicalResult CalyxToHWPass::runOnModule(ModuleOp module) {
  MLIRContext &context = getContext();

  ConversionTarget target(context);
  target.addIllegalDialect<CalyxDialect>();
  target.addLegalDialect<HWDialect>();
  target.addLegalDialect<CombDialect>();
  target.addLegalDialect<SeqDialect>();
  target.addLegalDialect<SVDialect>();

  RewritePatternSet patterns(&context);
  patterns.add<ConvertComponentOp>(&context);
  patterns.add<ConvertWiresOp>(&context);
  patterns.add<ConvertControlOp>(&context);
  patterns.add<ConvertCellOp>(&context);
  patterns.add<ConvertAssignOp>(&context);

  return applyPartialConversion(module, target, std::move(patterns));
}

std::unique_ptr<mlir::Pass> circt::createCalyxToHWPass() {
  return std::make_unique<CalyxToHWPass>();
}
