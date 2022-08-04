//===- HandshakeOps.cpp - Handshake MLIR Operations -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the Handshake operations struct.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include <set>

using namespace circt;
using namespace circt::handshake;

namespace circt {
namespace handshake {
#include "circt/Dialect/Handshake/HandshakeCanonicalization.h.inc"
}
} // namespace circt

static std::string defaultOperandName(unsigned int idx) {
  return "in" + std::to_string(idx);
}

namespace sost {
// Sized Operation with Single Type (SOST).
// These are operation on the format:
//   opname operands optAttrDict : dataType
// containing a 'size' (=operands.size()) and 'dataType' attribute.
// if 'explicitSize' is set, the operation is parsed as follows:
//   opname [$size] operands opAttrDict : dataType
// If the datatype of the operation is "None", the operation is also added a
// {control = true} attribute. if 'alwaysControl' is set, the control attribute
// is always set.

void addAttributes(OperationState &result, int size, Type dataType,
                   bool alwaysControl = false) {
  result.addAttribute(
      "size",
      IntegerAttr::get(IntegerType::get(result.getContext(), 32), size));
  result.addAttribute("dataType", TypeAttr::get(dataType));
  if (dataType.isa<NoneType>() || alwaysControl)
    result.addAttribute("control", BoolAttr::get(result.getContext(), true));
}

static ParseResult parseIntInSquareBrackets(OpAsmParser &parser, int &v) {
  if (parser.parseLSquare() || parser.parseInteger(v) || parser.parseRSquare())
    return failure();
  return success();
}

static ParseResult
parseOperation(OpAsmParser &parser,
               SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
               OperationState &result, int &size, Type &type, bool explicitSize,
               bool alwaysControl = false) {
  if (explicitSize)
    if (parseIntInSquareBrackets(parser, size))
      return failure();

  if (parser.parseOperandList(operands) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(type))
    return failure();

  if (!explicitSize)
    size = operands.size();

  sost::addAttributes(result, size, type, alwaysControl);
  return success();
}

static void printOp(OpAsmPrinter &p, Operation *op, bool explicitSize) {
  if (explicitSize) {
    int size = op->getAttrOfType<IntegerAttr>("size").getValue().getZExtValue();
    p << " [" << size << "]";
  }
  Type type = op->getAttrOfType<TypeAttr>("dataType").getValue();
  p << " " << op->getOperands();
  p.printOptionalAttrDict((op)->getAttrs(), {"size", "dataType", "control"});
  p << " : " << type;
}
} // namespace sost

void ForkOp::build(OpBuilder &builder, OperationState &result, Value operand,
                   int outputs) {
  auto type = operand.getType();

  // Fork has results as many as there are successor ops
  result.types.append(outputs, type);

  // Single operand
  result.addOperands(operand);
  sost::addAttributes(result, outputs, type);
}

static ParseResult parseForkOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> allOperands;
  Type type;
  ArrayRef<Type> operandTypes(type);
  SmallVector<Type, 1> resultTypes;
  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  int size;
  if (sost::parseOperation(parser, allOperands, result, size, type,
                           /*explicitSize=*/true))
    return failure();

  resultTypes.assign(size, type);
  result.addTypes(resultTypes);
  if (parser.resolveOperands(allOperands, operandTypes, allOperandLoc,
                             result.operands))
    return failure();
  return success();
}

ParseResult ForkOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseForkOp(parser, result);
}

void ForkOp::print(OpAsmPrinter &p) { sost::printOp(p, *this, true); }

namespace {

struct EliminateUnusedForkResultsPattern : mlir::OpRewritePattern<ForkOp> {
  using mlir::OpRewritePattern<ForkOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForkOp op,
                                PatternRewriter &rewriter) const override {
    std::set<unsigned> unusedIndexes;

    for (auto res : llvm::enumerate(op.getResults()))
      if (res.value().getUses().empty())
        unusedIndexes.insert(res.index());

    if (unusedIndexes.size() == 0)
      return failure();

    // Create a new fork op, dropping the unused results.
    rewriter.setInsertionPoint(op);
    auto newFork =
        rewriter.create<ForkOp>(op.getLoc(), op.getOperand(),
                                op.getNumResults() - unusedIndexes.size());
    rewriter.updateRootInPlace(op, [&] {
      unsigned i = 0;
      for (auto oldRes : llvm::enumerate(op.getResults()))
        if (unusedIndexes.count(oldRes.index()) == 0)
          oldRes.value().replaceAllUsesWith(newFork.getResult(i++));
    });
    rewriter.eraseOp(op);
    return success();
  }
};

struct EliminateForkToForkPattern : mlir::OpRewritePattern<ForkOp> {
  using mlir::OpRewritePattern<ForkOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForkOp op,
                                PatternRewriter &rewriter) const override {
    auto parentForkOp = op.getOperand().getDefiningOp<ForkOp>();
    if (!parentForkOp)
      return failure();

    /// Create the fork with as many outputs as the two source forks.
    /// Keeping the op.operand() output may or may not be redundant (dependning
    /// on if op is the single user of the value), but we'll let
    /// EliminateUnusedForkResultsPattern apply in that case.
    unsigned totalNumOuts = op.size() + parentForkOp.size();
    rewriter.updateRootInPlace(parentForkOp, [&] {
      /// Create a new parent fork op which produces all of the fork outputs and
      /// replace all of the uses of the old results.
      auto newParentForkOp = rewriter.create<ForkOp>(
          parentForkOp.getLoc(), parentForkOp.getOperand(), totalNumOuts);

      for (auto it :
           llvm::zip(parentForkOp->getResults(), newParentForkOp.getResults()))
        std::get<0>(it).replaceAllUsesWith(std::get<1>(it));
      rewriter.eraseOp(parentForkOp);

      /// Replace the results of the matches fork op with the corresponding
      /// results of the new parent fork op.
      rewriter.replaceOp(op, newParentForkOp.getResults().take_back(op.size()));
    });
    return success();
  }
};

} // namespace

void handshake::ForkOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                    MLIRContext *context) {
  results.insert<circt::handshake::EliminateSimpleForksPattern,
                 EliminateForkToForkPattern, EliminateUnusedForkResultsPattern>(
      context);
}

void LazyForkOp::build(OpBuilder &builder, OperationState &result,
                       Value operand, int outputs) {
  auto type = operand.getType();

  // Fork has results as many as there are successor ops
  result.types.append(outputs, type);

  // Single operand
  result.addOperands(operand);

  // Fork is control-only if it is the no-data output of a ControlMerge or a
  // StartOp
  auto *op = operand.getDefiningOp();
  bool isControl = isa_and_nonnull<ControlMergeOp, StartOp>(op) &&
                   operand == op->getResult(0);

  // Alternatively, the control signal could originate from a BlockArgument
  isControl = isControl || (operand.isa<BlockArgument>() &&
                            operand.getType().isa<NoneType>());
  sost::addAttributes(result, outputs, type, isControl);
}

ParseResult LazyForkOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseForkOp(parser, result);
}

void LazyForkOp::print(OpAsmPrinter &p) { sost::printOp(p, *this, true); }

void MergeOp::build(OpBuilder &builder, OperationState &result,
                    ValueRange operands) {
  assert(operands.size() != 0 &&
         "Expected at least one operand to this merge op.");
  auto type = operands.front().getType();
  result.types.push_back(type);
  result.addOperands(operands);
  sost::addAttributes(result, operands.size(), type);
}

ParseResult MergeOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> allOperands;
  Type type;
  ArrayRef<Type> operandTypes(type);
  SmallVector<Type, 1> resultTypes, dataOperandsTypes;
  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  int size;
  if (sost::parseOperation(parser, allOperands, result, size, type, false))
    return failure();

  dataOperandsTypes.assign(size, type);
  resultTypes.push_back(type);
  result.addTypes(resultTypes);
  if (parser.resolveOperands(allOperands, dataOperandsTypes, allOperandLoc,
                             result.operands))
    return failure();
  return success();
}

void MergeOp::print(OpAsmPrinter &p) { sost::printOp(p, *this, false); }

void MergeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.insert<circt::handshake::EliminateSimpleMergesPattern>(context);
}

/// Returns a dematerialized version of the value 'v', defined as the source of
/// the value before passing through a buffer or fork operation.
static Value getDematerialized(Value v) {
  Operation *parentOp = v.getDefiningOp();
  if (!parentOp)
    return v;

  return llvm::TypeSwitch<Operation *, Value>(parentOp)
      .Case<ForkOp>(
          [&](ForkOp op) { return getDematerialized(op.getOperand()); })
      .Case<BufferOp>(
          [&](BufferOp op) { return getDematerialized(op.getOperand()); })
      .Default([&](auto) { return v; });
}

namespace {

/// Eliminates muxes with identical data inputs. Data inputs are inspected as
/// their dematerialized versions. This has the side effect of any subsequently
/// unused buffers are DCE'd and forks are optimized to be narrower.
struct EliminateSimpleMuxesPattern : mlir::OpRewritePattern<MuxOp> {
  using mlir::OpRewritePattern<MuxOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MuxOp op,
                                PatternRewriter &rewriter) const override {
    Value firstDataOperand = getDematerialized(op.dataOperands()[0]);
    if (!llvm::all_of(op.dataOperands(), [&](Value operand) {
          return getDematerialized(operand) == firstDataOperand;
        }))
      return failure();
    rewriter.replaceOp(op, firstDataOperand);
    return success();
  }
};

struct EliminateUnaryMuxesPattern : OpRewritePattern<MuxOp> {
  using mlir::OpRewritePattern<MuxOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MuxOp op,
                                PatternRewriter &rewriter) const override {
    if (op.dataOperands().size() != 1)
      return failure();

    rewriter.replaceOp(op, op.dataOperands()[0]);
    return success();
  }
};

} // namespace

void MuxOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.insert<EliminateSimpleMuxesPattern, EliminateUnaryMuxesPattern>(
      context);
}

void MuxOp::build(OpBuilder &builder, OperationState &result, Value anyInput,
                  int inputs) {
  // Output type
  auto type = anyInput.getType();
  result.types.push_back(type);

  // Select operand
  result.addOperands(anyInput);

  // Data operands
  for (int i = 0, e = inputs; i < e; ++i)
    result.addOperands(anyInput);
  sost::addAttributes(result, inputs, type);
}

void MuxOp::build(OpBuilder &builder, OperationState &result, Value selOperand,
                  ValueRange _dataOperands) {
  assert(_dataOperands.size() != 0 && "Building mux with no inputs?");
  Type dataType = _dataOperands[0].getType();
  result.addTypes({dataType});
  result.addOperands({selOperand});
  result.addOperands(_dataOperands);
  sost::addAttributes(result, _dataOperands.size(), dataType);
}

std::string handshake::MuxOp::getOperandName(unsigned int idx) {
  return idx == 0 ? "select" : defaultOperandName(idx - 1);
}

ParseResult MuxOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand selectOperand;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> allOperands;
  Type selectType, dataType;
  SmallVector<Type, 1> dataOperandsTypes;
  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperand(selectOperand) || parser.parseLSquare() ||
      parser.parseOperandList(allOperands) || parser.parseRSquare() ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(selectType) || parser.parseComma() ||
      parser.parseType(dataType))
    return failure();

  int size = allOperands.size();
  sost::addAttributes(result, size, dataType);
  dataOperandsTypes.assign(size, dataType);
  result.addTypes(dataType);
  allOperands.insert(allOperands.begin(), selectOperand);
  if (parser.resolveOperands(
          allOperands,
          llvm::concat<const Type>(ArrayRef<Type>(selectType),
                                   ArrayRef<Type>(dataOperandsTypes)),
          allOperandLoc, result.operands))
    return failure();
  return success();
}

void MuxOp::print(OpAsmPrinter &p) {
  Type selectType = selectOperand().getType();
  auto ops = getOperands();
  p << ' ' << ops.front();
  p << " [";
  p.printOperands(ops.drop_front());
  p << "]";
  p.printOptionalAttrDict((*this)->getAttrs(), {"dataType", "size", "control"});
  p << " : " << selectType << ", " << dataType();
}

LogicalResult MuxOp::verify() {
  unsigned numDataOperands = static_cast<int>(dataOperands().size());
  auto selectType = selectOperand().getType();

  unsigned selectBits;
  if (auto integerType = selectType.dyn_cast<IntegerType>())
    selectBits = integerType.getWidth();
  else if (selectType.isIndex())
    selectBits = IndexType::kInternalStorageBitWidth;
  else
    return emitError("unsupported type for select operand: ") << selectType;

  double maxDataOperands = std::pow(2, selectBits);
  if (numDataOperands > maxDataOperands)
    return emitError("select bitwidth was ")
           << selectBits << ", which can mux "
           << static_cast<int64_t>(maxDataOperands) << " operands, but found "
           << numDataOperands << " operands";

  return success();
}

std::string handshake::ControlMergeOp::getResultName(unsigned int idx) {
  assert(idx == 0 || idx == 1);
  return idx == 0 ? "dataOut" : "index";
}

void ControlMergeOp::build(OpBuilder &builder, OperationState &result,
                           Value operand, int inputs) {
  auto type = operand.getType();
  result.types.push_back(type);
  // Second result gives the input index to the muxes
  // Number of bits depends on encoding (log2/1-hot)
  result.types.push_back(builder.getIndexType());

  // Operand to keep defining value (used when connecting merges)
  // Removed afterwards
  result.addOperands(operand);

  // Operands from predecessor blocks
  for (int i = 0, e = inputs; i < e; ++i)
    result.addOperands(operand);

  sost::addAttributes(result, inputs, type);
}

void ControlMergeOp::build(OpBuilder &builder, OperationState &result,
                           ValueRange operands) {
  auto type = operands[0].getType();
  result.types.push_back(type);
  // Selected index.
  result.types.push_back(builder.getIndexType());
  result.addOperands(operands);
  sost::addAttributes(result, operands.size(), type);
}

ParseResult ControlMergeOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> allOperands;
  Type type;
  ArrayRef<Type> operandTypes(type);
  SmallVector<Type, 1> resultTypes, dataOperandsTypes;
  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  int size;
  if (sost::parseOperation(parser, allOperands, result, size, type,
                           /*explicitSize=*/false))
    return failure();

  dataOperandsTypes.assign(size, type);
  resultTypes.push_back(type);
  resultTypes.push_back(IndexType::get(parser.getContext()));
  result.addTypes(resultTypes);
  if (parser.resolveOperands(allOperands, dataOperandsTypes, allOperandLoc,
                             result.operands))
    return failure();
  return success();
}

void ControlMergeOp::print(OpAsmPrinter &p) { sost::printOp(p, *this, false); }

LogicalResult FuncOp::verify() {
  // If this function is external there is nothing to do.
  if (isExternal())
    return success();

  // Verify that the argument list of the function and the arg list of the
  // entry block line up.  The trait already verified that the number of
  // arguments is the same between the signature and the block.
  auto fnInputTypes = getArgumentTypes();
  Block &entryBlock = front();

  for (unsigned i = 0, e = entryBlock.getNumArguments(); i != e; ++i)
    if (fnInputTypes[i] != entryBlock.getArgument(i).getType())
      return emitOpError("type of entry block argument #")
             << i << '(' << entryBlock.getArgument(i).getType()
             << ") must match the type of the corresponding argument in "
             << "function signature(" << fnInputTypes[i] << ')';

  // Verify that we have a name for each argument and result of this function.
  auto verifyPortNameAttr = [&](StringRef attrName,
                                unsigned numIOs) -> LogicalResult {
    auto portNamesAttr = (*this)->getAttrOfType<ArrayAttr>(attrName);

    if (!portNamesAttr)
      return emitOpError() << "expected attribute '" << attrName << "'.";

    auto portNames = portNamesAttr.getValue();
    if (portNames.size() != numIOs)
      return emitOpError() << "attribute '" << attrName << "' has "
                           << portNames.size()
                           << " entries but is expected to have " << numIOs
                           << ".";

    if (llvm::any_of(portNames,
                     [&](Attribute attr) { return !attr.isa<StringAttr>(); }))
      return emitOpError() << "expected all entries in attribute '" << attrName
                           << "' to be strings.";

    return success();
  };
  if (failed(verifyPortNameAttr("argNames", getNumArguments())))
    return failure();
  if (failed(verifyPortNameAttr("resNames", getNumResults())))
    return failure();

  // Verify that all memrefs have a corresponding extmemory operation
  for (auto arg : entryBlock.getArguments()) {
    if (!arg.getType().isa<MemRefType>())
      continue;
    if (arg.getUsers().empty() ||
        !isa<ExternalMemoryOp>(*arg.getUsers().begin()))
      return emitOpError("expected that block argument #")
             << arg.getArgNumber() << " is used by an 'extmemory' operation";
  }

  return success();
}

/// Parses a FuncOp signature using
/// mlir::function_interface_impl::parseFunctionSignature while getting access
/// to the parsed SSA names to store as attributes.
static ParseResult parseFuncOpArgs(
    OpAsmParser &parser, SmallVectorImpl<OpAsmParser::Argument> &entryArgs,
    SmallVectorImpl<Attribute> &argNames, SmallVectorImpl<Type> &resTypes,
    SmallVectorImpl<DictionaryAttr> &resAttrs) {
  auto *context = parser.getContext();

  bool isVariadic;
  if (mlir::function_interface_impl::parseFunctionSignature(
          parser, /*allowVariadic=*/true, entryArgs, isVariadic, resTypes,
          resAttrs)
          .failed())
    return failure();

  llvm::transform(entryArgs, std::back_inserter(argNames), [&](auto arg) {
    return StringAttr::get(context, arg.ssaName.name.drop_front());
  });

  return success();
}

/// Generates names for a handshake.func input and output arguments, based on
/// the number of args as well as a prefix.
static SmallVector<Attribute> getFuncOpNames(Builder &builder, TypeRange types,
                                             StringRef prefix) {
  SmallVector<Attribute> resNames;
  llvm::transform(
      llvm::enumerate(types), std::back_inserter(resNames), [&](auto it) {
        bool lastOperand = it.index() == types.size() - 1;
        std::string suffix = lastOperand && it.value().template isa<NoneType>()
                                 ? "Ctrl"
                                 : std::to_string(it.index());
        return builder.getStringAttr(prefix + suffix);
      });
  return resNames;
}

void handshake::FuncOp::build(OpBuilder &builder, OperationState &state,
                              StringRef name, FunctionType type,
                              ArrayRef<NamedAttribute> attrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getTypeAttrName(), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());

  if (const auto *argNamesAttrIt = llvm::find_if(
          attrs, [&](auto attr) { return attr.getName() == "argNames"; });
      argNamesAttrIt == attrs.end())
    state.addAttribute("argNames", builder.getArrayAttr({}));

  if (llvm::find_if(attrs, [&](auto attr) {
        return attr.getName() == "resNames";
      }) == attrs.end())
    state.addAttribute("resNames", builder.getArrayAttr({}));

  state.addRegion();
}

/// Helper function for appending a string to an array attribute, and
/// rewriting the attribute back to the operation.
static void addStringToStringArrayAttr(Builder &builder, Operation *op,
                                       StringRef attrName, StringAttr str) {
  llvm::SmallVector<Attribute> attrs;
  llvm::copy(op->getAttrOfType<ArrayAttr>(attrName).getValue(),
             std::back_inserter(attrs));
  attrs.push_back(str);
  op->setAttr(attrName, builder.getArrayAttr(attrs));
}

void handshake::FuncOp::resolveArgAndResNames() {
  Builder builder(getContext());

  /// Generate a set of fallback names. These are used in case names are
  /// missing from the currently set arg- and res name attributes.
  auto fallbackArgNames = getFuncOpNames(builder, getArgumentTypes(), "in");
  auto fallbackResNames = getFuncOpNames(builder, getResultTypes(), "out");
  auto argNames = getArgNames().getValue();
  auto resNames = getResNames().getValue();

  /// Use fallback names where actual names are missing.
  auto resolveNames = [&](auto &fallbackNames, auto &actualNames,
                          StringRef attrName) {
    for (auto fallbackName : llvm::enumerate(fallbackNames)) {
      if (actualNames.size() <= fallbackName.index())
        addStringToStringArrayAttr(
            builder, this->getOperation(), attrName,
            fallbackName.value().template cast<StringAttr>());
    }
  };
  resolveNames(fallbackArgNames, argNames, "argNames");
  resolveNames(fallbackResNames, resNames, "resNames");
}

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  StringAttr nameAttr;
  SmallVector<OpAsmParser::Argument> args;
  SmallVector<Type> resTypes;
  SmallVector<DictionaryAttr> resAttributes;
  SmallVector<Attribute> argNames;

  // Parse visibility.
  (void)mlir::impl::parseOptionalVisibilityKeyword(parser, result.attributes);

  // Parse signature
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes) ||
      parseFuncOpArgs(parser, args, argNames, resTypes, resAttributes))
    return failure();
  mlir::function_interface_impl::addArgAndResultAttrs(builder, result, args,
                                                      resAttributes);

  // Set function type
  SmallVector<Type> argTypes;
  for (auto arg : args)
    argTypes.push_back(arg.type);

  result.addAttribute(
      handshake::FuncOp::getTypeAttrName(),
      TypeAttr::get(builder.getFunctionType(argTypes, resTypes)));

  // Parse attributes
  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();

  // If argNames and resNames wasn't provided manually, infer argNames attribute
  // from the parsed SSA names and resNames from our naming convention.
  if (!result.attributes.get("argNames"))
    result.addAttribute("argNames", builder.getArrayAttr(argNames));
  if (!result.attributes.get("resNames")) {
    auto resNames = getFuncOpNames(builder, resTypes, "out");
    result.addAttribute("resNames", builder.getArrayAttr(resNames));
  }

  // Parse region
  auto *body = result.addRegion();
  return parser.parseRegion(*body, args);
}

void FuncOp::print(OpAsmPrinter &p) {
  mlir::function_interface_impl::printFunctionOp(p, *this, /*isVariadic=*/true);
}

namespace {
struct EliminateSimpleControlMergesPattern
    : mlir::OpRewritePattern<ControlMergeOp> {
  using mlir::OpRewritePattern<ControlMergeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ControlMergeOp op,
                                PatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult EliminateSimpleControlMergesPattern::matchAndRewrite(
    ControlMergeOp op, PatternRewriter &rewriter) const {
  auto dataResult = op.result();
  auto choiceResult = op.index();
  auto choiceUnused = choiceResult.use_empty();
  if (!choiceUnused && !choiceResult.hasOneUse())
    return failure();

  Operation *choiceUser;
  if (choiceResult.hasOneUse()) {
    choiceUser = choiceResult.getUses().begin().getUser();
    if (!isa<SinkOp>(choiceUser))
      return failure();
  }

  auto merge = rewriter.create<MergeOp>(op.getLoc(), op.dataOperands());

  for (auto &use : dataResult.getUses()) {
    auto *user = use.getOwner();
    rewriter.updateRootInPlace(
        user, [&]() { user->setOperand(use.getOperandNumber(), merge); });
  }

  if (choiceUnused) {
    rewriter.eraseOp(op);
    return success();
  }

  rewriter.eraseOp(choiceUser);
  rewriter.eraseOp(op);
  return success();
}

void ControlMergeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.insert<EliminateSimpleControlMergesPattern>(context);
}

void handshake::BranchOp::build(OpBuilder &builder, OperationState &result,
                                Value dataOperand) {
  auto type = dataOperand.getType();
  result.types.push_back(type);
  result.addOperands(dataOperand);

  // Branch is control-only if it is the no-data output of a ControlMerge or a
  // StartOp. This holds because Branches are inserted before Forks
  auto *op = dataOperand.getDefiningOp();
  bool isControl = isa_and_nonnull<ControlMergeOp, StartOp>(op) &&
                   dataOperand == op->getResult(0);

  // Alternatively, the control signal could originate from a BlockArgument
  isControl = isControl || (dataOperand.isa<BlockArgument>() &&
                            dataOperand.getType().isa<NoneType>());
  sost::addAttributes(result, 1, type, isControl);
}

void BranchOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<circt::handshake::EliminateSimpleBranchesPattern>(context);
}

ParseResult BranchOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> allOperands;
  Type type;
  ArrayRef<Type> operandTypes(type);
  SmallVector<Type, 1> dataOperandsTypes;
  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  int size;
  if (sost::parseOperation(parser, allOperands, result, size, type,
                           /*explicitSize=*/false))
    return failure();

  dataOperandsTypes.assign(size, type);
  result.addTypes({type});
  if (parser.resolveOperands(allOperands, dataOperandsTypes, allOperandLoc,
                             result.operands))
    return failure();
  return success();
}

void BranchOp::print(OpAsmPrinter &p) { sost::printOp(p, *this, false); }

ParseResult ConditionalBranchOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> allOperands;
  Type dataType;
  SmallVector<Type> operandTypes;
  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(dataType))
    return failure();

  if (allOperands.size() != 2)
    return parser.emitError(parser.getCurrentLocation(),
                            "Expected exactly 2 operands");

  result.addTypes({dataType, dataType});
  operandTypes.push_back(IntegerType::get(parser.getContext(), 1));
  operandTypes.push_back(dataType);
  if (parser.resolveOperands(allOperands, operandTypes, allOperandLoc,
                             result.operands))
    return failure();

  if (dataType.isa<NoneType>())
    result.addAttribute("control", BoolAttr::get(parser.getContext(), true));

  return success();
}

void ConditionalBranchOp::print(OpAsmPrinter &p) {
  Type type = dataOperand().getType();
  p << " " << getOperands();
  p.printOptionalAttrDict((*this)->getAttrs(), {"size", "dataType", "control"});
  p << " : " << type;
}

std::string handshake::ConditionalBranchOp::getOperandName(unsigned int idx) {
  assert(idx == 0 || idx == 1);
  return idx == 0 ? "cond" : "data";
}

std::string handshake::ConditionalBranchOp::getResultName(unsigned int idx) {
  assert(idx == 0 || idx == 1);
  return idx == ConditionalBranchOp::falseIndex ? "outFalse" : "outTrue";
}

void handshake::ConditionalBranchOp::build(OpBuilder &builder,
                                           OperationState &result,
                                           Value condOperand,
                                           Value dataOperand) {
  auto type = dataOperand.getType();
  result.types.append(2, type);
  result.addOperands(condOperand);
  result.addOperands(dataOperand);

  // Branch is control-only if it is the no-data output of a ControlMerge or a
  // StartOp This holds because Branches are inserted before Forks
  auto *op = dataOperand.getDefiningOp();
  bool isControl = isa_and_nonnull<ControlMergeOp, StartOp>(op) &&
                   dataOperand == op->getResult(0);

  if (isControl || type.isa<NoneType>())
    result.addAttribute("control", builder.getBoolAttr(true));
}

ParseResult SelectOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> allOperands;
  Type dataType;
  SmallVector<Type> operandTypes;
  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(dataType))
    return failure();

  if (allOperands.size() != 3)
    return parser.emitError(parser.getCurrentLocation(),
                            "Expected exactly 3 operands");

  result.addTypes({dataType});
  operandTypes.push_back(IntegerType::get(parser.getContext(), 1));
  operandTypes.push_back(dataType);
  operandTypes.push_back(dataType);
  if (parser.resolveOperands(allOperands, operandTypes, allOperandLoc,
                             result.operands))
    return failure();

  if (dataType.isa<NoneType>())
    result.addAttribute("control", BoolAttr::get(parser.getContext(), true));

  return success();
}

void SelectOp::print(OpAsmPrinter &p) {
  Type type = trueOperand().getType();
  p << " " << getOperands();
  p.printOptionalAttrDict((*this)->getAttrs(), {"size", "dataType", "control"});
  p << " : " << type;
}

std::string handshake::SelectOp::getOperandName(unsigned int idx) {
  switch (idx) {
  case 0:
    return "sel";
  case 1:
    return "true";
  case 2:
    return "false";
  default:
    llvm_unreachable("Expected exactly 3 operands");
  }
}

void handshake::SelectOp::build(OpBuilder &builder, OperationState &result,
                                Value condOperand, Value trueOperand,
                                Value falseOperand) {
  auto type = trueOperand.getType();
  result.types.append(1, type);
  result.addOperands(condOperand);
  result.addOperands(trueOperand);
  result.addOperands(falseOperand);
  result.addAttribute("dataType", TypeAttr::get(type));
  result.addAttribute(
      "control", BoolAttr::get(builder.getContext(), type.isa<NoneType>()));
}

void StartOp::build(OpBuilder &builder, OperationState &result) {
  // Control-only output, has no type
  auto type = builder.getNoneType();
  result.types.push_back(type);
  result.addAttribute("control", builder.getBoolAttr(true));
}

void EndOp::build(OpBuilder &builder, OperationState &result, Value operand) {
  result.addOperands(operand);
}

void handshake::ReturnOp::build(OpBuilder &builder, OperationState &result,
                                ValueRange operands) {
  result.addOperands(operands);
}

void SinkOp::build(OpBuilder &builder, OperationState &result, Value operand) {
  result.addOperands(operand);
  sost::addAttributes(result, 1, operand.getType());
}

void SinkOp::build(OpBuilder &builder, OperationState &odsState,
                   TypeRange resultTypes, ValueRange operands,
                   ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  build(builder, odsState, operands[0]);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ParseResult SinkOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> allOperands;
  Type type;
  ArrayRef<Type> operandTypes(type);
  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  int size;
  if (sost::parseOperation(parser, allOperands, result, size, type, false))
    return failure();

  if (parser.resolveOperands(allOperands, operandTypes, allOperandLoc,
                             result.operands))
    return failure();
  return success();
}

void SinkOp::print(OpAsmPrinter &p) { sost::printOp(p, *this, false); }

std::string handshake::ConstantOp::getOperandName(unsigned int idx) {
  assert(idx == 0);
  return "ctrl";
}

static void setupSourceOp(OperationState &result) {
  result.addTypes(NoneType::get(result.getContext()));
  sost::addAttributes(result, 1, result.types[0], /*alwaysControl=*/true);
}

void SourceOp::build(OpBuilder &builder, OperationState &result) {
  setupSourceOp(result);
}

ParseResult SourceOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  setupSourceOp(result);
  return success();
}

void SourceOp::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(), {"size", "dataType", "control"});
}

LogicalResult ConstantOp::verify() {
  // Verify that the type of the provided value is equal to the result type.
  if ((*this)->getAttr("value").getType() != getResult().getType())
    return emitOpError()
           << "constant value type differs from operation result type.";

  return success();
}

void handshake::ConstantOp::build(OpBuilder &builder, OperationState &result,
                                  Attribute value, Value operand) {
  result.addOperands(operand);

  auto type = value.getType();
  result.types.push_back(type);

  result.addAttribute("value", value);
}

void handshake::ConstantOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<circt::handshake::EliminateSunkConstantsPattern>(context);
}

void handshake::TerminatorOp::build(OpBuilder &builder, OperationState &result,
                                    ArrayRef<Block *> successors) {
  // Add all the successor blocks of the block which contains this terminator
  result.addSuccessors(successors);
}

LogicalResult BufferOp::verify() {
  // Verify that exactly 'size' number of initial values have been provided, if
  // an initializer list have been provided.
  if (auto initVals = initValues()) {
    if (!isSequential())
      return emitOpError()
             << "only bufferType buffers are allowed to have initial values.";

    auto nInits = initVals->size();
    if (nInits != size())
      return emitOpError() << "expected " << size() << " init values but got "
                           << nInits << ".";
  }

  return success();
}

void handshake::BufferOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<circt::handshake::EliminateSunkBuffersPattern>(context);
}

void handshake::BufferOp::build(OpBuilder &builder, OperationState &result,
                                Type innerType, int size, Value operand,
                                BufferTypeEnum bufferType) {
  result.addOperands(operand);
  sost::addAttributes(result, size, innerType);
  result.addTypes({innerType});
  result.addAttribute("bufferType",
                      BufferTypeEnumAttr::get(result.getContext(), bufferType));
}

ParseResult BufferOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> allOperands;
  Type type;
  ArrayRef<Type> operandTypes(type);
  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  int size;
  if (sost::parseIntInSquareBrackets(parser, size))
    return failure();

  auto bufferTypeAttr = BufferTypeEnumAttr::parse(parser, {});
  if (!bufferTypeAttr)
    return failure();

  result.addAttribute("bufferType", bufferTypeAttr);

  if (parser.parseOperandList(allOperands) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(type))
    return failure();

  sost::addAttributes(result, size, type, false);
  result.addTypes({type});
  if (parser.resolveOperands(allOperands, operandTypes, allOperandLoc,
                             result.operands))
    return failure();
  return success();
}

void BufferOp::print(OpAsmPrinter &p) {
  int size =
      (*this)->getAttrOfType<IntegerAttr>("size").getValue().getZExtValue();
  p << " [" << size << "]";
  p << " " << stringifyEnum(bufferType());
  Type type = (*this)->getAttrOfType<TypeAttr>("dataType").getValue();
  p << " " << (*this)->getOperands();
  p.printOptionalAttrDict((*this)->getAttrs(),
                          {"size", "dataType", "control", "bufferType"});
  p << " : " << type;
}

static std::string getMemoryOperandName(unsigned nStores, unsigned idx) {
  std::string name;
  if (idx < nStores * 2) {
    bool isData = idx % 2 == 0;
    name = isData ? "stData" + std::to_string(idx / 2)
                  : "stAddr" + std::to_string(idx / 2);
  } else {
    idx -= 2 * nStores;
    name = "ldAddr" + std::to_string(idx);
  }
  return name;
}

std::string handshake::MemoryOp::getOperandName(unsigned int idx) {
  return getMemoryOperandName(stCount(), idx);
}

static std::string getMemoryResultName(unsigned nLoads, unsigned nStores,
                                       unsigned idx) {
  std::string name;
  if (idx < nLoads)
    name = "ldData" + std::to_string(idx);
  else if (idx < nLoads + nStores)
    name = "stDone" + std::to_string(idx - nLoads);
  else
    name = "ldDone" + std::to_string(idx - nLoads - nStores);
  return name;
}

std::string handshake::MemoryOp::getResultName(unsigned int idx) {
  return getMemoryResultName(ldCount(), stCount(), idx);
}

LogicalResult MemoryOp::verify() {
  auto memrefType = memRefType();

  if (memrefType.getNumDynamicDims() != 0)
    return emitOpError()
           << "memref dimensions for handshake.memory must be static.";
  if (memrefType.getShape().size() != 1)
    return emitOpError() << "memref must have only a single dimension.";

  unsigned opStCount = stCount();
  unsigned opLdCount = ldCount();
  int addressCount = memrefType.getShape().size();

  auto inputType = inputs().getType();
  auto outputType = outputs().getType();
  Type dataType = memrefType.getElementType();

  unsigned numOperands = static_cast<int>(inputs().size());
  unsigned numResults = static_cast<int>(outputs().size());
  if (numOperands != (1 + addressCount) * opStCount + addressCount * opLdCount)
    return emitOpError("number of operands ")
           << numOperands << " does not match number expected of "
           << 2 * opStCount + opLdCount << " with " << addressCount
           << " address inputs per port";

  if (numResults != opStCount + 2 * opLdCount)
    return emitOpError("number of results ")
           << numResults << " does not match number expected of "
           << opStCount + 2 * opLdCount << " with " << addressCount
           << " address inputs per port";

  Type addressType = opStCount > 0 ? inputType[1] : inputType[0];

  for (unsigned i = 0; i < opStCount; i++) {
    if (inputType[2 * i] != dataType)
      return emitOpError("data type for store port ")
             << i << ":" << inputType[2 * i] << " doesn't match memory type "
             << dataType;
    if (inputType[2 * i + 1] != addressType)
      return emitOpError("address type for store port ")
             << i << ":" << inputType[2 * i + 1]
             << " doesn't match address type " << addressType;
  }
  for (unsigned i = 0; i < opLdCount; i++) {
    Type ldAddressType = inputType[2 * opStCount + i];
    if (ldAddressType != addressType)
      return emitOpError("address type for load port ")
             << i << ":" << ldAddressType << " doesn't match address type "
             << addressType;
  }
  for (unsigned i = 0; i < opLdCount; i++) {
    if (outputType[i] != dataType)
      return emitOpError("data type for load port ")
             << i << ":" << outputType[i] << " doesn't match memory type "
             << dataType;
  }
  for (unsigned i = 0; i < opStCount; i++) {
    Type syncType = outputType[opLdCount + i];
    if (!syncType.isa<NoneType>())
      return emitOpError("data type for sync port for store port ")
             << i << ":" << syncType << " is not 'none'";
  }
  for (unsigned i = 0; i < opLdCount; i++) {
    Type syncType = outputType[opLdCount + opStCount + i];
    if (!syncType.isa<NoneType>())
      return emitOpError("data type for sync port for load port ")
             << i << ":" << syncType << " is not 'none'";
  }

  return success();
}

std::string handshake::ExternalMemoryOp::getOperandName(unsigned int idx) {
  if (idx == 0)
    return "extmem";

  return getMemoryOperandName(stCount(), idx - 1);
}

std::string handshake::ExternalMemoryOp::getResultName(unsigned int idx) {
  return getMemoryResultName(ldCount(), stCount(), idx);
}

void ExternalMemoryOp::build(OpBuilder &builder, OperationState &result,
                             Value memref, ValueRange inputs, int ldCount,
                             int stCount, int id) {
  SmallVector<Value> ops;
  ops.push_back(memref);
  llvm::append_range(ops, inputs);
  result.addOperands(ops);

  auto memrefType = memref.getType().cast<MemRefType>();

  // Data outputs (get their type from memref)
  result.types.append(ldCount, memrefType.getElementType());

  // Control outputs
  result.types.append(stCount + ldCount, builder.getNoneType());

  // Memory ID (individual ID for each MemoryOp)
  Type i32Type = builder.getIntegerType(32);
  result.addAttribute("id", builder.getIntegerAttr(i32Type, id));
  result.addAttribute("ldCount", builder.getIntegerAttr(i32Type, ldCount));
  result.addAttribute("stCount", builder.getIntegerAttr(i32Type, stCount));
}

void MemoryOp::build(OpBuilder &builder, OperationState &result,
                     ValueRange operands, int outputs, int controlOutputs,
                     bool lsq, int id, Value memref) {
  result.addOperands(operands);

  auto memrefType = memref.getType().cast<MemRefType>();

  // Data outputs (get their type from memref)
  result.types.append(outputs, memrefType.getElementType());

  // Control outputs
  result.types.append(controlOutputs, builder.getNoneType());
  result.addAttribute("lsq", builder.getBoolAttr(lsq));
  result.addAttribute("memRefType", TypeAttr::get(memrefType));

  // Memory ID (individual ID for each MemoryOp)
  Type i32Type = builder.getIntegerType(32);
  result.addAttribute("id", builder.getIntegerAttr(i32Type, id));

  if (!lsq) {
    result.addAttribute("ldCount", builder.getIntegerAttr(i32Type, outputs));
    result.addAttribute(
        "stCount", builder.getIntegerAttr(i32Type, controlOutputs - outputs));
  }
}

bool handshake::MemoryOp::allocateMemory(
    llvm::DenseMap<unsigned, unsigned> &memoryMap,
    std::vector<std::vector<llvm::Any>> &store,
    std::vector<double> &storeTimes) {
  if (memoryMap.count(id()))
    return false;

  auto type = memRefType();
  std::vector<llvm::Any> in;

  ArrayRef<int64_t> shape = type.getShape();
  int allocationSize = 1;
  unsigned count = 0;
  for (int64_t dim : shape) {
    if (dim > 0)
      allocationSize *= dim;
    else {
      assert(count < in.size());
      allocationSize *= llvm::any_cast<APInt>(in[count++]).getSExtValue();
    }
  }
  unsigned ptr = store.size();
  store.resize(ptr + 1);
  storeTimes.resize(ptr + 1);
  store[ptr].resize(allocationSize);
  storeTimes[ptr] = 0.0;
  mlir::Type elementType = type.getElementType();
  int width = elementType.getIntOrFloatBitWidth();
  for (int i = 0; i < allocationSize; i++) {
    if (elementType.isa<mlir::IntegerType>()) {
      store[ptr][i] = APInt(width, 0);
    } else if (elementType.isa<mlir::FloatType>()) {
      store[ptr][i] = APFloat(0.0);
    } else {
      llvm_unreachable("Unknown result type!\n");
    }
  }

  memoryMap[id()] = ptr;
  return true;
}

std::string handshake::LoadOp::getOperandName(unsigned int idx) {
  unsigned nAddresses = addresses().size();
  std::string opName;
  if (idx < nAddresses)
    opName = "addrIn" + std::to_string(idx);
  else if (idx == nAddresses)
    opName = "dataFromMem";
  else
    opName = "ctrl";
  return opName;
}

std::string handshake::LoadOp::getResultName(unsigned int idx) {
  std::string resName;
  if (idx == 0)
    resName = "dataOut";
  else
    resName = "addrOut" + std::to_string(idx - 1);
  return resName;
}

void handshake::LoadOp::build(OpBuilder &builder, OperationState &result,
                              Value memref, ValueRange indices) {
  // Address indices
  // result.addOperands(memref);
  result.addOperands(indices);

  // Data type
  auto memrefType = memref.getType().cast<MemRefType>();

  // Data output (from load to successor ops)
  result.types.push_back(memrefType.getElementType());

  // Address outputs (to lsq)
  result.types.append(indices.size(), builder.getIndexType());
}

static ParseResult parseMemoryAccessOp(OpAsmParser &parser,
                                       OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> addressOperands,
      remainingOperands, allOperands;
  SmallVector<Type, 1> parsedTypes, allTypes;
  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();

  if (parser.parseLSquare() || parser.parseOperandList(addressOperands) ||
      parser.parseRSquare() || parser.parseOperandList(remainingOperands) ||
      parser.parseColon() || parser.parseTypeList(parsedTypes))
    return failure();

  // The last type will be the data type of the operation; the prior will be the
  // address types.
  Type dataType = parsedTypes.back();
  auto parsedTypesRef = llvm::makeArrayRef(parsedTypes);
  result.addTypes(dataType);
  result.addTypes(parsedTypesRef.drop_back());
  allOperands.append(addressOperands);
  allOperands.append(remainingOperands);
  allTypes.append(parsedTypes);
  allTypes.push_back(NoneType::get(result.getContext()));
  if (parser.resolveOperands(allOperands, allTypes, allOperandLoc,
                             result.operands))
    return failure();
  return success();
}

template <typename MemOp>
static void printMemoryAccessOp(OpAsmPrinter &p, MemOp op) {
  p << " [";
  p << op.addresses();
  p << "] " << op.data() << ", " << op.ctrl() << " : ";
  llvm::interleaveComma(op.addresses(), p, [&](Value v) { p << v.getType(); });
  p << ", " << op.data().getType();
}

ParseResult LoadOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseMemoryAccessOp(parser, result);
}

void LoadOp::print(OpAsmPrinter &p) { printMemoryAccessOp(p, *this); }

std::string handshake::StoreOp::getOperandName(unsigned int idx) {
  unsigned nAddresses = addresses().size();
  std::string opName;
  if (idx < nAddresses)
    opName = "addrIn" + std::to_string(idx);
  else if (idx == nAddresses)
    opName = "dataIn";
  else
    opName = "ctrl";
  return opName;
}

template <typename TMemoryOp>
static LogicalResult verifyMemoryAccessOp(TMemoryOp op) {
  if (op.addresses().size() == 0)
    return op.emitOpError() << "No addresses were specified";

  return success();
}

LogicalResult LoadOp::verify() { return verifyMemoryAccessOp(*this); }

std::string handshake::StoreOp::getResultName(unsigned int idx) {
  std::string resName;
  if (idx == 0)
    resName = "dataToMem";
  else
    resName = "addrOut" + std::to_string(idx - 1);
  return resName;
}

void handshake::StoreOp::build(OpBuilder &builder, OperationState &result,
                               Value valueToStore, ValueRange indices) {

  // Address indices
  result.addOperands(indices);

  // Data
  result.addOperands(valueToStore);

  // Data output (from store to LSQ)
  result.types.push_back(valueToStore.getType());

  // Address outputs (from store to lsq)
  result.types.append(indices.size(), builder.getIndexType());
}

LogicalResult StoreOp::verify() { return verifyMemoryAccessOp(*this); }

ParseResult StoreOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseMemoryAccessOp(parser, result);
}

void StoreOp::print(OpAsmPrinter &p) { return printMemoryAccessOp(p, *this); }

void JoinOp::build(OpBuilder &builder, OperationState &result,
                   ValueRange operands) {
  auto type = builder.getNoneType();
  result.types.push_back(type);

  result.addOperands(operands);
  sost::addAttributes(result, operands.size(), type);
}

ParseResult JoinOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> allOperands;
  Type type;
  ArrayRef<Type> operandTypes(type);
  SmallVector<Type, 1> dataOperandsTypes;
  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  int size;
  if (sost::parseOperation(parser, allOperands, result, size, type, false))
    return failure();

  dataOperandsTypes.assign(size, type);
  result.addTypes({type});
  if (parser.resolveOperands(allOperands, dataOperandsTypes, allOperandLoc,
                             result.operands))
    return failure();
  return success();
}

void JoinOp::print(OpAsmPrinter &p) { sost::printOp(p, *this, false); }

/// Based on mlir::func::CallOp::verifySymbolUses
LogicalResult InstanceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the module attribute was specified.
  auto fnAttr = this->moduleAttr();
  assert(fnAttr && "requires a 'module' symbol reference attribute");

  FuncOp fn = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid handshake function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError(
        "incorrect number of operands for the referenced handshake function");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (getOperand(i).getType() != fnType.getInput(i))
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;

  if (fnType.getNumResults() != getNumResults())
    return emitOpError(
        "incorrect number of results for the referenced handshake function");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
    if (getResult(i).getType() != fnType.getResult(i))
      return emitOpError("result type mismatch: expected result type ")
             << fnType.getResult(i) << ", but provided "
             << getResult(i).getType() << " for result number " << i;

  return success();
}

LogicalResult InstanceOp::verify() {
  if ((*this)->getNumOperands() == 0)
    return emitOpError() << "must provide at least a control operand.";

  if (!getControl().getType().dyn_cast<NoneType>())
    return emitOpError()
           << "last operand must be a control (none-typed) operand.";

  return success();
}

FunctionType InstanceOp::getModuleType() {
  return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

ParseResult UnpackOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand tuple;
  TupleType type;

  if (parser.parseOperand(tuple) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(type))
    return failure();

  if (parser.resolveOperand(tuple, type, result.operands))
    return failure();

  result.addTypes(type.getTypes());

  return success();
}

void UnpackOp::print(OpAsmPrinter &p) {
  p << " " << input();
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << input().getType();
}

ParseResult PackOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  TupleType type;

  if (parser.parseOperandList(operands) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(type))
    return failure();

  if (parser.resolveOperands(operands, type.getTypes(), allOperandLoc,
                             result.operands))
    return failure();

  result.addTypes(type);

  return success();
}

void PackOp::print(OpAsmPrinter &p) {
  p << " " << inputs();
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << result().getType();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  auto *parent = (*this)->getParentOp();
  auto function = dyn_cast<handshake::FuncOp>(parent);
  if (!function)
    return emitOpError("must have a handshake.func parent");

  // The operand number and types must match the function signature.
  const auto &results = function.getResultTypes();
  if (getNumOperands() != results.size())
    return emitOpError("has ")
           << getNumOperands() << " operands, but enclosing function returns "
           << results.size();

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (getOperand(i).getType() != results[i])
      return emitError() << "type of return operand " << i << " ("
                         << getOperand(i).getType()
                         << ") doesn't match function result type ("
                         << results[i] << ")";

  return success();
}

#define GET_OP_CLASSES
#include "circt/Dialect/Handshake/Handshake.cpp.inc"
