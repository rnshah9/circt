//===- MSFTOps.cpp - Implement MSFT dialect operations --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the MSFT dialect operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/MSFT/MSFTOps.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/ModuleImplementation.h"
#include "circt/Dialect/MSFT/MSFTAttributes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace msft;

//===----------------------------------------------------------------------===//
// Misc helper functions
//===----------------------------------------------------------------------===//

static bool hasAttribute(StringRef name, ArrayRef<NamedAttribute> attrs) {
  for (auto &argAttr : attrs)
    if (argAttr.getName() == name)
      return true;
  return false;
}

// Copied nearly exactly from hwops.cpp.
// TODO: Unify code once a `ModuleLike` op interface exists.
static void buildModule(OpBuilder &builder, OperationState &result,
                        StringAttr name, const hw::ModulePortInfo &ports) {
  using namespace mlir::function_interface_impl;

  // Add an attribute for the name.
  result.addAttribute(SymbolTable::getSymbolAttrName(), name);

  SmallVector<Attribute> argNames, resultNames;
  SmallVector<Type, 4> argTypes, resultTypes;
  SmallVector<Attribute> argAttrs, resultAttrs;
  auto exportPortIdent = StringAttr::get(builder.getContext(), "hw.exportPort");

  for (auto elt : ports.inputs) {
    if (elt.direction == hw::PortDirection::INOUT &&
        !elt.type.isa<hw::InOutType>())
      elt.type = hw::InOutType::get(elt.type);
    argTypes.push_back(elt.type);
    argNames.push_back(elt.name);
    Attribute attr;
    if (elt.sym && !elt.sym.empty())
      attr = builder.getDictionaryAttr({{exportPortIdent, elt.sym}});
    else
      attr = builder.getDictionaryAttr({});
    argAttrs.push_back(attr);
  }

  for (auto elt : ports.outputs) {
    resultTypes.push_back(elt.type);
    resultNames.push_back(elt.name);
    Attribute attr;
    if (elt.sym && !elt.sym.empty())
      attr = builder.getDictionaryAttr({{exportPortIdent, elt.sym}});
    else
      attr = builder.getDictionaryAttr({});
    resultAttrs.push_back(attr);
  }

  // Record the argument and result types as an attribute.
  auto type = builder.getFunctionType(argTypes, resultTypes);
  result.addAttribute(MSFTModuleOp::getFunctionTypeAttrName(result.name),
                      TypeAttr::get(type));
  result.addAttribute("argNames", builder.getArrayAttr(argNames));
  result.addAttribute("resultNames", builder.getArrayAttr(resultNames));
  result.addAttribute("parameters", builder.getDictionaryAttr({}));
  result.addAttribute(MSFTModuleOp::getArgAttrsAttrName(result.name),
                      builder.getArrayAttr(argAttrs));
  result.addAttribute(MSFTModuleOp::getResAttrsAttrName(result.name),
                      builder.getArrayAttr(resultAttrs));
  result.addRegion();
}

//===----------------------------------------------------------------------===//
// Custom directive parsers/printers
//===----------------------------------------------------------------------===//

static ParseResult parsePhysLoc(OpAsmParser &p, PhysLocationAttr &attr) {
  llvm::SMLoc loc = p.getCurrentLocation();
  StringRef devTypeStr;
  uint64_t x, y, num;

  if (p.parseKeyword(&devTypeStr) || p.parseKeyword("x") || p.parseColon() ||
      p.parseInteger(x) || p.parseKeyword("y") || p.parseColon() ||
      p.parseInteger(y) || p.parseKeyword("n") || p.parseColon() ||
      p.parseInteger(num))
    return failure();

  Optional<PrimitiveType> devType = symbolizePrimitiveType(devTypeStr);
  if (!devType) {
    p.emitError(loc, "Unknown device type '" + devTypeStr + "'");
    return failure();
  }
  PrimitiveTypeAttr devTypeAttr =
      PrimitiveTypeAttr::get(p.getContext(), *devType);
  attr = PhysLocationAttr::get(p.getContext(), devTypeAttr, x, y, num);
  return success();
}

static void printPhysLoc(OpAsmPrinter &p, Operation *, PhysLocationAttr loc) {
  p << stringifyPrimitiveType(loc.getPrimitiveType().getValue())
    << " x: " << loc.getX() << " y: " << loc.getY() << " n: " << loc.getNum();
}

static ParseResult parseListOptionalRegLocList(OpAsmParser &p,
                                               LocationVectorAttr &locs) {
  SmallVector<PhysLocationAttr, 32> locArr;
  TypeAttr type;
  if (p.parseAttribute(type) || p.parseLSquare() ||
      p.parseCommaSeparatedList(
          [&]() { return parseOptionalRegLoc(locArr, p); }) ||
      p.parseRSquare())
    return failure();

  if (failed(LocationVectorAttr::verify(
          [&p]() { return p.emitError(p.getNameLoc()); }, type, locArr)))
    return failure();
  locs = LocationVectorAttr::get(p.getContext(), type, locArr);
  return success();
}

static void printListOptionalRegLocList(OpAsmPrinter &p, Operation *,
                                        LocationVectorAttr locs) {
  p << locs.getType() << " [";
  llvm::interleaveComma(locs.getLocs(), p, [&p](PhysLocationAttr loc) {
    printOptionalRegLoc(loc, p);
  });
  p << "]";
}

static ParseResult parseImplicitInnerRef(OpAsmParser &p,
                                         hw::InnerRefAttr &innerRef) {
  SymbolRefAttr sym;
  if (p.parseAttribute(sym))
    return failure();
  auto loc = p.getCurrentLocation();
  if (sym.getNestedReferences().size() != 1)
    return p.emitError(loc, "expected <module sym>::<inner name>");
  innerRef = hw::InnerRefAttr::get(
      sym.getRootReference(),
      sym.getNestedReferences().front().getRootReference());
  return success();
}
void printImplicitInnerRef(OpAsmPrinter &p, Operation *,
                           hw::InnerRefAttr innerRef) {
  p << SymbolRefAttr::get(innerRef.getModule(),
                          {FlatSymbolRefAttr::get(innerRef.getName())});
}

/// Parse an parameter list if present. Same format as HW dialect.
/// module-parameter-list ::= `<` parameter-decl (`,` parameter-decl)* `>`
/// parameter-decl ::= identifier `:` type
/// parameter-decl ::= identifier `:` type `=` attribute
///
static ParseResult parseParameterList(OpAsmParser &parser,
                                      SmallVector<Attribute> &parameters) {

  return parser.parseCommaSeparatedList(
      OpAsmParser::Delimiter::OptionalLessGreater, [&]() {
        std::string name;
        Type type;
        Attribute value;

        if (parser.parseKeywordOrString(&name) || parser.parseColonType(type))
          return failure();

        // Parse the default value if present.
        if (succeeded(parser.parseOptionalEqual())) {
          if (parser.parseAttribute(value, type))
            return failure();
        }

        auto &builder = parser.getBuilder();
        parameters.push_back(hw::ParamDeclAttr::get(
            builder.getContext(), builder.getStringAttr(name), type, value));
        return success();
      });
}

/// Shim to also use this for the InstanceOp custom parser.
static ParseResult parseParameterList(OpAsmParser &parser,
                                      ArrayAttr &parameters) {
  SmallVector<Attribute> parseParameters;
  if (failed(parseParameterList(parser, parseParameters)))
    return failure();

  parameters = ArrayAttr::get(parser.getContext(), parseParameters);

  return success();
}

/// Print a parameter list for a module or instance. Same format as HW dialect.
static void printParameterList(OpAsmPrinter &p, Operation *op,
                               ArrayAttr parameters) {
  if (!parameters || parameters.empty())
    return;

  p << '<';
  llvm::interleaveComma(parameters, p, [&](Attribute param) {
    auto paramAttr = param.cast<hw::ParamDeclAttr>();
    p << paramAttr.getName().getValue() << ": " << paramAttr.getType();
    if (auto value = paramAttr.getValue()) {
      p << " = ";
      p.printAttributeWithoutType(value);
    }
  });
  p << '>';
}

static ParseResult parseModuleLikeOp(OpAsmParser &parser,
                                     OperationState &result,
                                     bool withParameters = false) {
  using namespace mlir::function_interface_impl;

  auto loc = parser.getCurrentLocation();

  SmallVector<OpAsmParser::Argument, 4> entryArgs;
  SmallVector<DictionaryAttr, 4> resultAttrs;
  SmallVector<Type, 4> resultTypes;
  auto &builder = parser.getBuilder();

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  if (withParameters) {
    // Parse the parameters
    DictionaryAttr paramsAttr;
    if (parser.parseAttribute(paramsAttr))
      return failure();
    result.addAttribute("parameters", paramsAttr);
  }

  // Parse the function signature.
  bool isVariadic = false;
  SmallVector<Attribute> resultNames;
  if (hw::module_like_impl::parseModuleFunctionSignature(
          parser, entryArgs, isVariadic, resultTypes, resultAttrs, resultNames))
    return failure();

  // Record the argument and result types as an attribute.  This is necessary
  // for external modules.
  SmallVector<Type> argTypes;
  for (auto arg : entryArgs)
    argTypes.push_back(arg.type);
  auto type = builder.getFunctionType(argTypes, resultTypes);
  result.addAttribute(MSFTModuleOp::getFunctionTypeAttrName(result.name),
                      TypeAttr::get(type));

  // If function attributes are present, parse them.
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  auto *context = result.getContext();

  if (hasAttribute("argNames", result.attributes) ||
      hasAttribute("resultNames", result.attributes)) {
    parser.emitError(
        loc, "explicit argNames and resultNames attributes not allowed");
    return failure();
  }

  // Use the argument and result names if not already specified.
  SmallVector<Attribute> argNames;
  if (!entryArgs.empty()) {
    for (auto &arg : entryArgs)
      argNames.push_back(
          hw::module_like_impl::getPortNameAttr(context, arg.ssaName.name));
  }

  result.addAttribute("argNames", ArrayAttr::get(context, argNames));
  result.addAttribute("resultNames", ArrayAttr::get(context, resultNames));

  assert(resultAttrs.size() == resultTypes.size());

  // Add the attributes to the module arguments.
  addArgAndResultAttrs(builder, result, entryArgs, resultAttrs,
                       MSFTModuleOp::getArgAttrsAttrName(result.name),
                       MSFTModuleOp::getResAttrsAttrName(result.name));

  // Parse the optional module body.
  auto regionSuccess =
      parser.parseOptionalRegion(*result.addRegion(), entryArgs);
  if (regionSuccess.has_value() && failed(*regionSuccess))
    return failure();

  return success();
}

template <typename ModuleTy>
static void printModuleLikeOp(mlir::FunctionOpInterface moduleLike,
                              OpAsmPrinter &p, Attribute parameters = nullptr) {
  using namespace mlir::function_interface_impl;

  auto argTypes = moduleLike.getArgumentTypes();
  auto resultTypes = moduleLike.getResultTypes();

  // Print the operation and the function name.
  p << ' ';
  p.printSymbolName(SymbolTable::getSymbolName(moduleLike).getValue());

  if (parameters) {
    // Print the parameterization.
    p << ' ';
    p.printAttribute(parameters);
  }

  p << ' ';
  bool needArgNamesAttr = false;
  hw::module_like_impl::printModuleSignature(p, moduleLike, argTypes,
                                             /*isVariadic=*/false, resultTypes,
                                             needArgNamesAttr);

  SmallVector<StringRef, 3> omittedAttrs;
  if (!needArgNamesAttr)
    omittedAttrs.push_back("argNames");
  omittedAttrs.push_back("resultNames");
  omittedAttrs.push_back("parameters");
  omittedAttrs.push_back(
      ModuleTy::getFunctionTypeAttrName(moduleLike->getName()));
  omittedAttrs.push_back(ModuleTy::getArgAttrsAttrName(moduleLike->getName()));
  omittedAttrs.push_back(ModuleTy::getResAttrsAttrName(moduleLike->getName()));

  printFunctionAttributes(p, moduleLike, omittedAttrs);

  // Print the body if this is not an external function.
  Region &mbody = moduleLike.getFunctionBody();
  if (!mbody.empty()) {
    p << ' ';
    p.printRegion(mbody, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
}

//===----------------------------------------------------------------------===//
// DynamicInstanceOp
//===----------------------------------------------------------------------===//

ArrayAttr DynamicInstanceOp::globalRefPath() {
  SmallVector<Attribute, 16> path;
  DynamicInstanceOp next = *this;
  do {
    path.push_back(next.getInstanceRefAttr());
    next = next->getParentOfType<DynamicInstanceOp>();
  } while (next);
  std::reverse(path.begin(), path.end());
  return ArrayAttr::get(getContext(), path);
}

//===----------------------------------------------------------------------===//
// InstanceOp
//===----------------------------------------------------------------------===//

LogicalResult InstanceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto *module =
      symbolTable.lookupNearestSymbolFrom(*this, getModuleNameAttr());
  if (module == nullptr)
    return emitError("Cannot find module definition '")
           << getModuleName() << "'";

  // It must be some sort of module.
  if (!hw::isAnyModule(module) &&
      !isa<MSFTModuleOp, MSFTModuleExternOp>(module))
    return emitError("symbol reference '")
           << getModuleName() << "' isn't a module";
  return success();
}

/// Lookup the module or extmodule for the symbol.  This returns null on
/// invalid IR.
Operation *InstanceOp::getReferencedModule() {
  auto topLevelModuleOp = (*this)->getParentOfType<ModuleOp>();
  if (!topLevelModuleOp)
    return nullptr;
  return topLevelModuleOp.lookupSymbol(getModuleName());
}

StringAttr InstanceOp::getResultName(size_t idx) {
  if (auto *refMod = getReferencedModule())
    return hw::getModuleResultNameAttr(refMod, idx);
  return StringAttr();
}

/// Suggest a name for each result value based on the saved result names
/// attribute.
void InstanceOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  // Provide default names for instance results.
  std::string name = instanceName().str() + ".";
  size_t baseNameLen = name.size();

  for (size_t i = 0, e = getNumResults(); i != e; ++i) {
    name.resize(baseNameLen);
    StringAttr resNameAttr = getResultName(i);
    if (resNameAttr)
      name += resNameAttr.getValue().str();
    else
      name += std::to_string(i);
    setNameFn(getResult(i), name);
  }
}

LogicalResult
InstanceOp::verifySignatureMatch(const hw::ModulePortInfo &ports) {
  if (ports.inputs.size() != getNumOperands())
    return emitOpError("wrong number of inputs (expected ")
           << ports.inputs.size() << ")";
  if (ports.outputs.size() != getNumResults())
    return emitOpError("wrong number of outputs (expected ")
           << ports.outputs.size() << ")";
  for (auto port : ports.inputs)
    if (getOperand(port.argNum).getType() != port.type)
      return emitOpError("in input port ")
             << port.name << ", expected type " << port.type << " got "
             << getOperand(port.argNum).getType();
  for (auto port : ports.outputs)
    if (getResult(port.argNum).getType() != port.type)
      return emitOpError("in output port ")
             << port.name << ", expected type " << port.type << " got "
             << getResult(port.argNum).getType();

  return success();
}

void InstanceOp::build(OpBuilder &builder, OperationState &state,
                       ArrayRef<Type> resultTypes, StringAttr sym_name,
                       FlatSymbolRefAttr moduleName, ArrayRef<Value> inputs) {
  build(builder, state, resultTypes, sym_name, moduleName, inputs, ArrayAttr(),
        SymbolRefAttr());
}

//===----------------------------------------------------------------------===//
// MSFTModuleOp
//===----------------------------------------------------------------------===//

/// Return an encapsulated set of information about input and output ports of
/// the specified module or instance.  The input ports always come before the
/// output ports in the list.
/// TODO: This should really be shared with the HW dialect instead of cloned.
/// Consider adding a `HasModulePorts` op interface to facilitate.
hw::ModulePortInfo MSFTModuleOp::getPorts() {
  SmallVector<hw::PortInfo> inputs, outputs;
  auto argTypes = getArgumentTypes();

  auto argNames = this->getArgNames();
  for (unsigned i = 0, e = argTypes.size(); i < e; ++i) {
    bool isInOut = false;
    auto type = argTypes[i];

    if (auto inout = type.dyn_cast<hw::InOutType>()) {
      isInOut = true;
      type = inout.getElementType();
    }

    auto direction =
        isInOut ? hw::PortDirection::INOUT : hw::PortDirection::INPUT;
    inputs.push_back({argNames[i].cast<StringAttr>(), direction, type, i});
  }

  auto resultNames = this->getResultNames();
  auto resultTypes = getResultTypes();
  for (unsigned i = 0, e = resultTypes.size(); i < e; ++i) {
    outputs.push_back({resultNames[i].cast<StringAttr>(),
                       hw::PortDirection::OUTPUT, resultTypes[i], i});
  }
  return hw::ModulePortInfo(inputs, outputs);
}

SmallVector<BlockArgument>
MSFTModuleOp::addPorts(ArrayRef<std::pair<StringAttr, Type>> inputs,
                       ArrayRef<std::pair<StringAttr, Value>> outputs) {
  auto *ctxt = getContext();
  Block *body = getBodyBlock();

  // Append new inputs.
  SmallVector<Type, 32> modifiedArgs(getArgumentTypes().begin(),
                                     getArgumentTypes().end());
  SmallVector<Attribute> modifiedArgNames(
      getArgNames().getAsRange<Attribute>());
  SmallVector<BlockArgument> newBlockArgs;
  for (auto ttPair : inputs) {
    modifiedArgNames.push_back(ttPair.first);
    modifiedArgs.push_back(ttPair.second);
    newBlockArgs.push_back(
        body->addArgument(ttPair.second, Builder(ctxt).getUnknownLoc()));
  }
  setArgNamesAttr(ArrayAttr::get(ctxt, modifiedArgNames));

  // Append new outputs.
  SmallVector<Type, 32> modifiedResults(getResultTypes().begin(),
                                        getResultTypes().end());
  SmallVector<Attribute> modifiedResultNames(
      getResultNames().getAsRange<Attribute>());
  Operation *terminator = body->getTerminator();
  SmallVector<Value, 32> modifiedOutputs(terminator->getOperands());
  for (auto tvPair : outputs) {
    modifiedResultNames.push_back(tvPair.first);
    modifiedResults.push_back(tvPair.second.getType());
    modifiedOutputs.push_back(tvPair.second);
  }
  setResultNamesAttr(ArrayAttr::get(ctxt, modifiedResultNames));
  terminator->setOperands(modifiedOutputs);

  // Finalize and return.
  setType(FunctionType::get(ctxt, modifiedArgs, modifiedResults));
  return newBlockArgs;
}

// Remove the ports at the specified indexes.
SmallVector<unsigned> MSFTModuleOp::removePorts(llvm::BitVector inputs,
                                                llvm::BitVector outputs) {
  MLIRContext *ctxt = getContext();
  FunctionType ftype = getFunctionType();
  Block *body = getBodyBlock();
  Operation *terminator = body->getTerminator();

  SmallVector<Type, 4> newInputTypes;
  SmallVector<Attribute, 4> newArgNames;
  unsigned originalNumArgs = ftype.getNumInputs();
  ArrayRef<Attribute> origArgNames = getArgNamesAttr().getValue();
  assert(origArgNames.size() == originalNumArgs);
  for (size_t i = 0; i < originalNumArgs; ++i) {
    if (!inputs.test(i)) {
      newInputTypes.emplace_back(ftype.getInput(i));
      newArgNames.emplace_back(origArgNames[i]);
    }
  }

  SmallVector<Type, 4> newResultTypes;
  SmallVector<Attribute, 4> newResultNames;
  unsigned originalNumResults = getNumResults();
  ArrayRef<Attribute> origResNames = getResultNamesAttr().getValue();
  assert(origResNames.size() == originalNumResults);
  for (size_t i = 0; i < originalNumResults; ++i) {
    if (!outputs.test(i)) {
      newResultTypes.emplace_back(ftype.getResult(i));
      newResultNames.emplace_back(origResNames[i]);
    }
  }

  setType(FunctionType::get(ctxt, newInputTypes, newResultTypes));
  setResultNamesAttr(ArrayAttr::get(ctxt, newResultNames));
  setArgNamesAttr(ArrayAttr::get(ctxt, newArgNames));

  // Build new operand list for output op. Construct an output mapping to
  // return as a side-effect.
  unsigned numResults = ftype.getNumResults();
  SmallVector<Value> newOutputValues;
  SmallVector<unsigned> newToOldResultMap;

  for (unsigned i = 0; i < numResults; ++i) {
    if (!outputs.test(i)) {
      newOutputValues.push_back(terminator->getOperand(i));
      newToOldResultMap.push_back(i);
    }
  }
  terminator->setOperands(newOutputValues);

  // Erase the arguments after setting the new output op operands since the
  // arguments might be used by output op.
  for (unsigned argNum = 0, e = body->getArguments().size(); argNum < e;
       ++argNum)
    if (inputs.test(argNum))
      body->getArgument(argNum).dropAllUses();
  body->eraseArguments(inputs);

  return newToOldResultMap;
}

void MSFTModuleOp::modifyPorts(
    llvm::ArrayRef<std::pair<unsigned int, circt::hw::PortInfo>> insertInputs,
    llvm::ArrayRef<std::pair<unsigned int, circt::hw::PortInfo>> insertOutputs,
    llvm::ArrayRef<unsigned int> eraseInputs,
    llvm::ArrayRef<unsigned int> eraseOutputs) {
  hw::modifyModulePorts(*this, insertInputs, insertOutputs, eraseInputs,
                        eraseOutputs, getBodyBlock());
}

void MSFTModuleOp::appendOutputs(
    ArrayRef<std::pair<StringAttr, Value>> outputs) {
  addPorts({}, outputs);
}

void MSFTModuleOp::build(OpBuilder &builder, OperationState &result,
                         StringAttr name, hw::ModulePortInfo ports,
                         ArrayRef<NamedAttribute> params) {
  buildModule(builder, result, name, ports);

  // Create a region and a block for the body.
  auto *bodyRegion = result.regions[0].get();
  Block *body = new Block();
  bodyRegion->push_back(body);

  // Add arguments to the body block.
  for (auto elt : ports.inputs)
    body->addArgument(elt.type, builder.getUnknownLoc());

  MSFTModuleOp::ensureTerminator(*bodyRegion, builder, result.location);
}

ParseResult MSFTModuleOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseModuleLikeOp(parser, result, /*withParameters=*/true);
}

void MSFTModuleOp::print(OpAsmPrinter &p) {
  printModuleLikeOp<MSFTModuleOp>(*this, p, getParametersAttr());
}

//===----------------------------------------------------------------------===//
// MSFTModuleExternOp
//===----------------------------------------------------------------------===//

/// Check parameter specified by `value` to see if it is valid within the scope
/// of the specified module `module`.  If not, emit an error at the location of
/// `usingOp` and return failure, otherwise return success.  If `usingOp` is
/// null, then no diagnostic is generated. Same format as HW dialect.
///
/// If `disallowParamRefs` is true, then parameter references are not allowed.
static LogicalResult checkParameterInContext(Attribute value, Operation *module,
                                             Operation *usingOp,
                                             bool disallowParamRefs) {
  // Literals are always ok.  Their types are already known to match
  // expectations.
  if (value.isa<IntegerAttr>() || value.isa<FloatAttr>() ||
      value.isa<StringAttr>() || value.isa<hw::ParamVerbatimAttr>())
    return success();

  // Check both subexpressions of an expression.
  if (auto expr = value.dyn_cast<hw::ParamExprAttr>()) {
    for (auto op : expr.getOperands())
      if (failed(
              checkParameterInContext(op, module, usingOp, disallowParamRefs)))
        return failure();
    return success();
  }

  // Parameter references need more analysis to make sure they are valid within
  // this module.
  if (auto parameterRef = value.dyn_cast<hw::ParamDeclRefAttr>()) {
    auto nameAttr = parameterRef.getName();

    // Don't allow references to parameters from the default values of a
    // parameter list.
    if (disallowParamRefs) {
      if (usingOp)
        usingOp->emitOpError("parameter ")
            << nameAttr << " cannot be used as a default value for a parameter";
      return failure();
    }

    // Find the corresponding attribute in the module.
    for (auto param : module->getAttrOfType<ArrayAttr>("parameters")) {
      auto paramAttr = param.cast<hw::ParamDeclAttr>();
      if (paramAttr.getName() != nameAttr)
        continue;

      // If the types match then the reference is ok.
      if (paramAttr.getType() == parameterRef.getType())
        return success();

      if (usingOp) {
        auto diag = usingOp->emitOpError("parameter ")
                    << nameAttr << " used with type " << parameterRef.getType()
                    << "; should have type " << paramAttr.getType();
        diag.attachNote(module->getLoc()) << "module declared here";
      }
      return failure();
    }

    if (usingOp) {
      auto diag = usingOp->emitOpError("use of unknown parameter ") << nameAttr;
      diag.attachNote(module->getLoc()) << "module declared here";
    }
    return failure();
  }

  if (usingOp)
    usingOp->emitOpError("invalid parameter value ") << value;
  return failure();
}

ParseResult MSFTModuleExternOp::parse(OpAsmParser &parser,
                                      OperationState &result) {
  using namespace mlir::function_interface_impl;

  auto loc = parser.getCurrentLocation();

  SmallVector<OpAsmParser::Argument, 4> entryArgs;
  SmallVector<DictionaryAttr, 4> resultAttrs;
  SmallVector<Type, 4> resultTypes;
  SmallVector<Attribute> parameters;
  auto &builder = parser.getBuilder();

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse the function signature.
  bool isVariadic = false;
  SmallVector<Attribute> resultNames;
  if (parseParameterList(parser, parameters) ||
      hw::module_like_impl::parseModuleFunctionSignature(
          parser, entryArgs, isVariadic, resultTypes, resultAttrs,
          resultNames) ||
      // If function attributes are present, parse them.
      parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Record the argument and result types as an attribute.  This is necessary
  // for external modules.
  SmallVector<Type> argTypes;
  for (auto arg : entryArgs)
    argTypes.push_back(arg.type);

  auto type = builder.getFunctionType(argTypes, resultTypes);
  result.addAttribute(MSFTModuleExternOp::getFunctionTypeAttrName(result.name),
                      TypeAttr::get(type));

  auto *context = result.getContext();

  if (hasAttribute("resultNames", result.attributes) ||
      hasAttribute("parameters", result.attributes)) {
    parser.emitError(
        loc, "explicit `resultNames` / `parameters` attributes not allowed");
    return failure();
  }

  // Use the argument and result names if not already specified.
  SmallVector<Attribute> argNames;
  if (!entryArgs.empty()) {
    for (auto &arg : entryArgs)
      argNames.push_back(
          hw::module_like_impl::getPortNameAttr(context, arg.ssaName.name));
  }

  // An explicit `argNames` attribute overrides the MLIR names.  This is how
  // we represent port names that aren't valid MLIR identifiers.  Result and
  // parameter names are printed quoted when they aren't valid identifiers, so
  // they don't need this affordance.
  if (!hasAttribute("argNames", result.attributes))
    result.addAttribute("argNames", ArrayAttr::get(context, argNames));
  result.addAttribute("resultNames", ArrayAttr::get(context, resultNames));
  result.addAttribute("parameters", ArrayAttr::get(context, parameters));

  assert(resultAttrs.size() == resultTypes.size());

  // Add the attributes to the function arguments.
  addArgAndResultAttrs(builder, result, entryArgs, resultAttrs,
                       MSFTModuleExternOp::getArgAttrsAttrName(result.name),
                       MSFTModuleExternOp::getResAttrsAttrName(result.name));

  // Extern modules carry an empty region to work with HWModuleImplementation.h.
  result.addRegion();

  return success();
}

void MSFTModuleExternOp::print(OpAsmPrinter &p) {
  using namespace mlir::function_interface_impl;

  auto typeAttr = (*this)->getAttrOfType<TypeAttr>(getFunctionTypeAttrName());
  FunctionType fnType = typeAttr.getValue().cast<FunctionType>();
  auto argTypes = fnType.getInputs();
  auto resultTypes = fnType.getResults();

  // Print the operation and the function name.
  p << ' ';
  p.printSymbolName(SymbolTable::getSymbolName(*this).getValue());

  // Print the parameter list if present.
  printParameterList(p, *this, (*this)->getAttrOfType<ArrayAttr>("parameters"));

  bool needArgNamesAttr = false;
  hw::module_like_impl::printModuleSignature(
      p, *this, argTypes, /*isVariadic=*/false, resultTypes, needArgNamesAttr);

  SmallVector<StringRef, 3> omittedAttrs;
  if (!needArgNamesAttr)
    omittedAttrs.push_back("argNames");
  omittedAttrs.push_back("resultNames");
  omittedAttrs.push_back("parameters");
  omittedAttrs.push_back(getFunctionTypeAttrName());
  omittedAttrs.push_back(getArgAttrsAttrName());
  omittedAttrs.push_back(getResAttrsAttrName());

  printFunctionAttributes(p, *this, omittedAttrs);
}

LogicalResult MSFTModuleExternOp::verify() {
  using namespace mlir::function_interface_impl;
  auto typeAttr = (*this)->getAttrOfType<TypeAttr>(getFunctionTypeAttrName());
  auto moduleType = typeAttr.getValue().cast<FunctionType>();
  auto argNames = (*this)->getAttrOfType<ArrayAttr>("argNames");
  auto resultNames = (*this)->getAttrOfType<ArrayAttr>("resultNames");
  if (argNames.size() != moduleType.getNumInputs())
    return emitOpError("incorrect number of argument names");
  if (resultNames.size() != moduleType.getNumResults())
    return emitOpError("incorrect number of result names");

  SmallPtrSet<Attribute, 4> paramNames;

  // Check parameter default values are sensible.
  for (auto param : (*this)->getAttrOfType<ArrayAttr>("parameters")) {
    auto paramAttr = param.cast<hw::ParamDeclAttr>();

    // Check that we don't have any redundant parameter names.  These are
    // resolved by string name: reuse of the same name would cause ambiguities.
    if (!paramNames.insert(paramAttr.getName()).second)
      return emitOpError("parameter ")
             << paramAttr << " has the same name as a previous parameter";

    // Default values are allowed to be missing, check them if present.
    auto value = paramAttr.getValue();
    if (!value)
      continue;

    auto typedValue = value.dyn_cast<mlir::TypedAttr>();
    if (!typedValue)
      return emitOpError("parameter ")
             << paramAttr << " should have a typed value; has value " << value;

    if (typedValue.getType() != paramAttr.getType())
      return emitOpError("parameter ")
             << paramAttr << " should have type " << paramAttr.getType()
             << "; has type " << typedValue.getType();

    // Verify that this is a valid parameter value, disallowing parameter
    // references.  We could allow parameters to refer to each other in the
    // future with lexical ordering if there is a need.
    if (failed(checkParameterInContext(value, *this, *this,
                                       /*disallowParamRefs=*/true)))
      return failure();
  }
  return success();
}

hw::ModulePortInfo MSFTModuleExternOp::getPorts() {
  using namespace mlir::function_interface_impl;

  SmallVector<hw::PortInfo> inputs, outputs;

  auto typeAttr =
      getOperation()->getAttrOfType<TypeAttr>(getFunctionTypeAttrName());
  auto moduleType = typeAttr.getValue().cast<FunctionType>();
  auto argTypes = moduleType.getInputs();
  auto resultTypes = moduleType.getResults();

  auto argNames = getOperation()->getAttrOfType<ArrayAttr>("argNames");
  for (unsigned i = 0, e = argTypes.size(); i < e; ++i) {
    bool isInOut = false;
    auto type = argTypes[i];

    if (auto inout = type.dyn_cast<hw::InOutType>()) {
      isInOut = true;
      type = inout.getElementType();
    }

    auto direction =
        isInOut ? hw::PortDirection::INOUT : hw::PortDirection::INPUT;

    inputs.push_back({argNames[i].cast<StringAttr>(), direction, type, i});
  }

  auto resultNames = getOperation()->getAttrOfType<ArrayAttr>("resultNames");
  for (unsigned i = 0, e = resultTypes.size(); i < e; ++i)
    outputs.push_back({resultNames[i].cast<StringAttr>(),
                       hw::PortDirection::OUTPUT, resultTypes[i], i});

  return hw::ModulePortInfo(inputs, outputs);
}

//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

void OutputOp::build(OpBuilder &builder, OperationState &result) {}

//===----------------------------------------------------------------------===//
// MSFT high level design constructs
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// SystolicArrayOp
//===----------------------------------------------------------------------===//

ParseResult SystolicArrayOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  uint64_t numRows, numColumns;
  Type rowType, columnType;
  OpAsmParser::UnresolvedOperand rowInputs, columnInputs;
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseLSquare() || parser.parseOperand(rowInputs) ||
      parser.parseColon() || parser.parseInteger(numRows) ||
      parser.parseKeyword("x") || parser.parseType(rowType) ||
      parser.parseRSquare() || parser.parseLSquare() ||
      parser.parseOperand(columnInputs) || parser.parseColon() ||
      parser.parseInteger(numColumns) || parser.parseKeyword("x") ||
      parser.parseType(columnType) || parser.parseRSquare())
    return failure();

  hw::ArrayType rowInputType = hw::ArrayType::get(rowType, numRows);
  hw::ArrayType columnInputType = hw::ArrayType::get(columnType, numColumns);
  SmallVector<Value> operands;
  if (parser.resolveOperands({rowInputs, columnInputs},
                             {rowInputType, columnInputType}, loc, operands))
    return failure();
  result.addOperands(operands);

  Type peOutputType;
  SmallVector<OpAsmParser::Argument> peArgs;
  if (parser.parseKeyword("pe")) {
    return failure();
  }
  llvm::SMLoc peLoc = parser.getCurrentLocation();
  if (parser.parseArgumentList(peArgs, AsmParser::Delimiter::Paren)) {
    return failure();
  }
  if (peArgs.size() != 2) {
    return parser.emitError(peLoc, "expected two operands");
  }

  peArgs[0].type = rowType;
  peArgs[1].type = columnType;

  if (parser.parseArrow() || parser.parseLParen() ||
      parser.parseType(peOutputType) || parser.parseRParen())
    return failure();

  result.addTypes({hw::ArrayType::get(
      hw::ArrayType::get(peOutputType, numColumns), numRows)});

  Region *pe = result.addRegion();

  peLoc = parser.getCurrentLocation();

  if (parser.parseRegion(*pe, peArgs))
    return failure();

  if (pe->getBlocks().size() != 1)
    return parser.emitError(peLoc, "expected one block for the PE");
  Operation *peTerm = pe->getBlocks().front().getTerminator();
  if (peTerm->getOperands().size() != 1)
    return peTerm->emitOpError("expected one return value");
  if (peTerm->getOperand(0).getType() != peOutputType)
    return peTerm->emitOpError("expected return type as given in parent: ")
           << peOutputType;

  return success();
}

void SystolicArrayOp::print(OpAsmPrinter &p) {
  hw::ArrayType rowInputType = getRowInputs().getType().cast<hw::ArrayType>();
  hw::ArrayType columnInputType =
      getColInputs().getType().cast<hw::ArrayType>();
  p << " [";
  p.printOperand(getRowInputs());
  p << " : " << rowInputType.getSize() << " x ";
  p.printType(rowInputType.getElementType());
  p << "] [";
  p.printOperand(getColInputs());
  p << " : " << columnInputType.getSize() << " x ";
  p.printType(columnInputType.getElementType());

  p << "] pe (";
  p.printOperand(getPe().getArgument(0));
  p << ", ";
  p.printOperand(getPe().getArgument(1));
  p << ") -> (";
  p.printType(getPeOutputs()
                  .getType()
                  .cast<hw::ArrayType>()
                  .getElementType()
                  .cast<hw::ArrayType>()
                  .getElementType());
  p << ") ";
  p.printRegion(getPe(), false);
}

//===----------------------------------------------------------------------===//
// LinearOp
//===----------------------------------------------------------------------===//

LogicalResult LinearOp::verify() {

  for (auto &op : *getBodyBlock()) {
    if (!isa<hw::HWDialect, comb::CombDialect, msft::MSFTDialect>(
            op.getDialect()))
      return emitOpError() << "expected only hw, comb, and msft dialect ops "
                              "inside the datapath.";
  }

  return success();
}

#define GET_OP_CLASSES
#include "circt/Dialect/MSFT/MSFT.cpp.inc"
