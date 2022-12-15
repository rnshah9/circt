//===- FlattenIO.cpp - HW I/O flattening pass -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;

static bool isStructType(Type type) {
  return hw::getCanonicalType(type).isa<hw::StructType>();
}

static hw::StructType getStructType(Type type) {
  return hw::getCanonicalType(type).dyn_cast<hw::StructType>();
}

// Legal if no in- or output type is a struct.
static bool isLegalFuncLikeOp(FunctionOpInterface moduleLikeOp) {
  bool legalResults =
      llvm::none_of(moduleLikeOp.getResultTypes(), isStructType);
  bool legalArgs = llvm::none_of(moduleLikeOp.getArgumentTypes(), isStructType);

  return legalResults && legalArgs;
}

static llvm::SmallVector<Type> getInnerTypes(hw::StructType t) {
  llvm::SmallVector<Type> inner;
  t.getInnerTypes(inner);
  for (auto [index, innerType] : llvm::enumerate(inner))
    inner[index] = hw::getCanonicalType(innerType);
  return inner;
}

namespace {

// Replaces an output op with a new output with flattened (exploded) structs.
struct OutputOpConversion : public OpConversionPattern<hw::OutputOp> {
  OutputOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                     DenseSet<Operation *> *opVisited)
      : OpConversionPattern(typeConverter, context), opVisited(opVisited) {}

  LogicalResult
  matchAndRewrite(hw::OutputOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Value> convOperands;

    // Flatten the operands.
    for (auto operand : adaptor.getOperands()) {
      if (auto structType = getStructType(operand.getType())) {
        auto explodedStruct = rewriter.create<hw::StructExplodeOp>(
            op.getLoc(), getInnerTypes(structType), operand);
        llvm::copy(explodedStruct.getResults(),
                   std::back_inserter(convOperands));
      } else
        convOperands.push_back(operand);
    }

    // And replace.
    rewriter.replaceOpWithNewOp<hw::OutputOp>(op, convOperands);
    opVisited->insert(op->getParentOp());
    return success();
  }
  mutable DenseSet<Operation *> *opVisited;
};

struct StructExplodeOpConversion
    : public OpConversionPattern<hw::StructExplodeOp> {
  StructExplodeOpConversion(TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(hw::StructExplodeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands());
    return success();
  }
};

using IOTypes = std::pair<TypeRange, TypeRange>;

struct IOInfo {
  // A mapping between an arg/res index and the struct type of the given field.
  DenseMap<unsigned, hw::StructType> argStructs, resStructs;

  // Records of the original arg/res types.
  TypeRange argTypes, resTypes;
};

class FlattenIOTypeConverter : public TypeConverter {
public:
  FlattenIOTypeConverter() {
    addConversion([](Type type, SmallVectorImpl<Type> &results) {
      auto structType = getStructType(type);
      if (!structType)
        results.push_back(type);
      else {
        for (auto field : structType.getElements())
          results.push_back(field.type);
      }
      return success();
    });

    addTargetMaterialization([](OpBuilder &builder, hw::StructType type,
                                ValueRange inputs, Location loc) {
      auto result = builder.create<hw::StructCreateOp>(loc, type, inputs);
      return result.getResult();
    });

    addTargetMaterialization([](OpBuilder &builder, hw::TypeAliasType type,
                                ValueRange inputs, Location loc) {
      auto structType = getStructType(type);
      assert(structType && "expected struct type");
      auto result = builder.create<hw::StructCreateOp>(loc, structType, inputs);
      return result.getResult();
    });
  }
};

} // namespace

template <typename... TOp>
static void addSignatureConversion(DenseMap<Operation *, IOInfo> &ioMap,
                                   ConversionTarget &target,
                                   RewritePatternSet &patterns,
                                   FlattenIOTypeConverter &typeConverter) {
  (mlir::populateFunctionOpInterfaceTypeConversionPattern<TOp>(patterns,
                                                               typeConverter),
   ...);

  // Legality is defined by a module having been processed once. This is due to
  // that a pattern cannot be applied multiple times (a 'pattern was already
  // applied' error - a case that would occur for nested structs). Additionally,
  // if a pattern could be applied multiple times, this would complicate
  // updating arg/res names.

  // Instead, we define legality as when a module has had a modification to its
  // top-level i/o. This ensures that only a single level of structs are
  // processed during signature conversion, which then allows us to use the
  // signature conversion in a recursive manner.
  target.addDynamicallyLegalOp<TOp...>([&](FunctionOpInterface moduleLikeOp) {
    if (isLegalFuncLikeOp(moduleLikeOp))
      return true;

    // This op is involved in conversion. Check if the signature has changed.
    auto ioInfoIt = ioMap.find(moduleLikeOp);
    if (ioInfoIt == ioMap.end()) {
      // Op wasn't primed in the map. Do the safe thing, assume
      // that it's not considered in this pass, and mark it as legal
      return true;
    }
    auto ioInfo = ioInfoIt->second;

    auto compareTypes = [&](TypeRange oldTypes, TypeRange newTypes) {
      return llvm::any_of(llvm::zip(oldTypes, newTypes), [&](auto typePair) {
        auto oldType = std::get<0>(typePair);
        auto newType = std::get<1>(typePair);
        return oldType != newType;
      });
    };
    if (compareTypes(moduleLikeOp.getResultTypes(), ioInfo.resTypes) ||
        compareTypes(moduleLikeOp.getArgumentTypes(), ioInfo.argTypes))
      return true;

    // We're pre-conversion for an op that was primed in the map - it will
    // always be illegal since it has to-be-converted struct types at its I/O.
    return false;
  });
}

template <typename T>
static bool hasUnconvertedOps(mlir::ModuleOp module) {
  return llvm::any_of(module.getBody()->getOps<T>(),
                      [](T op) { return !isLegalFuncLikeOp(op); });
}

template <typename T>
static DenseMap<Operation *, IOTypes> populateIOMap(mlir::ModuleOp module) {
  DenseMap<Operation *, IOTypes> ioMap;
  for (auto op : module.getOps<T>())
    ioMap[op] = {op.getArgumentTypes(), op.getResultTypes()};
  return ioMap;
}

static void updateNameAttribute(FunctionOpInterface op, StringRef attrName,
                                DenseMap<unsigned, hw::StructType> &structMap) {
  llvm::SmallVector<Attribute> newNames;
  auto oldNames = op.getOperation()
                      ->getAttrOfType<ArrayAttr>(attrName)
                      .getAsValueRange<StringAttr>();
  for (auto [i, oldName] : llvm::enumerate(oldNames)) {
    // Was this arg/res index a struct?
    auto it = structMap.find(i);
    if (it == structMap.end()) {
      // No, keep old name.
      newNames.push_back(StringAttr::get(op.getContext(), oldName));
      continue;
    }

    // Yes - create new names from the struct fields and the old name at the
    // index.
    auto structType = it->second;
    for (auto field : structType.getElements())
      newNames.push_back(
          StringAttr::get(op.getContext(), oldName + "." + field.name.str()));
  }
  op.getOperation()->setAttr(attrName,
                             ArrayAttr::get(op.getContext(), newNames));
}

template <typename T>
static DenseMap<Operation *, IOInfo> populateIOInfoMap(mlir::ModuleOp module) {
  DenseMap<Operation *, IOInfo> ioInfoMap;
  for (auto op : module.getOps<T>()) {
    IOInfo ioInfo;
    ioInfo.argTypes = op.getArgumentTypes();
    ioInfo.resTypes = op.getResultTypes();
    for (auto [i, arg] : llvm::enumerate(ioInfo.argTypes)) {
      if (auto structType = getStructType(arg))
        ioInfo.argStructs[i] = structType;
    }
    for (auto [i, res] : llvm::enumerate(ioInfo.resTypes)) {
      if (auto structType = getStructType(res))
        ioInfo.resStructs[i] = structType;
    }
    ioInfoMap[op] = ioInfo;
  }
  return ioInfoMap;
}

template <typename T>
static LogicalResult flattenOpsOfType(ModuleOp module, bool recursive) {
  auto *ctx = module.getContext();
  FlattenIOTypeConverter typeConverter;

  // Recursively (in case of nested structs) lower the module. We do this one
  // conversion at a time to allow for updating the arg/res names of the
  // module in between flattening each level of structs.
  while (hasUnconvertedOps<T>(module)) {
    ConversionTarget target(*ctx);
    RewritePatternSet patterns(ctx);
    target.addLegalDialect<hw::HWDialect>();

    // Record any struct types at the module signature. This will be used
    // post-conversion to update the argument and result names.
    auto ioInfoMap = populateIOInfoMap<T>(module);

    // Signature conversion and legalization patterns.
    addSignatureConversion<T>(ioInfoMap, target, patterns, typeConverter);

    // Argument conversion for output ops. Similarly to the signature
    // conversion, legality is based on the op having been visited once, due to
    // the possibility of nested structs.
    DenseSet<Operation *> opVisited;
    patterns.add<OutputOpConversion>(typeConverter, ctx, &opVisited);
    // patterns.add<StructExplodeOpConversion>(typeConverter, ctx);
    target.addDynamicallyLegalOp<hw::OutputOp>(
        [&](auto op) { return opVisited.contains(op->getParentOp()); });

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      return failure();

    // Update the arg/res names of the module.
    for (auto op : module.getOps<T>()) {
      auto ioInfo = ioInfoMap[op];
      updateNameAttribute(op, "argNames", ioInfo.argStructs);
      updateNameAttribute(op, "resultNames", ioInfo.resStructs);
    }

    // Break if we've only lowering a single level of structs.
    if (!recursive)
      break;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

template <typename... TOps>
static bool flattenIO(ModuleOp module, bool recursive) {
  return (failed(flattenOpsOfType<TOps>(module, recursive)) || ...);
}

namespace {

class FlattenIOPass : public circt::hw::FlattenIOBase<FlattenIOPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (flattenIO<hw::HWModuleOp, hw::HWModuleExternOp,
                  hw::HWModuleGeneratedOp>(module, recursive))
      signalPassFailure();
  };
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass initialization
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> circt::hw::createFlattenIOPass() {
  return std::make_unique<FlattenIOPass>();
}
