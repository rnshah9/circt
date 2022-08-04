//===- SVModule.cpp - SV API pybind module --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DialectModules.h"

#include "circt-c/Dialect/SV.h"
#include "mlir-c/Bindings/Python/Interop.h"

#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

#include "llvm/ADT/SmallVector.h"

#include "PybindUtils.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

using namespace mlir::python::adaptors;

void circt::python::populateDialectSVSubmodule(py::module &m) {
  m.doc() = "SV Python Native Extension";

  mlir_attribute_subclass(m, "SVAttributeAttr", svAttrIsASVAttributeAttr)
      .def_classmethod(
          "get",
          [](py::object cls, std::string name, py::object expressionObj,
             MlirContext ctxt) {
            MlirStringRef expression = {nullptr, 0};
            if (!expressionObj.is_none())
              expression = mlirStringRefCreateFromCString(
                  expressionObj.cast<std::string>().c_str());
            return cls(svSVAttributeAttrGet(
                ctxt, mlirStringRefCreateFromCString(name.c_str()),
                expression));
          },
          "Create a SystemVerilog attribute", py::arg(), py::arg("name"),
          py::arg("expression") = py::none(), py::arg("ctxt") = py::none())
      .def_property_readonly("name",
                             [](MlirAttribute self) {
                               MlirStringRef name =
                                   svSVAttributeAttrGetName(self);
                               return std::string(name.data, name.length);
                             })
      .def_property_readonly(
          "expression", [](MlirAttribute self) -> py::object {
            MlirStringRef name = svSVAttributeAttrGetExpression(self);
            if (name.data == nullptr)
              return py::none();
            return py::str(std::string(name.data, name.length));
          });

  mlir_attribute_subclass(m, "SVAttributesAttr", svAttrIsASVAttributesAttr)
      .def_classmethod(
          "get",
          [](py::object cls, MlirAttribute attributes, bool emitAsComments,
             MlirContext ctxt) {
            return cls(svSVAttributesAttrGet(ctxt, attributes, emitAsComments));
          },
          "Create SV attributes attr", py::arg(), py::arg("attributes"),
          py::arg("emit_as_comments") = py::none(),
          py::arg("ctxt") = py::none())
      .def_property_readonly("attributes",
                             [](MlirAttribute self) {
                               return svSVAttributesAttrGetAttributes(self);
                             })
      .def_property_readonly("emit_as_comments", [](MlirAttribute self) {
        return svSVAttributesAttrGetEmitAsComments(self);
      });
}
