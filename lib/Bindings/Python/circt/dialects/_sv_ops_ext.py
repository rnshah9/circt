#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from mlir.ir import ArrayAttr, Attribute, FlatSymbolRefAttr, OpView, StringAttr
from circt.dialects import sv, hw
import circt.support as support


class IfDefOp:

  def __init__(self, cond: Attribute, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {"cond": cond}
    regions = 2
    super().__init__(
        self.build_generic(attributes=attributes,
                           results=results,
                           operands=operands,
                           successors=None,
                           regions=regions,
                           loc=loc,
                           ip=ip))
    self.regions[0].blocks.append()
    self.regions[1].blocks.append()


class WireOp:

  def __init__(self,
               data_type,
               name,
               *,
               inner_sym=None,
               svAttributes=None,
               loc=None,
               ip=None):
    attributes = {"name": StringAttr.get(name)}
    if inner_sym is not None:
      attributes["inner_sym"] = StringAttr.get(inner_sym)
    if svAttributes is not None:
      attributes["svAttributes"] = ArrayAttr.get(svAttributes)
    OpView.__init__(
        self,
        self.build_generic(attributes=attributes,
                           results=[data_type],
                           operands=[],
                           successors=None,
                           regions=0,
                           loc=loc,
                           ip=ip))

  @staticmethod
  def create(data_type, name=None):
    if not isinstance(data_type, hw.InOutType):
      data_type = hw.InOutType.get(data_type)
    return sv.WireOp(data_type, name)


class RegOp:

  def __init__(self,
               data_type,
               name,
               *,
               inner_sym=None,
               svAttributes=None,
               loc=None,
               ip=None):
    attributes = {"name": StringAttr.get(name)}
    if inner_sym is not None:
      attributes["inner_sym"] = FlatSymbolRefAttr.get(inner_sym)
    if svAttributes is not None:
      attributes["svAttributes"] = ArrayAttr.get(svAttributes)
    OpView.__init__(
        self,
        self.build_generic(attributes=attributes,
                           results=[data_type],
                           operands=[],
                           successors=None,
                           regions=0,
                           loc=loc,
                           ip=ip))


class AssignOp:

  @staticmethod
  def create(dest, src):
    return sv.AssignOp(dest=dest, src=src)


class ReadInOutOp:

  @staticmethod
  def create(value):
    value = support.get_value(value)
    type = support.get_self_or_inner(value.type).element_type
    return sv.ReadInOutOp(type, value)
