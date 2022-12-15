#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from circt.dialects import esi as raw_esi, msft
from .pycde_types import ChannelType

import mlir.ir

from functools import singledispatchmethod


class ModuleDecl:
  """Represents an input or output port on a design module."""

  __slots__ = ["name", "_type"]

  def __init__(self, type: mlir.ir.Type, name: str = None):
    self.name: str = name
    self._type: mlir.ir.Type = type

  @property
  def type(self):
    return self._type


class Output(ModuleDecl):
  """Create an RTL-level output port"""


class OutputChannel(Output):
  """Create an ESI output channel port."""

  def __init__(self, type: mlir.ir.Type, name: str = None):
    type = ChannelType(raw_esi.ChannelType.get(type))
    super().__init__(type, name)


class Input(ModuleDecl):
  """Create an RTL-level input port."""


class Clock(Input):
  """Create a clock input"""

  def __init__(self, name: str = None):
    super().__init__(mlir.ir.IntegerType.get_signless(1), name)


class InputChannel(Input):
  """Create an ESI input channel port."""

  def __init__(self, type: mlir.ir.Type, name: str = None):
    type = ChannelType(raw_esi.ChannelType.get(type))
    super().__init__(type, name)


class AppID:
  AttributeName = "msft.appid"

  @singledispatchmethod
  def __init__(self, name: str, idx: int):
    self._appid = msft.AppIDAttr.get(name, idx)

  @__init__.register(mlir.ir.Attribute)
  def __init__mlir_attr(self, attr: mlir.ir.Attribute):
    self._appid = msft.AppIDAttr(attr)

  @property
  def name(self) -> str:
    return self._appid.name

  @property
  def index(self) -> int:
    return self._appid.index

  def __str__(self) -> str:
    return f"{self.name}[{self.index}]"


class _PyProxy:
  """Parent class for a Python object which has a corresponding IR op (i.e. a
  proxy class)."""

  __slots__ = ["name"]

  def __init__(self, name: str):
    self.name = name
