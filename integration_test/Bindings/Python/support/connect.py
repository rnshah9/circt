# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import mlir
import circt
from circt.support import connect
from circt.dialects import hw
from circt.esi import types


def build(mod, dummy_mod):
  # CHECK: %[[C0:.+]] = hw.constant 0
  const = hw.ConstantOp.create(types.i32, 0)
  inst = dummy_mod.instantiate("d")
  connect(inst.x, inst.y)
  connect(inst.x, const)
  connect(inst.x, const.result)
  # CHECK: hw.instance "d" @Dummy(x: %[[C0]]: i32)


with mlir.ir.Context() as ctx, mlir.ir.Location.unknown():
  circt.register_dialects(ctx)
  m = mlir.ir.Module.create()
  with mlir.ir.InsertionPoint(m.body):
    dummy = hw.HWModuleOp(name='Dummy',
                          input_ports=[("x", types.i32)],
                          output_ports=[("y", types.i32)],
                          body_builder=lambda m: {"y": m.x})

    hw.HWModuleOp(name='top',
                  input_ports=[],
                  output_ports=[],
                  body_builder=lambda top: build(top, dummy))
  print(m)
