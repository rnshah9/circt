# RUN: %PYTHON% py-split-input-file.py %s | FileCheck %s

from pycde import generator, types, Input, Output
from pycde.behavioral import If, Else, EndIf
from pycde.testing import unittestmodule

# CHECK-LABEL: msft.module @IfNestedTest {} (%a: ui8, %b: ui8, %cond: i1, %cond2: i1) -> (out: ui17, out2: ui24)
# CHECK:         %0 = hwarith.mul %a, %b {sv.namehint = "a_mul_b"} : (ui8, ui8) -> ui16
# CHECK:         %1 = comb.mux %cond2, %a, %b {sv.namehint = "x"} : ui8
# CHECK:         %2 = hwarith.mul %a, %1 {sv.namehint = "v_thenvalue"} : (ui8, ui8) -> ui16
# CHECK:         %3 = hwarith.mul %2, %b {sv.namehint = "u_thenvalue"} : (ui16, ui8) -> ui24
# CHECK:         %4 = hwarith.cast %b {sv.namehint = "v_elsevalue"} : (ui8) -> ui16
# CHECK:         %5 = hwarith.mul %4, %a {sv.namehint = "u_elsevalue"} : (ui16, ui8) -> ui24
# CHECK:         %6 = comb.mux %cond, %2, %4 {sv.namehint = "v"} : ui16
# CHECK:         %7 = comb.mux %cond, %3, %5 {sv.namehint = "u"} : ui24
# CHECK:         %8 = hwarith.add %6, %0 {sv.namehint = "v_plus_a_mul_b"} : (ui16, ui16) -> ui17
# CHECK:         msft.output %8, %7 : ui17, ui24


@unittestmodule()
class IfNestedTest:
  cond = Input(types.i1)
  cond2 = Input(types.i1)
  a = Input(types.ui8)
  b = Input(types.ui8)

  out = Output(types.ui17)
  out2 = Output(types.ui24)

  @generator
  def build(ports):
    w = ports.a * ports.b
    with If(ports.cond):
      with If(ports.cond2):
        x = ports.a
      with Else:
        x = ports.b
      EndIf()
      v = ports.a * x
      u = v * ports.b
    with Else:
      v = ports.b.as_uint(16)
      u = v * ports.a
    EndIf()

    ports.out2 = u
    ports.out = v + w


# CHECK-LABEL: msft.module @IfDefaultTest {} (%a: ui8, %b: ui8, %cond: i1, %cond2: i1) -> (out: ui8)
# CHECK:         [[r1:%.+]] = comb.mux %cond2, %b, %a {sv.namehint = "v_thenvalue"} : ui8
# CHECK:         [[r0:%.+]] = comb.mux %cond, [[r1]], %a {sv.namehint = "v"} : ui8
# CHECK:         msft.output [[r0]] : ui8


@unittestmodule()
class IfDefaultTest:
  cond = Input(types.i1)
  cond2 = Input(types.i1)
  a = Input(types.ui8)
  b = Input(types.ui8)

  out = Output(types.ui8)

  @generator
  def build(ports):
    v = ports.a
    with If(ports.cond):
      with If(ports.cond2):
        v = ports.b
      EndIf()
    EndIf()
    ports.out = v


# -----


@unittestmodule()
class IfMismatchErrorTest:
  cond = Input(types.i1)
  a = Input(types.ui8)
  b = Input(types.ui4)

  out = Output(types.ui8)

  @generator
  def build(ports):
    v = ports.a
    with If(ports.cond):
      v = ports.b
    # CHECK: TypeError: 'Then' and 'Else' values must have same type for
    EndIf()
    ports.out = v


# -----


@unittestmodule()
class IfMismatchEndIfTest:

  @generator
  def build(ports):
    # CHECK: AssertionError: EndIf() called without matching If()
    EndIf()


# -----


@unittestmodule()
class IfCondErrorTest:
  cond = Input(types.i2)

  @generator
  def build(ports):
    # CHECK: TypeError: 'Cond' bit width must be 1
    with If(ports.cond):
      pass
    EndIf()
