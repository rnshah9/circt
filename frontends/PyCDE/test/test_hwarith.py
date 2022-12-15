# RUN: %PYTHON% py-split-input-file.py %s | FileCheck %s

from pycde import Input, Output, generator
from pycde.testing import unittestmodule
from pycde.pycde_types import types


# CHECK: msft.module @InfixArith {} (%in0: si16, %in1: ui16)
# CHECK-NEXT:   %0 = hwarith.add %in0, %in1 {{({sv.namehint = ".*"} )?}}: (si16, ui16) -> si18
# CHECK-NEXT:   %1 = hwarith.sub %in0, %in1 {{({sv.namehint = ".*"} )?}}: (si16, ui16) -> si18
# CHECK-NEXT:   %2 = hwarith.mul %in0, %in1 {{({sv.namehint = ".*"} )?}}: (si16, ui16) -> si32
# CHECK-NEXT:   %3 = hwarith.div %in0, %in1 {{({sv.namehint = ".*"} )?}}: (si16, ui16) -> si16
# CHECK-NEXT:   %c-1_i16 = hw.constant -1 {{({sv.namehint = ".*"} )?}}: i16
# CHECK-NEXT:   %4 = hwarith.cast %c-1_i16 {{({sv.namehint = ".*"} )?}}: (i16) -> si16
# CHECK-NEXT:   %5 = hwarith.mul %in0, %4 {{({sv.namehint = ".*"} )?}}: (si16, si16) -> si32
# CHECK-NEXT:   msft.output
@unittestmodule(run_passes=True)
class InfixArith:
  in0 = Input(types.si16)
  in1 = Input(types.ui16)

  @generator
  def construct(ports):
    add = ports.in0 + ports.in1
    sub = ports.in0 - ports.in1
    mul = ports.in0 * ports.in1
    div = ports.in0 / ports.in1
    neg = -ports.in0


# -----


# CHECK: msft.module @InfixLogic {} (%in0: si16, %in1: ui16)
# CHECK-NEXT:  %0 = hwarith.cast %in0 {{({sv.namehint = ".*"} )?}}: (si16) -> i16
# CHECK-NEXT:  %1 = hwarith.cast %in1 {{({sv.namehint = ".*"} )?}}: (ui16) -> i16
# CHECK-NEXT:  %2 = comb.and bin %0, %1 {{({sv.namehint = ".*"} )?}}: i16
# CHECK-NEXT:  %3 = hwarith.cast %in0 {{({sv.namehint = ".*"} )?}}: (si16) -> i16
# CHECK-NEXT:  %4 = hwarith.cast %in1 {{({sv.namehint = ".*"} )?}}: (ui16) -> i16
# CHECK-NEXT:  %5 = comb.or bin %3, %4 {{({sv.namehint = ".*"} )?}}: i16
# CHECK-NEXT:  %6 = hwarith.cast %in0 {{({sv.namehint = ".*"} )?}}: (si16) -> i16
# CHECK-NEXT:  %7 = hwarith.cast %in1 {{({sv.namehint = ".*"} )?}}: (ui16) -> i16
# CHECK-NEXT:  %8 = comb.xor bin %6, %7 {{({sv.namehint = ".*"} )?}}: i16
# CHECK-NEXT:  %9 = hwarith.cast %in0 {{({sv.namehint = ".*"} )?}}: (si16) -> i16
# CHECK-NEXT:  %c-1_i16 = hw.constant -1 {{({sv.namehint = ".*"} )?}}: i16
# CHECK-NEXT:  %10 = comb.xor bin %9, %c-1_i16 {{({sv.namehint = ".*"} )?}}: i16
# CHECK-NEXT:  msft.output
@unittestmodule(run_passes=True)
class InfixLogic:
  in0 = Input(types.si16)
  in1 = Input(types.ui16)

  @generator
  def construct(ports):
    and_ = ports.in0 & ports.in1
    or_ = ports.in0 | ports.in1
    xor = ports.in0 ^ ports.in1
    inv = ~ports.in0


# -----


# CHECK: msft.module @InfixComparison {} (%in0: i16, %in1: i16)
# CHECK-NEXT:    %0 = comb.icmp bin eq %in0, %in1 {{({sv.namehint = ".*"} )?}}: i16
# CHECK-NEXT:    %1 = comb.icmp bin ne %in0, %in1 {{({sv.namehint = ".*"} )?}}: i16
# CHECK-NEXT:    msft.output
@unittestmodule(run_passes=True)
class InfixComparison:
  in0 = Input(types.i16)
  in1 = Input(types.i16)

  @generator
  def construct(ports):
    eq = ports.in0 == ports.in1
    neq = ports.in0 != ports.in1


# -----


# CHECK:  msft.module @Multiple {} (%in0: si16, %in1: si16) -> (out0: i16)
# CHECK-NEXT:    %0 = hwarith.add %in0, %in1 {{({sv.namehint = ".*"} )?}}: (si16, si16) -> si17
# CHECK-NEXT:    %1 = hwarith.add %0, %in0 {{({sv.namehint = ".*"} )?}}: (si17, si16) -> si18
# CHECK-NEXT:    %2 = hwarith.add %1, %in1 {{({sv.namehint = ".*"} )?}}: (si18, si16) -> si19
# CHECK-NEXT:    %3 = hwarith.cast %2 {{({sv.namehint = ".*"} )?}}: (si19) -> i16
# CHECK-NEXT:    msft.output %3 {{({sv.namehint = ".*"} )?}}: i16
@unittestmodule(run_passes=True)
class Multiple:
  in0 = Input(types.si16)
  in1 = Input(types.si16)
  out0 = Output(types.i16)

  @generator
  def construct(ports):
    ports.out0 = (ports.in0 + ports.in1 + ports.in0 + ports.in1).as_int(16)


# -----


# CHECK:  msft.module @Casting {} (%in0: i16)
# CHECK-NEXT:    %0 = hwarith.cast %in0 {{({sv.namehint = ".*"} )?}}: (i16) -> si16
# CHECK-NEXT:    %1 = hwarith.cast %in0 {{({sv.namehint = ".*"} )?}}: (i16) -> ui16
# CHECK-NEXT:    %2 = hwarith.cast %0 {{({sv.namehint = ".*"} )?}}: (si16) -> i16
# CHECK-NEXT:    %3 = hwarith.cast %in0 {{({sv.namehint = ".*"} )?}}: (i16) -> si8
# CHECK-NEXT:    %4 = hwarith.cast %in0 {{({sv.namehint = ".*"} )?}}: (i16) -> ui8
# CHECK-NEXT:    %5 = hwarith.cast %0 {{({sv.namehint = ".*"} )?}}: (si16) -> i8
# CHECK-NEXT:    %6 = hwarith.cast %0 {{({sv.namehint = ".*"} )?}}: (si16) -> si24
# CHECK-NEXT:    msft.output
@unittestmodule(run_passes=True)
class Casting:
  in0 = Input(types.i16)

  @generator
  def construct(ports):
    in0s = ports.in0.as_sint()
    in0u = ports.in0.as_uint()
    in0s_i = in0s.as_int()
    in0s8 = ports.in0.as_sint(8)
    in0u8 = ports.in0.as_uint(8)
    in0s_i8 = in0s.as_int(8)
    in0s_s24 = in0s.as_sint(24)


# -----


# CHECK: hw.module @Lowering<__INST_HIER: none = "INSTANTIATE_WITH_INSTANCE_PATH">(%in0: i16, %in1: i16) -> (out0: i16)
# CHECK-NEXT:    %0 = comb.add %in0, %in1 {{({sv.namehint = ".*"} )?}}: i16
# CHECK-NEXT:    hw.output %0 {{({sv.namehint = ".*"} )?}}: i16
@unittestmodule(generate=True, run_passes=True, print_after_passes=True)
class Lowering:
  in0 = Input(types.i16)
  in1 = Input(types.i16)
  out0 = Output(types.i16)

  @generator
  def construct(ports):
    ports.out0 = (ports.in0.as_sint() + ports.in1.as_sint()).as_int(16)
