# RUN: rm -rf %t
# RUN: %PYTHON% %s %t 2>&1 | FileCheck %s

from pycde import dim, module, generator, types, Input, Output
from pycde.constructs import SystolicArray
import pycde

import pycde.dialects.comb as comb

import sys


@module
class Top:
  clk = Input(types.i1)
  row_data = Input(dim(8, 3))
  col_data = Input(dim(8, 2))
  out = Output(dim(8, 2, 3))

  @generator
  def build(ports):
    # If we just feed constants, CIRCT pre-computes the outputs in the
    # generated Verilog! Keep these for demo purposes.
    # row_data = dim(8, 3)([1, 2, 3])
    # col_data = dim(8, 2)([4, 5])

    # CHECK-LABEL: %{{.+}} = msft.systolic.array [%{{.+}} : 3 x i8] [%{{.+}} : 2 x i8] pe (%arg0, %arg1) -> (i8) {
    # CHECK:         [[SUM:%.+]] = comb.add %arg0, %arg1 {sv.namehint = "sum"} : i8
    # CHECK:         [[SUMR:%.+]] = seq.compreg sym @sum__reg1 [[SUM]], %clk : i8
    # CHECK:         msft.pe.output [[SUMR]] : i8
    def pe(r, c):
      sum = comb.AddOp(r, c)
      sum.name = "sum"
      return sum.reg(ports.clk)

    pe_outputs = SystolicArray(ports.row_data, ports.col_data, pe)

    ports.out = pe_outputs


t = pycde.System([Top], name="SATest", output_directory=sys.argv[1])
t.generate()
print("=== Pre-pass mlir dump")
t.print()

print("=== Running passes")
t.run_passes()

print("=== Final mlir dump")
t.print()
# CHECK-LABEL: hw.module @Top<__INST_HIER: none = "INSTANTIATE_WITH_INSTANCE_PATH">(%clk: i1, %col_data: !hw.array<2xi8>, %row_data: !hw.array<3xi8>) -> (out: !hw.array<3xarray<2xi8>>)
# CHECK:         %sum__reg1_0_0 = sv.reg sym @sum__reg1  : !hw.inout<i8>
# CHECK:         sv.read_inout %sum__reg1_0_0 : !hw.inout<i8>

t.emit_outputs()
