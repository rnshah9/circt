# RUN: %PYTHON% py-split-input-file.py %s | FileCheck %s

from pycde import generator, dim, Clock, Input, Output, Value, types
from pycde.constructs import Mux
from pycde.testing import unittestmodule


def array_from_tuple(*input):
  return Value(input)


# CHECK-LABEL: msft.module @ComplexMux {} (%Clk: i1, %In: !hw.array<5xarray<4xi3>>, %Sel: i1) -> (Out: !hw.array<4xi3>, OutArr: !hw.array<2xarray<4xi3>>, OutInt: i1, OutSlice: !hw.array<3xarray<4xi3>>)
# CHECK:         %c3_i3 = hw.constant 3 : i3
# CHECK:         %0 = hw.array_get %In[%c3_i3] {sv.namehint = "In__3"} : !hw.array<5xarray<4xi3>>
# CHECK:         %In__3__reg1 = seq.compreg sym @In__3__reg1 %0, %Clk : !hw.array<4xi3>
# CHECK:         %In__3__reg2 = seq.compreg sym @In__3__reg2 %In__3__reg1, %Clk : !hw.array<4xi3>
# CHECK:         %In__3__reg3 = seq.compreg sym @In__3__reg3 %In__3__reg2, %Clk : !hw.array<4xi3>
# CHECK:         %c1_i3 = hw.constant 1 : i3
# CHECK:         [[R1:%.+]] = hw.array_get %In[%c1_i3] {sv.namehint = "In__1"} : !hw.array<5xarray<4xi3>>
# CHECK:         [[R2:%.+]] = hw.array_create [[R1]], %In__3__reg3 : !hw.array<4xi3>
# CHECK:         [[R3:%.+]] = hw.array_get [[R2]][%Sel] : !hw.array<2xarray<4xi3>>
# CHECK:         %c0_i3 = hw.constant 0 : i3
# CHECK:         [[R4:%.+]] = hw.array_get %In[%c0_i3] {sv.namehint = "In__0"} : !hw.array<5xarray<4xi3>>
# CHECK:         %c1_i3_0 = hw.constant 1 : i3
# CHECK:         [[R5:%.+]] = hw.array_get %In[%c1_i3_0] {sv.namehint = "In__1"} : !hw.array<5xarray<4xi3>>
# CHECK:         [[R6:%.+]] = hw.array_create [[R5]], [[R4]] : !hw.array<4xi3>
# CHECK:         [[R7:%.+]] = hw.array_slice %In[%c0_i3_1] {sv.namehint = "In_0upto3"} : (!hw.array<5xarray<4xi3>>) -> !hw.array<3xarray<4xi3>>
# CHECK:         [[R9:%.+]] = hw.array_get %8[%c0_i2] {sv.namehint = "In__0__0"} : !hw.array<4xi3>
# CHECK:         %c0_i2_3 = hw.constant 0 : i2
# CHECK:         [[R10:%.+]] = comb.concat %c0_i2_3, %Sel : i2, i1
# CHECK:         [[R11:%.+]] = comb.shru [[R9]], [[R10]] : i3
# CHECK:         [[R12:%.+]] = comb.extract [[R11]] from 0 : (i3) -> i1
# CHECK:         msft.output [[R3]], [[R6]], [[R12]], [[R7]] : !hw.array<4xi3>, !hw.array<2xarray<4xi3>>, i1, !hw.array<3xarray<4xi3>>


@unittestmodule()
class ComplexMux:

  Clk = Clock()
  In = Input(dim(3, 4, 5))
  Sel = Input(dim(1))
  Out = Output(dim(3, 4))
  OutArr = Output(dim(3, 4, 2))
  OutSlice = Output(dim(3, 4, 3))
  OutInt = Output(types.i1)

  @generator
  def create(ports):
    ports.Out = Mux(ports.Sel, ports.In[3].reg().reg(cycles=2), ports.In[1])

    ports.OutArr = array_from_tuple(ports.In[0], ports.In[1])
    ports.OutSlice = ports.In[0:3]

    ports.OutInt = ports.In[0][0][ports.Sel]


# -----

# CHECK-LABEL:  msft.module @Slicing {} (%In: !hw.array<5xarray<4xi8>>, %Sel2: i2, %Sel8: i8) -> (OutArrSlice2: !hw.array<2xarray<4xi8>>, OutArrSlice8: !hw.array<2xarray<4xi8>>, OutIntSlice: i2)
# CHECK:          [[R0:%.+]] = hw.array_get %In[%c0_i3] {sv.namehint = "In__0"} : !hw.array<5xarray<4xi8>>
# CHECK:          [[R1:%.+]] = hw.array_get %0[%c0_i2] {sv.namehint = "In__0__0"} : !hw.array<4xi8>
# CHECK:          [[R2:%.+]] = comb.concat %c0_i6, %Sel2 : i6, i2
# CHECK:          [[R3:%.+]] = comb.shru [[R1]], [[R2]] : i8
# CHECK:          [[R4:%.+]] = comb.extract [[R3]] from 0 : (i8) -> i2
# CHECK:          [[R5:%.+]] = comb.concat %false, %Sel2 : i1, i2
# CHECK:          [[R6:%.+]] = hw.array_slice %In[[[R5]]] : (!hw.array<5xarray<4xi8>>) -> !hw.array<2xarray<4xi8>>
# CHECK:          [[R7:%.+]] = comb.extract %Sel8 from 0 : (i8) -> i3
# CHECK:          [[R8:%.+]] = hw.array_slice %In[[[R7]]] : (!hw.array<5xarray<4xi8>>) -> !hw.array<2xarray<4xi8>>
# CHECK:          msft.output [[R6]], [[R8]], [[R4]] : !hw.array<2xarray<4xi8>>, !hw.array<2xarray<4xi8>>, i2


@unittestmodule()
class Slicing:
  In = Input(dim(8, 4, 5))
  Sel8 = Input(types.i8)
  Sel2 = Input(types.i2)

  OutIntSlice = Output(types.i2)
  OutArrSlice8 = Output(dim(8, 4, 2))
  OutArrSlice2 = Output(dim(8, 4, 2))

  @generator
  def create(ports):
    i = ports.In[0][0]
    ports.OutIntSlice = i.slice(ports.Sel2, 2)
    ports.OutArrSlice2 = ports.In.slice(ports.Sel2, 2)
    ports.OutArrSlice8 = ports.In.slice(ports.Sel8, 2)
