# RUN: %PYTHON% py-split-input-file.py %s | FileCheck %s

from pycde import Clock, Input, types, System
from pycde.module import AppID, externmodule, generator, module
from pycde.testing import unittestmodule


# CHECK: TypeError: Module parameter definitions cannot have *args
@externmodule
def foo(*args):
  pass


# -----


# CHECK: TypeError: Module parameter definitions cannot have **kwargs
@externmodule
def bar(**kwargs):
  pass


# -----


@unittestmodule()
class ClkError:
  a = Input(types.i32)

  @generator
  def build(ports):
    # CHECK: ValueError: If 'clk' not specified, must be in clock block
    ports.a.reg()


# -----


@unittestmodule()
class AppIDError:

  @generator
  def build(ports):
    c = types.i32(4)
    # CHECK: ValueError: AppIDs can only be attached to ops with symbols
    c.appid = AppID("c", 0)


# -----


@module
class Test:
  clk = Clock()
  x = Input(types.i32)

  @generator
  def build(ports):
    ports.x.reg(appid=AppID("reg", 5))


t = System([Test], name="Test")
t.generate()

inst = t.get_instance(Test)
# CHECK: reg[8] not found
inst.reg[8]

# -----


@unittestmodule()
class OperatorError:
  a = Input(types.i32)
  b = Input(types.si32)

  @generator
  def build(ports):
    # CHECK: Operator '+' is not supported on signless values. LHS operand should be cast .as_sint()/.as_uint().
    ports.a + ports.b


# -----


@unittestmodule()
class OperatorError2:
  a = Input(types.i32)
  b = Input(types.si32)

  @generator
  def build(ports):
    # CHECK: Operator '+' is not supported on signless values. RHS operand should be cast .as_sint()/.as_uint().
    ports.b + ports.a


# -----


@unittestmodule()
class OperatorError2:
  a = Input(types.i32)
  b = Input(types.si32)

  @generator
  def build(ports):
    # CHECK: Operator '==' requires LHS to be cast .as_int().
    ports.b == ports.a
