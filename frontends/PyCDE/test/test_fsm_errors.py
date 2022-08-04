# RUN: %PYTHON% py-split-input-file.py %s | FileCheck %s

from pycde import System, Input, Output, generator
from pycde.dialects import comb
from pycde import fsm
from pycde.pycde_types import types
from pycde.testing import unittestmodule


@fsm.machine()
class FSM:
  # CHECK: ValueError: Input port a has width 2. For now, FSMs only support i1 inputs.
  a = Input(types.i2)
  A = fsm.State(initial=True)


# -----


# CHECK: ValueError: No initial state specified, please create a state with `initial=True`.
@fsm.machine()
class FSM:
  a = Input(types.i1)
  A = fsm.State()


# -----


# CHECK: ValueError: Multiple initial states specified (B, A).
@fsm.machine()
class FSM:
  a = Input(types.i1)
  A = fsm.State(initial=True)
  B = fsm.State(initial=True)