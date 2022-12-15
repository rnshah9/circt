// REQUIRES: iverilog,cocotb

// This test is executed with all different buffering strategies

// RUN: hlstool %s --dynamic-firrtl --ir-input-level 1 --buffering-strategy=all --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: %PYTHON% %S/../cocotb_driver.py --objdir=%T --topLevel=top --pythonModule=tuple_packing --pythonFolder=%S %t.sv 2>&1 | FileCheck %s

// RUN: hlstool %s --dynamic-firrtl --ir-input-level 1 --buffering-strategy=allFIFO --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: %PYTHON% %S/../cocotb_driver.py --objdir=%T --topLevel=top --pythonModule=tuple_packing --pythonFolder=%S %t.sv 2>&1 | FileCheck %s

// RUN: hlstool %s --dynamic-firrtl --ir-input-level 1 --buffering-strategy=cycles --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: %PYTHON% %S/../cocotb_driver.py --objdir=%T --topLevel=top --pythonModule=tuple_packing --pythonFolder=%S %t.sv 2>&1 | FileCheck %s

// CHECK: ** TEST
// CHECK: ** TESTS=[[N:.*]] PASS=[[N]] FAIL=0 SKIP=0

module {
  handshake.func @top(%arg0: none, ...) -> (i32, none) attributes {argNames = ["inCtrl"], resNames = ["out0", "outCtrl"]} {
    %0:4 = fork [4] %arg0 : none
    %const0 = constant %0#0 {value = 123 : i32} : i32
    %const1 = constant %0#1 {value = 456 : i32} : i32
    %const2 = constant %0#2 {value = 0 : i32} : i32

    %tuple = handshake.pack %const0, %const1, %const2 : tuple<i32, i32, i32>
    %res:3 = handshake.unpack %tuple : tuple<i32, i32, i32>

    %sum = arith.addi %res#0, %res#1 : i32
    sink %res#2 : i32

    return %sum, %0#3 : i32, none
  }
}
