// RUN: circt-opt %s --verify-diagnostics --split-input-file | circt-opt --verify-diagnostics | FileCheck %s

// CHECK: module attributes {calyx.entrypoint = "main"} {
module attributes {calyx.entrypoint = "main"} {
  // CHECK-LABEL: calyx.component @A(%in: i8, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i8, %done: i1 {done}) {
  calyx.component @A(%in: i8, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i8, %done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }

  // CHECK-LABEL: calyx.component @B(%in: i8, %clk: i1 {clk}, %go: i1 {go}, %reset: i1 {reset}) -> (%out: i1, %done: i1 {done}) {
  calyx.component @B (%in: i8, %clk: i1 {clk}, %go: i1 {go}, %reset: i1 {reset}) -> (%out: i1, %done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }

  // CHECK-LABEL:   calyx.component @main(%clk: i1 {clk}, %go: i1 {go}, %reset: i1 {reset}) -> (%done: i1 {done}) {
  calyx.component @main(%clk: i1 {clk}, %go: i1 {go}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    // CHECK:      %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i8, i1, i1, i1, i8, i1
    // CHECK-NEXT: %r2.in, %r2.write_en, %r2.clk, %r2.reset, %r2.out, %r2.done = calyx.register @r2 : i1, i1, i1, i1, i1, i1
    // CHECK-NEXT: %mu.clk, %mu.reset, %mu.go, %mu.left, %mu.right, %mu.out, %mu.done = calyx.std_mult_pipe @mu : i1, i1, i1, i32, i32, i32, i1
    // CHECK-NEXT: %du.clk, %du.reset, %du.go, %du.left, %du.right, %du.out_quotient, %du.done = calyx.std_divu_pipe @du : i1, i1, i1, i32, i32, i32, i1
    // CHECK-NEXT: %m.addr0, %m.addr1, %m.write_data, %m.write_en, %m.clk, %m.read_data, %m.done = calyx.memory @m <[64, 64] x 8> [6, 6] : i6, i6, i8, i1, i1, i8, i1
    // CHECK-NEXT: %c0.in, %c0.go, %c0.clk, %c0.reset, %c0.out, %c0.done = calyx.instance @c0 of @A : i8, i1, i1, i1, i8, i1
    // CHECK-NEXT: %c1.in, %c1.go, %c1.clk, %c1.reset, %c1.out, %c1.done = calyx.instance @c1 of @A : i8, i1, i1, i1, i8, i1
    // CHECK-NEXT: %c2.in, %c2.clk, %c2.go, %c2.reset, %c2.out, %c2.done = calyx.instance @c2 of @B : i8, i1, i1, i1, i1, i1
    // CHECK-NEXT: %adder.left, %adder.right, %adder.out = calyx.std_add @adder : i8, i8, i8
    // CHECK-NEXT: %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1
    // CHECK-NEXT: %pad.in, %pad.out = calyx.std_pad @pad : i8, i9
    // CHECK-NEXT: %slice.in, %slice.out = calyx.std_slice @slice : i8, i7
    // CHECK-NEXT: %not.in, %not.out = calyx.std_not @not : i1, i1
    // CHECK-NEXT: %wire.in, %wire.out = calyx.std_wire @wire : i8, i8
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i8, i1, i1, i1, i8, i1
    %r2.in, %r2.write_en, %r2.clk, %r2.reset, %r2.out, %r2.done = calyx.register @r2 : i1, i1, i1, i1, i1, i1
    %mu.clk, %mu.reset, %mu.go, %mu.lhs, %mu.rhs, %mu.out, %mu.done = calyx.std_mult_pipe @mu : i1, i1, i1, i32, i32, i32, i1
    %du.clk, %du.reset, %du.go, %du.left, %du.right, %du.out, %du.done = calyx.std_divu_pipe @du : i1, i1, i1, i32, i32, i32, i1
    %m.addr0, %m.addr1, %m.write_data, %m.write_en, %m.clk, %m.read_data, %m.done = calyx.memory @m <[64, 64] x 8> [6, 6] : i6, i6, i8, i1, i1, i8, i1
    %c0.in, %c0.go, %c0.clk, %c0.reset, %c0.out, %c0.done = calyx.instance @c0 of @A : i8, i1, i1, i1, i8, i1
    %c1.in, %c1.go, %c1.clk, %c1.reset, %c1.out, %c1.done = calyx.instance @c1 of @A : i8, i1, i1, i1, i8, i1
    %c2.in, %c2.clk, %c2.go, %c2.reset, %c2.out, %c2.done = calyx.instance @c2 of @B : i8, i1, i1, i1, i1, i1
    %adder.left, %adder.right, %adder.out = calyx.std_add @adder : i8, i8, i8
    %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1
    %pad.in, %pad.out = calyx.std_pad @pad : i8, i9
    %slice.in, %slice.out = calyx.std_slice @slice : i8, i7
    %not.in, %not.out = calyx.std_not @not : i1, i1
    %wire.in, %wire.out = calyx.std_wire @wire : i8, i8
    %c1_i1 = hw.constant 1 : i1
    %c0_i1 = hw.constant 0 : i1
    %c0_i6 = hw.constant 0 : i6
    %c0_i8 = hw.constant 0 : i8

    calyx.wires {
      // CHECK:      calyx.assign %not.in = %r2.out : i1
      // CHECK-NEXT: calyx.assign %gt.left = %r2.out ? %adder.out : i8
      // CHECK-NEXT: calyx.assign %gt.left = %not.out ? %adder.out : i8
      // CHECK-NEXT: calyx.assign %r.in = %0 ? %c0_i8 : i8
      // CHECK-NEXT: %0 = comb.and %true, %true : i1
      calyx.assign %not.in = %r2.out : i1
      calyx.assign %gt.left = %r2.out ? %adder.out : i8
      calyx.assign %gt.left = %not.out ? %adder.out : i8
      calyx.assign %r.in = %0 ? %c0_i8 : i8
      %0 = comb.and %c1_i1, %c1_i1 : i1

      // CHECK: calyx.group @Group1 {
      calyx.group @Group1 {
        // CHECK: calyx.assign %c1.in = %c0.out : i8
        // CHECK-NEXT: calyx.group_done %c1.done : i1
        calyx.assign %c1.in = %c0.out : i8
        calyx.group_done %c1.done : i1
      }
      calyx.comb_group @ReadMemory {
        // CHECK: calyx.assign %m.addr0 = %c0_i6 : i6
        // CHECK-NEXT: calyx.assign %m.addr1 = %c0_i6 : i6
        // CHECK-NEXT: calyx.assign %gt.left = %m.read_data : i8
        // CHECK-NEXT: calyx.assign %gt.right = %c0_i8 : i8
        calyx.assign %m.addr0 = %c0_i6 : i6
        calyx.assign %m.addr1 = %c0_i6 : i6
        calyx.assign %gt.left = %m.read_data : i8
        calyx.assign %gt.right = %c0_i8 : i8
      }
      calyx.group @Group3 {
        calyx.assign %r.in = %c0.out : i8
        calyx.assign %r.write_en = %c1_i1 : i1
        calyx.group_done %r.done : i1
      }
    }
    calyx.control {
      // CHECK:      calyx.seq {
      // CHECK-NEXT: calyx.seq {
      // CHECK-NEXT: calyx.enable @Group1
      // CHECK-NEXT: calyx.enable @Group3
      // CHECK-NEXT: calyx.seq {
      // CHECK-NEXT: calyx.if %gt.out with @ReadMemory {
      // CHECK-NEXT: calyx.enable @Group1
      // CHECK-NEXT: } else {
      // CHECK-NEXT: calyx.enable @Group3
      // CHECK-NEXT: }
      // CHECK-NEXT: calyx.if %c2.out {
      // CHECK-NEXT: calyx.enable @Group1
      // CHECK-NEXT: }
      // CHECK-NEXT: calyx.while %gt.out with @ReadMemory {
      // CHECK-NEXT: calyx.while %c2.out {
      // CHECK-NEXT: calyx.enable @Group1
      // CHECK:      calyx.par {
      // CHECK-NEXT: calyx.enable @Group1
      // CHECK-NEXT: calyx.enable @Group3
      calyx.seq {
        calyx.seq {
          calyx.enable @Group1
          calyx.enable @Group3
          calyx.seq {
            calyx.if %gt.out with @ReadMemory {
              calyx.enable @Group1
            } else {
              calyx.enable @Group3
            }
            calyx.if %c2.out {
              calyx.enable @Group1
            }
            calyx.while %gt.out with @ReadMemory {
              calyx.while %c2.out {
                calyx.enable @Group1
              }
            }
          }
        }
        calyx.par {
          calyx.enable @Group1
          calyx.enable @Group3
        }
      }
    }
  }
}

// -----
// CHECK: module attributes {calyx.entrypoint = "A"} {
module attributes {calyx.entrypoint = "A"} {
  // CHECK: hw.module.extern @prim(%in: i32) -> (out: i32) attributes {filename = "test.v"}
  hw.module.extern @prim(%in: i32) -> (out: i32) attributes {filename = "test.v"}

  // CHECK: hw.module.extern @params<WIDTH: i32>(%in: !hw.int<#hw.param.decl.ref<"WIDTH">>) -> (out: !hw.int<#hw.param.decl.ref<"WIDTH">>) attributes {filename = "test.v"}
  hw.module.extern @params<WIDTH: i32>(%in: !hw.int<#hw.param.decl.ref<"WIDTH">>) -> (out: !hw.int<#hw.param.decl.ref<"WIDTH">>) attributes {filename = "test.v"}

  // CHECK-LABEL: calyx.component @A(%in_0: i32, %in_1: i32, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out_0: i32, %out_1: i32, %done: i1 {done})
  calyx.component @A(%in_0: i32, %in_1: i32, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out_0: i32, %out_1: i32, %done: i1 {done}) {
    // CHECK: %true = hw.constant true
    %c1_1 = hw.constant 1 : i1
    // CHECK-NEXT: %params_0.in, %params_0.out = calyx.primitive @params_0 of @params<WIDTH: i32 = 32> : i32, i32
    %params_0.in, %params_0.out = calyx.primitive @params_0 of @params<WIDTH: i32 = 32> : i32, i32
    // CHECK-NEXT: %prim_0.in, %prim_0.out = calyx.primitive @prim_0 of @prim : i32, i32
    %prim_0.in, %prim_0.out = calyx.primitive @prim_0 of @prim : i32, i32

    calyx.wires {
      // CHECK: calyx.assign %done = %true : i1
      calyx.assign %done = %c1_1 : i1
      // CHECK-NEXT: calyx.assign %params_0.in = %in_0 : i32
      calyx.assign %params_0.in = %in_0 : i32
      // CHECK-NEXT: calyx.assign %out_0 = %params_0.out : i32
      calyx.assign %out_0 = %params_0.out : i32
      // CHECK-NEXT: calyx.assign %prim_0.in = %in_1 : i32
      calyx.assign %prim_0.in = %in_1 : i32
      // CHECK-NEXT: calyx.assign %out_1 = %prim_0.out : i32
      calyx.assign %out_1 = %prim_0.out : i32
    }
    calyx.control {}
  } {static = 1}
}

// -----
// CHECK: module attributes {calyx.entrypoint = "A"} {
module attributes {calyx.entrypoint = "A"} {
  // CHECK: hw.module.extern @params<WIDTH: i32>(%in: !hw.int<#hw.param.decl.ref<"WIDTH">>, %clk: i1 {calyx.clk}, %go: i1 {calyx.go}) -> (out: !hw.int<#hw.param.decl.ref<"WIDTH">>, done: i1 {calyx.done}) attributes {filename = "test.v"}
  hw.module.extern @params<WIDTH: i32>(%in: !hw.int<#hw.param.decl.ref<"WIDTH">>, %clk: i1 {calyx.clk}, %go: i1 {calyx.go}) -> (out: !hw.int<#hw.param.decl.ref<"WIDTH">>, done: i1 {calyx.done}) attributes {filename = "test.v"}

  // CHECK-LABEL: calyx.component @A(%in_0: i32, %in_1: i32, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out_0: i32, %out_1: i32, %done: i1 {done})
  calyx.component @A(%in_0: i32, %in_1: i32, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out_0: i32, %out_1: i32, %done: i1 {done}) {
    // CHECK: %true = hw.constant true
    %c1_1 = hw.constant 1 : i1
    // CHECK-NEXT: %params_0.in, %params_0.clk, %params_0.go, %params_0.out, %params_0.done = calyx.primitive @params_0 of @params<WIDTH: i32 = 32> : i32, i1, i1, i32, i1
    %params_0.in, %params_0.clk, %params_0.go, %params_0.out, %params_0.done = calyx.primitive @params_0 of @params<WIDTH: i32 = 32> : i32, i1, i1, i32, i1

    calyx.wires {
      // CHECK: calyx.assign %done = %true : i1
      calyx.assign %done = %c1_1 : i1
      // CHECK-NEXT: calyx.assign %params_0.in = %in_0 : i32
      calyx.assign %params_0.in = %in_0 : i32
      // CHECK-NEXT: calyx.assign %out_0 = %params_0.out : i32
      calyx.assign %out_0 = %params_0.out : i32
    }
    calyx.control {}
  } {static = 1}
}
