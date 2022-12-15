// RUN: circt-opt %s | circt-opt | FileCheck %s

hw.module @d1(%clk : i1, %rst : i1) -> () {
// CHECK: %myMemory = seq.hlmem @myMemory %clk, %rst : <4xi32>
  %myMemory = seq.hlmem @myMemory %clk, %rst : <4xi32>

  %c0_i2 = hw.constant 0 : i2
  %c1_i1 = hw.constant 1 : i1
  %c42_i32 = hw.constant 42 : i32

  // CHECK: %myMemory_rdata = seq.read %myMemory[%c0_i2] rden %true {latency = 0 : i64} : !seq.hlmem<4xi32>
  %myMemory_rdata = seq.read %myMemory[%c0_i2] rden %c1_i1 { latency = 0} : !seq.hlmem<4xi32>

  // CHECK: seq.write %myMemory[%c0_i2] %c42_i32 wren %true {latency = 1 : i64} : !seq.hlmem<4xi32>
  seq.write %myMemory[%c0_i2] %c42_i32 wren %c1_i1 { latency = 1 } : !seq.hlmem<4xi32>
  hw.output
}

hw.module @d2(%clk : i1, %rst : i1) -> () {
// CHECK: %myMemory = seq.hlmem @myMemory %clk, %rst : <4x8xi32>
  %myMemory = seq.hlmem @myMemory %clk, %rst : <4x8xi32>

  %c0_i2 = hw.constant 0 : i2
  %c0_i3 = hw.constant 0 : i3
  %c1_i1 = hw.constant 1 : i1
  %c42_i32 = hw.constant 42 : i32

  // CHECK: %myMemory_rdata = seq.read %myMemory[%c0_i2, %c0_i3] {latency = 0 : i64} : !seq.hlmem<4x8xi32>
  %myMemory_rdata = seq.read %myMemory[%c0_i2, %c0_i3] { latency = 0} : !seq.hlmem<4x8xi32>

  // CHECK: seq.write %myMemory[%c0_i2, %c0_i3] %c42_i32 wren %true {latency = 1 : i64} : !seq.hlmem<4x8xi32>
  seq.write %myMemory[%c0_i2, %c0_i3] %c42_i32 wren %c1_i1 { latency = 1 } : !seq.hlmem<4x8xi32>
  hw.output
}

hw.module @d0(%clk : i1, %rst : i1) -> () {
// CHECK: %myMemory = seq.hlmem @myMemory %clk, %rst : <1xi32>
  %myMemory = seq.hlmem @myMemory %clk, %rst : <1xi32>

  %c0_i0 = hw.constant 0 : i0
  %c0_i3 = hw.constant 0 : i3
  %c1_i1 = hw.constant 1 : i1
  %c42_i32 = hw.constant 42 : i32

  // CHECK: %myMemory_rdata = seq.read %myMemory[%c0_i0] rden %true {latency = 0 : i64} : !seq.hlmem<1xi32>
  %myMemory_rdata = seq.read %myMemory[%c0_i0] rden %c1_i1 { latency = 0} : !seq.hlmem<1xi32>
  hw.output
}
