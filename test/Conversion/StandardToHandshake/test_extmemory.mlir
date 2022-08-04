// RUN: circt-opt -lower-std-to-handshake %s --split-input-file | FileCheck %s

// CHECK-LABEL:   handshake.func @main(
// CHECK-SAME:                         %[[VAL_0:.*]]: memref<4xi32>,
// CHECK-SAME:                         %[[VAL_1:.*]]: none, ...) -> (i32, none) attributes {argNames = ["in0", "inCtrl"], resNames = ["out0", "outCtrl"]} {
// CHECK:           %[[VAL_2:.*]]:2 = extmemory[ld = 1, st = 0] (%[[VAL_0]] : memref<4xi32>) (%[[VAL_3:.*]]) {id = 0 : i32} : (index) -> (i32, none)
// CHECK:           %[[VAL_4:.*]]:2 = fork [2] %[[VAL_1]] : none
// CHECK:           %[[VAL_5:.*]]:2 = fork [2] %[[VAL_4]]#1 : none
// CHECK:           %[[VAL_6:.*]] = join %[[VAL_5]]#1, %[[VAL_2]]#1 : none
// CHECK:           %[[VAL_7:.*]] = constant %[[VAL_5]]#0 {value = 0 : index} : index
// CHECK:           %[[VAL_8:.*]], %[[VAL_3]] = load {{\[}}%[[VAL_7]]] %[[VAL_2]]#0, %[[VAL_4]]#0 : index, i32
// CHECK:           return %[[VAL_8]], %[[VAL_6]] : i32, none
// CHECK:         }
func.func @main(%mem : memref<4xi32>) -> i32 {
  %idx = arith.constant 0 : index
  %0 = memref.load %mem[%idx] : memref<4xi32>
  return %0 : i32
}

// -----

// CHECK-LABEL: handshake.func @no_use(%arg0: memref<4xi32>, %arg1: none, ...) -> none
// CHECK:    extmemory[ld = 0, st = 0] (%arg0 : memref<4xi32>) () {id = 0 : i32} : () -> ()
// CHECK:    return %arg1 : none
// CHECK:  }
func.func @no_use(%mem : memref<4xi32>) {
  return
}
