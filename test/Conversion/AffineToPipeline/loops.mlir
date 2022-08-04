// RUN: circt-opt -convert-affine-to-pipeline %s | FileCheck %s

// CHECK-LABEL: func @minimal
func.func @minimal(%arg0 : memref<10xindex>) {
  // Setup constants.
  // CHECK: %[[LB:.+]] = arith.constant 0 : [[ITER_TYPE:.+]]
  // CHECK: %[[UB:.+]] = arith.constant [[TRIP_COUNT:.+]] : [[ITER_TYPE]]
  // CHECK: %[[STEP:.+]] = arith.constant 1 : [[ITER_TYPE]]

  // Pipeline header.
  // CHECK: pipeline.while II = 1 trip_count = [[TRIP_COUNT]] iter_args(%[[ITER_ARG:.+]] = %[[LB]]) : ([[ITER_TYPE]]) -> ()

  // Condition block.
  // CHECK: %[[COND_RESULT:.+]] = arith.cmpi ult, %[[ITER_ARG]]
  // CHECK: pipeline.register %[[COND_RESULT]]

  // First stage.
  // CHECK: %[[STAGE0:.+]] = pipeline.stage
  // CHECK: %[[ITER_INC:.+]] = arith.addi %[[ITER_ARG]], %[[STEP]]
  // CHECK: pipeline.register %[[ITER_INC]]

  // Pipeline terminator.
  // CHECK: pipeline.terminator iter_args(%[[STAGE0]]), results()

  affine.for %arg1 = 0 to 10 {
    affine.store %arg1, %arg0[%arg1] : memref<10xindex>
  }

  return
}

// CHECK-LABEL: func @dot
func.func @dot(%arg0: memref<64xi32>, %arg1: memref<64xi32>) -> i32 {
  // Pipeline boilerplate checked above, just check the stages computations.

  // First stage.
  // CHECK: %[[STAGE0:.+]]:3 = pipeline.stage
  // CHECK-DAG: %[[STAGE0_0:.+]] = memref.load %arg0[%arg2]
  // CHECK-DAG: %[[STAGE0_1:.+]] = memref.load %arg1[%arg2]
  // CHECK-DAG: %[[STAGE0_2:.+]] = arith.addi %arg2, %c1
  // CHECK: pipeline.register %[[STAGE0_0]], %[[STAGE0_1]], %[[STAGE0_2]]

  // Second stage.
  // CHECK: %[[STAGE1:.+]] = pipeline.stage
  // CHECK-DAG: %[[STAGE1_0:.+]] = arith.muli %[[STAGE0]]#0, %[[STAGE0]]#1 : i32
  // CHECK: pipeline.register %[[STAGE1_0]]

  // Third stage.
  // CHECK: %[[STAGE2:.+]] = pipeline.stage
  // CHECK-DAG: %[[STAGE2_0:.+]] = arith.addi %arg3, %2
  // CHECK: pipeline.register %[[STAGE2_0]]

  // Pipeline terminator.
  // CHECK: pipeline.terminator iter_args(%[[STAGE0]]#2, %[[STAGE2]]), results(%[[STAGE2]])

  %c0_i32 = arith.constant 0 : i32
  %0 = affine.for %arg2 = 0 to 64 iter_args(%arg3 = %c0_i32) -> (i32) {
    %1 = affine.load %arg0[%arg2] : memref<64xi32>
    %2 = affine.load %arg1[%arg2] : memref<64xi32>
    %3 = arith.muli %1, %2 : i32
    %4 = arith.addi %arg3, %3 : i32
    affine.yield %4 : i32
  }

  return %0 : i32
}

// CHECK-LABEL: func @affine_symbol
#map0 = affine_map<()[s0] -> (s0 + 1)>
func.func @affine_symbol(%arg0: i32) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[CAST:.+]] = arith.index_cast %arg0
  // CHECK-DAG: %[[UB:.+]] = arith.addi %[[CAST]], %[[C1]]
  %0 = arith.index_cast %arg0 : i32 to index
  // CHECK: arith.cmpi ult, %arg1, %[[UB]]
  %1 = affine.for %arg1 = 1 to #map0()[%0] iter_args(%arg2 = %c0_i32) -> (i32) {
    %2 = arith.index_cast %arg1 : index to i32
    %3 = arith.addi %arg2, %2 : i32
    affine.yield %3 : i32
  }
  return %1 : i32
}

// CHECK-LABEL: func @affine_dimension
#map1 = affine_map<(d0)[] -> (d0 + 1)>
func.func @affine_dimension(%arg0: i32) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[CAST:.+]] = arith.index_cast %arg0
  // CHECK-DAG: %[[UB:.+]] = arith.addi %[[CAST]], %[[C1]]
  %0 = arith.index_cast %arg0 : i32 to index
  // CHECK: arith.cmpi ult, %arg1, %[[UB]]
  %1 = affine.for %arg1 = 1 to #map1(%0) iter_args(%arg2 = %c0_i32) -> (i32) {
    %2 = arith.index_cast %arg1 : index to i32
    %3 = arith.addi %arg2, %2 : i32
    affine.yield %3 : i32
  }
  return %1 : i32
}
