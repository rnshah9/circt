// RUN: circt-opt -canonicalize='top-down=true region-simplify=true' %s | FileCheck %s

// CHECK-LABEL: hw.module @extract_noop(%arg0: i3) -> ("": i3) {
// CHECK-NEXT:    hw.output %arg0

hw.module @extract_noop(%arg0: i3) -> ("": i3) {
  %x = comb.extract %arg0 from 0 : (i3) -> i3
  hw.output %x : i3
}

// Constant Folding

// CHECK-LABEL: hw.module @extract_cstfold() -> (result: i3) {
// CHECK-NEXT:    %c-3_i3 = hw.constant -3 : i3
// CHECK-NEXT:    hw.output  %c-3_i3

hw.module @extract_cstfold() -> (result: i3) {
  %c42_i12 = hw.constant 42 : i12
  %x = comb.extract %c42_i12 from 3 : (i12) -> i3
  hw.output %x : i3
}

// CHECK-LABEL: hw.module @and_cstfold(%arg0: i7) -> (result: i7) {
// CHECK-NEXT:    %c1_i7 = hw.constant 1 : i7
// CHECK-NEXT:    %0 = comb.and %arg0, %c1_i7 : i7
// CHECK-NEXT:    hw.output %0 : i7

hw.module @and_cstfold(%arg0: i7) -> (result: i7) {
  %c11_i7 = hw.constant 11 : i7
  %c5_i7 = hw.constant 5 : i7
  %0 = comb.and %arg0, %c11_i7, %c5_i7 : i7
  hw.output %0 : i7
}

// CHECK-LABEL: hw.module @or_cstfold(%arg0: i7) -> (result: i7) {
// CHECK-NEXT:    %c15_i7 = hw.constant 15 : i7
// CHECK-NEXT:    %0 = comb.or %arg0, %c15_i7 : i7
// CHECK-NEXT:    hw.output %0 : i7

hw.module @or_cstfold(%arg0: i7) -> (result: i7) {
  %c11_i7 = hw.constant 11 : i7
  %c5_i7 = hw.constant 5 : i7
  %0 = comb.or %arg0, %c11_i7, %c5_i7 : i7
  hw.output %0 : i7
}

// CHECK-LABEL: hw.module @xor_cstfold(%arg0: i7) -> (result: i7) {
// CHECK-NEXT:    %c14_i7 = hw.constant 14 : i7
// CHECK-NEXT:    %0 = comb.xor %arg0, %c14_i7 : i7
// CHECK-NEXT:    hw.output %0 : i7

hw.module @xor_cstfold(%arg0: i7) -> (result: i7) {
  %c11_i7 = hw.constant 11 : i7
  %c5_i7 = hw.constant 5 : i7
  %0 = comb.xor %arg0, %c11_i7, %c5_i7 : i7
  hw.output %0 : i7
}

// CHECK-LABEL: hw.module @add_cstfold(%arg0: i7) -> (result: i7) {
// CHECK-NEXT:    %c15_i7 = hw.constant 15 : i7
// CHECK-NEXT:    %0 = comb.add %arg0, %c15_i7 : i7
// CHECK-NEXT:    hw.output %0 : i7
hw.module @add_cstfold(%arg0: i7) -> (result: i7) {
  %c10_i7 = hw.constant 10 : i7
  %c5_i7 = hw.constant 5 : i7
  %0 = comb.add %arg0, %c10_i7, %c5_i7 : i7
  hw.output %0 : i7
}

// CHECK-LABEL: hw.module @mul_cstfold(%arg0: i7) -> (result: i7) {
// CHECK-NEXT:    %c15_i7 = hw.constant 15 : i7
// CHECK-NEXT:    %0 = comb.mul %arg0, %c15_i7 : i7
// CHECK-NEXT:    hw.output %0 : i7
hw.module @mul_cstfold(%arg0: i7) -> (result: i7) {
  %c3_i7 = hw.constant 3 : i7
  %c5_i7 = hw.constant 5 : i7
  %0 = comb.mul %arg0, %c3_i7, %c5_i7 : i7
  hw.output %0 : i7
}

// CHECK-LABEL: hw.module @div_cstfold(%arg0: i7) -> (result: i7, a: i7, b: i7, c: i7) {
// CHECK-NEXT:    %c-3_i7 = hw.constant -3 : i7
// CHECK-NEXT:    %c2_i7 = hw.constant 2 : i7
// CHECK-NEXT:    hw.output %c2_i7, %arg0, %c-3_i7, %arg0 : i7, i7, i7, i7
hw.module @div_cstfold(%arg0: i7) -> (result: i7, a: i7, b: i7, c: i7) {
  %c1_i7 = hw.constant 1 : i7
  %c-3_i7 = hw.constant -3 : i7
  %c5_i7 = hw.constant 5 : i7
  %c10_i7 = hw.constant 10 : i7
  %a = comb.divu %c10_i7, %c5_i7 : i7
  %b = comb.divu %arg0, %c1_i7 : i7

  %c = comb.divs %c10_i7, %c-3_i7 : i7
  %d = comb.divs %arg0, %c1_i7 : i7

  hw.output %a, %b, %c, %d : i7, i7, i7, i7
}

// CHECK-LABEL: hw.module @mod_cstfold(%arg0: i7) -> (result: i7, a: i7, b: i7, c: i7) {
// CHECK-NEXT:    %c0_i7 = hw.constant 0 : i7
// CHECK-NEXT:    %c1_i7 = hw.constant 1 : i7
// CHECK-NEXT:    hw.output %c0_i7, %c0_i7, %c1_i7, %c0_i7 : i7, i7, i7, i7
hw.module @mod_cstfold(%arg0: i7) -> (result: i7, a: i7, b: i7, c: i7) {
  %c1_i7 = hw.constant 1 : i7
  %c-3_i7 = hw.constant -3 : i7
  %c5_i7 = hw.constant 5 : i7
  %c10_i7 = hw.constant 10 : i7
  %a = comb.modu %c10_i7, %c5_i7 : i7
  %b = comb.modu %arg0, %c1_i7 : i7

  %c = comb.mods %c10_i7, %c-3_i7 : i7
  %d = comb.mods %arg0, %c1_i7 : i7

  hw.output %a, %b, %c, %d : i7, i7, i7, i7
}
// CHECK-LABEL: hw.module @variadic_noop(%arg0: i11) -> (result: i11) {
// CHECK-NEXT:    hw.output %arg0

hw.module @variadic_noop(%arg0: i11) -> (result: i11) {
  %0 = comb.and %arg0 : i11
  %1 = comb.or  %0 : i11
  %2 = comb.xor %1 : i11
  %3 = comb.add %2 : i11
  %4 = comb.mul %3 : i11
  hw.output %4 : i11
}

// CHECK-LABEL: hw.module @and_annulment0(%arg0: i11, %arg1: i11) -> (result: i11) {
// CHECK-NEXT:    %c0_i11 = hw.constant 0 : i11
// CHECK-NEXT:    hw.output %c0_i11

hw.module @and_annulment0(%arg0: i11, %arg1: i11) -> (result: i11) {
  %c0_i11 = hw.constant 0 : i11
  %0 = comb.and %arg0, %arg1, %c0_i11 : i11
  hw.output %0 : i11
}

// CHECK-LABEL: hw.module @and_annulment1(%arg0: i7)
// CHECK-NEXT:    %c0_i7 = hw.constant 0 : i7
// CHECK-NEXT:    hw.output %c0_i7

hw.module @and_annulment1(%arg0: i7) -> (result: i7) {
  %c1_i7 = hw.constant 1 : i7
  %c2_i7 = hw.constant 2 : i7
  %c4_i7 = hw.constant 4 : i7
  %0 = comb.and %arg0, %c1_i7, %c2_i7, %c4_i7: i7
  hw.output %0 : i7
}

// CHECK-LABEL: hw.module @or_annulment0(%arg0: i11) -> (result: i11) {
// CHECK-NEXT:    %c-1_i11 = hw.constant -1 : i11
// CHECK-NEXT:    hw.output %c-1_i11

hw.module @or_annulment0(%arg0: i11) -> (result: i11) {
  %c-1_i11 = hw.constant -1 : i11
  %0 = comb.or %arg0, %arg0, %arg0, %c-1_i11 : i11
  hw.output %0 : i11
}

// CHECK-LABEL: hw.module @or_annulment1(%arg0: i3)
// CHECK-NEXT:    %c-1_i3 = hw.constant -1 : i3
// CHECK-NEXT:    hw.output %c-1_i3

hw.module @or_annulment1(%arg0: i3) -> (result: i3) {
  %c1_i3 = hw.constant 1 : i3
  %c2_i3 = hw.constant 2 : i3
  %c4_i3 = hw.constant 4 : i3
  %0 = comb.or %arg0, %c1_i3, %c2_i3, %c4_i3: i3
  hw.output %0 : i3
}

// CHECK-LABEL: hw.module @mul_annulment(%arg0: i11, %arg1: i11, %arg2: i11) -> (result: i11) {
// CHECK-NEXT:    %c0_i11 = hw.constant 0 : i11
// CHECK-NEXT:    hw.output %c0_i11

hw.module @mul_annulment(%arg0: i11, %arg1: i11, %arg2: i11) -> (result: i11) {
  %c0_i11 = hw.constant 0 : i11
  %0 = comb.mul %arg0, %c0_i11, %arg1 : i11
  hw.output %0 : i11
}

// CHECK-LABEL: hw.module @mul_overflow(%arg0: i2) -> (result: i2) {
// CHECK-NEXT:    %c0_i2 = hw.constant 0 : i2
// CHECK-NEXT:    hw.output %c0_i2

hw.module @mul_overflow(%arg0: i2) -> (result: i2) {
  %c2_i2 = hw.constant 2 : i2
  %0 = comb.mul %arg0, %c2_i2, %c2_i2 : i2
  hw.output %0 : i2
}

// Flattening

// CHECK-LABEL: hw.module @and_flatten_in_back(%arg0: i7, %arg1: i7, %arg2: i7) -> (result: i7) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.and %arg0, %arg1, %arg2 : i7
// CHECK-NEXT:    hw.output [[RES]] : i7

hw.module @and_flatten_in_back(%arg0: i7, %arg1: i7, %arg2: i7) -> (result: i7) {
  %and0 = comb.and %arg1, %arg2 : i7
  %0 = comb.and %arg0, %and0 : i7
  hw.output %0 : i7
}

// CHECK-LABEL: hw.module @and_flatten_in_middle(%arg0: i7, %arg1: i7, %arg2: i7, %arg3: i7) -> (result: i7) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.and %arg0, %arg1, %arg2, %arg3 : i7
// CHECK-NEXT:    hw.output [[RES]] : i7

hw.module @and_flatten_in_middle(%arg0: i7, %arg1: i7, %arg2: i7, %arg3: i7) -> (result: i7) {
  %and0 = comb.and %arg1, %arg2 : i7
  %0 = comb.and %arg0, %and0, %arg3 : i7
  hw.output %0 : i7
}

// CHECK-LABEL: hw.module @and_flatten_in_front(%arg0: i7, %arg1: i7, %arg2: i7) -> (result: i7) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.and %arg0, %arg1, %arg2 : i7
// CHECK-NEXT:    hw.output [[RES]] : i7

hw.module @and_flatten_in_front(%arg0: i7, %arg1: i7, %arg2: i7) -> (result: i7) {
  %and0 = comb.and %arg0, %arg1 : i7
  %0 = comb.and %and0, %arg2 : i7
  hw.output %0 : i7
}

// CHECK-LABEL: hw.module @or_flatten_in_back(%arg0: i7, %arg1: i7, %arg2: i7) -> (result: i7) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.or %arg0, %arg1, %arg2 : i7
// CHECK-NEXT:    hw.output [[RES]] : i7

hw.module @or_flatten_in_back(%arg0: i7, %arg1: i7, %arg2: i7) -> (result: i7) {
  %or0 = comb.or %arg1, %arg2 : i7
  %0 = comb.or %arg0, %or0 : i7
  hw.output %0 : i7
}

// CHECK-LABEL: hw.module @or_flatten_keep_root_name(%arg0: i7, %arg1: i7, %arg2: i7) -> (result: i7) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.or %arg0, %arg1, %arg2 {sv.namehint = "waterman"}  : i7
// CHECK-NEXT:    hw.output [[RES]] : i7

hw.module @or_flatten_keep_root_name(%arg0: i7, %arg1: i7, %arg2: i7) -> (result: i7) {
  %or0 = comb.or %arg1, %arg2 : i7
  %0 = comb.or %arg0, %or0 {sv.namehint="waterman"}: i7
  hw.output %0 : i7
}

// CHECK-LABEL: hw.module @or_flatten_in_middle(%arg0: i7, %arg1: i7, %arg2: i7, %arg3: i7) -> (result: i7) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.or %arg0, %arg1, %arg2, %arg3 : i7
// CHECK-NEXT:    hw.output [[RES]] : i7

hw.module @or_flatten_in_middle(%arg0: i7, %arg1: i7, %arg2: i7, %arg3: i7) -> (result: i7) {
  %or0 = comb.or %arg1, %arg2 : i7
  %0 = comb.or %arg0, %or0, %arg3 : i7
  hw.output %0 : i7
}

// CHECK-LABEL: hw.module @or_flatten_in_front(%arg0: i7, %arg1: i7, %arg2: i7) -> (result: i7) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.or %arg0, %arg1, %arg2 : i7
// CHECK-NEXT:    hw.output [[RES]] : i7

hw.module @or_flatten_in_front(%arg0: i7, %arg1: i7, %arg2: i7) -> (result: i7) {
  %or0 = comb.or %arg0, %arg1 : i7
  %0 = comb.or %or0, %arg2 : i7
  hw.output %0 : i7
}

// CHECK-LABEL: hw.module @xor_flatten_in_back(%arg0: i7, %arg1: i7, %arg2: i7) -> (result: i7) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.xor %arg0, %arg1, %arg2 : i7
// CHECK-NEXT:    hw.output [[RES]] : i7

hw.module @xor_flatten_in_back(%arg0: i7, %arg1: i7, %arg2: i7) -> (result: i7) {
  %xor0 = comb.xor %arg1, %arg2 : i7
  %0 = comb.xor %arg0, %xor0 : i7
  hw.output %0 : i7
}

// CHECK-LABEL: hw.module @xor_flatten_in_middle(%arg0: i7, %arg1: i7, %arg2: i7, %arg3: i7) -> (result: i7) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.xor %arg0, %arg1, %arg2, %arg3 : i7
// CHECK-NEXT:    hw.output [[RES]] : i7

hw.module @xor_flatten_in_middle(%arg0: i7, %arg1: i7, %arg2: i7, %arg3: i7) -> (result: i7) {
  %xor0 = comb.xor %arg1, %arg2 : i7
  %0 = comb.xor %arg0, %xor0, %arg3 : i7
  hw.output %0 : i7
}

// CHECK-LABEL: hw.module @xor_flatten_in_front(%arg0: i7, %arg1: i7, %arg2: i7) -> (result: i7) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.xor %arg0, %arg1, %arg2 : i7
// CHECK-NEXT:    hw.output [[RES]] : i7

hw.module @xor_flatten_in_front(%arg0: i7, %arg1: i7, %arg2: i7) -> (result: i7) {
  %xor0 = comb.xor %arg0, %arg1 : i7
  %0 = comb.xor %xor0, %arg2 : i7
  hw.output %0 : i7
}

hw.module @add_flatten_in_back(%arg0: i7, %arg1: i7, %arg2: i7) -> (result: i7) {
  %add0 = comb.add %arg1, %arg2 : i7
  %0 = comb.add %arg0, %add0 : i7
  hw.output %0 : i7
}

// CHECK-LABEL: hw.module @add_flatten_in_middle(%arg0: i7, %arg1: i7, %arg2: i7, %arg3: i7) -> (result: i7) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.add %arg0, %arg1, %arg2, %arg3 : i7
// CHECK-NEXT:    hw.output [[RES]] : i7

hw.module @add_flatten_in_middle(%arg0: i7, %arg1: i7, %arg2: i7, %arg3: i7) -> (result: i7) {
  %add0 = comb.add %arg1, %arg2 : i7
  %0 = comb.add %arg0, %add0, %arg3 : i7
  hw.output %0 : i7
}

// CHECK-LABEL: hw.module @add_flatten_in_front(%arg0: i7, %arg1: i7, %arg2: i7) -> (result: i7) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.add %arg0, %arg1, %arg2 : i7
// CHECK-NEXT:    hw.output [[RES]] : i7

hw.module @add_flatten_in_front(%arg0: i7, %arg1: i7, %arg2: i7) -> (result: i7) {
  %add0 = comb.add %arg0, %arg1 : i7
  %0 = comb.add %add0, %arg2 : i7
  hw.output %0 : i7
}

hw.module @mul_flatten_in_back(%arg0: i7, %arg1: i7, %arg2: i7) -> (result: i7) {
  %mul0 = comb.mul %arg1, %arg2 : i7
  %0 = comb.mul %arg0, %mul0 : i7
  hw.output %0 : i7
}

// CHECK-LABEL: hw.module @mul_flatten_in_middle(%arg0: i7, %arg1: i7, %arg2: i7, %arg3: i7) -> (result: i7) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.mul %arg0, %arg1, %arg2, %arg3 : i7
// CHECK-NEXT:    hw.output [[RES]] : i7

hw.module @mul_flatten_in_middle(%arg0: i7, %arg1: i7, %arg2: i7, %arg3: i7) -> (result: i7) {
  %mul0 = comb.mul %arg1, %arg2 : i7
  %0 = comb.mul %arg0, %mul0, %arg3 : i7
  hw.output %0 : i7
}

// CHECK-LABEL: hw.module @mul_flatten_in_front(%arg0: i7, %arg1: i7, %arg2: i7) -> (result: i7) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.mul %arg0, %arg1, %arg2 : i7
// CHECK-NEXT:    hw.output [[RES]] : i7

hw.module @mul_flatten_in_front(%arg0: i7, %arg1: i7, %arg2: i7) -> (result: i7) {
  %mul0 = comb.mul %arg0, %arg1 : i7
  %0 = comb.mul %mul0, %arg2 : i7
  hw.output %0 : i7
}

// Identities

// CHECK-LABEL: hw.module @and_identity(%arg0: i11, %arg1: i11) -> (result: i11) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.and %arg0, %arg1
// CHECK-NEXT:    hw.output [[RES]]

hw.module @and_identity(%arg0: i11, %arg1: i11) -> (result: i11) {
  %c-1_i11 = hw.constant -1 : i11
  %0 = comb.and %c-1_i11, %arg0, %arg1, %c-1_i11 : i11
  hw.output %0 : i11
}

// CHECK-LABEL: hw.module @or_identity(%arg0: i11, %arg1: i11) -> (result: i11) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.or %arg0, %arg1
// CHECK-NEXT:    hw.output [[RES]]

hw.module @or_identity(%arg0: i11, %arg1: i11) -> (result: i11) {
  %c0_i11 = hw.constant 0 : i11
  %0 = comb.or %arg0, %c0_i11, %arg1 : i11
  hw.output %0 : i11
}

// CHECK-LABEL: hw.module @xor_identity(%arg0: i11, %arg1: i11) -> (result: i11) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.xor %arg1, %arg0
// CHECK-NEXT:    hw.output [[RES]]

hw.module @xor_identity(%arg0: i11, %arg1: i11) -> (result: i11) {
  %c0_i11 = hw.constant 0 : i11
  %0 = comb.xor %c0_i11, %arg1, %arg0 : i11
  hw.output %0 : i11
}

// CHECK-LABEL: hw.module @add_identity(%arg0: i11, %arg1: i11) -> (result: i11) {
// CHECK-NEXT:   [[RES:%[0-9]+]] = comb.add %arg0, %arg1
// CHECK-NEXT:    hw.output [[RES]]

hw.module @add_identity(%arg0: i11, %arg1: i11) -> (result: i11) {
  %c0_i11 = hw.constant 0 : i11
  %0 = comb.add %arg0, %c0_i11, %arg1 : i11
  hw.output %0 : i11
}

// CHECK-LABEL: hw.module @mul_identity(%arg0: i11, %arg1: i11) -> (result: i11) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.mul %arg0, %arg1
// CHECK-NEXT:    hw.output [[RES]]

hw.module @mul_identity(%arg0: i11, %arg1: i11) -> (result: i11) {
  %c1_i11 = hw.constant 1 : i11
  %0 = comb.mul %arg0, %c1_i11, %arg1 : i11
  hw.output %0 : i11
}

// Idempotency

// CHECK-LABEL: hw.module @and_idempotent(%arg0: i11, %arg1: i11) -> (result: i11) {
// CHECK-NEXT:    %c9_i11 = hw.constant 9 : i11
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.and %arg0, %arg1, %c9_i11
// CHECK-NEXT:    hw.output [[RES]]

hw.module @and_idempotent(%arg0: i11, %arg1 : i11) -> (result: i11) {
  %c9_i11 = hw.constant 9 : i11
  %0 = comb.and %arg0, %arg1, %c9_i11, %c9_i11 : i11
  hw.output %0 : i11
}

// CHECK-LABEL: hw.module @or_idempotent(%arg0: i11, %arg1: i11) -> (result: i11) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.or %arg0, %arg1
// CHECK-NEXT:    hw.output [[RES]]

hw.module @or_idempotent(%arg0: i11, %arg1 : i11) -> (result: i11) {
  %0 = comb.or %arg0, %arg1, %arg1, %arg1 : i11
  hw.output %0 : i11
}

// CHECK-LABEL: hw.module @xor_idempotent(%arg0: i11, %arg1: i11, %arg2: i11) -> (result: i11) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.xor %arg0, %arg1
// CHECK-NEXT:    hw.output [[RES]]

hw.module @xor_idempotent(%arg0: i11, %arg1: i11, %arg2: i11) -> (result: i11) {
  %0 = comb.xor %arg0, %arg1, %arg2, %arg2 : i11
  hw.output %0 : i11
}

// CHECK-LABEL: hw.module @xor_idempotent_two_arguments(%arg0: i11) -> (result: i11) {
// CHECK-NEXT:    %c0_i11 = hw.constant 0 : i11
// CHECK-NEXT:    hw.output %c0_i11 : i11

hw.module @xor_idempotent_two_arguments(%arg0: i11) -> (result: i11) {
  %c0_i11 = hw.constant 0 : i11
  %0 = comb.xor %arg0, %arg0 : i11
  hw.output %0 : i11
}

// Add reduction to shift left and multiplication.

// CHECK-LABEL: hw.module @add_reduction1(%arg0: i11, %arg1: i11) -> (result: i11) {
// CHECK-NEXT:    %false = hw.constant false
// CHECK-NEXT:    [[EXTRACT:%[0-9]+]] = comb.extract %arg1 from 0 : (i11) -> i10
// CHECK-NEXT:    [[CONCAT:%[0-9]+]] = comb.concat [[EXTRACT]], %false : i10, i1
// CHECK-NEXT:    hw.output [[CONCAT]]

hw.module @add_reduction1(%arg0: i11, %arg1: i11) -> (result: i11) {
  %0 = comb.add %arg1, %arg1 : i11
  hw.output %0 : i11
}

// CHECK-LABEL: hw.module @add_reduction2(%arg0: i11, %arg1: i11) -> (result: i11) {
// CHECK-NEXT:    %c3_i11 = hw.constant 3 : i11
// CHECK-NEXT:   [[RES:%[0-9]+]] = comb.mul %arg1, %c3_i11
// CHECK-NEXT:    hw.output [[RES]]

hw.module @add_reduction2(%arg0: i11, %arg1: i11) -> (result: i11) {
  %0 = comb.add %arg1, %arg1, %arg1 : i11
  hw.output %0 : i11
}

// CHECK-LABEL: hw.module @add_reduction3(%arg0: i11, %arg1: i11) -> (result: i11) {
// CHECK-NEXT:    %c0_i3 = hw.constant 0 : i3
// CHECK-NEXT:    [[EXTRACT:%[0-9]+]] = comb.extract %arg1 from 0 : (i11) -> i8
// CHECK-NEXT:    [[CONCAT:%[0-9]+]] = comb.concat [[EXTRACT]], %c0_i3 : i8, i3
// CHECK-NEXT:    hw.output [[CONCAT]]

hw.module @add_reduction3(%arg0: i11, %arg1: i11) -> (result: i11) {
  %c7_i11 = hw.constant 7 : i11
  %0 = comb.mul %arg1, %c7_i11 : i11
  %1 = comb.add %arg1, %0 : i11
  hw.output %1 : i11
}

// Multiply reduction to shift left.

// CHECK-LABEL: hw.module @multiply_reduction(%arg0: i11, %arg1: i11) -> (result: i11) {
// CHECK-NEXT:    %false = hw.constant false
// CHECK-NEXT:    [[EXTRACT:%[0-9]+]] = comb.extract %arg1 from 0 : (i11) -> i10
// CHECK-NEXT:    [[CONCAT:%[0-9]+]] = comb.concat [[EXTRACT]], %false : i10, i1
// CHECK-NEXT:    hw.output [[CONCAT]]

hw.module @multiply_reduction(%arg0: i11, %arg1: i11) -> (result: i11) {
  %c2_i11 = hw.constant 2 : i11
  %0 = comb.mul %arg1, %c2_i11 : i11
  hw.output %0 : i11
}

// CHECK-LABEL: hw.module @parity_constant_folding1() -> (result: i1) {
// CHECK-NEXT:  %true = hw.constant true
// CHECK-NEXT:  hw.output %true : i1

hw.module @parity_constant_folding1() -> (result: i1) {
  %c4_i4 = hw.constant 4 : i4
  %0 = comb.parity %c4_i4 : i4
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @parity_constant_folding2() -> (result: i1) {
// CHECK-NEXT:  %false = hw.constant false
// CHECK-NEXT:  hw.output %false : i1
hw.module @parity_constant_folding2() -> (result: i1) {
  %c15_i4 = hw.constant 15 : i4
  %0 = comb.parity %c15_i4 : i4
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @concat_fold_0
// CHECK-NEXT:  %c120_i8 = hw.constant 120 : i8
hw.module @concat_fold_0() -> (result: i8) {
  %c7_i4 = hw.constant 7 : i4
  %c4_i3 = hw.constant 4 : i3
  %false = hw.constant false
  %0 = comb.concat %c7_i4, %c4_i3, %false : i4, i3, i1
  hw.output %0 : i8
}

// CHECK-LABEL: hw.module @concat_fold_1
// CHECK-NEXT:  %0 = comb.concat %arg0, %arg1, %arg2
hw.module @concat_fold_1(%arg0: i4, %arg1: i3, %arg2: i1) -> (result: i8) {
  %a = comb.concat %arg0, %arg1 : i4, i3
  %b = comb.concat %a, %arg2 : i7, i1
  hw.output %b : i8
}

// CHECK-LABEL: hw.module @concat_fold_3
// CHECK-NEXT:    %c60_i7 = hw.constant 60 : i7
// CHECK-NEXT:    %0 = comb.concat %c60_i7, %arg0 : i7, i1
hw.module @concat_fold_3(%arg0: i1) -> (result: i8) {
  %c7_i4 = hw.constant 7 : i4
  %c4_i3 = hw.constant 4 : i3
  %0 = comb.concat %c7_i4, %c4_i3, %arg0 : i4, i3, i1
  hw.output %0 : i8
}

// CHECK-LABEL: hw.module @concat_fold_4
hw.module @concat_fold_4(%arg0: i3) -> (result: i5) {
  // CHECK-NEXT: %0 = comb.extract %arg0 from 2 : (i3) -> i1
  %0 = comb.extract %arg0 from 2 : (i3) -> (i1)
  // CHECK-NEXT: %1 = comb.replicate %0 : (i1) -> i2
  %1 = comb.concat %0, %0, %arg0 : i1, i1, i3
  // CHECK-NEXT: %2 = comb.concat %1, %arg0 : i2, i3
  hw.output %1 : i5
}


// CHECK-LABEL: hw.module @concat_fold_5
// CHECK-NEXT:   %0 = comb.concat %arg0, %arg1 : i3, i3
// CHECK-NEXT:   hw.output %0, %arg0
hw.module @concat_fold_5(%arg0: i3, %arg1: i3) -> (result: i6, a: i3) {
  %0 = comb.extract %arg0 from 2 : (i3) -> (i1)
  %1 = comb.extract %arg0 from 0 : (i3) -> i2
  %2 = comb.concat %0, %1, %arg1 : i1, i2, i3

  %3 = comb.concat %0, %1 : i1, i2
  hw.output %2, %3 : i6, i3
}

// CHECK-LABEL: hw.module @concat_fold6(%arg0: i5, %arg1: i3) -> (result: i4) {
// CHECK-NEXT: %0 = comb.extract %arg0 from 1 : (i5) -> i4
// CHECK-NEXT: hw.output %0 : i4
hw.module @concat_fold6(%arg0: i5, %arg1: i3) -> (result: i4) {
  %0 = comb.extract %arg0 from 3 : (i5) -> i2
  %1 = comb.extract %arg0 from 1 : (i5) -> i2
  %2 = comb.concat %0, %1 : i2, i2
  hw.output %2 : i4
}

// CHECK-LABEL: hw.module @concat_fold7(%arg0: i5) -> (result: i20) {
// CHECK-NEXT: %0 = comb.replicate %arg0 : (i5) -> i20
// CHECK-NEXT: hw.output %0 : i20
hw.module @concat_fold7(%arg0: i5) -> (result: i20) {
  %0 = comb.concat %arg0, %arg0, %arg0, %arg0 : i5, i5, i5, i5
  hw.output %0 : i20
}

// CHECK-LABEL: hw.module @concat_fold8
hw.module @concat_fold8(%arg0: i5, %arg1: i3) -> (r0: i28, r1: i28, r2: i13) {
  %0 = comb.replicate %arg0 : (i5) -> i20

  // CHECK-NEXT: %0 = comb.replicate %arg0 : (i5) -> i25
  // CHECK-NEXT: %1 = comb.concat %arg1, %0 : i3, i25
  %1 = comb.concat %arg1, %arg0, %0 : i3, i5, i20

  // CHECK-NEXT: %2 = comb.replicate %arg0 : (i5) -> i25
  // CHECK-NEXT: %3 = comb.concat %arg1, %2 : i3, i25
  %2 = comb.concat %arg1, %0, %arg0 : i3, i20, i5

  // CHECK-NEXT: %4 = comb.replicate %arg0 : (i5) -> i10
  // CHECK-NEXT: %5 = comb.concat %arg1, %4 : i3, i10
  %3 = comb.concat %arg1, %arg0, %arg0 : i3, i5, i5

  // CHECK-NEXT: hw.output %1, %3, %5
  hw.output %1, %2, %3 : i28, i28, i13
}


// CHECK-LABEL: hw.module @mux_fold0(%arg0: i3, %arg1: i3)
// CHECK-NEXT:    hw.output %arg0 : i3
hw.module @mux_fold0(%arg0: i3, %arg1: i3) -> (result: i3) {
  %c1_i1 = hw.constant 1 : i1
  %0 = comb.mux %c1_i1, %arg0, %arg1 : i3
  hw.output %0 : i3
}

// CHECK-LABEL: hw.module @mux_fold1(%arg0: i1, %arg1: i3)
// CHECK-NEXT:    hw.output %arg1 : i3
hw.module @mux_fold1(%arg0: i1, %arg1: i3) -> (result: i3) {
  %0 = comb.mux %arg0, %arg1, %arg1 : i3
  hw.output %0 : i3
}

// CHECK-LABEL: hw.module @icmp_fold_constants()
// CHECK-NEXT:    %false = hw.constant false
// CHECK-NEXT:    hw.output %false : i1
hw.module @icmp_fold_constants() -> (result: i1) {
  %c2_i2 = hw.constant 2 : i2
  %c3_i2 = hw.constant 3 : i2
  %0 = comb.icmp uge %c2_i2, %c3_i2 : i2
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @icmp_fold_same_operands(%arg0: i2)
// CHECK-NEXT:    %false = hw.constant false
// CHECK-NEXT:    hw.output %false : i1
hw.module @icmp_fold_same_operands(%arg0: i2) -> (result: i1) {
  %0 = comb.icmp ugt %arg0, %arg0 : i2
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @icmp_fold_constant_rhs0(%arg0: i2) -> (result: i1) {
// CHECK-NEXT:    %false = hw.constant false
// CHECK-NEXT:    hw.output %false : i1
hw.module @icmp_fold_constant_rhs0(%arg0: i2) -> (result: i1) {
  %c3_i2 = hw.constant 3 : i2
  %0 = comb.icmp ugt %arg0, %c3_i2 : i2
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @icmp_fold_constant_rhs1(%arg0: i2) -> (result: i1) {
// CHECK-NEXT:    %false = hw.constant false
// CHECK-NEXT:    hw.output %false : i1
hw.module @icmp_fold_constant_rhs1(%arg0: i2) -> (result: i1) {
  %c-2_i2 = hw.constant -2 : i2
  %0 = comb.icmp slt %arg0, %c-2_i2 : i2
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @icmp_fold_constant_lhs0(%arg0: i2) -> (result: i1) {
// CHECK-NEXT:    %true = hw.constant true
// CHECK-NEXT:    hw.output %true : i1
hw.module @icmp_fold_constant_lhs0(%arg0: i2) -> (result: i1) {
  %c3_i2 = hw.constant 3 : i2
  %0 = comb.icmp uge %c3_i2, %arg0 : i2
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @icmp_fold_constant_lhs1(%arg0: i2) -> (result: i1) {
// CHECK-NEXT:    %true = hw.constant true
// CHECK-NEXT:    hw.output %true : i1
hw.module @icmp_fold_constant_lhs1(%arg0: i2) -> (result: i1) {
  %c-2_i2 = hw.constant -2 : i2
  %0 = comb.icmp sle %c-2_i2, %arg0 : i2
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @icmp_canonicalize0(%arg0: i2) -> (result: i1) {
// CHECK-NEXT:    %c-1_i2 = hw.constant -1 : i2
// CHECK-NEXT:    %0 = comb.icmp sgt %arg0, %c-1_i2 : i2
// CHECK-NEXT:    hw.output %0 : i1
hw.module @icmp_canonicalize0(%arg0: i2) -> (result: i1) {
  %c-1_i2 = hw.constant -1 : i2
  %0 = comb.icmp slt %c-1_i2, %arg0 : i2
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @icmp_canonicalize_ne(%arg0: i2) -> (result: i1) {
// CHECK-NEXT:    %c-2_i2 = hw.constant -2 : i2
// CHECK-NEXT:    %0 = comb.icmp ne %arg0, %c-2_i2 : i2
// CHECK-NEXT:    hw.output %0 : i1
hw.module @icmp_canonicalize_ne(%arg0: i2) -> (result: i1) {
  %c-2_i2 = hw.constant -2 : i2
  %0 = comb.icmp slt %c-2_i2, %arg0 : i2
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @icmp_canonicalize_eq(%arg0: i2) -> (result: i1) {
// CHECK-NEXT:    %c-2_i2 = hw.constant -2 : i2
// CHECK-NEXT:    %0 = comb.icmp eq %arg0, %c-2_i2 : i2
// CHECK-NEXT:    hw.output %0 : i1
hw.module @icmp_canonicalize_eq(%arg0: i2) -> (result: i1) {
  %c-1_i2 = hw.constant -1 : i2
  %0 = comb.icmp slt %arg0, %c-1_i2: i2
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @icmp_canonicalize_sgt(%arg0: i2) -> (result: i1) {
// CHECK-NEXT:    %c-1_i2 = hw.constant -1 : i2
// CHECK-NEXT:    %0 = comb.icmp sgt %arg0, %c-1_i2 : i2
// CHECK-NEXT:    hw.output %0 : i1
hw.module @icmp_canonicalize_sgt(%arg0: i2) -> (result: i1) {
  %c0_i2 = hw.constant 0 : i2
  %0 = comb.icmp sle %c0_i2, %arg0 : i2
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @shl_fold1() -> (result: i12) {
// CHECK-NEXT:   %c84_i12 = hw.constant 84 : i12
// CHECK-NEXT:   hw.output %c84_i12 : i12
hw.module @shl_fold1() -> (result: i12) {
  %c42_i12 = hw.constant 42 : i12
  %c1_i12 = hw.constant 1 : i12
  %0 = comb.shl %c42_i12, %c1_i12 : i12
  hw.output %0 : i12
}

// CHECK-LABEL: hw.module @shl_fold2() -> (result: i12) {
// CHECK-NEXT:   %c0_i12 = hw.constant 0 : i12
// CHECK-NEXT:   hw.output %c0_i12 : i12
hw.module @shl_fold2() -> (result: i12) {
  %c1_i12 = hw.constant 1 : i12
  %c10_i12 = hw.constant 12 : i12
  %0 = comb.shl %c1_i12, %c10_i12 : i12
  hw.output %0 : i12
}

// CHECK-LABEL: hw.module @shl_fold3(%arg0: i12) -> (result: i12) {
// CHECK-NEXT:   %c0_i12 = hw.constant 0 : i12
// CHECK-NEXT:   hw.output %c0_i12 : i12
hw.module @shl_fold3(%arg0: i12) -> (result: i12) {
  %c12_i12 = hw.constant 12 : i12
  %0 = comb.shl %arg0, %c12_i12 : i12
  hw.output %0 : i12
}

// CHECK-LABEL: hw.module @shl_fold4(%arg0: i12) -> (result: i12) {
// CHECK-NEXT:   hw.output %arg0 : i12
hw.module @shl_fold4(%arg0: i12) -> (result: i12) {
  %c0_i12 = hw.constant 0 : i12
  %0 = comb.shl %arg0, %c0_i12 : i12
  hw.output %0 : i12
}

// CHECK-LABEL: hw.module @shl_shift_to_extract_and_concat(%arg0: i12) -> (result: i12) {
// CHECK-NEXT:   %c0_i2 = hw.constant 0 : i2
// CHECK-NEXT:   %0 = comb.extract %arg0 from 0 : (i12) -> i10
// CHECK-NEXT:   %1 = comb.concat %0, %c0_i2 : i10, i2
// CHECK-NEXT:   hw.output %1
hw.module @shl_shift_to_extract_and_concat(%arg0: i12) -> (result: i12) {
  %c2_i12 = hw.constant 2 : i12
  %0 = comb.shl %arg0, %c2_i12 : i12
  hw.output %0 : i12
}

// CHECK-LABEL: hw.module @shru_fold1() -> (result: i12) {
// CHECK-NEXT:   %c21_i12 = hw.constant 21 : i12
// CHECK-NEXT:   hw.output %c21_i12 : i12
hw.module @shru_fold1() -> (result: i12) {
  %c42_i12 = hw.constant 42 : i12
  %c1_i12 = hw.constant 1 : i12
  %0 = comb.shru %c42_i12, %c1_i12 : i12
  hw.output %0 : i12
}

// CHECK-LABEL: hw.module @shru_fold2() -> (result: i12) {
// CHECK-NEXT:   %c2047_i12 = hw.constant 2047 : i12
// CHECK-NEXT:   hw.output %c2047_i12 : i12
hw.module @shru_fold2() -> (result: i12) {
  %c-1_i12 = hw.constant -1 : i12
  %c1_i12 = hw.constant 1 : i12
  %0 = comb.shru %c-1_i12, %c1_i12 : i12
  hw.output %0 : i12
}

// CHECK-LABEL: hw.module @shru_fold3(%arg0: i12) -> (result: i12) {
// CHECK-NEXT:   %c0_i12 = hw.constant 0 : i12
// CHECK-NEXT:   hw.output %c0_i12 : i12
hw.module @shru_fold3(%arg0: i12) -> (result: i12) {
  %c12_i12 = hw.constant 12 : i12
  %0 = comb.shru %arg0, %c12_i12 : i12
  hw.output %0 : i12
}

// CHECK-LABEL: hw.module @shru_fold4(%arg0: i12) -> (result: i12) {
// CHECK-NEXT:   hw.output %arg0 : i12
hw.module @shru_fold4(%arg0: i12) -> (result: i12) {
  %c0_i12 = hw.constant 0 : i12
  %0 = comb.shru %arg0, %c0_i12 : i12
  hw.output %0 : i12
}

// CHECK-LABEL: hw.module @shru_shift_to_extract_and_concat(%arg0: i12) -> (result: i12) {
// CHECK-NEXT:   %c0_i2 = hw.constant 0 : i2
// CHECK-NEXT:   %0 = comb.extract %arg0 from 2 : (i12) -> i10
// CHECK-NEXT:   %1 = comb.concat %c0_i2, %0 : i2, i10
// CHECK-NEXT:   hw.output %1
hw.module @shru_shift_to_extract_and_concat(%arg0: i12) -> (result: i12) {
  %c2_i12 = hw.constant 2 : i12
  %0 = comb.shru %arg0, %c2_i12 : i12
  hw.output %0 : i12
}

// CHECK-LABEL: hw.module @shrs_fold1() -> (result: i12) {
// CHECK-NEXT:   %c21_i12 = hw.constant 21 : i12
// CHECK-NEXT:   hw.output %c21_i12 : i12
hw.module @shrs_fold1() -> (result: i12) {
  %c42_i12 = hw.constant 42 : i12
  %c1_i12 = hw.constant 1 : i12
  %0 = comb.shrs %c42_i12, %c1_i12 : i12
  hw.output %0 : i12
}

// CHECK-LABEL: hw.module @shrs_fold2() -> (result: i12) {
// CHECK-NEXT:   %c-3_i12 = hw.constant -3 : i12
// CHECK-NEXT:   hw.output %c-3_i12 : i12
hw.module @shrs_fold2() -> (result: i12) {
  %c-5_i12 = hw.constant -5 : i12
  %c10_i12 = hw.constant 1 : i12
  %0 = comb.shrs %c-5_i12, %c10_i12 : i12
  hw.output %0 : i12
}

// CHECK-LABEL: hw.module @shrs_fold3(%arg0: i12) -> (result: i12) {
// CHECK-NEXT:   hw.output %arg0 : i12
hw.module @shrs_fold3(%arg0: i12) -> (result: i12) {
  %c0_i12 = hw.constant 0 : i12
  %0 = comb.shrs %arg0, %c0_i12 : i12
  hw.output %0 : i12
}

// CHECK-LABEL: hw.module @shru_shift_to_extract_and_concat0(%arg0: i12) -> (result: i12) {
// CHECK-NEXT:   %0 = comb.extract %arg0 from 11 : (i12) -> i1
// CHECK-NEXT:   %1 = comb.replicate %0 : (i1) -> i12
// CHECK-NEXT:   hw.output %1 : i12
hw.module @shru_shift_to_extract_and_concat0(%arg0: i12) -> (result: i12) {
  %c12_i12 = hw.constant 12 : i12
  %0 = comb.shrs %arg0, %c12_i12 : i12
  hw.output %0 : i12
}

// CHECK-LABEL: hw.module @shru_shift_to_extract_and_concat1(%arg0: i12) -> (result: i12) {
// CHECK-NEXT:   %0 = comb.extract %arg0 from 11 : (i12) -> i1
// CHECK-NEXT:   %1 = comb.replicate %0 : (i1) -> i2
// CHECK-NEXT:   %2 = comb.extract %arg0 from 2 : (i12) -> i10
// CHECK-NEXT:   %3 = comb.concat %1, %2 : i2, i10
// CHECK-NEXT:   hw.output %3
hw.module @shru_shift_to_extract_and_concat1(%arg0: i12) -> (result: i12) {
  %c2_i12 = hw.constant 2 : i12
  %0 = comb.shrs %arg0, %c2_i12 : i12
  hw.output %0 : i12
}

// CHECK-LABEL: hw.module @mux_canonicalize0(%a: i1, %b: i1) -> (result: i1) {
// CHECK-NEXT:   %0 = comb.or %a, %b : i1
// CHECK-NEXT: hw.output %0 : i1
hw.module @mux_canonicalize0(%a: i1, %b: i1) -> (result: i1) {
  %true = hw.constant true
  %0 = comb.mux %a, %true, %b : i1
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @mux_canonicalize1(%a: i1, %b: i1) -> (result: i1) {
// CHECK-NEXT:   %0 = comb.and %a, %b : i1
// CHECK-NEXT: hw.output %0 : i1
hw.module @mux_canonicalize1(%a: i1, %b: i1) -> (result: i1) {
  %false = hw.constant false
  %0 = comb.mux %a, %b, %false : i1
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @mux_canonicalize2(%a: i1, %b: i1) -> (result: i1) {
// CHECK-NEXT:   %0 = comb.or %a, %b : i1
// CHECK-NEXT: hw.output %0 : i1
hw.module @mux_canonicalize2(%a: i1, %b: i1) -> (result: i1) {
  %c-1_i1 = hw.constant -1 : i1
  %0 = comb.mux %a, %c-1_i1, %b : i1
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @mux_canonicalize3(%a: i1, %b: i1) -> (result: i1) {
// CHECK-NEXT:   %0 = comb.and %a, %b : i1
// CHECK-NEXT: hw.output %0 : i1
hw.module @mux_canonicalize3(%a: i1, %b: i1) -> (result: i1) {
  %c0_i1 = hw.constant 0 : i1
  %0 = comb.mux %a, %b, %c0_i1 : i1
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @icmp_fold_1bit_eq1(%arg: i1)
// CHECK-NEXT:   %true = hw.constant true
// CHECK-NEXT:   %0 = comb.xor %arg, %true : i1
// CHECK-NEXT:   %1 = comb.xor %arg, %true : i1
// CHECK-NEXT:   hw.output %0, %arg, %arg, %1 : i1, i1, i1, i1
// CHECK-NEXT:   }
hw.module @icmp_fold_1bit_eq1(%arg: i1) -> (result: i1, a: i1, b: i1, c: i1) {
  %zero = hw.constant 0 : i1
  %one = hw.constant 1 : i1
  %0 = comb.icmp eq  %zero, %arg : i1
  %1 = comb.icmp eq   %one, %arg : i1
  %2 = comb.icmp ne  %zero, %arg : i1
  %3 = comb.icmp ne   %one, %arg : i1
  hw.output %0, %1, %2, %3 : i1, i1, i1, i1
}

// CHECK-LABEL:  hw.module @sub_fold1(%arg0: i7) -> (result: i7) {
// CHECK-NEXT:    %c-1_i7 = hw.constant -1 : i7
// CHECK-NEXT:    hw.output %c-1_i7 : i7
hw.module @sub_fold1(%arg0: i7) -> (result: i7) {
  %c11_i7 = hw.constant 11 : i7
  %c5_i7 = hw.constant 12: i7
  %0 = comb.sub %c11_i7, %c5_i7 : i7
  hw.output %0 : i7
}

// CHECK-LABEL: hw.module @sub_fold2(%arg0: i7) -> (result: i7) {
// CHECK-NEXT:    hw.output %arg0 : i7
hw.module @sub_fold2(%arg0: i7) -> (result: i7) {
  %c0_i7 = hw.constant 0 : i7
  %0 = comb.sub %arg0, %c0_i7 : i7
  hw.output %0 : i7
}

// CHECK-LABEL:  hw.module @sub_fold3(%arg0: i7) -> (result: i7) {
// CHECK-NEXT:     %c0_i7 = hw.constant 0 : i7
// CHECK-NEXT:     hw.output %c0_i7 : i7
hw.module @sub_fold3(%arg0: i7) -> (result: i7) {
  %0 = comb.sub %arg0, %arg0 : i7
  hw.output %0 : i7
}

// CHECK-LABEL: issue955
// Incorrect constant folding with >64 bit constants.
hw.module @issue955() -> (result: i100, a: i100) {
  // 1 << 64
  %0 = hw.constant 18446744073709551616 : i100
  %1 = comb.and %0, %0 : i100

  // CHECK-DAG: = hw.constant 18446744073709551616 : i100

  // (1 << 64) + 1
  %2 = hw.constant 18446744073709551617 : i100
  %3 = comb.and %2, %2 : i100

  // CHECK-DAG: = hw.constant 18446744073709551617 : i100
  hw.output %1, %3 : i100, i100
}

// CHECK-LABEL: replicate_and_one_bit
hw.module @replicate_and_one_bit(%bit: i1) -> (a: i65, b: i8, c: i8) {
  %c-18446744073709551616_i65 = hw.constant -18446744073709551616 : i65
  %0 = comb.replicate %bit : (i1) -> i65
  %1 = comb.and %0, %c-18446744073709551616_i65 : i65
  // CHECK: [[A:%[0-9]+]] = comb.concat %bit, %c0_i64 : i1, i64

  %c4_i8 = hw.constant 4 : i8
  %2 = comb.replicate %bit : (i1) -> i8
  %3 = comb.and %2, %c4_i8 : i8
  // CHECK: [[B:%[0-9]+]] = comb.concat %c0_i5, %bit, %c0_i2 : i5, i1, i2

  %c1_i8 = hw.constant 1 : i8
  %4 = comb.and %2, %c1_i8 : i8
  // CHECK: [[C:%[0-9]+]] = comb.concat %c0_i7, %bit : i7, i1

  // CHECK: hw.output [[A]], [[B]], [[C]] :
  hw.output %1, %3, %4 : i65, i8, i8
}

// CHECK-LABEL: hw.module @wire0()
// CHECK-NEXT:    hw.output
hw.module @wire0() {
  %0 = sv.wire : !hw.inout<i1>
  hw.output
}

// CHECK-LABEL: hw.module @wire1()
// CHECK-NEXT:    hw.output
hw.module @wire1() {
  %0 = sv.wire : !hw.inout<i1>
  %1 = sv.read_inout %0 : !hw.inout<i1>
  hw.output
}

// CHECK-LABEL: hw.module @wire2()
// CHECK-NEXT:    hw.output
hw.module @wire2() {
  %c = hw.constant 1 : i1
  %0 = sv.wire : !hw.inout<i1>
  sv.assign %0, %c : i1
  hw.output
}

// CHECK-LABEL: hw.module @wire3()
// CHECK-NEXT:    hw.output
hw.module @wire3() {
  %c = hw.constant 1 : i1
  %0 = sv.wire : !hw.inout<i1>
  %1 = sv.read_inout %0 : !hw.inout<i1>
  sv.assign %0, %c :i1
  hw.output
}

// CHECK-LABEL: hw.module @wire4()
// CHECK-NEXT:   %true = hw.constant true
// CHECK-NEXT:   %0 = sv.wire sym @symName : !hw.inout<i1>
// CHECK-NEXT:   %1 = sv.read_inout %0 : !hw.inout<i1>
// CHECK-NEXT:   sv.assign %0, %true : i1
// CHECK-NEXT:   hw.output %1 : i1
hw.module @wire4() -> (result: i1) {
  %true = hw.constant true
  %0 = sv.wire sym @symName : !hw.inout<i1>
  %1 = sv.read_inout %0 : !hw.inout<i1>
  sv.assign %0, %true : i1
  hw.output %1 : i1
}

// CHECK-LABEL: hw.module @wire4_1()
// CHECK-NEXT:   %true = hw.constant true
// CHECK-NEXT:   hw.output %true : i1
hw.module @wire4_1() -> (result: i1) {
  %true = hw.constant true
  %0 = sv.wire : !hw.inout<i1>
  %1 = sv.read_inout %0 : !hw.inout<i1>
  sv.assign %0, %true : i1
  hw.output %1 : i1
}

// CHECK-LABEL: hw.module @wire5()
// CHECK-NEXT:   %wire_with_name = sv.wire sym @wire_with_name : !hw.inout<i1>
// CHECK-NEXT:   hw.output
hw.module @wire5() -> () {
  %wire_with_name = sv.wire sym @wire_with_name : !hw.inout<i1>
  hw.output
}

// CHECK-LABEL: hw.module @replicate
hw.module @replicate(%arg0: i7) -> (r1: i9, r2: i7) {
  %c2 = hw.constant 2 : i3
  %r1 = comb.replicate %c2 : (i3) -> i9

// CHECK-NEXT: %c146_i9 = hw.constant 146 : i9
  %r2 = comb.replicate %arg0 : (i7) -> i7

// CHECK-NEXT:  hw.output %c146_i9, %arg0
  hw.output %r1, %r2 : i9, i7
}

// CHECK-LABEL: hw.module @bitcast_canonicalization
hw.module @bitcast_canonicalization(%arg0: i4) -> (r1: i4, r2: !hw.array<2xi2>) {
  %id = hw.bitcast %arg0 : (i4) -> i4
  %a = hw.bitcast %arg0 : (i4) -> !hw.struct<a:i2, b:i2>
  %b = hw.bitcast %a : (!hw.struct<a:i2, b:i2>) -> !hw.array<2xi2>
  // CHECK-NEXT: %0 = hw.bitcast %arg0 : (i4) -> !hw.array<2xi2>
  // CHECK-NEXT: hw.output %arg0, %0
  hw.output %id, %b : i4, !hw.array<2xi2>
}

// CHECK-LABEL: hw.module @array_get1(%a0: i3, %a1: i3, %a2: i3) -> (r0: i3)
// CHECK-NEXT:    hw.output %a0 : i3
hw.module @array_get1(%a0: i3, %a1: i3, %a2: i3) -> (r0: i3) {
  %c0 = hw.constant 0 : i2
  %arr = hw.array_create %a2, %a1, %a0 : i3
  %r0 = hw.array_get %arr[%c0] : !hw.array<3xi3>
  hw.output %r0 : i3
}

// CHECK-LABEL: hw.module @struct_extract1(%a0: i3, %a1: i5) -> (r0: i3)
// CHECK-NEXT:    hw.output %a0 : i3
hw.module @struct_extract1(%a0: i3, %a1: i5) -> (r0: i3) {
  %s = hw.struct_create (%a0, %a1) : !hw.struct<foo: i3, bar: i5>
  %r0 = hw.struct_extract %s["foo"] : !hw.struct<foo: i3, bar: i5>
  hw.output %r0 : i3
}

// Ensure that canonicalizer works with hw.enum.constant.

hw.module @enum_constant() -> () {
  %0 = hw.enum.constant A : !hw.enum<A, B, C>
}


// == Begin: test cases from LowerToHW ==

// CHECK-LABEL:  hw.module @instance_ooo(%arg0: i2, %arg1: i2, %arg2: i3) -> (out0: i8) {
// CHECK-NEXT:    %false = hw.constant false
// CHECK-NEXT:    %myext.out = hw.instance "myext" @MyParameterizedExtModule(in: %3: i1) -> (out: i8)  {oldParameters = {DEFAULT = 0 : i64, DEPTH = 3.242000e+01 : f64, FORMAT = "xyz_timeout=%d\0A", WIDTH = 32 : i8}}
// CHECK-NEXT:    %0 = comb.concat %false, %arg0 : i1, i2
// CHECK-NEXT:    %1 = comb.concat %false, %arg0 : i1, i2
// CHECK-NEXT:    %2 = comb.add %0, %1 : i3
// CHECK-NEXT:    %3 = comb.icmp eq %2, %arg2 : i3
// CHECK-NEXT:    hw.output %myext.out : i8

hw.module @instance_ooo(%arg0: i2, %arg1: i2, %arg2: i3) -> (out0: i8) {
  %false = hw.constant false
    %.in.wire = sv.wire  : !hw.inout<i1>
    %0 = sv.read_inout %.in.wire : !hw.inout<i1>
    %myext.out = hw.instance "myext" @MyParameterizedExtModule(in: %0: i1) -> (out: i8)  {oldParameters = {DEFAULT = 0 : i64, DEPTH = 3.242000e+01 : f64, FORMAT = "xyz_timeout=%d\0A", WIDTH = 32 : i8}}
    %1 = comb.concat %false, %arg0 : i1, i2
    %2 = comb.concat %false, %arg0 : i1, i2
    %3 = comb.add %1, %2 : i3
    %4 = comb.icmp eq %3, %arg2 : i3
    sv.assign %.in.wire, %4 : i1
    hw.output %myext.out : i8
}

// CHECK-LABEL: hw.module @TestInstance(%u2: i2, %s8: i8, %clock: i1, %reset: i1) {
// CHECK-NEXT:   %c0_i2 = hw.constant 0 : i2
// CHECK-NEXT:   hw.instance "xyz" @Simple(in1: %0: i4, in2: %u2: i2, in3: %s8: i8)
// CHECK-NEXT:   %0 = comb.concat %c0_i2, %u2 : i2, i2
// CHECK-NEXT:   %myext.out = hw.instance "myext" @MyParameterizedExtModule(in: %reset: i1) -> (out: i8)  {oldParameters = {DEFAULT = 0 : i64, DEPTH = 3.242000e+01 : f64, FORMAT = "xyz_timeout=%d\0A", WIDTH = 32 : i8}}
// CHECK-NEXT:   hw.output
hw.module.extern @MyParameterizedExtModule(%in: i1) -> (out: i8) attributes {verilogName = "name_thing"}
hw.module.extern @Simple(%in1: i4, %in2: i2, %in3: i8)
hw.module @TestInstance(%u2: i2, %s8: i8, %clock: i1, %reset: i1) {
  %c0_i2 = hw.constant 0 : i2
  %.in1.wire = sv.wire  : !hw.inout<i4>
  %0 = sv.read_inout %.in1.wire : !hw.inout<i4>
  %.in2.wire = sv.wire  : !hw.inout<i2>
  %1 = sv.read_inout %.in2.wire : !hw.inout<i2>
  %.in3.wire = sv.wire  : !hw.inout<i8>
  %2 = sv.read_inout %.in3.wire : !hw.inout<i8>
  hw.instance "xyz" @Simple(in1: %0: i4, in2: %1: i2, in3: %2: i8) -> ()
  %3 = comb.concat %c0_i2, %u2 : i2, i2
  sv.assign %.in1.wire, %3 : i4
  sv.assign %.in2.wire, %u2 : i2
  sv.assign %.in3.wire, %s8 : i8
  %.in.wire = sv.wire  : !hw.inout<i1>
  %4 = sv.read_inout %.in.wire : !hw.inout<i1>
  %myext.out = hw.instance "myext" @MyParameterizedExtModule(in: %4: i1) -> (out: i8)  {oldParameters = {DEFAULT = 0 : i64, DEPTH = 3.242000e+01 : f64, FORMAT = "xyz_timeout=%d\0A", WIDTH = 32 : i8}}
  sv.assign %.in.wire, %reset : i1
  hw.output
}

// CHECK-LABEL:  hw.module @instance_cyclic(%arg0: i2, %arg1: i2) {
// CHECK-NEXT:    %myext.out = hw.instance "myext" @MyParameterizedExtModule(in: %0: i1)
// CHECK-NEXT:    %0 = comb.extract %myext.out from 2 : (i8) -> i1
// CHECK-NEXT:    hw.output
// CHECK-NEXT:  }
hw.module @instance_cyclic(%arg0: i2, %arg1: i2) {
  %.in.wire = sv.wire  : !hw.inout<i1>
  %0 = sv.read_inout %.in.wire : !hw.inout<i1>
  %myext.out = hw.instance "myext" @MyParameterizedExtModule(in: %0: i1) -> (out: i8)  {oldParameters = {DEFAULT = 0 : i64, DEPTH = 3.242000e+01 : f64, FORMAT = "xyz_timeout=%d\0A", WIDTH = 32 : i8}}
  %1 = comb.extract %myext.out from 2 : (i8) -> i1
  sv.assign %.in.wire, %1 : i1
  hw.output
}

  hw.module.extern @ZeroWidthPorts(%inA: i4) -> (outa: i4)
// CHECK-LABEL:  hw.module @ZeroWidthInstance(%iA: i4) -> (oA: i4) {
// CHECK-NEXT:    %myinst.outa = hw.instance "myinst" @ZeroWidthPorts(inA: %iA: i4) -> (outa: i4)
// CHECK-NEXT:    hw.output %myinst.outa : i4
// CHECK-NEXT:  }
hw.module @ZeroWidthInstance(%iA: i4) -> (oA: i4) {
  %.inA.wire = sv.wire  : !hw.inout<i4>
  %0 = sv.read_inout %.inA.wire : !hw.inout<i4>
  %myinst.outa = hw.instance "myinst" @ZeroWidthPorts(inA: %0: i4) -> (outa: i4)
  sv.assign %.inA.wire, %iA : i4
  hw.output %myinst.outa : i4
}

// CHECK-LABEL:  hw.module @unintializedWire(%clock1: i1, %clock2: i1, %inpred: i1, %indata: i42) -> (result: i42, result2: i42) {
// CHECK-NEXT:    %c0_i4 = hw.constant 0 : i4
// CHECK-NEXT:    %true = hw.constant true
// CHECK-NEXT:    %_M.ro_data_0, %_M.rw_data_0 = hw.instance "_M" @FIRRTLMem_1_1_1_42_12_0_1_0(ro_clock_0: %clock1: i1, ro_en_0: %true: i1, ro_addr_0: %c0_i4: i4) -> (ro_data_0: i42, rw_data_0: i42)
// CHECK-NEXT:    hw.output %_M.ro_data_0, %_M.rw_data_0 : i42, i42
// CHECK-NEXT:  }
hw.module.extern @FIRRTLMem_1_1_1_42_12_0_1_0(%ro_clock_0: i1, %ro_en_0: i1, %ro_addr_0: i4) -> (ro_data_0: i42, rw_data_0: i42)
hw.module @unintializedWire(%clock1: i1, %clock2: i1, %inpred: i1, %indata: i42) -> (result: i42, result2: i42) {
  %c0_i3 = hw.constant 0 : i3
  %true = hw.constant true
  %false = hw.constant false
  %.read.clk.wire = sv.wire  : !hw.inout<i1>
  %0 = sv.read_inout %.read.clk.wire : !hw.inout<i1>
  %.read.en.wire = sv.wire  : !hw.inout<i1>
  %1 = sv.read_inout %.read.en.wire : !hw.inout<i1>
  %.read.addr.wire = sv.wire  : !hw.inout<i4>
  %2 = sv.read_inout %.read.addr.wire : !hw.inout<i4>
  %.rw.clk.wire = sv.wire  : !hw.inout<i1>
  %3 = sv.read_inout %.rw.clk.wire : !hw.inout<i1>
  %.rw.en.wire = sv.wire  : !hw.inout<i1>
  %4 = sv.read_inout %.rw.en.wire : !hw.inout<i1>
  %.rw.addr.wire = sv.wire  : !hw.inout<i4>
  %5 = sv.read_inout %.rw.addr.wire : !hw.inout<i4>
  %.rw.wmode.wire = sv.wire  : !hw.inout<i1>
  %6 = sv.read_inout %.rw.wmode.wire : !hw.inout<i1>
  %.rw.wmask.wire = sv.wire  : !hw.inout<i1>
  %7 = sv.read_inout %.rw.wmask.wire : !hw.inout<i1>
  %.rw.wdata.wire = sv.wire : !hw.inout<i42>
  %8 = sv.read_inout %.rw.wdata.wire : !hw.inout<i42>
  %.write.clk.wire = sv.wire  : !hw.inout<i1>
  %9 = sv.read_inout %.write.clk.wire : !hw.inout<i1>
  %.write.en.wire = sv.wire  : !hw.inout<i1>
  %10 = sv.read_inout %.write.en.wire : !hw.inout<i1>
  %.write.addr.wire = sv.wire  : !hw.inout<i4>
  %11 = sv.read_inout %.write.addr.wire : !hw.inout<i4>
  %.write.mask.wire = sv.wire  : !hw.inout<i1>
  %12 = sv.read_inout %.write.mask.wire : !hw.inout<i1>
  %.write.data.wire = sv.wire  : !hw.inout<i42>
  %13 = sv.read_inout %.write.data.wire : !hw.inout<i42>
  %_M.ro_data_0, %_M.rw_rdata_0 = hw.instance "_M" @FIRRTLMem_1_1_1_42_12_0_1_0
     (ro_clock_0: %0: i1, ro_en_0: %1: i1, ro_addr_0: %2: i4) -> (ro_data_0: i42, rw_data_0: i42)

  %14 = sv.read_inout %.read.addr.wire : !hw.inout<i4>
  %c0_i4 = hw.constant 0 : i4
  sv.assign %.read.addr.wire, %c0_i4 : i4
  %15 = sv.read_inout %.read.en.wire : !hw.inout<i1>
  sv.assign %.read.en.wire, %true : i1
  %16 = sv.read_inout %.read.clk.wire : !hw.inout<i1>
  sv.assign %.read.clk.wire, %clock1 : i1
  %17 = sv.read_inout %.rw.addr.wire : !hw.inout<i4>
  %c0_i4_0 = hw.constant 0 : i4
  sv.assign %.rw.addr.wire, %c0_i4_0 : i4
  %18 = sv.read_inout %.rw.en.wire : !hw.inout<i1>
  sv.assign %.rw.en.wire, %true : i1
  %19 = sv.read_inout %.rw.clk.wire : !hw.inout<i1>
  sv.assign %.rw.clk.wire, %clock1 : i1
  %20 = sv.read_inout %.rw.wmask.wire : !hw.inout<i1>
  sv.assign %.rw.wmask.wire, %true : i1
  %21 = sv.read_inout %.rw.wmode.wire : !hw.inout<i1>
  sv.assign %.rw.wmode.wire, %true : i1
  %22 = sv.read_inout %.write.addr.wire : !hw.inout<i4>
  %c0_i4_1 = hw.constant 0 : i4
  sv.assign %.write.addr.wire, %c0_i4_1 : i4
  %23 = sv.read_inout %.write.en.wire : !hw.inout<i1>
  sv.assign %.write.en.wire, %inpred : i1
  %24 = sv.read_inout %.write.clk.wire : !hw.inout<i1>
  sv.assign %.write.clk.wire, %clock2 : i1
  %25 = sv.read_inout %.write.data.wire : !hw.inout<i42>
  sv.assign %.write.data.wire, %indata : i42
  %26 = sv.read_inout %.write.mask.wire : !hw.inout<i1>
  sv.assign %.write.mask.wire, %true : i1
  hw.output %_M.ro_data_0, %_M.rw_rdata_0 : i42, i42
}

// CHECK-LABEL: hw.module @uninitializedWireAggregate
hw.module @uninitializedWireAggregate() -> (result1: !hw.struct<a: i1, b: i1>,
                                            result2: !hw.struct<a: i1, b: !hw.array<10x!hw.struct<a: i1, b: i1>>>)
{
  %0 = sv.wire : !hw.inout<!hw.struct<a: i1, b: i1>>
  %1 = sv.read_inout %0 : !hw.inout<!hw.struct<a: i1, b: i1>>
  %2 = sv.wire : !hw.inout<!hw.struct<a: i1, b: !hw.array<10x!hw.struct<a: i1, b: i1>>>>
  %3 = sv.read_inout %2 :  !hw.inout<!hw.struct<a: i1, b: !hw.array<10x!hw.struct<a: i1, b: i1>>>>

  hw.output %1, %3 : !hw.struct<a: i1, b: i1>, !hw.struct<a: i1, b: !hw.array<10x!hw.struct<a: i1, b: i1>>>
  // CHECK-NEXT: %x_i2 = sv.constantX : !hw.struct<a: i1, b: i1>
  // CHECK-NEXT: %x_i21 = sv.constantX : !hw.struct<a: i1, b: !hw.array<10xstruct<a: i1, b: i1>>>
  // CHECK-NEXT: hw.output %x_i2, %x_i21
}

// CHECK-LABEL:  hw.module @IncompleteRead(%clock1: i1) {
// CHECK-NEXT:    %c0_i4 = hw.constant 0 : i4
// CHECK-NEXT:    %true = hw.constant true
// CHECK-NEXT:    %_M.ro_data_0 = hw.instance "_M" @FIRRTLMem_1_0_0_42_12_0_1_0(ro_clock_0: %clock1: i1, ro_en_0: %true: i1, ro_addr_0: %c0_i4: i4) -> (ro_data_0: i42)
// CHECK-NEXT:    hw.output
// CHECK-NEXT:  }
hw.module.extern @FIRRTLMem_1_0_0_42_12_0_1_0(%ro_clock_0: i1, %ro_en_0: i1, %ro_addr_0: i4) -> (ro_data_0: i42)
hw.module @IncompleteRead(%clock1: i1) {
  %c0_i3 = hw.constant 0 : i3
  %true = hw.constant true
  %false = hw.constant false
  %.read.clk.wire = sv.wire  : !hw.inout<i1>
  %0 = sv.read_inout %.read.clk.wire : !hw.inout<i1>
  %.read.en.wire = sv.wire  : !hw.inout<i1>
  %1 = sv.read_inout %.read.en.wire : !hw.inout<i1>
  %.read.addr.wire = sv.wire  : !hw.inout<i4>
  %2 = sv.read_inout %.read.addr.wire : !hw.inout<i4>
  %_M.ro_data_0 = hw.instance "_M" @FIRRTLMem_1_0_0_42_12_0_1_0(ro_clock_0: %0: i1, ro_en_0: %1: i1, ro_addr_0: %2: i4) -> (ro_data_0: i42)
  %3 = sv.read_inout %.read.addr.wire : !hw.inout<i4>
  %c0_i4 = hw.constant 0 : i4
  sv.assign %.read.addr.wire, %c0_i4 : i4
  %4 = sv.read_inout %.read.en.wire : !hw.inout<i1>
  sv.assign %.read.en.wire, %true : i1
  %5 = sv.read_inout %.read.clk.wire : !hw.inout<i1>
  sv.assign %.read.clk.wire, %clock1 : i1
  hw.output
}

// CHECK-LABEL:  hw.module @foo() {
// CHECK-NEXT:    %io_cpu_flush.wire = sv.wire sym @io_cpu_flush.wire  : !hw.inout<i1>
// CHECK-NEXT:    hw.instance "fetch" @bar(io_cpu_flush: %0: i1) -> ()
// CHECK-NEXT:    %0 = sv.read_inout %io_cpu_flush.wire : !hw.inout<i1>
// CHECK-NEXT:    hw.output
hw.module.extern @bar(%io_cpu_flush: i1)
hw.module @foo() {
  %io_cpu_flush.wire = sv.wire sym @io_cpu_flush.wire  : !hw.inout<i1>
  %.io_cpu_flush.wire = sv.wire  : !hw.inout<i1>
  %0 = sv.read_inout %.io_cpu_flush.wire : !hw.inout<i1>
  hw.instance "fetch" @bar(io_cpu_flush: %0: i1) -> ()
  %1 = sv.read_inout %io_cpu_flush.wire : !hw.inout<i1>
  sv.assign %.io_cpu_flush.wire, %1 : i1
  %2 = sv.read_inout %io_cpu_flush.wire : !hw.inout<i1>
  %hits_1_7 = sv.wire  : !hw.inout<i1>
  sv.assign %hits_1_7, %2 : i1
  hw.output
}

// CHECK-LABEL:  hw.module @MemDepth1(%clock: i1, %en: i1, %addr: i1) -> (data: i32) {
// CHECK-NEXT:    %mem0.0 = hw.instance "mem0" @FIRRTLMem_1_0_0_32_1_0_1_1(x: %clock: i1, y: %en: i1, z: %addr: i1) -> ("": i32)
// CHECK-NEXT:    hw.output %mem0.0 : i32
// CHECK-NEXT:  }
hw.module.extern @FIRRTLMem_1_0_0_32_1_0_1_1(%x: i1, %y: i1, %z: i1) -> ("": i32)
hw.module @MemDepth1(%clock: i1, %en: i1, %addr: i1) -> (data: i32) {
  %.load0.clk.wire = sv.wire  : !hw.inout<i1>
  %0 = sv.read_inout %.load0.clk.wire : !hw.inout<i1>
  %.load0.en.wire = sv.wire  : !hw.inout<i1>
  %1 = sv.read_inout %.load0.en.wire : !hw.inout<i1>
  %.load0.addr.wire = sv.wire  : !hw.inout<i1>
  %2 = sv.read_inout %.load0.addr.wire : !hw.inout<i1>
  %mem0.ro_data_0 = hw.instance "mem0" @FIRRTLMem_1_0_0_32_1_0_1_1("x": %0: i1, "y": %1: i1, "z": %2: i1) -> ("": i32)
  %3 = sv.read_inout %.load0.clk.wire : !hw.inout<i1>
  sv.assign %.load0.clk.wire, %clock : i1
  %4 = sv.read_inout %.load0.addr.wire : !hw.inout<i1>
  sv.assign %.load0.addr.wire, %addr : i1
  %5 = sv.read_inout %.load0.en.wire : !hw.inout<i1>
  sv.assign %.load0.en.wire, %en : i1
  hw.output %mem0.ro_data_0 : i32
}

// == End: test cases from LowerToHW ==

// CHECK-LABEL: hw.module @ExtractOfInject
hw.module @ExtractOfInject(%a: !hw.struct<a: i1>, %v: i1) -> (result: i1) {
  %b = hw.struct_inject %a["a"], %v : !hw.struct<a: i1>
  %c = hw.struct_extract %b["a"] : !hw.struct<a: i1>
  // CHECK: hw.output %v
  hw.output %c : i1
}

// CHECK-LABEL: hw.module @ExtractCycle
hw.module @ExtractCycle(%a: !hw.struct<a: i1>, %v: i1) -> (result: i1) {
  %b = hw.struct_inject %b["a"], %v : !hw.struct<a: i1>
  %c = hw.struct_extract %b["a"] : !hw.struct<a: i1>
  // CHECK: hw.output %v
  hw.output %c : i1
}

// CHECK-LABEL: hw.module @ExtractOfUnrelatedInject
hw.module @ExtractOfUnrelatedInject(%a: !hw.struct<a: i1, b: i1>, %v: i1) -> (result: i1) {
  %b = hw.struct_inject %a["b"], %v : !hw.struct<a: i1, b: i1>
  %c = hw.struct_extract %b["a"] : !hw.struct<a: i1, b: i1>
  // CHECK: [[STRUCT:%.+]] = hw.struct_extract %a["a"] : !hw.struct<a: i1, b: i1>
  // CHECK-NEXT: hw.output [[STRUCT]]
  hw.output %c : i1
}

// CHECK-LABEL: hw.module @InjectOnInject
hw.module @InjectOnInject(%a: !hw.struct<a: i1>, %p: i1, %q: i1) -> (result: !hw.struct<a: i1>) {
  %b = hw.struct_inject %a["a"], %p : !hw.struct<a: i1>
  %c = hw.struct_inject %b["a"], %q : !hw.struct<a: i1>
  // CHECK: [[STRUCT:%.+]] = hw.struct_inject %a["a"], %q
  // CHECK-NEXT: hw.output [[STRUCT]]
  hw.output %c : !hw.struct<a: i1>
}

