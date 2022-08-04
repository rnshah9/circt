// RUN: circt-opt %s | circt-opt | FileCheck %s
// RUN: circt-opt %s -test-ssp-roundtrip | circt-opt | FileCheck %s

// 1) tests the plain parser/printer roundtrip.
// 2) roundtrips via the scheduling infra (i.e. populates a `Problem` instance and reconstructs the SSP IR from it.)

// CHECK: ssp.instance "no properties" of "Problem" {
// CHECK:   library {  
// CHECK:     operator_type @NoProps
// CHECK:   }
// CHECK:   graph {
// CHECK:     %[[op_0:.*]] = operation<> @Op0()
// CHECK:     operation<>(%[[op_0]])
// CHECK:     operation<>(@Op0)
// CHECK:     operation<>(%[[op_0]], @Op0)
// CHECK:   }
// CHECK: }
ssp.instance "no properties" of "Problem" {
  library {
    operator_type @NoProps
  }
  graph {
    %0 = operation<> @Op0()
    operation<>(%0)
    operation<>(@Op0)
    operation<>(%0, @Op0)
  }
}

// CHECK: ssp.instance "arbitrary_latencies" of "Problem" {
// CHECK:   library {
// CHECK:     operator_type @unit [latency<1>]
// CHECK:     operator_type @extr [latency<0>]
// CHECK:     operator_type @add [latency<3>]
// CHECK:     operator_type @mult [latency<6>]
// CHECK:     operator_type @sqrt [latency<10>]
// CHECK:   }
// CHECK:   graph {
// CHECK:     %[[op_0:.*]] = operation<@extr>() [t<0>]
// CHECK:     %[[op_1:.*]] = operation<@extr>() [t<10>]
// CHECK:     %[[op_2:.*]] = operation<@mult>(%[[op_0]], %[[op_0]]) [t<20>]
// CHECK:     %[[op_3:.*]] = operation<@mult>(%[[op_1]], %[[op_1]]) [t<30>]
// CHECK:     %[[op_4:.*]] = operation<@add>(%[[op_2]], %[[op_3]]) [t<40>]
// CHECK:     %[[op_5:.*]] = operation<@sqrt>(%[[op_4]]) [t<50>]
// CHECK:     operation<@unit>(%[[op_5]]) [t<60>]
// CHECK:   }
// CHECK: }
ssp.instance "arbitrary_latencies" of "Problem" {
  library {
    operator_type @unit [latency<1>]
    operator_type @extr [latency<0>]
    operator_type @add [latency<3>]
    operator_type @mult [latency<6>]
    operator_type @sqrt [latency<10>]
  }
  graph {
    %0 = operation<@extr>() [t<0>]
    %1 = operation<@extr>() [t<10>]
    %2 = operation<@mult>(%0, %0) [t<20>]
    %3 = operation<@mult>(%1, %1) [t<30>]
    %4 = operation<@add>(%2, %3) [t<40>]
    %5 = operation<@sqrt>(%4) [t<50>]
    operation<@unit>(%5) [t<60>]
  }
}

// CHECK: ssp.instance "self_arc" of "CyclicProblem" [II<3>] {
// CHECK:   library {
// CHECK:     operator_type @unit [latency<1>]
// CHECK:     operator_type @_3 [latency<3>]
// CHECK:   }
// CHECK:   graph {
// CHECK:     %[[op_0:.*]] = operation<@unit>() [t<0>]
// CHECK:     %[[op_1:.*]] = operation<@_3> @self(%[[op_0]], %[[op_0]], @self [dist<1>]) [t<1>]
// CHECK:     operation<@unit>(%[[op_1]]) [t<4>]
// CHECK:   }
// CHECK: }
ssp.instance "self_arc" of "CyclicProblem" [II<3>] {
  library {
    operator_type @unit [latency<1>]
    operator_type @_3 [latency<3>]
  }
  graph {
    %0 = operation<@unit>() [t<0>]
    %1 = operation<@_3> @self(%0, %0, @self [dist<1>]) [t<1>]
    operation<@unit>(%1) [t<4>]
  }
}

// CHECK: ssp.instance "multiple_oprs" of "SharedOperatorsProblem" {
// CHECK:   library {
// CHECK:     operator_type @slowAdd [latency<3>, limit<2>]
// CHECK:     operator_type @fastAdd [latency<1>, limit<1>]
// CHECK:     operator_type @_0 [latency<0>]
// CHECK:     operator_type @_1 [latency<1>]
// CHECK:   }
// CHECK:   graph {
// CHECK:     %[[op_0:.*]] = operation<@slowAdd>() [t<0>]
// CHECK:     %[[op_1:.*]] = operation<@slowAdd>() [t<1>]
// CHECK:     %[[op_2:.*]] = operation<@fastAdd>() [t<0>]
// CHECK:     %[[op_3:.*]] = operation<@slowAdd>() [t<1>]
// CHECK:     %[[op_4:.*]] = operation<@fastAdd>() [t<1>]
// CHECK:     %[[op_5:.*]] = operation<@_0>(%[[op_0]], %[[op_1]], %[[op_2]], %[[op_3]], %[[op_4]]) [t<10>]
// CHECK:     operation<@_1>() [t<10>]
// CHECK:   }
// CHECK: }
ssp.instance "multiple_oprs" of "SharedOperatorsProblem" {
  library {
    operator_type @slowAdd [latency<3>, limit<2>]
    operator_type @fastAdd [latency<1>, limit<1>]
    operator_type @_0 [latency<0>]
    operator_type @_1 [latency<1>]
  }
  graph {
    %0 = operation<@slowAdd>() [t<0>]
    %1 = operation<@slowAdd>() [t<1>]
    %2 = operation<@fastAdd>() [t<0>]
    %3 = operation<@slowAdd>() [t<1>]
    %4 = operation<@fastAdd>() [t<1>]
    %5 = operation<@_0>(%0, %1, %2, %3, %4) [t<10>]
    operation<@_1>() [t<10>]
  }
}

// CHECK: ssp.instance "canis14_fig2" of "ModuloProblem" [II<3>] {
// CHECK:   library {
// CHECK:     operator_type @MemPort [latency<1>, limit<1>]
// CHECK:     operator_type @Add [latency<1>]
// CHECK:     operator_type @Implicit [latency<0>]
// CHECK:   }
// CHECK:   graph {
// CHECK:     %[[op_0:.*]] = operation<@MemPort> @load_A(@store_A [dist<1>]) [t<2>]
// CHECK:     %[[op_1:.*]] = operation<@MemPort> @load_B() [t<0>]
// CHECK:     %[[op_2:.*]] = operation<@Add> @add(%[[op_0]], %[[op_1]]) [t<3>]
// CHECK:     operation<@MemPort> @store_A(%[[op_2]]) [t<4>]
// CHECK:     operation<@Implicit> @last(@store_A) [t<5>]
// CHECK:   }
// CHECK: }
ssp.instance "canis14_fig2" of "ModuloProblem" [II<3>] {
  library {
    operator_type @MemPort [latency<1>, limit<1>]
    operator_type @Add [latency<1>]
    operator_type @Implicit [latency<0>]
  }
  graph {
    %0 = operation<@MemPort> @load_A(@store_A [dist<1>]) [t<2>]
    %1 = operation<@MemPort> @load_B() [t<0>]
    %2 = operation<@Add> @add(%0, %1) [t<3>]
    operation<@MemPort> @store_A(%2) [t<4>]
    operation<@Implicit> @last(@store_A) [t<5>]
  }
}
