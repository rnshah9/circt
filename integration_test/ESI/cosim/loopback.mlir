// REQUIRES: esi-cosim
// RUN: circt-opt %s --esi-connect-services > %t4.mlir
// RUN: circt-opt %t4.mlir --lower-esi-to-physical --lower-esi-ports --lower-esi-to-hw | circt-opt --export-verilog -o %t3.mlir > %t1.sv
// RUN: circt-translate %t4.mlir -export-esi-capnp -verify-diagnostics > %t2.capnp
// RUN: esi-cosim-runner.py --schema %t2.capnp %s %t1.sv
// PY: import loopback as test
// PY: rpc = test.LoopbackTester(rpcschemapath, simhostport)
// PY: rpc.test_two_chan_loopback(25)
// PY: rpc.test_i32(25)
// PY: rpc.test_keytext(25)

hw.module @intLoopback(%clk:i1, %rst:i1) -> () {
  %cosimRecv = esi.cosim %clk, %rst, %bufferedResp, 1 {name="IntTestEP"} : !esi.channel<i32> -> !esi.channel<i32>
  %bufferedResp = esi.buffer %clk, %rst, %cosimRecv {stages=1} : i32
}

!KeyText = !hw.struct<text: !hw.array<6xi14>, key: !hw.array<4xi8>>
hw.module @twoListLoopback(%clk:i1, %rst:i1) -> () {
  %cosim = esi.cosim %clk, %rst, %resp, 2 {name="KeyTextEP"} : !esi.channel<!KeyText> -> !esi.channel<!KeyText>
  %resp = esi.buffer %clk, %rst, %cosim {stages=4} : !KeyText
}

esi.service.decl @HostComms {
  esi.service.to_server @Send : !esi.channel<!esi.any>
  esi.service.to_client @Recv : !esi.channel<i8>
}

hw.module @TwoChanLoopback(%clk: i1) -> () {
  %dataIn = esi.service.req.to_client <@HostComms::@Recv> (["loopback_tohw"]) : !esi.channel<i8>
  esi.service.req.to_server %dataIn -> <@HostComms::@Send> (["loopback_fromhw"]) : !esi.channel<i8>
}

hw.module @top(%clk:i1, %rst:i1) -> () {
  hw.instance "intLoopbackInst" @intLoopback(clk: %clk: i1, rst: %rst: i1) -> ()
  hw.instance "twoListLoopbackInst" @twoListLoopback(clk: %clk: i1, rst: %rst: i1) -> ()

  esi.service.instance @HostComms impl as  "cosim" opts {EpID_start = 10} (%clk, %rst) : (i1, i1) -> ()
  hw.instance "TwoChanLoopback" @TwoChanLoopback(clk: %clk: i1) -> ()
}
