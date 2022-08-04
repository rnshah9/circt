// REQUIRES: capnp
// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s
// RUN: circt-opt %s --esi-create-capnp-schema=schema-file=%t1.capnp --lower-esi-ports --lower-esi-to-hw --export-verilog -verify-diagnostics | FileCheck --check-prefix=COSIM %s
// Disable the SV test : circt-opt %s --lower-esi-ports --lower-esi-to-hw | circt-opt --export-verilog | FileCheck --check-prefix=SV %s

hw.module.extern @Sender() -> (x: !esi.channel<si14>)
hw.module.extern @Reciever(%a: !esi.channel<i32>)
hw.module.extern @ArrReciever(%x: !esi.channel<!hw.array<4xsi64>>)

// CHECK-LABEL: hw.module.extern @Sender() -> (x: !esi.channel<si14>)
// CHECK-LABEL: hw.module.extern @Reciever(%a: !esi.channel<i32>)
// CHECK-LABEL: hw.module.extern @ArrReciever(%x: !esi.channel<!hw.array<4xsi64>>)

hw.module @top(%clk:i1, %rst:i1) -> () {
  hw.instance "recv" @Reciever (a: %cosimRecv: !esi.channel<i32>) -> ()
  // CHECK:  hw.instance "recv" @Reciever(a: %0: !esi.channel<i32>) -> ()

  %send.x = hw.instance "send" @Sender () -> (x: !esi.channel<si14>)
  // CHECK:  %send.x = hw.instance "send" @Sender() -> (x: !esi.channel<si14>)

  %cosimRecv = esi.cosim %clk, %rst, %send.x, 1 {name="TestEP"} : !esi.channel<si14> -> !esi.channel<i32>
  // CHECK:  esi.cosim %clk, %rst, %send.x, 1 {name = "TestEP"} : !esi.channel<si14> -> !esi.channel<i32>

  %send2.x = hw.instance "send2" @Sender () -> (x: !esi.channel<si14>)
  // CHECK:  %send2.x = hw.instance "send2" @Sender() -> (x: !esi.channel<si14>)

  %cosimArrRecv = esi.cosim %clk, %rst, %send2.x, 2 {name="ArrTestEP"} : !esi.channel<si14> -> !esi.channel<!hw.array<4xsi64>>
  // CHECK:  esi.cosim %clk, %rst, %send2.x, 2 {name = "ArrTestEP"} : !esi.channel<si14> -> !esi.channel<!hw.array<4xsi64>>

  hw.instance "arrRecv" @ArrReciever (x: %cosimArrRecv: !esi.channel<!hw.array<4 x si64>>) -> ()

  // Ensure that the file hash is deterministic.
  // COSIM: @0xccf233b58d85e822;
  // COSIM-LABEL: struct Si14 @0x9bd5e507cce05cc1
  // COSIM:         i @0 :Int16;
  // COSIM-LABEL: struct I32 @0x92cd59dfefaacbdb
  // COSIM:         i @0 :UInt32;
  // Ensure the standard RPC interface is tacked on.
  // COSIM: interface CosimDpiServer
  // COSIM: list @0 () -> (ifaces :List(EsiDpiInterfaceDesc));
  // COSIM: open @1 [S, T] (iface :EsiDpiInterfaceDesc) -> (iface :EsiDpiEndpoint(S, T));

  // COSIM-LABEL: hw.module @top(%clk: i1, %rst: i1)
  // COSIM: %TestEP.DataOutValid, %TestEP.DataOut, %TestEP.DataInReady = hw.instance "TestEP" @Cosim_Endpoint<ENDPOINT_ID: i32 = 1, SEND_TYPE_ID: ui64 = 11229133067582987457, SEND_TYPE_SIZE_BITS: i32 = 128, RECV_TYPE_ID: ui64 = 10578209918096690139, RECV_TYPE_SIZE_BITS: i32 = 128>(clk: %clk: i1, rst: %rst: i1, DataOutReady: %{{.+}}: i1, DataInValid: %{{.+}}: i1, DataIn: %{{.+}}: !hw.array<128xi1>) -> (DataOutValid: i1, DataOut: !hw.array<128xi1>, DataInReady: i1)

  // SV: assign _T.valid = TestEP_DataOutValid;
  // SV: assign _T.data = dataSection[6'h0+:32];
  // SV: Reciever recv (
  // SV:   .a ({{.+}}.source)
  // SV: );
  // SV: {{.+}}.ready = TestEP_DataInReady;
  // SV: Sender send (
  // SV:   .x ({{.+}}.sink)
  // SV: );
  // SV: Cosim_Endpoint #(
  // SV:   .ENDPOINT_ID(32'd1),
  // SV:   .RECV_TYPE_ID(64'd10578209918096690139),
  // SV:   .RECV_TYPE_SIZE_BITS(32'd128),
  // SV:   .SEND_TYPE_ID(64'd11229133067582987457),
  // SV:   .SEND_TYPE_SIZE_BITS(32'd128)
  // SV: ) TestEP (
  // SV:   .clk (clk),
  // SV:   .rst (rst),
  // SV:   .DataOutReady ({{.+}}.ready),
  // SV:   .DataInValid ({{.+}}.valid),
  // SV:   .DataIn ({50'h0, {{.+}}.data, {16'h0, 16'h1, 30'h0, 2'h0}})
  // SV:   .DataOutValid (TestEP_DataOutValid),
  // SV:   .DataOut (TestEP_DataOut),
  // SV:   .DataInReady (TestEP_DataInReady)
  // SV: );
  // SV: rootPointer = TestEP_DataOut[{{.+}}+:64];
  // SV: dataSection = TestEP_DataOut[{{.+}}+:64];

  // SV:  assign [[IF1:.+]].ready = ArrTestEP_DataInReady;
  // SV:  Sender send2 (
  // SV:    .x ([[IF1]].sink)
  // SV:  );
  // SV:  Cosim_Endpoint #(
  // SV:    .ENDPOINT_ID(32'd2),
  // SV:    .RECV_TYPE_ID(64'd16793803313215739890),
  // SV:    .RECV_TYPE_SIZE_BITS(32'd384),
  // SV:    .SEND_TYPE_ID(64'd11229133067582987457),
  // SV:    .SEND_TYPE_SIZE_BITS(32'd128)
  // SV:  ) ArrTestEP (
  // SV:    .clk          (clk),
  // SV:    .rst         (rst),
  // SV:    .DataOutReady ({{.+}}.ready),
  // SV:    .DataInValid  ([[IF1]].valid),
  // SV:    .DataIn       ({50'h0, [[IF1]].data, {16'h0, 16'h1, 30'h0, 2'h0}}),
  // SV:    .DataOutValid (ArrTestEP_DataOutValid),
  // SV:    .DataOut      (ArrTestEP_DataOut),
  // SV:    .DataInReady  (ArrTestEP_DataInReady)
  // SV:  );
  // SV:  always @(posedge clk) begin
  // SV:    if (ArrTestEP_DataOutValid) begin
  // SV:      assert({{.+}} == 32'h0);
  // SV:      assert({{.+}} == 16'h0);
  // SV:      assert({{.+}} == 16'h1);
  // SV:      assert({{.+}} == 2'h1);
  // SV:      assert({{.+}} == 3'h5);
  // SV:      assert({{.+}} <= 29'h4);
  // SV:    end
  // SV:  end // always @(posedge)
  // SV:  assign rootPointer{{.*}} = ArrTestEP_DataOut[9'h0+:64];
  // SV:  assign ptrSection = ArrTestEP_DataOut[9'h40+:64];
  // SV:  assign l_ptr = ptrSection[6'h0+:64];
  // SV:  wire [29:0] {{.+}} = l_ptr[6'h2+:30] + 30'h80;
  // SV:  wire [3:0][63:0] {{.+}} = /*cast(bit[3:0][63:0])*/ArrTestEP_DataOut[{{.+}}+:256];

  // The decode part is missing, but ExportVerilog is currently not using the
  // names I'm assigning since it's inlining them. More work on ExportVerilog is
  // necessary to improve it and I'll fill in the rest when this is done.
}
