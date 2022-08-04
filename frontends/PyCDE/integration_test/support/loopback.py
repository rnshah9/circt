#!/usr/bin/python3

import esi_cosim
import random


class LoopbackTester(esi_cosim.CosimBase):
  """Provides methods to test the loopback simulations."""

  def test_two_chan_loopback(self, num_msgs):
    to_hw = self.openEP(1001, sendType=self.schema.I1, recvType=self.schema.I32)
    from_hw = self.openEP(1002,
                          sendType=self.schema.I32,
                          recvType=self.schema.I1)
    for _ in range(num_msgs):
      data = random.randint(0, 2**32 - 1)
      print(f"Sending {data}")
      to_hw.send(self.schema.I32.new_message(i=data))
      result = self.readMsg(from_hw, self.schema.I32)
      print(f"Got {result}")
      assert (result.i == data)
