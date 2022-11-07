#!/usr/bin/env python
# coding: utf-8

 
# P1 (Encoder, PyProc) --> P2 (NCProcess) --> P3(Receiver, PyProc)


from bz2 import compress
from typing import Dict, Tuple
import numpy as np
from PIL import Image
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.resources import CPU, Loihi2NeuroCore
from lava.magma.core.run_configs import Loihi2HwCfg, Loihi2SimCfg
from lava.magma.core.process.variable import Var

from lava.proc import io
from lava.proc.io.encoder import Compression
from lava.lib.dl.netx.utils import NetDict

from lava.utils.system import Loihi2
if Loihi2.is_loihi2_available:
    from lava.proc import embedded_io as eio


# P1 encoder & sender.
class P1(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get('shape', (2,))
        self.data = Var(shape=(2,), init=kwargs.pop("data", 0))
        self.vth = Var(shape=(1,), init=kwargs.pop("vth", 0))
        self.out1 = OutPort(shape=shape)


#P2 container.
class P2(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (2, ))
        self.s_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)


# P3 receiver.
class P3(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get('shape', (2,))
        self.inp3 = InPort(shape=shape)
        self.out3 = OutPort(shape=shape)


#PyProcModel implementing P1
@implements(proc=P1, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyProcModelA(PyLoihiProcessModel):
    out1: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    data: np.ndarray =   LavaPyType(np.ndarray, int)
    vth: np.ndarray =   LavaPyType(np.ndarray, int)

    def run_spk(self):
        self.data[:] = self.data + 1
        print("data: {}\n".format(self.data))
        s_out = self.data >= self.vth
        self.data[s_out] = 0  # Reset voltage to 0
        self.out1.send(s_out)


#Container process model (NC).
@implements(proc=P2, protocol=LoihiProtocol)
@requires(Loihi2NeuroCore)
class PyProcModelB(AbstractSubProcessModel):
    s_in: PyInPort = LavaPyType(np.ndarray, np.int32)
    s_out: PyOutPort = LavaPyType(np.ndarray, np.int32)

    def __init__(self, proc: AbstractProcess) -> None:
        
        self.adapter = eio.spike.PyToNxAdapter(shape=(2,))
        self.adapter2 = eio.spike.NxToPyAdapter(shape=(2,))

        # connect Process inport to SubProcess 1 Input
        proc.in_ports.s_in.connect(self.adapter.inp)
        # SubProcess 1 Output to SubProcess 2 Input
        self.adapter.out.connect(self.adapter2.inp)
        # SubProcess 2 Output to Process Output
        self.adapter2.out.connect(proc.out_ports.s_out)


#PyProcModel implementing P3
@implements(proc=P3, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')       
class PyProcModelC(PyLoihiProcessModel):
    inp3: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    out3: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def run_spk(self):
        in_data3 = self.inp3.recv()
        print("P3 received: {}\n".format(in_data3))
        self.out3.send(in_data3)



# Define data & threshold
data = np.array([1, 2])
vth= 3

# Instantiate
sender1 = P1(data=data,vth=vth)
sender2 = P2()
sender3 = P3()

# Connecting output port to an input port
sender1.out1.connect(sender2.s_in)
sender2.s_out.connect(sender3.inp3)


from lava.magma.core.run_configs import Loihi2HwCfg
from lava.magma.core.run_conditions import RunSteps

sender1.run(RunSteps(num_steps=3), Loihi2HwCfg())
sender1.stop()

