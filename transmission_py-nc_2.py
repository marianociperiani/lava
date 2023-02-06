#!/usr/bin/env python
# coding: utf-8
"""
	                			  ----------------------------------
	                			 /    P2 (Container Process)      / 
                                /             |                  / 
     P1 (Encoder, PyProc) ---> /              |		            / ----> P4(Receiver, PyProc)
                              / PyToNx-> P3 (NcProc) -> NxToPy / 
	                	     ----------------------------------
"""
import numpy as np
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.nc.model import NcProcessModel
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.nc.ports import NcInPort, NcOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.nc.type import LavaNcType
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.resources import CPU, Loihi2NeuroCore
from lava.magma.core.run_configs import Loihi2HwCfg, Loihi2SimCfg
from lava.magma.core.process.variable import Var
from lava.magma.core.model.nc.var import NcVar
from lava.proc.embedded_io.spike import PyToNxAdapter, NxToPyAdapter

from lava.utils.system import Loihi2
if Loihi2.is_loihi2_available:
    from lava.proc import embedded_io as eio

"""Process"""
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

# P3 NC.
class P3(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get('shape', (2,))
        self.inp3 = InPort(shape=shape)
        self.out3 = OutPort(shape=shape)


# P4 receiver.
class P4(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get('shape', (2,))
        self.inp4 = InPort(shape=shape)
        self.out4 = OutPort(shape=shape)



"""ProcessModels"""

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


#NC Process Model.
@implements(proc=P3, protocol=LoihiProtocol)
@requires(Loihi2NeuroCore)
class PyProcModelC(NcProcessModel):
    inp3:  NcInPort = LavaNcType(NcInPort, np.int32, precision=24)
    out3: NcOutPort = LavaNcType(NcOutPort, np.int32, precision=24)

    def run_spk(self):
        in_data3 = self.inp3.recv()
        self.out3.send(in_data3)


#Container process model.
@implements(proc=P2, protocol=LoihiProtocol)
@requires(CPU)
class PyProcModelB(AbstractSubProcessModel):
    def __init__(self, proc):
        
        self.py2nx = PyToNxAdapter(shape=(2, ))
        self.nx2py = NxToPyAdapter(shape=(2, ))
        self.p3=P3()
        # connect Process inport to SubProcess 1 Input
        proc.in_ports.s_in.connect(self.py2nx.inp)
        # SubProcess 1 Output to SubProcess 2 Input
        self.py2nx.out.connect(self.p3.inp3)
        # SubProcess 2 Output to Process Output
        self.p3.out3.connect(self.nx2py.inp)
        self.nx2py.out.connect(proc.out_ports.s_out)



#PyProcModel implementing P4
@implements(proc=P4, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')       
class PyProcModelD(PyLoihiProcessModel):
    inp4: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    out4: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def run_spk(self):
        in_data4 = self.inp4.recv()
        print("P4 received: {}\n".format(in_data4))
        self.out4.send(in_data4)



# Define data & threshold
data = np.array([1, 0])
vth= 3

# Instantiate
sender1 = P1(data=data,vth=vth) 
sender2 = P2() #containing process
sender4 = P4()

# Connecting output port to an input port
sender1.out1.connect(sender2.s_in)
sender2.s_out.connect(sender4.inp4)


from lava.magma.core.run_configs import Loihi2HwCfg
from lava.magma.core.run_conditions import RunSteps
sender1.run(RunSteps(num_steps=3), Loihi2HwCfg(select_sub_proc_model=True))
sender1.stop()