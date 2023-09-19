from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.nc.ports import NcInPort, NcOutPort
from lava.magma.core.model.c.ports import COutPort, CInPort
from lava.magma.core.model.nc.type import LavaNcType
try:
    from lava.magma.core.model.nc.net import NetL2
except ImportError:

    class NetL2:
        pass
from lava.magma.core.model.nc.tables import Nodes
from lava.magma.core.resources import Loihi2NeuroCore
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.type import PyInPort, PyOutPort, LavaPyType
from lava.magma.core.model.c.type import LavaCType, LavaCDataType
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.resources import LMT, CPU
from lava.magma.core.process.ports.ports import OutPort, InPort
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.c.model import CLoihiProcessModel
from lava.magma.core.model.nc.model import AbstractNcProcessModel
from lava.magma.core.model.nc.var import NcVar
from lava.proc.lif.process import AbstractLIF
from lava.magma.core.learning.constants import W_TRACE_FRACTIONAL_PART
from lava.proc.dense.process import Dense
from lava.magma.core.process.process import LogConfig
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2HwCfg
from lava.magma.core.process.variable import Var
from lava.magma.core.callback_fx import NxSdkCallbackFx
from lava.utils.profiler import Profiler
import logging

import os
import matplotlib.pyplot as plt
import numpy as np
import typing as ty


class InputProcess(AbstractProcess):
    def __init__(self, shape):
        super().__init__()
        self.out_port = OutPort(shape)


@implements(proc=InputProcess, protocol=LoihiProtocol)
@requires(LMT)
class InputProcessModel(CLoihiProcessModel):
    out_port: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)

    @property
    def source_file_name(self) -> str:
        return "input_injector.c"


class OutputProcess(AbstractProcess):
    def __init__(self, shape):
        super().__init__()
        self.inp = InPort(shape)
        self.in_sync = InPort(shape=(1, ))
        self.out = OutPort(shape=(100,))


@implements(proc=OutputProcess, protocol=LoihiProtocol)
@requires(LMT)
class OutputProcessModel(CLoihiProcessModel):
    """C-process model for NeuroCore to Python spike transmission."""
    inp: CInPort = LavaCType(CInPort, d_type=LavaCDataType.INT32)
    in_sync: CInPort = LavaCType(CInPort, d_type=LavaCDataType.INT32)
    out: COutPort = LavaCType(COutPort, d_type=LavaCDataType.INT32)

    @property
    def source_file_name(self) -> str:
        return "output_snip.c"


# Output receiver
class OutputReceiver(AbstractProcess):
    """Process for receiving data from Output."""
    def __init__(self,
                 *,
                 shape: ty.Tuple[int, ...]) -> None:
        super().__init__(shape=shape)
        self.shape = shape
        self.inp = InPort(shape=self.shape)
        self.out_sync = OutPort(shape=(1, ))
        buffer_shape = (100, 100)
        self.data = Var(shape=buffer_shape, init=np.zeros(buffer_shape))


@implements(proc=OutputReceiver, protocol=LoihiProtocol)
@requires(CPU)
class OutputReceiverModel(PyLoihiProcessModel):
    """Fixed point ring buffer receive process model."""
    inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32)
    out_sync: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32)
    data: np.ndarray = LavaPyType(np.ndarray, np.int32)

    def post_guard(self) -> bool:
        return True

    def run_post_mgmt(self) -> None:
        """Receive spikes and store in an internal variable"""
        if self.time_step == 100:
            # Sending a signal to Loihi to sync on the 100th time step
            # Loihi will be waiting to receive this message
            # This is necessary, outherwise Loihi would have finished and
            # closed the channel 'inp' to retrieve the data, i.e., the program
            # would be stalling waiting on the data.
            
            # After both CPU and Loihi backend are synced on the 100 time step
            # retrieve the data from Loihi
            for i in range(100):
                self.out_sync.send(np.array([i], np.int32))
                data = self.inp.recv()
                self.data[..., i] = data


class Probes(NxSdkCallbackFx):
    def __init__(self):
        self.trace_probes = []
        self.syn_probes = []
        self.u_probes = []
        self.v_probes = []
        self.spikes_probe = []

    def pre_run_callback(self, board, var_id_to_model_map):
        nxChip = board.nxChips[0]
        nxCorePost = nxChip.nxCores[0]
        # nxCorePre = nxChip.nxCores[1]
        mon = board.monitor
        pCore = nxChip.cpuCores[0]
        # self.u_probes.append(
        #     mon.probe(
        #         nxCorePre.neuronInterface.group[0].cxState,
        #         [0],
        #         "u"
        #     )[0]
        # )
        self.u_probes.append(
            mon.probe(
                nxCorePost.neuronInterface.group[0].cxState,
                [20],
                "u"
            )[0]
        )
        # self.v_probes.append(
        #     mon.probe(
        #         nxCorePre.neuronInterface.group[0].cxState,
        #         [0],
        #         "v"
        #     )[0]
        # )
        self.v_probes.append(
            mon.probe(
                nxCorePost.neuronInterface.group[0].cxState,
                [20],
                "v"
            )[0]
        )
        # self.spikes_probe.append(
        #     mon.probe(pCore.spikeCounter, [0], 'spike_count')[0]
        # )
        # self.spikes_probe.append(
        #     mon.probe(pCore.spikeCounter, [1], 'spike_count')[0]
        # )
        # self.trace_probes.append(
        #     mon.probe(
        #         nxCorePost.synapseMap.profile[0],
        #         [0],
        #         "traceEntry[0].doubleTraceEntry.x1",
        #     )[0]
        # )
        # self.trace_probes.append(
        #     mon.probe(
        #         nxCorePost.synapseMap.profile[0],
        #         [0],
        #         "traceEntry[0].doubleTraceEntry.x2",
        #     )[0]
        # )
        # self.trace_probes.append(
        #     mon.probe(
        #         nxCorePost.neuronInterface.group[0].stdpPostState,
        #         [0],
        #         "yepoch0"
        #     )[0]
        # )
        # self.trace_probes.append(
        #     mon.probe(
        #         nxCorePost.neuronInterface.group[0].stdpPostState,
        #         [0],
        #         "yepoch1"
        #     )[0]
        # )
        # self.trace_probes.append(
        #     mon.probe(
        #         nxCorePost.neuronInterface.group[0].stdpPostState,
        #         [0],
        #         "yepoch2"
        #     )[0]
        # )
        # self.syn_probes.append(
        #     mon.probe(nxCorePost.synapseMem.entry[0], [0], "wgt")[0]
        # )

    def post_run_callback(self, board, var_id_to_model_map):
        # print("LIF-pre spikes", self.spikes_probe[0].data)
        # print("LIF-post spikes", self.spikes_probe[1].data)
        # print("LIF-pre u", self.u_probes[0].data)
        # print("LIF-pre v", self.v_probes[0].data)
        np.save("probe_u.npy", self.u_probes[0].data)
        np.save("probe_v.npy", self.v_probes[0].data)
        print("LIF-post u", self.u_probes[0].data)
        print("LIF-post v", self.v_probes[0].data)
        # print("x1", self.trace_probes[0].data)
        # print("x2", self.trace_probes[1].data)
        # print("y1", self.trace_probes[2].data)
        # print("y2", self.trace_probes[3].data)
        # print("y3", self.trace_probes[4].data)
        # print("wgt", self.syn_probes[0].data)


class CustomLIF(AbstractLIF):
    """Leaky-Integrate-and-Fire (LIF) neural Process.

    LIF dynamics abstracts to:
    u[t] = u[t-1] * (1-du) + a_in         # neuron current
    v[t] = v[t-1] * (1-dv) + u[t] + bias  # neuron voltage
    s_out = v[t] > vth                    # spike if threshold is exceeded
    v[t] = 0                              # reset at spike

    Parameters
    ----------
    shape : tuple(int)
        Number and topology of LIF neurons.
    u : float, list, numpy.ndarray, optional
        Initial value of the neurons' current.
    v : float, list, numpy.ndarray, optional
        Initial value of the neurons' voltage (membrane potential).
    du : float, optional
        Inverse of decay time-constant for current decay. Currently, only a
        single decay can be set for the entire population of neurons.
    dv : float, optional
        Inverse of decay time-constant for voltage decay. Currently, only a
        single decay can be set for the entire population of neurons.
    bias_mant : float, list, numpy.ndarray, optional
        Mantissa part of neuron bias.
    bias_exp : float, list, numpy.ndarray, optional
        Exponent part of neuron bias, if needed. Mostly for fixed point
        implementations. Ignored for floating point implementations.
    vth : float, optional
        Neuron threshold voltage, exceeding which, the neuron will spike.
        Currently, only a single threshold can be set for the entire
        population of neurons.

    Example
    -------
    >>> lif = LIF(shape=(200, 15), du=10, dv=5)
    This will create 200x15 LIF neurons that all have the same current decay
    of 10 and voltage decay of 5.
    """

    def __init__(
        self,
        *,
        shape: ty.Tuple[int, ...],
        u: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        v: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        du: ty.Optional[float] = 0,
        dv: ty.Optional[float] = 0,
        bias_mant: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        bias_exp: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        t_half: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        charging_bias: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        vth: ty.Optional[float] = 10,
        name: ty.Optional[str] = None,
        log_config: ty.Optional[LogConfig] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            shape=shape,
            u=u,
            v=v,
            du=du,
            dv=dv,
            bias_mant=bias_mant,
            bias_exp=bias_exp,
            charging_bias=charging_bias,
            t_half=t_half,
            name=name,
            log_config=log_config,
            **kwargs,
        )

        self.t_half = Var(shape=(1,), init=t_half)
        self.charging_bias = Var(shape=(1,), init=charging_bias)
        self.vth = Var(shape=(1,), init=vth)


@implements(proc=CustomLIF, protocol=LoihiProtocol)
@requires(Loihi2NeuroCore)
@tag("ucoded")
class CustomLIFModel(AbstractNcProcessModel):
    """Implementation of a Leaky Integrate-and-Fire (LIF) neural process
    model that defines the behavior of micro-coded (ucoded) LIF neurons on
    Loihi 2. In its current form, this process model exactly matches the
    hardcoded LIF behavior with one exception: To improve performance by a
    factor of two, the negative saturation behavior of v is switched off.
    """

    # Declare port implementation
    a_in: NcInPort = LavaNcType(NcInPort, np.int16, precision=16)
    s_out: NcOutPort = LavaNcType(NcOutPort, np.int32, precision=24)
    # Declare variable implementation
    u: NcVar = LavaNcType(NcVar, np.int32, precision=24)
    v: NcVar = LavaNcType(NcVar, np.int32, precision=24)
    du: NcVar = LavaNcType(NcVar, np.int16, precision=12)
    dv: NcVar = LavaNcType(NcVar, np.int16, precision=12)
    charging_bias = LavaNcType(NcVar, np.int16, precision=12)
    t_half = LavaNcType(NcVar, np.int16, precision=12)
    bias_mant: NcVar = LavaNcType(NcVar, np.int16, precision=13)
    bias_exp: NcVar = LavaNcType(NcVar, np.int16, precision=3)
    vth: NcVar = LavaNcType(NcVar, np.int32, precision=17)

    def allocate(self, net: NetL2):
        """Allocates neural resources in 'virtual' neuro core."""

        shape = np.product(list(self.proc_params["shape"]))

        curr_dir = os.path.dirname(os.path.realpath(__file__))
        ucode_file = os.path.join(curr_dir, "customLIF.dasm")

        # vth_reg = np.left_shift(self.vth.var.init, 6)
        vth_reg = self.vth.var.init

        # Allocate neurons
        neurons_cfg: Nodes = net.neurons_cfg.allocate_ucode(
            shape=(1,),
            ucode=ucode_file,
            vth=vth_reg,
            t_half=self.t_half,
            charging_bias=self.charging_bias,
            du=4096 - self.du.var.get(),
            dv=4096 - self.dv.var.get(),
        )
        neurons: Nodes = net.neurons.allocate_ucode(
            shape=shape,
            u=self.u,
            v=self.v,
            bias=self.bias_mant.var.get() * 2 ** (self.bias_exp.var.get()),
        )

        # Allocate output axons
        ax_out: Nodes = net.ax_out.allocate(shape=shape, num_message_bits=0)

        # Connect InPort of Process to neurons
        self.a_in.connect(neurons)
        # Connect Nodes
        neurons.connect(neurons_cfg)
        neurons.connect(ax_out)
        # Connect output axon to OutPort of Process
        ax_out.connect(self.s_out)


def calculate_weights(N):
    """
    Calculate the weights based on the DFT equation
    """
    c = 2 * np.pi/N
    n = np.arange(N).reshape(N, 1)
    k = np.arange(N).reshape(1, N)
    trig_factors = np.dot(n, k) * c
    real_weights = np.cos(trig_factors)[:N//2]
    imag_weights = -np.sin(trig_factors)[:N//2]
    weights = np.vstack((real_weights, imag_weights))*127
    weights = weights.astype(np.int8)
    return weights


def parse_txt_data(fname, N, T):
    w=[]
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    fname = os.path.join(curr_dir, fname)
    with open(fname, "r") as f:
        a = f.read()
    data_list = [int(x) for x in a.split(",")]

    m=(1-T)/(np.max(data_list)-np.min(data_list))
    b=1-m*np.max(data_list)

    w=[j * m + b for j in data_list]

    data_array = np.array(w)
    spike_array = np.zeros((N, T))


    for i in range(N):
        if int(data_array[i])==T:
            data_array[i]=data_array[i]-1
        spike_array[i][int(data_array[i])] = 1
    return spike_array


def parse_out_data(spike_train, T):
    spike_times =  np.argmax(spike_train, axis=1)
    formatted_spike_times = 0.75*T - spike_times
    sft = np.sqrt(formatted_spike_times[1:50]**2 + formatted_spike_times[51:]**2)
    np.save("spike_train.npy", spike_train)
    np.save("spike_times.npy", spike_times)
    np.save("sft.npy", sft)
    return sft


def plot_results(sft, in_data):
    fft = np.fft.fft(in_data)
    fft_abs = np.abs(fft[1:49])
    fig, ax = plt.subplots(2)
    ax[0].plot(fft_abs, color="cornflowerblue")
    ax[1].plot(sft, color="cornflowerblue")
    ax[0].set_title("FFT")
    ax[0].set_yticks([])
    ax[1].set_title("Spiking FT")
    ax[1].set_yticks([])
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    plt.tight_layout()
    fig.savefig("ft_plot.pdf")


def main(N=100, T=100, SFT=True, profile=False):
    #np.set_printoptions(threshold=np.inf)
    if not SFT:
        bias=20
        sample_data = np.zeros((N, T))
        sample_data[:, :T//2] = np.eye(N)
        alpha = 0.5
        vth = bias * T//2
        weights = np.eye(N)*100
        vth = alpha*0.25*weights[0].sum()*T
    else:
        sample_data = parse_txt_data("512-samples.txt", N, T)
        #print(sample_data)
        weights = calculate_weights(N)
        alpha = 0.25
        vth = alpha*0.25*weights[0].sum()*T
        bias = 4*vth / T

    # Used to generate the input data for the CProc
    #np.savetxt("input_spikes.txt", sample_data.astype(int), fmt='%i', delimiter=", ")

    input_injector = InputProcess(shape=(N,))
    dense1 = Dense(weights=weights)
    lif1 = CustomLIF(shape=(N,), t_half=T//2, vth=vth, charging_bias=bias)
    output = OutputProcess(shape=(N,))
    receiver = OutputReceiver(shape=(100,))

    input_injector.out_port.connect(dense1.s_in)
    dense1.a_out.connect(lif1.a_in)
    lif1.s_out.connect(output.inp)
    receiver.out_sync.connect(output.in_sync)
    output.out.connect(receiver.inp)
    
    if profile:
        run_config = Loihi2HwCfg()
        profiler = Profiler.init(run_config)

        profiler.execution_time_probe(num_steps=T-1)
        # profiler.energy_probe(num_steps=T)
        # profiler.activity_probe()
        # profiler.memory_probe()

    else:
        probes = Probes()
        run_config = Loihi2HwCfg()

    #lif1._log_config.level = logging.INFO
    lif1.run(condition=RunSteps(num_steps=T), run_cfg=run_config)

    # Spiking data from output is always recorded
    spike_out = receiver.data.get()

    if not profile:
        print("Currents:\n {}".format(lif1.u.get()))
        print("Voltages:\n {}".format(lif1.v.get()))
    lif1.stop()

    # Output spikes are saved to file
    print("Count: ", np.sum(spike_out))  # This is a check (should be 156)
    np.savetxt("array_output.txt", spike_out.astype(int), fmt='%i', delimiter=", ")

    if profile:
        profiler.execution_time[:]
        print(profiler.execution_time)
        #print(f"Total execution time: {np.round(np.sum(profiler.execution_time), 6)} s")
        # Using all the profiler probes at the same time seems to not work
        # print(f"Total power: {np.round(profiler.power, 6)} W")
        # print(f"Total energy: {np.round(profiler.energy, 6)} J")
        # print(f"Static energy: {np.round(profiler.static_energy, 6)} J")
        # print(f"Power Breakdown: {np.round(profiler.power_breakdown(), 6)} W")
        # print(f"Power Breakdown: {np.round(profiler.energy_breakdown(), 6)} µJ")
        # profiler.plot_activity(file="./core_activity")
        # profiler.plot_memory_util("./memory_util")
        profiler.plot_execution_time("./execution_time")
        # print(profiler.statement)
    else:
        ...
        # This does not work anymore with the new spike_out
        sft = parse_out_data(spike_out, T)
        plot_results(sft, sample_data)
    print("DONE")


if __name__ == "__main__":
    main()
