from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.nc.ports import NcInPort, NcOutPort
from lava.magma.core.model.nc.type import LavaNcType
try:
    from lava.magma.core.model.nc.net import NetL2
except ImportError:

    class NetL2:
        pass
from lava.magma.core.model.nc.tables import Nodes
from lava.magma.core.resources import NeuroCore, Loihi2NeuroCore
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.nc.model import AbstractNcProcessModel
from lava.magma.core.model.nc.var import NcVar
from lava.proc.lif.process import LIF, AbstractLIF
from lava.magma.core.learning.constants import W_TRACE_FRACTIONAL_PART

from lava.proc.embedded_io.spike import PyToNxAdapter, NxToPyAdapter
from lava.proc.io.source import RingBuffer as SpikeGenerator
from lava.proc.io.sink import RingBuffer as Sink
from lava.proc.dense.process import Dense
from lava.magma.core.process.process import LogConfig
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2HwCfg
from lava.magma.core.process.variable import Var
from lava.magma.core.callback_fx import NxSdkCallbackFx
from lava.utils.profiler import Profiler


import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import typing as ty
import termplotlib as tpl
import sys

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
    with open(fname, "r") as f:
        a = f.read()
    data_list = [int(x) for x in a.split(" ")]
    m=(1-T)/(np.max(data_list)-np.min(data_list))
    b=1-m*np.max(data_list)

    w=[int(j * m + b) for j in data_list]

    data_array = np.array(w)
    spike_array = np.zeros((N, T))
    

    for i in range(N):
        data_array[i]=int(data_array[i])-1
        spike_array[i][int(data_array[i])] = 1

    np.savetxt('time_coded_array.txt', data_array, delimiter=' ', fmt='%f')
    np.savetxt('sample_data.txt', spike_array)
    plot_input_spikes(spike_array)
    return spike_array,m,b,data_array

def parse_out_data(spike_train, T):
    spike_times =  np.argmax(spike_train, axis=1)
    formatted_spike_times = 0.75*T - spike_times
    sft = np.sqrt(formatted_spike_times[1:128]**2 + formatted_spike_times[129:]**2)
    np.save("spike_times.npy", spike_times)
    np.savetxt('spike_times.txt', spike_times, delimiter=',', fmt='%d')
    np.savetxt('formatted_spike_times.txt', formatted_spike_times, delimiter=',', fmt='%f')
    np.save("sft.npy", sft)
    return sft


def plot_results(sft, in_data):
    np.savetxt('in_data.txt', in_data, fmt='%10.2f')
    fft = np.fft.fft(in_data)
    np.savetxt('fft.txt', fft,delimiter=',', fmt='%10.2f')
    fft_abs = np.abs(fft)
    np.savetxt('fft_abs.txt',fft_abs)
    
    fig, ax = plt.subplots(2)
    ax[0].plot(fft[1:128], color="cornflowerblue")
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

def plot_input_spikes(spike_array):
    # Create a 2D array where each row represents a sequence of samples (time steps)
    data = spike_array
    
    # Create a plot for each row in the array
    #for i in range(data.shape[0]):
    # Create a plot for the first ten rows (neurons)
    for i in range(10):
        plt.plot(data[i], 'x-', label=f"Neuron {i+1}") 
    
    # Add labels and legend
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    
    # Save the plot as a PNG file
    plt.savefig("input_spikes_plot.png")
    
    # Close the plot
    plt.close()
    


def main(N=256, T=20, SFT=True, profile=True):
    np.set_printoptions(threshold=sys.maxsize)
    if not SFT:
        bias=20
        sample_data = np.zeros((N, T))
        sample_data[:, :T//2] = np.eye(N)
        alpha = 0.5
        vth = bias * T//2
        weights = np.eye(N)*100
        vth = alpha*0.25*weights[0].sum()*T
    else:
        sample_data,slope,variation,time_coded_array = parse_txt_data("256-samples.txt", N, T)
        
        weights = calculate_weights(N)
        alpha = 0.25
        vth = alpha*0.25*weights[0].sum()*T
        bias = 1000*vth / T

        print("Slope (Silent Stage)")
        print(slope)
        print("Step")
        print(variation)
        print("Vth")
        print(vth)        
        print("Time Coded Array")
        print(time_coded_array)
        print("Sample Data (Input Array)")
        print(sample_data)
        sample_data = np.array(sample_data)

    
    sg = SpikeGenerator(data=sample_data)
    py2nx = PyToNxAdapter(shape=(N, ))
    dense1 = Dense(weights=weights)
    lif1 = CustomLIF(shape=(N,), t_half=T//2, vth=vth, charging_bias=bias)
    dense2 = Dense(weights=np.eye(N))
    nx2py = NxToPyAdapter(shape=(N, ))
    sink = Sink(shape=(N, ), buffer=T)


    sg.s_out.connect(py2nx.inp)
    py2nx.out.connect(dense1.s_in)
    dense1.a_out.connect(lif1.a_in)
    lif1.s_out.connect(dense2.s_in)
    dense2.s_in.connect(nx2py.inp)
    nx2py.out.connect(sink.a_in)

    if profile:
        run_config = Loihi2HwCfg()
        profiler = Profiler.init(run_config)
        # profiler.energy_probe(num_steps=T)
        # profiler.activity_probe()
        # profiler.memory_probe()
        profiler.execution_time_probe(num_steps=T)
    else:
        run_config = Loihi2HwCfg()

    lif1.run(condition=RunSteps(num_steps=T), run_cfg=run_config)

    if not profile:
        spike_out = sink.data.get()
        print("Sent spikes:", sg.data.get(), "\nReceived spikes:", spike_out)
        np.savetxt('sent_spikes.txt', sg.data.get(), delimiter=',', fmt='%d')
        np.savetxt('received_spikes.txt', spike_out, delimiter=',', fmt='%d')
        print("Currents:\n {}".format(lif1.u.get()))
        print("Voltages:\n {}".format(lif1.v.get()))
    lif1.stop()

    if profile:
        # profiler.execution_time[:]
        print(f"Total execution time: {np.round(np.sum(profiler.execution_time), 6)} s")
        # print(f"Total power: {np.round(profiler.power, 6)} W")
        # print(f"Total energy: {np.round(profiler.energy, 6)} J")
        # print(f"Static energy: {np.round(profiler.static_energy, 6)} J")
        print(f"Total execution time: {np.round(np.sum(profiler.execution_time), 6)} s")
        profiler.plot_execution_time("./execution_time2")
    else:
        sft = parse_out_data(spike_out, T)

        print("SFT")
        print(sft)
        np.savetxt('sft.txt', sft,delimiter=',', fmt='%10.2f')
        plot_results(sft, sample_data)
    print("DONE")



if __name__ == "__main__":
    main()
