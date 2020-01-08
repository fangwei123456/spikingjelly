import torch
import torch.nn.functional as F
import neuron
import simulating
import encoding
import connection
import learning
import connection.transform as tf
from matplotlib import pyplot


if __name__ == "__main__":

    sim = simulating.Simulator()
    sim.append(learning.STDPModule(tf.SpikeCurrent(amplitude=0.2),
                                   connection.Linear(2, 1),
                                   neuron.IFNode(shape=[1], r=1.0, v_threshold=1.0),
                                   tau_a=10.0,
                                   tau_b=10.0,
                                   learning_rate=1e-3
                                   ))

    in_spike_list0 = []
    in_spike_list1 = []
    out_spike_list = []
    w_list0 = []
    w_list1 = []

    for i in range(600):
        if i < 400:
            in_spike = torch.ones(size=[2], dtype=torch.bool)
        else:
            in_spike = torch.zeros(size=[2], dtype=torch.bool)
            in_spike[1] = True

        out_spike = sim.step(in_spike)
        in_spike_list0.append(in_spike[0].float().item())
        in_spike_list1.append(in_spike[1].float().item())

        out_spike_list.append(out_spike.float().item())

        w_list0.append(sim.module_list[-1].module_list[2].w[:, 0].item())
        w_list1.append(sim.module_list[-1].module_list[2].w[:, 1].item())

    pyplot.plot(in_spike_list0, c='r', label='in_spike[0]')
    pyplot.plot(in_spike_list1, c='g', label='in_spike[1]')
    pyplot.legend()
    pyplot.show()
    pyplot.plot(out_spike_list, label='out_spike')
    pyplot.legend()
    pyplot.show()
    pyplot.plot(w_list0, c='r', label='w[0]')
    pyplot.plot(w_list1, c='g', label='w[1]')
    pyplot.legend()
    pyplot.show()





