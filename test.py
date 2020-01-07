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
    sim.append(encoding.ConstantEncoder())
    sim.append(tf.SpikeCurrent(amplitude=0.01))
    sim.append(neuron.IFNode(shape=[1], r=0.5, v_threshold=1.0))
    sim.append(tf.SpikeCurrent(amplitude=0.4))
    sim.append(neuron.IFNode(shape=[1], r=2.0, v_threshold=1.0))
    sim.append(tf.ExpDecayCurrent(tau=5.0, amplitude=1.0))
    sim.append(neuron.LIFNode(shape=[1], r=5.0, v_threshold=1.0, tau=10.0))
    v = []
    v.extend(([], [], []))

    for i in range(1000):
        output_data = sim.step(torch.ones(size=[1], dtype=torch.bool))

        #print(i, sim.pipeline)
        for j in range(3):
            v[j].append(sim.module_list[2 * j + 2].v.item())

    pyplot.plot(v[0])
    pyplot.show()
    pyplot.plot(v[1])
    pyplot.show()
    pyplot.plot(v[2])
    pyplot.show()






