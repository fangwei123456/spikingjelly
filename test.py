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

    sim = learning.STDPLearning(1, 0.1, -1, 0.1, 0.01)
    sim.append(encoding.ConstantEncoder(shape=[1]))
    sim.append(tf.SpikeCurrent(amplitude=1.0))
    sim.append(connection.Linear(in_num=1, out_num=1), is_learning=True)
    sim.append(neuron.IFNode(shape=[1], r=0.5, v_threshold=1.0))
    sim.append(tf.SpikeCurrent(amplitude=0.4))
    sim.append(neuron.IFNode(shape=[1], r=2.0, v_threshold=1.0))

    for i in range(1000):

        output_data = sim.step(torch.randint(low=0, high=2, size=[1]).bool())






