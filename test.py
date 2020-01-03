import torch
import torch.nn.functional as F
import neuron
from matplotlib import pyplot

if __name__ == "__main__":
    if_node = neuron.IFNode([1], r=1.0, v_threshold=1.0)
    v = []
    for i in range(1000):
        if_node(0.01)
        v.append(if_node.v.item())

    pyplot.plot(v)
    pyplot.show()

    lif_node = neuron.LIFNode([1], r=9.0, v_threshold=1.0, tau=20.0)
    v = []

    for i in range(1000):
        if i < 500:
            lif_node(0.1)
        else:
            lif_node(0)
        v.append(lif_node.v.item())

    pyplot.plot(v)
    pyplot.show()


