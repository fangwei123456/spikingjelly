import torch
import torch.nn.functional as F
import neuron
from matplotlib import pyplot

if __name__ == "__main__":
    if_node = neuron.IFNode([1], 1.0)
    v = []
    for i in range(1000):
        if_node(0.01)
        v.append(if_node.v.item())

    pyplot.plot(v)
    pyplot.show()

    lif_node = neuron.LIFNode([1], v_threshold=1.0, tau=25)
    v = []

    for i in range(100):
        if i < 20:
            lif_node(0.1)
        else:
            lif_node(0)
        v.append(lif_node.v.item())
        print(v[-1])

    pyplot.plot(v)
    pyplot.show()



