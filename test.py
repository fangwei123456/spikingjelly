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

    pe = encoding.PoissonEncoder()
    x = torch.rand(size=[8])
    print(x)
    for i in range(10):
        print(pe(x))






