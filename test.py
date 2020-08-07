import sys
print(sys.path)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from SpikingFlow.clock_driven import neuron, encoding, accelerating, surrogate, functional
from matplotlib import pyplot as plt


if __name__ == '__main__':
    plt.style.use(['muted'])
    fig = plt.figure(dpi=200)
    x = torch.arange(-2.5, 2.5, 0.001)
    plt.plot(x.data, surrogate.heaviside(x), label='heaviside', linestyle='-.')
    surrogate_function = surrogate.BilinearLeakyReLU(w=1, c=0, spiking=False)
    x = x.data.clone()
    x.requires_grad_(True)
    y = surrogate_function(x)
    plt.plot(x.data, y.data, label='primitive, w=1, c=0.01')
    z = y.sum()
    z.backward()
    plt.plot(x.data, x.grad, label='gradient, w=1, c=0.01')
    plt.xlim(-2, 2)
    plt.legend()
    plt.title('BilinearLeakyReLU surrogate function')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.grid(linestyle='--')
    plt.show()