import torch
from spikingjelly.activation_based import neuron
from spikingjelly import visualizing
from matplotlib import pyplot as plt
import numpy as np

T = 150
x_seq = torch.rand([T, 1])
neu = neuron.SlidingPSN(k=4)

x_seq[50:100].fill_(0.6)
x_seq[100:].fill_(0)

for p in neu.parameters():
    p.requires_grad = False
with torch.no_grad():
    weight = neu.gen_gemm_weight(x_seq.shape[0])
    h_seq = torch.addmm(neu.bias, weight, x_seq.flatten(1)).view(x_seq.shape)

    s_seq = (h_seq >= 0.).float()
dpi = 150
figsize = (8, 6)
visualizing.plot_one_neuron_v_s(h_seq.flatten().numpy() - neu.bias.item(), s_seq.flatten().numpy(), v_threshold=-neu.bias.item(),
                                v_reset=None,
                                figsize=figsize, dpi=dpi)
plt.show()