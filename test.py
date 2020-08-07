import sys
print(sys.path)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from SpikingFlow.clock_driven import neuron, encoding, accelerating, surrogate, functional

import time


if __name__ == '__main__':
    x = torch.rand([4]) - 0.5
    # x = x.to('cuda:1')

    alpha = 2
    x = x.data.clone()
    x.requires_grad_(True)

    x = x.data.clone()
    x.requires_grad_(True)
    sg = surrogate.Sigmoid(alpha)
    t1 = time.time()
    for i in range(1):
        z = sg(x)
        z.sum().backward()
        x.grad.zero_()

    t1 = time.time() - t1
    print('c++', t1 / 100)