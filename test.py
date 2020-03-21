import torch
import torch.nn as nn
import torch.nn.functional as F

fc = nn.Linear(in_features=1, out_features=1)
print('weight', fc.weight, 'bias', fc.bias)
x = torch.rand(size=[1])
print('x', x)
y = fc(x)
print('y', y)
print('fc.weight.matmul(x) + fc.bias = ', fc.weight.matmul(x) + fc.bias)