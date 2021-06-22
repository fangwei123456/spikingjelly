import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._memory = {}
        self._memory_rv = {}

    def register_memory(self, name: str, value):
        self._memory[name] = value
        self._memory_rv[name] = value

    def reset(self):
        for key in self._memory.keys():
            self._memory[key] = self._memory_rv[key]
            
    def __getattr__(self, name: str):
        if name in self._memory:
            return self._memory[name]
        else:
            return super().__getattr__(name)
        
    def detach(self):
        for key in self._memory.keys():
            if isinstance(self._memory[key], torch.Tensor):
                self._memory[key].detach_()

    def _apply(self, fn):
        for key, value in self._memory:
            if isinstance(value, torch.Tensor):
                self._memory[key] = fn(value)

        for key, value in self._memory_rv:
            if isinstance(value, torch.Tensor):
                self._memory_rv[key] = fn(value)
        return super()._apply(fn)





