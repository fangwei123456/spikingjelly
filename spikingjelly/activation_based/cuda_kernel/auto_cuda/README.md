# Auto CUDA

`auto_cuda` contains the experimental CUDA code generator used by CuPy neuron
kernels. It owns generic kernel-building primitives and does not own concrete
neuron implementations; those live in `cuda_kernel/neuron_kernel`.

The generator currently traces a Python `neuronal_charge` function containing
the arithmetic operators `+`, `-`, `*`, and `/`, then emits forward and backward
CUDA source.

## Example

```python
import torch

from spikingjelly.activation_based import surrogate
from spikingjelly.activation_based.cuda_kernel.auto_cuda.generator import (
    analyse_graph,
    gen_backward_codes,
    gen_forward_codes,
)


def lif_charge(
    x: torch.Tensor, v_last: torch.Tensor, tau: float, v_reset: float
):
    return v_last + (x - (v_last - v_reset)) / tau


input_nodes, inter_nodes, output_nodes, commands = analyse_graph(
    lif_charge,
    requires_grad=(True, True, False, False),
)
forward_codes, _, cuda_commands = gen_forward_codes(
    input_nodes,
    inter_nodes,
    output_nodes,
    commands,
    hard_reset=True,
)
backward_codes, _, _ = gen_backward_codes(
    cuda_commands,
    input_nodes,
    output_nodes,
    commands,
    hard_reset=True,
    detach_reset=True,
    surrogate_fuction=surrogate.ATan(),
)
```

`surrogate_fuction` keeps its historical spelling because it is a public keyword
of `gen_backward_codes`.
