from spikingjelly.activation_based.auto_cuda.generator import analyse_graph, gen_forward_codes, gen_backward_codes
from spikingjelly.activation_based import surrogate

import torch
if __name__ == '__main__':

    def lif_charge(x: torch.Tensor, v_last: torch.Tensor, tau: float, v_reset: float):
        h = v_last + (x - (v_last - v_reset)) / tau
        return h


    input_nodes, inter_nodes, output_nodes, cmds = analyse_graph(lif_charge,
                                                                 requires_grad=(True, True, False, False))

    forward_codes, forward_kernel_name, cuda_cmds = gen_forward_codes(input_nodes, inter_nodes, output_nodes, cmds,
                                                              hard_reset=True)

    backward_codes, backward_kernel_name, input_bp_vars = gen_backward_codes(cuda_cmds, input_nodes, output_nodes, cmds,
                                                                    hard_reset=True,
                                                                    detach_reset=True,
                                                                    surrogate_fuction=surrogate.ATan())

    print(f'forward_codes = \n{forward_codes}')
    print(f'backward_codes = \n{backward_codes}')