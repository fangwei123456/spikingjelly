from matplotlib.pyplot import step
import numpy as np
import torch

from .quantize import step_quantize, quantize_8b, right_shift_to_zero
from .base import MemoryModule
from .surrogate import cuba_spike


SCALE_RHO_MULT = 0.1
TAU_RHO_MULT = 100


def _listep_forward(input, decay, state, w_scale, dtype = torch.int32):
    scaled_state = (state * w_scale).clone().detach().to(
        dtype = dtype, device = input.device
    )
    decay_int = (1<<12) - decay.clone().detach().to(
        dtype = dtype, device = input.device
    )
    output = right_shift_to_zero(scaled_state * decay_int, 12) + \
        (w_scale * input).to(dtype = dtype)
    return output / w_scale


def _listep_backward(grad_output, output, decay, input):
    decay_factor = 1 - decay / (1<<12)
    grad_input = grad_output
    grad_state = grad_output * decay_factor
    grad_decay = - grad_output * input
    if torch.numel(decay) == 1:
        grad_decay = torch.sum(grad_decay)
    else:
        grad_decay = torch.sum(grad_decay, dim = 0)
    return grad_input, grad_decay, grad_output


class LeakyIntegratorStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, decay, state, w_scale):
        output = _listep_forward(input, decay, state, w_scale, dtype = torch.int64)
        ctx.save_for_backward(output, decay, input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, decay, input = ctx.saved_tensors
        grad_input, grad_decay, grad_state = _listep_backward(grad_output, output, decay, input)
        return grad_input, grad_decay, grad_state, None


class CubaLIFNode(MemoryModule):
    def __init__(
        self, threshold, current_decay, voltage_decay,
        v_reset=0., tau_grad=1, scale_grad=1, scale=1 << 6,
        norm=None, dropout=None,
        shared_param=True, requires_grad=False, graded_spike=False,
        step_mode = "s", backend = "torch",
        store_v_seq: bool = False, store_i_seq: bool = False,
    ):
        super().__init__()
        self.step_mode = step_mode
        self.backend = backend
        self.store_v_seq = store_v_seq
        self.store_i_seq = store_i_seq
        self.v_reset = v_reset

        self.p_scale = 1<<12
        self.w_scale = int(scale)
        self.s_scale = int(scale * (1<<6))

        self._threshold = int(threshold*self.w_scale) / self.w_scale
        self.tau_rho = tau_grad * self._threshold
        self.scale_rho = scale_grad
        self.shared_param = shared_param
        self.requires_grad = requires_grad

        if norm is not None:
            self.norm = norm()
            if hasattr(self.norm, "pre_hook_fx"):
                self.norm.pre_hook_fx = self.quantize_8bit
        else:
            self.norm = None
        self.drop = dropout

        self.graded_spike = graded_spike
        self.threshold_eps = 0.01 / self.s_scale

        if self.shared_param:
            if not np.isscalar(current_decay) or not np.isscalar(voltage_decay):
                raise AssertionError(
                    f"current_decay and voltage_decay should be scalars when"
                    f"shared_param = True."
                )
            self.register_parameter(
                "current_decay",
                torch.nn.Parameter(
                    torch.FloatTensor([self.p_scale * current_decay]),
                    requires_grad = self.requires_grad,
                )
            )
            self.register_parameter(
                "voltage_decay",
                torch.nn.Parameter(
                    torch.FloatTensor([self.p_scale * voltage_decay]),
                    requires_grad = self.requires_grad,
                )
            )
        else:
            if np.isscalar(current_decay) is True:  # 1% jitter for now
                self.current_decay_min = current_decay * 0.99
                self.current_decay_max = current_decay * 1.01
            else:
                if len(current_decay) != 2:
                    raise AssertionError(
                        f'Expected current decay to be of length 2'
                    )
                self.current_decay_min = current_decay[0]
                self.current_decay_max = current_decay[1]
            if np.isscalar(voltage_decay) is True:
                self.voltage_decay_min = voltage_decay * 0.99
                self.voltage_decay_max = voltage_decay * 1.01
            else:
                if len(voltage_decay) != 2:
                    raise AssertionError(
                        f'Expected voltage decay to be of length 2'
                    )
                self.voltage_decay_min = voltage_decay[0]
                self.voltage_decay_max = voltage_decay[1]
            self.register_parameter(
                'current_decay',
                torch.nn.Parameter(
                    torch.FloatTensor([self.p_scale * self.current_decay_min]),
                    requires_grad=self.requires_grad,
                )
            )
            self.register_parameter(
                'voltage_decay',
                torch.nn.Parameter(
                    torch.FloatTensor([self.p_scale * self.voltage_decay_min]),
                    requires_grad=self.requires_grad,
                )
            )

        self.register_memory(
            "current_state",
            torch.zeros(1, dtype = torch.float)
        )
        self.register_memory(
            "voltage_state",
            torch.zeros(1, dtype = torch.float)
        )
        self.register_memory("shape", None)
        self.register_memory("num_neuron", None)

        self.clamp()

    def quantize_8bit(self, x, descale = False):
        return quantize_8b(x, scale = self.w_scale, descale = descale)

    def clamp(self):
        with torch.no_grad():
            self.current_decay.data.clamp_(0, self.p_scale)
            self.voltage_decay.data.clamp_(0, self.p_scale)

    @property
    def store_v_seq(self):
        return self._store_v_seq

    @store_v_seq.setter
    def store_v_seq(self, value: bool):
        self._store_v_seq = value
        if value:
            if not hasattr(self, "v_seq"):
                self.register_memory("v_seq", None)

    @property
    def store_i_seq(self):
        return self._store_i_seq

    @store_i_seq.setter
    def store_i_seq(self, value: bool):
        self._store_i_seq = value
        if value:
            if not hasattr(self, "i_seq"):
                self.register_memory("i_seq", None)

    @property
    def threshold(self):
        return self._threshold

    @property
    def v_th_mant(self):
        return int(self.threshold * self.w_scale)

    @property
    def device(self):
        return self.current_decay.device

    @property
    def cx_current_decay(self):
        """The compartment current decay parameter to be used for configuring
        Loihi hardware."""
        self.clamp()
        val = step_quantize(self.current_decay).cpu().data.numpy().astype(int)
        if len(val) == 1:
            return val[0]
        return val

    @property
    def cx_voltage_decay(self):
        """The compartment voltage decay parameter to be used for configuring
        Loihi hardware."""
        self.clamp()
        val = step_quantize(self.voltage_decay).cpu().data.numpy().astype(int)
        if len(val) == 1:
            return val[0]
        return val

    @property
    def ref_delay(self):
        """Refractory delay."""
        return 1

    @property
    def scale(self):
        """Scale difference between slayer representation and hardware
        representation of the variable states."""
        return self.w_scale

    @property
    def device_params(self):
        """Dictionary of device parameters."""
        return {
            'type': 'CUBA',
            'iDecay': self.cx_current_decay,
            'vDecay': self.cx_voltage_decay,
            'vThMant': self.v_th_mant,
            'refDelay': self.ref_delay,
            'gradedSpike': self.graded_spike,
        }

    @staticmethod
    @torch.jit.script
    def jit_hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
        v = (1. - spike) * v + spike * v_reset
        return v

    @staticmethod
    @torch.jit.script
    def jit_soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
        v = v - spike * v_threshold
        return v

    def neuronal_charge_1st(self, x: torch.Tensor):
        self.shape = x.shape[1:] # x.shape = [batch_size, ......]
        if len(self.shape) <= 0:
            raise AssertionError(
                "x.shape of neuronal_charge() should be"
                "[batch_size, ......]"
            )
        self.num_neurons = int(np.prod(self.shape))

        if not self.shared_param:
            cd = self.current_decay_min \
                + (self.current_decay_max - self.current_decay_min) \
                * torch.rand(self.shape, dtype = torch.float).to(self.device)
            vd = self.voltage_decay_min \
                + (self.voltage_decay_max - self.voltage_decay_min) \
                * torch.rand(self.shape, dtype = torch.float).to(self.device)
            self.current_decay.data = self.p_scale * cd
            self.voltage_decay.data = self.p_scale * vd

            del self.current_decay_max
            del self.current_decay_min
            del self.voltage_decay_max
            del self.voltage_decay_min

        self.current_state = self.current_state * torch.ones(x.shape).to(
            dtype = self.current_state.dtype,
            device = self.current_state.device,
        )
        self.voltage_state = self.voltage_state * torch.ones(x.shape).to(
            dtype = self.voltage_state.dtype,
            device = self.voltage_state.device,
        )


    def neuronal_charge(self, x: torch.Tensor):
        if self.shape is None: # the method is called for the first time
            self.neuronal_charge_1st(x)
        else:
            if x.shape[1:] != self.shape:
                raise AssertionError(
                    f"x.shape of neuronal_charge should be"
                    f"{self.shape}"
                )
        if self.requires_grad:
            self.clamp()

        current = LeakyIntegratorStep.apply(
            x,
            step_quantize(self.current_decay),
            self.current_state.contiguous(),
            self.s_scale,
        )
        if self.norm is not None:
            current = self.norm(current)
        voltage = LeakyIntegratorStep.apply(
            current,
            step_quantize(self.voltage_decay),
            self.voltage_state.contiguous(),
            self.s_scale,
        )

        self.current_state = current
        self.voltage_state = voltage

    def neuronal_fire(self):
        return cuba_spike.apply(
            self.voltage_state,
            self.threshold + self.threshold_eps,
            self.tau_rho * TAU_RHO_MULT,
            self.scale_rho * SCALE_RHO_MULT,
            self.graded_spike,
            1,
        )

    def neuronal_reset(self, spike):
        if self.v_reset is None:
            self.voltage_state = self.jit_soft_reset(self.voltage_state, spike, self.threshold+self.threshold_eps)
        else:
            self.voltage_state = self.jit_hard_reset(self.voltage_state, spike, 0.)

    def single_step_forward(self, x):
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        if self.drop is not None:
            spike = self.drop(spike)
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        if self.store_i_seq:
            i_seq = []

        for t in range(T):
            y = self.single_step_forward(x_seq[t])
            y_seq.append(y.unsqueeze(0))
            if self.store_v_seq:
                v_seq.append(self.voltage_state.unsqueeze(0))
            if self.store_i_seq:
                i_seq.append(self.current_state.unsqueeze(0))

        if self.store_v_seq:
            self.v_seq = torch.cat(v_seq, dim = 0)
        if self.store_i_seq:
            self.i_seq = torch.cat(i_seq, dim = 0)

        return torch.cat(y_seq, dim = 0)