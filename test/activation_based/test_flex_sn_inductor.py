"""M1 tests for FlexSN inductor backend (eager path only).

Validates that the new ``flex_sn_scan`` HOP + ``FlexSN(backend="inductor")``
produce numerically identical output/state sequences to the reference
``backend="torch"`` implementation on a simple LIF single-step core.

Inductor lowering and Triton codegen are M3 scope and are not exercised here,
so these tests do not require a CUDA GPU.
"""
from __future__ import annotations

import pytest
import torch

from spikingjelly.activation_based.neuron.flexsn import FlexSN
from spikingjelly.activation_based.triton_kernel.flex_sn_inductor import (
    flex_sn_scan,
)


def _lif_core(x: torch.Tensor, v: torch.Tensor):
    tau = 2.0
    v_threshold = 1.0
    v_reset = 0.0
    h = v + (x - (v - v_reset)) / tau
    s = (h >= v_threshold).to(h.dtype)
    v_new = h * (1.0 - s) + v_reset * s
    return s, v_new


@pytest.fixture
def rng():
    return torch.Generator().manual_seed(42)


@pytest.mark.parametrize("T", [1, 4, 16])
@pytest.mark.parametrize("shape", [(8,), (4, 8), (2, 3, 8)])
def test_hop_matches_manual_loop(rng, T, shape):
    x = torch.randn((T, *shape), generator=rng)
    v0 = torch.zeros(shape)

    s_seq, v_seq = flex_sn_scan(_lif_core, 1, 1, 1, x, v0)

    expected_s, expected_v = [], []
    v = v0.clone()
    for t in range(T):
        s, v = _lif_core(x[t], v)
        expected_s.append(s)
        expected_v.append(v)
    expected_s = torch.stack(expected_s, dim=0)
    expected_v = torch.stack(expected_v, dim=0)

    torch.testing.assert_close(s_seq, expected_s)
    torch.testing.assert_close(v_seq, expected_v)


@pytest.mark.parametrize("T", [1, 4, 16])
@pytest.mark.parametrize("shape", [(8,), (4, 8)])
def test_inductor_backend_matches_torch_backend(rng, T, shape):
    x = torch.randn((T, *shape), generator=rng)

    torch_neuron = FlexSN(
        core=_lif_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="torch",
    )
    inductor_neuron = FlexSN(
        core=_lif_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="inductor",
    )

    torch_out = torch_neuron(x)
    inductor_out = inductor_neuron(x)

    torch.testing.assert_close(inductor_out, torch_out)
    assert len(inductor_neuron.states) == 1
    torch.testing.assert_close(inductor_neuron.states[0], torch_neuron.states[0])


def test_inductor_backend_store_state_seqs(rng):
    T, N = 8, 16
    x = torch.randn((T, N), generator=rng)

    torch_neuron = FlexSN(
        core=_lif_core, num_inputs=1, num_states=1, num_outputs=1,
        step_mode="m", backend="torch", store_state_seqs=True,
    )
    inductor_neuron = FlexSN(
        core=_lif_core, num_inputs=1, num_states=1, num_outputs=1,
        step_mode="m", backend="inductor", store_state_seqs=True,
    )

    torch_neuron(x)
    inductor_neuron(x)

    assert len(inductor_neuron.state_seqs) == 1
    torch.testing.assert_close(
        inductor_neuron.state_seqs[0], torch_neuron.state_seqs[0]
    )


def test_inductor_backend_skips_flex_sn_kernel_construction():
    neuron = FlexSN(
        core=_lif_core, num_inputs=1, num_states=1, num_outputs=1,
        step_mode="m", backend="inductor",
    )
    assert neuron.kernel is None


def test_inductor_backend_in_supported_backends():
    neuron = FlexSN(
        core=_lif_core, num_inputs=1, num_states=1, num_outputs=1,
        step_mode="m", backend="inductor",
    )
    assert "inductor" in neuron.supported_backends


def test_hop_rejects_wrong_arity():
    x = torch.randn(4, 8)
    v0 = torch.zeros(8)
    extra = torch.zeros(8)

    with pytest.raises(ValueError, match="expected 2 tensor args"):
        flex_sn_scan(_lif_core, 1, 1, 1, x, v0, extra)


def test_hop_rejects_mismatched_T():
    x1 = torch.randn(4, 8)
    x2 = torch.randn(5, 8)
    v0 = torch.zeros(8)

    def two_input_core(a, b, v):
        return a + b, v + a

    with pytest.raises(ValueError, match="leading dim"):
        flex_sn_scan(two_input_core, 2, 1, 1, x1, x2, v0)
