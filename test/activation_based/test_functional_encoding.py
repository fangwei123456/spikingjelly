import inspect
import math

import pytest
import torch
import torch.nn.functional as F

from spikingjelly.activation_based import encoding, functional
from spikingjelly.activation_based.functional import (
    encoding as functional_encoding,
    forward,
    layer as functional_layer,
    loss,
    misc,
    net_config,
    neuron as functional_neuron,
    online_learning,
)


def _assert_close(actual, expected):
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_stateful_encoder_single_step_advances_wrapped_time_state():
    spike = torch.arange(12).reshape(3, 2, 2)

    out0, t0 = functional.stateful_encoder_single_step(spike, 0, 3)
    out1, t1 = functional.stateful_encoder_single_step(spike, t0, 3)
    out2, t2 = functional.stateful_encoder_single_step(spike, t1, 3)

    _assert_close(out0, spike[0])
    assert out0.data_ptr() == spike[0].data_ptr()
    assert t0 == 1
    _assert_close(out1, spike[1])
    assert t1 == 2
    _assert_close(out2, spike[2])
    assert t2 == 0


@pytest.mark.parametrize("enc_function", ["linear", "log"])
def test_latency_encode_matches_existing_formula(enc_function):
    x = torch.tensor([[0.0, 0.5, 1.0]])
    T = 5
    spike = functional.latency_encode(x, T, enc_function)
    if enc_function == "log":
        alpha = math.exp(T - 1.0) - 1.0
        t_f = (T - 1.0 - torch.log(alpha * x + 1.0)).round().long()
    else:
        t_f = ((T - 1.0) * (1.0 - x)).round().long()
    expected = F.one_hot(t_f, num_classes=T).to(x).permute(2, 0, 1)

    _assert_close(spike, expected)


def test_latency_encode_rejects_unknown_encoding_function():
    with pytest.raises(NotImplementedError):
        functional.latency_encode(torch.rand(2, 3), 4, "unknown")


def test_periodic_encoder_uses_functional_state_after_first_materialization():
    spike = torch.randn(3, 2)
    encoder = encoding.PeriodicEncoder(spike)

    out0 = encoder(spike)
    out1 = encoder(torch.empty(0))
    out2 = encoder(torch.empty(0))
    out3 = encoder(torch.empty(0))

    assert encoder.spike is spike
    _assert_close(out0, spike[0])
    _assert_close(out1, spike[1])
    _assert_close(out2, spike[2])
    _assert_close(out3, spike[0])
    assert encoder.t == 1


def test_latency_encoder_materializes_spike_once_then_ignores_later_input():
    x = torch.tensor([0.0, 0.5, 1.0])
    encoder = encoding.LatencyEncoder(T=4)

    first = encoder(x)
    stored_spike = encoder.spike
    second = encoder(torch.ones_like(x))

    _assert_close(first, stored_spike[0])
    _assert_close(second, stored_spike[1])
    assert encoder.spike is stored_spike
    assert encoder.t == 2


def test_latency_encoder_single_step_encode_delegates_to_functional_encoding():
    x = torch.tensor([0.0, 0.5, 1.0])
    encoder = encoding.LatencyEncoder(T=4, enc_function="log")

    encoder.single_step_encode(x)

    _assert_close(encoder.spike, functional.latency_encode(x, 4, "log", encoder.alpha))


def test_latency_encoder_uses_its_alpha_state():
    x = torch.tensor([0.5])
    encoder = encoding.LatencyEncoder(T=4, enc_function="log")
    encoder.alpha = 0.0

    encoder.single_step_encode(x)

    _assert_close(encoder.spike, functional.latency_encode(x, 4, "log", alpha=0.0))
    assert encoder.spike[-1].item() == 1.0


def test_weighted_phase_encode_matches_existing_binary_decomposition():
    x = torch.tensor([192 / 256, 1 / 256, 128 / 256, 255 / 256])
    expected = torch.tensor(
        [
            [1.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
        ]
    )

    spike = functional.weighted_phase_encode(x, 8)

    _assert_close(spike, expected)


def test_weighted_phase_encode_rejects_values_outside_supported_range():
    with pytest.raises(AssertionError):
        functional.weighted_phase_encode(torch.tensor([1.0]), 4)


def test_weighted_phase_encode_does_not_mutate_input():
    x = torch.tensor([0.25, 0.5])
    x_before = x.clone()

    functional.weighted_phase_encode(x, 3)

    assert torch.equal(x, x_before)


def test_weighted_phase_encoder_single_step_encode_delegates_to_functional_encoding():
    x = torch.tensor([0.25, 0.5])
    encoder = encoding.WeightedPhaseEncoder(K=3)

    encoder.single_step_encode(x)

    _assert_close(encoder.spike, functional.weighted_phase_encode(x, 3))


def test_stateful_encoder_top_level_exports():
    for name in functional_encoding.__all__:
        assert getattr(functional, name) is getattr(functional_encoding, name)


def test_functional_encoding_exports_do_not_collide_with_existing_functional_modules():
    modules = (
        functional_encoding,
        forward,
        functional_layer,
        loss,
        misc,
        net_config,
        functional_neuron,
        online_learning,
    )
    seen = {}
    for module in modules:
        for name in module.__all__:
            previous = seen.setdefault(name, module.__name__)
            assert previous == module.__name__, (
                f"`{name}` exported by both {previous} and {module.__name__}"
            )


def test_functional_encoding_public_functions_have_annotations_and_bilingual_docstrings():
    for name in functional_encoding.__all__:
        fn = getattr(functional_encoding, name)
        signature = inspect.signature(fn)
        assert signature.return_annotation is not inspect.Signature.empty, name
        for parameter in signature.parameters.values():
            assert parameter.annotation is not inspect.Parameter.empty, (
                name,
                parameter.name,
            )

        doc = inspect.getdoc(fn)
        assert doc is not None, name
        assert f".. _{name}-cn:" in doc
        assert f".. _{name}-en:" in doc
        assert ":param" in doc
        assert ":return:" in doc
        assert ":rtype:" in doc
