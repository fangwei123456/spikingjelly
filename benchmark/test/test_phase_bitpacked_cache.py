from types import SimpleNamespace

import pytest
import torch
from torch import nn


def test_phase_bitpacked_cache_append_and_reorder():
    pytest.importorskip("triton")
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "4.40.1" or not torch.cuda.is_available():
        pytest.skip("Requires the historical phase benchmark environment")

    from benchmark.snn_llm.phase_bitpacked_cache import (
        _PhaseBitpackedCache,
        _phase_bitpacked_cache,
    )
    from transformers.cache_utils import DynamicCache

    class Neuron:
        T = 8
        spike_one = False
        v0 = 0.0

        def get_grain_power(self, step, mode="d"):
            base = {"d": 0.75, "h": 0.5, "theta": 0.625}[mode]
            return torch.tensor(base ** (step + 1), dtype=torch.bfloat16)

    def reference(value, neuron):
        batch, time, heads, length, width = value.shape
        temporal = value.permute(1, 0, 3, 2, 4).reshape(
            time, batch, length, heads * width
        )
        voltage = torch.zeros_like(temporal[0])
        for step in range(time):
            voltage = voltage + temporal[step]
        sign = voltage.sign()
        voltage = voltage.abs() + neuron.v0
        events = []
        for step in range(time):
            event = (voltage - neuron.get_grain_power(step, "theta") > 0).to(
                voltage.dtype
            )
            events.append(sign * neuron.get_grain_power(step) * event)
            voltage = voltage - neuron.get_grain_power(step, "h") * event
        return (
            torch.stack(events)
            .reshape(time, batch, length, heads, width)
            .permute(1, 0, 3, 2, 4)
        )

    neuron = Neuron()
    identity = SimpleNamespace(input_quantizer=neuron, use_act_quant=True)
    attention = SimpleNamespace(
        q_proj=nn.Linear(4, 4, bias=False).to(device="cuda", dtype=torch.bfloat16),
        k_Identity=identity,
        v_Identity=identity,
    )
    cache = _PhaseBitpackedCache([SimpleNamespace(self_attn=attention)])
    assert cache.to_legacy_cache() is cache
    torch.manual_seed(9)
    key = torch.randn(2, 8, 2, 3, 4, device="cuda").to(torch.bfloat16)
    value = torch.randn_like(key)
    key[0, :, 0, 0] = 0
    with pytest.raises(RuntimeError, match="inference-only"):
        cache.update(key, value, 0)
    for length in (3, 1):
        with torch.inference_mode():
            decoded_key, decoded_value = cache.update(
                key[:, :, :, -length:], value[:, :, :, -length:], 0
            )
        if length == 3:
            full_key, full_value = key, value
        else:
            full_key = torch.cat((key, key[:, :, :, -1:]), dim=3)
            full_value = torch.cat((value, value[:, :, :, -1:]), dim=3)
        torch.testing.assert_close(
            decoded_key, reference(full_key, neuron), rtol=0, atol=0
        )
        torch.testing.assert_close(
            decoded_value, reference(full_value, neuron), rtol=0, atol=0
        )
        assert cache.get_seq_length() == full_key.shape[3]
    indices = torch.tensor([1, 0], device="cuda")
    cache.reorder_cache(indices)
    reordered = cache._unpack(
        cache.key_cache[0], cache._neuron_values[0][0][2], key.dtype
    )
    torch.testing.assert_close(
        reordered, reference(full_key, neuron)[indices], rtol=0, atol=0
    )
    assert (
        cache.key_cache[0].numel() * cache.key_cache[0].element_size() * 8
        == full_key.numel() * full_key.element_size()
    )
    descriptor = DynamicCache.__dict__["from_legacy_cache"]
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])
    )
    with _phase_bitpacked_cache(model) as active:
        assert not identity.use_act_quant
        assert DynamicCache.from_legacy_cache(active) is active
    assert identity.use_act_quant
    assert DynamicCache.__dict__["from_legacy_cache"] is descriptor
