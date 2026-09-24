"""Inference-only KV cache for the historical T=8 phase-SpikingLLM attention."""

from contextlib import contextmanager

import torch
import triton
import triton.language as tl
from transformers.cache_utils import DynamicCache


@triton.jit
def _unpack_bits(code_ptr, power_ptr, output_ptr, elements, block: tl.constexpr):
    index = tl.program_id(0) * block + tl.arange(0, block)
    valid = index < elements * 8
    step = index // elements
    code = tl.load(code_ptr + index % elements, mask=valid, other=0).to(tl.int32)
    power = tl.load(power_ptr + step, mask=valid, other=0)
    sign = ((code >> 8) & 3) - 1
    event = ((code >> step) & 1) != 0
    value = tl.where(event & (sign > 0), power, tl.where(event & (sign < 0), -power, 0))
    tl.store(output_ptr + index, value, mask=valid)


@triton.jit
def _pack_bits(
    input_ptr,
    reset_ptr,
    threshold_ptr,
    v0_ptr,
    output_ptr,
    elements,
    sequence: tl.constexpr,
    heads: tl.constexpr,
    width: tl.constexpr,
    block: tl.constexpr,
):
    index = tl.program_id(0) * block + tl.arange(0, block)
    valid = index < elements
    dtype = input_ptr.dtype.element_ty
    voltage = tl.full((block,), 0, tl.float32)
    for step in tl.static_range(8):
        current = tl.load(input_ptr + step * elements + index, mask=valid, other=0)
        voltage = (voltage + current.to(tl.float32)).to(dtype).to(tl.float32)
    sign_code = tl.where(voltage > 0, 2, tl.where(voltage < 0, 0, 1))
    voltage = (tl.abs(voltage) + tl.load(v0_ptr)).to(dtype).to(tl.float32)
    bits = tl.full((block,), 0, tl.int32)
    for step in tl.static_range(8):
        threshold = tl.load(threshold_ptr + step).to(tl.float32)
        reset = tl.load(reset_ptr + step).to(tl.float32)
        event = ((voltage - threshold).to(dtype).to(tl.float32) > 0).to(tl.int32)
        bits = bits | (event << step)
        decrement = (reset * event.to(tl.float32)).to(dtype).to(tl.float32)
        voltage = (voltage - decrement).to(dtype).to(tl.float32)
    batch = index // (sequence * heads * width)
    token = index // (heads * width) % sequence
    head = index // width % heads
    channel = index % width
    destination = ((batch * heads + head) * sequence + token) * width + channel
    tl.store(output_ptr + destination, (sign_code << 8) | bits, mask=valid)


class _PhaseBitpackedCache(DynamicCache):
    """Store the historical phase FSNeuron's sign and eight events per KV scalar.

    The pinned Transformers 4.40.1 model must retain this cache object; its raw
    int16 entries are not a legacy floating-point KV tuple.
    """

    def __init__(self, layers):
        super().__init__()
        if not hasattr(self, "key_cache"):
            raise RuntimeError("The phase benchmark requires Transformers 4.40.1")
        self._neuron_values = []
        for layer in layers:
            attention = layer.self_attn
            device = attention.q_proj.weight.device
            dtype = attention.q_proj.weight.dtype
            if device.type != "cuda" or dtype != torch.bfloat16:
                raise ValueError("The phase bitpacked cache requires CUDA BF16 weights")
            pair = []
            for identity in (attention.k_Identity, attention.v_Identity):
                neuron = identity.input_quantizer
                if neuron.T != 8 or getattr(neuron, "spike_one", False):
                    raise ValueError(
                        "The phase bitpacked cache requires T=8 and spike_one=False"
                    )
                powers = tuple(
                    torch.stack(
                        [neuron.get_grain_power(step, mode=mode) for step in range(8)]
                    )
                    .reshape(-1)
                    .to(device=device, dtype=dtype)
                    .contiguous()
                    .detach()
                    for mode in ("h", "theta", "d")
                )
                v0 = torch.as_tensor(neuron.v0, device=device, dtype=dtype).detach()
                if any(power.numel() != 8 for power in powers) or v0.numel() != 1:
                    raise ValueError(
                        "The phase bitpacked cache requires scalar neuron powers"
                    )
                pair.append((*powers, v0))
            self._neuron_values.append(tuple(pair))

    def to_legacy_cache(self):
        return self

    def _pack(self, value, parameters):
        batch, time, heads, length, width = value.shape
        if time != 8:
            raise ValueError("The phase bitpacked cache requires eight temporal slices")
        value = value.permute(1, 0, 3, 2, 4).contiguous()
        code = torch.empty(
            (batch, heads, length, width), device=value.device, dtype=torch.int16
        )
        reset, threshold, _, v0 = parameters
        _pack_bits[(triton.cdiv(code.numel(), 256),)](
            value,
            reset,
            threshold,
            v0,
            code,
            code.numel(),
            length,
            heads,
            width,
            256,
            enable_fp_fusion=False,
        )
        return code

    @staticmethod
    def _unpack(code, powers, dtype):
        output = torch.empty((8, *code.shape), device=code.device, dtype=dtype)
        _unpack_bits[(triton.cdiv(output.numel(), 1024),)](
            code, powers, output, code.numel(), 1024
        )
        return output.permute(1, 0, 2, 3, 4)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if torch.is_grad_enabled():
            raise RuntimeError("The phase bitpacked cache is inference-only")
        key = self._pack(key_states, self._neuron_values[layer_idx][0])
        value = self._pack(value_states, self._neuron_values[layer_idx][1])
        if layer_idx == 0:
            self._seen_tokens += key.shape[-2]
        if layer_idx == len(self.key_cache):
            self.key_cache.append(key)
            self.value_cache.append(value)
        else:
            self.key_cache[layer_idx] = torch.cat(
                (self.key_cache[layer_idx], key), dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                (self.value_cache[layer_idx], value), dim=-2
            )
        return (
            self._unpack(
                self.key_cache[layer_idx],
                self._neuron_values[layer_idx][0][2],
                key_states.dtype,
            ),
            self._unpack(
                self.value_cache[layer_idx],
                self._neuron_values[layer_idx][1][2],
                value_states.dtype,
            ),
        )


@contextmanager
def _phase_bitpacked_cache(model):
    """Temporarily connect the pinned single-threaded phase model to its cache."""
    layers = model.model.layers
    cache = _PhaseBitpackedCache(layers)
    identities = [
        identity
        for layer in layers
        for identity in (layer.self_attn.k_Identity, layer.self_attn.v_Identity)
    ]
    previous = [identity.use_act_quant for identity in identities]
    original_descriptor = DynamicCache.__dict__["from_legacy_cache"]
    original_method = DynamicCache.from_legacy_cache

    def preserve_cache(cls, past):
        return cache if past is cache else original_method(past)

    try:
        DynamicCache.from_legacy_cache = classmethod(preserve_cache)
        for identity in identities:
            identity.use_act_quant = False
        yield cache
    finally:
        DynamicCache.from_legacy_cache = original_descriptor
        for identity, enabled in zip(identities, previous):
            identity.use_act_quant = enabled
