from collections import namedtuple
import io
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from spikingjelly.activation_based import base, functional, neuron, surrogate
from spikingjelly.activation_based.ann2snn import (
    Converter,
    STATransformerRecipe,
    TransformerSpikeEquivalentRecipe,
)
from spikingjelly.activation_based.ann2snn.operators import (
    TDConv2d,
    TDGELU,
    TDLayerNorm,
    TDLinear,
    TDMultiheadAttention,
    TDScaledDotProductAttention,
)
from spikingjelly.activation_based.ann2snn.recipes.sta_transformer import (
    _STATIC_TENSOR_KWARGS,
    _STAAnalogLinear,
    _STAOnlineGELU,
    _STAOnlineLayerNorm,
    _STAOnlineMultiheadAttention,
    _STAStatelessTensorOp,
    _STASpikeEncoder,
)


TinyModelOutput = namedtuple("TinyModelOutput", ["logits", "hidden"])


def _first_real_then_zero_sequence(x: torch.Tensor, time_steps: int) -> torch.Tensor:
    steps = [x]
    steps.extend(torch.zeros_like(x) for _ in range(time_steps - 1))
    return torch.stack(steps, dim=0)


def _run_online_steps(module: nn.Module, x_seq: torch.Tensor):
    reset = getattr(module, "_reset_sta_state", None)
    if reset is not None:
        reset()
    outputs = []
    for x in x_seq:
        outputs.append(module(x))
    return torch.stack(outputs, dim=0)


def _sequence_value(value, time_steps: int, preserve_tensor: bool = False):
    if torch.is_tensor(value):
        if not preserve_tensor and not value.is_floating_point():
            raise TypeError("Test helper only sequences floating data tensors.")
        return (
            value
            if preserve_tensor
            else _first_real_then_zero_sequence(value, time_steps)
        )
    if isinstance(value, tuple):
        items = [_sequence_value(x, time_steps, preserve_tensor) for x in value]
        return type(value)(*items) if hasattr(value, "_fields") else tuple(items)
    if isinstance(value, list):
        return [_sequence_value(x, time_steps, preserve_tensor) for x in value]
    if isinstance(value, dict):
        return {
            key: _sequence_value(
                x,
                time_steps,
                preserve_tensor=preserve_tensor or key in _STATIC_TENSOR_KWARGS,
            )
            for key, x in value.items()
        }
    return value


def _sequence_inputs(converted: nn.Module, *args, **kwargs):
    time_steps = converted.time_steps
    return (
        tuple(_sequence_value(arg, time_steps) for arg in args),
        _sequence_value(kwargs, time_steps),
    )


def _slice_step_value(value, step: int, preserve_tensor: bool = False):
    if torch.is_tensor(value):
        return value if preserve_tensor else value[step]
    if isinstance(value, tuple):
        items = [_slice_step_value(x, step, preserve_tensor) for x in value]
        return type(value)(*items) if hasattr(value, "_fields") else tuple(items)
    if isinstance(value, list):
        return [_slice_step_value(x, step, preserve_tensor) for x in value]
    if isinstance(value, dict):
        return {
            key: _slice_step_value(x, step, preserve_tensor)
            for key, x in value.items()
        }
    return value


def _run_converted_step_loop(converted: nn.Module, *args, **kwargs):
    functional.set_step_mode(converted, "s")
    functional.reset_net(converted)
    seq_args, seq_kwargs = _sequence_inputs(converted, *args, **kwargs)
    outputs = []
    for step in range(converted.time_steps):
        step_args = tuple(
            _slice_step_value(arg, step)
            for arg in seq_args
        )
        step_kwargs = {
            key: _slice_step_value(
                value,
                step,
                preserve_tensor=key in _STATIC_TENSOR_KWARGS,
            )
            for key, value in seq_kwargs.items()
        }
        outputs.append(converted(*step_args, **step_kwargs))
    return torch.stack(outputs, dim=0)


def _run_converted_multistep(converted: nn.Module, *args, **kwargs):
    functional.set_step_mode(converted, "m")
    functional.reset_net(converted)
    seq_args, seq_kwargs = _sequence_inputs(converted, *args, **kwargs)
    return converted(*seq_args, **seq_kwargs)


def _sum_time_value(value):
    if torch.is_tensor(value):
        return value.sum(dim=0)
    if isinstance(value, tuple):
        items = [_sum_time_value(x) for x in value]
        return type(value)(*items) if hasattr(value, "_fields") else tuple(items)
    if isinstance(value, list):
        return [_sum_time_value(x) for x in value]
    if isinstance(value, dict):
        return {key: _sum_time_value(x) for key, x in value.items()}
    if value is None:
        return None
    raise TypeError("Test helper can only sum tensor outputs.")


def _run_converted_readout(converted: nn.Module, *args, **kwargs):
    return _sum_time_value(_run_converted_multistep(converted, *args, **kwargs))


def _assert_converted_readout_matches_ann(
    converted: nn.Module,
    model: nn.Module,
    *args,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    **kwargs,
) -> None:
    assert torch.allclose(
        _run_converted_readout(converted, *args, **kwargs),
        model(*args, **kwargs),
        atol=atol,
        rtol=rtol,
    )


def _measure_min_forward_seconds(
    module: nn.Module,
    x: torch.Tensor,
    warmup: int = 3,
    repeat: int = 7,
) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            module(x)
        elapsed = []
        for _ in range(repeat):
            start = time.perf_counter()
            module(x)
            elapsed.append(time.perf_counter() - start)
    return min(elapsed)


def _copy_linear_to_td(source: nn.Linear, target: TDLinear) -> None:
    with torch.no_grad():
        target.weight.copy_(source.weight)
        if source.bias is None:
            assert target.bias is None
        else:
            target.bias.copy_(source.bias)


def _copy_layer_norm_to_td(source: nn.LayerNorm, target: TDLayerNorm) -> None:
    with torch.no_grad():
        if source.weight is None:
            assert target.weight is None
        else:
            target.weight.copy_(source.weight)
        if source.bias is None:
            assert target.bias is None
        else:
            target.bias.copy_(source.bias)


def _copy_mha_to_td(
    source: nn.MultiheadAttention, target: TDMultiheadAttention
) -> None:
    with torch.no_grad():
        q_weight, k_weight, v_weight = source.in_proj_weight.chunk(3, dim=0)
        target.q_proj.weight.copy_(q_weight)
        target.k_proj.weight.copy_(k_weight)
        target.v_proj.weight.copy_(v_weight)
        if source.in_proj_bias is None:
            assert target.q_proj.bias is None
            assert target.k_proj.bias is None
            assert target.v_proj.bias is None
        else:
            q_bias, k_bias, v_bias = source.in_proj_bias.chunk(3, dim=0)
            target.q_proj.bias.copy_(q_bias)
            target.k_proj.bias.copy_(k_bias)
            target.v_proj.bias.copy_(v_bias)
        target.out_proj.weight.copy_(source.out_proj.weight)
        if source.out_proj.bias is None:
            assert target.out_proj.bias is None
        else:
            target.out_proj.bias.copy_(source.out_proj.bias)


def test_sta_sequence_linear_matches_online_linear():
    torch.manual_seed(71)
    source = nn.Linear(5, 7).eval()
    online = _STAAnalogLinear(source).eval()
    sequence = TDLinear(5, 7).eval()
    _copy_linear_to_td(source, sequence)
    x_seq = _first_real_then_zero_sequence(torch.randn(3, 5), time_steps=6)

    online_seq = _run_online_steps(online, x_seq)
    sequence_seq = sequence(x_seq)

    assert torch.allclose(sequence_seq, online_seq, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        sequence_seq.cumsum(dim=0),
        online_seq.cumsum(dim=0),
        atol=1e-6,
        rtol=1e-6,
    )


def test_sta_sequence_layer_norm_matches_online_layer_norm():
    torch.manual_seed(72)
    source = nn.LayerNorm(5).eval()
    online = _STAOnlineLayerNorm(source).eval()
    sequence = TDLayerNorm(5).eval()
    _copy_layer_norm_to_td(source, sequence)
    x_seq = _first_real_then_zero_sequence(torch.randn(3, 4, 5), time_steps=6)

    online_seq = _run_online_steps(online, x_seq)
    sequence_seq = sequence(x_seq)

    assert torch.allclose(sequence_seq, online_seq, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        sequence_seq.cumsum(dim=0),
        online_seq.cumsum(dim=0),
        atol=1e-6,
        rtol=1e-6,
    )


def test_sta_sequence_gelu_matches_online_gelu():
    torch.manual_seed(73)
    source = nn.GELU(approximate="tanh").eval()
    online = _STAOnlineGELU(source).eval()
    sequence = TDGELU(approximate="tanh").eval()
    x_seq = _first_real_then_zero_sequence(torch.randn(3, 4, 5), time_steps=6)

    online_seq = _run_online_steps(online, x_seq)
    sequence_seq = sequence(x_seq)

    assert torch.allclose(sequence_seq, online_seq, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        sequence_seq.cumsum(dim=0),
        online_seq.cumsum(dim=0),
        atol=1e-6,
        rtol=1e-6,
    )


def test_sta_sequence_mha_matches_online_mha():
    torch.manual_seed(74)
    source = nn.MultiheadAttention(
        embed_dim=8,
        num_heads=2,
        dropout=0.0,
        batch_first=True,
    ).eval()
    online = _STAOnlineMultiheadAttention(source).eval()
    sequence = TDMultiheadAttention(embed_dim=8, num_heads=2).eval()
    _copy_mha_to_td(source, sequence)
    x_seq = _first_real_then_zero_sequence(torch.randn(2, 4, 8), time_steps=5)

    online_outputs = []
    online._reset_sta_state()
    for x in x_seq:
        y, weights = online(x, x, x, need_weights=False)
        assert weights is None
        online_outputs.append(y)
    online_seq = torch.stack(online_outputs, dim=0)
    sequence_seq, sequence_weights = sequence(
        x_seq, x_seq, x_seq, need_weights=False
    )

    assert sequence_weights is None
    assert torch.allclose(sequence_seq, online_seq, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        sequence_seq.cumsum(dim=0),
        online_seq.cumsum(dim=0),
        atol=1e-6,
        rtol=1e-6,
    )


def test_sta_sequence_spike_encoder_clamps_zero_threshold():
    encoder = _STASpikeEncoder(torch.zeros(3), channel_dim=-1, step_mode="m")
    x_seq = torch.zeros(4, 2, 3)

    y_seq = encoder(x_seq)

    assert torch.isfinite(y_seq).all()
    assert torch.count_nonzero(y_seq) == 0


def test_sta_sequence_spike_encoder_rejects_invalid_multistep_input():
    encoder = _STASpikeEncoder(torch.ones(3), channel_dim=-1, step_mode="m")

    with pytest.raises(ValueError, match="at least one data dimension"):
        encoder(torch.zeros(4))
    with pytest.raises(ValueError, match="non-empty time dimension"):
        encoder(torch.zeros(0, 2, 3))


class TinyUnsupportedSequenceOpClassifier(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1)


def _activation_aware_calibration_channel_last(
    activation: torch.Tensor,
    threshold_std_scale: float = 3.0,
    eps: float = 1e-6,
):
    reduce_dims = tuple(range(activation.dim() - 1))
    offset = activation.mean(dim=reduce_dims)
    threshold = activation.std(dim=reduce_dims, unbiased=False) * threshold_std_scale
    threshold = torch.clamp(threshold, min=eps)
    return threshold.detach(), offset.detach()


class TinyTDTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.norm1 = TDLayerNorm(embed_dim)
        self.q_proj = TDLinear(embed_dim, embed_dim)
        self.k_proj = TDLinear(embed_dim, embed_dim)
        self.v_proj = TDLinear(embed_dim, embed_dim)
        self.attn = TDScaledDotProductAttention()
        self.out_proj = TDLinear(embed_dim, embed_dim)
        self.norm2 = TDLayerNorm(embed_dim)
        self.fc1 = TDLinear(embed_dim, mlp_dim)
        self.act = TDGELU()
        self.fc2 = TDLinear(mlp_dim, embed_dim)

    def _split_heads(self, x_seq: torch.Tensor) -> torch.Tensor:
        t, batch_size, seq_len, _ = x_seq.shape
        x_seq = x_seq.reshape(t, batch_size, seq_len, self.num_heads, self.head_dim)
        return x_seq.transpose(2, 3)

    def _merge_heads(self, x_seq: torch.Tensor) -> torch.Tensor:
        t, batch_size, _, seq_len, _ = x_seq.shape
        x_seq = x_seq.transpose(2, 3).contiguous()
        return x_seq.reshape(t, batch_size, seq_len, self.embed_dim)

    @staticmethod
    def _ann_linear(module: TDLinear, x_cum: torch.Tensor) -> torch.Tensor:
        return F.linear(x_cum, module.weight, module.bias)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        norm_seq = self.norm1(x_seq)
        q_seq = self._split_heads(self.q_proj(norm_seq))
        k_seq = self._split_heads(self.k_proj(norm_seq))
        v_seq = self._split_heads(self.v_proj(norm_seq))
        attn_seq = self._merge_heads(self.attn(q_seq, k_seq, v_seq))
        x_seq = x_seq + self.out_proj(attn_seq)

        mlp_seq = self.fc1(self.norm2(x_seq))
        mlp_seq = self.act(mlp_seq)
        mlp_seq = self.fc2(mlp_seq)
        return x_seq + mlp_seq

    def ann_reference(self, x_cum: torch.Tensor) -> torch.Tensor:
        norm_cum = F.layer_norm(
            x_cum,
            self.norm1.normalized_shape,
            self.norm1.weight,
            self.norm1.bias,
            self.norm1.eps,
        )
        q_cum = self._split_heads(self._ann_linear(self.q_proj, norm_cum))
        k_cum = self._split_heads(self._ann_linear(self.k_proj, norm_cum))
        v_cum = self._split_heads(self._ann_linear(self.v_proj, norm_cum))
        attn_cum = F.scaled_dot_product_attention(
            q_cum,
            k_cum,
            v_cum,
            dropout_p=0.0,
        )
        x_cum = x_cum + self._ann_linear(self.out_proj, self._merge_heads(attn_cum))

        mlp_cum = F.layer_norm(
            x_cum,
            self.norm2.normalized_shape,
            self.norm2.weight,
            self.norm2.bias,
            self.norm2.eps,
        )
        mlp_cum = self._ann_linear(self.fc1, mlp_cum)
        mlp_cum = F.gelu(mlp_cum, approximate=self.act.approximate)
        mlp_cum = self._ann_linear(self.fc2, mlp_cum)
        return x_cum + mlp_cum


class TinyActivationAwareTDTransformerBlock(TinyTDTransformerBlock):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        v_threshold: torch.Tensor,
        v_offset: torch.Tensor,
        surrogate_function: surrogate.SurrogateFunctionBase,
    ) -> None:
        super().__init__(embed_dim, num_heads, mlp_dim)
        self.activation_neuron = neuron.ActivationAwareIFNode(
            v_threshold=v_threshold,
            v_offset=v_offset,
            channel_dim=-1,
            surrogate_function=surrogate_function,
            step_mode="m",
        )
        self.hidden_spike_seq = None

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        norm_seq = self.norm1(x_seq)
        q_seq = self._split_heads(self.q_proj(norm_seq))
        k_seq = self._split_heads(self.k_proj(norm_seq))
        v_seq = self._split_heads(self.v_proj(norm_seq))
        attn_seq = self._merge_heads(self.attn(q_seq, k_seq, v_seq))
        x_seq = x_seq + self.out_proj(attn_seq)

        mlp_seq = self.fc1(self.norm2(x_seq))
        mlp_seq = self.act(mlp_seq)
        mlp_seq = self.activation_neuron(mlp_seq)
        self.hidden_spike_seq = mlp_seq
        mlp_seq = self.fc2(mlp_seq)
        return x_seq + mlp_seq


class TinyANNFunctionalSDPATransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.norm1 = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, embed_dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.unflatten(-1, (self.num_heads, self.head_dim)).transpose(-3, -2)

    @staticmethod
    def _merge_heads(x: torch.Tensor) -> torch.Tensor:
        return x.transpose(-3, -2).flatten(-2)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        norm = self.norm1(x)
        q = self._split_heads(self.q_proj(norm))
        k = self._split_heads(self.k_proj(norm))
        v = self._split_heads(self.v_proj(norm))
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
        )
        x = x + self.out_proj(self._merge_heads(attn))
        return x + self.fc2(self.act(self.fc1(self.norm2(x))))


class TinyANNMHATransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        norm = self.norm1(x)
        attn, _ = self.attn(
            norm,
            norm,
            norm,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = x + attn
        return x + self.fc2(self.act(self.fc1(self.norm2(x))))


class TinyANNFFNBlock(nn.Module):
    def __init__(self, embed_dim: int = 16, mlp_dim: int = 32, depth: int = 2) -> None:
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, mlp_dim),
                    nn.GELU(),
                    nn.Linear(mlp_dim, embed_dim),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyImageTransformerClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch = nn.Conv2d(3, 8, kernel_size=4, stride=4)
        self.norm = nn.LayerNorm(8)
        self.fc1 = nn.Linear(8, 16)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(16, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = self.fc2(self.act(self.fc1(x)))
        return x.mean(dim=1)


class TinyFunctionalViewTransformerClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch = nn.Conv2d(3, 8, kernel_size=4, stride=4)
        self.norm = nn.LayerNorm(8)
        self.fc1 = nn.Linear(8, 16)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(16, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.transpose(torch.flatten(self.patch(x), start_dim=2), 1, 2)
        x = self.norm(x)
        x = self.fc2(self.act(self.fc1(x)))
        return x.mean(dim=1)


class TinyTensorOpAdapterModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        x = torch.flatten(x, start_dim=2)
        return x.mean(dim=1)


class TinyFunctionalTensorOpAdapterModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.unsqueeze(x, dim=1)
        x = torch.transpose(input=x, dim0=2, dim1=3)
        x = torch.flatten(input=x, start_dim=2)
        return torch.mean(input=x, dim=1)


class TinyConstantTransformerClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias_token = nn.Parameter(torch.randn(1, 1, 8))
        self.norm = nn.LayerNorm(8)
        self.fc = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.bias_token
        return self.fc(self.norm(x)).mean(dim=1)


class TinyMaskBufferTransformerClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("attention_mask", torch.tensor([[True, False]]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.masked_fill(self.attention_mask.unsqueeze(-1), 0.0).mean(dim=1)


class TinyFloatMaskBufferTransformerClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("padding_mask", torch.zeros(1, 1, 2, 2))
        self.padding_mask[..., 1] = -10000.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.scaled_dot_product_attention(
            x.unsqueeze(1),
            x.unsqueeze(1),
            x.unsqueeze(1),
            attn_mask=self.padding_mask,
            dropout_p=0.0,
        ).squeeze(1)
        return x.mean(dim=1)


class TinyPositionalSDPAMaskTransformerClassifier(nn.Module):
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = F.scaled_dot_product_attention(
            x.unsqueeze(1),
            x.unsqueeze(1),
            x.unsqueeze(1),
            attn_mask,
            0.0,
        ).squeeze(1)
        return x.mean(dim=1)


class TinyKeywordTransformerClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4)
        self.fc = nn.Linear(4, 3)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.fc(self.norm(pixel_values)).mean(dim=1)


class TinyNamedTupleOutputTransformerClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4)
        self.fc = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> TinyModelOutput:
        hidden = self.norm(x)
        logits = self.fc(hidden).mean(dim=1)
        return TinyModelOutput(logits=logits, hidden=hidden)


class TinyKeywordDictOutputTransformerClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4)
        self.fc = nn.Linear(4, 3)

    def forward(self, pixel_values: torch.Tensor) -> dict:
        logits = self.fc(self.norm(pixel_values)).mean(dim=1)
        return {"logits": logits}


class TinyConvPaddingClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch = nn.Conv2d(
            3,
            4,
            kernel_size=3,
            padding="same",
            padding_mode="reflect",
        )
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x).mean(dim=(2, 3))
        return self.fc(x)


class TinyHeadClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4)
        self.body = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(self.norm(x))).mean(dim=1)


class TinyMHAWeightsBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            8,
            2,
            dropout=0.0,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            need_weights=True,
        )


class TinyMHADefaultWeightsBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            8,
            2,
            dropout=0.0,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.attn(x, x, x)
        return output.mean(dim=1)


class TinyPositionalMHAMaskBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            8,
            2,
            dropout=0.0,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output, _ = self.attn(
            x,
            x,
            x,
            key_padding_mask,
            False,
            attn_mask,
        )
        return output.mean(dim=1)


class TinyFunctionalLinearClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3, 4))
        self.bias = nn.Parameter(torch.randn(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias).mean(dim=1)


class TinyFunctionalLinearKwargClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3, 4))
        self.bias = nn.Parameter(torch.randn(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, weight=self.weight, bias=self.bias).mean(dim=1)


class TinyTwoInputClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4)
        self.fc = nn.Linear(4, 2)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return self.fc(self.norm(x0 + x1)).mean(dim=1)


class TinyFiveInputClassifier(nn.Module):
    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
    ) -> torch.Tensor:
        return (x0 + x1 + x2 + x3 + x4).mean(dim=1)


def _apply_ann_to_cumulative(
    block: nn.Module,
    x_seq: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    x_cum = x_seq.cumsum(dim=0)
    return torch.stack([block(x_cum[t], **kwargs) for t in range(x_seq.shape[0])])


def test_tiny_transformer_block_matches_ann_reference_on_cumulative_input():
    block = TinyTDTransformerBlock(embed_dim=8, num_heads=2, mlp_dim=16)
    x_seq = torch.randn(5, 2, 4, 8)

    y_seq = block(x_seq)
    expected = block.ann_reference(x_seq.cumsum(dim=0))

    assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-5, rtol=1e-5)


def test_tiny_transformer_block_final_cumulative_output_matches_total_input():
    block = TinyTDTransformerBlock(embed_dim=8, num_heads=2, mlp_dim=16)
    x_seq = torch.randn(4, 2, 5, 8)

    y_seq = block(x_seq)
    expected = block.ann_reference(x_seq.cumsum(dim=0))

    assert torch.allclose(
        y_seq.cumsum(dim=0)[-1],
        expected[-1],
        atol=1e-5,
        rtol=1e-5,
    )


def test_tiny_transformer_block_uses_affine_td_linear_layers():
    block = TinyTDTransformerBlock(embed_dim=8, num_heads=2, mlp_dim=16)

    linear_layers = [
        block.q_proj,
        block.k_proj,
        block.v_proj,
        block.out_proj,
        block.fc1,
        block.fc2,
    ]

    assert all(layer.bias is not None for layer in linear_layers)


def test_tiny_transformer_block_autograd_smoke():
    block = TinyTDTransformerBlock(embed_dim=8, num_heads=2, mlp_dim=16)
    x_seq = torch.randn(3, 2, 4, 8, requires_grad=True)

    y_seq = block(x_seq)
    y_seq.square().sum().backward()

    assert x_seq.grad is not None
    assert torch.isfinite(x_seq.grad).all()
    for parameter in block.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert block.q_proj.bias.grad is not None
    assert torch.isfinite(block.q_proj.bias.grad).all()


def test_activation_aware_calibration_for_tiny_transformer_hidden_activation():
    torch.manual_seed(0)
    probe_block = TinyTDTransformerBlock(embed_dim=8, num_heads=2, mlp_dim=16)
    x_seq = torch.randn(5, 2, 4, 8)
    hidden_seq = probe_block.act(probe_block.fc1(probe_block.norm2(x_seq)))

    threshold, offset = _activation_aware_calibration_channel_last(hidden_seq)

    assert threshold.shape == (16,)
    assert offset.shape == (16,)
    assert torch.isfinite(threshold).all()
    assert torch.isfinite(offset).all()
    assert (threshold > 0).all()


def test_activation_aware_tiny_transformer_block_forward_and_spike_sanity():
    torch.manual_seed(1)
    probe_block = TinyTDTransformerBlock(embed_dim=8, num_heads=2, mlp_dim=16)
    x_seq = torch.linspace(-1.0, 1.0, steps=5 * 2 * 4 * 8).view(5, 2, 4, 8)
    hidden_seq = probe_block.act(probe_block.fc1(probe_block.norm2(x_seq)))
    threshold, offset = _activation_aware_calibration_channel_last(hidden_seq)
    block = TinyActivationAwareTDTransformerBlock(
        embed_dim=8,
        num_heads=2,
        mlp_dim=16,
        v_threshold=threshold,
        v_offset=offset,
        surrogate_function=surrogate.DeterministicPass(),
    )

    y_seq = block(x_seq)

    assert y_seq.shape == x_seq.shape
    assert torch.isfinite(y_seq).all()
    assert torch.isfinite(y_seq.cumsum(dim=0)).all()
    assert block.activation_neuron.v_threshold.shape == (16,)
    assert block.activation_neuron.v_offset.shape == (16,)
    assert block.hidden_spike_seq is not None
    assert torch.isfinite(block.hidden_spike_seq).all()
    assert block.hidden_spike_seq.sum() > 0

    functional.reset_net(block)
    y_seq_after_reset = block(x_seq)
    assert torch.allclose(y_seq, y_seq_after_reset)


def test_activation_aware_tiny_transformer_block_autograd_smoke():
    torch.manual_seed(2)
    probe_block = TinyTDTransformerBlock(embed_dim=8, num_heads=2, mlp_dim=16)
    calibration_seq = torch.randn(4, 2, 3, 8)
    hidden_seq = probe_block.act(probe_block.fc1(probe_block.norm2(calibration_seq)))
    threshold, offset = _activation_aware_calibration_channel_last(hidden_seq)
    block = TinyActivationAwareTDTransformerBlock(
        embed_dim=8,
        num_heads=2,
        mlp_dim=16,
        v_threshold=threshold,
        v_offset=offset,
        surrogate_function=surrogate.Sigmoid(),
    )
    x_seq = torch.randn(4, 2, 3, 8, requires_grad=True)

    y_seq = block(x_seq)
    y_seq.square().mean().backward()

    assert x_seq.grad is not None
    assert torch.isfinite(x_seq.grad).all()
    for parameter in block.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_converter_functional_sdpa_transformer_block_matches_ann_reference():
    block = TinyANNFunctionalSDPATransformerBlock(
        embed_dim=8,
        num_heads=2,
        mlp_dim=16,
    )
    block.eval()
    converted = Converter(recipe=TransformerSpikeEquivalentRecipe()).convert(block)
    modules = dict(converted.named_modules())
    x_seq = torch.randn(5, 2, 4, 8)
    attn_mask = torch.tensor(
        [
            [0.0, 0.0, float("-inf"), float("-inf")],
            [0.0, 0.0, 0.0, float("-inf")],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    y_seq = converted(x_seq, attn_mask)
    expected = block(x_seq.cumsum(dim=0), attn_mask=attn_mask)

    assert isinstance(modules["norm1"], TDLayerNorm)
    assert isinstance(modules["q_proj"], TDLinear)
    assert isinstance(modules["act"], TDGELU)
    assert any(
        isinstance(module, TDScaledDotProductAttention) for module in modules.values()
    )
    assert not any(
        node.op == "call_function" and node.target is F.scaled_dot_product_attention
        for node in converted.graph.nodes
    )
    assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-5, rtol=1e-5)


def test_converter_mha_transformer_block_matches_ann_reference():
    block = TinyANNMHATransformerBlock(embed_dim=8, num_heads=2, mlp_dim=16)
    block.eval()
    converted = Converter(recipe=TransformerSpikeEquivalentRecipe()).convert(block)
    modules = dict(converted.named_modules())
    x_seq = torch.randn(4, 2, 5, 8)
    attn_mask = torch.zeros(5, 5)
    attn_mask[:, -1] = float("-inf")

    y_seq = converted(x_seq, attn_mask)
    expected = _apply_ann_to_cumulative(block, x_seq, attn_mask=attn_mask)

    assert isinstance(modules["norm1"], TDLayerNorm)
    assert isinstance(modules["attn"], TDMultiheadAttention)
    assert isinstance(modules["fc1"], TDLinear)
    assert isinstance(modules["act"], TDGELU)
    assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-5, rtol=1e-5)


def test_converter_transformer_block_autograd_smoke():
    block = TinyANNMHATransformerBlock(embed_dim=8, num_heads=2, mlp_dim=16)
    converted = Converter(recipe=TransformerSpikeEquivalentRecipe()).convert(block)
    x_seq = torch.randn(3, 2, 4, 8, requires_grad=True)

    y_seq = converted(x_seq)
    y_seq.square().sum().backward()

    assert x_seq.grad is not None
    assert torch.isfinite(x_seq.grad).all()
    for parameter in converted.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_sta_transformer_recipe_converts_affine_layers_and_runs_inner_steps():
    torch.manual_seed(3)
    model = TinyImageTransformerClassifier().eval()
    calibration = [(torch.randn(2, 3, 16, 16), torch.zeros(2, dtype=torch.long))]
    recipe = STATransformerRecipe(
        dataloader=calibration,
        time_steps=4,
        threshold_mode="mse",
    )

    converted = Converter(recipe=recipe).convert(model)
    modules = dict(converted.named_modules())
    x = torch.randn(2, 3, 16, 16)
    y_seq = _run_converted_multistep(converted, x)
    y = y_seq.sum(dim=0)

    assert converted.time_steps == 4
    assert not hasattr(modules["patch"], "v_threshold")
    assert not hasattr(modules["fc1"], "v_threshold")
    assert not hasattr(modules["fc2"], "v_threshold")
    assert y_seq.shape == (4, 2, 5)
    assert y.shape == (2, 5)
    assert torch.isfinite(y).all()


def test_sta_transformer_recipe_converted_model_is_plain_module():
    converted = Converter(recipe=STATransformerRecipe(time_steps=4)).convert(
        TinyANNFFNBlock(embed_dim=8, mlp_dim=16).eval()
    )

    assert isinstance(converted, nn.Module)
    assert not isinstance(converted, base.StepModule)
    assert not isinstance(converted, base.MemoryModule)
    assert not hasattr(converted, "step_mode")
    assert not hasattr(converted, "encode_inputs")
    assert not hasattr(converted, "sum_time")
    assert not hasattr(converted, "add_outputs")


def test_sta_transformer_test_sequence_helper_rejects_nonfloating_data():
    with pytest.raises(TypeError, match="floating data tensors"):
        _sequence_value(torch.ones(2, 4, dtype=torch.long), time_steps=4)


def test_sta_transformer_recipe_set_step_mode_reaches_inner_modules():
    converted = Converter(recipe=STATransformerRecipe(time_steps=4)).convert(
        TinyTensorOpAdapterModel().eval()
    )
    step_modules = [m for m in converted.modules() if isinstance(m, base.StepModule)]

    assert len(step_modules) > 0

    functional.set_step_mode(converted, "s")
    for module in step_modules:
        assert module.step_mode == "s"
    functional.set_step_mode(converted, "m")
    for module in step_modules:
        assert module.step_mode == "m"


def test_sta_transformer_recipe_tensor_ops_single_loop_match_multistep():
    torch.manual_seed(91)
    converted = Converter(recipe=STATransformerRecipe(time_steps=4)).convert(
        TinyTensorOpAdapterModel().eval()
    )
    x = torch.randn(2, 3, 4)
    op_names = {
        module.op_name
        for module in converted.modules()
        if isinstance(module, _STAStatelessTensorOp)
    }

    assert {"mean", "flatten", "transpose", "unsqueeze"}.issubset(op_names)
    assert torch.allclose(
        _run_converted_multistep(converted, x),
        _run_converted_step_loop(converted, x),
        atol=1e-6,
        rtol=1e-6,
    )


def test_sta_transformer_recipe_functional_tensor_ops_match_multistep():
    torch.manual_seed(92)
    converted = Converter(recipe=STATransformerRecipe(time_steps=4)).convert(
        TinyFunctionalTensorOpAdapterModel().eval()
    )
    x = torch.randn(2, 3, 4)

    assert torch.allclose(
        _run_converted_multistep(converted, x),
        _run_converted_step_loop(converted, x),
        atol=1e-6,
        rtol=1e-6,
    )


def test_sta_transformer_recipe_stateless_tensor_ops_are_picklable():
    converted = Converter(recipe=STATransformerRecipe(time_steps=4)).convert(
        TinyTensorOpAdapterModel().eval()
    )

    torch.save(converted, io.BytesIO())


def test_sta_transformer_recipe_mha_single_loop_matches_multistep():
    torch.manual_seed(75)
    model = TinyANNMHATransformerBlock(embed_dim=8, num_heads=2, mlp_dim=16).eval()
    calibration = [(torch.randn(2, 4, 8),)]
    converted = Converter(
        recipe=STATransformerRecipe(
            dataloader=calibration,
            time_steps=5,
            mode="equivalent",
        )
    ).convert(model)
    x = torch.randn(2, 4, 8)

    single_seq = _run_converted_step_loop(converted, x)
    multi_seq = _run_converted_multistep(converted, x)

    assert torch.allclose(multi_seq, single_seq, atol=1e-5, rtol=1e-5)


def test_sta_transformer_recipe_multistep_timing_smoke_matches_single_loop():
    torch.manual_seed(750)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = TinyANNFFNBlock(embed_dim=16, mlp_dim=32, depth=2).eval()
        time_steps = 128
        converted = Converter(
            recipe=STATransformerRecipe(
                time_steps=time_steps,
                mode="equivalent",
            )
        ).convert(model)
        x = torch.randn(1, 4, 16)

        single_seq = _run_converted_step_loop(converted, x)
        multi_seq = _run_converted_multistep(converted, x)
        assert torch.allclose(multi_seq, single_seq, atol=1e-5, rtol=1e-5)

        seq_args, seq_kwargs = _sequence_inputs(converted, x)
        functional.set_step_mode(converted, "m")
        sequence_seconds = _measure_min_forward_seconds(
            converted, *seq_args, **seq_kwargs
        )
        functional.set_step_mode(converted, "s")
        single_seconds = _measure_min_forward_seconds(
            lambda value: _run_converted_step_loop(converted, value), x
        )

        assert single_seconds > 0
        assert sequence_seconds > 0
    finally:
        torch.set_num_threads(old_threads)


def test_sta_transformer_recipe_spiking_encoder_single_loop_matches_multistep():
    torch.manual_seed(76)
    model = TinyANNMHATransformerBlock(embed_dim=8, num_heads=2, mlp_dim=16).eval()
    calibration = [(torch.randn(2, 4, 8),)]
    converted = Converter(
        recipe=STATransformerRecipe(
            dataloader=calibration,
            time_steps=5,
            mode="spiking_encoder",
        )
    ).convert(model)
    x = torch.randn(2, 4, 8)

    single_seq = _run_converted_step_loop(converted, x)
    multi_seq = _run_converted_multistep(converted, x)

    assert torch.allclose(multi_seq, single_seq, atol=1e-5, rtol=1e-5)


def test_sta_transformer_recipe_vit_shaped_classifier_single_loop_matches_multistep():
    torch.manual_seed(77)
    model = TinyImageTransformerClassifier().eval()
    calibration = [(torch.randn(2, 3, 16, 16),)]
    converted = Converter(
        recipe=STATransformerRecipe(
            dataloader=calibration,
            time_steps=4,
            mode="spiking_encoder",
        )
    ).convert(model)
    x = torch.randn(2, 3, 16, 16)

    single_seq = _run_converted_step_loop(converted, x)
    multi_seq = _run_converted_multistep(converted, x)

    assert torch.allclose(multi_seq, single_seq, atol=1e-5, rtol=1e-5)


def test_sta_transformer_recipe_replaces_conv2d_with_td_conv2d():
    model = TinyImageTransformerClassifier().eval()

    converted = Converter(
        recipe=STATransformerRecipe(
            dataloader=[(torch.randn(2, 3, 16, 16),)],
            time_steps=4,
            mode="spiking_encoder",
        )
    ).convert(model)

    assert isinstance(dict(converted.named_modules())["patch"], TDConv2d)


def test_sta_transformer_recipe_functional_view_single_loop_matches_multistep():
    torch.manual_seed(78)
    model = TinyFunctionalViewTransformerClassifier().eval()
    calibration = [(torch.randn(2, 3, 16, 16),)]
    converted = Converter(
        recipe=STATransformerRecipe(
            dataloader=calibration,
            time_steps=4,
            mode="spiking_encoder",
        )
    ).convert(model)
    x = torch.randn(2, 3, 16, 16)

    assert torch.allclose(
        _run_converted_multistep(converted, x),
        _run_converted_step_loop(converted, x),
        atol=1e-5,
        rtol=1e-5,
    )


def test_sta_transformer_recipe_rejects_spiking_affine():
    model = TinyImageTransformerClassifier().eval()

    with pytest.raises(ValueError, match="spiking affine"):
        Converter(
            recipe=STATransformerRecipe(
                dataloader=[(torch.randn(2, 3, 16, 16),)],
                time_steps=4,
                mode="spiking_affine",
            )
        ).convert(model)


def test_sta_transformer_recipe_rejects_unsupported_method():
    model = TinyUnsupportedSequenceOpClassifier().eval()

    with pytest.raises(ValueError, match="tensor method 'permute'"):
        Converter(
            recipe=STATransformerRecipe(
                time_steps=4,
                mode="equivalent",
            )
        ).convert(model)


def test_sta_transformer_recipe_mha_default_need_weights_runs():
    torch.manual_seed(79)
    model = TinyMHADefaultWeightsBlock().eval()
    x = torch.randn(2, 4, 8)
    converted = Converter(
        recipe=STATransformerRecipe(
            time_steps=4,
            mode="equivalent",
        )
    ).convert(model)

    assert torch.allclose(
        _run_converted_multistep(converted, x).sum(dim=0),
        _run_converted_step_loop(converted, x).sum(dim=0),
        atol=1e-5,
        rtol=1e-5,
    )


def test_sta_transformer_recipe_rejects_mha_attention_weights():
    model = TinyMHAWeightsBlock().eval()

    with pytest.raises(ValueError, match="need_weights=False|attention weights"):
        Converter(
            recipe=STATransformerRecipe(
                time_steps=4,
                mode="equivalent",
            )
        ).convert(model)


def test_sta_transformer_recipe_equivalent_replaces_mha_with_td_mha():
    model = TinyANNMHATransformerBlock(embed_dim=8, num_heads=2, mlp_dim=16).eval()

    converted = Converter(
        recipe=STATransformerRecipe(
            time_steps=4,
            mode="equivalent",
        )
    ).convert(model)

    assert isinstance(dict(converted.named_modules())["attn"], TDMultiheadAttention)


def test_sta_transformer_recipe_spiking_encoder_stacks_mha_encoder():
    model = TinyANNMHATransformerBlock(embed_dim=8, num_heads=2, mlp_dim=16).eval()

    converted = Converter(
        recipe=STATransformerRecipe(
            dataloader=[(torch.randn(2, 4, 8),)],
            time_steps=4,
            mode="spiking_encoder",
        )
    ).convert(model)
    modules = dict(converted.named_modules())

    assert isinstance(modules["attn"], TDMultiheadAttention)
    assert hasattr(modules["getitem_sta_encoder"], "v_threshold")


def test_sta_transformer_recipe_rejects_key_padding_mask():
    model = TinyPositionalMHAMaskBlock().eval()

    with pytest.raises(ValueError, match="key_padding_mask"):
        Converter(
            recipe=STATransformerRecipe(
                time_steps=4,
                mode="equivalent",
            )
        ).convert(model)


def test_sta_transformer_recipe_forward_does_not_auto_reset_state():
    torch.manual_seed(31)
    model = TinyImageTransformerClassifier().eval()
    calibration = [(torch.randn(2, 3, 16, 16),)]
    converted = Converter(
        recipe=STATransformerRecipe(dataloader=calibration, time_steps=4)
    ).convert(model)
    x = torch.randn(2, 3, 16, 16)

    y0_seq = _run_converted_multistep(converted, x)
    seq_args, seq_kwargs = _sequence_inputs(converted, x)
    y1_seq_without_reset = converted(*seq_args, **seq_kwargs)
    functional.reset_net(converted)
    y1_seq_after_reset = converted(*seq_args, **seq_kwargs)

    assert not torch.allclose(y0_seq, y1_seq_without_reset)
    assert torch.allclose(y0_seq, y1_seq_after_reset, atol=1e-6, rtol=1e-6)


def test_sta_transformer_recipe_reset_net_resets_inner_memory_modules():
    torch.manual_seed(310)
    model = TinyImageTransformerClassifier().eval()
    calibration = [(torch.randn(2, 3, 16, 16),)]
    converted = Converter(
        recipe=STATransformerRecipe(dataloader=calibration, time_steps=4)
    ).convert(model)
    x = torch.randn(2, 3, 16, 16)
    functional.set_step_mode(converted, "m")
    seq_args, seq_kwargs = _sequence_inputs(converted, x)

    y0_seq = converted(*seq_args, **seq_kwargs)
    functional.reset_net(converted)
    y1_seq = converted(*seq_args, **seq_kwargs)

    assert torch.allclose(y0_seq, y1_seq, atol=1e-6, rtol=1e-6)


def test_sta_transformer_recipe_preserves_final_wrapper_training_flag():
    model = TinyImageTransformerClassifier().train()
    calibration = [(torch.randn(2, 3, 16, 16),)]

    converted = Converter(
        recipe=STATransformerRecipe(dataloader=calibration, time_steps=4)
    ).convert(model)

    assert converted.training is False
    assert model.training is True


def test_sta_transformer_recipe_equivalent_mode_skips_calibration_loop():
    class CountingCalibration:
        def __iter__(self):
            raise AssertionError("equivalent mode should not iterate calibration")

    model = TinyImageTransformerClassifier().eval()

    converted = Converter(
        recipe=STATransformerRecipe(
            dataloader=CountingCalibration(),
            time_steps=4,
            mode="equivalent",
        )
    ).convert(model)

    assert torch.isfinite(
        _run_converted_readout(converted, torch.randn(2, 3, 16, 16))
    ).all()


def test_sta_transformer_recipe_preserves_nonzero_conv_padding_mode():
    torch.manual_seed(39)
    model = TinyConvPaddingClassifier().eval()
    calibration = [(torch.randn(2, 3, 8, 8),)]
    converted = Converter(
        recipe=STATransformerRecipe(dataloader=calibration, time_steps=4)
    ).convert(model)
    x = torch.randn(2, 3, 8, 8)

    _assert_converted_readout_matches_ann(converted, model, x)


def test_sta_transformer_recipe_equivalent_mha_matches_ann():
    torch.manual_seed(32)
    model = TinyANNMHATransformerBlock(embed_dim=8, num_heads=2, mlp_dim=16).eval()
    calibration = [(torch.randn(2, 4, 8),)]
    converted = Converter(
        recipe=STATransformerRecipe(dataloader=calibration, time_steps=4)
    ).convert(model)
    modules = dict(converted.named_modules())
    x = torch.randn(2, 4, 8)

    _assert_converted_readout_matches_ann(converted, model, x)
    assert isinstance(modules["attn"], TDMultiheadAttention)
    assert modules["attn"].batch_first is True


def test_sta_transformer_recipe_preserves_static_attention_mask_kwargs():
    torch.manual_seed(40)
    model = TinyANNMHATransformerBlock(embed_dim=8, num_heads=2, mlp_dim=16).eval()
    calibration = [(torch.randn(2, 4, 8),)]
    converted = Converter(
        recipe=STATransformerRecipe(dataloader=calibration, time_steps=4)
    ).convert(model)
    x = torch.randn(2, 4, 8)
    attn_mask = torch.zeros(4, 4)
    attn_mask[:, -1] = float("-inf")

    assert torch.allclose(
        _run_converted_readout(converted, x, attn_mask=attn_mask),
        model(x, attn_mask=attn_mask),
        atol=1e-5,
        rtol=1e-5,
    )


def test_sta_transformer_recipe_returns_attention_weight_deltas():
    model = TinyMHAWeightsBlock().eval()

    with pytest.raises(ValueError, match="attention weights|need_weights"):
        Converter(recipe=STATransformerRecipe(time_steps=4)).convert(model)


def test_sta_transformer_recipe_rejects_functional_linear():
    torch.manual_seed(42)
    model = TinyFunctionalLinearClassifier().eval()

    with pytest.raises(ValueError, match="function node.*linear"):
        Converter(recipe=STATransformerRecipe(time_steps=4)).convert(model)


def test_sta_transformer_recipe_rejects_functional_keyword_linear():
    torch.manual_seed(13)
    model = TinyFunctionalLinearKwargClassifier().eval()

    with pytest.raises(ValueError, match="function node.*linear"):
        Converter(recipe=STATransformerRecipe(time_steps=4)).convert(model)


def test_sta_transformer_recipe_calibrates_explicit_multi_input_batch():
    torch.manual_seed(43)
    model = TinyTwoInputClassifier().eval()
    x0 = torch.randn(2, 4, 4)
    x1 = torch.randn(2, 4, 4)
    calibration = [((x0, x1), torch.zeros(2, dtype=torch.long))]

    converted = Converter(
        recipe=STATransformerRecipe(
            dataloader=calibration,
            time_steps=4,
            mode="spiking_encoder",
        )
    ).convert(model)

    assert torch.isfinite(_run_converted_readout(converted, x0, x1)).all()


def test_sta_transformer_recipe_state_buffers_rebuild_after_dtype_change():
    torch.manual_seed(44)
    model = TinyImageTransformerClassifier().eval()
    calibration = [(torch.randn(2, 3, 16, 16),)]
    converted = Converter(
        recipe=STATransformerRecipe(
            dataloader=calibration,
            time_steps=4,
            mode="spiking_encoder",
        )
    ).convert(model)
    act = dict(converted.named_modules())["act"]

    act(torch.randn(2, 4, 16))
    act.double()
    y = act(torch.randn(2, 4, 16, dtype=torch.float64))

    assert y.dtype == torch.float64
    assert torch.isfinite(y).all()


def test_sta_transformer_recipe_spiking_encoder_mode_encodes_nonlinear_outputs():
    torch.manual_seed(8)
    model = TinyImageTransformerClassifier().eval()
    calibration = [(torch.randn(2, 3, 16, 16),)]

    converted = Converter(
        recipe=STATransformerRecipe(
            dataloader=calibration,
            time_steps=4,
            mode="spiking_encoder",
            threshold_mode="mse",
        )
    ).convert(model)
    modules = dict(converted.named_modules())
    y = _run_converted_readout(converted, torch.randn(2, 3, 16, 16))

    assert not hasattr(modules["fc1"], "v_threshold")
    assert not hasattr(modules["fc2"], "v_threshold")
    assert hasattr(modules["norm_sta_encoder"], "v_threshold")
    assert hasattr(modules["act_sta_encoder"], "v_threshold")
    assert y.shape == (2, 5)
    assert torch.isfinite(y).all()


def test_sta_transformer_recipe_spiking_encoder_mode_encodes_mha_output():
    torch.manual_seed(33)
    model = TinyANNMHATransformerBlock(embed_dim=8, num_heads=2, mlp_dim=16).eval()
    calibration = [(torch.randn(2, 4, 8),)]
    converted = Converter(
        recipe=STATransformerRecipe(
            dataloader=calibration,
            time_steps=4,
            mode="spiking_encoder",
        )
    ).convert(model)
    modules = dict(converted.named_modules())
    y = _run_converted_readout(converted, torch.randn(2, 4, 8))

    assert hasattr(modules["getitem_sta_encoder"], "v_threshold")
    assert torch.isfinite(y).all()


def test_sta_transformer_recipe_rejects_spiking_affine_threshold_mode_max_path():
    torch.manual_seed(34)
    model = TinyImageTransformerClassifier().eval()
    calibration = [(torch.randn(2, 3, 16, 16),)]

    with pytest.raises(ValueError, match="spiking affine"):
        Converter(
            recipe=STATransformerRecipe(
                dataloader=calibration,
                time_steps=4,
                mode="spiking_affine",
                threshold_mode="max",
            )
        ).convert(model)


def test_sta_transformer_recipe_raises_when_observer_never_calibrated():
    model = TinyImageTransformerClassifier().eval()

    with pytest.raises(ValueError, match="spiking affine"):
        Converter(
            recipe=STATransformerRecipe(
                dataloader=[],
                time_steps=4,
                mode="spiking_affine",
            )
        ).convert(model)


def test_sta_transformer_recipe_time_steps_affects_mse_thresholds():
    torch.manual_seed(4)
    model_t2 = TinyImageTransformerClassifier().eval()
    model_t8 = TinyImageTransformerClassifier().eval()
    model_t8.load_state_dict(model_t2.state_dict())
    calibration = [(torch.randn(2, 3, 16, 16),)]

    converted_t2 = Converter(
        recipe=STATransformerRecipe(
            dataloader=calibration,
            time_steps=2,
            mode="spiking_encoder",
            threshold_mode="mse",
        )
    ).convert(model_t2)
    converted_t8 = Converter(
        recipe=STATransformerRecipe(
            dataloader=calibration,
            time_steps=8,
            mode="spiking_encoder",
            threshold_mode="mse",
        )
    ).convert(model_t8)

    threshold_t2 = dict(converted_t2.named_modules())[
        "norm_sta_encoder"
    ].v_threshold
    threshold_t8 = dict(converted_t8.named_modules())[
        "norm_sta_encoder"
    ].v_threshold

    assert not torch.allclose(threshold_t2, threshold_t8)


def test_sta_transformer_recipe_threshold_scale_rescales_thresholds():
    torch.manual_seed(5)
    model_base = TinyImageTransformerClassifier().eval()
    model_scaled = TinyImageTransformerClassifier().eval()
    model_scaled.load_state_dict(model_base.state_dict())
    calibration = [(torch.randn(2, 3, 16, 16),)]

    converted_base = Converter(
        recipe=STATransformerRecipe(
            dataloader=calibration,
            time_steps=4,
            mode="spiking_encoder",
            threshold_mode="mse",
        )
    ).convert(model_base)
    converted_scaled = Converter(
        recipe=STATransformerRecipe(
            dataloader=calibration,
            time_steps=4,
            mode="spiking_encoder",
            threshold_mode="mse",
            threshold_scale=0.5,
        )
    ).convert(model_scaled)

    threshold_base = dict(converted_base.named_modules())[
        "norm_sta_encoder"
    ].v_threshold
    threshold_scaled = dict(converted_scaled.named_modules())[
        "norm_sta_encoder"
    ].v_threshold

    assert torch.allclose(threshold_scaled, threshold_base * 0.5)


def test_sta_transformer_recipe_preserves_source_requires_grad_flags():
    model = TinyImageTransformerClassifier().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    calibration = [(torch.randn(2, 3, 16, 16),)]

    converted = Converter(
        recipe=STATransformerRecipe(dataloader=calibration, time_steps=4)
    ).convert(model)

    assert not any(parameter.requires_grad for parameter in converted.parameters())


def test_sta_transformer_recipe_equivalent_mode_does_not_require_dataloader():
    model = TinyImageTransformerClassifier().eval()
    converted = Converter(
        recipe=STATransformerRecipe(dataloader=None, time_steps=4)
    ).convert(model)
    x = torch.randn(2, 3, 16, 16)

    _assert_converted_readout_matches_ann(converted, model, x)


def test_sta_transformer_recipe_num_calibration_batches_limits_observer_updates():
    torch.manual_seed(35)
    model = TinyImageTransformerClassifier().eval()
    calibration = [(torch.randn(2, 3, 16, 16),) for _ in range(3)]
    recipe = STATransformerRecipe(
        dataloader=calibration,
        time_steps=4,
        mode="spiking_encoder",
        num_calibration_batches=1,
    )

    Converter(recipe=recipe).convert(model)

    assert recipe._observers
    assert all(
        observer.num_batches_tracked == 1 for observer in recipe._observers.values()
    )


def test_sta_transformer_recipe_dict_batch_and_dict_output_are_supported():
    torch.manual_seed(36)
    model = TinyKeywordDictOutputTransformerClassifier().eval()
    calibration = [{"pixel_values": torch.randn(2, 4, 4)}]
    converted = Converter(
        recipe=STATransformerRecipe(dataloader=calibration, time_steps=4)
    ).convert(model)

    x = torch.randn(2, 4, 4)
    output = _run_converted_readout(converted, pixel_values=x)

    assert set(output.keys()) == {"logits"}
    assert torch.allclose(
        output["logits"],
        model(pixel_values=x)["logits"],
        atol=1e-5,
        rtol=1e-5,
    )


def test_sta_transformer_recipe_preserves_namedtuple_outputs():
    model = TinyNamedTupleOutputTransformerClassifier().eval()
    converted = Converter(
        recipe=STATransformerRecipe(dataloader=None, time_steps=4)
    ).convert(model)
    x = torch.randn(2, 4, 4)
    output = _run_converted_readout(converted, x)
    expected = model(x)

    assert isinstance(output, TinyModelOutput)
    assert torch.allclose(output.logits, expected.logits, atol=1e-5, rtol=1e-5)
    assert torch.allclose(output.hidden, expected.hidden, atol=1e-5, rtol=1e-5)


def test_sta_transformer_recipe_kwargs_calibration_path():
    torch.manual_seed(37)
    model = TinyKeywordTransformerClassifier().eval()
    calibration = [{"pixel_values": torch.randn(2, 4, 4)}]
    converted = Converter(
        recipe=STATransformerRecipe(dataloader=calibration, time_steps=4)
    ).convert(model)
    x = torch.randn(2, 4, 4)

    assert torch.allclose(
        _run_converted_readout(converted, pixel_values=x),
        model(x),
        atol=1e-5,
        rtol=1e-5,
    )


def test_sta_transformer_recipe_rejects_spike_classifier_affine_path():
    torch.manual_seed(38)
    model_default = TinyHeadClassifier().eval()
    model_with_head = TinyHeadClassifier().eval()
    model_with_head.load_state_dict(model_default.state_dict())
    calibration = [(torch.randn(2, 4, 4),)]

    with pytest.raises(ValueError, match="spiking affine"):
        Converter(
            recipe=STATransformerRecipe(
                dataloader=calibration,
                time_steps=4,
                mode="spiking_affine",
            )
        ).convert(model_default)
    with pytest.raises(ValueError, match="spiking affine"):
        Converter(
            recipe=STATransformerRecipe(
                dataloader=calibration,
                time_steps=4,
                mode="spiking_affine",
                spike_classifier=True,
            )
        ).convert(model_with_head)


def test_sta_transformer_recipe_rejects_spike_conv2d_explicitly():
    torch.manual_seed(6)
    model = TinyImageTransformerClassifier().eval()
    calibration = [(torch.randn(2, 3, 16, 16),)]

    with pytest.raises(ValueError, match="spiking affine"):
        Converter(
            recipe=STATransformerRecipe(
                dataloader=calibration,
                time_steps=4,
                mode="spiking_affine",
                threshold_mode="mse",
                spike_conv2d=True,
            )
        ).convert(model)


def test_sta_transformer_recipe_wraps_tensor_constants_as_first_step_inputs():
    torch.manual_seed(7)
    model = TinyConstantTransformerClassifier().eval()
    calibration = [(torch.randn(2, 4, 8),)]
    converted = Converter(
        recipe=STATransformerRecipe(
            dataloader=calibration,
            time_steps=4,
            threshold_mode="mse",
        )
    ).convert(model)
    x = torch.randn(2, 4, 8)
    tensor_get_attrs = [
        node
        for node in converted.graph.nodes
        if node.op == "get_attr"
        and torch.is_tensor(
            STATransformerRecipe._get_attr_value(converted, node.target)
        )
    ]

    assert tensor_get_attrs == []
    _assert_converted_readout_matches_ann(converted, model, x)


def test_sta_transformer_recipe_does_not_wrap_nonfloating_tensor_constants():
    model = TinyMaskBufferTransformerClassifier().eval()
    calibration = [(torch.randn(2, 2, 4),)]

    converted = Converter(
        recipe=STATransformerRecipe(dataloader=calibration, time_steps=4)
    ).convert(model)
    tensor_get_attr_values = [
        STATransformerRecipe._get_attr_value(converted, node.target)
        for node in converted.graph.nodes
        if node.op == "get_attr"
        and torch.is_tensor(
            STATransformerRecipe._get_attr_value(converted, node.target)
        )
    ]

    assert any(value.dtype == torch.bool for value in tensor_get_attr_values)
    assert not any(
        name.startswith("sta_time_constant")
        for name, _module in converted.named_modules()
    )


def test_sta_transformer_recipe_rejects_functional_sdpa_with_static_control_constant():
    model = TinyFloatMaskBufferTransformerClassifier().eval()

    with pytest.raises(ValueError, match="scaled_dot_product_attention"):
        Converter(recipe=STATransformerRecipe(time_steps=4)).convert(model)


def test_sta_transformer_recipe_rejects_functional_sdpa_positional_attention_masks():
    model = TinyPositionalSDPAMaskTransformerClassifier().eval()

    with pytest.raises(ValueError, match="scaled_dot_product_attention"):
        Converter(recipe=STATransformerRecipe(time_steps=4)).convert(model)


def test_sta_transformer_recipe_does_not_preserve_ordinary_fourth_positional_input():
    model = TinyFiveInputClassifier().eval()
    converted = Converter(
        recipe=STATransformerRecipe(dataloader=None, time_steps=4)
    ).convert(model)
    inputs = tuple(torch.randn(2, 4) for _ in range(5))

    _assert_converted_readout_matches_ann(converted, model, *inputs)


def test_sta_transformer_recipe_static_mask_tensor_helper_keeps_static_kwargs():
    converted = Converter(
        recipe=STATransformerRecipe(dataloader=None, time_steps=2)
    ).convert(nn.Identity())
    mask = torch.tensor([[True, False]])
    nested = {"attention_mask": (mask, [mask])}

    encoded = _sequence_inputs(converted, **nested)[1]

    assert encoded["attention_mask"][0] is mask
    assert encoded["attention_mask"][1][0] is mask


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"dataloader": None, "mode": "spiking_encoder"}, "dataloader"),
        ({"dataloader": None, "spike_linear": True}, "dataloader"),
        ({"dataloader": None, "spike_conv2d": True}, "dataloader"),
        ({"time_steps": 0}, "time_steps"),
        ({"time_steps": True}, "time_steps"),
        ({"mode": "missing"}, "mode"),
        ({"threshold_mode": "missing"}, "threshold_mode"),
        ({"threshold_scale": 0.0}, "threshold_scale"),
        ({"spike_linear": 1}, "spike_linear"),
        ({"spike_conv2d": 1}, "spike_conv2d"),
        ({"spike_classifier": 1}, "spike_classifier"),
        ({"momentum": 1.5}, "momentum"),
        ({"num_calibration_batches": 0}, "num_calibration_batches"),
        ({"num_calibration_batches": True}, "num_calibration_batches"),
        ({"show_progress": 1}, "show_progress"),
        ({"eps": 0.0}, "eps"),
    ],
)
def test_sta_transformer_recipe_validate_errors(kwargs, match):
    model = TinyImageTransformerClassifier().eval()
    dataloader = kwargs.pop("dataloader", [(torch.randn(2, 3, 16, 16),)])
    recipe = STATransformerRecipe(
        dataloader=dataloader,
        **kwargs,
    )

    with pytest.raises(ValueError, match=match):
        Converter(recipe=recipe).convert(model)
