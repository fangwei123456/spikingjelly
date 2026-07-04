import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import functional, neuron, surrogate
from spikingjelly.activation_based.ann2snn.operators import (
    SNNElementWiseProduct,
    SNNMatrixOperator,
    TDLinear,
)


def _table(scale=1.0):
    return neuron.FewSpikeTable(
        theta=torch.tensor([0.25, 0.5, 0.75]) * scale,
        h=torch.tensor([0.2, 0.3, 0.4]) * scale,
        d=torch.tensor([1.0, 2.0, 4.0]) * scale,
    )


def _two_step_table(scale=1.0):
    return neuron.FewSpikeTable(
        theta=torch.tensor([0.25, 0.5]) * scale,
        h=torch.tensor([0.1, 0.1]) * scale,
        d=torch.tensor([1.0, 2.0]) * scale,
    )


def _few_spike_node(table=None, surrogate_function=None):
    if table is None:
        table = _table()
    if surrogate_function is None:
        surrogate_function = surrogate.DeterministicPass()
    return neuron.FewSpikeNode(
        table=table,
        surrogate_function=surrogate_function,
        step_mode="m",
    )


def _split_heads(x, num_heads):
    if x.shape[-1] % num_heads != 0:
        raise ValueError("embed_dim must be divisible by num_heads.")
    head_dim = x.shape[-1] // num_heads
    x = x.reshape(*x.shape[:-1], num_heads, head_dim)
    return x.transpose(-3, -2)


def _merge_heads(x):
    x = x.transpose(-3, -2).contiguous()
    return x.reshape(*x.shape[:-2], x.shape[-2] * x.shape[-1])


class _FewSpikeSelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, score_node):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.q_proj = TDLinear(embed_dim, embed_dim, bias=False)
        self.k_proj = TDLinear(embed_dim, embed_dim, bias=False)
        self.v_proj = TDLinear(embed_dim, embed_dim, bias=False)
        self.score_node = score_node
        self.score_matmul = SNNMatrixOperator()
        self.context_matmul = SNNMatrixOperator()
        self.out_proj = TDLinear(embed_dim, embed_dim, bias=False)

    def _project_sequence(self, x_seq):
        q_seq = _split_heads(self.q_proj(x_seq), self.num_heads)
        k_seq = _split_heads(self.k_proj(x_seq), self.num_heads)
        v_seq = _split_heads(self.v_proj(x_seq), self.num_heads)
        return q_seq, k_seq, v_seq

    def _score_sequence_from_projected(self, q_seq, k_seq):
        return self.score_matmul(q_seq, k_seq.transpose(-1, -2))

    def score_sequence(self, x_seq):
        q_seq, k_seq, _ = self._project_sequence(x_seq)
        return self._score_sequence_from_projected(q_seq, k_seq)

    def _encoded_score_sequence_from_projected(self, q_seq, k_seq):
        return self.score_node(self._score_sequence_from_projected(q_seq, k_seq))

    def encoded_score_sequence(self, x_seq):
        return self.score_node(self.score_sequence(x_seq))

    def context_sequence(self, x_seq):
        q_seq, k_seq, v_seq = self._project_sequence(x_seq)
        score_seq = self._encoded_score_sequence_from_projected(q_seq, k_seq)
        return self.context_matmul(score_seq, v_seq)

    def forward(self, x_seq):
        context_seq = _merge_heads(self.context_sequence(x_seq))
        return self.out_proj(context_seq)


class _FewSpikeGatedMLPBlock(nn.Module):
    def __init__(self, embed_dim, hidden_features, activation, gate):
        super().__init__()
        self.up_proj = TDLinear(embed_dim, hidden_features, bias=False)
        self.gate_proj = TDLinear(embed_dim, hidden_features, bias=False)
        self.activation = activation
        self.gate = gate
        self.product = SNNElementWiseProduct()
        self.down_proj = TDLinear(hidden_features, embed_dim, bias=False)

    def forward(self, x_seq):
        up_seq = self.activation(self.up_proj(x_seq))
        gate_seq = self.gate(self.gate_proj(x_seq))
        return self.down_proj(self.product(up_seq, gate_seq))


class _TinyFewSpikeDecoderBlock(nn.Module):
    """Simplified decoder toy: no LayerNorm, Softmax, dropout, mask, or KV-cache."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        hidden_features,
        score_node,
        activation,
        gate,
    ):
        super().__init__()
        self.attn = _FewSpikeSelfAttentionBlock(embed_dim, num_heads, score_node)
        self.mlp = _FewSpikeGatedMLPBlock(embed_dim, hidden_features, activation, gate)

    def forward(self, x_seq):
        x_seq = x_seq + self.attn(x_seq)
        return x_seq + self.mlp(x_seq)


class _TinyFewSpikeDecoderStack(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x_seq):
        for layer in self.layers:
            x_seq = layer(x_seq)
        return x_seq


def _single_step_attention_reference_from_analog(block, x):
    q = _split_heads(F.linear(x, block.q_proj.weight, None), block.num_heads)
    k = _split_heads(F.linear(x, block.k_proj.weight, None), block.num_heads)
    v = _split_heads(F.linear(x, block.v_proj.weight, None), block.num_heads)
    score = torch.matmul(q, k.transpose(-1, -2))
    score = block.score_node.single_step_forward(score)
    context = torch.matmul(score, v)
    context = _merge_heads(context)
    return F.linear(context, block.out_proj.weight, None)


def _single_step_attention_reference(block, x_seq):
    return _single_step_attention_reference_from_analog(block, x_seq.sum(dim=0))


def _single_step_gated_reference_from_analog(block, x):
    up = F.linear(x, block.up_proj.weight, None)
    gate = F.linear(x, block.gate_proj.weight, None)
    up = block.activation.single_step_forward(up)
    gate = block.gate.single_step_forward(gate)
    return F.linear(up * gate, block.down_proj.weight, None)


def _single_step_decoder_reference_from_analog(block, x):
    x = x + _single_step_attention_reference_from_analog(block.attn, x)
    return x + _single_step_gated_reference_from_analog(block.mlp, x)


def _single_step_decoder_reference(block, x_seq):
    return _single_step_decoder_reference_from_analog(block, x_seq.sum(dim=0))


def _single_step_stack_reference(stack, x_seq):
    x = x_seq.sum(dim=0)
    for layer in stack.layers:
        x = _single_step_decoder_reference_from_analog(layer, x)
    return x


def _assert_finite_gradients(module, x_seq):
    assert x_seq.grad is not None
    assert torch.isfinite(x_seq.grad).all()
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_single_head_attention_final_sum_matches_single_step_reference():
    table = _table()
    block = _FewSpikeSelfAttentionBlock(
        embed_dim=4,
        num_heads=1,
        score_node=_few_spike_node(table),
    )
    x_seq = torch.randn(table.K, 2, 3, 4)

    y_seq = block(x_seq)
    functional.reset_net(block)
    expected = _single_step_attention_reference(block, x_seq)

    assert y_seq.shape == (table.K, 2, 3, 4)
    assert torch.allclose(y_seq.sum(dim=0), expected, atol=1e-5, rtol=1e-5)


def test_multi_head_attention_keeps_score_and_context_sequences():
    table = _table()
    block = _FewSpikeSelfAttentionBlock(
        embed_dim=6,
        num_heads=2,
        score_node=_few_spike_node(table),
    )
    x_seq = torch.randn(table.K, 2, 4, 6)

    score_seq = block.score_sequence(x_seq)
    functional.reset_net(block)
    encoded_score_seq = block.score_node(score_seq)
    functional.reset_net(block)
    context_seq = block.context_sequence(x_seq)
    functional.reset_net(block)
    y_seq = block(x_seq)
    functional.reset_net(block)
    expected = _single_step_attention_reference(block, x_seq)

    assert score_seq.shape == (table.K, 2, 2, 4, 4)
    assert encoded_score_seq.shape == (table.K, 2, 2, 4, 4)
    assert context_seq.shape == (table.K, 2, 2, 4, 3)
    assert y_seq.shape == (table.K, 2, 4, 6)
    assert torch.allclose(y_seq.sum(dim=0), expected, atol=1e-5, rtol=1e-5)


def test_attention_oat_score_branch_matches_single_step_reference():
    normal = _two_step_table()
    outlier = _two_step_table(scale=10.0)
    block = _FewSpikeSelfAttentionBlock(
        embed_dim=4,
        num_heads=1,
        score_node=neuron.OutlierAwareThresholdNode(
            table=normal,
            outlier_table=outlier,
            split_threshold=1.0,
            clamp_value=2.0,
            surrogate_function=surrogate.DeterministicPass(),
            step_mode="m",
        ),
    )
    x_seq = torch.tensor(
        [
            [
                [[1.0, 0.2, -0.5, 0.3], [0.4, -1.2, 0.6, 0.1]],
                [[-0.6, 0.7, 0.2, -0.4], [0.9, 0.1, -0.3, 0.5]],
            ],
            [
                [[0.8, -0.1, 0.4, 0.5], [0.5, -0.4, 0.3, 0.2]],
                [[-0.2, 0.5, 0.6, -0.1], [0.7, 0.3, -0.2, 0.4]],
            ],
        ]
    )

    y_seq = block(x_seq)
    functional.reset_net(block)
    expected = _single_step_attention_reference(block, x_seq)

    assert y_seq.shape == (normal.K, 2, 2, 4)
    assert torch.allclose(y_seq.sum(dim=0), expected, atol=1e-5, rtol=1e-5)


def test_attention_hg_score_branch_matches_single_step_reference():
    table_low = _two_step_table()
    table_mid = _two_step_table(scale=2.0)
    table_high = _two_step_table(scale=4.0)
    block = _FewSpikeSelfAttentionBlock(
        embed_dim=4,
        num_heads=1,
        score_node=neuron.HGNode(
            tables=[table_low, table_mid, table_high],
            gate_thresholds=[0.0, 1.0],
            surrogate_function=surrogate.DeterministicPass(),
            step_mode="m",
        ),
    )
    x_seq = torch.randn(table_low.K, 2, 3, 4)

    y_seq = block(x_seq)
    functional.reset_net(block)
    expected = _single_step_attention_reference(block, x_seq)

    assert y_seq.shape == (table_low.K, 2, 3, 4)
    assert torch.allclose(y_seq.sum(dim=0), expected, atol=1e-5, rtol=1e-5)


def test_attention_rejects_wrong_time_length_from_score_node():
    table = _table()
    block = _FewSpikeSelfAttentionBlock(
        embed_dim=4,
        num_heads=1,
        score_node=_few_spike_node(table),
    )

    with pytest.raises(ValueError, match="K=3"):
        block(torch.randn(table.K - 1, 2, 3, 4))


def test_decoder_block_final_sum_matches_reference_after_reset():
    table = _table()
    block = _TinyFewSpikeDecoderBlock(
        embed_dim=4,
        num_heads=1,
        hidden_features=5,
        score_node=_few_spike_node(table),
        activation=_few_spike_node(table),
        gate=_few_spike_node(table),
    )
    x_seq = torch.randn(table.K, 2, 3, 4)

    y0 = block(x_seq)
    functional.reset_net(block)
    y1 = block(x_seq)
    functional.reset_net(block)
    expected = _single_step_decoder_reference(block, x_seq)

    assert y0.shape == (table.K, 2, 3, 4)
    assert torch.allclose(y0, y1)
    assert torch.allclose(y0.sum(dim=0), expected, atol=1e-4, rtol=1e-4)


def test_decoder_block_autograd_with_sigmoid_surrogates():
    table = _table()
    block = _TinyFewSpikeDecoderBlock(
        embed_dim=4,
        num_heads=1,
        hidden_features=5,
        score_node=_few_spike_node(table, surrogate.Sigmoid()),
        activation=_few_spike_node(table, surrogate.Sigmoid()),
        gate=_few_spike_node(table, surrogate.Sigmoid()),
    )
    x_seq = torch.randn(table.K, 2, 3, 4, requires_grad=True)

    y_seq = block(x_seq)
    functional.reset_net(block)
    expected = _single_step_decoder_reference(block, x_seq)
    y_seq.square().sum().backward()

    assert y_seq.shape == (table.K, 2, 3, 4)
    assert torch.allclose(y_seq.sum(dim=0), expected, atol=1e-4, rtol=1e-4)
    _assert_finite_gradients(block, x_seq)


def test_synthetic_decoder_stack_benchmark_smoke():
    torch.manual_seed(0)
    table = _table()
    vocab_size = 11
    embed_dim = 4
    embedding = nn.Embedding(vocab_size, embed_dim)
    token_ids = torch.tensor([[0, 3, 7], [2, 5, 1]])
    dense_tokens = embedding(token_ids)
    coeffs = torch.tensor([0.5, -0.25, 0.75]).view(table.K, 1, 1, 1)
    x_seq = dense_tokens.unsqueeze(0) * coeffs

    stack = _TinyFewSpikeDecoderStack(
        [
            _TinyFewSpikeDecoderBlock(
                embed_dim=embed_dim,
                num_heads=1,
                hidden_features=5,
                score_node=_few_spike_node(table, surrogate.Sigmoid()),
                activation=_few_spike_node(table, surrogate.Sigmoid()),
                gate=_few_spike_node(table, surrogate.Sigmoid()),
            ),
            _TinyFewSpikeDecoderBlock(
                embed_dim=embed_dim,
                num_heads=2,
                hidden_features=6,
                score_node=_few_spike_node(table, surrogate.Sigmoid()),
                activation=_few_spike_node(table, surrogate.Sigmoid()),
                gate=_few_spike_node(table, surrogate.Sigmoid()),
            ),
        ]
    )

    y_seq = stack(x_seq)
    functional.reset_net(stack)
    expected = _single_step_stack_reference(stack, x_seq)
    y_seq.square().sum().backward()

    assert y_seq.shape == (table.K, 2, 3, embed_dim)
    assert torch.isfinite(y_seq).all()
    # Two decoder layers with Sigmoid surrogates accumulate more float32 error.
    assert torch.allclose(y_seq.sum(dim=0), expected, atol=1e-3, rtol=1e-3)
    assert embedding.weight.grad is not None
    assert torch.isfinite(embedding.weight.grad).all()
    for parameter in stack.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
