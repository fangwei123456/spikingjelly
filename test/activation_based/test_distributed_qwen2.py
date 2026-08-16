import torch
import sys
from types import ModuleType, SimpleNamespace

from spikingjelly.activation_based.ann2snn.qcfs import SignedQCFSSequenceEncoder
from spikingjelly.activation_based.ann2snn.recipes.qwen2 import Qwen2SNNCalibration
from benchmark.snn_llm.qwen2 import (
    _encode_envelope,
    _fuse_qkv,
    _qwen2_architecture,
    _qwen2_rotary_base,
    _validate_qwen2_calibration,
    load_hf_qwen2_weights,
)
from spikingjelly.activation_based.distributed.llm.temporal import (
    pack_time_batch,
    unpack_time_batch,
)
from spikingjelly.activation_based.memopt import NullSpikeCompressor


def test_qwen_qcfs_envelope_matches_existing_reconstruction():
    time_steps = 4
    scale = torch.tensor([0.25, 0.5, 1.0])
    sequence = torch.randn(time_steps, 2, 5, 3, requires_grad=True)
    envelope = pack_time_batch(sequence)
    optimized_sequence = sequence.detach().clone().requires_grad_(True)

    encoded = unpack_time_batch(
        _encode_envelope(envelope, scale, time_steps), time_steps
    )
    optimized = unpack_time_batch(
        _encode_envelope(
            pack_time_batch(optimized_sequence),
            scale,
            time_steps,
            NullSpikeCompressor(),
        ),
        time_steps,
    )
    expected = SignedQCFSSequenceEncoder(
        scale, time_steps, collect_statistics=False
    ).reconstruct(sequence.sum(0))
    encoded.sum().backward()
    optimized.sum().backward()

    assert torch.equal(encoded.sum(0), expected)
    assert torch.equal(optimized, encoded)
    assert torch.count_nonzero(encoded[1:]) == 0
    assert sequence.grad is not None and torch.isfinite(sequence.grad).all()
    assert torch.equal(optimized_sequence.grad, sequence.grad)


def test_qwen_qkv_import_interleaves_query_groups():
    query = torch.arange(8).reshape(8, 1)
    key = torch.tensor([[100], [101], [102], [103]])
    value = torch.tensor([[200], [201], [202], [203]])

    fused = _fuse_qkv(query, key, value, num_attention_heads=4, num_query_groups=2)

    assert torch.equal(
        fused.flatten(),
        torch.tensor([0, 1, 2, 3, 100, 101, 200, 201, 4, 5, 6, 7, 102, 103, 202, 203]),
    )


def test_qwen_checkpoint_import_rejects_bias_mismatch(monkeypatch):
    core = ModuleType("megatron.core")
    core.parallel_state = SimpleNamespace(
        get_tensor_model_parallel_rank=lambda: 0,
        get_tensor_model_parallel_world_size=lambda: 1,
    )
    megatron = ModuleType("megatron")
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    source_attention = SimpleNamespace(
        q_proj=torch.nn.Linear(1, 8, bias=True),
        k_proj=torch.nn.Linear(1, 4, bias=True),
        v_proj=torch.nn.Linear(1, 4, bias=True),
    )
    source = SimpleNamespace(
        model=SimpleNamespace(
            layers=[SimpleNamespace(self_attn=source_attention)],
        ),
        config=SimpleNamespace(num_attention_heads=4, num_key_value_heads=2),
    )
    target = SimpleNamespace(
        pre_process=False,
        decoder=SimpleNamespace(
            layers=[
                SimpleNamespace(
                    layer_number=1,
                    self_attention=SimpleNamespace(
                        linear_qkv=SimpleNamespace(weight=torch.empty(16, 1), bias=None)
                    ),
                )
            ]
        ),
    )

    try:
        load_hf_qwen2_weights(target, source)
    except ValueError as error:
        assert "QKV bias settings must match" in str(error)
    else:
        raise AssertionError("Mismatched QKV bias settings must fail.")


def test_qwen_rotary_base_accepts_only_default_rope():
    config = SimpleNamespace(
        rope_parameters={"rope_type": "default", "rope_theta": 1_000_000.0}
    )
    assert _qwen2_rotary_base(config) == 1_000_000.0

    config.rope_parameters["rope_type"] = "yarn"
    try:
        _qwen2_rotary_base(config)
    except ValueError as error:
        assert "default RoPE" in str(error)
    else:
        raise AssertionError(
            "Non-default RoPE must fail instead of silently diverging."
        )


def test_qwen_checkpoint_architecture_excludes_local_source_path():
    config = SimpleNamespace(
        model_type="qwen2",
        vocab_size=128,
        max_position_embeddings=64,
        num_hidden_layers=2,
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=32,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )

    architecture = _qwen2_architecture(config)

    assert architecture["hidden_size"] == 16
    assert "source" not in architecture


def test_qwen_calibration_matches_source_channels_and_positive_scales():
    source = SimpleNamespace(
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=32,
    )
    calibration = Qwen2SNNCalibration(
        input_scale=torch.ones(16),
        layer_scales=(
            {
                "query": torch.ones(16),
                "key": torch.ones(8),
                "value": torch.ones(8),
                "mlp": torch.ones(32),
            },
        ),
        time_steps=2,
        calibration_levels=2,
        calibration_quantile=1.0,
        calibration_reservoir_size=8,
        calibration_seed=1,
        valid_token_count=1,
    )
    _validate_qwen2_calibration(calibration, source)

    calibration.layer_scales[0]["key"][0] = 0
    try:
        _validate_qwen2_calibration(calibration, source)
    except ValueError as error:
        assert "positives" in str(error)
    else:
        raise AssertionError("Non-positive calibration scales must fail.")
