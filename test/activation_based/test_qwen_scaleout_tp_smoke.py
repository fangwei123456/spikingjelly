import json

import pytest
import torch

from spikingjelly.activation_based.ann2snn.operators import TDLinear

from benchmark.snn_llm.qwen_conversion import scaleout_tp_smoke as runner


def test_qwen_tp_mapping_scales_with_model_depth():
    plan = runner._qwen_tdlinear_tp_plan(36)

    assert len(plan) == 252
    assert sum(value == "td_colwise_replicated" for value in plan.values()) == 180
    assert sum(value == "td_rowwise_replicated" for value in plan.values()) == 72
    assert "layers.0.q_proj" in plan
    assert "layers.35.down_proj" in plan
    assert all("embed_tokens" not in name and "lm_head" not in name for name in plan)
    assert 512 in runner.TIME_STEP_CHOICES


def test_gathered_rowwise_execution_preserves_tdlinear_math_without_dtensor():
    model = torch.nn.ModuleDict({"row": TDLinear(4, 3), "col": TDLinear(4, 3)})
    value = torch.randn(2, 5, 4)
    expected = model["row"].ann_forward(value)
    col_forward = model["col"].ann_forward

    runner._install_gathered_rowwise_execution(
        model,
        {"row": "td_rowwise_replicated", "col": "td_colwise_replicated"},
    )

    torch.testing.assert_close(model["row"].ann_forward(value), expected)
    assert model["col"].ann_forward.__func__ is col_forward.__func__


def test_full_tensor_materialization_uses_dtensor_like_contract():
    class FakeDTensor:
        def __init__(self):
            self.calls = 0

        def full_tensor(self):
            self.calls += 1
            return torch.tensor([1.0])

    value = FakeDTensor()

    assert runner._materialize_full_tensor(value).item() == 1.0
    assert value.calls == 1


def test_full_tensor_materialization_unwraps_parameter_dtensor_contract():
    class RawDTensor:
        def __init__(self):
            self.calls = 0

        def full_tensor(self):
            self.calls += 1
            return torch.tensor([2.0])

    class ParameterDTensor:
        _is_param = True

        def __init__(self):
            self.data = RawDTensor()

        def full_tensor(self):
            raise AssertionError("Parameter wrapper must not materialize directly.")

    value = ParameterDTensor()

    assert runner._materialize_full_tensor(value).item() == 2.0
    assert value.data.calls == 1


def test_tp_environment_requires_two_gpus_and_gseries_guard(monkeypatch):
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setenv("WORLD_SIZE", "1")
    with pytest.raises(RuntimeError, match="world_size=2"):
        runner._validate_distributed_environment()

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.delenv("NCCL_P2P_DISABLE", raising=False)
    with pytest.raises(RuntimeError, match="NCCL_P2P_DISABLE=1"):
        runner._validate_distributed_environment()


def test_rank_zero_is_only_report_writer(tmp_path):
    report = {"finite": 1.0}

    assert runner._write_rank0_report(tmp_path, report, rank=1) is None
    path = runner._write_rank0_report(tmp_path, report, rank=0)

    assert json.loads(path.read_text()) == report


def test_tp_comparison_enforces_logits_cache_loss_tokens_and_reset():
    reference = {
        "logits": [[1.0, 2.0]],
        "loss": 3.0,
        "cache_relative_l2": 0.1,
        "token_ids": [4, 5],
        "reset_error": 0.0,
    }
    candidate = dict(reference)
    metrics = runner._compare_outputs(reference, candidate, label="test")
    assert metrics["logits_relative_l2"] == 0.0

    candidate["token_ids"] = [4, 6]
    with pytest.raises(ValueError, match="token IDs"):
        runner._compare_outputs(reference, candidate, label="test")

    candidate = dict(reference)
    candidate["logits"] = [[10.0, -10.0]]
    with pytest.raises(ValueError, match="logits_relative_l2"):
        runner._compare_outputs(reference, candidate, label="test")

    candidate = dict(reference)
    candidate["cache_relative_l2"] = 0.13
    metrics = runner._compare_outputs(reference, candidate, label="test")
    assert metrics["cache_relative_l2_delta"] == pytest.approx(0.03)

    candidate["cache_relative_l2"] = 0.21
    with pytest.raises(ValueError, match="cache_relative_l2"):
        runner._compare_outputs(reference, candidate, label="test")

    invalid_reference = dict(reference)
    invalid_reference["cache_relative_l2"] = float("nan")
    with pytest.raises(ValueError, match="delta must be finite"):
        runner._compare_outputs(invalid_reference, reference, label="test")


def test_tp_signed_quality_uses_single_model_gates_not_signed_bit_parity():
    exact = {
        "logits": [[1.0, 0.0], [0.0, 1.0]],
        "loss": 2.0,
        "cache_relative_l2": 0.01,
        "reset_error": 0.0,
    }
    signed = {
        "logits": [[0.8, 0.2], [0.4, 0.6]],
        "loss": 2.2,
        "cache_relative_l2": 0.1,
        "reset_error": 0.0,
    }

    metrics = runner._signed_quality(signed, exact)

    assert metrics["logits_relative_l2"] < 0.5
    assert metrics["top1_agreement"] == 1.0
