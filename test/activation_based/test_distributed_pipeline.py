# ruff: noqa: F401,F403,F405
from spikingjelly.activation_based.distributed.pipeline import memopt as pipeline_memopt
from spikingjelly.activation_based.distributed.pipeline import (
    runtime as pipeline_runtime,
)
from spikingjelly.activation_based.distributed.pipeline.partition import (
    _partition_costs_contiguously,
)
from spikingjelly.activation_based.distributed.pipeline.runtime import (
    _build_snn_pipeline_runtime,
)
from spikingjelly.activation_based.distributed.pipeline.spikformer import (
    _SpikformerPipelineStage,
)
from test.activation_based._distributed_dtensor_test_support import *


def test_cost_partition_never_exceeds_requested_parts_at_float_boundary():
    costs = [
        63.2983219267329,
        993.0491635748301,
        479.44063272103153,
        319.43720121332564,
        729.1624014985915,
        24.291858945771793,
        434.24914484586606,
        664.413839099525,
        962.1362249074848,
        761.6377781461243,
        885.159209602491,
        118.90590716525107,
        429.77060562282907,
    ]

    assert _partition_costs_contiguously(costs, 2) == [8, 5]


def test_cifar10dvs_vgg_pipeline_module_matches_baseline():
    torch.manual_seed(0)
    baseline = CIFAR10DVSVGG(dropout=0.0, backend="torch").eval()
    x = torch.randn(1, 2, 2, 48, 48)
    pipeline_module = _build_cifar10dvs_vgg_pipeline_module(
        copy.deepcopy(baseline),
        num_logical_stages=2,
        example_input=x,
    ).eval()
    reference = baseline(x)
    functional.reset_net(pipeline_module)
    result = pipeline_module(x)
    torch.testing.assert_close(reference, result, rtol=1e-5, atol=1e-6)


def test_spikformer_pipeline_module_matches_baseline():
    torch.manual_seed(0)
    baseline = spikformer_ti(
        T=2, img_size_h=64, img_size_w=64, num_classes=11, backend="torch"
    ).eval()
    x = torch.randn(2, 3, 64, 64)
    pipeline_module = _build_spikformer_pipeline_module(
        copy.deepcopy(baseline),
        num_logical_stages=3,
        example_input=x,
    ).eval()
    functional.reset_net(baseline)
    reference = baseline(x)
    functional.reset_net(pipeline_module)
    result = pipeline_module(x)
    torch.testing.assert_close(reference, result, rtol=1e-5, atol=1e-6)


def test_spikformer_pipeline_middle_stage_rejects_4d_block_input():
    stage = _SpikformerPipelineStage(blocks=[nn.Identity()])

    with pytest.raises(ValueError, match="expects 5D"):
        stage(torch.randn(2, 3, 4, 4))


def test_measure_module_cost_uses_autograd_inside_no_grad():
    module = nn.Linear(3, 2)
    x = torch.randn(4, 3)

    with torch.no_grad():
        _output, cost = _measure_module_cost(module, x)

    assert cost > 0
    assert module.weight.grad is None


def test_cifar10dvs_vgg_pipeline_runtime_supports_interleaved_single_rank():
    with single_rank_process_group():
        model = CIFAR10DVSVGG(dropout=0.0, backend="torch").eval()
        x = torch.randn(2, 2, 2, 48, 48)
        runtime = configure_cifar10dvs_vgg_pipeline(
            copy.deepcopy(model),
            example_input=x,
            device=torch.device("cpu"),
            n_microbatches=2,
            pp_schedule="interleaved",
            pp_virtual_stages=2,
        )
        assert runtime.schedule_kind == "interleaved"
        assert runtime.virtual_pipeline_size == 2
        assert len(runtime.stage_modules) == 2


def test_spikformer_pipeline_runtime_supports_zero_bubble_single_rank():
    with single_rank_process_group():
        model = spikformer_ti(
            T=2, img_size_h=64, img_size_w=64, num_classes=11, backend="torch"
        ).eval()
        x = torch.randn(2, 3, 64, 64)
        runtime = configure_spikformer_pipeline(
            copy.deepcopy(model),
            example_input=x,
            device=torch.device("cpu"),
            n_microbatches=2,
            pp_schedule="zero_bubble",
            pp_virtual_stages=2,
            pp_delay_wgrad=True,
        )
        assert runtime.schedule_kind == "zero_bubble"
        assert runtime.delayed_wgrad is True
        assert len(runtime.stage_modules) == 2


def test_build_snn_pipeline_runtime_moves_dry_run_to_target_device(monkeypatch):
    class DeviceCheckingStage(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(1))
            self.seen_devices = []

        def forward(self, x):
            param_device = next(self.parameters()).device.type
            self.seen_devices.append((param_device, x.device.type))
            return x

    class PipelineModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.stages = nn.ModuleList([DeviceCheckingStage()])

    class FakePipelineStage:
        def __init__(
            self,
            module,
            *,
            stage_index,
            num_stages,
            device,
            input_args,
            output_args,
            group,
        ):
            assert next(module.parameters()).device.type == "meta"
            assert input_args.device.type == "meta"
            assert output_args.device.type == "meta"

    class FakeSchedule:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(pipeline_runtime, "PipelineStage", FakePipelineStage)
    monkeypatch.setattr(pipeline_runtime, "ScheduleGPipe", FakeSchedule)

    with single_rank_process_group():
        pipeline_module = PipelineModule()
        runtime = _build_snn_pipeline_runtime(
            pipeline_module,
            example_input=torch.randn(2, 3),
            device=torch.device("meta"),
            n_microbatches=1,
            stage_index=0,
            model_family="toy",
            schedule_kind="gpipe",
        )

    assert pipeline_module.stages[0].seen_devices == [("meta", "meta")]
    assert runtime.stage_input_examples[0].device.type == "meta"


def test_recommend_pipeline_memopt_stages_prefers_heavy_stages():
    selected = recommend_pipeline_memopt_stages(
        (1.0, 8.0, 3.0, 7.0), stage_budget_ratio=0.5
    )
    assert selected == (1, 3)


def test_recommend_pipeline_memopt_stages_rejects_nan_ratio():
    with pytest.raises(ValueError, match="finite number"):
        recommend_pipeline_memopt_stages((1.0, 2.0), stage_budget_ratio=float("nan"))


def test_pipeline_memopt_uses_rank_zero_stage_selection(monkeypatch):
    runtime = SNNPipelineRuntime(
        schedule=None,
        stage_module=nn.Identity(),
        stage_modules=(),
        local_stage_indices=(),
        stage_index=0,
        num_stages=2,
        device=torch.device("cpu"),
        n_microbatches=1,
        model_family="cifar10dvs_vgg",
        split_points=("stages.1",),
        stage_costs=(10.0, 1.0),
    )

    monkeypatch.setattr(pipeline_memopt.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        pipeline_memopt.dist,
        "broadcast",
        lambda selected, **_kwargs: selected.fill_(1),
    )

    runtime, _, applied = apply_pipeline_stage_memopt(runtime, memopt_level=1)

    assert applied is False
    assert runtime.memopt_selected_stage_indices == (1,)


def test_apply_pipeline_stage_memopt_only_wraps_selected_heavy_stage():
    torch.manual_seed(0)
    model = CIFAR10DVSVGG(dropout=0.0, backend="torch").eval()
    stage = _CIFAR10DVSVGGPipelineStage(
        feature_modules=[copy.deepcopy(model.features[0])],
        classifier=None,
        transpose_input=True,
    ).eval()
    wrapped_stage = _MicrobatchResetStage(stage)
    runtime = SNNPipelineRuntime(
        schedule=None,
        stage_module=wrapped_stage,
        stage_modules=(wrapped_stage,),
        local_stage_indices=(0,),
        stage_index=0,
        num_stages=2,
        device=torch.device("cpu"),
        n_microbatches=2,
        model_family="cifar10dvs_vgg",
        split_points=("stages.1",),
        stage_costs=(10.0, 1.0),
        stage_input_examples=(torch.randn(1, 2, 2, 48, 48),),
    )
    runtime, optimize_ms, applied = apply_pipeline_stage_memopt(
        runtime,
        memopt_level=1,
        compress_x=False,
        stage_budget_ratio=0.5,
    )
    assert applied is True
    assert optimize_ms >= 0.0
    assert runtime.memopt_selected_stage_indices == (0,)
    assert isinstance(runtime.stage_module.inner.features[0], GCContainer)


def test_apply_pipeline_stage_memopt_supports_legacy_memopt_signature(monkeypatch):
    torch.manual_seed(0)
    model = CIFAR10DVSVGG(dropout=0.0, backend="torch").eval()
    stage = _CIFAR10DVSVGGPipelineStage(
        feature_modules=[copy.deepcopy(model.features[0])],
        classifier=None,
        transpose_input=True,
    ).eval()
    wrapped_stage = _MicrobatchResetStage(stage)
    runtime = SNNPipelineRuntime(
        schedule=None,
        stage_module=wrapped_stage,
        stage_modules=(wrapped_stage,),
        local_stage_indices=(0,),
        stage_index=0,
        num_stages=2,
        device=torch.device("cpu"),
        n_microbatches=2,
        model_family="cifar10dvs_vgg",
        split_points=("stages.1",),
        stage_costs=(10.0, 1.0),
        stage_input_examples=(torch.randn(1, 2, 2, 48, 48),),
    )

    import spikingjelly.activation_based.memopt as memopt

    calls = {"count": 0}

    def fake_memory_optimization(module, target_types, dummy_input, compress_x, level):
        calls["count"] += 1
        return module

    monkeypatch.setattr(memopt, "memory_optimization", fake_memory_optimization)

    runtime, optimize_ms, applied = apply_pipeline_stage_memopt(
        runtime,
        memopt_level=1,
        compress_x=False,
        stage_budget_ratio=0.5,
        use_plan_cache=True,
    )

    assert applied is True
    assert optimize_ms >= 0.0
    assert calls["count"] == 1


def test_apply_pipeline_stage_memopt_moves_dummy_input_to_runtime_device(monkeypatch):
    torch.manual_seed(0)
    model = CIFAR10DVSVGG(dropout=0.0, backend="torch").eval()
    stage = _CIFAR10DVSVGGPipelineStage(
        feature_modules=[copy.deepcopy(model.features[0])],
        classifier=None,
        transpose_input=True,
    ).eval()
    wrapped_stage = _MicrobatchResetStage(stage)
    runtime = SNNPipelineRuntime(
        schedule=None,
        stage_module=wrapped_stage,
        stage_modules=(wrapped_stage,),
        local_stage_indices=(0,),
        stage_index=0,
        num_stages=2,
        device=torch.device("meta"),
        n_microbatches=2,
        model_family="cifar10dvs_vgg",
        split_points=("stages.1",),
        stage_costs=(10.0, 1.0),
        stage_input_examples=(torch.randn(1, 2, 2, 48, 48),),
    )

    import spikingjelly.activation_based.memopt as memopt

    def fake_memory_optimization(module, target_types, dummy_input, compress_x, level):
        assert next(module.parameters()).device.type == "meta"
        assert dummy_input[0].device.type == "meta"
        return module

    monkeypatch.setattr(memopt, "memory_optimization", fake_memory_optimization)

    _, _, applied = apply_pipeline_stage_memopt(
        runtime,
        memopt_level=1,
        compress_x=False,
        stage_budget_ratio=0.5,
        use_plan_cache=False,
    )

    assert applied is True


def test_parse_pipeline_layout_validates_counts():
    counts = parse_pipeline_layout("1|2|3", 3, 6)
    assert counts == (1, 2, 3)
    with pytest.raises(ValueError, match="requires 6 units"):
        parse_pipeline_layout("1|2|2", 3, 6)


def test_resolve_pipeline_schedule_kind_rules():
    assert resolve_pipeline_schedule_kind("auto", 1, False) == "1f1b"
    assert resolve_pipeline_schedule_kind("auto", 2, False) == "interleaved"
    assert resolve_pipeline_schedule_kind("auto", 2, True) == "zero_bubble"
    with pytest.raises(ValueError, match="requires pp_virtual_stages >= 2"):
        resolve_pipeline_schedule_kind("interleaved", 1, False)
    with pytest.raises(ValueError, match="does not support pp_virtual_stages=2"):
        resolve_pipeline_schedule_kind("gpipe", 2, False)
    with pytest.raises(ValueError, match="does not support pp_virtual_stages=2"):
        resolve_pipeline_schedule_kind("1f1b", 2, False)


def test_make_pipeline_outputs_contiguous_clones_views():
    base = torch.randn(2, 3, 4)
    view = base.transpose(0, 1)
    out = _make_pipeline_outputs_contiguous(view)
    torch.testing.assert_close(out, view)
    assert out.data_ptr() != view.data_ptr()


def test_make_pipeline_outputs_contiguous_reuses_contiguous_tensors():
    value = torch.randn(2, 3, 4)
    out = _make_pipeline_outputs_contiguous(value)
    assert out is value


def test_cifar_pipeline_transposes_on_first_non_empty_stage():
    torch.manual_seed(0)
    baseline = CIFAR10DVSVGG(dropout=0.0, backend="torch").eval()
    example = torch.randn(1, 2, 2, 48, 48)
    pipeline = _build_cifar10dvs_vgg_pipeline_module(
        copy.deepcopy(baseline),
        num_logical_stages=2,
        example_input=example,
        layout_counts=(0, len(list(baseline.features.children())) + 1),
    )
    assert pipeline.stages[0].transpose_input is False
    assert pipeline.stages[1].transpose_input is True
    functional.reset_net(baseline)
    reference = baseline(example)
    functional.reset_net(pipeline)
    result = pipeline(example)
    torch.testing.assert_close(reference, result, rtol=1e-5, atol=1e-6)


def test_spikformer_pipeline_attaches_patch_embed_to_first_non_empty_stage():
    torch.manual_seed(0)
    baseline = spikformer_ti(
        T=2,
        img_size_h=64,
        img_size_w=64,
        num_classes=11,
        backend="torch",
    ).eval()
    example = torch.randn(1, 3, 64, 64)
    pipeline = _build_spikformer_pipeline_module(
        copy.deepcopy(baseline),
        num_logical_stages=2,
        example_input=example,
        layout_counts=(0, len(baseline.blocks) + 2),
    )
    assert pipeline.stages[0].patch_embed is None
    assert pipeline.stages[1].patch_embed is not None
    functional.reset_net(baseline)
    reference = baseline(example)
    functional.reset_net(pipeline)
    result = pipeline(example)
    torch.testing.assert_close(reference, result, rtol=1e-5, atol=1e-6)
