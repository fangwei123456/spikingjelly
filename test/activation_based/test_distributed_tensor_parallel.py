# ruff: noqa: F401,F403,F405
import pickle

from spikingjelly.activation_based import base
from spikingjelly.activation_based.distributed.tensor_parallel.state import (
    _has_tensor_shard_input_validator,
)

from test.activation_based._distributed_dtensor_test_support import *


class _ResetCountingMemoryModule(base.MemoryModule):
    def __init__(self):
        super().__init__()
        self.register_memory("v", 0.0)
        self.reset_calls = 0

    def single_step_forward(self, x: torch.Tensor):
        self.v = x * 2
        return self.v

    def reset(self):
        self.reset_calls += 1
        super().reset()


def test_tp_communication_debug_stats_recorded_for_rowwise_conv(
    monkeypatch: pytest.MonkeyPatch,
):
    conv = distributed_dtensor.ChannelShardConv1d(
        nn.Conv1d(4, 6, kernel_size=1, bias=True),
        process_group=None,
        mode="rowwise",
    )
    x = torch.randn(2, 4, 8)
    distributed_dtensor.reset_tp_communication_debug_stats()
    distributed_dtensor.enable_tp_communication_debug(True)
    try:
        conv(x)
        stats = distributed_dtensor.get_tp_communication_debug_stats()
        assert stats["all_reduce_calls"] == 0
        assert stats["all_reduce_bytes"] == 0

        conv.world_size = 2
        conv.process_group = None
        calls = {"count": 0}

        def _fake_all_reduce(tensor, group=None):
            calls["count"] += 1
            return tensor

        monkeypatch.setattr(distributed_dtensor.dist, "all_reduce", _fake_all_reduce)
        conv(x)
        stats = distributed_dtensor.get_tp_communication_debug_stats()
        assert calls["count"] == 1
        assert stats["all_reduce_calls"] == 1
        assert stats["all_reduce_bytes"] > 0
    finally:
        distributed_dtensor.enable_tp_communication_debug(False)
        distributed_dtensor.reset_tp_communication_debug_stats()


def test_channel_shard_conv2d_preserves_nonzero_padding_mode():
    torch.manual_seed(0)
    source = nn.Conv2d(
        2,
        4,
        kernel_size=3,
        padding=1,
        padding_mode="reflect",
        bias=True,
    )
    x = torch.randn(3, 2, 8, 8)
    reference = source(x)
    delattr(source, "_reversed_padding_repeated_twice")

    wrapped = ChannelShardConv2d(source, process_group=None, mode="colwise")
    torch.testing.assert_close(wrapped(x), reference)


def test_channel_shard_conv2d_preserves_string_padding_fallback():
    torch.manual_seed(0)
    source = nn.Conv2d(
        2,
        4,
        kernel_size=3,
        padding="same",
        padding_mode="reflect",
        bias=True,
    )
    x = torch.randn(3, 2, 8, 8)
    reference = source(x)
    delattr(source, "_reversed_padding_repeated_twice")

    wrapped = ChannelShardConv2d(source, process_group=None, mode="colwise")
    torch.testing.assert_close(wrapped(x), reference)


def test_channel_shard_conv1d_preserves_padding_and_multistep_shape():
    torch.manual_seed(0)
    source = nn.Conv1d(
        2,
        4,
        kernel_size=3,
        padding=1,
        padding_mode="reflect",
        bias=True,
    )
    source.step_mode = "m"
    x = torch.randn(5, 3, 2, 8)
    reference = source(x.flatten(0, 1)).view(5, 3, 4, 8)
    delattr(source, "_reversed_padding_repeated_twice")

    wrapped = ChannelShardConv1d(source, process_group=None, mode="colwise")
    torch.testing.assert_close(wrapped(x), reference)


def test_channel_shard_conv_preserves_requires_grad_flags():
    for conv_cls, wrapper_cls in (
        (nn.Conv1d, ChannelShardConv1d),
        (nn.Conv2d, ChannelShardConv2d),
    ):
        for mode in ("colwise", "rowwise"):
            source = conv_cls(4, 8, kernel_size=1, bias=True)
            source.weight.requires_grad_(False)
            source.bias.requires_grad_(False)

            wrapped = wrapper_cls(source, process_group=None, mode=mode)

            assert wrapped.weight.requires_grad is False
            assert wrapped.bias.requires_grad is False


def test_channel_shard_batch_norm_preserves_requires_grad_flags():
    for bn_cls, wrapper_cls in (
        (nn.BatchNorm1d, ChannelShardBatchNorm1d),
        (nn.BatchNorm2d, ChannelShardBatchNorm2d),
    ):
        source = bn_cls(4)
        source.weight.requires_grad_(False)
        source.bias.requires_grad_(False)

        wrapped = wrapper_cls(source, process_group=None)

        assert wrapped.weight.requires_grad is False
        assert wrapped.bias.requires_grad is False


def test_make_tensor_shard_memory_module_uses_channel_dim_for_single_step():
    source = neuron.IFNode(step_mode="s")
    module = make_tensor_shard_memory_module(
        source,
        shard_dim=2,
        logical_dim_size=4,
        process_group=None,
    )
    x = torch.ones(2, 4, 3, 3)

    assert module is not source
    assert type(module) is type(source)
    assert module(x).shape == x.shape


def test_make_tensor_shard_memory_module_rejects_invalid_shard_configuration(
    monkeypatch: pytest.MonkeyPatch,
):
    for shard_dim in (3, -3):
        module = make_tensor_shard_memory_module(
            neuron.IFNode(step_mode="s"), shard_dim, None, None
        )
        with pytest.raises(ValueError, match="is invalid for input"):
            module(torch.ones(2, 4))

    process_group = object()
    monkeypatch.setattr(torch.distributed, "get_rank", lambda _group: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group: 2)
    with pytest.raises(ValueError, match="logical_dim_size=3 must be divisible"):
        make_tensor_shard_memory_module(
            neuron.IFNode(step_mode="s"), -1, 3, process_group
        )


def test_make_tensor_shard_memory_module_preserves_memory_interface():
    source = _ResetCountingMemoryModule()
    module = make_tensor_shard_memory_module(
        source,
        shard_dim=-1,
        logical_dim_size=3,
        process_group=None,
    )
    module(torch.ones(2, 3, requires_grad=True))

    assert list(source.named_memories()) == [("v", 0.0)]
    assert module.get_reset_value("v") == 0.0
    assert module.v.requires_grad

    module.detach()
    assert not module.v.requires_grad

    functional.reset_net(module)
    assert module.reset_calls == 1
    assert module.v == 0.0


def test_make_tensor_shard_memory_module_preserves_source_runtime_options():
    source = neuron.IFNode(step_mode="m", backend="torch", store_v_seq=True)
    module = make_tensor_shard_memory_module(source, -1, 4, None)

    assert module.step_mode == "m"
    assert module.backend == "torch"
    assert module.store_v_seq is True


def test_make_tensor_shard_memory_module_is_idempotent_and_serializable():
    source = neuron.IFNode(step_mode="s")
    module = make_tensor_shard_memory_module(source, -1, 4, None)

    assert make_tensor_shard_memory_module(module, -1, 4, None) is module
    assert not source._forward_pre_hooks

    for restored in (copy.deepcopy(module), pickle.loads(pickle.dumps(module))):
        assert make_tensor_shard_memory_module(restored, -1, 4, None) is restored
        assert restored(torch.ones(2, 4)).shape == (2, 4)
        with pytest.raises(ValueError, match="Expected local shard size 4"):
            restored(torch.ones(2, 3))


def test_tensor_shard_validator_precedes_existing_hook_and_accepts_keyword_input():
    events = []
    source = neuron.IFNode(step_mode="s")
    source.register_forward_pre_hook(lambda _module, _args: events.append("source"))
    module = make_tensor_shard_memory_module(source, -1, 4, None)

    assert module(x=torch.ones(2, 4)).shape == (2, 4)
    assert events == ["source"]

    events.clear()
    with pytest.raises(ValueError, match="Expected local shard size 4"):
        module(x=torch.ones(2, 3))
    assert events == []

    with pytest.raises(TypeError, match="first positional argument or the 'x'"):
        module(input=torch.ones(2, 4))


def test_tensor_shard_memory_module_keeps_source_state_dict_paths():
    source = neuron.ParametricLIFNode(step_mode="s")
    module = make_tensor_shard_memory_module(source, -1, 4, None)

    assert tuple(module.state_dict()) == ("w",)
    assert "inner" not in dict(module.named_modules())
    module.load_state_dict(source.state_dict(), strict=True)


def test_tensor_shard_memory_module_legacy_alias_points_to_factory():
    source = neuron.IFNode(step_mode="s")
    module = TensorShardMemoryModule(source, -1, 4, None)

    assert distributed_dtensor.TensorShardMemoryModule is TensorShardMemoryModule
    assert module is not source
    assert type(module) is type(source)
    assert _has_tensor_shard_input_validator(module)


def test_tensor_shard_memory_module_compiles_for_valid_input():
    module = make_tensor_shard_memory_module(neuron.IFNode(step_mode="s"), -1, 4, None)
    compiled = torch.compile(module, backend="eager", fullgraph=True)

    assert compiled(torch.ones(2, 4)).shape == (2, 4)


def test_channel_shard_conv2d_colwise_all_reduces_input_gradient(monkeypatch):
    torch.manual_seed(0)
    source = nn.Conv2d(2, 4, kernel_size=1, bias=False)
    wrapped = ChannelShardConv2d(source, process_group=None, mode="colwise")
    wrapped.world_size = 2
    x = torch.randn(1, 2, 3, 3, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    calls = {"count": 0}

    def _fake_all_reduce(tensor, group=None):
        calls["count"] += 1
        tensor.mul_(2)
        return tensor

    monkeypatch.setattr(distributed_dtensor.dist, "is_available", lambda: True)
    monkeypatch.setattr(distributed_dtensor.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed_dtensor.dist, "all_reduce", _fake_all_reduce)

    wrapped(x).sum().backward()
    torch.nn.functional.conv2d(
        x_ref, wrapped.weight, wrapped.bias, wrapped.stride
    ).sum().backward()

    assert calls["count"] == 1
    torch.testing.assert_close(x.grad, x_ref.grad * 2)


def test_colwise_backward_all_reduce_contiguous_grad(monkeypatch):
    from spikingjelly.activation_based.distributed.tensor_parallel.channel import (
        _ColwiseBackwardAllReduce,
    )

    x = torch.randn(2, 3, requires_grad=True)
    non_contiguous_grad = torch.ones(3, 2).t()
    assert not non_contiguous_grad.is_contiguous()
    calls = {"count": 0}

    def _fake_all_reduce(tensor, group=None):
        calls["count"] += 1
        assert tensor.is_contiguous()
        return tensor

    monkeypatch.setattr(distributed_dtensor.dist, "is_available", lambda: True)
    monkeypatch.setattr(distributed_dtensor.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed_dtensor.dist, "all_reduce", _fake_all_reduce)

    y = _ColwiseBackwardAllReduce.apply(x, None, 2)
    y.backward(non_contiguous_grad)

    assert calls["count"] == 1
    torch.testing.assert_close(x.grad, non_contiguous_grad)


def test_channel_shard_conv1d_colwise_all_reduces_input_gradient(monkeypatch):
    torch.manual_seed(0)
    source = nn.Conv1d(2, 4, kernel_size=1, bias=False)
    wrapped = ChannelShardConv1d(source, process_group=None, mode="colwise")
    wrapped.world_size = 2
    x = torch.randn(1, 2, 5, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    calls = {"count": 0}

    def _fake_all_reduce(tensor, group=None):
        calls["count"] += 1
        tensor.mul_(2)
        return tensor

    monkeypatch.setattr(distributed_dtensor.dist, "is_available", lambda: True)
    monkeypatch.setattr(distributed_dtensor.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed_dtensor.dist, "all_reduce", _fake_all_reduce)

    wrapped(x).sum().backward()
    torch.nn.functional.conv1d(
        x_ref, wrapped.weight, wrapped.bias, wrapped.stride
    ).sum().backward()

    assert calls["count"] == 1
    torch.testing.assert_close(x.grad, x_ref.grad * 2)


def test_channel_shard_batch_norm2d_supports_cumulative_momentum():
    torch.manual_seed(0)
    source = nn.BatchNorm2d(4, momentum=None)
    wrapped = ChannelShardBatchNorm2d(source, process_group=None)
    x = torch.randn(3, 4, 5, 5)

    torch.testing.assert_close(wrapped(x), source(x))
    assert wrapped.num_batches_tracked.item() == source.num_batches_tracked.item()


def test_channel_shard_batch_norm1d_supports_cumulative_momentum():
    torch.manual_seed(0)
    source = nn.BatchNorm1d(4, momentum=None)
    wrapped = ChannelShardBatchNorm1d(source, process_group=None)
    x = torch.randn(3, 4, 5)

    torch.testing.assert_close(wrapped(x), source(x))
    assert wrapped.num_batches_tracked.item() == source.num_batches_tracked.item()


def test_materialize_dtensor_output_recurses_nested_containers():
    class FakeDTensor:
        def __init__(self, value):
            self.value = value

        def full_tensor(self):
            return self.value

    result = materialize_dtensor_output(
        {
            "a": FakeDTensor(torch.tensor([1])),
            "b": [FakeDTensor(torch.tensor([2])), (FakeDTensor(torch.tensor([3])),)],
        }
    )

    torch.testing.assert_close(result["a"], torch.tensor([1]))
    torch.testing.assert_close(result["b"][0], torch.tensor([2]))
    torch.testing.assert_close(result["b"][1][0], torch.tensor([3]))


def test_auto_tensor_parallel_plan_and_forward_match_single_rank():
    with single_rank_process_group():
        torch.manual_seed(0)
        baseline = ToyDistributedSNN()
        candidate = copy.deepcopy(baseline)

        x = torch.randn(5, 2, 8)
        _reset_net(baseline)
        reference = baseline(x)

        config = SNNDistributedConfig(
            device_type="cpu",
            mesh_shape=(1,),
            tensor_parallel_roots=["features"],
            auto_tensor_parallel=True,
            enable_data_parallel=False,
        )
        distributed_model, mesh, analysis = configure_snn_distributed(candidate, config)

        assert mesh.ndim == 1
        assert analysis.tensor_parallel_candidate_names == ("features.0", "features.2")

        _reset_net(distributed_model)
        result = materialize_dtensor_output(distributed_model(x))

        torch.testing.assert_close(reference, result)
        assert type(distributed_model.features[1]) is type(baseline.features[1])
        assert not hasattr(distributed_model.features[1], "inner")
        assert _has_tensor_shard_input_validator(distributed_model.features[1])

        plan = auto_build_tensor_parallel_plan(
            candidate, tensor_parallel_roots=["features"]
        )
        assert set(plan.keys()) == {"features.0", "features.2"}


def test_replace_module_by_name_replaces_nested_indexed_module():
    model = nn.Sequential(nn.ModuleList([nn.Identity(), nn.Linear(2, 2)]))
    replacement = nn.Linear(2, 3)

    _replace_module_by_name(model, "0.1", replacement)

    assert model[0][1] is replacement


def test_wrap_tp_memory_modules_skips_stateless_layers_before_memory_module():
    module = nn.Sequential(
        nn.Linear(4, 6),
        nn.Dropout(p=0.0),
        nn.Identity(),
        neuron.ParametricLIFNode(step_mode="s"),
    )

    wrapped = wrap_tp_memory_modules(
        module,
        {"0": "colwise_local_output"},
        process_group=None,
    )

    assert wrapped is module
    assert _has_tensor_shard_input_validator(module[3])
    assert module[3](torch.ones(2, 6)).shape == (2, 6)
    with pytest.raises(ValueError, match="Expected local shard size 6"):
        module[3](torch.ones(2, 5))


def test_configure_snn_distributed_supports_experimental_conv_tp_on_real_snn():
    with single_rank_process_group():
        torch.manual_seed(0)
        baseline = CIFAR10DVSVGG(dropout=0.0, backend="torch")
        candidate = copy.deepcopy(baseline)
        baseline.eval()
        candidate.eval()

        x = torch.randn(1, 2, 2, 48, 48)
        _reset_net(baseline)
        reference = baseline(x)

        config = SNNDistributedConfig(
            device_type="cpu",
            mesh_shape=(1,),
            auto_tensor_parallel=True,
            tensor_parallel_roots=["classifier"],
            experimental_conv_tensor_parallel=True,
            conv_tensor_parallel_roots=["features"],
            enable_data_parallel=False,
        )
        distributed_model, _, _ = configure_snn_distributed(candidate, config)

        _reset_net(distributed_model)
        result = materialize_dtensor_output(distributed_model(x))

        torch.testing.assert_close(reference, result, rtol=1e-5, atol=1e-6)
        assert (
            "ChannelShardConv2d"
            in type(distributed_model.features[0].proj_bn[-2]).__name__
        )
        assert type(distributed_model.features[0].neuron) is type(
            baseline.features[0].neuron
        )
        assert (
            distributed_model.features[0].neuron.v.shape[1]
            == distributed_model.features[0].proj_bn[-2].out_channels
        )


def test_cifar10dvs_vgg_tp_helper_after_memopt_level2_single_rank():
    with single_rank_process_group():
        torch.manual_seed(0)
        baseline = CIFAR10DVSVGG(dropout=0.0, backend="torch").eval()
        x = torch.randn(1, 2, 2, 48, 48)
        candidate = copy.deepcopy(baseline).eval()
        candidate.features[0] = GCContainer(None, candidate.features[0])
        distributed_model, _, _ = configure_snn_distributed(
            candidate,
            SNNDistributedConfig(
                device_type="cpu",
                mesh_shape=(1,),
                tensor_parallel_roots=["classifier"],
                auto_tensor_parallel=True,
                experimental_conv_tensor_parallel=True,
                conv_tensor_parallel_roots=["features"],
                enable_data_parallel=False,
            ),
        )
        _reset_net(baseline)
        reference = baseline(x)
        _reset_net(distributed_model)
        result = materialize_dtensor_output(distributed_model(x))
        torch.testing.assert_close(reference, result, rtol=1e-5, atol=1e-6)
        assert _has_tensor_shard_input_validator(
            distributed_model.features[0][0].neuron
        )


def test_spikformer_tp_helper_after_memopt_level2_single_rank():
    with single_rank_process_group():
        torch.manual_seed(0)
        baseline = spikformer_ti(
            T=2, img_size_h=64, img_size_w=64, num_classes=11, backend="torch"
        ).eval()
        x = torch.randn(2, 3, 64, 64)
        candidate = copy.deepcopy(baseline).eval()
        candidate.patch_embed.stages[0] = GCContainer(
            None, candidate.patch_embed.stages[0]
        )
        candidate.blocks[0] = GCContainer(None, candidate.blocks[0])
        distributed_model, _, _ = configure_snn_distributed(
            candidate,
            SNNDistributedConfig(
                device_type="cpu",
                mesh_shape=(1,),
                tensor_parallel_roots=["head"],
                auto_tensor_parallel=True,
                experimental_spikformer_tensor_parallel=True,
                spikformer_tensor_parallel_roots=["blocks"],
                experimental_spikformer_patch_stem_tensor_parallel=True,
                spikformer_patch_stem_tensor_parallel_roots=["patch_embed"],
                enable_data_parallel=False,
            ),
        )
        _reset_net(baseline)
        reference = baseline(x)
        _reset_net(distributed_model)
        result = materialize_dtensor_output(distributed_model(x))
        torch.testing.assert_close(reference, result, rtol=1e-5, atol=1e-6)
        assert _has_tensor_shard_input_validator(
            distributed_model.patch_embed.stages[0][0].neuron
        )
        assert (
            "ChannelShardConv1d"
            in type(distributed_model.blocks[0][0].attn.qkv_conv_bn[0]).__name__
        )
        assert _has_tensor_shard_input_validator(
            distributed_model.blocks[0][0].mlp.neuron1
        )


def test_spikformer_head_tp_helper_single_rank():
    with single_rank_process_group():
        torch.manual_seed(0)
        baseline = spikformer_ti(
            T=2, img_size_h=64, img_size_w=64, num_classes=11, backend="torch"
        ).eval()
        candidate = copy.deepcopy(baseline).eval()
        distributed_model, mesh, analysis = configure_snn_distributed(
            candidate,
            SNNDistributedConfig(
                device_type="cpu",
                mesh_shape=(1,),
                tensor_parallel_roots=["head"],
                auto_tensor_parallel=True,
                experimental_spikformer_tensor_parallel=True,
                spikformer_tensor_parallel_roots=["blocks"],
                experimental_spikformer_patch_stem_tensor_parallel=True,
                spikformer_patch_stem_tensor_parallel_roots=["patch_embed"],
                enable_data_parallel=False,
            ),
        )
        x = torch.randn(2, 3, 64, 64)
        _reset_net(baseline)
        reference = baseline(x)
        _reset_net(distributed_model)
        result = materialize_dtensor_output(distributed_model(x))
        torch.testing.assert_close(reference, result, rtol=1e-5, atol=1e-6)
        assert mesh.ndim == 1
        assert analysis.tensor_parallel_candidate_names == ("head",)
        assert isinstance(
            distributed_model.patch_embed.stages[0].neuron, base.MemoryModule
        )
        assert not hasattr(distributed_model.patch_embed.stages[0].neuron, "inner")
        assert (
            "ChannelShardConv2d"
            in type(distributed_model.patch_embed.stages[0].conv_bn.block[0]).__name__
        )


def test_spikformer_fsdp2_single_rank_smoke():
    with single_rank_process_group():
        torch.manual_seed(0)
        candidate = spikformer_ti(
            T=2, img_size_h=64, img_size_w=64, num_classes=11, backend="torch"
        ).eval()
        analysis = analyze(candidate)
        distributed_plan = plan(
            analysis=analysis,
            objective="memory",
            topology={"dp": 1},
            backend="torch",
            batch_size=1,
            model_family="spikformer",
            mode="fsdp2",
        )
        runtime = apply(model=candidate, plan=distributed_plan, device_type="cpu")
        mesh = runtime.mesh
        assert mesh is not None
        assert tuple(mesh.shape) == tuple(distributed_plan.topology.mesh_shape)
        assert runtime.plan is distributed_plan


def test_spikformer_patch_stem_tp_helper_handles_patch_embed_root():
    with single_rank_process_group():
        torch.manual_seed(0)
        candidate = spikformer_ti(
            T=2, img_size_h=64, img_size_w=64, num_classes=11, backend="torch"
        ).eval()
        distributed_model, _, _ = configure_snn_distributed(
            candidate,
            SNNDistributedConfig(
                device_type="cpu",
                mesh_shape=(1,),
                tensor_parallel_roots=["head"],
                auto_tensor_parallel=True,
                experimental_spikformer_tensor_parallel=True,
                spikformer_tensor_parallel_roots=["blocks"],
                experimental_spikformer_patch_stem_tensor_parallel=True,
                spikformer_patch_stem_tensor_parallel_roots=["patch_embed"],
                enable_data_parallel=False,
            ),
        )
        assert (
            "ChannelShardConv2d"
            in type(distributed_model.patch_embed.stages[0].conv_bn.block[0]).__name__
        )
        assert isinstance(
            distributed_model.patch_embed.stages[0].neuron, base.MemoryModule
        )
        assert not hasattr(distributed_model.patch_embed.stages[0].neuron, "inner")


def test_spikformer_patch_stem_tp_helper_handles_single_stage_root_colwise():
    with single_rank_process_group():
        torch.manual_seed(0)
        candidate = spikformer_ti(
            T=2, img_size_h=64, img_size_w=64, num_classes=11, backend="torch"
        ).eval()
        distributed_model, _, _ = configure_snn_distributed(
            candidate,
            SNNDistributedConfig(
                device_type="cpu",
                mesh_shape=(1,),
                auto_tensor_parallel=False,
                experimental_spikformer_patch_stem_tensor_parallel=True,
                spikformer_patch_stem_tensor_parallel_roots=["patch_embed.stages.0"],
                enable_data_parallel=False,
            ),
        )
        shard_conv = distributed_model.patch_embed.stages[0].conv_bn.block[0]
        assert "ChannelShardConv2d" in type(shard_conv).__name__
        assert shard_conv.mode == "colwise"
        next_conv = distributed_model.patch_embed.stages[1].conv_bn.block[0]
        assert "ChannelShardConv2d" in type(next_conv).__name__
        assert next_conv.mode == "rowwise"


def test_spikformer_patch_stem_tp_helper_handles_seq_to_ann_root():
    with single_rank_process_group():
        stem = (
            spikformer_ti(
                T=2, img_size_h=64, img_size_w=64, num_classes=11, backend="torch"
            )
            .patch_embed.stages[0]
            .conv_bn.block
        )
        module = nn.Sequential(stem, copy.deepcopy(stem))

        distributed_model, _, _ = configure_snn_distributed(
            module,
            SNNDistributedConfig(
                device_type="cpu",
                mesh_shape=(1,),
                auto_tensor_parallel=False,
                experimental_spikformer_patch_stem_tensor_parallel=True,
                spikformer_patch_stem_tensor_parallel_roots=["0"],
                enable_data_parallel=False,
            ),
        )

        assert "ChannelShardConv2d" in type(distributed_model[0][0]).__name__


def test_spikformer_patch_stem_tp_helper_rejects_unpaired_isolated_root():
    with single_rank_process_group():
        torch.manual_seed(0)
        candidate = spikformer_ti(
            T=2, img_size_h=64, img_size_w=64, num_classes=11, backend="torch"
        ).eval()
        with pytest.raises(ValueError, match="at least two consecutive stem blocks"):
            configure_snn_distributed(
                candidate,
                SNNDistributedConfig(
                    device_type="cpu",
                    mesh_shape=(1,),
                    auto_tensor_parallel=False,
                    experimental_spikformer_patch_stem_tensor_parallel=True,
                    spikformer_patch_stem_tensor_parallel_roots=[
                        "patch_embed.stages.3"
                    ],
                    enable_data_parallel=False,
                ),
            )
