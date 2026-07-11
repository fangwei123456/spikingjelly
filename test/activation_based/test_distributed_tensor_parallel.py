# ruff: noqa: F401,F403,F405
from test.activation_based.test_distributed_dtensor import *
from test.activation_based.test_distributed_dtensor import (
    _ToyNonCallableReset,
    _ToyResetCounter,
    _load_train_distributed_module,
    _reset_net,
    _train_args,
    _train_runtime,
)


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
        assert isinstance(distributed_model.features[1], TensorShardMemoryModule)

        plan = auto_build_tensor_parallel_plan(
            candidate, tensor_parallel_roots=["features"]
        )
        assert set(plan.keys()) == {"features.0", "features.2"}

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
        assert isinstance(distributed_model.features[0].neuron, TensorShardMemoryModule)
        assert (
            distributed_model.features[0].neuron.inner.v.shape[1]
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

def test_spikformer_tp_plus_memopt_level1_single_rank():
    pass

def test_spikformer_fsdp2_tp_plus_memopt_level1_single_rank():
    pass

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
            distributed_model.patch_embed.stages[0].neuron, TensorShardMemoryModule
        )
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
        assert runtime.mesh is not None

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
            distributed_model.patch_embed.stages[0].neuron, TensorShardMemoryModule
        )

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
