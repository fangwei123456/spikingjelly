# ruff: noqa: F401,F403,F405
from test.activation_based._distributed_dtensor_test_support import *


def test_plan_returns_structured_plan_from_analysis():
    model = ToyDistributedSNN()
    analysis = analyze(model, roots=["features"])
    distributed_plan = plan(
        analysis=analysis,
        objective="speed",
        topology={"dp": 1},
        backend="inductor",
        batch_size=8,
        model_family="toy_snn",
        features=DistributedFeatureSet(allow_pipeline=False),
    )
    assert isinstance(distributed_plan, SNNDistributedPlan)
    assert distributed_plan.objective == "speed"
    assert isinstance(distributed_plan.topology, SNNDistributedTopology)
    assert distributed_plan.topology.world_size == 1
    assert distributed_plan.tensor_parallel_roots == ("features",)


def test_plan_accepts_missing_model_family():
    model = ToyDistributedSNN()
    analysis = analyze(model, roots=["features"])
    distributed_plan = plan(
        analysis=analysis,
        objective="speed",
        topology={"dp": 1},
        backend="inductor",
        batch_size=8,
    )
    assert isinstance(distributed_plan, SNNDistributedPlan)
    assert distributed_plan.model_family == "generic"


def test_plan_respects_allow_zero_optimizer_flag():
    model = ToyDistributedSNN()
    analysis = analyze(model, roots=["features"])
    distributed_plan = plan(
        analysis=analysis,
        objective="speed",
        topology={"dp": 2},
        backend="inductor",
        batch_size=8,
        features=DistributedFeatureSet(allow_zero_optimizer=False),
    )
    assert distributed_plan.mode == "dp"
    assert distributed_plan.optimizer_strategy == "none"


def test_plan_allows_explicit_mode_override_for_advanced_users():
    model = ToyDistributedSNN()
    analysis = analyze(model, roots=["features"])
    distributed_plan = plan(
        analysis=analysis,
        objective="speed",
        topology={"tp": 2},
        backend="inductor",
        batch_size=8,
        mode="tp",
    )
    assert distributed_plan.mode == "tp"
    assert distributed_plan.optimizer_strategy == "none"
    assert distributed_plan.topology.mesh_shape == (2,)


def test_plan_rejects_tensor_parallel_without_candidates():
    model = nn.Sequential(neuron.IFNode(step_mode="m"))
    analysis = analyze(model)
    with pytest.raises(
        ValueError, match="requires at least one tensor-parallel candidate"
    ):
        plan(
            analysis=analysis,
            objective="memory",
            topology={"tp": 2},
            backend="inductor",
            batch_size=8,
            mode="tp",
        )


def test_plan_rejects_pipeline_when_feature_flag_disables_it():
    model = ToyDistributedSNN()
    analysis = analyze(model, roots=["features"])
    with pytest.raises(NotImplementedError, match="Pipeline parallelism"):
        plan(
            analysis=analysis,
            objective="capacity",
            topology={"pp": 2},
            backend="inductor",
            batch_size=8,
            mode="pp",
            features=DistributedFeatureSet(allow_pipeline=False),
        )


def test_recommend_snn_distributed_strategy_speed_prefers_dp_zero():
    recommendation = recommend_snn_distributed_strategy(
        model="spikformer_ti",
        world_size=2,
        prefer="speed",
        batch_size=4,
        zero_redundancy_optimizer_available=True,
        pipelining_available=True,
        fsdp2_available=True,
        tensor_parallel_available=True,
    )
    assert recommendation.mode == "dp"
    assert recommendation.optimizer_sharding == "zero"
    assert recommendation.memopt_level == 0


def test_recommend_snn_distributed_strategy_memory_prefers_fsdp2_tp():
    recommendation = recommend_snn_distributed_strategy(
        model="spikformer_ti",
        world_size=4,
        prefer="memory",
        batch_size=4,
        zero_redundancy_optimizer_available=True,
        pipelining_available=True,
        fsdp2_available=True,
        tensor_parallel_available=True,
    )
    assert recommendation.mode == "fsdp2_tp"
    assert recommendation.memopt_level == 1
    assert recommendation.mesh_shape == (2, 2)


def test_recommend_snn_distributed_strategy_capacity_prefers_pp():
    recommendation = recommend_snn_distributed_strategy(
        model="cifar10dvs_vgg",
        world_size=4,
        prefer="capacity",
        batch_size=8,
        zero_redundancy_optimizer_available=True,
        pipelining_available=True,
        fsdp2_available=True,
        tensor_parallel_available=True,
    )
    assert recommendation.mode == "pp"
    assert recommendation.memopt_level == 1
    assert recommendation.pp_microbatches == 8
    assert recommendation.pp_schedule == "interleaved"
    assert recommendation.pp_virtual_stages == 2
    assert recommendation.pp_layout is None
    assert recommendation.pp_delay_wgrad is False


def test_recommend_snn_distributed_strategy_capacity_degrades_virtual_stages_for_small_batch():
    recommendation = recommend_snn_distributed_strategy(
        model="cifar10dvs_vgg",
        world_size=4,
        prefer="capacity",
        batch_size=4,
        zero_redundancy_optimizer_available=True,
        pipelining_available=True,
        fsdp2_available=True,
        tensor_parallel_available=True,
    )
    assert recommendation.mode == "pp"
    assert recommendation.pp_microbatches == 4
    assert recommendation.pp_schedule == "1f1b"
    assert recommendation.pp_virtual_stages == 1


def test_recommend_snn_distributed_strategy_capacity_falls_back_when_batch_too_small_for_pp():
    recommendation = recommend_snn_distributed_strategy(
        model="cifar10dvs_vgg",
        world_size=4,
        prefer="capacity",
        batch_size=2,
        zero_redundancy_optimizer_available=True,
        pipelining_available=True,
        fsdp2_available=True,
        tensor_parallel_available=True,
    )
    assert recommendation.mode == "fsdp2_tp"
    assert recommendation.memopt_level == 1
    assert recommendation.mesh_shape == (2, 2)
    assert any("global batch is smaller" in note for note in recommendation.rationale)
    assert not any(
        "Pipeline APIs are unavailable" in note for note in recommendation.rationale
    )
    assert not any("prefer='memory'" in note for note in recommendation.rationale)


def test_recommended_pipeline_microbatches_rejects_too_small_batch():
    with pytest.raises(ValueError, match=r"batch_size .* must be >= num_stages"):
        distributed_dtensor.recommended_pipeline_microbatches(2, 4)


def test_recommended_pipeline_microbatches_rejects_uneven_fallback():
    with pytest.raises(ValueError, match="must be divisible"):
        distributed_dtensor.recommended_pipeline_microbatches(37, 8)
