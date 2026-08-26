import copy
import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from spikingjelly.activation_based import functional, layer, neuron
from spikingjelly.activation_based.distributed.tensor_parallel import (
    ChannelShardBatchNorm2d,
    ChannelShardConv2d,
)
from spikingjelly.activation_based.memopt import checkpoint_module


def _vision_tp_worker(rank: int, store_path: str) -> None:
    from spikingjelly.activation_based.model.sew_resnet import (
        SEWResNet34Builder,
        SEWResNet34Config,
    )
    from spikingjelly.activation_based.model.spikformer import (
        SpikformerBuilder,
        SpikformerConfig,
    )

    dist.init_process_group(
        "gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=2,
    )
    try:
        cases = (
            (
                SEWResNet34Config(time_steps=2, num_classes=7, image_size=32),
                SEWResNet34Builder,
            ),
            (
                SpikformerConfig(
                    time_steps=2,
                    num_classes=7,
                    image_height=32,
                    image_width=32,
                ),
                SpikformerBuilder,
            ),
        )
        for config, builder_cls in cases:
            torch.manual_seed(17)
            reference, _, _, _ = builder_cls(config).build(
                process_group=None,
                memopt_process_group=None,
                pipeline_rank=0,
                pipeline_size=1,
                pipeline_microbatches=1,
                device=torch.device("cpu"),
                micro_batch_size=1,
                memopt_level=0,
                memopt_compress_inputs=False,
                memopt_checkpoint_budget="memory",
            )
            torch.manual_seed(17)
            candidate, _, _, _ = builder_cls(config).build(
                process_group=dist.group.WORLD,
                memopt_process_group=dist.group.WORLD,
                pipeline_rank=0,
                pipeline_size=1,
                pipeline_microbatches=1,
                device=torch.device("cpu"),
                micro_batch_size=1,
                memopt_level=0,
                memopt_compress_inputs=False,
                memopt_checkpoint_budget="memory",
            )
            reference.eval()
            candidate.eval()
            torch.manual_seed(19)
            x_reference = torch.randn(
                config.time_steps, 1, 3, 32, 32, requires_grad=True
            )
            x_candidate = x_reference.detach().clone().requires_grad_(True)

            reference_output = reference(x_reference)
            candidate_output = candidate(x_candidate)
            torch.testing.assert_close(candidate_output, reference_output)
            reference_output.square().mean().backward()
            candidate_output.square().mean().backward()
            torch.testing.assert_close(x_candidate.grad, x_reference.grad)
    finally:
        dist.destroy_process_group()


def _channel_tp_worker(rank: int, store_path: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=2,
    )
    try:
        torch.manual_seed(7)
        conv1 = layer.Conv2d(4, 8, kernel_size=1, bias=False, step_mode="m")
        bn1 = layer.BatchNorm2d(8, step_mode="m")
        spike = neuron.IFNode(step_mode="m")
        conv2 = layer.Conv2d(8, 6, kernel_size=1, bias=True, step_mode="m")

        local_conv1 = ChannelShardConv2d(
            copy.deepcopy(conv1), dist.group.WORLD, "colwise"
        )
        local_bn1 = ChannelShardBatchNorm2d(copy.deepcopy(bn1), dist.group.WORLD)
        local_spike = copy.deepcopy(spike)
        local_conv2 = ChannelShardConv2d(
            copy.deepcopy(conv2), dist.group.WORLD, "rowwise"
        )

        x_reference = torch.randn(3, 2, 4, 5, 5, requires_grad=True)
        x_local = x_reference.detach().clone().requires_grad_(True)
        reference = conv2(spike(bn1(conv1(x_reference))))
        candidate = local_conv2(local_spike(local_bn1(local_conv1(x_local))))
        torch.testing.assert_close(candidate, reference)

        reference.square().mean().backward()
        candidate.square().mean().backward()
        torch.testing.assert_close(x_local.grad, x_reference.grad)

        start = rank * 4
        end = start + 4
        torch.testing.assert_close(
            local_conv1.weight.grad, conv1.weight.grad[start:end]
        )
        torch.testing.assert_close(local_bn1.running_mean, bn1.running_mean[start:end])
        torch.testing.assert_close(
            local_conv2.weight.grad, conv2.weight.grad[:, start:end]
        )

        functional.reset_net(local_spike)
        assert local_spike.v == 0.0
    finally:
        dist.destroy_process_group()


def test_channel_tp_matches_dense_multistep_forward_and_backward():
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(
            _channel_tp_worker,
            args=(os.path.join(directory, "store"),),
            nprocs=2,
            join=True,
        )


def test_channel_shard_batch_norm_updates_stats_once_under_memopt():
    source = layer.BatchNorm2d(4, step_mode="m")
    batch_norm = ChannelShardBatchNorm2d(source, None)
    module = checkpoint_module(batch_norm)
    x = torch.randn(2, 3, 4, 5, 5, requires_grad=True)

    module(x).sum().backward()

    assert batch_norm.num_batches_tracked.item() == 1


def test_builtin_vision_tensor_parallel_strategies_match_dense_models():
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(
            _vision_tp_worker,
            args=(os.path.join(directory, "store"),),
            nprocs=2,
            join=True,
        )
