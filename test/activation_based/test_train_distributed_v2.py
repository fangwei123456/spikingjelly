import pytest
import torch
import torch.nn as nn
from spikingjelly.activation_based import distributed as sjdist
from spikingjelly.activation_based.examples.memopt.models import CIFAR10DVSVGG
from torch.utils.data import TensorDataset

from test.activation_based._distributed_test_utils import single_rank_process_group


@pytest.mark.skipif(
    not (sjdist.DTENSOR_AVAILABLE and sjdist.FSDP2_AVAILABLE),
    reason="DTensor DeviceMesh or FSDP2 APIs are unavailable in the current PyTorch build.",
)
def test_new_distributed_api_supports_manual_training_loop_single_rank():
    with single_rank_process_group():
        torch.manual_seed(0)
        model = CIFAR10DVSVGG(dropout=0.0, backend="torch").to("cpu")
        dataset = TensorDataset(
            torch.randn(4, 2, 2, 48, 48),
            torch.tensor([0, 1, 2, 3]),
        )

        analysis = sjdist.analyze(
            model,
            model_family="cifar10dvs_vgg",
        )
        distributed_plan = sjdist.plan(
            analysis=analysis,
            objective="speed",
            topology={"dp": 1},
            backend="torch",
            batch_size=2,
            model_family="cifar10dvs_vgg",
            mode="fsdp2",
            features=sjdist.DistributedFeatureSet(
                allow_experimental_conv_tp=False,
            ),
        )
        runtime = sjdist.apply(
            model=model,
            plan=distributed_plan,
            device_type="cpu",
        )
        assert runtime.plan is not None
        assert runtime.plan.mode == "fsdp2"

        optimizer = runtime.build_optimizer(
            optimizer_cls=torch.optim.SGD,
            lr=1e-3,
        )
        loader = runtime.prepare_dataloader(
            dataset=dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=False,
        )
        criterion = nn.CrossEntropyLoss()

        first_param_before = next(runtime.model.parameters()).detach().clone()
        last_loss = None

        runtime.model.train()
        for images, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = runtime.model(images.float())
            logits, labels = runtime.prepare_classification_output(logits, labels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            runtime.reset_state()
            last_loss = loss

        assert last_loss is not None
        assert torch.isfinite(last_loss)
        first_param_after = next(runtime.model.parameters()).detach()
        assert not torch.equal(first_param_before, first_param_after)
