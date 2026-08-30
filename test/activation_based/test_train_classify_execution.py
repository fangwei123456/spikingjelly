import pytest
import torch

from spikingjelly.activation_based import neuron
from spikingjelly.activation_based.examples.common.train_classify import Trainer


def test_common_trainer_exposes_mutually_exclusive_execution_mode():
    args = (
        Trainer()
        .get_args_parser()
        .parse_args(
            ["--execution-mode", "cuda_graph", "--cuda-graph-warmup-steps", "13"]
        )
    )

    assert args.execution_mode == "cuda_graph"
    assert args.cuda_graph_warmup_steps == 13
    assert not hasattr(args, "compile")


def test_common_trainer_rejects_zero_cuda_graph_warmup():
    trainer = Trainer()
    args = trainer.get_args_parser().parse_args(
        ["--execution-mode", "cuda_graph", "--cuda-graph-warmup-steps", "0"]
    )

    with pytest.raises(ValueError, match="positive"):
        trainer.main(args)


def test_common_trainer_rejects_cuda_graph_on_cpu():
    trainer = Trainer()
    args = trainer.get_args_parser().parse_args(
        ["--execution-mode", "cuda_graph", "--device", "cpu"]
    )

    with pytest.raises(ValueError, match="CUDA device"):
        trainer.main(args)


@pytest.mark.parametrize(
    "options",
    [
        ["--compile-mode", "max-autotune"],
        ["--compile-backend", "eager"],
        ["--execution-mode", "cuda_graph", "--compile-eval"],
    ],
)
def test_common_trainer_rejects_compile_options_outside_compile_mode(options):
    trainer = Trainer()
    args = trainer.get_args_parser().parse_args(options)

    with pytest.raises(ValueError, match="execution-mode compile"):
        trainer.main(args)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_common_trainer_captures_forward_and_backward():
    trainer = Trainer()
    args = trainer.get_args_parser().parse_args(
        [
            "--execution-mode",
            "cuda_graph",
            "--cuda-graph-warmup-steps",
            "1",
            "--disable-amp",
            "--disable-pinmemory",
        ]
    )
    args.distributed = False
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 5), neuron.LIFNode(step_mode="s", backend="torch")
    ).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batches = [(torch.randn(2, 4), torch.tensor([0, 1])) for _ in range(3)]

    trainer.train_one_epoch(
        model,
        torch.nn.CrossEntropyLoss(),
        optimizer,
        batches,
        torch.device("cuda"),
        0,
        args,
    )

    assert trainer._cuda_graph_training_step.stats.captures == 1
    assert trainer._cuda_graph_training_step.stats.replays == 1
