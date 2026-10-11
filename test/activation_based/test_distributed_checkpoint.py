import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from spikingjelly.activation_based.distributed.llm.checkpoint import save_checkpoint


def test_checkpoint_file_path_notifies_every_rank(tmp_path, monkeypatch):
    core = ModuleType("megatron.core")
    core.dist_checkpointing = SimpleNamespace()
    megatron = ModuleType("megatron")
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")

    rank = 0
    status = None

    def broadcast(value, src):
        nonlocal status
        assert src == 0
        if rank == 0:
            status = value.clone()
        else:
            value.copy_(status)

    def barrier():
        raise AssertionError("barrier must not run")

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    monkeypatch.setattr(torch.distributed, "barrier", barrier)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_text("not a directory")
    arguments = dict(
        checkpoint_dir=checkpoint,
        model=None,
        optimizer=None,
        scheduler=None,
        optimizer_step=1,
        consumed_samples=1,
        recipe={},
    )

    with pytest.raises(NotADirectoryError):
        save_checkpoint(**arguments)
    rank = 1
    with pytest.raises(RuntimeError, match="Rank 0 could not prepare"):
        save_checkpoint(**arguments)
