import os
import tempfile
from contextlib import contextmanager

import torch.distributed as dist


@contextmanager
def single_rank_process_group():
    if dist.is_initialized():
        if dist.get_world_size() != 1:
            raise RuntimeError(
                "single_rank_process_group() requires world_size == 1 "
                "when reusing an initialized process group."
            )
        if dist.get_backend() != "gloo":
            raise RuntimeError(
                "single_rank_process_group() reuses an existing process group "
                f"with backend '{dist.get_backend()}', but expects 'gloo'."
            )
        yield
        return

    fd, path = tempfile.mkstemp()
    os.close(fd)
    init_method = "file:///" + path.replace("\\", "/")
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=0,
        world_size=1,
    )
    try:
        yield
    finally:
        dist.destroy_process_group()
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
