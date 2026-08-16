import torch

from benchmark.snn_llm.qwen2 import _InputQCFSRMSNorm
from benchmark.snn_llm.spikelm import _ElasticBiSpike


def test_checkpoint_keeps_calibration_but_not_ephemeral_membrane():
    spike = _ElasticBiSpike(time_steps=4, decay=0.25, amplitude=1.0)
    qwen_input = _InputQCFSRMSNorm(
        config=object(),
        hidden_size=3,
        eps=1e-6,
        scale=torch.tensor([0.25, 0.5, 1.0]),
        time_steps=4,
        use_snn_memopt=False,
    )

    assert "v" not in spike.state_dict()
    assert torch.equal(
        qwen_input.state_dict()["qcfs_scale"], torch.tensor([0.25, 0.5, 1.0])
    )
