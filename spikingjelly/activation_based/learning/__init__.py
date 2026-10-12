"""Learning methods: local synaptic plasticity and gradient-based online training.

学习方法：局部突触可塑性与基于梯度的在线训练。
"""

from . import online_learning, plasticity
from .plasticity import (
    MSTDPETLearner,
    MSTDPLearner,
    STDPLearner,
    mstdp_linear_single_step,
    mstdpet_linear_single_step,
    stdp_conv1d_single_step,
    stdp_conv2d_single_step,
    stdp_linear_single_step,
    stdp_multi_step,
)

__all__ = [
    "plasticity",
    "online_learning",
    "STDPLearner",
    "MSTDPLearner",
    "MSTDPETLearner",
    "stdp_linear_single_step",
    "mstdp_linear_single_step",
    "mstdpet_linear_single_step",
    "stdp_conv1d_single_step",
    "stdp_conv2d_single_step",
    "stdp_multi_step",
]
