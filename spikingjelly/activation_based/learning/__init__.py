"""Learning methods: local synaptic plasticity and gradient-based online training.

学习方法：局部突触可塑性与基于梯度的在线训练。
"""

from . import online_learning, plasticity
from .plasticity import (
    MSTDPETLearner,
    MSTDPLearner,
    STDPLearner,
)

__all__ = [
    "plasticity",
    "online_learning",
    "STDPLearner",
    "MSTDPLearner",
    "MSTDPETLearner",
]
