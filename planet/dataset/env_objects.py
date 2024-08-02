from torch import Tensor
from dataclasses import dataclass


@dataclass
class EnvStep:
    observation: Tensor
    action: Tensor
    reward: float
    done: int


@dataclass
class EnvSequence:
    observations: Tensor
    actions: Tensor
    rewards: Tensor
    dones: Tensor
