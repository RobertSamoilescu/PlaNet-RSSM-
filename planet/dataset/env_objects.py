from torch import Tensor
from dataclasses import dataclass
from typing_extensions import Self


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

    def to(self, device: str) -> Self:
        self.observations = self.observations.float().to(device)
        self.actions = self.actions.float().to(device)
        self.rewards = self.rewards.float().to(device)
        self.dones = self.dones.float().to(device)
        return self
