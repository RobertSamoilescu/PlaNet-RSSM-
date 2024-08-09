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
    
    def cuda(self) -> Self:
        self.observations = self.observations.float().cuda()
        self.actions = self.actions.float().cuda()
        self.rewards = self.rewards.float().cuda()
        self.dones = self.dones.float().cuda()
        return self
