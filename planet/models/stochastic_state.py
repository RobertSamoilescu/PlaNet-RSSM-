import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from typing import Tuple


class StochasticStateModel(nn.Module):
    def __init__(
        self,
        hidden_state_size: int,
        state_size: int,
        hidden_layer_size: int,
    ) -> None:
        super(StochasticStateModel, self).__init__()
        self.fc = nn.Linear(hidden_state_size, hidden_layer_size)
        self.mean_head = nn.Linear(hidden_layer_size, state_size)
        self.std_head = nn.Linear(hidden_layer_size, state_size)

    def forward(self, hidden_state: Tensor) -> Tuple[Tensor, Tensor]:
        x = F.relu(self.fc(hidden_state))
        mean = self.mean_head(x)
        std = F.softplus(self.std_head(x)) + 0.1
        return torch.distributions.Normal(mean, std)
