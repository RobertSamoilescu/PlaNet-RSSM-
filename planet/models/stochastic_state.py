import torch
import torch.nn as nn
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
        self.fc1 = nn.Linear(hidden_state_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.mean_head = nn.Linear(hidden_layer_size, state_size)
        self.log_std_head = nn.Linear(hidden_layer_size, state_size)

    def forward(self, hidden_state: Tensor) -> Tuple[Tensor, Tensor]:
        x = torch.relu(self.fc1(hidden_state))
        x = torch.relu(self.fc2(x))
        return self.mean_head(x), self.log_std_head(x)
