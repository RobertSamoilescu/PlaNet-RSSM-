import torch
import torch.nn as nn
from torch import Tensor


class RewardModel(nn.Module):
    def __init__(
        self, 
        hidden_state_size: int,
        state_size: int,
        hidden_layer_size: int,
     ) -> None:
        super(RewardModel, self).__init__()
        input_size = hidden_state_size + state_size
        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, 1)

    def forward(self, hidden_state: Tensor, state: Tensor) -> Tensor:
        x = torch.cat([hidden_state, state], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
