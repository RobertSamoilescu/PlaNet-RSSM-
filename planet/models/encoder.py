import torch
import torch.nn as nn
from torch import Tensor


class EncoderModel(nn.Module):
    def __init__(
            self, 
            hidden_state_size: int, 
            observation_size: int, 
            state_size: int,
            hidden_layer_size: int,
        ) -> None:
        super(EncoderModel, self).__init__()
        input_size = hidden_state_size + observation_size
        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.mean_head = nn.Linear(hidden_layer_size, state_size)
        self.log_std_head = nn.Linear(hidden_layer_size, state_size)

    def forward(self, hidden_state: Tensor, observation: Tensor) -> Tensor:
        x = torch.cat([hidden_state, observation], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.mean_head(x), self.log_std_head(x)
        