import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import Tuple


class DeterministicStateModel(nn.Module):
    def __init__(
        self,
        hidden_state_size: int,
        state_size: int,
        action_size: int,
        hidden_layer_size: int,
        **kwargs
    ) -> None:
        super(DeterministicStateModel, self).__init__()
        kwargs.pop("batch_first", None)
        input_size = state_size + action_size
        self.fc = nn.Linear(input_size, hidden_layer_size)
        self.gru = nn.GRUCell(hidden_layer_size, hidden_state_size)

    def forward(
        self, hidden_state: Tensor, state: Tensor, action: Tensor
    ) -> Tuple[Tensor, Tensor]:
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc(x))
        return self.gru(x, hidden_state)
