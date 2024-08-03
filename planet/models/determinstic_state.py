import torch
import torch.nn as nn
from torch import Tensor

from typing import Tuple


class DeterministicStateModel(nn.Module):
    def __init__(self, hidden_state_size: int, state_size: int, action_size: int, **kwargs) -> None:
        super(DeterministicStateModel, self).__init__()
        kwargs.pop("batch_first", None)
        input_size = state_size + action_size
        self.gru = nn.GRU(input_size, hidden_state_size, batch_first=True, **kwargs)

    def forward(self, hidden_state: Tensor, state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        input_state = torch.cat([state, action], dim=-1)
        _, next_hidden_state = self.gru(input_state, hidden_state)
        return next_hidden_state
