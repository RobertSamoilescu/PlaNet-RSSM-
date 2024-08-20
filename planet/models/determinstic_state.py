import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import Tuple


class DeterministicStateModel(nn.Module):
    def __init__(
        self,
        rnn_hidden_size: int,
        state_size: int,
        action_size: int,
        hidden_size: int = 200,
        **kwargs
    ) -> None:
        """Deterministic state model

        :param rnn_hidden_size: hidden size of the rnn
        :param state_size: size of the state
        :param action_size: size of the action
        :param hidden_size: size of the hidden layer
        """
        super(DeterministicStateModel, self).__init__()
        kwargs.pop("batch_first", None)
        input_size = state_size + action_size
        self.fc = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRUCell(hidden_size, rnn_hidden_size)

    def forward(
        self, hidden_state: Tensor, state: Tensor, action: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass

        :param hidden_state: hidden state of the rnn
        :param state: state tensor
        :param action: action tensor
        :return: output, hidden state
        """
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc(x))
        return self.gru(x, hidden_state)
