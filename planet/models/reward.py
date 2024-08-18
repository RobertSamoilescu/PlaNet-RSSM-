import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor


class RewardModel(nn.Module):
    def __init__(
        self,
        rnn_hidden_size: int,
        state_size: int,
        hidden_size: int = 200,
    ) -> None:
        """Reward model

        :param rnn_hidden_size: hidden size of the rnn
        :param state_size: size of the state
        :param hidden_size: size of the hidden layer
        """
        super(RewardModel, self).__init__()
        input_size = rnn_hidden_size + state_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, hidden_state: Tensor, state: Tensor) -> Tensor:
        """Forward pass

        :param hidden_state: hidden state of the rnn
        :param state: state tensor
        :return: reward tensor
        """
        x = torch.cat([hidden_state, state], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
