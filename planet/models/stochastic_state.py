import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from typing import Tuple


class StochasticStateModel(nn.Module):
    def __init__(
        self,
        rnn_hidden_size: int,
        state_size: int,
        hidden_size: int = 200,
        min_std: float = 0.1,
    ) -> None:
        """Stochastic state model

        :param rnn_hidden_size: hidden size of the rnn
        :param state_size: size of the state
        :param hidden_size: size of the hidden layer
        :param min_std: minimum standard deviation
        """
        super(StochasticStateModel, self).__init__()
        self.min_std = min_std

        self.fc = nn.Linear(rnn_hidden_size, hidden_size)
        self.mean_head = nn.Linear(hidden_size, state_size)
        self.std_head = nn.Linear(hidden_size, state_size)

    def forward(self, rnn_hidden_state: Tensor) -> torch.distributions.Normal:
        """Forward pass

        :param hidden_state: hidden state of the rnn
        :return: normal distribution of the state
        """
        x = F.relu(self.fc(rnn_hidden_state))
        mean = self.mean_head(x)
        std = F.softplus(self.std_head(x)) + self.min_std
        return torch.distributions.Normal(mean, std)
