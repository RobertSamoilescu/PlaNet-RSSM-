import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class StochasticStateModel(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, latent_size: int
    ) -> None:
        super(StochasticStateModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean_head = nn.Linear(hidden_size, latent_size)
        self.log_std_head = nn.Linear(hidden_size, latent_size)

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        x = torch.relu(self.fc1(input))
        x = torch.relu(self.fc2(x))
        return self.mean_head(x), self.log_std_head(x)
