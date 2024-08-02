import torch
import torch.nn as nn
from torch import Tensor


class RewardModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, input: Tensor) -> Tensor:
        x = torch.relu(self.fc1(input))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
