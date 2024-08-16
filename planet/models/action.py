import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor


class ActionModel(nn.Module):
    def __init__(
        self,
        hidden_layer_size: int,
        state_size: int,
        action_size: int,
    ) -> None:
        super(ActionModel, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_layer_size)
        self.ln1 = nn.LayerNorm(hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.ln2 = nn.LayerNorm(hidden_layer_size)
        self.mean = nn.Linear(hidden_layer_size, action_size)
        self.std = nn.Linear(hidden_layer_size, action_size)

    def forward(self, hidden_state: Tensor, state: Tensor):
        x = torch.cat([hidden_state, state], dim=-1)
        x = F.elu(self.ln1(self.fc1(x)))
        x = F.elu(self.ln2(self.fc2(x)))
        mean = self.mean(x)
        std = F.softplus(self.std(x)) + 0.1
        return torch.distributions.Normal(mean, std)
