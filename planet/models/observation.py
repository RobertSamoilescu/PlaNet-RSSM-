import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor


class ObservationModel(nn.Module):
    def __init__(
        self,
        hidden_state_size: int,
        state_size: int,
        observation_size: int,
        hidden_layer_size: int,
    ) -> None:
        super(ObservationModel, self).__init__()
        input_size = hidden_state_size + state_size

        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, observation_size)

    def forward(self, hidden_state: Tensor, state: Tensor) -> Tensor:
        x = torch.cat([hidden_state, state], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ImageObservationModel(nn.Module):
    def __init__(
        self,
        hidden_state_size: int,
        state_size: int,
        observation_size: int,
    ):
        super(ImageObservationModel, self).__init__()
        self.fc = nn.Linear(state_size + hidden_state_size, observation_size)
        self.dc1 = nn.ConvTranspose2d(
            observation_size, 128, kernel_size=5, stride=2
        )
        self.dc2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)
        self.dc3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2)
        self.dc4 = nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2)

    def forward(self, hidden_state: Tensor, state: Tensor) -> Tensor:
        x = self.fc(torch.cat([state, hidden_state], dim=1))
        x = x.view(x.size(0), 1024, 1, 1)
        x = F.relu(self.dc1(x))
        x = F.relu(self.dc2(x))
        x = F.relu(self.dc3(x))
        return self.dc4(x)
