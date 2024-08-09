import torch
import torch.nn as nn
from torch.nn import functional as F
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
        self.fc = nn.Linear(input_size, hidden_layer_size)
        self.mean_head = nn.Linear(hidden_layer_size, state_size)
        self.std_head = nn.Linear(hidden_layer_size, state_size)

    def forward(self, hidden_state: Tensor, observation: Tensor) -> Tensor:
        x = torch.cat([hidden_state, observation], dim=-1)
        x = F.relu(self.fc(x))
        mean = self.mean_head(x)
        std = F.softplus(self.std_head(x)) + 0.1
        return torch.distributions.Normal(mean, std)


class ImageEncoderModel(nn.Module):
    """
    Encoder to embed image observation (3, 64, 64) to vector (1024,)
    """

    def __init__(
        self,
        hidden_state_size: int,
        observation_size: int,
        state_size: int,
        hidden_layer_size: int,
    ):
        super(ImageEncoderModel, self).__init__()
        # conv part
        self.cv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2)
        self.cv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.cv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.cv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2)

        # fc part
        input_size = hidden_state_size + observation_size
        self.fc = nn.Linear(input_size, hidden_layer_size)
        self.mean_head = nn.Linear(hidden_layer_size, state_size)
        self.std_head = nn.Linear(hidden_layer_size, state_size)

    def forward(self, hidden_state: Tensor, observation: Tensor):
        observation = F.relu(self.cv1(observation))
        observation = F.relu(self.cv2(observation))
        observation = F.relu(self.cv3(observation))
        observation = F.relu(self.cv4(observation)).reshape(
            observation.size(0), -1
        )

        x = torch.cat([hidden_state, observation], dim=-1)
        x = F.relu(self.fc(x))
        mean = self.mean_head(x)
        std = F.softplus(self.std_head(x)) + 0.1
        return torch.distributions.Normal(mean, std)
