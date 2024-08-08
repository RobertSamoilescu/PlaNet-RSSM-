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
        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.mean_head = nn.Linear(hidden_layer_size, state_size)
        self.std_head = nn.Linear(hidden_layer_size, state_size)

    def forward(self, hidden_state: Tensor, observation: Tensor) -> Tensor:
        x = torch.cat([hidden_state, observation], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.mean_head(x)
        std = F.softplus(self.std_head(x)) + 0.1
        return mean, std


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, d_latent: int = 128):
        super(Encoder, self).__init__()
        self.d_latent = d_latent
        self.in_channels = in_channels

        modules = []
        hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc = nn.Linear(hidden_dims[-1] * 4, self.d_latent)

    def forward(self, x) -> Tensor:
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        return F.relu(self.fc(x))


class ImageEncoderModel(nn.Module):
    def __init__(
        self,
        hidden_state_size: int,
        observation_size: int,
        state_size: int,
        hidden_layer_size: int,
        in_channels: int = 3,
    ) -> None:
        super(ImageEncoderModel, self).__init__()
        input_size = hidden_state_size + observation_size

        self.obs_encoder = Encoder(
            in_channels=in_channels, d_latent=observation_size
        )
        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.mean_head = nn.Linear(hidden_layer_size, state_size)
        self.std_head = nn.Linear(hidden_layer_size, state_size)

    def forward(self, hidden_state: Tensor, observation: Tensor) -> Tensor:
        observation = self.obs_encoder(observation)
        x = torch.cat([hidden_state, observation], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.mean_head(x)
        std = F.softplus(self.std_head(x)) + 0.1
        return mean, std
