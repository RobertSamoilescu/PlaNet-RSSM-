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


class Decoder(nn.Module):
    def __init__(self, d_latent: int = 3, out_channels: int = 3):
        super(Decoder, self).__init__()
        self.d_latent = d_latent
        self.out_channels = out_channels

        modules = []
        hidden_dims = [512, 256, 128, 64, 32]

        self.decoder_input = nn.Linear(d_latent, hidden_dims[0] * 4)

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2,
                                       padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor):
        x = self.decoder_input(z)
        x = x.view(-1, 512, 2, 2)
        x = self.decoder(x)
        return self.final_layer(x)
    
    
class ImageObservationModel(nn.Module):
    def __init__(
        self,
        hidden_state_size: int,
        state_size: int,
        observation_size: int,
        hidden_layer_size: int,
        out_channels: int = 3,
    ) -> None:
        super(ImageObservationModel, self).__init__()
        input_size = hidden_state_size + state_size

        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, observation_size)
        self.decoder = Decoder(d_latent=state_size, out_channels=out_channels)

    def forward(self, hidden_state: Tensor, state: Tensor) -> Tensor:
        x = torch.cat([hidden_state, state], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.decoder(x)