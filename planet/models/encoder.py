import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor


class EncoderModel(nn.Module):
    def __init__(
        self,
        rnn_hidden_size: int,
        observation_size: int,
        state_size: int,
        hidden_size: int = 200,
        min_std: float = 0.1,
    ) -> None:
        """Encoder model

        :param rnn_hidden_size: hidden size of the rnn
        :param observation_size: size of the observation
        :param state_size: size of the state
        :param hidden_size: size of the hidden layer
        :param min_std: minimum standard deviation
        """
        super(EncoderModel, self).__init__()
        self.min_std = min_std

        input_size = rnn_hidden_size + observation_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean_head = nn.Linear(hidden_size, state_size)
        self.std_head = nn.Linear(hidden_size, state_size)

    def forward(
        self, hidden_state: Tensor, observation: Tensor
    ) -> torch.distributions.Normal:
        """Forward pass

        :param hidden_state: hidden state of the rnn
        :param observation: observation tensor
        :return: normal distribution of the state
        """
        x = torch.cat([hidden_state, observation], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean_head(x)
        std = F.softplus(self.std_head(x)) + self.min_std
        return torch.distributions.Normal(mean, std)


class ImageEncoderModel(nn.Module):
    """
    Encoder to embed image observation (3, 64, 64) to vector (1024,)
    """

    def __init__(
        self,
        rnn_hidden_size: int,
        observation_size: int,
        state_size: int,
        hidden_size: int = 200,
        min_std: float = 0.1,
    ):
        """Image encoder model

        :param rnn_hidden_size: hidden size of the rnn
        :param observation_size: size of the observation
        :param state_size: size of the state
        :param hidden_size: size of the hidden layer
        :param min_std: minimum standard deviation
        """
        super(ImageEncoderModel, self).__init__()
        self.min_std = min_std

        # conv part
        self.cv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2)
        self.cv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.cv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.cv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2)

        # fc part
        input_size = rnn_hidden_size + observation_size
        self.fc = nn.Linear(input_size, hidden_size)
        self.mean_head = nn.Linear(hidden_size, state_size)
        self.std_head = nn.Linear(hidden_size, state_size)

    def forward(
        self, hidden_state: Tensor, observation: Tensor
    ) -> torch.distributions.Normal:
        """Forward pass

        :param hidden_state: hidden state of the rnn
        :param observation: observation tensor
        :return: normal distribution of the state
        """
        observation = F.relu(self.cv1(observation))
        observation = F.relu(self.cv2(observation))
        observation = F.relu(self.cv3(observation))
        observation = F.relu(self.cv4(observation)).reshape(
            observation.size(0), -1
        )

        x = torch.cat([hidden_state, observation], dim=-1)
        x = F.relu(self.fc(x))
        mean = self.mean_head(x)
        std = F.softplus(self.std_head(x)) + self.min_std
        return torch.distributions.Normal(mean, std)
