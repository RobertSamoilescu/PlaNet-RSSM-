import torch.nn as nn
from torch import Tensor

from typing import Tuple


class DeterministicStateModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, **kwargs) -> None:
        super(DeterministicStateModel, self).__init__()
        kwargs.pop("batch_first", None)
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, **kwargs)

    def forward(self, input: Tensor, h_0: Tensor) -> Tuple[Tensor, Tensor]:
        output, h_n = self.gru(input, h_0)
        return output, h_n
