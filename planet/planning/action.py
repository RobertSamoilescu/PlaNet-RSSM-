import torch


class ActionPlanner:
    def __init__(self, H: int, action_size: int) -> None:
        self.H = H
        self.action_size = action_size

        self.mean = torch.zeros(H, action_size)
        self.std = torch.ones(H, action_size)

    def sample(self, n: int) -> torch.Tensor:
        dist = torch.distributions.Normal(self.mean, self.std)
        return dist.sample((n,))

    def update(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean = mean
        self.std = std
