import torch


class Action:
    def __init__(self, action_size: int):
        self.action_size = action_size
        self._mean = torch.zeros(action_size)
        self._std = torch.ones(action_size)

    def sample(self) -> torch.Tensor:
        dist = torch.distributions.Normal(self._mean, self._std)
        return dist.sample()

    @property
    def mean(self) -> torch.Tensor:
        return self._mean

    @property
    def std(self) -> torch.Tensor:
        return self._std

    def update(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self._mean = mean
        self._std = std


class ActionPlanner:
    def __init__(self, H: int, action_size: int) -> None:
        self.H = H
        self.action_size = action_size
        self.actions = [Action(action_size) for _ in range(H)]

    def sample(self) -> torch.Tensor:
        return torch.stack([action.sample() for action in self.actions])

    def update(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        for i, action in enumerate(self.actions):
            action.update(mean[i], std[i])

    def cuda(self):
        for action in self.actions:
            action._mean = action._mean.cuda()
            action._std = action._std.cuda()
        return self
