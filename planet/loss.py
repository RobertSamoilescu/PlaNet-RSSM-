import torch
from torch.distributions import Normal


def compute_observation_loss(
    gt_obs: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    mse_loss = torch.nn.functional.mse_loss(obs, gt_obs, reduction="none")
    mse_loss = mse_loss.reshape(mse_loss.shape[0], -1)
    return (mse_loss.sum(axis=-1) * mask).sum()  # type: ignore[call-overload]


def compute_reward_loss(
    gt_reward: torch.Tensor, reward: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    mse_loss = torch.nn.functional.mse_loss(
        reward, gt_reward, reduction="none"
    )
    return (mse_loss * mask).sum()


def compute_kl_divergence(
    posterior: Normal,
    prior_dist: Normal,
    mask: torch.Tensor,
    free_nats: float,
) -> torch.Tensor:
    kl = torch.distributions.kl_divergence(posterior, prior_dist).sum(dim=-1)
    kl = kl.clamp(min=free_nats)
    return (kl * mask).sum()
