import torch
import torch.nn as nn
from torch.distributions import Normal

from typing import Dict

from planet.dataset.buffer import SequenceBuffer
from planet.dataset.env_objects import EnvSequence


def _zero_grad(optimizers: Dict[str, torch.optim.Optimizer]):
    for optimizer in optimizers.values():
        optimizer.zero_grad()


def _clip_grad_norm(models: Dict[str, nn.Module], grad_norm: float = 1.0):
    for model in models.values():
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)


def _gradient_step(optimizers: Dict[str, torch.optim.Optimizer]):
    for optimizer in optimizers.values():
        optimizer.step()


def _compute_observation_loss(
    batch: EnvSequence, t: int, next_obs: torch.Tensor
) -> torch.Tensor:
    # extract ground truth observation
    batch_obs = batch.observations[:, t]
    assert batch_obs.shape == next_obs.shape

    obs_dist = Normal(batch_obs, torch.ones_like(batch_obs))
    return -(obs_dist.log_prob(next_obs).sum(axis=-1) * (1 - batch.dones[:, t])).sum()


def _compute_reward_loss(
    batch: EnvSequence, t: int, next_reward: torch.Tensor
) -> torch.Tensor:
    # extract ground truth reward
    batch_rewards = batch.rewards[:, t]
    assert batch_rewards.shape == next_reward.shape

    reward_dist = Normal(batch_rewards, torch.ones_like(batch_rewards))
    return -(reward_dist.log_prob(next_reward) * (1 - batch.dones[:, t])).sum()


def _compute_kl_divergence(
    batch: EnvSequence,
    t: int,
    enc_state_dist: Normal,
    next_state_dist: Normal,
) -> torch.Tensor:
    return (
        torch.distributions.kl.kl_divergence(enc_state_dist, next_state_dist).sum(axis=-1) * 
        (1 - batch.dones[:, t])
    ).sum()
        

def train_step(
    buffer: SequenceBuffer,
    B: int, 
    L: int,
    models: Dict[str, nn.Module],
    optimizers: Dict[str, torch.optim.Optimizer],
    hidden_state_size: int,
    state_size: int,
) -> torch.Tensor:
    """
    Perform a single training step.

    :param buffer: Buffer containing sequences of environment steps.
    :param B: Batch size.
    :param L: Sequence length.
    :param models: Dictionary containing the models.
    :param optimizers: Dictionary containing the optimizers.
    :param hidden_state_size: Size of the hidden state.
    :param state_size: Size of the state.
    """
    # define loss variables
    obs_loss = reward_loss = kl_div = 0

    # sample a batch of experiences
    batch = buffer.sample_batch(B, L)

    # initialize first hidden state
    # TODO: is this correct?
    hidden_state = torch.zeros(1, B, hidden_state_size)
    
    # get the first approximation of the state
    # TODO: is this correct?
    enc_mean_state, enc_log_std_state = models['enc_model'](
        hidden_state=hidden_state.reshape(B, hidden_state_size),
        observation=batch.observations[:, 0],
    )
    enc_state_dist = Normal(enc_mean_state, enc_log_std_state.exp())

    for t in range(1, L):
        # sample current state
        state = enc_state_dist.rsample()

        # compute deterministic hidden state
        next_hidden_state = models['det_state_model'](
            hidden_state=hidden_state,
            state=state.reshape(B, 1, state_size), 
            action=batch.actions[:, t-1:t]
        )

        # compute next state based on the deterministic hidden state
        mean_next_state, log_std_next_state  = models['stoch_state_model'](
            hidden_state=next_hidden_state.reshape(B, hidden_state_size),
        )

        # sample next state
        next_state_dist = Normal(mean_next_state, log_std_next_state.exp())
        next_state = next_state_dist.rsample()

        # compute next observation based on the 
        # hidden state and the next state
        next_obs = models['obs_model'](
            hidden_state=next_hidden_state.reshape(B, hidden_state_size),
            state=next_state,
        )

        # compute next reward based on the hidden state 
        # and the next state
        next_reward = models['reward_obs_model'](
            hidden_state=next_hidden_state.reshape(B, hidden_state_size),
            state=next_state,
        ).reshape(B)

        # compute next approximation of the state
        enc_mean_state, enc_log_std_state = models['enc_model'](
            hidden_state=next_hidden_state.reshape(B, hidden_state_size),
            observation=batch.observations[:, t],
        )
        enc_state_dist = Normal(enc_mean_state, enc_log_std_state.exp())

        # some sanity checks
        assert enc_mean_state.shape == mean_next_state.shape
        assert enc_log_std_state.shape == log_std_next_state.shape
        
        # comput losses
        obs_loss += _compute_observation_loss(batch, t, next_obs)
        reward_loss += _compute_reward_loss(batch, t, next_reward)
        kl_div += _compute_kl_divergence(batch, t, enc_state_dist, next_state_dist)

    # compute average loss
    obs_loss = obs_loss / (1 - batch.dones[:, 1:]).sum()
    reward_loss = reward_loss / (1 - batch.dones[:, 1:]).sum()
    kl_div = kl_div / (1 - batch.dones[:, 1:]).sum()

    # zero gradients
    _zero_grad(optimizers)

    # compute loss
    loss = obs_loss + reward_loss + kl_div
    loss.backward()

    # clip gradients
    _clip_grad_norm(models)
    
    # gradient step
    _gradient_step(optimizers)
    return loss


def train(
    train_steps: int,
    buffer: SequenceBuffer,
    B: int, 
    L: int,
    models: Dict[str, nn.Module],
    optimizers: Dict[str, torch.optim.Optimizer],
    hidden_state_size: int,
    state_size: int,
    log_interval: int = 10,
) -> None:
    """
    Perform training.

    :param train_steps: Number of training steps.
    :param buffer: Buffer containing sequences of environment steps.
    :param B: Batch size.
    :param L: Sequence length.
    :param models: Dictionary containing the models.
    :param optimizers: Dictionary containing the optimizers.
    :param hidden_state_size: Size of the hidden state.
    """
    for i in range(train_steps):
        loss = train_step(buffer, B, L, models, optimizers, hidden_state_size, state_size)

        if i % log_interval == 0:
            print(f"Loss: {loss.item()}")