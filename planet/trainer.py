import torch
import torch.nn as nn
from torch.distributions import Normal

import gym
from typing import Dict
from tqdm import tqdm

from planet.dataset.buffer import SequenceBuffer
from planet.dataset.env_objects import EnvSequence, EnvStep
from planet.utils.sample import init_buffer
from planet.planning.planner import latent_planning


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
        

def model_train_step(
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
        hidden_state = models['det_state_model'](
            hidden_state=hidden_state,
            state=state.reshape(B, 1, state_size), 
            action=batch.actions[:, t-1:t]
        )

        # compute next state based on the deterministic hidden state
        mean_next_state, log_std_next_state  = models['stoch_state_model'](
            hidden_state=hidden_state.reshape(B, hidden_state_size),
        )

        # sample next state
        next_state_dist = Normal(mean_next_state, log_std_next_state.exp())
        next_state = next_state_dist.rsample()

        # compute next observation based on the 
        # hidden state and the next state
        next_obs = models['obs_model'](
            hidden_state=hidden_state.reshape(B, hidden_state_size),
            state=next_state,
        )

        # compute next reward based on the hidden state 
        # and the next state
        next_reward = models['reward_obs_model'](
            hidden_state=hidden_state.reshape(B, hidden_state_size),
            state=next_state,
        ).reshape(B)

        # compute next approximation of the state
        enc_mean_state, enc_log_std_state = models['enc_model'](
            hidden_state=hidden_state.reshape(B, hidden_state_size),
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
    env: gym.Env,
    train_steps: int,
    T: int,
    R: int,
    S: int,
    C: int,
    B: int, 
    L: int,
    H: int,
    I: int,
    J: int,
    K: int,
    models: Dict[str, nn.Module],
    optimizers: Dict[str, torch.optim.Optimizer],
    hidden_state_size: int,
    state_size: int,
    action_size: int,
    log_interval: int = 10,
) -> None:
    """
    Perform training.

    :param env: The environment.
    :param train_steps: Number of training steps.
    :param T: Maximum number of steps per episode.
    :param R: Action repeat.
    :param S: Random seeds episodes.
    :param C: Collect interval
    :param B: Batch size.
    :param L: Sequence length.
    :param models: Dictionary containing the models.
    :param optimizers: Dictionary containing the optimizers.
    :param hidden_state_size: Size of the hidden state.
    """
    # running statitsics
    running_loss = None
    running_reward = None

    # initialize buffer with S random seeds episodes
    buffer = SequenceBuffer()
    buffer = init_buffer(
        buffer=buffer, 
        env=env, 
        num_sequences=S,
        max_sequence_len=T
    )

    for i in range(train_steps):

        # model fitting
        for s in range(C):
            loss = model_train_step(
                buffer=buffer, 
                B=B, L=L, 
                models=models, 
                optimizers=optimizers, 
                hidden_state_size=hidden_state_size, 
                state_size=state_size
            )
            
            running_loss = loss if running_loss is None else 0.99 * running_loss + 0.01 * loss

        if i % log_interval == 0:
            print(f"Iter: {i}, Loss: {running_loss.item()}")

        # data collection
        sequence = []
        episode_reward = 0
        obs, _ = env.reset()

        # initialize hidden state and state belief
        hidden_state = torch.zeros(1, 1, hidden_state_size)

        for t in tqdm(range(T // R)):
            observation = torch.from_numpy(obs).reshape(1, -1)
            mean_state, log_std_state = models['enc_model'](
                hidden_state=hidden_state.reshape(1, -1),
                observation=observation
            )

            action = latent_planning(
                H=H,
                I=I,
                J=J,
                K=K,
                hidden_state=hidden_state,
                current_state_belief=(mean_state, log_std_state),
                deterministic_state_model=models['det_state_model'],
                stochastic_state_model=models['stoch_state_model'],
                reward_model=models['reward_obs_model'],
                action_size=action_size,
            )

            # add exploration noise
            action += torch.randn_like(action) * 0.3

            # take action in the environment
            reward_sum = 0
            for _ in range(R):
                _obs, _reward, terminated, truncated, _ = env.step(action.numpy())
                reward_sum += _reward    

                done = terminated or truncated
                if done:
                    break
                
            # add step to the sequence
            sequence.append(
                EnvStep(
                    observation=torch.from_numpy(obs),
                    action=action,
                    reward=reward_sum,
                    done=0
                )
            )

            # update episode reward
            episode_reward += reward_sum

            if done:
                break

            # update observation
            obs = _obs

            # update hidden state
            hidden_state = models['det_state_model'](
                hidden_state=hidden_state,
                state=mean_state.reshape(1, 1, -1),
                action=action.reshape(1, 1, -1)
            )

        # update running reward
        running_reward = episode_reward if running_reward is None else 0.99 * running_reward + 0.01 * episode_reward
        if i % log_interval == 0:
            print(f"Iter: {i}, Reward: {running_reward}")

        # add the sequence to the buffer
        buffer.add_sequence(sequence)

          