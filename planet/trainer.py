import torch
import torch.nn as nn
from torch.distributions import Normal

import gym
import math
from tqdm import tqdm
from typing import Dict, Optional, Tuple, List

from planet.dataset.buffer import SequenceBuffer
from planet.dataset.env_objects import EnvStep
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
    gt_obs: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    mse_loss = torch.nn.functional.mse_loss(obs, gt_obs, reduction="none")
    mse_loss = mse_loss.reshape(mse_loss.shape[0], -1)
    return (mse_loss.mean(axis=-1) * mask).sum()


def _compute_reward_loss(
    gt_reward: torch.Tensor, reward: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    mse_loss = torch.nn.functional.mse_loss(reward, gt_reward, reduction="none")
    return (mse_loss * mask).sum()


def _compute_kl_divergence(
    enc_state_dist: Normal,
    next_state_dist: Normal,
    mask: torch.Tensor,
) -> torch.Tensor:
    return (
        torch.distributions.kl.kl_divergence(
            enc_state_dist, next_state_dist
        ).sum(axis=-1)
        * mask
    ).sum()


def _set_models_eval(models: Dict[str, nn.Module]):
    for model in models.values():
        model.eval()


def _set_models_train(models: Dict[str, nn.Module]):
    for model in models.values():
        model.train()


def model_train_step(
    buffer: SequenceBuffer,
    B: int,
    L: int,
    models: Dict[str, nn.Module],
    optimizers: Dict[str, torch.optim.Optimizer],
    hidden_state_size: int,
    state_size: int,
    action_size: int,
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
    _set_models_train(models)

    # define loss variables
    obs_loss = reward_loss = kl_div = 0

    # sample a batch of experiences
    batch = buffer.sample_batch(B, L)
    batch.observations = batch.observations.float().cuda()
    batch.actions = batch.actions.float().cuda()
    batch.rewards = batch.rewards.float().cuda()
    batch.dones = batch.dones.float().cuda()

    # initialize first hidden state
    # TODO: is this correct?
    hidden_state = torch.zeros(1, B, hidden_state_size).cuda()

    # get the first approximation of the state
    # TODO: is this correct?
    enc_mean_state, enc_log_std_state = models["enc_model"](
        hidden_state=hidden_state.reshape(B, hidden_state_size),
        observation=batch.observations[:, 0],
    )
    enc_state_dist = Normal(enc_mean_state, enc_log_std_state.exp())
    enc_state = enc_state_dist.rsample()

    for t in range(1, L):
        # compute deterministic hidden state
        next_hidden_state = models["det_state_model"](
            hidden_state=hidden_state,
            state=enc_state.reshape(B, 1, state_size),
            action=batch.actions[:, t - 1].reshape(B, 1, action_size),
        )

        # compute the prior distribution of the next state
        mean_next_state, log_std_next_state = models["stoch_state_model"](
            hidden_state=next_hidden_state.reshape(B, hidden_state_size),
        )
        next_state_dist = Normal(mean_next_state, log_std_next_state.exp())

        # compute next approximation of the state
        enc_next_mean_state, enc_log_std_next_state = models["enc_model"](
            hidden_state=next_hidden_state.reshape(B, hidden_state_size),
            observation=batch.observations[:, t],
        )
        enc_next_state_dist = Normal(
            enc_next_mean_state, enc_log_std_next_state.exp()
        )

        # compute the kl divergence between the posterior
        # and the prior of the state
        mask = 1 - batch.dones[:, t - 1]
        kl_div += _compute_kl_divergence(
            enc_next_state_dist, next_state_dist, mask
        )

        # sample next state
        enc_next_state = enc_next_state_dist.rsample()

        # compute next observation based on the
        # hidden state and the next state
        obs = models["obs_model"](
            hidden_state=next_hidden_state.reshape(B, hidden_state_size),
            state=enc_next_state,
        )

        # compute next reward based on the hidden state
        # and the next state
        reward = models["reward_obs_model"](
            hidden_state=next_hidden_state.reshape(B, hidden_state_size),
            state=enc_next_state,
        ).reshape(B)

        # comput observation loss
        gt_obs = batch.observations[:, t]
        obs_loss += _compute_observation_loss(gt_obs, obs, mask)

        # compute reward loss
        gt_reward = batch.rewards[:, t - 1]
        reward_loss += _compute_reward_loss(gt_reward, reward, mask)

        # update hidden state
        hidden_state = next_hidden_state
        enc_state = enc_next_state

    # zero gradients
    _zero_grad(optimizers)

    # compute average loss
    mask_sum = (1 - batch.dones[:, :-1]).sum()
    obs_loss /= mask_sum
    reward_loss /= mask_sum
    kl_div /= mask_sum

    # compute loss
    loss = obs_loss + reward_loss + kl_div
    loss.backward()

    # clip gradients
    _clip_grad_norm(models)

    # gradient step
    _gradient_step(optimizers)
    return loss.item(), obs_loss.item(), reward_loss.item(), kl_div.item()


@torch.no_grad()
def evaluate(
    env: gym.Env,
    buffer: SequenceBuffer,
    T: int,
    H: int,
    I: int,
    J: int,
    K: int,
    models: Dict[str, nn.Module],
    hidden_state_size: int,
    action_size: int,
    action_noise: Optional[float] = None
) -> Tuple[List[EnvStep], float]:
    _set_models_eval(models)

    sequence = []
    episode_reward = 0

    # reset environment
    obs, _ = env.reset()

    # initialize hidden state and state belief
    hidden_state = torch.zeros(1, 1, hidden_state_size).cuda()

    for _ in tqdm(range(T)):
        observation = torch.from_numpy(obs).float().reshape(1, -1).cuda()
        enc_mean_state, enc_log_std_state = models["enc_model"](
            hidden_state=hidden_state.reshape(1, -1),
            observation=observation,
        )
        enc_state = torch.distributions.Normal(
            enc_mean_state, enc_log_std_state.exp()
        ).sample()

        action = latent_planning(
            H=H,
            I=I,
            J=J,
            K=K,
            hidden_state=hidden_state,
            current_state_belief=enc_state,
            deterministic_state_model=models["det_state_model"],
            stochastic_state_model=models["stoch_state_model"],
            reward_model=models["reward_obs_model"],
            action_size=action_size,
        )

        # add exploration noise
        if action_noise is not None:
            action += torch.randn_like(action) * math.sqrt(action_noise)

        # take action in the environment
        action_cpu = action.cpu()
        next_obs, reward, terminated, truncated, _ = env.step(
            action_cpu.numpy()
        )

        # update episode reward and add
        # step to the sequence
        episode_reward += reward
        done = 1 if terminated or truncated else 0
        sequence.append(
            EnvStep(
                observation=torch.from_numpy(obs).float(),
                action=action_cpu.float(),
                reward=float(reward),
                done=float(done),
            )
        )

        if done == 1:
            break

        # update observation
        obs = next_obs

        # update hidden state
        hidden_state = models["det_state_model"](
            hidden_state=hidden_state,
            state=enc_state.reshape(1, 1, -1),
            action=action.reshape(1, 1, -1),
        )

    return sequence, episode_reward




def data_collection(
    env: gym.Env,
    buffer: SequenceBuffer,
    T: int,
    H: int,
    I: int,
    J: int,
    K: int,
    models: Dict[str, nn.Module],
    hidden_state_size: int,
    action_size: int,
    action_noise: float = 0.1
):
    """
    Data collection step.

    :param env: The environment.
    :param buffer: Buffer containing sequences of environment steps.
    :param T: Maximum number of steps per episode.
    :param H: Planning horizon distance.
    :param I: Optimization iterations.
    :param J: Candidates per iteration.
    :param K: Top-K candidates to keep.
    :param models: Dictionary containing the models.
    :param hidden_state_size: Size of the hidden state.
    :param action_size: Size of the action space.
    :return: The episode reward.
    """
    sequence, episode_reward = evaluate(
        env=env,
        buffer=buffer,
        T=T,
        H=H,
        I=I,
        J=J,
        K=K,
        models=models,
        hidden_state_size=hidden_state_size,
        action_size=action_size,
        action_noise=action_noise
    )
    buffer.add_sequence(sequence)
    return episode_reward


def train(
    env: gym.Env,
    train_steps: int,
    T: int,
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
    evaluate_interval: int = 10,
    action_noise: Optional[float] = None
) -> None:
    """
    Perform training.

    :param env: The environment.
    :param train_steps: Number of training steps.
    :param T: Maximum number of steps per episode.
    :param S: Random seeds episodes.
    :param C: Collect interval
    :param B: Batch size.
    :param L: Sequence length.
    :param models: Dictionary containing the models.
    :param optimizers: Dictionary containing the optimizers.
    :param hidden_state_size: Size of the hidden state.
    :param state_size: Size of the state.
    :param action_size: Size of the action space.
    """
    # running statitsics
    running_loss = None
    running_reward = None

    running_obs_loss = None
    running_reward_loss = None
    running_kl_div = None

    # initialize buffer with S random seeds episodes
    buffer = SequenceBuffer()
    buffer = init_buffer(
        buffer=buffer, env=env, num_sequences=S, max_sequence_len=T
    )

    for i in range(train_steps):

        # model fitting
        for s in range(C):
            loss, obs_loss, reward_loss, kl_div = model_train_step(
                buffer=buffer,
                B=B,
                L=L,
                models=models,
                optimizers=optimizers,
                hidden_state_size=hidden_state_size,
                state_size=state_size,
                action_size=action_size,
            )

            running_loss = (
                loss
                if running_loss is None
                else 0.99 * running_loss + 0.01 * loss
            )

            running_obs_loss = (
                obs_loss
                if running_obs_loss is None
                else 0.99 * running_obs_loss + 0.01 * obs_loss
            )

            running_reward_loss = (
                reward_loss
                if running_reward_loss is None
                else 0.99 * running_reward_loss + 0.01 * reward_loss
            )

            running_kl_div = (
                kl_div
                if running_kl_div is None
                else 0.99 * running_kl_div + 0.01 * kl_div
            )

        # data collection
        episode_reward = data_collection(
            env=env,
            buffer=buffer,
            T=T,
            H=H,
            I=I,
            J=J,
            K=K,
            models=models,
            hidden_state_size=hidden_state_size,
            action_size=action_size,
            action_noise=action_noise
        )

        # update running reward
        running_reward = (
            episode_reward
            if running_reward is None
            else 0.9 * running_reward + 0.1 * episode_reward
        )

        if i % log_interval == 0:
            print(
                f"Iter: {i}, Loss: {running_loss}, Reward: {running_reward}"
            )
            print(
                f"Obs Loss: {running_obs_loss}, Reward Loss: {running_reward_loss}, KL Div: {running_kl_div}"
            )

        if i % evaluate_interval == 0:
            _, episode_reward = evaluate(
                env=env,
                buffer=buffer,
                T=T,
                H=H,
                I=I,
                J=J,
                K=K,
                models=models,
                hidden_state_size=hidden_state_size,
                action_size=action_size,
            )
            print(
                f"Evaluation: Iter: {i}, Reward: {episode_reward}"
            )