import torch
import torch.nn as nn
from torch.distributions import Normal

import os
import gym
import math
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional, Tuple, List, Any

from planet.dataset.buffer import SequenceBuffer
from planet.dataset.env_objects import EnvStep
from planet.utils.sample import init_buffer
from planet.utils.envs import make_env
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
    return 0.5 * (mse_loss.sum(axis=-1) * mask).sum()


def _compute_reward_loss(
    gt_reward: torch.Tensor, reward: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    mse_loss = torch.nn.functional.mse_loss(
        reward, gt_reward, reduction="none"
    )
    return 0.5 * (mse_loss * mask).sum()


def _compute_kl_divergence(
    posterior: Normal,
    prior_dist: Normal,
    mask: torch.Tensor,
    free_nats: float,
) -> torch.Tensor:
    kl = torch.distributions.kl_divergence(posterior, prior_dist).sum(dim=-1)
    kl = kl.clamp(min=free_nats)
    return (kl * mask).sum()


def _set_models_eval(models: Dict[str, nn.Module]):
    for model in models.values():
        model.eval()


def _set_models_train(models: Dict[str, nn.Module]):
    for model in models.values():
        model.train()


def save_models(
    models: Dict[str, nn.Module],
    optimizers: Dict[str, torch.optim.Optimizer],
    path: str,
):
    checkpoint = {}

    for key, model in models.items():
        checkpoint[key] = model.state_dict()

    for key, optimizer in optimizers.items():
        checkpoint[key] = optimizer.state_dict()

    torch.save(checkpoint, path)


def load_models(
    models: Dict[str, nn.Module],
    optimizers: Dict[str, torch.optim.Optimizer],
    path: str,
):
    checkpoint = torch.load(path)

    for key, model in models.items():
        model.load_state_dict(checkpoint[key])

    for key, optimizer in optimizers.items():
        optimizer.load_state_dict(checkpoint[key])


class PlanetTrainer:
    def __init__(
        self,
        config: Dict[str, Any],
        models: Dict[str, nn.Module],
        optimizers: Dict[str, torch.optim.Optimizer],
    ) -> None:

        self.config = config
        self.models = models
        self.optimizers = optimizers

    def model_fit_step(self) -> Dict[str, float]:
        # define loss variables
        obs_loss = reward_loss = kl_div = 0

        # sample a batch of experiences
        batch = self.buffer.sample_batch(
            B=self.config["train_config"]["B"],
            L=self.config["train_config"]["L"],
        ).cuda()

        # initialize first hidden state
        hidden_state = torch.zeros(
            self.config["train_config"]["B"],
            self.config["state_config"]["hidden_state_size"],
        ).cuda()

        # initialize posterior distribution
        posterior_dist = self.models["enc_model"](
            observation=batch.observations[:, 0], hidden_state=hidden_state
        )

        # sample initial state
        posterior = posterior_dist.rsample()

        for t in range(1, self.config["train_config"]["L"]):
            # compute deterministic hidden state
            next_hidden_state = self.models["det_state_model"](
                hidden_state=hidden_state,
                state=posterior,
                action=batch.actions[:, t - 1],
            )

            # compute the prior distribution of the next state
            next_prior_dist = self.models["stoch_state_model"](
                next_hidden_state
            )

            # compute next approximation of the state
            next_posterior_dist = self.models["enc_model"](
                hidden_state=next_hidden_state,
                observation=batch.observations[:, t],
            )

            # compute the kl divergence between the posterior
            mask = 1 - batch.dones[:, t - 1]
            kl_div += _compute_kl_divergence(
                posterior=next_posterior_dist,
                prior_dist=next_prior_dist,
                mask=mask,
                free_nats=self.config["train_config"]["free_nats"],
            )

            # sample next state
            next_posterior = next_posterior_dist.rsample()

            # compute next observation based on the
            # hidden state and the next state
            obs = self.models["obs_model"](
                hidden_state=next_hidden_state,
                state=next_posterior,
            )

            # compute next reward based on the hidden state
            # and the next state
            reward = self.models["reward_obs_model"](
                hidden_state=next_hidden_state,
                state=next_posterior,
            ).reshape(-1)

            # comput observation loss
            gt_obs = batch.observations[:, t]
            obs_loss += _compute_observation_loss(gt_obs, obs, mask)

            # compute reward loss
            gt_reward = batch.rewards[:, t - 1]
            reward_loss += _compute_reward_loss(gt_reward, reward, mask)

            # update hidden state
            hidden_state = next_hidden_state
            posterior = next_posterior

        # zero gradients
        _zero_grad(self.optimizers)

        # compute average loss
        mask_sum = (1 - batch.dones[:, :-1]).sum()
        obs_loss /= mask_sum
        reward_loss /= mask_sum
        kl_div /= mask_sum

        # compute loss
        loss = obs_loss + reward_loss + kl_div
        loss.backward()

        # clip gradients
        _clip_grad_norm(self.models, 1000)

        # gradient step
        _gradient_step(self.optimizers)
        return {
            "loss": loss.item(),
            "obs_loss": obs_loss.item(),
            "reward_loss": reward_loss.item(),
            "kl_div": kl_div.item(),
        }

    def update_model_running_loss(
        self, loss: Dict[str, float], running_stats: Dict[str, float]
    ) -> None:
        for key, value in loss.items():
            running_stats[key] = (
                value
                if running_stats.get(key) is None
                else 0.99 * running_stats[key] + 0.01 * value
            )
        return running_stats

    def update_data_collection_running_reward(
        self, episode_reward: float, running_stats: Dict[str, float]
    ) -> float:
        running_stats["reward"] = (
            episode_reward
            if running_stats.get("reward") is None
            else 0.9 * running_stats["reward"] + 0.1 * episode_reward
        )
        return running_stats

    @torch.no_grad()
    def collect_episode(
        self, env: Any, action_noise: Optional[float] = None
    ) -> Tuple[List[EnvStep], float]:
        _set_models_eval(self.models)

        # reset environment
        sequence, episode_reward = [], 0
        obs, _ = env.reset()

        # initialize hidden state and state belief
        hidden_state = torch.zeros(
            1, self.config["state_config"]["hidden_state_size"]
        ).cuda()

        max_episode_length = self.config["train_config"]["max_episode_length"]
        action_repeat = self.config["train_config"]["action_repeat"]
        T = max_episode_length // action_repeat

        for _ in range(T):
            observation = torch.from_numpy(obs).float().unsqueeze(0).cuda()
            posterior_dist = self.models["enc_model"](
                hidden_state=hidden_state,
                observation=observation,
            )
            action = latent_planning(
                H=self.config["train_config"]["H"],
                I=self.config["train_config"]["I"],
                J=self.config["train_config"]["J"],
                K=self.config["train_config"]["K"],
                hidden_state=hidden_state,
                current_state_belief=posterior_dist,
                deterministic_state_model=self.models["det_state_model"],
                stochastic_state_model=self.models["stoch_state_model"],
                reward_model=self.models["reward_obs_model"],
                action_size=self.config["state_config"]["action_size"],
            )

            # add exploration noise
            if action_noise is not None:
                action += torch.randn_like(action) * action_noise

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
            hidden_state = self.models["det_state_model"](
                hidden_state=hidden_state,
                state=posterior_dist.sample(),
                action=action.unsqueeze(0),
            )

        return sequence, episode_reward

    def data_collection(self) -> float:
        """Data collection step."""
        sequence, episode_reward = self.collect_episode(
            env=self.env,
            action_noise=self.config["train_config"]["action_noise"],
        )
        self.buffer.add_sequence(sequence)
        return episode_reward

    def train_step(self, i: int, running_stats: Dict[str, float] = {}):
        _set_models_train(self.models)
        
        # fit the world model
        for _ in range(self.config["train_config"]["C"]):
            model_loss = self.model_fit_step()
            running_stats = self.update_model_running_loss(
                model_loss, running_stats
            )

        # data collection
        reward = self.data_collection()
        running_stats = self.update_data_collection_running_reward(
            reward, running_stats
        )

        if i % self.config["train_config"]["log_interval"] == 0:
            print(
                f"Iter: {i}, "
                f"Loss: {running_stats['loss']}, Obs Loss: {running_stats['obs_loss']}, "
                f"Reward Loss: {running_stats['reward_loss']}, KL Div: {running_stats['kl_div']}, ",
                f"Collection reward: {running_stats['reward']}",
            )

    def evaluate(self, seed: int = 0) -> float:
        # set models to evaluation mode
        _set_models_eval(self.models)

        # create evaluation environment with deterministic seed
        _env_config = self.config["env_config"].copy()
        if _env_config["env_type"] == "gym":
            _env_config["seed"] = seed
        else:
            _env_config["task_kwargs"] = _env_config.get("task_kwargs", {})
            _env_config["task_kwargs"]["random"] = seed

        episode_rewards = []
        env = make_env(_env_config)
        num_eval_episodes = self.config["eval_config"]["num_eval_episodes"]

        for _ in tqdm(range(num_eval_episodes), desc="Evaluation"):
            episode_rewards.append(
                self.collect_episode(
                    env=env,
                    action_noise=None,
                )[1]
            )

        return np.mean(episode_rewards)

    def fit(self):
        best_score = -np.inf
        train_steps = self.config["train_config"]["train_steps"]

        checkpoint_path = os.path.join(
            self.config["train_config"]["checkpoint_dir"], "best_model.pth"
        )
        if not os.path.exists(self.config["train_config"]["checkpoint_dir"]):
            os.makedirs(self.config["train_config"]["checkpoint_dir"])

        # create environment and buffer
        self.env = make_env(self.config["env_config"])
        self.buffer = SequenceBuffer()

        # initialize buffer with S random seeds episodes
        max_episode_length = self.config["train_config"]["max_episode_length"]
        action_repeat = self.config["train_config"]["action_repeat"]
        T = max_episode_length // action_repeat
        self.buffer = init_buffer(
            buffer=self.buffer,
            env=self.env,
            num_sequences=self.config["train_config"]["S"],
            max_sequence_len=T,
        )

        for i in range(train_steps):
            self.train_step(i)

            if i % self.config["eval_config"]["eval_interval"] == 0:
                mean_reward = self.evaluate()
                print(f"Evaluation: Iter: {i}, Mean reward: {mean_reward}")

                if mean_reward > best_score:
                    best_score = mean_reward
                    save_models(self.models, self.optimizers, checkpoint_path)
