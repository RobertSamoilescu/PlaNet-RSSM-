import torch
import torch.nn as nn
from torch.distributions import Normal

import os
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional, Tuple, List, Any

from planet.dataset.buffer import ReplayBuffer
from planet.dataset.env_objects import EnvStep
from planet.utils.sample import init_buffer
from planet.utils.envs import make_env
from planet.planning.planner import latent_planning


def _zero_grad(optimizers: Dict[str, torch.optim.Optimizer]):
    for optimizer in optimizers.values():
        optimizer.zero_grad()


def _gradient_step(optimizers: Dict[str, torch.optim.Optimizer]):
    for optimizer in optimizers.values():
        optimizer.step()


@torch.no_grad()
def collect_episode(
    env: Any,
    models: Dict[str, nn.Module],
    rnn_hidden_size: int,
    action_size: int,
    max_episode_length: int,
    H: int = 15,
    I: int = 10,
    J: int = 1_000,
    K: int = 100,
    action_min: float = -1.0,
    action_max: float = 1.0,
    action_noise: Optional[float] = None,
    device: str = "cuda",
) -> Tuple[List[EnvStep], float]:
    """Collect an episode using the world model.

    :param env: The environment to collect the episode
    :param model: Dictionary of models
    :param rnn_hidden_size: The size of the rnn hidden state
    :param H: The number of latent samples
    :param I: The number of action sequences
    :param J: The number of action samples
    :param K: The number of planning steps
    :param action_min: The minimum action value
    :param action_max: The maximum action value
    :param action_noise: The action noise
    :param device: The device to run the model
    :return: A tuple of list of environment steps and the total reward
    """
    _set_models_eval(models)

    # reset environment
    sequence, episode_reward = [], 0
    obs_np, _ = env.reset()

    # initialize hidden state
    hidden_state = torch.zeros(1, rnn_hidden_size).to(device)

    for _ in tqdm(range(max_episode_length), desc="Collecting"):
        # compute posterior distribution based on the current observation
        obs = torch.from_numpy(obs_np).float().to(device)
        posterior_dist = models["enc_model"](
            hidden_state=hidden_state,
            observation=obs.unsqueeze(0),
        )

        # compute action using latent planning
        action = latent_planning(
            H=H,
            I=I,
            J=J,
            K=K,
            hidden_state=hidden_state,
            current_state_belief=posterior_dist,
            deterministic_state_model=models["det_state_model"],
            stochastic_state_model=models["stoch_state_model"],
            reward_model=models["reward_obs_model"],
            action_size=action_size,
            action_min=action_min,
            action_max=action_max,
        )

        # add exploration noise
        if action_noise is not None:
            action += torch.randn_like(action) * action_noise
            action = torch.clamp(action, action_min, action_max)

        # take action in the environment
        action_np = action.cpu().numpy()
        next_obs_np, reward, terminated, truncated, _ = env.step(action_np)

        # update episode reward and
        # add step to the sequence
        episode_reward += reward
        done = 1 if terminated or truncated else 0
        sequence.append(
            EnvStep(
                observation=torch.from_numpy(obs_np),
                action=torch.from_numpy(action_np),
                reward=reward,
                done=done,
            )
        )

        if done == 1:
            break

        # update observation
        obs_np = next_obs_np

        # update hidden state
        hidden_state = models["det_state_model"](
            hidden_state=hidden_state,
            state=posterior_dist.sample(),
            action=action.unsqueeze(0),
        )

    return sequence, episode_reward


def _compute_observation_loss(
    gt_obs: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    mse_loss = torch.nn.functional.mse_loss(obs, gt_obs, reduction="none")
    mse_loss = mse_loss.reshape(mse_loss.shape[0], -1)
    return (mse_loss.sum(axis=-1) * mask).sum()  # type: ignore[call-overload]


def _compute_reward_loss(
    gt_reward: torch.Tensor, reward: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    mse_loss = torch.nn.functional.mse_loss(
        reward, gt_reward, reduction="none"
    )
    return (mse_loss * mask).sum()


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
        models: Dict[str, nn.Module],
        config: Dict[str, Any],
    ) -> None:
        """Initialize the trainer with models and config.

        :param models: A dictionary of models
        :param config: A dictionary of configuration parameters
        """
        self.config = config
        self.models = models

        # get all parameters
        self.all_params = []
        for model in models.values():
            self.all_params += list(model.parameters())

        # initialize optimizers
        self.optimizers: Dict[str, torch.optim.Optimizer] = {
            "all_models": torch.optim.Adam(
                self.all_params, lr=self.config["train_config"]["lr"]
            )
        }

        # create checkpoint directory
        self.best_checkpoint_path = os.path.join(
            self.config["train_config"]["checkpoint_dir"], "best_model.pth"
        )
        self.latest_checkpoint_path = os.path.join(
            self.config["train_config"]["checkpoint_dir"], "latest_model.pth"
        )
        if not os.path.exists(self.config["train_config"]["checkpoint_dir"]):
            os.makedirs(self.config["train_config"]["checkpoint_dir"])

        # create environment
        self.env = make_env(self.config["env_config"])

        # create replay buffer
        self.buffer = ReplayBuffer()

        # compute max episode length
        T = self.config["train_config"]["T"]
        skip = self.config["env_config"]["skip"]
        self.max_episode_len = T // skip

    def model_fit_step(self) -> Dict[str, float]:
        # define loss variables
        obs_loss = reward_loss = kl_div = torch.tensor(0.0)
        device = self.config["model_config"].get("device", "cuda")

        # sample a batch of experiences
        batch = self.buffer.sample_batch(
            B=self.config["train_config"]["B"],
            L=self.config["train_config"]["L"],
        ).to(device)

        # initialize first hidden state
        hidden_state = torch.zeros(
            self.config["train_config"]["B"],
            self.config["model_config"]["rnn_hidden_size"],
        ).to(device)

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
        torch.nn.utils.clip_grad_norm_(
            self.all_params, self.config["train_config"]["grad_norm_clip"]
        )

        # gradient step
        _gradient_step(self.optimizers)
        return {
            "loss": loss.item(),
            "obs_loss": obs_loss.item(),
            "reward_loss": reward_loss.item(),
            "kl_div": kl_div.item(),
        }

    def update_running_stats(
        self, stats: Dict[str, float], running_stats: Dict[str, float]
    ) -> Dict[str, float]:
        """Update running statistics.

        :param stats: A dictionary of stats values
        :param running_stats: A dictionary of running statistics
        :return: A dictionary of updated running statistics
        """
        for key, value in stats.items():
            running_stats[key] = (
                value
                if running_stats.get(key) is None
                else 0.99 * running_stats[key] + 0.01 * value
            )
        return running_stats

    def data_collection(self) -> Dict[str, float]:
        """Data collection step."""
        sequence, reward = collect_episode(
            env=self.env,
            models=self.models,
            rnn_hidden_size=self.config["model_config"]["rnn_hidden_size"],
            action_size=self.config["model_config"]["action_size"],
            max_episode_length=self.max_episode_len,
            H=self.config["train_config"]["H"],
            I=self.config["train_config"]["I"],
            J=self.config["train_config"]["J"],
            K=self.config["train_config"]["K"],
            action_min=self.config["train_config"]["action_min"],
            action_max=self.config["train_config"]["action_max"],
            action_noise=self.config["train_config"]["action_noise"],
        )
        self.buffer.add_sequence(sequence)
        return {"reward": reward}

    def train_step(self, i: int, running_stats: Dict[str, float] = {}):
        _set_models_train(self.models)

        # fit the world model
        for _ in tqdm(
            range(self.config["train_config"]["C"]), desc="Model fit"
        ):
            stats = self.model_fit_step()
            running_stats = self.update_running_stats(stats, running_stats)

        # data collection
        stats = self.data_collection()
        running_stats = self.update_running_stats(stats, running_stats)

        if i % self.config["train_config"]["log_interval"] == 0:
            print(
                "Iter: %d, "
                "Loss: %.4f, Obs Loss: %.4f, Reward Loss: %.4f, KL Div: %.4f, "
                "Collection reward: %.4f"
                % (
                    i,
                    running_stats["loss"],
                    running_stats["obs_loss"],
                    running_stats["reward_loss"],
                    running_stats["kl_div"],
                    running_stats["reward"],
                )
            )

    def _evaluate(self) -> List[float]:
        _set_models_eval(self.models)
        episode_rewards = []

        num_eval_episodes = self.config["eval_config"]["num_eval_episodes"]
        for _ in tqdm(range(num_eval_episodes), desc="Evaluation"):
            episode_rewards.append(
                collect_episode(
                    env=self.env,
                    models=self.models,
                    rnn_hidden_size=self.config["model_config"][
                        "rnn_hidden_size"
                    ],
                    action_size=self.config["model_config"]["action_size"],
                    max_episode_length=self.max_episode_len,
                    H=self.config["train_config"]["H"],
                    I=self.config["train_config"]["I"],
                    J=self.config["train_config"]["J"],
                    K=self.config["train_config"]["K"],
                    action_min=self.config["train_config"]["action_min"],
                    action_max=self.config["train_config"]["action_max"],
                    action_noise=None,
                )[1]
            )

        return episode_rewards

    def evaluate_step(self, i: int, best_score: Dict[str, float] = {}) -> None:
        eval_interval = self.config["eval_config"]["eval_interval"]
        train_steps = self.config["train_config"]["train_steps"]

        if "mean_reward" not in best_score:
            best_score["mean_reward"] = -np.inf

        if i % eval_interval == 0 or i == train_steps - 1:
            rewards = self._evaluate()
            mean_reward = np.mean(rewards).item()

            if mean_reward > best_score["mean_reward"]:
                # update best score and save model
                best_score["mean_reward"] = mean_reward
                save_models(
                    models=self.models,
                    optimizers=self.optimizers,
                    path=self.best_checkpoint_path,
                )

            # save latest model
            save_models(
                models=self.models,
                optimizers=self.optimizers,
                path=self.latest_checkpoint_path,
            )

            print(
                "Iter: %d, Mean Reward: %.4f, Median Reward: %.4f, "
                "Std Reward: %.4f, Min Reward: %.4f, Max Reward: %.4f"
                % (
                    i,
                    mean_reward,
                    np.median(rewards),
                    np.std(rewards),
                    np.min(rewards),
                    np.max(rewards),
                )
            )

    def fit(self):
        train_steps = self.config["train_config"]["train_steps"]

        # initialize buffer with S random seeds episodes
        self.buffer = init_buffer(
            buffer=self.buffer,
            env=self.env,
            num_sequences=self.config["train_config"]["S"],
            max_sequence_len=self.max_episode_len,
        )

        for i in range(train_steps):
            self.train_step(i)
            self.evaluate_step(i)
